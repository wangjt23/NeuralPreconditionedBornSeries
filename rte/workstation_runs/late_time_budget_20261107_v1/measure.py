"""Same-block GPU late-correction cost discriminator; no training or teachers.

Native production solves are the timing authority. Thin-hook solves give a
diagnostic partition with overhead reported, never a fictitious successful tail.
"""
import ast
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
import random
import statistics
import time

import torch

REPOSITORY = Path('/data/wjt/NeuralPreconditionedBornSeries')
POLICY = json.loads((Path(__file__).resolve().parent/'policy.json').read_text())
sys.path.insert(0, str(REPOSITORY/POLICY['backend_dir']))
from direct_segmented_tsa import DirectSegmentedTSA
from direct_full_graph import DirectFullGraph
import rte.augmented_fgmres as production
from rte.benchmark_tail_solver import load_model, solve
from rte.global_source_predictor import balance_integrated_source
from rte.train_source_predictor import build_base, material_operator
from rte.run_large_angular_spectral_experiments import ExperimentConfig


def need(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync():
    torch.cuda.synchronize()


def diagnostic_core():
    original = next(n for n in ast.parse(Path(production.__file__).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'augmented_fgmres')
    function = copy.deepcopy(original)
    function.args.kwonlyargs.append(ast.arg(arg='event_sink'))
    function.args.kw_defaults.append(ast.Name(id='_cost_noop', ctx=ast.Load()))

    class Hooks(ast.NodeTransformer):
        def visit_Continue(self, node):
            return [ast.parse("event_sink('reject', locals())").body[0], node]

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == 'j':
                return [node, ast.parse("event_sink('commit', locals())").body[0]]
            return node

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'true_residual' for t in node.targets):
                return [ast.parse("event_sink('final_start', locals())").body[0], node,
                        ast.parse("event_sink('final_end', locals())").body[0]]
            return node

    function = Hooks().visit(function)
    clean = copy.deepcopy(function)
    clean.args.kwonlyargs, clean.args.kw_defaults = [], []

    class Erase(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'event_sink':
                return None
            return self.generic_visit(node)

    need(ast.dump(Erase().visit(clean), include_attributes=False) == ast.dump(original, include_attributes=False),
         'diagnostic hooks changed production arithmetic/control')
    namespace = dict(production.__dict__, _cost_noop=lambda *args: None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 '<thin-cost-hooks>', 'exec'), namespace)
    return namespace['augmented_fgmres']


class Phases:
    def __init__(self):
        self.points = []
        self.prefix_seen = False
        self.pool = [torch.cuda.Event(enable_timing=True) for _ in range(12)]
        for event in self.pool:
            event.record()
        sync()  # diagnostic preparation only, before its solve timer

    def mark(self, name):
        event = self.pool[len(self.points)]
        event.record()
        self.points.append((name, time.perf_counter(), event))

    def hook(self, event, frame):
        if event == 'commit':
            if frame['classical_calls'] == 4 and not frame['attempted'] and not self.prefix_seen:
                self.prefix_seen = True
                self.mark('prefix4')
            if frame['neural']:
                self.mark('proposal_end')
        elif event == 'reject':
            self.mark('proposal_end')
        elif event in ('final_start', 'final_end'):
            self.mark(event)

    def export(self):
        sync()
        first = self.points[0]
        return [{'name': n, 'host_since_start': t-first[1],
                 'cuda_event_since_start': first[2].elapsed_time(e)/1000.}
                for n, t, e in self.points]


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    root = Path('/data/wjt/NeuralPreconditionedBornSeries')
    here = Path(__file__).resolve().parent
    policy = json.loads((here/'policy.json').read_text())
    for path, expected in policy['source_sha256'].items():
        need(sha(root/path) == expected, 'input/source hash mismatch: '+path)
    out = here/'results'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    sync(); began = time.perf_counter()
    model = load_model(root/policy['checkpoint'], False, 'cuda')
    sync(); model_load = time.perf_counter()-began
    traced_core = diagnostic_core()
    rows, traces, setup_rows = [], [], []
    settings = dict(complete=False, policy=policy, model_load_seconds=model_load,
                    host_pid=os.getpid(), torch=torch.__version__, training=False,
                    heldout9626_accessed=False, hooks_erase_to_original=True)

    def save():
        for filename, value in [('rows', rows), ('traces', traces), ('setup', setup_rows), ('settings', settings)]:
            (out/(filename+'.json')).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    def accepted(high, field, status, margin, baseline):
        qnorm = float(high.source.norm())
        physical = float((high.source-high.A_op(field)).norm())
        delta = physical/margin
        denominator = float(field.norm())-delta
        bound = delta/denominator if denominator > 0 else float('inf')
        difference = 0. if baseline is None else float((field-baseline).norm()/baseline.norm())
        need(bool(torch.isfinite(field).all()) and status['failed'] == 0 and physical/qnorm <= 1e-5
             and bound <= 1e-3 and difference <= 1e-4, 'original-equation/field acceptance failed')
        return dict(true_residual=physical/qnorm, evaluated_bound=bound, relative_field_difference=difference)

    for grid in (128, 256):
        data = torch.load(root/policy['data'][str(grid)], map_location='cpu', weights_only=True)
        cfg = ExperimentConfig(size=grid, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
        # One current conflict check at each grid boundary, not a broad resource
        # audit or historical-lock gate. The queue checks before worker startup.
        import subprocess
        gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        others = [int(x.strip()) for x in gpu_pids.splitlines() if x.strip().isdigit() and int(x.strip()) != os.getpid()]
        need(not others, 'new competing GPU process: '+str(others))
        for index in policy['ids']:
            sync(); began = time.perf_counter()
            low, high = material_operator(cfg, data, torch.tensor([index]), .1, 'cuda')
            sync(); physical_setup = time.perf_counter()-began
            sync(); began = time.perf_counter()
            phase = high.phase_matrix.double()*high.weights.double()[None, :]
            mu = torch.linalg.eigvalsh((phase+phase.T)/2).max()
            margin = float((high.sigma_t-high.sigma_s*mu).min())
            need(margin > 0, 'evaluated physical margin not positive')
            sync(); acceptance_setup = time.perf_counter()-began
            depths = [8, 16] if grid == 128 else [16]
            bases, wrappers, costs = {}, {}, {}
            for depth in depths:
                sync(); began = time.perf_counter(); base = build_base(low, 'tsa'+str(depth)); sync()
                bases[depth] = base
                costs[(depth, 'base')] = time.perf_counter()-began
                order = ['seg', 'full'] if index % 2 == 0 else ['full', 'seg']
                for backend in order:
                    sync(); began = time.perf_counter()
                    wrappers[(depth, backend)] = (DirectSegmentedTSA(base, low.source, chunk=1, capture_first=False)
                                                   if backend == 'seg' else DirectFullGraph(base, low.source))
                    sync(); costs[(depth, backend)] = time.perf_counter()-began
            setup_rows.append(dict(grid=grid, index=index, physical=physical_setup, acceptance=acceptance_setup,
                component_costs={str(k):v for k,v in costs.items()}))
            methods = {}
            for backend in ('seg', 'full'):
                methods['classic_'+backend] = dict(depth=8 if grid == 128 else 16, backend=backend,
                    recipe='reference_tail' if grid == 128 else 'tsa16_zero', neural=False)
                if grid == 128:
                    methods['tsa16_'+backend] = dict(depth=16, backend=backend, recipe='tsa16_zero', neural=False)
                methods['aligned_'+backend] = dict(depth=16, backend=backend, recipe='tsa16_zero', neural=True)

            def execute(config, phases=None):
                wrapper = wrappers[(config['depth'], config['backend'])]
                base = bases[config['depth']]
                if phases is None and not config['neural']:
                    return solve(config['recipe'], low, high, {'tsa'+str(config['depth']):wrapper}, {})
                initial = None
                if config['recipe'] == 'reference_tail':
                    initial = base.reference.reference_transport_solve(low.source).to(torch.complex128)
                    initial = balance_integrated_source(high, initial, high.source)
                if phases is not None:
                    phases.mark('initialization_end')
                inject = None
                if config['neural'] or config['recipe'] == 'reference_tail':
                    def inject(x, r):
                        if phases is not None: phases.mark('callback_start')
                        ref = base.reference.reference_transport_solve(r.to(torch.complex64))
                        if phases is not None: phases.mark('reference_end')
                        value = model(low.sigma_t, low.sigma_s, r.to(torch.complex64), ref).to(torch.complex128) if config['neural'] else ref
                        if phases is not None: phases.mark('inference_end')
                        return value
                core = traced_core if phases is not None else production.augmented_fgmres
                kwargs = {'event_sink':phases.hook} if phases is not None else {}
                return core(high.A_op, high.source, lambda r:wrapper(r.to(torch.complex64)).to(torch.complex128),
                            initial, inject=inject, **kwargs)

            baseline = None
            native_fields, native_status = {}, {}
            order = list(methods)
            rng = random.Random(71107+grid+index)
            for repeat in range(-1, policy['repeats']):
                rng.shuffle(order)
                # Reference field uses a complete native solve and is itself
                # part of the matched measured/warmup rows, not a free teacher.
                if repeat == -1:
                    order.remove('classic_seg'); order.insert(0, 'classic_seg')
                for method in order:
                    config = methods[method]
                    sync(); began = time.perf_counter()
                    field, status = execute(config)
                    status.pop('trace', None); status.pop('initial_residual', None)
                    sync(); solver_time = time.perf_counter()-began
                    if baseline is None:
                        baseline = field.detach().clone()
                    sync(); began = time.perf_counter()
                    accuracy = accepted(high, field, status, margin, baseline)
                    sync(); check_time = time.perf_counter()-began
                    prep = physical_setup+acceptance_setup+costs[(config['depth'], 'base')]+costs[(config['depth'], config['backend'])]
                    rows.append(dict(grid=grid, index=index, method=method, repeat=repeat,
                        solver_seconds=solver_time, acceptance_seconds=check_time,
                        accepted_seconds=solver_time+check_time, setup_seconds=prep,
                        Q1_seconds=prep+solver_time+check_time, status=status, **accuracy))
                    native_fields[method], native_status[method] = field, status
            for method, config in methods.items():
                if not config['neural']:
                    continue  # cost partition only for the actual neural path
                phases = Phases()
                sync(); began = time.perf_counter(); phases.mark('start')
                field, status = execute(config, phases)
                status.pop('trace', None); status.pop('initial_residual', None)
                sync(); core_time = time.perf_counter()-began
                phases.mark('core_return')
                need(torch.equal(field, native_fields[method]) and status == native_status[method], 'thin hooks changed field/status')
                sync(); began = time.perf_counter()
                accuracy = accepted(high, field, status, margin, baseline)
                sync(); check_time = time.perf_counter()-began
                traces.append(dict(grid=grid, index=index, method=method, solver_seconds=core_time,
                    acceptance_seconds=check_time, phases=phases.export(), status=status,
                    bitwise_vs_native=True, **accuracy))
            save()
            print('CONTEXT_DONE',grid,index,flush=True)
            del native_fields, wrappers, bases, baseline, low, high
            sync(); torch.cuda.empty_cache()
    settings.update(complete=True, wall_seconds=time.perf_counter()-started,
        measured_solves=sum(r['repeat'] >= 0 for r in rows), warmup_solves=sum(r['repeat'] < 0 for r in rows),
        traced_diagnostic_solves=len(traces), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), workload='single RHS per material; preloaded model warm timing; global model load separately recorded',
        trace_timing='CPU wall plus asynchronous CUDA event spans; no phase synchronization; native timing is authority',
        primary_Q1_backend='seg; frozen from prior independent full16 confirmation', secondary_warm_backend='full; both classical variants retained')
    save();print(json.dumps(settings),flush=True)


if __name__ == '__main__':
    run()
