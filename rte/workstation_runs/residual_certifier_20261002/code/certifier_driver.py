"""Bounded FWI saved-field and manufactured-fixture certification; no solves."""
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
import torch

from interval_fft import certify_residual, load_native_cpu_backend
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
from rte.train_source_predictor import build_base, material_operator, source_family

START = time.perf_counter()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def tensor_array(v):
    return v.detach().cpu().numpy().copy()

def array_identity(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return dict(dtype=str(a.dtype), shape=list(a.shape), sha256=h.hexdigest(), bytes=a.nbytes)

@torch.no_grad()
def run():
    root = Path(__file__).resolve().parent.parent
    policy = json.loads((root/'policy.json').read_text())
    assert policy['phase'] == 'fresh_original_H_interval_certifier_v1'
    assert policy['max_full_solves'] == 0 and policy['development_ids'] == [0, 3]
    assert policy['tau_rational'] == [1, 100000] and policy['reserve_rational'] == [1, 100]
    assert socket.gethostname() == 'FWI-Standard-PC-i440FX-PIIX-1996'
    assert subprocess.check_output(['id', '-un'], text=True).strip() == 'wjt'
    assert Path.cwd() == Path('/data/wjt/NeuralPreconditionedBornSeries')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == policy['head']
    for name, digest in policy['code_sha256'].items():
        assert sha(root/name) == digest, name
    torch.set_num_threads(1)
    output = root/'results'; output.mkdir(exist_ok=False)
    guard = load_native_cpu_backend(root/'code/native_cpu.so')
    counters = dict(saved_teacher_certificates=0, manufactured_fixture_certificates=0,
                    negative_control_certificates=0, explicit_gpu_H_actions=0,
                    preconditioner_actions=0, full_solves=0, training_steps=0)
    costs = dict(parent_hash_load_seconds=0., physical_setup_seconds=0.,
                 base_capture_setup_seconds=0., preconditioner_seconds=0.,
                 explicit_gpu_H_seconds=0., transfers_seconds=0.,
                 certificate_seconds=0., json_save_seconds=0.)
    rows = []
    deadline = START + policy['soft_seconds']
    peak_allocated = peak_reserved = 0
    def limit():
        nonlocal peak_allocated, peak_reserved
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        if time.perf_counter() > deadline:
            raise TimeoutError('Frozen bounded certifier wall budget exceeded')
        if peak_reserved > policy['max_reserved_bytes']:
            raise MemoryError('Frozen GPU memory bound exceeded')
    def save(complete, error=None):
        began = time.perf_counter()
        (output/'certificates.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
        costs['json_save_seconds'] += time.perf_counter()-began
        settings = dict(complete=complete, error=error, counters=counters, costs=costs,
                        native_cpu_guard=guard, elapsed_seconds=time.perf_counter()-START,
                        cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        gpu_peak_allocated_bytes=peak_allocated, gpu_peak_reserved_bytes=peak_reserved,
                        global_gpu_contract_established=False, source_exported=False,
                        original_candidates_and_buffers_remain_on_FWI=True,
                        heldout_or_reserved_seed_access=False,
                        policy_sha256=sha(root/'policy.json'),
                        endpoint_sharedlib_sha256=sha(root/'code/native_cpu.so'),
                        torch=str(torch.__version__), numpy=str(np.__version__),
                        cuda=str(torch.version.cuda), full_solves=0)
        (output/'settings.json').write_text(json.dumps(settings, indent=2, allow_nan=False)+'\n')
    def timed_H(high, candidate):
        torch.cuda.synchronize(); began=time.perf_counter()
        image=high.A_op(candidate)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-began
        counters['explicit_gpu_H_actions'] += 1; costs['explicit_gpu_H_seconds'] += elapsed
        return image, elapsed
    def operands(high, candidate, source):
        torch.cuda.synchronize(); began=time.perf_counter()
        a=dict(source=tensor_array(source), candidate=tensor_array(candidate),
               stream_symbol=tensor_array(high._stream_symbol(1, torch.complex128, 'cuda')),
               sigma_t=tensor_array(high.sigma_t), sigma_s=tensor_array(high.sigma_s),
               phase=tensor_array(high.phase_matrix), weights=tensor_array(high.weights))
        costs['transfers_seconds'] += time.perf_counter()-began
        return a
    def record(kind,n,index,high,x,q,parent=None,gpu_image=None,gpu_seconds=0.):
        limit();verification_began=time.perf_counter(); a=operands(high,x,q)
        identities={k:array_identity(v)for k,v in a.items()}
        began=time.perf_counter()
        cert=certify_residual(**a,residual_tolerance=Fraction(1,100000),
                              solution_tolerance=Fraction(1,1000))
        costs['certificate_seconds'] += time.perf_counter()-began
        if kind=='original_source_zero':
            assert cert['source_relative_norm_defined']
            assert not cert['accepted'] and not cert['residual_accepted']
            assert not cert['reserve_center_accepted']
            assert Fraction.from_float(cert['residual_norm_lower'])>Fraction(1,100000)*Fraction.from_float(cert['source_norm_upper'])
        assert identities=={k:array_identity(v)for k,v in a.items()},'Input mutation'
        complete_verification_seconds=time.perf_counter()-verification_began
        if gpu_image is None:
            gpu_image,gpu_seconds=timed_H(high,x)
        ordinary=float((q-gpu_image).norm()/q.norm())
        row=dict(kind=kind,size=n,index=index,development_seed=4626,
                 parent=parent,buffer_identities=identities,certificate=cert,
                 ordinary_gpu_residual_diagnostic=ordinary,
                 explicit_gpu_H_seconds=gpu_seconds,
                 complete_verification_seconds=complete_verification_seconds,
                 verification_cost_scope='Capture actualD64 and all CPUtransfers, both bitidentityhash passes, complete certifier; excludes separate roundedGPU diagnostic and JSON persistence',
                 output_representation='Actual immutable saved/constructed complex128 bits',
                 source_representation='Actual immutable captured source bits',
                 no_input_mutation=True)
        rows.append(row)
        counters[{'saved_teacher':'saved_teacher_certificates',
                  'manufactured_M_source':'manufactured_fixture_certificates',
                  'original_source_zero':'negative_control_certificates'}[kind]] += 1
        limit();save(False)
    try:
        for n,parent in policy['teacher_banks'].items():
            n=int(n); limit(); began=time.perf_counter()
            path=Path(parent['path']); assert sha(path)==parent['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True)
            bank=payload['bank']; assert bank['target'].dtype==torch.complex128
            assert bank['q'].dtype==torch.complex64
            assert bank['q'].shape==(16,12,n,n)
            costs['parent_hash_load_seconds'] += time.perf_counter()-began
            cfg=ExperimentConfig(size=n,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
            for index in policy['development_ids']:
                limit();torch.cuda.synchronize();began=time.perf_counter()
                t,s,q,x=[bank[k][index:index+1].to('cuda')for k in ('t','s','q','target')]
                low=make_operator(cfg,t,s,q,'cuda'); high=high_precision_copy(low)
                assert torch.equal(high.source, q.to(torch.complex128))
                q=high.source;torch.cuda.synchronize()
                costs['physical_setup_seconds'] += time.perf_counter()-began
                record('saved_teacher',n,index,high,x,q,parent=parent)
                if n==128:
                    record('original_source_zero',n,index,high,torch.zeros_like(x),q,parent=parent)
                del low,high,t,s,q,x
            del payload,bank
        n=256;limit();began=time.perf_counter()
        parent=policy['n256_material_parent'];assert sha(parent['path'])==parent['sha256']
        data=torch.load(parent['path'],map_location='cpu',weights_only=True)
        cfg,regenerated=source_family(n,16,4626)
        for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):
            assert torch.equal(data[key],regenerated[key]),key
        costs['parent_hash_load_seconds'] += time.perf_counter()-began
        sys.path.insert(0,str(Path(policy['frozen_capture_code'])))
        from direct_segmented_tsa import DirectSegmentedTSA
        for index in policy['development_ids']:
            limit();torch.cuda.synchronize();began=time.perf_counter()
            low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
            torch.cuda.synchronize();costs['physical_setup_seconds'] += time.perf_counter()-began
            began=time.perf_counter();base=build_base(low,'tsa16')
            wrapper=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False)
            torch.cuda.synchronize();costs['base_capture_setup_seconds'] += time.perf_counter()-began
            began=time.perf_counter();x=wrapper(low.source).clone()
            torch.cuda.synchronize();costs['preconditioner_seconds'] += time.perf_counter()-began
            counters['preconditioner_actions'] += 1
            assert x.dtype==torch.complex128
            image,elapsed=timed_H(high,x);q=image.clone()
            record('manufactured_M_source',n,index,high,x,q,parent=parent,
                   gpu_image=image,gpu_seconds=elapsed)
            record('original_source_zero',n,index,high,torch.zeros_like(x),high.source,parent=parent)
            del base,wrapper,low,high,x,q,image
        assert counters['saved_teacher_certificates']==4
        assert counters['manufactured_fixture_certificates']==2
        assert counters['negative_control_certificates']==4
        assert counters['full_solves']==0 and counters['training_steps']==0
        assert len(rows)==policy['max_certificates']
        assert counters['explicit_gpu_H_actions']==policy['max_explicit_H_actions']
        assert counters['preconditioner_actions']==policy['max_M_actions']
        limit();save(True)
    except BaseException as exc:
        save(False,dict(type=type(exc).__name__,message=str(exc)))
        raise
    print(json.dumps(dict(complete=True,counters=counters,costs=costs,
                          elapsed_seconds=time.perf_counter()-START),indent=2))

if __name__=='__main__':
    run()
