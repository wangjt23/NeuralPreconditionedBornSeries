"""Measure surviving transport errors in radial, characteristic and angular coordinates.

Uses existing development validation tensors, never the frozen comparison set.
Reference solutions and outer Krylov arithmetic are FP64 at unchanged geometry.
"""

import argparse
import json
from pathlib import Path

import torch

from rte.compiled_hss import CompiledHSS
from rte.mixed_precision_transport import high_precision_copy
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint, make_operator, save_checkpoint, source_from_data
from rte.run_neural_resolution_study import build_model
from rte.run_residual_subspace_experiment import write_csv
from rte.train_krylov_transport import short_fgmres


def energy_coordinates(op, value, coarse_size=32):
    spectrum = torch.fft.fftn(value, dim=(-2, -1), norm='ortho')
    energy = spectrum.abs().square()
    total = energy.sum().clamp_min(1e-300)
    kx, ky = op.kx*op.dx*op.shape[0]/(2*torch.pi), op.ky*op.dx*op.shape[1]/(2*torch.pi)
    outside = (kx.abs() >= coarse_size/2) | (ky.abs() >= coarse_size/2)
    scale = op.sigma_t.mean((-2, -1), keepdim=True)
    near = op._stream_symbol(len(value), value.dtype, value.device).abs() <= scale
    angular = torch.fft.fft(value, dim=1, norm='ortho').abs().square()
    fraction = lambda mask: float((energy*mask).sum()/total)
    high_energy = (energy*outside).sum()
    return dict(outside_coarse_fraction=fraction(outside), near_characteristic_fraction=fraction(near),
        outside_and_near_fraction=fraction(outside & near),
        near_fraction_within_outside=float((energy*(outside & near)).sum()/high_energy.clamp_min(1e-300)),
        outside_near_volume_fraction=float((outside & near).double().mean()),
        scalar_angular_fraction=float(angular[:, 0].sum()/angular.sum().clamp_min(1e-300)))


def commutator_metrics(op, value):
    collision = lambda v: op.sigma_t*v-op.scattering_op(v)
    def constant_collision(v):
        phase = op.phase_matrix.to(v.dtype)
        scatter = torch.einsum('lm,bmxy->blxy', phase, op.weights[None, :, None, None]*v)
        return op.sigma_t.mean((-2, -1), keepdim=True)*v-op.sigma_s.mean((-2, -1), keepdim=True)*scatter
    sc, cs = op.streaming_op(collision(value)), collision(op.streaming_op(value))
    constant = op.streaming_op(constant_collision(value))-constant_collision(op.streaming_op(value))
    total = sc-cs
    norm = lambda v: torch.linalg.vector_norm(v)
    denominator = (norm(sc)+norm(cs)).clamp_min(1e-300)
    return dict(normalized_commutator=float(norm(total)/denominator),
        constant_material_commutator_part=float(norm(constant)/denominator),
        varying_material_commutator_part=float(norm(total-constant)/denominator))


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows, probes = [], []
    specs = json.loads(Path(args.specs).read_text())
    for snapshot in args.datasets:
        payload = torch.load(snapshot, map_location='cpu', weights_only=True)
        data = payload['data']
        if payload['fingerprint'] != dataset_fingerprint(data):
            raise ValueError('dataset fingerprint mismatch')
        size = data['sigma_t'].shape[-1]
        cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, regimes='mixed', eta_scale=0.)
        for index in range(min(args.count, len(data['sigma_t']))):
            ids = torch.tensor([index])
            source = source_from_data(cfg, data, ids, torch.device('cpu'))
            low = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], source, torch.device('cpu'))
            high = high_precision_copy(low)
            exact, trace, _, _ = fgmres_real_stacked_with_trace(high.A_op, high.source, CompiledHSS(high, 8),
                tol=1e-10, maxiter=120, restart=20, max_restarts=5)
            if trace[-1] > 1e-9:
                raise RuntimeError(f'reference failed N{size} case {index}: {trace[-1]}')
            rates = low.sigma_t-low.sigma_s*torch.fft.fft(low.phase_matrix[0]*low.weights).real[None, :, None, None]
            low_rate, high_rate = float(rates.min()), float(rates.max())
            for spec in specs:
                checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True) if spec.get('checkpoint') else None
                model = build_model(spec, low, checkpoint).eval()
                squared, approximate = short_fgmres(high.A_op, model, high.source, 3)
                residual, error = high.source-high.A_op(approximate), exact-approximate
                common = dict(size=size, index=index, method=spec['label'], reference_residual=trace[-1],
                    three_step_relative_residual=float(squared.sqrt()),
                    relative_solution_error=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(exact)),
                    collision_condition=high_rate/low_rate,
                    classical_hss_bound=(high_rate**.5-low_rate**.5)/(high_rate**.5+low_rate**.5))
                for kind, value in [('residual', residual), ('error', error)]:
                    rows.append(dict(**common, field=kind, **energy_coordinates(high, value), **commutator_metrics(high, value)))
                save_checkpoint(out/f'fields_n{size}_{index}_{spec["label"]}.pt',
                    dict(error=error.cpu(), residual=residual.cpu(), reference=exact.cpu()))
                # Angle-zero manufactured errors: kx=0 is characteristic even
                # at large ky. Compare equally high-frequency streaming modes.
                x = torch.arange(size, dtype=torch.float64)/size
                for kx, ky in [(0, 1), (0, size//4), (size//4, 0), (size//4, size//4)]:
                    plane = torch.exp(2j*torch.pi*(kx*x[:, None]+ky*x[None, :]))
                    error_probe = torch.zeros_like(high.source)
                    error_probe[:, 0] = plane
                    rhs = high.A_op(error_probe)
                    remaining = error_probe-model(rhs.to(torch.complex64)).to(torch.complex128)
                    probes.append(dict(size=size, index=index, method=spec['label'], kx=kx, ky=ky,
                        error_ratio=float(torch.linalg.vector_norm(remaining)/torch.linalg.vector_norm(error_probe))))
                write_csv(out/'coordinates.csv', rows)
                write_csv(out/'plane_waves.csv', probes)
            print('analyzed development medium', size, index, flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), specs=specs,
        scope='development-only structural diagnostic; not additional confirmation or timing',
        masks='strict N32 coarse band; near characteristic iff abs(Omega.k) <= mean(sigma_t)',
        commutator='exact discrete [S,C]; C split into spatial average and variation, not an alias-free Leibniz formula'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--specs', required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
