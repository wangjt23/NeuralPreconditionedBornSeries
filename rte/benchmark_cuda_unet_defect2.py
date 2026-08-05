"""Correctness and inference benchmark for the defect2 CUDA operators."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rte.cuda_unet_defect2 import enable_cuda_defect2, fused_circular_conv2d, fused_scale_gate
from rte.npbs_unet_ablation import NPBSUNetPreconditioner
from rte.run_hard_case_experiments import make_case


def timed(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def relative_error(a, b):
    return float((a - b).abs().max() / torch.clamp(b.abs().max(), min=1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="mixed")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="rte/cuda_unet_defect2_benchmark.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(2026)
    device = torch.device("cuda")
    rte, _ = make_case(args.case, args.size, args.n_angles, torch.float32, 0.1, device=device)
    model_ref = NPBSUNetPreconditioner(
        rte,
        form="defect2",
        base_kind="transport_dsa",
        width=16,
        levels=3,
        gate_scale=0.1,
        scale_aware_gates=True,
        scale_gate_channel_mix=True,
    ).to(device).eval()
    model_cuda = NPBSUNetPreconditioner(
        rte,
        form="defect2",
        base_kind="transport_dsa",
        width=16,
        levels=3,
        gate_scale=0.1,
        scale_aware_gates=True,
        scale_gate_channel_mix=True,
    ).to(device).eval()
    model_cuda.load_state_dict(model_ref.state_dict())
    residual = rte.source.clone()

    x = torch.randn(2, 7, args.size, args.size, device=device)
    weight = torch.randn(11, 7, 3, 3, device=device)
    bias = torch.randn(11, device=device)
    reference_conv = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), weight, bias=bias)
    cuda_conv = fused_circular_conv2d(x, weight, bias)

    gate_x = torch.randn(2, 16, args.size, args.size, device=device)
    gate_coeff = torch.randn(2, 8, args.size, args.size, device=device)
    gate_weight = torch.randn(16, 8, 1, 1, device=device)
    gate_bias = torch.randn(16, device=device)
    gate_global = torch.randn(2, 16, device=device)
    gate_matrix = torch.randn(2, 16, 16, device=device)
    gate_gain = 0.1
    gate_mix_gain = 0.05
    reference_gate = gate_x * (
        1.0
        + gate_gain
        * torch.tanh(F.conv2d(gate_coeff, gate_weight, gate_bias) + gate_global[:, :, None, None])
    )
    reference_gate = reference_gate + gate_mix_gain * torch.einsum("bcd,bdhw->bchw", gate_matrix, gate_x) / (16.0**0.5)
    cuda_gate = fused_scale_gate(
        gate_x,
        gate_coeff,
        gate_weight,
        gate_bias,
        gate_global,
        matrix=gate_matrix,
        bounded_gain=gate_gain,
        mix_gain=gate_mix_gain,
    )

    reference_output = model_ref(residual)
    enable_counts = enable_cuda_defect2(model_cuda)
    cuda_output = model_cuda(residual)
    torch.cuda.synchronize()
    reference_ms = timed(lambda: model_ref(residual), args.warmup, args.iterations)
    cuda_ms = timed(lambda: model_cuda(residual), args.warmup, args.iterations)

    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "case": args.case,
        "size": args.size,
        "n_angles": args.n_angles,
        "operator_counts": enable_counts,
        "primitive_max_abs": {
            "circular_conv2d": float((cuda_conv - reference_conv).abs().max()),
            "fused_scale_gate": float((cuda_gate - reference_gate).abs().max()),
        },
        "primitive_relative_error": {
            "circular_conv2d": relative_error(cuda_conv, reference_conv),
            "fused_scale_gate": relative_error(cuda_gate, reference_gate),
        },
        "model_max_abs": float((cuda_output - reference_output).abs().max()),
        "model_relative_error": relative_error(cuda_output, reference_output),
        "reference_ms": reference_ms,
        "cuda_operator_ms": cuda_ms,
        "speedup": reference_ms / cuda_ms,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
