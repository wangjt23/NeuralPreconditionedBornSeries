"""Inference-only CUDA operators for the scale-aware defect2 U-Net."""

from pathlib import Path
import importlib.util
import os
import subprocess
import sysconfig
import tempfile
import types

import torch
import torch.nn.functional as F

_EXTENSION = None


def load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("defect2 CUDA operators require a CUDA device")
    from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths

    source_dir = Path(__file__).resolve().parent
    build_dir = Path(tempfile.gettempdir()) / "npbs_unet_defect2_cuda_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    name = "npbs_unet_defect2_cuda"
    shared = build_dir / (name + ".so")
    cpp_obj = build_dir / "cuda_unet_defect2.cpp.o"
    cu_obj = build_dir / "cuda_unet_defect2.cu.o"
    includes = []
    for path in include_paths(cuda=True) + [sysconfig.get_path("include")]:
        includes.extend(["-I", path])
    abi = str(int(torch.compiled_with_cxx11_abi()))
    common = ["-std=c++17", "-O3", "-fPIC", f"-D_GLIBCXX_USE_CXX11_ABI={abi}", f"-DTORCH_EXTENSION_NAME={name}"]
    cpp_cmd = [os.environ.get("CXX", "g++"), *common, *includes, "-c", str(source_dir / "cuda_unet_defect2.cpp"), "-o", str(cpp_obj)]
    nvcc = str(Path(CUDA_HOME or "/usr/local/cuda") / "bin" / "nvcc")
    major, minor = torch.cuda.get_device_capability()
    cu_cmd = [nvcc, "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
              f"-D_GLIBCXX_USE_CXX11_ABI={abi}", f"-DTORCH_EXTENSION_NAME={name}",
              f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}", *includes, "-c",
              str(source_dir / "cuda_unet_defect2_v2.cu"), "-o", str(cu_obj)]
    lib_flags = []
    for path in library_paths(cuda=True):
        lib_flags.extend(["-L", path])
    link_cmd = [os.environ.get("CXX", "g++"), "-shared", str(cpp_obj), str(cu_obj), "-o", str(shared), *lib_flags,
                "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-ltorch_python", "-lc10", "-lc10_cuda", "-lcudart",
                "-Wl,-rpath," + str(library_paths(cuda=True)[0]), "-Wl,-rpath," + str(library_paths(cuda=True)[1])]
    if not shared.exists():
        for command in (cpp_cmd, cu_cmd, link_cmd):
            subprocess.run(command, check=True)
    spec = importlib.util.spec_from_file_location(name, shared)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    _EXTENSION = extension
    return _EXTENSION


def fused_circular_conv2d(x, weight, bias=None, stride=1):
    if not x.is_cuda or torch.is_grad_enabled() or x.dtype != torch.float32:
        padding = weight.shape[-1] // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="circular") if padding else x
        return F.conv2d(x, weight, bias=bias, stride=stride)
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    return load_extension().circular_conv2d(x.contiguous(), weight.contiguous(), bias.contiguous(), int(stride))


def fused_scale_gate(x, coeff, local_weight, local_bias, global_bias, matrix=None, bounded_gain=0.0, mix_gain=0.0):
    if not x.is_cuda or torch.is_grad_enabled() or x.dtype != torch.float32:
        local = F.conv2d(coeff, local_weight, local_bias)
        out = x * (1.0 + float(bounded_gain) * torch.tanh(local + global_bias[:, :, None, None]))
        if matrix is not None:
            out = out + float(mix_gain) * torch.einsum("bcd,bdhw->bchw", matrix, x) / (x.shape[1] ** 0.5)
        return out
    if local_bias is None:
        local_bias = torch.zeros(local_weight.shape[0], device=x.device, dtype=x.dtype)
    if matrix is None:
        matrix = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
    return load_extension().fused_scale_gate(x.contiguous(), coeff.contiguous(), local_weight.contiguous(),
                                              local_bias.contiguous(), global_bias.contiguous(), matrix.contiguous(),
                                              float(bounded_gain), float(mix_gain))


def _cuda_conv_forward(self, x):
    if x.is_cuda and not torch.is_grad_enabled():
        bias = self.conv.bias if self.conv.bias is not None else self._cuda_zero_bias
        return load_extension().circular_conv2d(x.contiguous(), self.conv.weight.contiguous(), bias.contiguous(), int(self.conv.stride[0]))
    return self._cuda_original_forward(x)


def _cuda_gate_forward(self, x, coeff):
    if not x.is_cuda or torch.is_grad_enabled():
        return self._cuda_original_forward(x, coeff)
    if coeff.shape[-2:] != x.shape[-2:]:
        coeff = F.interpolate(coeff, size=x.shape[-2:], mode="bilinear", align_corners=False)
    summary = torch.cat([coeff.mean(dim=(-2, -1)),
                         torch.sqrt(torch.clamp(coeff.var(dim=(-2, -1), unbiased=False), min=1e-12)),
                         coeff.amin(dim=(-2, -1)), coeff.amax(dim=(-2, -1)),
                         self._cuda_level_code.view(1, 4).expand(coeff.shape[0], -1)], dim=1)
    global_bias = self.summary(summary)
    matrix = torch.tanh(self.mix(summary)) if self.channel_mix else None
    matrix = matrix.view(coeff.shape[0], x.shape[1], x.shape[1]) if matrix is not None else None
    return fused_scale_gate(x, coeff, self.local.conv.weight, self.local.conv.bias, global_bias, matrix,
                            self._cuda_bounded_gain, self._cuda_mix_gain)


def enable_cuda_defect2(model):
    """Install the fused kernels and cache all inference-only scalar metadata."""
    load_extension()
    from rte.npbs_unet_ablation import _CircularConv2d, ScaleAwareCoeffGate
    convolutions = 0
    gates = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, _CircularConv2d):
                if module.conv.bias is None:
                    module._cuda_zero_bias = torch.zeros(module.conv.out_channels, device=module.conv.weight.device, dtype=module.conv.weight.dtype)
                if not getattr(module, "_cuda_defect2_patched", False):
                    module._cuda_original_forward = module.forward
                    module.forward = types.MethodType(_cuda_conv_forward, module)
                    module._cuda_defect2_patched = True
                convolutions += 1
            elif isinstance(module, ScaleAwareCoeffGate):
                module._cuda_level_code = module.level_code(1, module.gain_logit.device, torch.float32)[0].detach()
                module._cuda_bounded_gain = float(module.gate_scale * (1.0 + 0.5 * torch.tanh(module.gain_logit)).detach().cpu())
                module._cuda_mix_gain = float(0.5 * module.gate_scale * (1.0 + 0.5 * torch.tanh(module.mix_gain_logit)).detach().cpu()) if module.channel_mix else 0.0
                if not getattr(module, "_cuda_defect2_patched", False):
                    module._cuda_original_forward = module.forward
                    module.forward = types.MethodType(_cuda_gate_forward, module)
                    module._cuda_defect2_patched = True
                gates += 1
    model.cuda_defect2_operator = True
    return {"circular_convolutions": convolutions, "scale_gates": gates}
