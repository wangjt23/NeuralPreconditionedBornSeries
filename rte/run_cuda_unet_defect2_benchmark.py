"""Run the defect2 CUDA benchmark in actual inference mode."""

import runpy
import torch

torch.set_grad_enabled(False)
runpy.run_module("rte.benchmark_cuda_unet_defect2", run_name="__main__")
