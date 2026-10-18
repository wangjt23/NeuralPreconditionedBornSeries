"""Lazy loading for the original tensor-only CDR sample format."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class ConvectionDiffusionDataset(Dataset):
    """Load ``kappa (N,N), v (N,N,2), f (N,N)`` as four CPU channels."""

    def __init__(self, data_path, limit=None, resolution=None):
        if limit is not None and limit <= 0:
            raise ValueError("sample limit must be positive")
        self.paths = sorted(Path(data_path).glob("sample_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No sample_*.pt files in {data_path}; generate data first")
        if limit is not None:
            if len(self.paths) < limit:
                raise ValueError(f"{data_path}: requested {limit} samples, found {len(self.paths)}")
            self.paths = self.paths[:limit]
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(sample, dict) or not {"kappa", "v", "f"} <= sample.keys():
            raise ValueError(f"{path}: expected a tensor dictionary containing kappa, v, f")
        kappa, velocity, rhs = (sample[key] for key in ("kappa", "v", "f"))
        if not all(isinstance(field, torch.Tensor) for field in (kappa, velocity, rhs)):
            raise ValueError(f"{path}: all fields must be tensors")
        if (kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]
                or rhs.shape != kappa.shape or velocity.shape != (*kappa.shape, 2)):
            raise ValueError(f"{path}: expected kappa/f (N,N) and v (N,N,2) on a square grid")
        if self.resolution is not None and kappa.shape != (self.resolution, self.resolution):
            raise ValueError(f"{path}: sample resolution differs from model resolution {self.resolution}")
        fields = torch.cat((kappa[None], velocity.permute(2, 0, 1), rhs[None])).float()
        if not torch.isfinite(fields).all() or not (kappa > 0).all():
            raise ValueError(f"{path}: expected finite fields and strictly positive diffusion")
        return fields
