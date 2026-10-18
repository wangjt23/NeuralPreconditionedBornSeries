"""Lazy OpenFWI velocity loading and physical parameter presets.

Dataset indices preserve the original experiment splits. Raw 3D volumes are
resized one at a time, rather than stacking hundreds of native volumes in RAM.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class unspvd_sos_Dataset(Dataset):
    """Return velocity tensors shaped (1, N, N) or (1, N, N, N).

    In 2D, test_point is a zero-based offset across files of 500 samples.
    Kimberlina uses years 0..35 for training and year 100 for testing; its
    year-based split intentionally ignores test_point, matching the source.
    """

    _TRAIN_YEARS = (0, 1, 2, 5, 10, 15, 20, 25, 30, 35)

    def __init__(self, sos_path, data_size=256, data_type="openfwi_CurveFault_B",
                 data_num=3000, test=False, test_point=6000):
        super().__init__()
        self.root = Path(sos_path).expanduser()
        self.size = int(data_size)
        self.data_type = data_type
        self.count = int(data_num)
        self.offset = int(test_point) if test else 0
        self.test = bool(test)
        self._cache = {}
        if self.count <= 0 or self.size <= 0 or self.offset < 0:
            raise ValueError("data_num/data_size must be positive and test_point nonnegative")
        if data_type not in ("openfwi_CurveVel_A", "openfwi_CurveFault_B", "openfwi3d"):
            raise ValueError(f"Unsupported dataset {data_type!r}; see helmholtz/README.md")
        if data_type == "openfwi3d" and self.count > (63 if test else 630):
            raise ValueError("Kimberlina split contains 63 test or 630 training volumes")

    def __len__(self):
        return self.count

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = {}  # Open file mappings independently in loader workers.
        return state

    def _load_npy(self, filename):
        path = self.root / filename
        if filename not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[filename] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._cache[filename]

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return torch.stack([self[i] for i in indices])
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.data_type == "openfwi3d":
            year = 100 if self.test else self._TRAIN_YEARS[index // 63]
            piece = index % 63 + 1
            path = self.root / f"year{year}_cut{piece}.bin"
            volume = np.memmap(path, mode="r", dtype=np.float32, shape=(400, 400, 350))
            tensor = torch.from_numpy(np.array(volume, copy=True))[None, None]
            return F.interpolate(tensor, size=(self.size,) * 3,
                                 mode="trilinear", align_corners=False)[0]
        sample = self.offset + index
        file_id, within_file = divmod(sample, 500)
        if self.data_type == "openfwi_CurveVel_A":
            filename = f"model{file_id + 1}.npy"
        else:
            filename = f"vel{file_id // 36 + 6}_1_{file_id % 36}.npy"
        array = self._load_npy(filename)
        tensor = torch.from_numpy(np.array(array[within_file], dtype=np.float32, copy=True))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"Expected velocity shape (1, H, W), got {tuple(tensor.shape)}")
        return F.interpolate(tensor[None], size=(self.size,) * 2,
                             mode="bilinear", align_corners=False)[0]


@dataclass
class DataConfig:
    """Physical parameters and local velocity-data location."""

    size: int
    freq: float
    dx: float
    data_num: int
    sos_mean: float
    sos_max: float
    sos_min: float
    dir_path: str
    data_type: str


_PRESETS = {
    "openfwi_CurveVel_A": dict(size=256, freq=64000., dx=1 / 256,
        data_num=30000, sos_mean=3000., sos_max=4500., sos_min=1500.,
        dir_path="openfwi/CurveVel_A/model", data_type="openfwi_CurveVel_A"),
    "openfwi_CurveFault_B": dict(size=256, freq=64935., dx=1 / 256,
        data_num=54000, sos_mean=3035.7, sos_max=4499., sos_min=1500.,
        dir_path="openfwi/CurveFault_B", data_type="openfwi_CurveFault_B"),
    "openfwi3d": dict(size=192, freq=76800., dx=1 / 192,
        data_num=630, sos_mean=2999.217642, sos_max=4201.001465, sos_min=2636.670410,
        dir_path="openfwi/3DKimberlina-V1", data_type="openfwi3d"),
}


def Dataconfig(user_data_type, data_dir=None, **overrides):
    """Load a preset; explicit data_dir overrides NPBS_DATA_ROOT (default: data).

    Numeric overrides permit changing grid size/frequency without editing source.
    Preset values preserve the original experiment configuration.
    """
    if user_data_type not in _PRESETS:
        raise ValueError(f"Unknown data preset {user_data_type!r}; choose from {sorted(_PRESETS)}")
    values = dict(_PRESETS[user_data_type])
    root = os.environ.get("NPBS_DATA_ROOT", "data")
    values["dir_path"] = os.path.expandvars(os.path.expanduser(
        str(data_dir) if data_dir is not None else os.path.join(root, values["dir_path"])
    ))
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"Unknown data configuration fields: {sorted(unknown)}")
    values.update(overrides)
    return DataConfig(**values)
