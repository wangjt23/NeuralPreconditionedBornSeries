"""Convert a trusted legacy tuple into a portable tensor-only data dictionary."""

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nonlinear.data import as_fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-pickle", action="store_true",
                        help="Allow arbitrary legacy pickle code; use only for data you trust")
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=not args.trust_pickle)
    if isinstance(payload, dict):
        state, correction = payload["u"], payload.get("du")
    elif isinstance(payload, (list, tuple)) and len(payload) >= 2:
        state, correction = payload[:2]
    else:
        raise ValueError("expected a data dictionary or a (u, du[, solution]) tuple")
    data = {"u": as_fields(torch.as_tensor(state)),
            "metadata": {"converted_legacy_data": True}}
    if correction is not None:
        data["du"] = as_fields(torch.as_tensor(correction))
        if data["u"].shape != data["du"].shape:
            raise ValueError("u and du must have the same shape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved {len(data['u'])} samples to {args.output}; unused reference fields were omitted")


if __name__ == "__main__":
    main()
