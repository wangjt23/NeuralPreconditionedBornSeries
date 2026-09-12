#!/usr/bin/env python3
"""Export the completed laptop dataset and hash its artifacts; never train or solve."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / "rte/handoff/2026-09-12"
RUN = ROOT / "rte/results/collision_transport_ensemble_seed1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export():
    import numpy
    import scipy
    import torch
    from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint
    from rte.train_collision_transport_ensemble import material_data

    settings = json.loads((RUN / "settings.json").read_text())
    cfg = ExperimentConfig(dataset_size=settings["train_size"], eval_size=settings["val_size"],
                           test_size=settings["test_size"], size=settings["size"],
                           n_angles=settings["n_angles"], anisotropy_g=settings["g"],
                           regimes=settings["regimes"], batch_size=settings["batch_size"], eta_scale=0.)
    expected = {**settings["fingerprints"], "test": (RUN / "test_fingerprint.txt").read_text().strip()}
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for split, count, offset in (("train", settings["train_size"], 1000),
                                 ("validation", settings["val_size"], 2000),
                                 ("test", settings["test_size"], 3000)):
        data = material_data(cfg, count, settings["seed"] + offset)
        fingerprint = dataset_fingerprint(data)
        if fingerprint != expected[split]:
            raise ValueError(f"Regenerated {split} differs from original run; do not overwrite evidence")
        path = BUNDLE / f"{split}.pt"
        if path.exists():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            if dataset_fingerprint(previous["data"]) != fingerprint:
                raise ValueError(f"Existing {split} snapshot differs")
        else:
            torch.save({"schema": 1, "data": data, "g": settings["g"],
                        "fingerprint": fingerprint}, path)
        print(f"{split}: {count} materials, {fingerprint}")
    checkpoint = torch.load(RUN / "last.pt", map_location="cpu", weights_only=True)
    metadata = BUNDLE / "provenance.json"
    if not metadata.exists():
        metadata.write_text(json.dumps({"status": "local training stopped; epoch budget completed",
            "checkpoint_epoch": checkpoint["epoch"], "selected_epoch": checkpoint["best_epoch"],
            "updates": checkpoint["updates"], "fingerprints": expected,
            "python": sys.version, "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "local_parent": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "fetched_remote_parent": subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip(),
            "residuals": "Fresh online probes are not stored; last.pt contains epoch-boundary RNG state",
            "sources": "Compact source parameters are included in every split snapshot"}, indent=2) + "\n")


def manifest_paths():
    paths = []
    for folder in (ROOT / "rte/results", BUNDLE):
        paths.extend(p for p in folder.rglob("*") if p.is_file() and p.name != "SHA256SUMS.tsv"
                     and p.name != ".DS_Store" and "__pycache__" not in p.parts)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Read-only SHA-256 check, no Torch required")
    args = parser.parse_args()
    manifest = BUNDLE / "SHA256SUMS.tsv"
    if args.verify:
        with manifest.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        for row in rows:
            path = (ROOT / row["path"]).resolve()
            if ROOT not in path.parents or not path.is_file():
                raise ValueError(f"Unsafe or missing artifact: {row['path']}")
            if path.stat().st_size != int(row["bytes"]) or digest(path) != row["sha256"]:
                raise ValueError(f"Artifact mismatch: {row['path']}")
        print(f"VERIFIED {len(rows)} files, {sum(int(r['bytes']) for r in rows):,} bytes")
        return
    export()
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256"])
        for path in manifest_paths():
            writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)])
    print(f"Saved {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
