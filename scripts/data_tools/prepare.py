#!/usr/bin/env python3
"""Prepare alpha, static GT and separate cumulative insert/delete experiments."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import struct
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
NAMES = ("base_img_emb.fvecs", "base_text_emb.fvecs",
         "query_img_emb.fvecs", "query_text_emb.fvecs")


def inspect_fvecs(path):
    """Validate every dimension header, record length and float value."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError(f"Empty/truncated fvecs: {path}")
        dim, = struct.unpack("<i", header)
        if dim <= 0 or size % (4 * (dim + 1)):
            raise ValueError(f"Invalid dimension/record size: {path}")
        stream.seek(0)
        fmt = struct.Struct(f"<i{dim}f")
        count = size // fmt.size
        for i in range(count):
            row = fmt.unpack(stream.read(fmt.size))
            if row[0] != dim or not all(math.isfinite(x) for x in row[1:]):
                raise ValueError(f"Invalid record {i}: {path}")
    return {"count": count, "dim": dim, "record_bytes": fmt.size, "bytes": size}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_backend():
    source = HERE / "exact_gt.cpp"
    binary = HERE / ".build" / "exact_gt"
    binary.parent.mkdir(exist_ok=True)
    if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        command = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(command + ["-O3", "-std=c++17", "-pthread", str(source),
                                  "-o", str(binary)], check=True)
    return binary


def copy_prefix(source, target, size):
    with source.open("rb") as src, target.open("xb") as dst:
        remaining = size
        while remaining:
            block = src.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError(f"Source truncated while copying: {source}")
            dst.write(block)
            remaining -= len(block)


def generate(args):
    dataset = args.dataset.resolve()
    output = (args.output or dataset / "prepared").resolve()
    if output.exists():
        raise ValueError(f"Output already exists (choose another --output): {output}")
    if any(c in str(output) for c in "\n\r\t"):
        raise ValueError("Output path must not contain tabs or newlines")
    if args.rounds <= 0 or args.k <= 0 or args.threads <= 0:
        raise ValueError("rounds, k and threads must be positive")
    if not (0 <= args.alpha_min <= args.alpha_max <= 1):
        raise ValueError("Require 0 <= alpha-min <= alpha-max <= 1")
    paths = {name: dataset / name for name in NAMES}
    info = {name: inspect_fvecs(path) for name, path in paths.items()}
    n = info[NAMES[0]]["count"]
    nq = info[NAMES[2]]["count"]
    if info[NAMES[1]]["count"] != n or info[NAMES[3]]["count"] != nq:
        raise ValueError("Paired modality counts must agree")
    if any(info[NAMES[i]]["dim"] != info[NAMES[i + 2]]["dim"] for i in (0, 1)):
        raise ValueError("Base/query dimensions must agree within each modality")
    try:
        rate = Decimal(args.update_rate)
        batch_decimal = Decimal(n) * rate
    except InvalidOperation as exc:
        raise ValueError("Invalid update rate") from exc
    if not rate.is_finite() or not 0 < rate <= 1 or batch_decimal != batch_decimal.to_integral_value():
        raise ValueError("update-rate must give a positive integral batch (relative to FULL base)")
    batch = int(batch_decimal)
    initial = n // 2
    if (batch <= 0 or initial < args.k or n - args.rounds * batch < args.k or
            initial + args.rounds * batch > n):
        raise ValueError("Not enough vectors for k, half-base insertion or requested update rounds")
    backend = args.backend.resolve() if args.backend else build_backend()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        rng = random.Random(args.seed)
        alpha_path = stage / "query_alpha.fvecs"
        with alpha_path.open("wb") as binary, (stage / "query_alpha.txt").open("w") as text:
            for qi in range(nq):
                # Round to float32 before both storage and GT use.
                value, = struct.unpack("<f", struct.pack("<f", rng.uniform(args.alpha_min, args.alpha_max)))
                binary.write(struct.pack("<if", 1, value))
                text.write(f"{qi} {value:.9g}\n")
        insert_base = stage / "base_for_insert"
        insert_base.mkdir()
        for name in NAMES[:2]:
            copy_prefix(paths[name], insert_base / name, initial * info[name]["record_bytes"])

        jobs = [{"file": "static_gt.ivecs", "active_range": [0, n]}]
        experiments = {}
        for mode in ("insert", "delete"):
            (stage / mode / "traces").mkdir(parents=True)
            (stage / mode / "gt").mkdir()
            states, traces = [], []
            for r in range(args.rounds + 1):
                active = [0, initial + r * batch] if mode == "insert" else [r * batch, n]
                job = {"file": f"{mode}/gt/round_{r:02d}.ivecs", "active_range": active}
                jobs.append(job)
                states.append({"after_updates": r, **job})
                if r == args.rounds:
                    break
                start = initial + r * batch if mode == "insert" else r * batch
                trace = f"{mode}/traces/round_{r:02d}_trace.bin"
                with (stage / trace).open("wb") as stream:
                    stream.write(struct.pack("<i", batch))
                    stream.write(struct.pack(f"<{batch}i", *range(start, start + batch)))
                traces.append({"file": trace, "ids": [start, start + batch], "count": batch,
                               "gt_before": job["file"],
                               "gt_after": f"{mode}/gt/round_{r + 1:02d}.ivecs"})
            experiments[mode] = {"id_flag": 0 if mode == "insert" else 1,
                                 "states": states, "traces": traces}
        job_path = stage / "jobs.tsv"
        with job_path.open("w") as stream:
            for job in jobs:
                begin, end = job["active_range"]
                stream.write(f"{begin}\t{end}\t{stage / job['file']}\n")
        print(f"Base={n}, queries={nq}, initial insert base={initial}; "
              f"{args.rounds} rounds x {batch} IDs per mode", flush=True)
        subprocess.run([str(backend), *(str(paths[name]) for name in NAMES),
                        str(alpha_path), str(job_path), str(args.k), str(args.threads)], check=True)
        job_path.unlink()
        manifest = {
            "schema_version": 1, "dataset": str(dataset), "k": args.k,
            "distance": "alpha * L2(img) + (1-alpha) * L2(text)",
            "distance_note": "L2 = sqrt(sum((base-query)^2)); no normalization; ties by original ID",
            "id_ranges": "zero-based, half-open [begin,end); never renumbered",
            "alpha": {"file": "query_alpha.fvecs", "min": args.alpha_min, "max": args.alpha_max,
                      "seed": args.seed, "distribution": "uniform", "count": nq},
            "update_rate": str(rate), "update_rate_denominator": "full base count",
            "batch_size": batch, "rounds": args.rounds, "base_for_insert_range": [0, initial],
            "inputs": {name: {**info[name], "sha256": sha256(path)} for name, path in paths.items()},
            "static_gt": jobs[0], "experiments": experiments,
        }
        manifest["outputs"] = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                               for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(f"Prepared dataset: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=HERE.parent / "datasets" / "oi_mini")
    parser.add_argument("--output", type=Path, help="Default: DATASET/prepared; must not exist")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--update-rate", default="0.01", help="Fraction of FULL base per round")
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--backend", type=Path, help="Use precompiled exact_gt instead of compiling")
    args = parser.parse_args()
    try:
        generate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
