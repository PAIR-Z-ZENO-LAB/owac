"""Copy unchanged training files to private storage with bounded buffers and hashes."""

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from owac.raw.all_bench_distillation import load_inventory


def cache_path(source, root):
    return Path(root) / (hashlib.sha256(str(source).encode()).hexdigest() + ".npz")


def copy_file(source, target):
    temporary = target.with_suffix(".partial")
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("wb") as writer:
        os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while block := reader.read(8 * 1024**2):
            writer.write(block)
            digest.update(block)
            os.posix_fadvise(
                reader.fileno(), reader.tell() - len(block), len(block), os.POSIX_FADV_DONTNEED
            )
        writer.flush()
        os.fdatasync(writer.fileno())
    with temporary.open("rb") as reader:
        copied = hashlib.file_digest(reader, "sha256").hexdigest()
        os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if copied != digest.hexdigest():
        raise RuntimeError(f"Private copy differs from source: {source}")
    temporary.replace(target)
    return {
        "source": str(source),
        "target": str(target),
        "bytes": target.stat().st_size,
        "sha256": copied,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sources, payload = load_inventory(args.inventory)
    selected = [
        source
        for source in sources
        if source.domain == "libero" and source.split in ("train", "validation")
    ]
    paths = sorted({source.path for source in selected})
    files = args.output / "files"
    files.mkdir(parents=True, exist_ok=True)
    required = sum(path.stat().st_size for path in paths)
    if required > shutil.disk_usage(args.output).free:
        raise OSError("Insufficient private storage for the complete unchanged data subset")
    targets = {path: cache_path(path, files) for path in paths}
    started, transferred = time.time(), 0
    manifest = args.output / "files.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as pool, manifest.open("w") as stream:
        futures = [pool.submit(copy_file, path, targets[path]) for path in paths]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            transferred += row["bytes"]
            if completed % 16 == 0 or completed == len(paths):
                status = {
                    "files": completed,
                    "total_files": len(paths),
                    "bytes": transferred,
                    "total_bytes": required,
                    "elapsed_seconds": time.time() - started,
                }
                (args.output / "status.json").write_text(json.dumps(status, indent=2))
                print(json.dumps(status), flush=True)
    remap = {source.group_id: str(targets[source.path]) for source in selected}
    for group in payload["groups"]:
        if group["group_id"] in remap:
            group["path"] = remap[group["group_id"]]
    (args.output / "inventory.json").write_text(json.dumps(payload, indent=2))
    (args.output / "complete.json").write_text(
        json.dumps({"files": len(paths), "bytes": transferred, "native_exit": 0}, indent=2)
    )


if __name__ == "__main__":
    main()
