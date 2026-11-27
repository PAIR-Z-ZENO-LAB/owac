"""Verify and align each finalized bowl episode as its transfer finishes."""

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from owac.raw.teleop_mcap import prepare_group, write_inventory


def main(root):
    receipts = root / "receipts"
    files = defaultdict(dict)
    for line in (receipts / "recorder.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = Path(relative)
        files[Path(*path.parts[:2])][path] = digest
    groups = []
    output = root / "aligned"
    for episode, members in sorted(files.items()):
        receipt = receipts / (str(episode).replace("/", "-") + ".json")
        while not all((root / "raw" / path).is_file() for path in members):
            time.sleep(3)
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if (
                previous["files"] == list(members.values())
                and Path(previous["row"]["path"]).is_file()
            ):
                groups.append(previous["row"])
                continue
        for path, expected in members.items():
            with (root / "raw" / path).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Recorder digest differs for {path}")
        row = prepare_group(root / "raw" / episode, output, {"episode_005", "episode_010"})
        groups.append(row)
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(files=list(members.values()), row=row), indent=2))
        tmp.replace(receipt)
        print(episode, row["split"], row["num_frames"], flush=True)
    write_inventory(output, groups)
    for suite in sorted({g["suite"] for g in groups}):
        link = root.parent / "datasets" / "owac-dataset" / suite
        if not link.exists():
            link.symlink_to(root / "raw" / suite, target_is_directory=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
