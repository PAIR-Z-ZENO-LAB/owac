"""Materialize the published split and training recipes for a local data and asset root."""

import argparse
import json
from pathlib import Path

from owac.paper.checkpoint import TOKENIZER

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ("j3", "j4", "j5", "regression", "flow")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(
    dataset, work, recipes=ROOT / "configs/paper", split=ROOT / "data/manifests/paper-rawvla.jsonl"
):
    work = work.resolve()
    dataset = dataset.resolve()
    work.mkdir(parents=True, exist_ok=True)
    groups = [json.loads(line) for line in split.read_text().splitlines()]
    for group in groups:
        group["path"] = str(dataset / "data" / group["domain"] / "raw" / group["relpath"])
    inventory = work / "inventory.json"
    write(inventory, {"schema": "owac.paper-inventory.v1", "groups": groups})
    for name in RECIPES:
        cfg = json.loads((recipes / f"{name}.json").read_text())
        cfg["data"]["inventory"] = str(inventory)
        if name in ("flow", "regression"):
            cfg["assets"] = {
                "j5": str(work / "assets/j5/encoder.pt"),
                "language_encoder": str(work / "assets/t5"),
                "tokenizer": str(TOKENIZER),
            }
        if name in ("j4", "j5"):
            previous = "j3" if name == "j4" else "j4"
            cfg["training"]["initialize_from"] = str(work / previous / "latest.pt")
        if name == "j5":
            cfg["relations"]["anchor"] = str(work / "j3/encoder.pt")
            cfg["evaluation"]["baseline_tokens"] = str(work / "j4/visuals/030000/tokens.npz")
        write(work / f"{name}.json", cfg)
    return len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="RawVLA-Bench checkout containing data/"
    )
    parser.add_argument("--work", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--recipes", type=Path, default=ROOT / "configs/paper")
    parser.add_argument("--split", type=Path, default=ROOT / "data/manifests/paper-rawvla.jsonl")
    args = parser.parse_args()
    count = prepare(args.dataset, args.work, args.recipes, args.split)
    print(f"Wrote {count} trajectory paths and five configurations to {args.work.resolve()}")


if __name__ == "__main__":
    main()
