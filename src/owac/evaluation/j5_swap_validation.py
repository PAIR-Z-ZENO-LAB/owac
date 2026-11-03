"""Compare module swaps on the prior held-out gripper-transition diagnostic frames."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_data import action_chunk
from owac.training.raw_ssl_data import FrameReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9027)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads((args.root / "production.json").read_text())
    stats = json.loads(Path(cfg["assets"]["statistics"]).read_text())["norm_stats"]["actions"]
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [
        s
        for s in sources
        if s.domain == "libero" and s.split == "validation" and s.lighting == "Normal"
    ]
    selected = [
        s
        for s in sources
        if "between_the_plate_and_the_ramekin" in s.task_name
        or s.task_name == "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    ]
    observations = []
    for source in selected:
        with np.load(source.path) as z:
            actions = z["actions"]
            prompt = json.loads(str(z["metadata_json"].item()))["task_description"]
        transitions = np.flatnonzero(np.diff(actions[:, 6] > 0)) + 1
        contact = {
            int(t + d) for t in transitions for d in range(-5, 6) if 0 <= t + d < len(actions)
        }
        anchors = np.linspace(
            0, source.num_frames - 1, min(8, source.num_frames), dtype=int
        ).tolist()
        readers = [FrameReader(source.path, k) for k in ["agentview_raw_uint8", "wrist_raw_uint8"]]
        for frame in sorted(set(anchors) | contact):
            seed = (
                70000 + sources.index(source) * 8 + anchors.index(frame)
                if frame in anchors
                else 80000 + frame
            )
            torch.manual_seed(seed)
            noise = torch.randn((1, 10, 32), device="cuda:1").cpu().numpy()[0]
            observations.append(
                dict(
                    source=source.group_id,
                    frame=frame,
                    anchor=frame in anchors,
                    contact=frame in contact,
                    request=dict(
                        raw=np.stack([r.frame(frame) for r in readers]), prompt=prompt, noise=noise
                    ),
                    target=action_chunk(actions, frame, stats)[:, :7],
                    valid=np.arange(10) + frame < len(actions),
                )
            )
        for r in readers:
            r.close()
    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    rows = []
    for mode in ["before", "after", "vision", "without_bridge"]:
        for o in observations:
            pred = client.infer(dict(o["request"], variant=mode))["normalized_actions"]
            v = o["valid"]
            target = o["target"]
            rows.append(
                dict(
                    variant=mode,
                    **{k: o[k] for k in ["source", "frame", "anchor", "contact"]},
                    squared_error=float(((pred - target)[v] ** 2).sum()),
                    valid_steps=int(v.sum()),
                    gripper_errors=int(((pred[:, 6] > 0) != (target[:, 6] > 0))[v].sum()),
                )
            )
        print(mode, len(observations), flush=True)
    (args.root / "weight-audit/contact-validation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
