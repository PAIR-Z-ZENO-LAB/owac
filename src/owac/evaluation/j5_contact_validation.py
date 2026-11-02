"""Inspect held-out action error around gripper transitions in two affected tasks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from owac.evaluation.j5_regression_probe import compact_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_data import action_chunk
from owac.training.raw_ssl_data import FrameReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.root / "production.json").read_text())
    torch.set_num_threads(4)
    policies = [
        compact_policy(args.root / "normal/step-003000.pt", "cuda:1"),
        WebsocketClientPolicy(host="127.0.0.1", port=9025),
    ]
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
    rows = []
    for source in selected:
        with np.load(source.path) as z:
            actions = z["actions"]
            prompt = json.loads(str(z["metadata_json"].item()))["task_description"]
        transitions = np.flatnonzero(np.diff(actions[:, 6] > 0)) + 1
        contact = set(
            int(t + offset)
            for t in transitions
            for offset in range(-5, 6)
            if 0 <= t + offset < len(actions)
        )
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
            request = dict(
                raw=np.stack([r.frame(frame) for r in readers]), prompt=prompt, noise=noise
            )
            target = action_chunk(actions, frame, stats)[:, :7]
            valid = np.arange(10) + frame < len(actions)
            metrics = []
            for policy in policies:
                prediction = policy.infer(request)["normalized_actions"]
                metrics.append(
                    dict(
                        squared_error=float(((prediction - target)[valid] ** 2).sum()),
                        valid_steps=int(valid.sum()),
                        executed_squared_error=float(
                            ((prediction - target)[:5][valid[:5]] ** 2).sum()
                        ),
                        executed_steps=int(valid[:5].sum()),
                        gripper_errors=int(
                            ((prediction[:, 6] > 0) != (target[:, 6] > 0))[valid].sum()
                        ),
                    )
                )
            rows.append(
                dict(
                    source=source.group_id,
                    frame=frame,
                    anchor=frame in anchors,
                    contact=frame in contact,
                    metrics=metrics,
                )
            )
        for reader in readers:
            reader.close()
        print(source.group_id, len(rows), flush=True)
    (args.root / "regression-audit/contact-validation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
