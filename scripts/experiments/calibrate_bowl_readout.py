"""Fit the relative weight of the two learned OWAC estimates on complete training recordings."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank, score_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("experiment", "parent", "urdf", "run"):
        p.add_argument(arg, type=Path)
    p.add_argument("--jaw-response", action="store_true")
    args = p.parse_args()
    args.run.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    model, step = load_export(args.parent)
    cfg = model.cfg
    text = fixed_language(model)
    src, _ = load_inventory(args.experiment / "aligned/inventory.json")
    track = json.loads((args.experiment / "field-tracking.json").read_text())
    hand = NeroHand(args.urdf, "cpu")

    def bank_for(split):
        bank = RecoveryBank(
            [s for s in src if s.split == split], args.experiment / "visual", cfg, track
        )
        bank.configure_output(cfg)
        bank.load_spatial(args.experiment / "spatial")
        return bank

    bank = bank_for("train")
    rows = []
    weights = model.base_logit.detach().sigmoid().cpu().numpy()
    # Hand and finger outputs have different error costs. Fit the two shares separately.
    # Keep both camera paths in the arm estimate; include each endpoint for the single grip axis.
    arms = [0.25, float(weights[:7].mean()), 0.75]
    grip = [0.0, 0.25, float(weights[7]), 0.75, 1.0]

    def set_shares(arm, jaw):
        share = model.base_logit.detach().new_tensor([arm] * 7 + [jaw])
        with torch.no_grad():
            model.base_logit.copy_(torch.logit(share, eps=torch.finfo(share.dtype).eps))

    set_params = model.set_jaw_response if args.jaw_response else set_shares
    if args.jaw_response:
        # Time constants expressed as fractions/multiples of the fitted physical gripper lag.
        # Include the identity and align nonzero lead candidates to predicted 30 Hz targets.
        tau = track["servo"][7]["tau_s"]
        gains = [1.0, *[float(1 - np.exp(-1 / (30 * tau * ratio))) for ratio in (0.5, 1, 2)]]
        grid = itertools.product(gains, [0, 1, 2, 3])
        names = ("gain", "lead")
    else:
        grid = itertools.product(arms, grip)
        names = ("arm", "gripper")
    for a, b in grid:
        set_params(a, b)
        summary, _ = score_replay(bank, bank.rollout(model, text, 5), hand)
        rows.append({names[0]: a, names[1]: b, "summary": summary})
        print(a, b, summary, flush=True)
    selected = min(
        rows, key=lambda r: (-r["summary"]["events"], r["summary"]["four_events_xyz_mm"])
    )
    set_params(selected[names[0]], selected[names[1]])
    del bank
    bank = bank_for("validation")
    error = bank.rollout(model, text, 5)
    summary, episodes = score_replay(bank, error, hand)
    setting = {k: selected[k] for k in names}
    if args.jaw_response:
        cfg["model"]["aperture_response"] = setting
    cfg["jaw_calibration" if args.jaw_response else "readout_calibration"] = dict(
        **setting,
        split="train",
        objective="event count then mean 3D distance at four expert grip transitions",
    )
    atomic_save(dict(config=cfg, step=step, model=model.state_dict()), args.run / "policy.pt")
    (args.run / "selection.json").write_text(
        json.dumps(
            dict(candidates=rows, selected=selected, summary=summary, episodes=episodes), indent=2
        )
    )
    for name, a, n in zip(bank.names, bank.starts, bank.sizes, strict=True):
        path = args.run / (name + ".npz")
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            path,
            observed=bank.q[a : a + n].cpu().numpy(),
            simulated=(bank.q + error)[a : a + n].cpu().numpy(),
        )
    print("validation", summary, flush=True)


if __name__ == "__main__":
    main()
