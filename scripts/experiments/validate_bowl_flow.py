"""Compare bowl policies with free-running joints and an independent recorded camera clock."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory, read_npz_member
from owac.runtime.bowl_flow import STROKES, BowlFlow
from owac.runtime.bowl_postprocess import BowlPostprocessor, hand_center
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank


@torch.inference_mode()
def evaluate(parent, old, out):
    torch.set_num_threads(2)
    model, _ = load_export(parent / "motion-history/policy-best.pt")
    text = fixed_language(model)
    params = json.loads((out / "flow-policy.json").read_text())
    oldcfg = json.loads((old / "bowl-postprocess.json").read_text())
    track = json.loads((parent / "field-tracking.json").read_text())
    clips = [
        e for e in load_inventory(parent / "aligned/inventory.json")[0] if e.split == "validation"
    ]
    bank = RecoveryBank(clips, parent / "visual", model.cfg, track)
    bank.configure_output(model.cfg)
    bank.load_spatial(parent / "spatial")
    names = [n.replace("/", "-") for n in bank.names]
    obs = [dict(np.load(old / "features" / (n + ".npz"))) for n in names]
    u2 = [read_npz_member(ep.path, "u2_raw.npy") for ep in clips]
    events = [np.load(out / "keyframes" / (n + ".npz"))["index"][1:5] for n in names]
    geometr = params["geometry"]
    tau = np.array([x["tau_s"] for x in track["servo"]])
    delay = np.array([x["delay_s"] for x in track["servo"]])
    lag = model.cfg["model"]["state_history_lags"]
    field = list(
        map(json.loads, next((out / "field").glob("commands-*.jsonl")).read_text().splitlines())
    )
    bias = np.median(
        [np.array(r["feedback_rad_m"]) - r["latest_mux_rad_m"] for r in field[-120:]], axis=0
    )
    schedules = json.loads((parent / "execution-comparison.json").read_text())["gaps"]
    results = {}
    base_proc = BowlFlow.load(out / "flow-policy.json")
    for variant in ("previous", "flow"):
        results[variant] = []
        for schedule in schedules:
            if variant == "previous":
                proc = [BowlPostprocessor(oldcfg) for _ in names]
            else:
                proc = [BowlFlow.__new__(BowlFlow) for _ in names]
                for p in proc:
                    p.__dict__ = dict(base_proc.__dict__)
                    from collections import deque

                    p.samples = deque(maxlen=params["layout_frames"])
                    p.stroke_samples = deque(maxlen=params["stroke_window"])
                    from owac.runtime.bowl_postprocess import JointPlanSmoother

                    p.arm = JointPlanSmoother(params["joint_penalty"], params["joint_limits_rad"])
                    p.reset()
            q = np.stack([e["q"][0] for e in obs]).astype(float)
            pending = q.copy()
            previous = q.copy()
            traces = [q.copy()]
            log = [[] for _ in names]
            changes = [[] for _ in names]
            issued = [[] for _ in names]
            phases = [[] for _ in names]
            preview = []
            overhead = []
            selected_case = variant == "flow" and len(results[variant]) == 0
            got_open = np.zeros(len(q), dtype=bool)
            num = np.full(len(q), 0)
            step = chunk = 0
            max_s = 55
            done = np.zeros(len(q), bool)
            finish = [None] * len(q)
            while step < 30 * max_s and not done.all():
                now = step / 30
                capture = [max(0, np.searchsorted(e["t"], now, side="right") - 1) for e in obs]
                ix = bank.tensor(np.array(capture) + bank.starts).long()
                history = np.stack([traces[max(step - d, 0)] for d in lag], axis=1)
                plan = (
                    bank.predict(model, text, ix, bank.tensor(q), bank.tensor(history))
                    .cpu()
                    .numpy()[:, :4]
                )
                gap = schedule[chunk % len(schedule)]
                for tick in range(gap + 4):
                    row = tick - gap
                    prevq = q.copy()
                    for i, p in enumerate(proc):
                        if row < 0 or done[i]:
                            continue
                        before = p.phase
                        image_index = -1
                        if variant == "flow" and row == 0 and p.needs_overhead(q[i]):
                            grid = (
                                u2[i][capture[i]]
                                .reshape(120, 4, 160, 4)
                                .mean((1, 3), dtype=np.float32)
                            )
                            p.observe(grid, q[i])
                            if selected_case and i == 0:
                                image_index = len(overhead)
                                overhead.append(grid)
                        pending[i] = p.command(
                            plan[i], row, q[i], obs[i]["scene"][capture[i]].astype("f4"), step / 30
                        )
                        pending[i] = np.clip(pending[i], track["low"], track["high"])
                        issued[i].append(pending[i].copy())
                        phases[i].append(p.phase)
                        if selected_case and i == 0:
                            preview.append(
                                (
                                    image_index,
                                    row,
                                    step / 30,
                                    q[i].copy(),
                                    plan[i].copy(),
                                    obs[i]["scene"][capture[i]].astype("f4"),
                                    pending[i].copy(),
                                    p.phase,
                                )
                            )
                        if before != p.phase:
                            log[i].append(
                                dict(
                                    t=step / 30,
                                    phase=p.phase,
                                    jaw=float(pending[i, 7]),
                                    xyz=hand_center(q[i], geometr).tolist(),
                                    reference=getattr(p, "source", None),
                                    distance_m=getattr(p, "distance", None),
                                )
                            )
                        if variant == "flow" and p.phase == "finished":
                            done[i] = True
                            finish[i] = step / 30
                    dt = 1 / 30
                    a = np.exp(-np.minimum(dt, delay) / tau)
                    b = np.exp(-np.maximum(dt - delay, 0) / tau)
                    q = (q * a + (1 - a) * (previous + bias)) * b + (1 - b) * (pending + bias)
                    q[:, 7] = q[:, 7].clip(0, 0.1)
                    previous = pending.copy()
                    for i in range(len(q)):
                        split = oldcfg["width_split_m"]
                        oldw, neww = prevq[i, 7], q[i, 7]
                        isopen = neww > split
                        if oldw <= split and isopen:
                            got_open[i] = True
                        if (
                            got_open[i]
                            and (
                                (num[i] % 2 == 0 and oldw > split >= neww)
                                or (num[i] % 2 == 1 and oldw <= split < neww)
                            )
                            and num[i] < 4
                        ):
                            j = num[i]
                            xyz = hand_center(q[i], geometr)
                            expert = obs[i]["xyz"][events[i][j]]
                            changes[i].append(
                                dict(
                                    event=STROKES[j],
                                    t=(step + 1) / 30,
                                    xyz_m=xyz.tolist(),
                                    expert_m=expert.tolist(),
                                    xyz_error_mm=float(1000 * np.linalg.norm(xyz - expert)),
                                )
                            )
                            num[i] += 1
                    traces.append(q.copy())
                    step += 1
                chunk += 1
            if selected_case:
                cols = list(zip(*preview, strict=True))
                np.savez_compressed(
                    out / "host-parity-input.npz",
                    overhead=overhead,
                    **dict(
                        zip(
                            (
                                "image",
                                "row",
                                "time",
                                "feedback",
                                "plan",
                                "scene",
                                "target",
                                "phase",
                            ),
                            cols,
                            strict=True,
                        )
                    ),
                )
            rows = {}
            for i, n in enumerate(names):
                cmds = np.asarray(issued[i])
                phase_labels = np.asarray(phases[i])
                closing = {}
                for phase in STROKES:
                    a = cmds[phase_labels == phase, :7]
                    if len(a):
                        closing[phase] = float(np.rad2deg(np.ptp(a, axis=0)).max())
                v = np.rad2deg(np.max(abs(np.diff(cmds[:, :7], axis=0)), axis=-1))
                rows[n] = dict(
                    commanded_step_deg=np.quantile(v, [0.5, 0.95, 1]).tolist(),
                    joint_excursion_during_stroke_deg=closing,
                    width_events=changes[i],
                    phase_changes=log[i],
                    end=proc[i].phase,
                    finish_s=finish[i],
                    stack_verdict=getattr(proc[i], "stack_verdict", None),
                    flower_observed=getattr(proc[i], "flower_observed", None),
                    remaining_m=getattr(proc[i], "distance", None),
                    centers=getattr(proc[i], "centers", None).tolist()
                    if getattr(proc[i], "centers", None) is not None
                    else None,
                )
            results[variant].append(dict(gaps=schedule, episodes=rows))
            print(
                variant,
                schedule,
                [
                    (n, v["end"], len(v["width_events"]), v["finish_s"], v["remaining_m"])
                    for n, v in rows.items()
                ],
                flush=True,
            )
            (out / "feedback-flow.json").write_text(
                json.dumps(
                    dict(
                        method=(
                            "Independent recorded stereo clock, simulated state and history, "
                            "identified servo time constants and stationary bias. Recorded "
                            "objects do not respond to simulated actions."
                        ),
                        bias=bias.tolist(),
                        results=results,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("parent", "old", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    evaluate(a.parent, a.old, a.out)
