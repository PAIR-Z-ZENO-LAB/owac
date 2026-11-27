"""Keep the operator's banana-lift verdicts for one real-robot session."""

import json
import os
from collections import Counter
from datetime import datetime


class GraspTrials:
    def __init__(self, directory, preset="direct", speed=20):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = directory / f"{stamp}-{os.getpid()}.jsonl"
        self.path.touch(exist_ok=False)
        self.preset = preset
        self.success = "placed" if preset == "placement" else "lifted"
        self.speed = speed
        self.number = 0
        self.active = None
        self.counts = Counter()
        self.samples = []
        self.next_sample = 0

    def _record(self, event, policy, **fields):
        row = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "trial": self.number,
            "policy": policy,
            "execution": self.preset,
            "driver_speed_percent": self.speed,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def begin(self, policy, now, sent):
        if self.active is None:
            self.number += 1
            self._record("start", policy)
            self.active = (policy, now, sent)
            self.samples = []
            self.next_sample = now

    def observe(self, now, sent, feedback, model, issued, mux):
        """Keep a 5 Hz sample of the command path and the actual pose for this attempt."""
        self.next_sample = now + 0.2
        _, start, previous = self.active
        self.samples.append(
            {
                "t": round(now - start, 3),
                "commands": sent - previous,
                "feedback_stamp_ns": feedback.stamp if feedback is not None else None,
                "feedback_rad_m": feedback.values.tolist() if feedback is not None else None,
                "model_rad_m": model.tolist() if sent > previous and model is not None else None,
                "policy_rad_m": issued.tolist() if sent > previous and issued is not None else None,
                "mux_stamp_ns": mux[0] if mux else None,
                "mux_rad_m": mux[1].tolist() if mux else None,
            }
        )

    def finish(self, outcome, now, sent):
        if self.active is None:
            return
        policy, start, previous = self.active
        commands = sent - previous
        if not commands:
            outcome = "unscored"
        self._record(
            "end",
            policy,
            outcome=outcome,
            commands=commands,
            elapsed_s=round(now - start, 2),
            samples=self.samples,
        )
        self.counts[policy, outcome] += 1
        self.active = None

    def status(self, policy, now):
        successes = self.counts[policy, self.success]
        total = successes + self.counts[policy, "missed"]
        score = f"{successes}/{total}（{successes / total:.0%}）" if total else "0/0"
        if self.active:
            _, start, _ = self.active
            trial = f"第 {self.number} 次  {now - start:.0f} 秒  等待判定"
        else:
            trial = "按 R 开始下一次"
        task = "放入篮子" if self.preset == "placement" else "抓起"
        return f"当前权重{task} {score}    {trial}"
