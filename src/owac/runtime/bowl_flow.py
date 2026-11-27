"""Demonstration-conditioned OWAC arm motion in measured bowl frames, with gripper feedback."""

import json
from collections import deque
from pathlib import Path

import numpy as np

from owac.runtime.bowl_localization import BowlLocator, StackAppearance, raw_grid
from owac.runtime.bowl_postprocess import JointPlanSmoother, hand_center
from owac.runtime.fruit_live import FEEDBACK_SECONDS, HZ

STAGES = ("pick_flower", "place_flower", "pick_black", "place_black", "retreat")
STROKES = ("close_flower", "open_flower", "close_black", "open_black")
TARGETS = (0, 2, 1, 3)


def pose_jacobian(q, geometry):
    """Hand center, rotation, and 6-by-7 geometric Jacobian in the robot base."""
    q = np.atleast_2d(q)
    n = len(q)
    offsets, rotations, tip = (np.asarray(geometry[k]) for k in ("xyz", "rot", "tip"))
    eye = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    pos = np.zeros((n, 3))
    axes, origins = [], []
    for j in range(7):
        pos = pos + eye @ offsets[j]
        eye = eye @ rotations[j]
        origins.append(pos)
        axes.append(eye[..., 2])
        co, si = np.cos(q[:, j]), np.sin(q[:, j])
        rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        rot[:, 0, 0] = rot[:, 1, 1] = co
        rot[:, 1, 0], rot[:, 0, 1] = si, -si
        eye = eye @ rot
    center = pos + eye @ tip
    axes = np.stack(axes, axis=1)
    linear = np.cross(axes, center[:, None] - np.stack(origins, axis=1))
    return center, eye, np.concatenate((linear, axes), axis=2).transpose(0, 2, 1)


def bounded_pose_step(jac, position, angle, q, limits):
    # Translation has first priority. Within its null space retain the demonstrated
    # wrist orientation. Re-solve with a saturated axis fixed at its physical limit.
    count = len(q)
    free = np.ones((count, 7), dtype=bool)
    residual = np.column_stack((position, angle))
    fixed = np.zeros_like(q)
    row = np.arange(count)
    for _ in range(8):
        active_jac = jac * free[:, None, :]
        jp, jr = active_jac[:, :3], active_jac[:, 3:]
        inverse = np.linalg.pinv(jp, rcond=1.0e-4)
        error = residual - (jac @ fixed[..., None])[..., 0]
        translation = (inverse @ error[:, :3, None])[..., 0]
        null = np.eye(7)[None] - inverse @ jp
        projected = jr @ null
        orientation = (
            np.linalg.pinv(projected, rcond=0.005)
            @ (error[:, 3:, None] - jr @ translation[..., None])
        )[..., 0]
        update = fixed + (translation + (null @ orientation[..., None])[..., 0]) * free
        trial = q + update
        feasible = np.clip(trial, *limits)
        violation = abs(trial - feasible) * free
        worst = violation.argmax(1)
        invalid = violation[row, worst] > 1.0e-10
        if not invalid.any():
            return update
        rr = row[invalid]
        jj = worst[invalid]
        fixed[rr, jj] = feasible[rr, jj] - q[rr, jj]
        free[rr, jj] = False
    return np.clip(q + update, *limits) - q


def translate_path(q, offsets, geometry, limits):
    """Apply a base-frame translation subject to joint limits and preserve wrist orientation."""
    pos, orientation, _ = pose_jacobian(q, geometry)
    goal = pos + offsets
    out = np.clip(q.astype(float), *limits)
    for _ in range(12):
        pos, rot, jac = pose_jacobian(out, geometry)
        rot_error = 0.5 * np.cross(rot.transpose(0, 2, 1), orientation.transpose(0, 2, 1)).sum(1)
        shift = goal - pos
        if np.max(abs(shift)) < 1.0e-7 and np.max(abs(rot_error)) < 1.0e-6:
            break
        step = bounded_pose_step(jac, shift, rot_error, out, limits)
        # Trust-region step for numerical convergence only. Commands use the separate plan smoother.
        fraction = np.minimum(1.0, 0.15 / np.maximum(abs(step).max(1), 1.0e-12))
        out = np.clip(out + step * fraction[:, None], *limits)
        if np.max(abs(step)) < 1.0e-7:
            break
    return out


def adapt_contact(
    ref, weights, stage, bank, example, scene_centers, locator, planes, geometry, limits
):
    slot = TARGETS[stage]
    level = planes[stage]
    support_level = level
    source = bank["centers"][example, slot]
    if stage in (1, 3):
        support_level = planes[0 if stage == 1 else 1]
        source = bank["landed"][example, stage // 2]
    center = np.r_[locator.table_xy(source, level), level]
    destination = np.r_[locator.table_xy(scene_centers[slot], support_level), level]
    # Use the demonstrated center of the grasped bowl as a point rigidly attached
    # to the hand. A joint limit can require a small change in wrist orientation.
    # Rotation about this center keeps the bowl aligned with the measured rim.
    palm, rotation, _ = pose_jacobian(ref[-1], geometry)
    offset = rotation[0].T @ (center - palm[0])
    contact_geometry = dict(geometry, tip=np.asarray(geometry["tip"]) + offset)
    path = translate_path(ref, weights * (destination - center), contact_geometry, limits)
    return path, contact_geometry, destination


def select_costs(bank, stage, measured, scene, centers):
    poses = bank["paths"][:, stage, 0]
    obs = bank["scenes"][:, stage]
    body = np.sqrt(np.mean((poses - measured[:7]) ** 2, axis=-1))
    visual = np.sqrt(np.mean((obs - scene) ** 2, axis=-1))
    if stage < 4:
        p = TARGETS[stage]
        layout = np.linalg.norm(bank["centers"][:, p] - centers[p], axis=-1)
    else:
        layout = np.zeros_like(body)
    return (
        np.stack((body, visual, layout, bank["central"][:, stage]), axis=1)
        / bank["cost_scale"][stage]
    )


def curve(phase):
    return phase * phase * (3 - 2 * phase)


def sample_path(path, fractions):
    p = np.clip(fractions, 0, 1) * (len(path) - 1)
    lo = np.minimum(p.astype(int), len(path) - 2)
    alpha = (p - lo)[..., None]
    return path[lo] * (1 - alpha) + path[lo + 1] * alpha


class BowlFlow:
    def __init__(self, config, bank):
        self.profile = config
        self.bank = bank
        self.locator = BowlLocator(config["locator"])
        self.stack = StackAppearance(
            bank["stack_templates"], bank["stack_exposures"], bank["stack_cutoffs"], self.locator.r
        )
        self.geo = config["geometry"]
        self.limits = np.asarray(config["joint_limits_rad"])
        self.arm = JointPlanSmoother(config["joint_penalty"], self.limits)
        self.samples = deque(maxlen=config["layout_frames"])
        self.reset()

    @classmethod
    def load(cls, path):
        path = Path(path)
        cfg = json.loads(path.read_text())
        with np.load(path.parent / cfg["trajectory_weights"], allow_pickle=False) as f:
            return cls(cfg, dict(f))

    def reset(self):
        self.arm.reset()
        self.samples.clear()
        self.stack_frames = deque(maxlen=self.profile["layout_frames"])
        self.stage = 0
        self.phase = "locate_bowls"
        self.centers = None
        self.last_capture = None
        self.recheck = None
        self.path = None
        self.score = None
        self.distance = None
        self.source = None
        self.grasp_ref = None
        self.stack_verdict = None
        self.stack_scores = None
        self.flower_observed = None
        self.retired_at = None
        self.stroke = False
        self.elapsed = 0.0
        self.arm_plan = None
        self.hold = None
        self.stroke_samples = deque(maxlen=self.profile["stroke_window"])

    def stack_visible(self, measured):
        if self.stage != 2 or self.centers is None or measured is None:
            return False
        p = self.locator.camera @ np.r_[hand_center(measured, self.geo), 1.0]
        return np.linalg.norm(p[:2] / p[2] - self.centers[2]) > 2 * self.locator.r

    def needs_overhead(self, measured):
        return (
            self.centers is None
            or self.recheck is not None
            or self.phase == "inspect_stack"
            or self.stack_visible(measured)
        )

    def observe(self, raw, measured=None):
        if self.stack_visible(measured):
            self.stack_frames.append(raw_grid(raw))
        if self.phase == "inspect_stack":
            self.samples.append(raw_grid(raw))
            if len(self.samples) == self.samples.maxlen:
                label, _, self.stack_scores = self.stack.read(
                    np.mean(self.samples, axis=0),
                    self.centers[2],
                    self.profile["stack_displacement_px"][1],
                )
                self.stack_verdict = (
                    "unconfirmed" if label < 0 else ("white", "flower", "black")[label]
                )
                self.phase = "finished"
            return
        if self.centers is None:
            self.samples.append(raw_grid(raw))
            if len(self.samples) < self.samples.maxlen:
                return
            pts, scores, thresholds = self.locator.locate(np.mean(self.samples, axis=0))
            self.score = float(np.min(scores - thresholds))
            distances = np.linalg.norm(
                pts[np.triu_indices(3, 1)[0]] - pts[np.triu_indices(3, 1)[1]], axis=1
            )
            if self.score >= 0 and np.min(distances) > 2 * (self.locator.r - 2):
                self.centers = np.vstack([pts, pts[2]])
                self.phase = STAGES[0]
            return
        if self.recheck is None:
            return
        # Reobserve the unobstructed black source at the first placement, and the
        # actual patterned rim on the stack when the arm reaches the black source.
        slot, obj = self.recheck
        if slot == 3:
            frame = np.mean(self.stack_frames, axis=0) if self.stack_frames else raw
            label, pts, self.stack_scores = self.stack.read(
                frame, self.centers[2], self.profile["stack_displacement_px"][0]
            )
            self.flower_observed = label == 1
            if self.flower_observed:
                self.centers[3] = pts[1]
            self.recheck = None
            return
        prior = self.centers[slot]
        r = self.locator.r
        old = self.locator.bounds
        x, y = np.rint(prior).astype(int)
        self.locator.bounds = [
            max(old[0], x - r),
            max(old[1], y - r),
            min(old[2], x + r + 1),
            min(old[3], y + r + 1),
        ]
        try:
            pts, scores, cut = self.locator.locate(raw, (obj,))
        finally:
            self.locator.bounds = old
        if scores[0] >= cut[0] and np.linalg.norm(pts[0] - prior) <= r:
            self.centers[slot] = pts[0]
        self.recheck = None

    def suspend(self):
        self.arm.reset()
        self.samples.clear()
        self.stack_frames.clear()
        self.stroke_samples.clear()
        self.arm_plan = None

    def observe_frame(self, observation):
        if observation.capture == self.last_capture or not self.needs_overhead(
            observation.feedback.values
        ):
            return
        from owac.raw.teleop_mcap import unpack_bg10

        frame = observation.frames[0]
        self.observe(
            unpack_bg10(frame.data, frame.width, frame.height, frame.stride),
            observation.feedback.values,
        )
        self.last_capture = observation.capture

    def begin(self, measured, scene):
        s = self.stage
        costs = select_costs(self.bank, s, measured, scene, self.centers)
        k = int(np.argmin(costs @ self.bank["readout"][s]))
        if s in (0, 2):
            self.grasp_ref = k
        elif s in (1, 3):
            # Preserve the rigid rim-to-hand transform of the actual pickup across its transfer.
            k = self.grasp_ref
        self.source = str(self.bank["names"][k])
        ref = self.bank["paths"][k, s].astype(float)
        weights = curve(np.linspace(0, 1, len(ref)))[:, None]
        ref = ref + (1 - weights) * (measured[:7] - ref[0])
        self.contact_geometry = self.geo
        if s < 4:
            ref, self.contact_geometry, self.goal = adapt_contact(
                ref,
                weights,
                s,
                self.bank,
                k,
                self.centers,
                self.locator,
                self.profile["plane_m"],
                self.geo,
                self.limits,
            )
        self.path = np.clip(ref, *self.limits)
        center, rotation, _ = pose_jacobian(self.path[-1], self.contact_geometry)
        if s == 4:
            self.goal = center[0]
        self.orientation = rotation[0]
        self.contact_hold = None
        self.duration = float(self.bank["duration"][s])
        self.elapsed = 0.0
        self.stroke = False
        self.phase = STAGES[s]
        self.arm_plan = None

    def command(self, plan, row, measured, scene, now):
        measured = np.asarray(measured, dtype=float)
        if self.centers is None or scene is None:
            return measured.copy()
        if self.phase in ("finished", "inspect_stack"):
            return np.r_[self.hold, self.profile["open_width_m"]]
        if self.path is None and self.recheck is not None:
            width = (
                self.profile["open_width_m"] if self.stage == 2 else self.profile["closed_width_m"]
            )
            return np.r_[self.arm.last, width]
        if self.path is None:
            self.begin(measured, scene)
        dt = 0 if self.arm.last_time is None else now - self.arm.last_time
        if 0 < dt <= FEEDBACK_SECONDS:
            self.elapsed = min(self.duration, self.elapsed + dt)
        s = self.stage
        center, rotation, _ = pose_jacobian(measured[:7], self.contact_geometry)
        self.distance = float(np.linalg.norm(center[0] - self.goal))
        reached = (
            self.elapsed >= self.duration and self.distance <= self.profile["contact_radius_m"][s]
        )
        if reached and s < 4:
            angle = float(
                np.arccos(np.clip(((rotation[0] * self.orientation).sum() - 1) / 2, -1, 1))
            )
            reached = angle <= self.profile["contact_angle_rad"][s]
        if reached and s < 4 and not self.stroke:
            self.stroke = True
            self.stroke_samples.clear()
            self.contact_hold = (
                self.arm.last if self.arm.last is not None else measured[:7]
            ).copy()
            self.phase = STROKES[s]
            if s in (1, 2):
                self.recheck = (1, 1) if s == 1 else (3, 0)
        closing = s in (0, 2)
        open_now = (closing and not self.stroke) or (not closing and self.stroke) or s == 4
        width = self.profile["open_width_m"] if open_now else self.profile["closed_width_m"]
        # Fit an arm command sequence on a changing OWAC plan, then keep the final
        # contact pose fixed throughout the measured opening or closing stroke.
        if self.stroke:
            target = self.contact_hold
            self.arm.last, self.arm.last_time = target.copy(), now
            self.arm.plan = None
            self.arm.velocity.fill(0)
        else:
            previous = (
                self.arm.last.copy()
                if self.arm.last is not None
                else np.clip(measured[:7], *self.limits)
            )
            replan = row == 0 or self.arm_plan is None or self.arm.plan is None
            if replan:
                frac = np.minimum(
                    1, (self.elapsed + (np.arange(len(plan)) - row) / HZ) / self.duration
                ).clip(0, 1)
                guide = sample_path(self.path, curve(frac))
                delta = np.clip(
                    plan[:, :7] - measured[:7],
                    -self.bank["native_span"][s],
                    self.bank["native_span"][s],
                )
                guide += self.bank["native_gain"][s] * np.sin(np.pi * frac[:, None]) ** 2 * delta
                # Correct the observed following error as the final waypoint comes to rest.
                terminal = frac >= 1
                guide[terminal] += np.clip(
                    self.path[-1] - measured[:7],
                    -self.bank["track_span"][s],
                    self.bank["track_span"][s],
                )
                self.arm_plan = guide
            # A phase change can happen in the middle of a four-action native plan.
            if replan and row:
                target = self.arm.step(self.arm_plan[row:], measured, now, 0)
                self.arm.plan = np.vstack([np.tile(measured[:7], (row, 1)), self.arm.plan])
            else:
                target = self.arm.step(self.arm_plan, measured, now, row)
            # During a camera gap the driver holds its previous command. Apply one
            # demonstrated 30 Hz increment per emitted target, including the first
            # target after a gap. Scale all seven joints along the same segment.
            step = target - previous
            ratio = np.max(abs(step) / self.profile["command_step_rad"])
            if ratio > 1:
                target = previous + step / ratio
                self.arm.last = target.copy()
                self.arm.velocity = (target - previous) / (dt * HZ if dt > 0 else 1.0)
                self.arm.plan = None
        acknowledged = (
            (
                measured[7] <= self.profile["ack_width_m"][s]
                if closing
                else measured[7] >= self.profile["ack_width_m"][s]
            )
            if s < 4
            else False
        )
        if self.stroke and closing:
            self.stroke_samples.append(float(measured[7]))
            if len(self.stroke_samples) == self.stroke_samples.maxlen:
                # A rim can stop the jaws before reaching the unloaded closed width.
                contact_settled = (
                    max(self.stroke_samples) < self.profile["width_split_m"]
                    and np.ptp(self.stroke_samples) <= self.profile["held_width_variation_m"]
                )
                acknowledged = acknowledged or contact_settled
        if self.stroke and acknowledged:
            self.stage += 1
            self.path = None
            self.phase = STAGES[self.stage]
            self.arm_plan = None
        elif s == 4 and reached:
            self.phase = "inspect_stack"
            self.retired_at = now
            self.samples.clear()
            self.hold = target.copy()
        return np.r_[target, width]

    def diagnostics(self):
        return dict(
            flow_stage=self.phase,
            flow_reference=self.source,
            flow_distance_m=self.distance,
            flower_on_white_observed=self.flower_observed,
            stack_appearance=self.stack_verdict,
            stack_scores=self.stack_scores,
            bowl_centers_u2_quarter=None if self.centers is None else self.centers.tolist(),
        )
