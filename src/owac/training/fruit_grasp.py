"""NERO hand pose and the approach to the first banana grasp in a fruit recording."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch


def closing_stroke(width, episode):
    half = np.quantile(width, [0.05, 0.95]).mean()
    falls = np.flatnonzero((width[:-1] > half) & (width[1:] <= half)) + 1
    if not len(falls):
        raise ValueError(f"No closing stroke found in {episode}")
    return falls[0], half


def grasp_phase(src, seconds):
    with np.load(src.path) as f:
        observed = f["observation_ns"]
        command_time = f["command_ns"]
        width = f["commands"][:, 7]
        closing, _ = closing_stroke(width, src.group_id)
        event = command_time[closing]
        remaining = (event - observed) / 1e9
        return remaining, (remaining >= 0) & (remaining <= seconds)


def urdf_transform(joint):
    origin = joint.find("origin")
    xyz = [float(x) for x in origin.get("xyz").split()]
    r, p, y = [float(x) for x in origin.get("rpy").split()]
    a, b, c = math.cos(r), math.cos(p), math.cos(y)
    u, v, w = math.sin(r), math.sin(p), math.sin(y)
    return xyz, [
        [c * b, c * v * u - w * a, c * v * a + w * u],
        [w * b, w * v * u + c * a, w * v * a - c * u],
        [-v, b * u, b * a],
    ]


def nero_pose(joints, xyz, rot, tip, basis):
    q = joints[..., :7].float()
    c, s = q.cos(), q.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    rz = torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*q.shape, 3, 3)
    R = rot[0] @ rz[..., 0, :, :]
    p = xyz[0]
    for i in range(1, 7):
        p = p + (R @ xyz[i])
        R = (R @ rot[i]) @ rz[..., i, :, :]
    return p + R @ tip, R @ basis


class NeroHand:
    """Rigid transform at the midpoint between the two URDF finger joint origins."""

    def __init__(self, folder, device):
        folder = Path(folder)
        links = ET.parse(folder / "nero_description.urdf").getroot()
        flange = ET.parse(folder / "nero_with_gripper_flange_description.xacro").getroot()
        hand = ET.parse(folder / "nero_with_gripper_description.xacro").getroot()
        joints = [links.find(f"joint[@name='joint{i}']") for i in range(1, 8)]
        if any([float(x) for x in j.find("axis").get("xyz").split()] != [0, 0, 1] for j in joints):
            raise ValueError("NERO chain expects local Z revolute axes")
        xyz, rpy = zip(*(urdf_transform(j) for j in joints), strict=True)
        self.xyz = torch.tensor(xyz, device=device, dtype=torch.float32)
        self.rot = torch.tensor(rpy, device=device, dtype=torch.float32)
        offset, frame = urdf_transform(flange.find("joint[@name='gripper_flange_joint']"))
        base, align = urdf_transform(hand.find("joint[@name='gripper_base_joint']"))
        origins = [urdf_transform(hand.find(f"joint[@name='gripper_joint{i}']"))[0] for i in (1, 2)]
        frame, align = np.asarray(frame), np.asarray(align)
        offset = offset + frame @ (base + align @ np.mean(origins, axis=0))
        self.tip = torch.tensor(offset, device=device, dtype=torch.float32)
        self.basis = torch.tensor(frame @ align, device=device, dtype=torch.float32)

    def pose(self, joints):
        return nero_pose(joints, self.xyz, self.rot, self.tip, self.basis)

    def squared_errors(self, predicted, target):
        p, r = self.pose(predicted)
        t, s = self.pose(target)
        return (p - t).square().sum(-1), (r - s).square().sum((-2, -1)) / 2

    def reference_scale(self, bank):
        state = bank.state[:, None, :].expand_as(bank.absolute)
        p, r = self.squared_errors(state, bank.absolute)
        relative = state - state * bank.relative_mask
        hold = (relative - bank.low) * 2 / bank.scale - 1
        mse = (hold - bank.normalized).square().mean(-1)
        return torch.stack([value[bank.valid].mean() for value in (p, r, mse)])

    def loss(self, predicted, target, valid, scales):
        pos, rot = self.squared_errors(predicted, target)
        return ((pos / scales[0] + rot / scales[1]) * 0.5)[valid].mean() * scales[2]


def pose_metrics(hand, predicted, target, mask):
    p, r = hand.pose(predicted[mask])
    t, s = hand.pose(target[mask])
    distance = (p - t).norm(dim=-1) * 1000
    relative = r.transpose(-1, -2) @ s
    # atan2 is accurate near zero, and also handles rotations close to pi.
    skew = torch.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        -1,
    )
    angle = torch.atan2(skew.norm(dim=-1), relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
    return dict(
        commands=len(distance),
        position_mm=distance.mean().item(),
        orientation_deg=angle.rad2deg().mean().item(),
    )


def approach_masks(remaining, valid, frequency, seconds):
    time = remaining[:, None] - np.arange(valid.shape[1])[None, :] / frequency
    near = (remaining[:, None] <= seconds) & (remaining[:, None] >= 0) & valid
    return near & (time >= 0), near & (np.abs(time) <= 0.5 / frequency)


@torch.inference_mode()
def evaluate_approach(model, bank, hand, scales, remaining, seconds):
    approach, contact = approach_masks(
        remaining, bank.valid.cpu().numpy(), bank.cfg["observation"]["frequency_hz"], seconds
    )
    indices = np.flatnonzero(approach.any(1) | contact.any(1))
    ix = bank.tensor(indices)
    target = bank.absolute[ix]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        parts = [bank.physical(model.sample(bank.batch(i)), i) for i in ix.split(16)]
    pred = torch.cat(parts)
    mask = bank.tensor(approach[indices])
    p, r = hand.squared_errors(pred, target)
    scores = pose_metrics(hand, pred, target, mask)
    scores["normalized_error"] = ((p / scales[0] + r / scales[1])[mask] * 0.5).mean().item()
    close_mask = bank.tensor(contact[indices])
    close = pose_metrics(hand, pred, target, close_mask)
    scores.update({"closing_" + k: v for k, v in close.items()})
    if bank.cfg["evaluation"].get("grasp_by_subset"):
        episode_ids = bank.episode[indices]
        distances = {}
        for episode, src in enumerate(bank.sources):
            rows = bank.tensor(episode_ids == episode)
            value = pose_metrics(hand, pred[rows], target[rows], close_mask[rows])["position_mm"]
            distances.setdefault(src.suite, []).append(value)
        scores.update({k + "_closing_position_mm": float(np.mean(v)) for k, v in distances.items()})
    return scores


def banana_image_errors(prob, truth, grid, patch):
    height, width = grid
    y, x = torch.meshgrid(
        (torch.arange(height, device=prob.device) + 0.5) * patch,
        (torch.arange(width, device=prob.device) + 0.5) * patch,
        indexing="ij",
    )
    xy = torch.stack((x, y), -1).reshape(-1, 2)

    def readout(mass):
        mass = mass.float() / mass.float().sum(-1, keepdim=True)
        centre = mass @ xy
        offset = xy[None] - centre[:, None]
        cov = offset.transpose(1, 2) @ (mass[:, :, None] * offset)
        angle = 0.5 * torch.atan2(2 * cov[:, 0, 1], cov[:, 0, 0] - cov[:, 1, 1])
        return centre, angle

    a, angle = readout(prob)
    b, ref = readout(truth)
    axial = ((angle - ref + torch.pi / 2) % torch.pi - torch.pi / 2).abs()
    return dict(
        centre_px=(a - b).norm(dim=-1).mean().item(), axis_deg=axial.rad2deg().mean().item()
    )


@torch.inference_mode()
def banana_validation(model, train, val):
    ix = val.tensor(val.starts)
    reference = train.banana_mass[train.tensor(train.starts)].float().mean(0, keepdim=True)
    patch = val.cfg["model"]["raw_encoder"]["patch_size"]
    grid = tuple(size // patch for size in val.cfg["observation"]["raw_shape"])
    target = val.banana_mass[ix]
    results = {}
    for key, exp in (("native", 0), ("minus2ev", 2)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.predict(val.batch(ix, exposure=exp), return_goal=True)[2]
        results.update(
            {
                f"banana/{key}_{k}": v
                for k, v in banana_image_errors(
                    logits.float().softmax(-1), target, grid, patch
                ).items()
            }
        )
    results.update(
        {
            f"banana/train_mean_{k}": v
            for k, v in banana_image_errors(
                reference.expand_as(target), target, grid, patch
            ).items()
        }
    )
    return results
