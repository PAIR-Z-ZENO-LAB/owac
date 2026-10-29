"""Export actual RAW/DINO patch tokens and a shared teacher-fitted PCA display."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.training.raw_jepa_train import load_inventory, write_json
from owac.training.raw_semantic import build_semantic
from owac.training.raw_semantic_train import DinoTargets, PairedWindows


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    state = torch.load(args.encoder, map_location="cpu", weights_only=False, mmap=True)
    config, step = state["config"], state["step"]
    model = build_semantic(config).eval()
    model.load_state_dict(state["encoder"])
    del state
    model.to(args.device)
    teacher = DinoTargets(config["semantic"]["teacher"], args.device)
    sources, _ = load_inventory(config["data"]["inventory"])
    config["training"]["working_cache_sources"] = 2
    dataset = PairedWindows(sources, config)
    arrays, samples = {}, []
    lights = ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver")
    with torch.inference_mode():
        for i, light in enumerate(lights):
            source = sorted(
                [
                    s
                    for s in dataset.sources.values()
                    if s.split == "validation" and s.lighting == light
                ],
                key=lambda s: s.group_id,
            )[0]
            end = (source.num_frames - 1 + 5) // 2
            raw, rgb = dataset.paired([(source.group_id, end)], args.device, augment=False)
            arrays[f"{i}_raw_tokens"] = model(raw[:1])[0].float().cpu().numpy()
            full = teacher.model(
                pixel_values=(rgb[:1] - teacher.mean) / teacher.std
            ).last_hidden_state
            arrays[f"{i}_dino_full_tokens"] = full[0].float().cpu().numpy()
            arrays[f"{i}_dino_tokens"] = arrays[f"{i}_dino_full_tokens"][
                1 + teacher.model.config.num_register_tokens :
            ]
            arrays[f"{i}_rgb"] = rgb[0].permute(1, 2, 0).cpu().numpy()
            arrays[f"{i}_raw_preview"] = raw[0, -1].permute(1, 2, 0).cpu().numpy() ** (1 / 2.2)
            samples.append(dict(lighting=light, source=source.group_id, endpoint=end, camera=0))

    # One basis fitted only to the normal-light reference, shared across all displays.
    reference = F.normalize(torch.from_numpy(arrays["0_dino_tokens"]), dim=-1)
    center = reference.mean(0)
    _, singular, vh = torch.linalg.svd(reference - center, full_matrices=False)
    basis = vh[:3].T
    reference_scores = (reference - center) @ basis
    lower, upper = reference_scores.amin(0), reference_scores.amax(0)
    arrays.update(
        pca_center=center.numpy(),
        pca_basis=basis.numpy(),
        pca_lower=lower.numpy(),
        pca_upper=upper.numpy(),
    )
    stats = []
    for i, sample in enumerate(samples):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), dpi=150, layout="constrained")
        axes[0].imshow(np.clip(arrays[f"{i}_raw_preview"], 0, 1))
        axes[0].set_title("RAW preview (gamma only)")
        axes[1].imshow(arrays[f"{i}_rgb"])
        axes[1].set_title("Paired normal-light RGB")
        row = dict(sample)
        for ax, name in zip(axes[2:], ("raw", "dino"), strict=True):
            tokens = torch.from_numpy(arrays[f"{i}_{name}_tokens"])
            scores = (F.normalize(tokens, dim=-1) - center) @ basis
            colors = (scores - lower) / (upper - lower)
            ax.imshow(colors.clamp(0, 1).reshape(16, 16, 3).numpy(), interpolation="nearest")
            ax.set_title(f"{name.upper()} tokens: shared PCA")
            row[name] = dict(
                shape=list(tokens.shape),
                dtype=str(tokens.dtype),
                mean_token_norm=float(tokens.norm(dim=-1).mean()),
                token_xy_8_8_first12=tokens[8 * 16 + 8, :12].tolist(),
                display_clipped_components=float(((colors < 0) | (colors > 1)).float().mean()),
            )
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"J2S step {step} | {sample['lighting']} | 1024D -> 3D, no RGB reconstruction")
        fig.savefig(args.output / f"{i}_{sample['lighting']}_pca.png")
        plt.close(fig)
        stats.append(row)
    np.savez_compressed(args.output / "tokens.npz", **arrays)
    write_json(
        args.output / "tokens.json",
        dict(
            step=step,
            samples=stats,
            dino_full_shape=list(arrays["0_dino_full_tokens"].shape),
            register_tokens=teacher.model.config.num_register_tokens,
            pca_explained_variance=float(singular[:3].square().sum() / singular.square().sum()),
            protocol=(
                "FP32, fixed held sources/middle endpoint/camera0. Saved raw unnormalized tokens. "
                "Display L2-normalizes each patch; PCA fitted to normal-light DINO only, "
                "same center/basis/range for every panel, clipped to [0,1]. "
                "Colors are feature projections, not object labels or reconstructed RGB."
            ),
        ),
    )
    print(f"Saved step {step} tokens and five shared-PCA panels", flush=True)


if __name__ == "__main__":
    main()
