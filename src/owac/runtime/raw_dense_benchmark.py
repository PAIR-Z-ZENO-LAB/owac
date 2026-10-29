"""Held-source dense geometry and task-retrieval proxies against DINOv3 RGB."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.raw.all_bench_distillation import load_inventory, read_npz_member
from owac.training.raw_dense_refinement import crop_correspondence
from owac.training.raw_jepa import build_jepa
from owac.training.raw_jepa_train import RawWindows, write_json


def match_metrics(a, b, ia, ib):
    a, b = F.normalize(a[:, ia].float(), dim=-1), F.normalize(b[:, ib].float(), dim=-1)
    predicted = (a @ b.transpose(-1, -2)).argmax(-1)
    indices = ib[predicted]
    error = (
        (indices % 16 - ib % 16).float().square() + (indices // 16 - ib // 16).float().square()
    ).sqrt() * 14
    return error


def main():
    from transformers import AutoModel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dino", default="/root/autodl-pub/checkpoints/perception/dinov3-vitl16")
    p.add_argument("--dino-size", type=int, choices=(224, 256), default=224)
    p.add_argument(
        "--per-light", type=int, default=0, help="0: every held source; >0: diagnostic subset"
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = "cuda"
    state = torch.load(args.encoder, map_location="cpu", weights_only=False)
    cfg = dict(state["config"])
    cfg.pop("initial_encoder", None)
    semantic = state.get("schema") == "owac.raw-semantic.encoder.v1"
    if semantic:
        from owac.training.raw_semantic import build_semantic

        encoder = build_semantic(cfg)
    else:
        model, target, projector = build_jepa(cfg, "cpu")
        encoder = model.encoder
        del target, projector
    encoder.load_state_dict(state["encoder"])
    step = state["step"]
    del state
    encoder.eval().to(device)
    dino = AutoModel.from_pretrained(args.dino, local_files_only=True).eval().to(device)
    sources, _ = load_inventory(cfg["data"]["inventory"])
    cfg["training"]["working_cache_sources"] = 2
    ds = RawWindows(sources, cfg)
    selected = sorted(
        [s for s in ds.sources.values() if s.split == "validation"], key=lambda s: s.group_id
    )
    if args.per_light:
        selected = [
            s
            for light in sorted({s.lighting for s in selected})
            for s in [x for x in selected if x.lighting == light][: args.per_light]
        ]
    selected_ids = {s.group_id for s in selected}
    keys = [k for k in ds.validation_keys() if k[0] in selected_ids]
    rows, pooled, identities = [], defaultdict(list), []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    origins = [(0, 0), (0, 28), (28, 0)]
    with torch.inference_mode():
        for index, (sid, end) in enumerate(keys):
            source = ds.sources[sid]
            raw = ds.native_batch([(sid, end)], device)
            rgbpath = Path(str(source.path).replace("/libero/raw/", "/libero/rgb/"))
            rgb = (
                torch.stack(
                    [
                        torch.from_numpy(
                            np.array(read_npz_member(rgbpath, key + ".npy")[end], copy=True)
                        ).permute(2, 0, 1)
                        for key in ["agentview_rgb_uint8", "wrist_rgb_uint8"]
                    ]
                )
                .to(device)
                .float()
                / 255
            )
            features = defaultdict(list)
            for y, x in origins:
                rawcrop = raw[..., y : y + 224, x : x + 224]
                rgbcrop = rgb[..., y : y + 224, x : x + 224]
                # FP32 inference, same 224-pixel crop. DINO native14x14 -> common16x16 centers.
                features["raw_final"].append(encoder(rawcrop))
                if not semantic:
                    features["raw_block9"].append(encoder(rawcrop, stop_at=9))
                if args.dino_size != 224:
                    rgbcrop = F.interpolate(
                        rgbcrop,
                        size=(args.dino_size, args.dino_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                tokens = dino(pixel_values=(rgbcrop - mean) / std).last_hidden_state
                tokens = tokens[:, 1 + dino.config.num_register_tokens :]
                grid = args.dino_size // 16
                dense = tokens.transpose(1, 2).reshape(2, -1, grid, grid)
                dense = F.interpolate(dense, size=(16, 16), mode="bilinear", align_corners=False)
                features["dinov3_rgb"].append(dense.flatten(2).transpose(1, 2))
            for name, views in features.items():
                pooled[name].append(views[0].mean(1).cpu())
                for direction in (1, 2):
                    ia, ib = crop_correspondence(origins[0], origins[direction], device=device)
                    errors = match_metrics(views[0], views[direction], ia, ib).cpu()
                    for camera in (0, 1):
                        values = errors[camera]
                        rows.append(
                            dict(
                                model=name,
                                source=sid,
                                endpoint=end,
                                camera=camera,
                                lighting=source.lighting,
                                direction=direction,
                                queries=len(values),
                                mean_px=float(values.mean()),
                                p90_px=float(values.quantile(0.9)),
                                exact_fraction=float((values == 0).float().mean()),
                            )
                        )
            identities.extend(
                [
                    dict(source=sid, task=source.task_name, lighting=source.lighting, camera=c)
                    for c in (0, 1)
                ]
            )
            if (index + 1) % 10 == 0:
                write_json(
                    args.output / "status.json", dict(windows=index + 1, total=len(keys), step=step)
                )
                print(index + 1, len(keys), flush=True)
    summary = {}
    for name in pooled:
        part = [r for r in rows if r["model"] == name]
        summary[name] = {}
        for group in ["all"] + sorted({r["lighting"] for r in part}):
            group_rows = [r for r in part if group == "all" or r["lighting"] == group]
            summary[name][group] = {
                k: float(np.mean([r[k] for r in group_rows]))
                for k in ["mean_px", "p90_px", "exact_fraction"]
            }
            summary[name][group]["queries"] = sum(r["queries"] for r in group_rows)
        emb = F.normalize(torch.cat(pooled[name]).float(), dim=-1)
        sim = emb @ emb.T
        for i, identity in enumerate(identities):
            excluded = torch.tensor([other["source"] == identity["source"] for other in identities])
            sim[i, excluded] = -torch.inf
        nearest = sim.argmax(1).tolist()
        eligible = [
            i
            for i, a in enumerate(identities)
            if any(a["task"] == b["task"] and a["source"] != b["source"] for b in identities)
        ]
        summary[name]["task_retrieval_proxy"] = {
            "correct": sum(
                identities[i]["task"] == identities[nearest[i]]["task"] for i in eligible
            ),
            "eligible": len(eligible),
            "total": len(identities),
        }
    write_json(
        args.output / "results.json",
        dict(
            step=step,
            rows=rows,
            summary=summary,
            protocol=(
                "FP32; three endpoints/source, both cameras; native224 crops displaced28px; "
                "border excluded; DINOv3-L/16 normal-light paired RGB endpoint versus "
                "six-frame pseudo-RAW; DINO input/grid recorded separately; same source excluded "
                "in task retrieval; task label proxy is not object semantics; "
                "p90 is mean of per-view p90, not pooled quantile"
            ),
            source_count=len(selected),
            dino_input_size=args.dino_size,
            dino_native_grid=args.dino_size // 16,
            windows=len(keys),
            subset=bool(args.per_light),
        ),
    )
    if semantic:
        from owac.runtime.raw_semantic_visualize import export_queries

        export_queries(args.encoder, args.output / "queries", device="cpu")
    write_json(args.output / "status.json", dict(status="completed", windows=len(keys), step=step))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
