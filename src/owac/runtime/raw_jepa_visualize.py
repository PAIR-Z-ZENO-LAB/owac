"""Small deterministic held-source feature visualization export (CPU, no training)."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.training.raw_jepa import block_mask, build_jepa, hide_pixels
from owac.training.raw_jepa_train import RawWindows, load_inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    receipt = json.loads((a.run_root / 'checkpoint.json').read_text())
    if not receipt['complete']:
        raise RuntimeError('Checkpoint write in progress; rerun after completion')
    # Eager CPU load: subsequent training writes cannot mutate our in-memory snapshot.
    state = torch.load(a.run_root / 'latest.pt', map_location='cpu', weights_only=False)
    cfg = state['config']
    online, target, projector = build_jepa(cfg, 'cpu')
    del projector
    sources, _ = load_inventory(cfg['data']['inventory'])
    cfg['training']['working_cache_sources'] = 2
    dataset = RawWindows(sources, cfg)
    selected = []
    for light in ['Normal', 'Low', 'ExtremeLow', 'Over', 'ExtremeOver']:
        selected.extend(sorted([s for s in dataset.sources.values()
                                if s.split == 'validation' and s.lighting == light],
                               key=lambda s: s.group_id)[:2])
    keys = [(s.group_id, e) for s in selected
            for e in [(s.num_frames - 1 + 5)//2, min(s.num_frames-1, (s.num_frames-1+5)//2+15)]]
    images, initial, trained, errors, masks = [], [], [], [], []
    mask = block_mask(16, random.Random(101), **{k: cfg['mask'][k] for k in ['blocks','scale']})
    online.eval()
    with torch.inference_mode():
        for i, key in enumerate(keys):
            raw = dataset.batch([key], 'cpu')[:1]
            images.append(raw[0,-1].permute(1,2,0).numpy())
            initial.append(online.encoder(raw).float().numpy()[0])
            print('initial', i, flush=True)
        online.load_state_dict(state['online'])
        target.load_state_dict(state['target'])
        step = state['step']
        del state
        for i, key in enumerate(keys):
            raw = dataset.batch([key], 'cpu')[:1]
            trained.append(online.encoder(raw).float().numpy()[0])
            pred = online(raw, mask)
            truth = F.layer_norm(target(raw).float(), (1152,))[:,mask]
            err = F.smooth_l1_loss(pred.float(), truth, reduction='none').mean(-1)[0]
            grid = np.full(256, np.nan, dtype=np.float32)
            grid[mask.numpy()] = err.numpy()
            errors.append(grid.reshape(16,16))
            masks.append(hide_pixels(raw,mask,14)[0,-1].permute(1,2,0).numpy())
            print('trained', i, flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output/'features.npz', images=np.array(images), initial=np.array(initial),
                        trained=np.array(trained), errors=np.array(errors), masked=np.array(masks))
    (a.output/'manifest.json').write_text(json.dumps({'step':step, 'keys':keys,
        'lights':[s.lighting for s in selected for _ in range(2)],
        'selection':'first two sorted validation sources per lighting; midpoint and +15 frames; camera0',
        'display':'RAW pseudo-RGB, optional display-only gamma; model inputs unchanged',
        'initialization':'public pi05 vision + identical seed101 new stem and zero gate',
        'checkpoint':'latest.pt eager CPU snapshot; no optimizer updates'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
