"""Layerwise translation probe: zero fill versus native overlapping RAW crops."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from owac.training.raw_jepa import build_jepa
from owac.training.raw_jepa_train import RawWindows, load_inventory
from owac.runtime.raw_jepa_translation import correspondence, translate


def extract(encoder, raw):
    values = {}
    def capture(name):
        def hook(module, args, output):
            values[name] = output[0][0].float().clone()
        return hook
    def input_hook(module, args):
        values['stem_gate'] = (args[0][0] - encoder.vision.embeddings.position_embedding.weight).float().clone()
        values['position_input'] = args[0][0].float().clone()
    handles = [encoder.vision.encoder.layers[0].register_forward_pre_hook(input_hook)]
    for n in (3, 9, 15, 21, 27):
        handles.append(encoder.vision.encoder.layers[n-1].register_forward_hook(capture(f'block{n:02}')))
    try:
        values['final_norm'] = encoder(raw)[0].float()
    finally:
        for handle in handles:
            handle.remove()
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    receipt = json.loads((args.run_root/'checkpoint.json').read_text())
    if not receipt['encoder_complete']:
        raise RuntimeError('Encoder export not complete')
    state = torch.load(args.run_root/'encoder.pt', map_location='cpu', weights_only=False)
    cfg = state['config']; torch.manual_seed(cfg['training']['seed'])
    model, target, projector = build_jepa(cfg, 'cpu'); del target, projector
    model.eval(); cfg['training']['working_cache_sources'] = 2
    sources, _ = load_inventory(cfg['data']['inventory']); ds = RawWindows(sources, cfg)
    selected = []
    for light in ['Normal','Low','ExtremeLow','Over','ExtremeOver']:
        selected.extend(sorted([s for s in ds.sources.values() if s.split=='validation' and s.lighting==light], key=lambda s:s.group_id)[:2])
    keys = [(s.group_id,(s.num_frames-1+5)//2) for s in selected]
    rows=[]; arrays={}
    with torch.inference_mode():
        for stage in ['initial','trained']:
            if stage=='trained': model.encoder.load_state_dict(state['encoder'])
            for i,(source,end) in enumerate(keys):
                raw=ds.batch([(source,end)],'cpu')[:1]
                native=ds.cache.get(source).burst(end)[:1]
                if native.shape[-2:] != (256,256):
                    raise ValueError(f'Expected native 256x256, got {native.shape}')
                views={'base':raw,'zero_x':translate(raw,28,0),'zero_y':translate(raw,0,28),
                       'crop0':native[...,:224,:224], 'crop_x':native[...,:224,28:252], 'crop_y':native[...,28:252,:224]}
                features={name:extract(model.encoder,value) for name,value in views.items()}
                for protocol,a,b,dx,dy in [('zero_x','base','zero_x',28,0),('zero_y','base','zero_y',0,28),
                                           ('crop_x','crop_x','crop0',28,0),('crop_y','crop_y','crop0',0,28)]:
                    if stage=='initial':
                        arrays[f'{i}_{protocol}_a']=views[a][0,-1].permute(1,2,0).numpy()
                        arrays[f'{i}_{protocol}_b']=views[b][0,-1].permute(1,2,0).numpy()
                    for layer in features[a]:
                        result=correspondence(features[a][layer],features[b][layer],dx,dy)
                        valid=result['valid']; errors=result['error'][valid]
                        rows.append(dict(stage=stage,sample=i,lighting=selected[i].lighting,protocol=protocol,layer=layer,
                                         n=int(valid.sum()),mean=float(errors.mean()),median=float(np.median(errors)),
                                         exact_fraction=float((errors==0).mean()),stationary_fraction=float((result['predicted'][valid]==np.arange(256)[valid]).mean())))
                        arrays[f'{stage}_{i}_{protocol}_{layer}_error']=result['error']
                        arrays[f'{stage}_{i}_{protocol}_{layer}_predicted']=result['predicted']
                print(stage,i,flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'maps.npz',**arrays)
    (args.output/'results.json').write_text(json.dumps(dict(step=state['step'],keys=keys,rows=rows,
      protocol='CPU FP32; camera0; 6-frame causal reset per view; 10 fixed held-source windows; cosine NN; 28px exact two-patch displacement; 224 valid source patches per direction; zero: 256->224 full-frame resize then fill; crop: native256, origins(0,0)/(28,0)/(0,28),224 square without resize or fill; crop candidate matching restricted to shared support; blocks before post norm; stem_gate before position embedding'),indent=2)+'\n')

if __name__=='__main__': main()
