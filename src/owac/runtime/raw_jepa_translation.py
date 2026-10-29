"""Known-translation correspondence diagnostic on held-source RAW windows."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from owac.training.raw_jepa import build_jepa
from owac.training.raw_jepa_train import RawWindows, load_inventory


def translate(raw, dx, dy):
    result = torch.zeros_like(raw)
    result[..., dy:, dx:] = raw[..., :raw.shape[-2]-dy, :raw.shape[-1]-dx]
    return result


def correspondence(a, b, dx, dy):
    yy, xx = np.mgrid[:16,:16]
    centers = np.stack([xx.ravel()*14+7, yy.ravel()*14+7], -1)
    truth = centers + [dx,dy]
    # Entire source patch stays in frame; entire destination patch avoids zero fill.
    valid = (truth[:,0]+7<=224)&(truth[:,1]+7<=224)
    candidates = (centers[:,0]-7>=dx)&(centers[:,1]-7>=dy)
    similarity = F.normalize(a,dim=-1) @ F.normalize(b,dim=-1).T
    similarity[:,~torch.from_numpy(candidates)] = -torch.inf
    predicted = similarity.argmax(-1).numpy()
    error = np.linalg.norm(centers[predicted]-truth,axis=-1)
    floor = np.linalg.norm(centers[candidates][None]-truth[:,None],axis=-1).min(-1)
    return dict(predicted=predicted,error=error,valid=valid,truth=truth,centers=centers,
                quantization_floor=floor,stationary_error=np.full(256,np.hypot(dx,dy)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(8)
    receipt=json.loads((args.run_root/'checkpoint.json').read_text())
    if not receipt['encoder_complete']: raise RuntimeError('Encoder export not complete')
    state=torch.load(args.run_root/'encoder.pt',map_location='cpu',weights_only=False)
    cfg=state['config'];torch.manual_seed(cfg['training']['seed'])
    model,target,projector=build_jepa(cfg,'cpu');del target,projector
    model.eval();sources,_=load_inventory(cfg['data']['inventory']);cfg['training']['working_cache_sources']=2
    ds=RawWindows(sources,cfg);selected=[]
    for light in ['Normal','Low','ExtremeLow','Over','ExtremeOver']:
        selected.extend(sorted([s for s in ds.sources.values() if s.split=='validation' and s.lighting==light],key=lambda s:s.group_id)[:2])
    keys=[(s.group_id,(s.num_frames-1+5)//2) for s in selected]
    shifts=[(28,0),(0,28),(7,0),(0,7)];data={};rows=[]
    with torch.inference_mode():
        for stage in ['initial','trained']:
            if stage=='trained':model.encoder.load_state_dict(state['encoder'])
            for i,key in enumerate(keys):
                raw=ds.batch([key],'cpu')[:1];a=model.encoder(raw)[0].float()
                if stage=='initial':data[f'image_{i}']=raw[0,-1].permute(1,2,0).numpy()
                for k,(dx,dy) in enumerate(shifts):
                    moved=translate(raw,dx,dy);b=model.encoder(moved)[0].float();result=correspondence(a,b,dx,dy)
                    for name,value in result.items():data[f'{stage}_{i}_{k}_{name}']=value
                    valid=result['valid'];err=result['error'][valid]
                    rows.append(dict(stage=stage,sample=i,lighting=selected[i].lighting,dx=dx,dy=dy,n=int(valid.sum()),mean=float(err.mean()),median=float(np.median(err)),stationary=float(np.hypot(dx,dy)),floor=float(result['quantization_floor'][valid].mean()),same_position_fraction=float((result['predicted'][valid]==np.arange(256)[valid]).mean())))
                print(stage,i,flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'translation.npz',**data)
    (args.output/'results.json').write_text(json.dumps(dict(step=state['step'],keys=keys,shifts=shifts,rows=rows,protocol='CPU FP32; 6 frames shifted identically after resize; camera0; first two sorted validation sources per light; zero padding; full-patch overlap only; cosine nearest neighbor; no subpatch refinement'),indent=2)+'\n')


if __name__=='__main__':main()
