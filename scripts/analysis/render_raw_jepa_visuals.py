from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
r=p.parse_args().root
z=np.load(r/'features.npz'); meta=json.loads((r/'manifest.json').read_text()); step=meta['step']
imgs=z['images']; before=z['initial']; after=z['trained']; lights=meta['lights']
# Same display transform across all views; no per-image contrast stretching.
display=np.clip(imgs,0,1)**(1/2.2)
def pic(ax,x,title):
 ax.imshow(x); ax.set_title(title,fontsize=9); ax.axis('off')
idx=[0,4,8,12,16]
fig,axes=plt.subplots(5,4,figsize=(12,14))
vmax=np.nanmax(z['errors'][idx])
for row,i in enumerate(idx):
 pic(axes[row,0],imgs[i],lights[i]+' | raw /255')
 pic(axes[row,1],display[i],'Display gamma 1/2.2')
 pic(axes[row,2],z['masked'][i]**(1/2.2),'Masked input (same mask all 6 frames)')
 ax=axes[row,3]; ax.imshow(display[i],extent=(0,16,16,0)); im=ax.imshow(z['errors'][i],cmap='magma',vmin=0,vmax=vmax,alpha=.85,extent=(0,16,16,0)); ax.axis('off'); ax.set_title('Masked-token SmoothL1 error',fontsize=9)
fig.colorbar(im,ax=axes[:,3],shrink=.6);fig.suptitle(f'1. RAW masked prediction | checkpoint {step} | visible pixels have no error value')
fig.savefig(r/'01_prediction.png',dpi=130);plt.close(fig)
# Shared PCA basis and shared color scaling across both models and all samples.
x=np.concatenate([before.reshape(-1,1152),after.reshape(-1,1152)])
mu=x.mean(0); _,_,v=np.linalg.svd((x-mu)[::8],full_matrices=False)
y=(x-mu)@v[:3].T
lo,hi=np.percentile(y,[1,99],axis=0); colors=np.clip((y-lo)/(hi-lo),0,1).reshape(2,20,16,16,3)
fig,axes=plt.subplots(5,3,figsize=(10,14))
for row,i in enumerate(idx):
 pic(axes[row,0],display[i],lights[i]+' | display RAW')
 pic(axes[row,1],colors[0,i],'Public vision + new RAW stem (init)')
 pic(axes[row,2],colors[1,i],f'Trained encoder {step}')
fig.suptitle('2. Spatial features | shared PCA basis / shared color scale\nColors are feature coordinates, NOT semantic labels')
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(r/'02_features.png',dpi=130);plt.close(fig)
# Three preselected positions; no object annotation or ground truth matching.
points=[(6,6),(8,8),(10,10)]; palette=['cyan','lime','magenta']
def norm(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)
fig,axes=plt.subplots(5,4,figsize=(12,14)); matches=[]
for row,i in enumerate(idx):
 pic(axes[row,0],display[i],lights[i]+' | query t')
 for (yy,xx),c in zip(points,palette): axes[row,0].plot(xx*14+7,yy*14+7,'o',color=c)
 for col,features in [(1,before),(2,after)]:
  pic(axes[row,col],display[i+1],('Init' if col==1 else 'Trained')+' matches at t+15')
  sim=norm(features[i])@norm(features[i+1]).T
  for (yy,xx),c in zip(points,palette):
   j=sim[yy*16+xx].argmax(); ay,ax=divmod(int(j),16)
   axes[row,col].plot(ax*14+7,ay*14+7,'o',color=c)
   matches.append(dict(sample=i,model=col,query=[yy,xx],match=[ay,ax],cosine=float(sim[yy*16+xx,j])))
 sim=norm(after[i])[8*16+8]@norm(after[i+1]).T
 ax=axes[row,3]; ax.imshow(display[i+1],extent=(0,16,16,0)); im=ax.imshow(sim.reshape(16,16),cmap='viridis',vmin=-1,vmax=1,alpha=.7);ax.axis('off');ax.set_title('Center query cosine heatmap',fontsize=9)
fig.suptitle('3. Cross-frame point matching | fixed query points, not annotated objects\nSame source / camera; matches may reflect position rather than object identity')
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(r/'03_matching.png',dpi=130);plt.close(fig)
# Search across different sources only, using mean-pooled encoder outputs.
features=norm(after.mean(1)); sims=features@features.T
fig,axes=plt.subplots(5,5,figsize=(15,14)); retrieval=[]
for row,i in enumerate(idx):
 pic(axes[row,0],display[i],lights[i]+' | query')
 eligible=[j for j in range(0,20,2) if j//2!=i//2]
 ranked=sorted(eligible,key=lambda j:-sims[i,j])[:4]
 retrieval.append(dict(query=i,neighbors=ranked,scores=[float(sims[i,j]) for j in ranked]))
 for col,j in enumerate(ranked,1): pic(axes[row,col],display[j],f'#{col} {lights[j]} | cos {sims[i,j]:.3f}\nsource {j//2}, frame {meta["keys"][j][1]}')
fig.suptitle('4. Nearest observation retrieval | trained mean-pooled features\n10 held-source candidates; same source excluded; not a retrieval accuracy evaluation')
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(r/'04_retrieval.png',dpi=130);plt.close(fig)
(r/'diagnostics.json').write_text(json.dumps(dict(matches=matches,retrieval=retrieval),indent=2))
(r/'index.html').write_text('<meta charset="utf-8"><title>J1 RAW-JEPA 可视化</title><style>body{max-width:1400px;margin:auto;font:18px sans-serif;background:#eee}img{width:100%}</style><h1>J1 RAW-JEPA — '+str(step)+'步检查点</h1><p>验证集固定样本，未接动作专家或外部RAW教师。亮度增强仅用于显示。PCA颜色不是语义标签；点匹配无人工真值；检索排除同源。</p>'+''.join(f'<h2>{s}</h2><img src="{f}">' for s,f in [('遮挡预测误差','01_prediction.png'),('初始化与训练后空间特征','02_features.png'),('跨帧点匹配','03_matching.png'),('相似观测检索','04_retrieval.png')]))
print('Rendered',step)
