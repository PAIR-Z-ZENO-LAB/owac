"""Export every J1 diagnostic panel as a standalone 500px scientific figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path('outputs/j1-visuals-20261029');out=r/'compact';out.mkdir(exist_ok=True)
z=np.load(r/'features.npz');d=json.loads((r/'diagnostics.json').read_text());meta=json.loads((r/'manifest.json').read_text())
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
imgs=np.clip(z['images'],0,1)**(1/2.2)
x=np.concatenate([z['initial'].reshape(-1,1152),z['trained'].reshape(-1,1152)])
mu=x.mean(0);_,_,v=np.linalg.svd((x-mu)[::8],full_matrices=False);y=(x-mu)@v[:3].T
lo,hi=np.percentile(y,[1,99],axis=0);c=np.clip((y-lo)/(hi-lo),0,1).reshape(2,20,16,16,3)
sections=[]
def card(name,img,title,overlay=None,point=None):
 fig=plt.figure(figsize=(5,5),dpi=100);ax=fig.add_axes([0,0,1,1]);ax.imshow(img,extent=(0,224,224,0),interpolation='nearest')
 if overlay is not None: ax.imshow(overlay,extent=(0,224,224,0),cmap='magma',vmin=0,vmax=error_max,alpha=.85,interpolation='nearest')
 if point is not None:
  px,py,color,mark=point;ax.scatter(px,py,s=220,c=color,marker=mark,edgecolors='white',linewidths=2,zorder=5)
 ax.text(5,9,title,va='top',fontsize=13,color='white',bbox=dict(facecolor='black',alpha=.8,pad=4))
 ax.set_xlim(0,224);ax.set_ylim(224,0);ax.axis('off');fig.savefig(out/name);plt.close(fig)
 return name
error_max=float(np.nanmax(z['errors'][[0,4,8,12,16]]))
for i,label,slug in [(0,'正常光','normal'),(4,'弱光','low'),(8,'极弱光','extreme_low'),(12,'过亮','over'),(16,'极亮','extreme_over')]:
 panels=[]
 def add(kind,img,title,**kw):
  name=f'{slug}_{kind}.png';card(name,img,title,**kw);panels.append((title,name))
 add('input',imgs[i],label+'｜完整观测（仅显示提亮）')
 add('mask',z['masked'][i]**(1/2.2),label+'｜黑块是模型看不到的区域')
 add('error',imgs[i],label+'｜越亮，特征预测误差越大',overlay=z['errors'][i])
 add('initial',c[0,i],label+'｜训练前特征（非语义标签）')
 add('trained',c[1,i],label+'｜79,000步特征（非语义标签）')
 add('query',imgs[i],label+'｜青点：待匹配位置',point=(91,91,'cyan','o'))
 m=next(m for m in d['matches'] if m['sample']==i and m['model']==2 and m['query']==[6,6]);py,px=m['match']
 add('match',imgs[i+1],label+'｜15帧后，红叉：模型匹配处',point=(px*14+7,py*14+7,'red','X'))
 row=next(x for x in d['retrieval'] if x['query']==i)
 for rank,j in enumerate(row['neighbors'],1):
  light={'Normal':'正常光','Low':'弱光','ExtremeLow':'极弱光','Over':'过亮','ExtremeOver':'极亮'}[meta['lights'][j]]
  add(f'retrieval{rank}',imgs[j],f'检索第{rank}名｜{light}｜相似度{row["scores"][rank-1]:.3f}')
 sections.append((label,panels))
html='<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>J1完整小图</title><style>body{font:18px sans-serif;max-width:560px;margin:20px auto;padding:12px}img{display:block;width:100%;max-width:500px;height:auto;object-fit:contain}figure{margin:20px 0}figcaption{margin:10px 0}</style><h1>79,000步：完整单图版</h1><p>每图500×500，无并排画布。亮度增强只用于显示；PCA颜色不是物体类别。跨帧匹配没有人工真值，检索仅10条候选轨迹并排除同源。</p><p>误差热图共用色标：深色0，亮色'+f'{error_max:.3f}'+'；透明处未评价。</p>'
for label,panels in sections:
 html+='<h2>'+label+'</h2>'+''.join(f'<figure><figcaption>{title}</figcaption><a href="{name}"><img src="{name}" width="500" height="500"></a></figure>' for title,name in panels)
(out/'index.html').write_text(html)
(r/'simple'/'index.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../compact/index.html"><a href="../compact/index.html">打开完整单图版</a>')
print('55 standalone panels generated')
