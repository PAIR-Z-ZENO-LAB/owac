"""Compact correspondence errors against known translation ground truth."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path('outputs/j1-translation-20261029');d=json.loads((r/'results.json').read_text());z=np.load(r/'translation.npz')
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
links=[]
for i in [0,2,4,6,8]:
 for k,(dx,dy) in enumerate(d['shifts']):
  base=f'trained_{i}_{k}_';valid=z[base+'valid'];img=z[f'image_{i}']**(1/2.2)
  moved=np.zeros_like(img);moved[dy:,dx:]=img[:224-dy,:224-dx]
  for kind in ['initial','trained','improvement','lines']:
   fig,ax=plt.subplots(figsize=(5,5),dpi=100);fig.subplots_adjust(left=.02,right=.86,bottom=.02,top=.88)
   name=f'{i}_{k}_{kind}.png'
   if kind=='lines':
    ax.imshow(moved,extent=(0,224,224,0));centers=z[base+'centers'];truth=z[base+'truth'];pred=centers[z[base+'predicted']]
    for q in [85,102,136,170]:
     ax.plot([truth[q,0],pred[q,0]],[truth[q,1],pred[q,1]],color='yellow',lw=1.5)
     ax.scatter(*truth[q],c='lime',marker='o',s=70,edgecolors='black');ax.scatter(*pred[q],c='red',marker='x',s=90,linewidths=2)
    title='绿圈：正确位置；红叉：模型位置'
   else:
    if kind=='improvement':
     val=z[f'initial_{i}_{k}_error']-z[base+'error'];cmap='RdBu';vmin,vmax=-100,100;title='训练改善：蓝色变好，红色变差'
    else:
     val=z[f'{kind}_{i}_{k}_error'].copy();cmap='magma';vmin,vmax=0,224;title=('训练前' if kind=='initial' else '训练后')+'：匹配位置误差（像素）'
    val=val.copy();val[~valid]=np.nan
    ax.set_facecolor('#ccc');im=ax.imshow(val.reshape(16,16),cmap=cmap,vmin=vmin,vmax=vmax,extent=(0,224,224,0));fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
   ax.axis('off');ax.set_xlim(0,224);ax.set_ylim(224,0)
   fig.suptitle(title+f'\n右移{dx}、下移{dy}像素',fontsize=13)
   fig.savefig(r/name);plt.close(fig);links.append((i,k,title,name))
html='<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{max-width:550px;margin:auto;font:18px sans-serif}img{width:100%;max-width:500px;height:auto}</style><h1>平移真值对照：'+str(d['step'])+'步</h1><p>灰色不评价。误差图色标一致；改善图正值代表训练后误差降低。7像素平移无法精确落在14像素特征网格，最小位置误差为7像素。连线选固定四点，不按效果筛选。绿圈是真值，红叉是预测，黄线越长错得越远。</p>'
for i,k,title,name in links:html+=f'<h3>样本{i} / 位移{d["shifts"][k]}：{title}</h3><img src="{name}">'
(r/'index.html').write_text(html)
for dx,dy in d['shifts']:
 for stage in ['initial','trained']:
  rows=[x for x in d['rows'] if x['dx']==dx and x['dy']==dy and x['stage']==stage]
  print(dx,dy,stage,'mean',np.mean([x['mean'] for x in rows]),'same',np.mean([x['same_position_fraction'] for x in rows]))
