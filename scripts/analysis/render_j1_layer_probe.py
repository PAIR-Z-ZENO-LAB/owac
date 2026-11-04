"""Readable layerwise paired error curves and per-source deltas."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path('outputs/j1-layer-probe-20261029'); d=json.loads((r/'results.json').read_text()); rows=d['rows']
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
layers=['stem_gate','position_input','block03','block09','block15','block21','block27','final_norm']
labels=['stem','加位置','第3层','第9层','第15层','第21层','第27层','最终LN']
summary=[]
for protocol,title in [('zero','零填充平移'),('crop','无填充重叠裁剪')]:
 fig,ax=plt.subplots(figsize=(6,5),dpi=120,layout='constrained')
 for stage,label in [('initial','初始化'),('trained',f"训练后 {d['step']}步")]:
  vals=[]
  for layer in layers:
   part=[x for x in rows if x['stage']==stage and x['layer']==layer and x['protocol'].startswith(protocol)]
   value=float(np.mean([x['mean'] for x in part]));vals.append(value)
   summary.append(dict(protocol=protocol,stage=stage,layer=layer,mean=value,exact=float(np.mean([x['exact_fraction'] for x in part]))))
  ax.plot(range(8),vals,'o-',label=label)
 ax.axhline(28,color='gray',ls='--',label='始终匹配原坐标：28px')
 ax.set_xticks(range(8),labels,rotation=35); ax.set_ylabel('平均位置误差（像素，越低越好）'); ax.set_title(title+'：不同层的匹配能力\n10源 × 水平/垂直28px，训练前后配对')
 ax.legend(fontsize=9);ax.grid(alpha=.2);fig.savefig(r/f'{protocol}_curve.png');plt.close(fig)
 deltas=[]
 for i in range(10):
  deltas.append([np.mean([x['mean'] for x in rows if x['sample']==i and x['stage']=='trained' and x['layer']==l and x['protocol'].startswith(protocol)])-np.mean([x['mean'] for x in rows if x['sample']==i and x['stage']=='initial' and x['layer']==l and x['protocol'].startswith(protocol)]) for l in layers])
 fig,ax=plt.subplots(figsize=(6,5),dpi=120,layout='constrained');limit=max(abs(np.asarray(deltas)).max(),1)
 im=ax.imshow(deltas,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto');ax.set_xticks(range(8),labels,rotation=35)
 ax.set_yticks(range(10),['正常1','正常2','低光1','低光2','极低1','极低2','过曝1','过曝2','极过1','极过2'])
 ax.set_title(title+'：训练后 − 初始化误差\n红色变差，蓝色变好');fig.colorbar(im,ax=ax,label='像素');fig.savefig(r/f'{protocol}_delta.png');plt.close(fig)
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
html='<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{max-width:760px;margin:auto;font:18px sans-serif}img{width:100%;height:auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>逐层空间匹配诊断</h1><p>同10条验证源、camera0、六帧因果窗口。28像素等于2个patch，量化下界0。每方向每源224个有效查询，共4480个查询/层/模型/协议。零填充使用原训练缩放；裁剪使用原始256像素图像，不缩放、不填充，因此两协议的视野和尺度也不同，不能仅将协议差异归因于填充值。</p>'
for protocol in ['zero','crop']:
 for kind in ['curve','delta']:html+=f'<img src="{protocol}_{kind}.png">'
html+='<p>stem为时序融合后、位置编码前；第27层为最终LayerNorm前。匹配均直接用余弦最近邻，无额外训练。初始化对照也使用新RAW stem，不代表原RGB模型的能力。</p>'
(r/'index.html').write_text(html)
for x in summary: print(x)
