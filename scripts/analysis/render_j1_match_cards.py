"""Compact standalone point-match cards, from saved model outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path('outputs/j1-visuals-20261029')
z=np.load(r/'features.npz'); d=json.loads((r/'diagnostics.json').read_text())
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
m=next(m for m in d['matches'] if m['sample']==0 and m['model']==2 and m['query']==[6,6])
y,x=m['match']
for idx,name,title,xy,color,marker,textpos in [(0,'query_full','① 查询：夹爪附近',(91,91),'cyan','o',(25,35)),(1,'match_full','② 模型匹配：桌面底部',(x*14+7,y*14+7),'red','X',(80,180))]:
 fig,ax=plt.subplots(figsize=(5,5),dpi=100)
 fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
 ax.imshow(np.clip(z['images'][idx],0,1)**(1/2.2))
 ax.scatter(*xy,s=220,c=color,marker=marker,edgecolors='white',linewidths=2,zorder=5)
 ax.annotate(title,xy=xy,xytext=textpos,fontsize=14,color='white',weight='bold',bbox=dict(facecolor='black',alpha=.8,pad=5),arrowprops=dict(arrowstyle='->',color=color,lw=3))
 ax.set_xlim(-.5,223.5);ax.set_ylim(223.5,-.5);ax.axis('off')
 fig.savefig(r/'simple'/f'{name}.png');plt.close(fig)
