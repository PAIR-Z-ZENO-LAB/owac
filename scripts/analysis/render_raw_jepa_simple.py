"""Readable Chinese single-example cards from the saved J1 feature export."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path('outputs/j1-visuals-20261029'); out=r/'simple';out.mkdir(exist_ok=True)
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
plt.rcParams['font.size']=16
z=np.load(r/'features.npz'); d=json.loads((r/'diagnostics.json').read_text()); meta=json.loads((r/'manifest.json').read_text())
imgs=np.clip(z['images'],0,1)**(1/2.2)
def setup(title,subtitle):
 fig,ax=plt.subplots(1,2,figsize=(11,6));fig.subplots_adjust(left=.04,right=.96,bottom=.16,top=.76,wspace=.12)
 fig.suptitle(title,fontsize=23,y=.98);fig.text(.5,.87,subtitle,ha='center',fontsize=15)
 for a in ax:a.axis('off')
 return fig,ax
def finish(fig,name,note):
 fig.text(.5,.055,note,ha='center',fontsize=14);fig.savefig(out/name,dpi=120);plt.close(fig)
for i,label in [(0,'正常光'),(4,'弱光'),(8,'极弱光')]:
 suffix={0:'normal',4:'low',8:'extreme_low'}[i]
 fig,ax=setup('① 挡住画面后，哪里猜得差？',f'{label} · 79,000步 · 猜的是特征，不是补全照片')
 ax[0].imshow(z['masked'][i]**(1/2.2));ax[0].set_title('模型看到的画面：黑块被遮挡')
 ax[1].imshow(imgs[i],extent=(0,16,16,0));v=np.nanmax(z['errors'][i]);im=ax[1].imshow(z['errors'][i],extent=(0,16,16,0),cmap='magma',vmin=0,vmax=v,alpha=.85);ax[1].set_title('亮色块：预测误差较大')
 fig.colorbar(im,ax=ax[1],shrink=.65,pad=.02)
 finish(fig,f'01_{suffix}.png','只评价被遮挡的区域；没有覆盖颜色的区域不代表误差为零。')
 fig,ax=setup('③ 过了15帧，还能找到同一个地方吗？',f'{label} · 青点是查询位置，红叉是模型找到的位置')
 ax[0].imshow(imgs[i]);ax[0].set_title('较早一帧：记住青点附近')
 ax[0].plot(91,91,'o',color='cyan',markersize=12,markeredgecolor='black')
 ax[1].imshow(imgs[i+1]);ax[1].set_title('15帧后：模型找到了红叉处')
 m=next(m for m in d['matches'] if m['sample']==i and m['model']==2 and m['query']==[6,6]);y,x=m['match'];ax[1].plot(x*14+7,y*14+7,'x',color='red',markersize=17,markeredgewidth=4)
 note='此例找错了：查询在夹爪附近，结果落到桌面底部。' if i==0 else '点留在相近坐标，不足以证明模型在跟踪物体；这里没有人工对应真值。'
 finish(fig,f'03_{suffix}.png',note)
# Use the same shared PCA convention as original, retaining init comparison in a separate card.
x=np.concatenate([z['initial'].reshape(-1,1152),z['trained'].reshape(-1,1152)]);mu=x.mean(0);_,_,v=np.linalg.svd((x-mu)[::8],full_matrices=False);y=(x-mu)@v[:3].T;lo,hi=np.percentile(y,[1,99],axis=0);c=np.clip((y-lo)/(hi-lo),0,1).reshape(2,20,16,16,3)
fig,ax=setup('② 模型看到的“区域差别”像物体吗？','左边看真实物体，右边看训练后特征；颜色不是物体类别')
ax[0].imshow(imgs[0]);ax[0].set_title('原始观测（仅显示提亮）');ax[1].imshow(c[1,0],interpolation='nearest');ax[1].set_title('训练后：每个色块对应一块图像')
finish(fig,'02_normal.png','看色块是否跟随物体轮廓；这张图尚不能证明模型分清了杯子、机械臂和桌面。')
fig,ax=setup('② 补充：训练前后到底改变了什么？','相同观测、相同PCA坐标和颜色尺度；颜色变化不等于语义变好')
ax[0].imshow(c[0,0]);ax[0].set_title('训练前：公开视觉＋新RAW入口');ax[1].imshow(c[1,0]);ax[1].set_title('训练后：79,000步')
finish(fig,'02_compare.png','特征明显改变，但仍需通过物体对应来判断改变是否有用。')
for i,suffix,label in [(0,'normal','正常光'),(4,'low','弱光')]:
 row=next(x for x in d['retrieval'] if x['query']==i);j=row['neighbors'][0]
 fig,ax=setup('④ 模型觉得哪张画面最像？',f'{label}查询 · 从10条验证轨迹中检索，排除查询自身轨迹')
 ax[0].imshow(imgs[i]);ax[0].set_title('给模型的查询图');ax[1].imshow(imgs[j]);ax[1].set_title('排第一的结果：'+{'Over':'过亮','ExtremeOver':'极亮','Low':'弱光','Normal':'正常光','ExtremeLow':'极弱光'}[meta['lights'][j]])
 finish(fig,f'04_{suffix}.png','比较物体和布局是否相似；明暗也可能影响排序，这不是检索准确率测试。')
files=['03_normal.png','01_normal.png','02_normal.png','02_compare.png','04_normal.png','03_low.png','01_low.png','04_low.png','03_extreme_low.png','01_extreme_low.png']
(out/'index.html').write_text('<meta charset="utf-8"><title>J1 中文单例图集</title><style>body{font:20px sans-serif;max-width:1100px;margin:auto;background:#fafafa}img{width:100%;height:auto;margin-bottom:36px}a{color:#1467a3}</style><h1>79,000步：直观看模型学到了什么</h1><p>一张图只讲一件事。全部沿用上一版真实特征，无重新训练。显示统一提亮，输入没有改变。</p>'+''.join(f'<a href="{f}"><img src="{f}"></a>' for f in files))
print(out)
