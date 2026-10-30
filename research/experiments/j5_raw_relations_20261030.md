# J5：保留弱光结构，恢复区分与学习 RAW 时序对应

## 范围与诊断

用户授权按讨论方案迭代、在服务器训练测试、SwanLab监测；复用已有能力。分支 `lzh/raw-temporal-relations`，继承 J4 最终30000步纯RAW EMA的完整encoder和原型头，重开Adam与30000新更新日程。J3/J4原权重、输出和蒸馏路线均保留。无RGB训练/重建/特征监督、无动作/语言标签、无外部预训练光流或视觉网络。部署仍是单帧RAW→同一套token供PCA/query使用，没有把多帧输入收益冒称单帧结果。

固定20图诊断：J3/J4原始patch间非对角平均余弦0.30736/0.58933；去空间均值后约-0.00193/0.00446。后者接近0部分是中心化本身的性质，不是物体区分改善的证据。人工看中心化query：大片泛亮降低，但物体仍未清楚分组。中心化后的全图集特征谱有效秩87.12→26.87，支持公共成分增强之外还存在维度集中。诊断产物 `outputs/j5-raw-relations-20261030/`。最终展示仍用原始输出余弦，不把临时中心化热图替代模型能力。

## 本轮改动

1. **保留曝光能力**：继续同源RAW低曝光噪声视图和DINO/iBOT/KoLeo。曝光token余弦权重由2降至0.5；J4已完成课程，本轮从0–6EV采样。真实其他四照明保持原值。
2. **空间关系参照**：固定自身J3最终encoder，仅对Normal RAW原观测计算patch余弦关系，要求对应退化学生输出保留此关系。排除masked行列及对角线，MSE权重4。参考不是RGB模型；它只能保护已有关系，不能生成不存在的语义。
3. **实际输出多样性**：在每rank每视图4×4空间抽样的可见patch上，单位归一化特征乘sqrt(width)，做VICReg式方差及去冗余约束，权重0.5/0.02。约束实际backbone tokens而非仅投影头；不是强制所有patch两两互斥。矩统计基于本rank当前batch，不增加大缓存。
4. **过去RAW软对应**：随机当前帧t≥1，同相机读取t-gap，gap在1到min(t,4)间抽样，与第一个global视图共享裁剪、翻转、缩放及增益。EMA分别编码当前/过去RAW，以空间中心化特征在±3patch邻域做temperature0.07的软匹配。前后向概率乘积给出连续cycle权重；不设人工置信度通过阈值、不把未匹配像素硬标为背景。学生可见内容token与匹配聚合的EMA过去内容特征对齐；模糊对应自然贡献更小。权重2，前2000更新线性加入。

时序分支是自监督对应学习，不是已验证的物体运动分割。相机运动、静态背景和共同运动仍可能产生歧义；没有把slot、运动标签、多帧部署融合或对象掩码说成已经实现。

## 来源与配置

- DINOv3 Gram anchoring：借鉴patch关系保留；本轮是自己的J3 RAW参考，非官方完整训练配方。https://github.com/facebookresearch/dinov3/blob/main/dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml
- VICReg：借鉴方差/协方差约束，改为backbone可见空间token采样。https://arxiv.org/abs/2105.04906
- 时序软匹配是当前实现的实验设计，不能声称复现运动分组论文。

双5090、global192、ViT-S/16、2×256global/8×96local不变；lr3e-5、warmup1000、cosine至2e-6、EMA.996→1、30k新更新。新配置 `configs/experiments/j5_raw_relations.json`。seed20261031是随机种子，不是日期标记。

## 内存与运行

保留精确帧pread、局部fadvise释放、worker4/rank、prefetch1和8CPU线程/rank。每样本仅增加一张过去RAW crop，不读整段轨迹、不创建光流或teacher token全量缓存。冻结J3和过去EMA额外forward不保留梯度。

工程20步正常退出：约365.5源图/秒、0.525秒/步、峰值allocated22.98GiB/rank、总RAM38.84GiB、共享8.40GiB、OOM0。20步初测时temporal_weight=0.5，正式配置调整为2；两步恢复验证使用最终配置。工程输出独立，不计入正式曲线。

11项针对性测试通过：原读取/曝光/恢复测试，新增关系掩码、已知内容置换对应、匹配模糊性、时间源边界、共享几何、实际token梯度。新增代码ruff通过；全库make check仍有既有213项lint错误，保留日志，未伪报全库通过。

## 评价与产物

每1000更新保存完整online/EMA/Adam/RNG/采样位置checkpoint与独立encoder；固定20验证源全五照明，RGB只在评估图展示，不送入encoder。保留0/-2/-4/-6EV同源曝光偏移匹配探针。新增原始patch余弦/按域照明分项/有效秩；这些是机制诊断，不是表征成功门槛。

每次评估再生成4张与J4最终的同样本对比，包含RAW、配对RGB参考、共同PCA与原始query[-1,1]；覆盖两域Normal和ExtremeLow。仍须人工看区域轮廓和物体区分，不能只看有效秩升高或余弦降低。

SwanLab：https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-raw-relations-20261030

远端 `/root/owac/repo/outputs/j5-raw-relations-20261030/`，冻结执行源码 `source/src`。J3参照路径写入配置，完整原文件保留；续训内部完整恢复，阶段初始化不伪称保留J4 Adam动量。无失败自动重启、无新定时任务或任意数值验收门槛。
