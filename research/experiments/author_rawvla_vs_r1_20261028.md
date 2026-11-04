# 原作者RawVLA π0.5配方与R1对照

本对照依据服务器固定author-main/configs/libero/pi05.yaml→configs/base/rawvla_libero.yaml→rawvla.yaml及其固定StarVLA训练/数据实现。是公开默认配方，不能冒充论文每次实验的最终运行收据。此处不使用本项目155步教师候选或init0教师选型代替作者训练。

双方都不是全模型随机初始化：作者以已适配LIBERO的π0.5冻结策略为底座训练Neural ISP；R1继承同类视觉与后端，重置RAW stem/gate，冻结语言/动作后端、训练整个视觉头。

|项目|作者RAW π0.5公开配方|本次R1|
|---|---|---|
|数据|RawVLA-Bench LIBERO RAW训练轨迹；发布共1771条，README命令覆盖cache_root|同一共享LIBERO源库，按源1421/177/173拆分，五照明|
|取样|每源32 samples索引组织，随机current，virtual_length100000|固定stride5端点，从25开始，按epoch打乱|
|历史|6 stride5端点，每端点6相邻帧，起始不足复制首帧|6 stride5反传端点＋此前因果prefix状态；prefix无梯度|
|图像标签|默认LIBERO配置无rgb_cache_root；亮度目标0.48与色度输出包络先验|无图像监督|
|动作标签|真实7维动作，horizon10；越过末尾重复末动作并计入loss|真实7维，horizon10；越过末尾mask不计loss|
|动作归一化|pi05_quantile：max(range,1e-6)，clip[-1,1]|相同|
|状态|quantile+clip，discrete_state_input=false，实际不消费|相同|
|padding|动作补0至32维|相同|
|动作loss|Gaussian/Beta(1.5,1) native flow velocity MSE，32维均值|同一算法，带有效未来mask|
|总loss|0.15 action +0.01 exposure +0.001 chroma prior|1.0 action，无辅助损失|
|训练参数|仅raw_frontend；vlm/action_head冻结eval|RAW stem/gate/SigLIP/projector，428579952参数；语言/动作后端冻结|
|优化器|AdamW lr1e-4，betas(.9,.95)，wd1e-8|AdamW lr1e-5，betas(.9,.999)，wd.01|
|调度|warmup100，cosine至1e-6|warmup100，恒定1e-5|
|batch|每设备1，累积8；global依实际进程数|global16|
|更新|公开默认2000|目标100000，现暂停29071|
|grad clip|1.0|1.0|

结论：本地R1的动作clip没有偏离作者RAW配方；之前官方RGB不clip是另一层对照。最大的机制差异是作者保留整个RGB视觉/动作系统，仅让ISP输出适合它的RGB，并有亮度/色度约束；R1替换RAW入口、更新整个视觉表示，只有动作信号。作者成功不能直接归因于相同纯动作loss可训练任意RAW视觉头。

## 加入π0.5原论文（2025-04-22 v1）

这是原论文的预训练/后训练，不是开源pi05_libero微调配方。

|项目|π0.5原论文|原作者RAW-VLA π0.5|本次R1|
|---|---|---|---|
|初始化|PaliGemma初始化VLM；后训练加入随机动作专家|已有π0.5 LIBERO＋新ISP|已有视觉/后端＋新stem/gate|
|数据|跨机器人示范、语义子任务、网页多模态；后训练加入口头指导|LIBERO RAW数据|同源五照明RAW，1421/177/173|
|输入|RGB、任务文本、离散状态token|RAW历史、任务文本；LIBERO不使用状态|同左，但RAW直接到tokens|
|监督|FAST动作token、文本/语义标签；后训练加入连续动作|真实动作＋亮度/色度先验|真实动作|
|loss|预训练CE；后训练CE＋10×flow matching|0.15flow＋0.01曝光＋0.001色度|纯flow|
|训练范围|VLM学习离散任务，后训练联合训练并加入动作专家|仅ISP|整个视觉头，后端冻结|
|动作窗口|50步|10步|10步|
|归一化|各数据集q01/q99；低维补零|pi05_quantile＋clip|同左|
|显式clip|正文未明确区分分位数映射和越界硬裁剪，不能据此确定|有|有|
|更新|预训练280k＋后训练80k|默认2000|目标100000|
|lr/batch/优化器|本次核对正文及附录未取得完整配方，不用LIBERO公开配置代填|见上表|见上表|

来源：https://arxiv.org/html/2504.16054v1 ，第IV-B/IV-C/IV-D及附录A-E。论文完整训练包含离散交叉熵监督，不能简化成“纯flow loss训练整个视觉编码器”。CE+10flow的系数取决于其归约定义，不能直接与R1系数1比较有效梯度大小。
