# J2S：完整 RAW 稠密语义编码器（2026-10-29）

## 决策与目标

用户明确目标是完整RAW encoder，效果参照DINOv3的稠密语义相似度图，并授权根据数据/算力自主选择训练路线。因此选择训练期冻结DINOv3-L/16配对RGB语义监督，部署期只用RAW encoder。1421条LIBERO训练源、双RTX5090适合已有语义迁移；不把有限操作场景自监督描述成重新获得开放世界语义的保证。

这是继承J1完整视觉权重后的新目标/架构训练，重建优化器与scheduler，非旧J1精确续训。旧J1/82000、R1/29071及蒸馏最终权重全部保留。旧J1已停止，新训练不是继续旧遮挡预测目标。用户示例中的水果市场不是本数据域，不能用LIBERO指标宣称已达到开放域DINOv3能力。

## 完整编码器

部署：六帧因果pseudo-RAW → 双响应stem → 局部内容门控时序融合 → 27层SigLIP主干 → 9/18/27层可学习融合 → 1024维稠密token。全部427404532个参数可训练，不是只训练adapter。

- 测量输入：当前数据三通道uint8 pseudo-RAW/255，不是原生Bayer。保留线性分支；添加 `log1p(255*x)/log(256)` 特征分支，3×3卷积在64维stem特征处合并，不生成RGB。新log卷积零初始化，初始精确保留旧stem行为；255来自现有uint8编码范围，不冒充物理噪声参数/传感器标定。
- 时序融合：当前token、历史token及两者差的通道LayerNorm特征输入局部线性门控，叠加继承的通道gate。每窗口reset、只看当前及过去六帧；局部门控零初始化等于旧gate。它可学习抑制局部冲突，但没有显式光流或配准，不能声称已解决运动模糊。
- 空间/语义：保留完整27层；9/18/27层经共享post-LN后由softmax权重融合，初始logits=[2,0,0]。新1024维投影是编码器输出头，全部层收到语义梯度。早层不是最终固定答案；融合权重可学习并上报SwanLab。
- 输出：256×1024非图像tokens。导出的encoder.pt不含教师/动作模型/ISP/教师缓存。架构构建仅依赖已固定的视觉实现代码，不读取公开政策权重或DINO权重；纯encoder恢复可独立进行。

## 监督与数据

冻结DINOv3-L/16是训练目标，不是部署组件。RGB metadata明确是相同时刻、相机的default-light配对渲染；RAW是五照明扰动。存在教师光照信息优势，可能对极低光不可观测内容形成推断，后续必须检查误匹配而非只看亮丽热图。

保持LIBERO train1421/validation177/test173、两相机，训练41377窗口，6连续帧、endpoint stride5。没有混入RoboTwin，没有读test。随机196/224/256大小裁剪与水平翻转对RAW全部六帧/RGB当前帧采用相同几何。RAW输入缩放224、RGB输入256；学生patch14、教师patch16均输出16×16，覆盖相同视野。

损失：逐位置 `1-cos(student, teacher)` ＋ 2×归一化patch Gram MSE ＋ 1×池化特征cosine loss。教师固定、无EMA漂移、无RGB像素重建、无动作损失。Gram权重2参考DINOv3，整体组合与本地适配超参属于工程选择，不是论文原配方复现。新目标直接约束最终输出的语义坐标及关系，不能仅靠平移匹配宣称达到目标。

## 优化、保存与监控

目标100000优化器更新，global32（初始2rank×micro2×accum8；性能修复后2rank×micro8×accum2），原FP32参数/Adam＋BF16前向、checkpoint重算。继承参数峰值lr1e-5，新log分支/内容gate/融合/语义头lr1e-4，warmup2000，cosine尾值10%；AdamW(.9,.95)，wd.04，gradclip1。

每500步在完整177验证源、531窗口/1062视图上记录固定语义loss，分五照明。step1、每1000及最终/用户暂停保存完整optimizer/scheduler/双rank RNG/采样cursor，先临时文件完整写入/fsync再原子替换。SIGUSR1在更新边界保存暂停。不自动设置新的loss停止阈值，不通过短跑偷偷替代100k。

380GiB上限保持；用户授权清理340.914GiB教师缓存后，启动一次性按较宽口径盘点现有产物66.537854GiB（包含原计数外资产），余量充分容纳本run独立完整检查点及原子写入临时副本。未删任何关键权重/视频。

- 配置：`configs/experiments/j2s_raw_semantic_100k.json`；执行配置在远端run/config.json，含实际storage baseline。
- 模型/训练：`src/owac/training/raw_semantic.py`、`raw_semantic_train.py`。
- 远端：`/root/owac/repo/outputs/j2s-raw-semantic-20261029/`。
- SwanLab：https://swanlab.cn/@Jeong/OWAC/runs/owac-j2s-raw-semantic-100k-20261029 。从启动同步训练/验证、实际更新、融合权重、GPU与磁盘。
- launch.py记录原生退出；100k正常完成后自动执行完整验证源 `raw_dense_benchmark --dino-size 256`，单独保留evaluation.exit.json。训练成功与评估成功分开。

## 真实检查与已知边界

13项模型行为测试及13项研究台测试，共26项通过。真实预检曾因直接SiglipVisionConfig创建了原模型没有的pooling head而strict加载失败；已改为与原PaliGemma相同vision config，无宽松加载、无丢弃不匹配参数。修复后真实2窗口/双相机反向通过：学生/教师均[4,256,1024]，427404532参数全部有梯度，教师无梯度；log分支梯度范数0.836209、内容gate0.064264、最后block0.086470；新增优化0、峰值allocated约5.10GiB。随后独立双卡正式训练已实际产生step1完整检查点，并继续推进。

全库make check仍被已有未提交诊断脚本的213项lint错误阻断。本任务相关文件单独ruff/format、26测试通过，不能声称全库通过。

全验证源旧表征对照见 `j2_raw_encoder_20261029.md`。新增评估必须同时记录语义loss、跨源任务检索代理、空间对应、固定查询点热图，最终还需要物体/部件真值才能确认同类语义质量。训练中和step1检查点不等于已达到DINOv3。

## 独立恢复检查

step1完整状态在独立进程CPU恢复：447/447参数张量对应447份Adam状态，scheduler.last_epoch=1，采样epoch0/cursor1，双rank RNG均在。将初始化权重、公开政策权重、DINO路径设为不可用路径后仍能构建与strict重载encoder；未加载教师或历史初始化，新增优化0。五照明固定查询特征已从导出encoder独立提取，训练起点图保留于outputs/j2s-query-start-20261029，不能称作训练后结果。

执行配置继承了J1部分说明字段（如student.temporal/projector），它们不参与J2S构建；实际结构以上述模型源码和本记录为准。导出不包含原2048维政策projector。

## 首次完整更新后验证：500步

531窗口全部计入，固定DINOv3目标。初始化→500步：dense cosine loss 1.006392→0.357349（平均余弦-0.006392→0.642651）；Gram MSE 0.099527→0.020169；global cosine loss 1.019595→0.096672；总loss2.225041→0.494360。同期归一化token空间std0.021823→0.020339。只证明目标对齐改善，未证明对象/部件识别、跨实例检索、闭环或与DINOv3能力等同。训练继续100k，未调整既定更新数。

最终评估已串接固定五照明查询affinities、原始NPZ和同色标PNG生成；训练起点同一组图保留供比较。

## 性能修复

保持模型/损失/global32/100k目标，实测吞吐提升约2.44倍，GPU采样均值约79%/80%；原内存增长主要是文件页缓存，另修复mmap恢复中CPU计数器持有checkpoint映射的问题。完整暂停于1162、1709并原状态恢复。见 [性能与内存记录](j2s_performance_20261029.md)。

## 用户最终暂停（2026-10-29）

64366步在更新边界完整保存，status=paused、checkpoint complete/encoder_complete=true、native_exit=0，GPU进程已退出。远端outputs/j2s-raw-semantic-20261029/latest.pt与encoder.pt保留，另以硬链接归档至attempts/user-pause-64366；原子替换保存不会覆盖归档inode。包含Adam/scheduler/双rank RNG/采样游标；历史日志、验证、评估与33085/48000/50000快照保留。本地退出收据outputs/j2s-user-pause-64366。未经新授权不恢复。
