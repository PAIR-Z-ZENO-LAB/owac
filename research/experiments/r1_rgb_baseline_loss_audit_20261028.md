# R1动作质量与官方RGB基线核对

训练暂停29,071步。本次仅阅读官方公开源码、服务器固定StarVLA源码和执行离线推理；无权重更新。官方main来源核对日期2026-10-28，不等于已取得发布权重的完整历史训练收据。

## 官方RGB基线

官方OpenPI的pi05_libero：pi05=True、action_horizon10、discrete_state_input=False、extra_delta_transform=False；动作32维，真实7维后补零；Gaussian noise、Beta(1.5,1)时间，速度MSE按所有维度均值。默认freeze_filter=Nothing，联合微调；从pi05_base初始化；batch256、lr5e-5、warmup10000、EMA0.999、30000更新。以上训练步数只是对照，R1原授权目标仍100000。

当前R1：原LIBERO模型初始化视觉主干/投影，重置stem/gate，仅视觉头更新；batch16、lr1e-5、warmup100、无EMA，真实末尾有效时间掩码。原生flow定义一致，但不是同一训练设置。

重要修正：当前R1 quantile+clip与固定StarVLA rawvla_npz_datasets分支一致，但官方OpenPI Normalize._normalize_quantile使用(range+1e-6)且不clip。状态不消费与官方LIBERO配置一致；动作clip则是实际训练目标差异。此前归一化排查只能排除已测的符号/主要尺度错误和状态clip作用，不能说与官方完全一致。

## 当前检查点实测

4个真实训练样本（其中3个task6），每样本同一noise、t=0.05/0.25/0.5/0.75/0.95，分解flow速度MSE；另从同一noise运行10步采样并与示范比较。诊断为定向小样本，不能代替验证集总体统计或成功率。

在t=0.5：

|样本|全部32维flow MSE|真实7维|补零25维|
|---|---:|---:|---:|
|stove/demo10 e210|0.017274|0.078789|0.0000497|
|task6/demo1 e25|0.008101|0.036807|0.0000637|
|task6/demo10 e25|0.047709|0.217853|0.0000693|
|task6/demo12 e65|0.095483|0.435971|0.0001457|

总MSE被补零维度稀释，真实动作MSE约为其4.5倍；但padding误差贡献很小，不能据此声称模型只学补零或padding梯度支配训练。官方同样32维均值，不能把它判为算法bug。

10步采样动作的归一化误差例子：task6/demo10旋转第1维MSE0.50178；task6/demo12夹爪MSE1.58943（RMSE1.2607）。低总flow loss不是动作已准确的证据。噪声条件单点速度误差与沿生成轨迹累积后的动作误差是不同指标；本次不能从4例建立可靠相关性。

## 迁移建议

先保持原生flow目标与32维定义；将真实7维、xyz/rotation/gripper、padding25与flow时间分组单独记录。验证时固定观测和noise进行10步动作生成，记录前5执行步及全部10步的归一化/物理动作误差、夹爪符号不一致率。比较时保留示范多解性，不以离线MSE取代闭环。

若要对齐官方RGB预处理，应将动作quantile改成官方不clip公式并用明确的新实验配置记录；此前训练不可冒充已用该协议。先在同一示范上比较官方RGB原模型、历史蒸馏学生和R1的采样动作误差，确定是视觉替换后的问题还是共用数据/动作接口问题。官方全模型微调不能无条件搬入用户限定的冻结后端实验，学习率和batch也不能直接照抄。当前尚未改训练配置或恢复训练。

## 来源与证据

- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py
- 本地诊断：outputs/r1-raw-action-20261027/gradient-diagnostic/flow-quality-probe.json、flow_quality_probe.py
- 服务器固定源码：starvla-pinned-source/examples/simBenchmarks/LIBERO/train_files/openpi/pi05_libero_8gpu.yaml、starVLA/dataloader/rawvla_npz_datasets.py、starVLA/model/framework/VLM4A/PI0.py

## 不裁剪动作目标验证

固定29,071检查点、同4个观测、相同noise和5个flow时间，将目标切换为官方quantile无clip（同时采用range+1e-6）。只改变验证目标，不执行optimizer.step，不会改变部署采样策略。逐样本结果见outputs/r1-raw-action-20261027/gradient-diagnostic/unclipped-comparison.json。该对照只能量化目标处理对误差统计的影响，不能回答重新训练后的闭环收益。
