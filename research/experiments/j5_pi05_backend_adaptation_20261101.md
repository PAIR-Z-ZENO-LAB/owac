# J5-ACT2：保留视觉能力，开放新模态后端适配

日期：2026-11-01。分支：`lzh/j5-language-adaptation`。

用户授权实现、服务器训练与闭环验证。目标是检验“线性桥接和过度冻结后端未完成RAW适配”的假设；当前工程运行不能证明动作收益。

## 监测与动作验证更新（2026-11-01）

按用户要求，训练在477步完整保存/native0退出后恢复，未重置权重、Adam或数据游标。迁移前的latest/best、配置、日志保存在`attempts/monitor-v2-477/`；旧执行源码保留，新增`source-monitor-v2/`执行快照。

当前看板：[OWAC-Action-Training](https://swanlab.cn/@Jeong/OWAC-Action-Training/runs/owac-j5-act2-normal-mse-20261101)。仅四组：01训练目标、02动作验证、03优化器、04运行资源。原训练曲线按实际step回填；去掉重复的基础LR（与桥接LR相同），停用SDK自动资源曲线，保留显式记录的内存/吞吐。闭环结果写入独立`OWAC-Action-Evaluation`，不混入训练面板。

SwanLab图表按项目共享，已存在图表不能靠再次`define_metric`移动（[官方定义](https://docs.swanlab.cn/api/py-define_metric.html)）。本次使用新项目隔离历史视觉/工程图表；**旧OWAC页面的历史分组未删除**。浏览器连接不可用，因此没有声称已通过网页清空旧分组。API已读回新项目25个自定义指标，仅4个命名前缀；自动SYSTEM标量为空，MSE477步数值与本地一致。

主指标`validation/action_mse`为实际10步Euler采样动作与示范的**7维归一化动作MSE**；不是flow loss，排除25个潜空间补零维度和轨迹末尾重复补齐动作。按照有效动作步加权聚合。验证扩展为每条轨迹均匀8个固定时刻、固定噪声；这是覆盖全部验证轨迹的固定帧子集，不是逐帧全量验证。当前Normal32条轨迹、256时刻、2272有效动作步。训练集与测试集未混入。

辅助指标：动作MAE；实际每轮执行前5步的MSE；反归一化后平移/旋转三维RMSE（原始控制命令单位，不宣称米或弧度）；夹爪命令正负判断错误率。进入五光照后在同一验证组内补充每光照MSE；单光照阶段不重复创建Normal与全局两张相同曲线。样本数与有效动作数同时记录。

477步基线：MSE **0.1995621**、MAE **0.2923230**、前5步MSE **0.1997837**、平移RMSE **0.3977135**、旋转RMSE **0.0690166**、夹爪错误率 **9.4630%**。best与学习率停滞判断切换为MSE，旧MAE选择分数不沿用，新基线从477建立，原best保留在迁移归档。闭环成功率继续独立检验实际控制能力。

针对MSE维度/尾部屏蔽、有效动作加权、夹爪反归一化及数据读取的5项测试通过；定向ruff通过。make check仍为既有213项lint错误。服务器真实验证后已继续更新，OOM0。500步MSE=0.2124411、MAE=0.2992522，未刷新477步best；选择状态实测为metric=validation/action_mse、best_step=477、bad_evals=1。

## 初始化与训练范围

继承 `/root/owac/repo/outputs/j5-pi05-bridge-20261031/expert/policy-final-ema.pt`（expert第5000步）。保留J5 12818 RAW编码器、原桥接和既有专家V/O/MLP LoRA权重。RAW编码器保持FP32/eval并冻结，官方后端基础权重与动作/时间接口冻结。

- 桥接：`old_linear(x) + W2 GELU(W1 LN(x))`，384→1024→2048；W2初始化为零，保留原映射。
- 语言18层：Q/K/V/O及gate/up/down全部rank16 LoRA。
- 专家18层：原rank8 V/O/MLP适配器直接保留，增加rank8 Q/K。
- 可训练参数26,359,552；新结构使用新AdamW状态。原始官方32维flow MSE不变，没有RGB/教师/attention/PCA辅助目标。

| 参数组 | 正常光峰值LR | 五光照峰值LR |
|---|---:|---:|
| 桥接（原线性及新增残差） | 3e-5 | 1e-5 |
| 语言LoRA | 5e-6 | 2e-6 |
| 专家V/O/MLP LoRA | 5e-6 | 2e-6 |
| 专家Q/K LoRA | 1e-6 | 5e-7 |

两阶段各6000更新，300步warmup后cosine；沿用既有验证降LR及完整model/Adam回滚。五光照阶段从开发闭环选出的正常光权重开始，以较低LR、新阶段Adam训练；不是恢复旧失败任务。EMA decay=0，导出的`ema`即在线权重。

## 数据、闭环与资源

保留原LIBERO source split。第一阶段实际只取Normal源及Normal验证源；第二阶段恢复五光照。训练只用RAW、指令与动作。配置中的1421/177/173是过滤光照前的原分区计数。

正常光在1000、3000、6000步暂停计算并执行完整开发闭环：四个suite各task0、每任务initial_index5..9，共20回合/检查点，Normal、RAW。与既有正式对照的初态0..4分开；这不是未见任务泛化实验。按成功回合数选择，平局保留较早检查点。如果三个候选全部失败，记录`normal-control-not-established`及全部失败轨迹，进入诊断，不自动把失败设置扩展到五光照。此分支不是能力验收通过。

五光照训练结束后复用既有完整RAW/RGB对照：40任务×5光照×5初态，各1000回合。RGB只用于参考评估。参考模型本身经过LIBERO微调，因此不能把差距解释为纯模态因果效应。

双RTX5090 DDP、global256。先实测micro64，再验证micro128减少累积开销；每rank两个worker、prefetch1，逐帧pread并释放文件页，有限reader/label缓存。完整分段恢复保留Adam、RNG、数据游标和选择状态。正式起点只清零工程短测的bad-eval计数，避免把相隔2步的工程验证当作500步间隔的训练停滞；best及权重/Adam不重置。SwanLab记录各组LR、损失、动作采样误差、内存及独立开发闭环结果。

## 实际证据

- 改动相关8项测试通过，涵盖残差初始化保持旧映射、扩展LoRA保留既有适配器及优化器分组。
- 定向ruff/编译通过。`make check`仍报既有213项lint问题，没有修改无关文件来消除它们。
- 服务器micro64完成3步训练、Normal动作采样；3→5完整恢复完成，两次native exit0。第3步Normal MAE7=0.31717，是新过滤/噪声索引协议下的工程值，不直接与旧五光照汇总比较。
- micro64实测显存峰值16.75GiB/卡，训练时主机72.9–73.2GiB，cgroup OOM/kill=0。加载时观察到约83GiB；主机占用主要是文件缓存，不能把它称作模型驻留内存。
- micro128完成5→7完整恢复，native exit0；峰值27.20GiB/卡，主机74.21GiB，OOM/kill=0。短测首更新7.36秒，尚不能宣称比micro64更快。
- 导出并独立重载第7步模型，LIBERO goal/task6、Normal、初态5完整执行300步，native exit0、成功0/1。保存视频、输入、动作与噪声；不是5步smoke或轨迹回放。这只证明真实执行链路贯通，没有动作收益结论。
- 正式正常光阶段已从第7步恢复并推进至第10步，SwanLab在线；第10步6.62秒/更新，显存27.20GiB/卡、主机80.67GiB（其中文件缓存71.00GiB）、OOM/kill=0。首段目标1000；后续实际进度以SwanLab及服务器状态为准。

旧全量评估为本轮释放GPU而中断，未删结果：RAW完成204/1000、成功0；RGB完成281/1000、成功170。共同完成204对为RAW0/204、RGB128/204。SIGINT触发SwanLab结束后控制器日志调用报错，原生退出1并清理子进程；保留原错误，不伪写成功退出。未完成回合不计分母。

## 位置

- 服务器根目录：`/root/owac/repo/outputs/j5-pi05-adapt-20261101/`；`source/`为执行源码快照。
- `engineering/`：工程3→5→7恢复、checkpoint、原生日志及收据。
- `normal/`、`development-001000/003000/006000/`、`five-light/`：分阶段产物；`curriculum-status.json`与`development-results.json`记录实际进度。
- [旧正常光SwanLab（截至477）](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-act2-normal-20261101)
- [工程SwanLab](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-act2-engineering-20261101)
