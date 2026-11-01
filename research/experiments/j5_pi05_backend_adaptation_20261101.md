# J5-ACT2：保留视觉能力，开放新模态后端适配

日期：2026-11-01。分支：`lzh/j5-language-adaptation`。

用户授权实现、服务器训练与闭环验证。目标是检验“线性桥接和过度冻结后端未完成RAW适配”的假设；当前工程运行不能证明动作收益。

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
- [正常光SwanLab](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-act2-normal-20261101)
- [工程SwanLab](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-act2-engineering-20261101)
