# J5 分阶段动作策略闭环（2026-11-01）

用户要求：训练结束后验证闭环成功率。本轮只评估已选定检查点，不继续训练或按测试表现调权重。

## 训练与选型

桥接、选择性动作专家 LoRA 两阶段均完成6000更新，native_exit=0。固定177源验证动作采样MAE7：桥接best6000=0.385287255，专家best5000=0.339392245，因此按既定验证规则选择expert5000。两个阶段记录的host峰值分别73.40/73.04GiB，OOM均0。没有触发学习率停滞回退（lr_factor均1），不能声称验证了回退的长期效果。

## 评估入口修复

原自动评估在SwanLab初始化阶段报 `Run id should not be provided when resume=never`，发生在启动策略/仿真前，原尝试闭环场次0。修复为显式 `resume="allow"`；原训练源码快照、pipeline.log、pipeline.exit.json与原finish文件保留。新评估使用独立 `evaluation-fix/src` 快照，训练文件/检查点未修改。

本次导出的policy元数据实查step5000、ranks=[0,8]，RAW编码器/语言/QK/动作时间接口冻结，训练参数仍为连接层与专家V/O/MLP LoRA。EMA decay=0，导出字段虽沿用ema名称，数值对应online。

代码ruff通过；make check仍阻断于此前213项lint错误。实际SwanLab启动和真实仿真回合已运行，不能用这些工程检查替代成功率。

## 协议与产物

- LIBERO spatial/object/goal/10，每套10任务，五照明，每任务5个固定初态；每角色1000场。RAW与RGB共享物理初态、照明参数、动作噪声种子；每5步重规划，10步flow采样，原生env.check_success判定。
- RGB参考是官方pi05_libero；RAW后端从pi05-base适配。它是独立参考，不是等训练预算的纯模态因果对照。
- 原始视频、initial.npz、actions.npz、episodes.jsonl均保存，不删除失败、不以5步smoke计成功率。
- 服务器根 `/root/owac/repo/outputs/j5-pi05-bridge-20261031/expert/closed-loop`；新进程回执 `/root/owac/repo/outputs/j5-pi05-bridge-20261031/evaluation-fix/exit.json`。
- SwanLab：https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-bridge-eval-20261031 。
- 初筛正常光第一场视频已复制到本地 `outputs/j5-action-closed-loop-20261101/`，并排视频只用于直观查看，不替代native结果；RGB成功后定格补齐RAW回合时长，不用于时延比较。

## 五照明单任务初筛

首个预定任务libero_spatial/task0（将盘子和小碗之间的黑碗放到盘子上），每照明5固定初态：

| 条件 | RAW | 官方RGB |
| --- | --- | --- |
| Normal | 0/5 | 5/5 |
| Low | 0/5 | 5/5 |
| ExtremeLow | 0/5 | 0/5 |
| Over | 0/5 | 4/5 |
| ExtremeOver | 0/5 | 0/5 |
| 合计 | 0/25（0%） | 14/25（56%） |

全部50场均为原生完整回合，非5步smoke。RAW失败发生于正常光及弱光，不能只归因于极弱光；本任务尚未建立有效控制。RGB极端照明同样困难。该结果只覆盖首个预定任务，不能写成40任务总体成功率，也不能证明RAW模态本身不适合动作。

原始episode快照和逐条记录汇总保存在本地outputs/j5-action-closed-loop-20261101/。全量每角色1000场仍在后台继续，当前无全量最终成功率。
