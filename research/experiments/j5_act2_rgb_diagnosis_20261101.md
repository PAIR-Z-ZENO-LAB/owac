# J5-ACT2 1000步：配对RGB闭环与动作诊断

## 结论

本次正常光开发集，RAW **0/20**，官方LIBERO RGB **20/20**。四个任务各5个初态，逐场核对物理初态、光照参数、环境种子和扩散噪声全部一致。当前主要问题已经发生在正常光，不能只归因于弱光。动作误差、指令干预和部署复现共同指向：当前RAW策略的任务条件控制能力仍弱，而不是完全忽略图像或已发现明确的部署接线错误。

这是1000步阶段诊断，不是6000步最终结论；也不是仅替换视觉编码器的公平消融。RAW继承通用pi05-base及之前的RAW适配，RGB使用已经针对LIBERO微调的官方权重。不能据此断言J5视觉编码器本身无效。

## 身份、配置和范围

- 日期：2026-11-01，Asia/Shanghai；负责人：lzh/Codex；证据层：LIBERO原生仿真闭环及离线动作诊断；诊断完成。
- 基础代码：`adcea5b7581fc332ce2cc4cd1761105c5e803ec5`，分支`lzh/j5-language-adaptation`。工作区原有未提交修改保留；本次新增`j5_policy_diagnosis.py`及课程恢复修复，diff保存在本地诊断输出中。
- 服务器工作目录：`/root/owac/repo`；Python：`/root/owac/runtime-j5-pi05/bin/python`；官方openpi固定源码：`/root/owac/openpi-official-15a9616`。
- 远端根目录R：`outputs/j5-pi05-adapt-20261101`；诊断目录D：`R/diagnosis-1000-rgb`。完整解析配置：`D/config.json`；RAW检查点：`R/normal/step-001000.pt`；部署导出：`D/policy-final-ema.pt`。
- 原RAW闭环：`R/development-001000/closed-loop/raw`；新增RGB闭环：`D/closed-loop/rgb`。LIBERO四套件task0，各初态5–9，Normal；10步去噪、动作块10、每次执行5步；单场上限分别220/280/300/520控制步，沿用原开发评估配置。
- 数据清单：`outputs/raw-distillation-all-bench-20261022/inventory.json`，沿用LIBERO轨迹级拆分，无本轮重划分。训练和RAW诊断不使用RGB监督；RGB只用于参考策略评估与展示。
- 两张RTX5090，容器内存116GiB，推理BF16。诊断期间原训练在1047步完整保存/native0暂停，结束后恢复；未新增训练配方。诊断总GPU时间、峰值资源未单独计量，不宣称资源效率收益。
- 原始帧为RawVLA-Bench模拟数据，不是真机传感器实验；摄像机视角、旋转和动作归一化沿用现有真实执行路径。

## 配对闭环

| 套件task0 | RAW | 官方RGB |
| --- | ---: | ---: |
| libero_spatial：黑碗放到盘子 | 0/5 | 5/5 |
| libero_object：alphabet soup放入篮子 | 0/5 | 5/5 |
| libero_goal：打开中间抽屉 | 0/5 | 5/5 |
| libero_10：两件物体放入篮子 | 0/5 | 5/5 |
| 合计 | **0/20** | **20/20** |

独立计数单位为一次完整回合，成功取环境原生判定。20对条件核对见`D/paired-conditions.json`。RGB新增评估`closed-loop/exit.json`为native_exit=0；没有将旧初态0–4的历史RGB结果混入本次比较。20场是开发检查，不能解释为全LIBERO成功率。

初态5的最终帧显示：RAW黑碗尚未放上盘子、目标罐尚未入篮、中间抽屉未拉开、双物体搬运未完成；RGB对应目标完成。两列取各自回合结束帧，终止步数不同，不能当作同一时刻的运动轨迹。完整视频与动作数组均保留。

## 同源验证数据上的动作误差

32条Normal验证轨迹，每条8个均匀时刻，256 anchors、2272有效动作步。比较7维归一化采样动作，排除轨迹尾部补齐；同样10步去噪、固定`70000+anchor_index`生成的噪声。RGB读取同源默认光照RGB，不是RAW Normal的完全相同光度输入。官方RGB权重的训练数据暴露范围没有在本轮核实；这些数值是现成策略能力参照，不能作为公平的留出泛化比较。

| 指标 | RAW 1000 | 官方RGB |
| --- | ---: | ---: |
| action MSE | 0.178478 | 0.018424 |
| 执行前5步action MSE | 0.181805 | 0.014739 |
| action MAE | 0.270585 | 0.056616 |
| 平移指令RMSE | 0.354143 | 0.085545 |
| 旋转指令RMSE | 0.064889 | 0.020734 |
| 夹爪符号错误率 | 10.04% | 1.54% |

平移/旋转是控制器命令空间数值，不是米/弧度的实际位姿误差。RAW全块MSE约为RGB的9.69倍，执行前5步差距更大，不能用“后续重规划会修正”解释掉。RAW从477步0.19956改善到1000步0.17848，但尚未转化为闭环成功。

在同一验证集合，无图像常数动作参考：训练集全局均值MSE=0.283041，按任务训练均值=0.261223，物理零动作=0.286772。RAW优于这些参考，因此不是完全没有学习；离现成RGB控制水平仍远。

## 输入干预与训练/部署一致性

每场初始观测固定噪声：分别替换为另一个套件的指令或画面，其他输入不变。表中为前5步×7维归一化动作的RMSE，再对20场平均。不同套件画面/指令是分布外配对，仅测输入依赖，不能直接证明正确的语义理解。

| 干预 | RAW | RGB |
| --- | ---: | ---: |
| 换指令 | 0.04710 | 0.14479 |
| 换画面 | 0.20825 | 0.19969 |
| 原输入复现原始计划max abs | 0 | 0 |

RAW明显响应画面，指令改变引起的动作变化约为RGB的三分之一。支持“任务条件约束偏弱”这一判断；尚未通过修复消融证明它是唯一原因，也不能由相似的换图幅度推断视觉理解同样好。

四套件初态5另测训练验证用joint forward与部署cached forward：RAW归一化动作RMSE分别0.002960、0.002991、0.003100、0.004811。这比策略间误差小，当前未见足以解释0/20的明显推理路径分裂；有限初态检查不等于排除所有实现问题。旧expert检查点的attention/token测量没有冒充本次1000步的测量。

## 数据和模型设置中的具体问题

1. **初始化并不对等。** RAW的语言/动作主体来自通用pi05-base，继承前一轮RAW LoRA；RGB来自官方LIBERO专门微调策略。当前RAW同时承担新视觉接口适配和LIBERO控制能力学习。不能把这两件事的综合差距全部归咎于RAW视觉。
2. **当前Normal阶段独立示范偏少。** 全部LIBERO训练1421条，但筛Normal后278条。当前四任务分别12、4、5、7条Normal示范。先前适配确实接触过五光照1421条，不能说整个训练史只有278条。不同光照是不同轨迹，不是同一示范的五份副本。
3. **当前验证覆盖不足。** Normal验证32条只覆盖23/40任务；当前双物体任务没有Normal验证轨迹。全局MSE下降可能偏向已覆盖任务，不能代表所有任务控制恢复。
4. **适配仍未完成，但不是旧的“语言完全冻结”。** 当前已有残差非线性桥接、语言LoRA rank16和动作专家LoRA rank8，包括新加Q/K适配；约2636万参数训练，RAW视觉及基础大部分权重冻结。输入干预与动作误差说明现有适配尚不足，不能简单重复上一版单线性桥接的解释。

现有数据没有显示损失爆炸是主要原因，学习率自适应也无法补上任务数据、初始化和模态对应关系。是否需要增加可训练容量，要在控制初始化和数据后判断。

## 下一项建议（本轮未启动）

先做**相同LIBERO语言/动作后端初始化**的受控RAW实验：保留J5视觉能力，RAW替换视觉输入路径，继承参考策略已学到的任务控制；训练仍只使用RAW及动作标签，不使用RGB图像监督或DINO蒸馏。必须明确该控制实验继承了RGB训练的策略后端，不能称为整套策略从RAW零起点学习。

这样才能区分“接口没有把已有控制能力唤起”和“需要重新学习任务控制”。先验证接近、抓取/拉动及放置是否恢复，再讨论扩大解冻范围；同时补齐按任务的验证覆盖，保留执行前5步误差、平移/旋转、夹爪和闭环指标。现有训练的最终结果仍待3000/6000步原计划评估，不因1000步诊断提前宣称最终失败。

## 命令、产物和恢复

所有命令在服务器`/root/owac/repo`执行，`PYTHONPATH`包含仓库src、固定openpi/src与openpi-client/src，`JAX_PLATFORMS=cpu`。

```bash
/root/owac/runtime-j5-pi05/bin/python -m owac.runtime.j5_action_finish \
  --config outputs/j5-pi05-adapt-20261101/diagnosis-1000-rgb/config.json \
  --output outputs/j5-pi05-adapt-20261101/diagnosis-1000-rgb \
  --checkpoint outputs/j5-pi05-adapt-20261101/normal/step-001000.pt
```

- 干预入口：`python -m owac.evaluation.j5_policy_diagnosis`，必填config/checkpoint/episodes/prompts/output/role。RAW本地加载、cuda:1；RGB使用`--port 9015`复用同一官方基线服务。提示词取原生benchmark，保存于`D/prompts.json`。结果为`D/{raw,rgb}-probe/{metrics.json,actions.npz}`。
- 离线RGB验证执行脚本：本地`outputs/j5-act2-diagnosis-20261101/rgb-dataset-probe.py`，结果`D/rgb-dataset-metrics.json`、`rgb-dataset-actions.npz`；数据清点和常数动作参考：`D/dataset-diagnosis.json`。
- 本地证据根：`outputs/j5-act2-diagnosis-20261101`，包含20场原始动作结果、四套件初态5双方视频、最终帧拼图和干预结果。模型检查点保留服务器，不为诊断重复下载。
- 原训练1047步完整恢复状态保留在`R/attempts/rgb-diagnosis-1047`。课程控制器修复为读取已有`development-results.json`并跳过已完成1000步评估；新源码快照`R/source-diagnosis-resume`。从`normal/latest.pt`恢复，已实测更新1050，下一节点3000；优化配方未改，OOM计数0。1050步主存82.69GiB中匿名8.20GiB、文件缓存73.10GiB，GPU峰值分配约27.20GiB。
- RGB基线及离线动作指标已上传[OWAC对应运行](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-act2-dev1000-rgb-20261101)，动作指标进入统一“02 动作验证”。本次未重启其他历史训练。

工程检查：新增诊断入口与课程恢复代码通过Ruff、格式与编译检查；动作指标及监测相关测试4项通过。`make check`仍被原有其他文件的213项lint错误阻断，本轮未修改它们。小型配置/诊断产物校验值见本地`diagnosis-manifest.json`，原执行脚本与工作区diff一并保存。
