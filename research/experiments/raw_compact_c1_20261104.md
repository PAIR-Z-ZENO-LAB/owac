# C1 RAW 紧凑策略训练与多任务闭环

最终状态：80,000更新及全部预定评价已完成，选定20,000步EMA，五照明平均1513/2000＝75.65%；按光照、任务类别及RawVLA官方结果的完整横评见[结果报告](raw_compact_c1_results_20261104.md)。以下按时间保留研究过程与历史状态。

2026-11-04 Asia/Shanghai。用户授权自主研究、重构、成熟模块迁移及服务器训练，不设额外 GPU 资源授权上限。训练仍排除 RGB 教师及外部视觉预训练。

## 版本与协议

分支 `lzh/raw-compact-policy`，起点 `27ca6b1`。旧工作区存在不属于本任务的未提交修改，保持保留。执行快照位于服务器 `outputs/raw-compact-c1-20261104/source/src`，解析配置 `config.json`；本地设计配置为 `configs/experiments/raw_compact_c1.json`。上游 CoRP SHA、许可证与本地改动见模型 `_vendor/provenance.json`。

[完整研究假设及协议](../../docs/plans/RAW_COMPACT_POLICY_2026-11-04.md)。这是充分训练的能力实验：相对旧策略同时改变结构和数据覆盖，不作单因素归因。训练仅原1421源/211114帧，验证177源，测试173源不参与训练与选型；三通道 pseudo-RAW 与真实 Bayer 分开解释。

模型48,850,866参数，可训练48,850,482；384维mask token不参与动作输入因此冻结。J5初始化；其余网络随机初始化；T5文本tokenizer无网络权重。40条语言条件各有不同编码。输入双视角256×256 RAW、9维本体、32文本token，输出12×7动作，每5步重新观察。

环境为服务器双RTX5090 32GiB、Python3.12.3、Torch2.12.1+cu132、Transformers4.57.0，BF16混合精度。CPU环境3项测试通过、完整模型测试因未安装可选transformers跳过；服务器实际环境4项行为测试全部通过，包括有效动作/终端掩码、前缀padding、动作缓存梯度及三种输入依赖。不是机器人成功证据。

## 执行

工作目录 `/root/owac/repo`，运行时 `/root/owac/runtime-j5-pi05/bin/python`；`PYTHONPATH` 指向本轮 `source/src`。

```bash
python -m owac.training.compact_data --config outputs/raw-compact-c1-20261104/raw_compact_c1.json --output outputs/raw-compact-c1-20261104/config.json
python -m owac.runtime.compact_experiment --config outputs/raw-compact-c1-20261104/config.json --output outputs/raw-compact-c1-20261104
```

控制器按配置依次训练并做原生闭环，模型、EMA、Adam、RNG完整续接；不是每段重新开始。每个阶段保留退出码、完整检查点和逐场视频/动作。首轮开发Normal初态5–9，独立最终初态20–29五照明。初始1000步40场仅观察，不与后续200场混合选型。

SwanLab项目OWAC，组 `C-raw-compact`，job_type=`raw-proprio-language-flow`，run `owac-raw-compact-j5-20261104`，名称 `C1｜紧凑动作联合训练｜LIBERO五照明RAW｜J5+本体+任务语言｜全训练集80k｜20261104`。

## 实际进展

数据准备完成，训练控制器已启动。真实更新、吞吐、显存、闭环成功率及失败解释在下方追加；当前不宣称性能改善。

### 1000步与首次多任务闭环

1000真实更新完成并完整保存，训练native0。双卡全局256，稳定更新约0.20秒、约1270样本/秒，峰值allocated9.02GiB/卡；这是训练局部测量，不是全流程时延。1000步EMA全五照明验证MSE0.2785686、执行前5步MSE0.2776437，15042有效动作步；不可与旧Normal验证混作同协议比较。

首个评估入口因未加入OpenPI源码路径而退出，尚未产生episode；原日志保存在`attempts/eval-001000-import-error`。修正服务进程PYTHONPATH后使用同一1000权重执行，无额外训练。40任务各初态5、Normal，完整40场native0：spatial0/10、object0/10、goal1/10、long0/10，合计1/40。唯一成功为goal/task7 turn_on_the_stove，277步。该成绩仅表明早期策略出现一次成功，不证明紧凑方案优于旧方案。

从1000完整模型/EMA/Adam/两rank RNG恢复，已实际越过1680更新，继续5000节点。初步假设仍待检验：1000刚结束warmup且EMA有历史滞后，当前0.278动作误差及39次失败不足以区分训练不足和表示不足。后续200场开发结果与失败视频用于复核，不依据任务分类loss下降推断控制成功。

补充分照明验证记录与闭环SwanLab汇总，不改损失或采样；5000段已加载旧代码，后续段采用补充记录。初始训练源码副本保存在`source-revisions/initial-training`。SwanLab训练已完成首段495条记录上传；[训练运行](https://swanlab.cn/@Jeong/OWAC/runs/owac-raw-compact-j5-20261104)。

本地9项相关测试通过、1项可选环境跳过，服务器完整4项通过；新改模块Ruff通过。全库`make check`仍为213项既有lint错误，没有将其误报为通过。已建立本聊天30分钟研究跟进，正常无变化不重复通知，新闭环或假设改变时更新。

### 5000步：首次完整200场闭环

训练与评估均native0，200场全部完成，没有删除失败、重试episode或缩短原生任务步数。Normal、40任务、每任务初态5–9：

| 套件 | 成功/场数 | 成功率 |
|---|---:|---:|
| LIBERO-Spatial | 39/50 | 78% |
| LIBERO-Object | 42/50 | 84% |
| LIBERO-Goal | 34/50 | 68% |
| LIBERO-10 | 13/50 | 26% |
| 总体 | 128/200 | 64% |

与旧J5+π0.5本体方案最佳5000步重合的20初态，旧6/20、新13/20：新增成功7场、退化0场。两者seed和lighting_parameters逐场一致。旧方案只训练Normal278源，新方案使用五照明1421源，后端及动作预测长度也有变化，因此这是能力比较，不能单独归因于紧凑结构；动作噪声维度不同，不声称噪声张量相同。旧方案没有对应40任务200场总分。

5000步EMA五照明验证：动作MSE0.0740198，前5步MSE0.0737689，夹爪符号错误率0.0433453；验证有效动作15042步。1000步MSE0.2785686到5000步0.0740198的下降与本轮闭环进展同时出现，但检查点仍按同一200场闭环选择。

原始证据在远端`development-005000/`，包括两个分片的200份视频、动作/本体记录和汇总；`eval-005000.exit.json`保存原生命令与native0。本地结果与可重复分析在`outputs/raw-compact-c1-20261104/development-005000/`，`comparison.json`含逐场旧新配对，`task-success.png`展示40任务分布。

#### 假设复核与继续研究

结果支持继续推进RAW表征与紧凑动作头联合训练，尚未达到全面高成功率。空间和物体任务已有较强早期控制能力，长任务是主要短板：task6、8、9各0/5；多物体task0和7各1/5。查看object/task1/init8、object/task3/init8、spatial/task4/init5及long/task0/init5视频，能看到目标接近/部分搬运，但出现接触停滞或后续步骤未完成。少量视频不能确定单一失败根因。

训练帧分布为long82773、goal39429、object50486、spatial38426；每任务均匀采样意味着long的平均每帧呈现量更少。先从5000完整模型/EMA/Adam/RNG继续到10000，已真实推进5120步，不因早期长任务不足立即增加结构。下一节点检验长任务和接触失败是否随训练改善；持续存在的状态歧义再支持因果历史或末端运动学本体对照。完整消融不是继续提升成功率的前置条件。

另对四套件训练轨迹12个时刻核对8维末端本体：最大位置分量差0.000908米、轴角差0.002421弧度、夹爪差0.000125米。接近但非完全相同，C1仍保持已训练的9维关节/夹爪输入；没有把此诊断当成加末端输入的收益证据。细节在`diagnostics/eef-contract/`。

5000评估后的调度改为八个仿真进程共享两个无episode状态的策略服务，每套件按互斥任务分为两组，保留原套件GPU/seed、初态、动作周期与成功判据。当前5000结果仍由原两个环境产生，新并行度将在10000正式评估实测。补充3项任务覆盖/种子保持测试和1项最后子进程非零退出测试，均通过；修正轮询结束时漏查最后子进程退出码的边界。修改模块Ruff通过，`make check`仍为既有213项错误。旧执行器/配置保存在`source-revisions/evaluation-two-clients`。

[5000步闭环SwanLab](https://swanlab.cn/@Jeong/OWAC/runs/owac-raw-compact-j5-20261104-development-005000)。首轮继续既定训练/开发节点，最终按开发结果选择检查点，再做未用于选型的初态20–29五照明2000场；当前64%不能代替该独立评价。

### 定时研究跟进取消

2026-11-04，用户要求终止定时任务并改为自行通过SwanLab监测。已删除本聊天30分钟heartbeat `raw`，未停止服务器训练或阶段评估控制器。核对配置：本轮累计80000次优化器更新、全局batch256；实查6920步，当前阶段目标10000，后续闭环节点20000/40000/60000/80000，随后执行已配置的独立最终评价。各阶段从完整检查点累计续训，不分别重置为80000步。

### 闭环曲线接入训练 run

2026-11-04，按用户要求修复闭环呈现。旧汇总器使用`训练ID-输出目录名`建立独立run，并以评估场次数作为step；因此每个checkpoint只有一个散点，训练run的闭环section为空。现在训练与评估共用`compact_swanlab`入口，保留原训练名称、group、job_type，评估以`resume="must"`续写训练ID，不另建评估run。`success/*`归入“05 闭环评估”，横轴为checkpoint的实际训练步数。新summary保存checkpoint内的step；旧summary仅按原`step-NNNNNN.pt`命名契约补读。

200场开发集沿用`success/overall`及四个`suite/lighting`指标，1000步40场初筛记为`success/screening/*`，独立最终评价记为`success/independent-final/*`。不同协议分开记录，最终选定checkpoint的开发成绩不会被独立最终评价覆盖；单次初筛仍只有一个点。训练启动时补录已完成且尚未关联的summary，上传完成后保存逐评估`swanlab-training-link.json`收据，恢复时避免重复写入。

服务器原控制器调度短暂停顿用于同步与补录，评估子进程正常完成；20k评估已在修改前加载旧代码，其独立run也保留历史。源码备份为`source-revisions/swanlab-training-link-20261104`，更新执行副本`source/src`后恢复同一控制器PID673053。没有重算评估、修改模型/优化器或重置训练。40k段已用新日志入口启动，实查超过20040更新。

通过SwanLab API从原训练run读回完整序列（均为Normal、40任务、每任务初态5–9、每点200场）：

| 训练步数 | 总体 | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|---:|
| 5000 | 128/200（64%） | 39/50 | 42/50 | 34/50 | 13/50 |
| 10000 | 152/200（76%） | 42/50 | 44/50 | 42/50 | 24/50 |
| 20000 | 157/200（78.5%） | 43/50 | 44/50 | 44/50 | 26/50 |

1000步初筛1/40已单独读回。证据为本地与远端`outputs/raw-compact-c1-20261104/cloud-closed-loop-linked.json`、`swanlab-closed-loop-backfill.log`及`swanlab-deployment.json`。这里是既有原生开发评估结果的监测修复，不是新的实验或独立最终评价。浏览器控制入口不可用，验证采用云端完整指标API，没有声称完成网页截图核验。

本地相关测试14通过、1项因缺少可选transformers跳过；服务器真实runtime中7项日志专项通过。修改模块Ruff/格式及语法检查通过。全库`make check`仍被已有213项lint错误阻断，输出保存在`swanlab-make-check.log`。

手动补录入口（仅在训练/评估日志会话之间运行，避免并发修改同一run状态）：

```bash
PYTHONPATH=outputs/raw-compact-c1-20261104/source/src \
  /root/owac/runtime-j5-pi05/bin/python -m owac.runtime.compact_swanlab \
  --config outputs/raw-compact-c1-20261104/config.json \
  --output outputs/raw-compact-c1-20261104
```

实现依据：[SwanLab恢复实验](https://docs.swanlab.cn/guide_cloud/experiment_track/resume-experiment.html)、[指标分组接口](https://docs.swanlab.cn/api/py-define_metric.html)，并核对服务器SDK0.10.3。旧独立run和原始逐场证据不删除，后续查看原[训练run](https://swanlab.cn/@Jeong/OWAC/runs/owac-raw-compact-j5-20261104)的闭环section。

### 合并核对后清理重复云端 run

2026-11-04 12:37 Asia/Shanghai，用户进一步要求“先合并然后清理多余的离散run”，覆盖上段保留重复云端run的安排。本次范围仅C1的`development-001000`、`development-005000`、`development-010000`、`development-020000`四个已完成run。通过API读取各run全部6个标量指标，与原始summary、训练run对应训练步数逐项比对，共24个值完全一致；四个来源均无媒体或系统指标。1000步的40场初筛映射到`success/screening/*`，其余仍为200场开发曲线。

删除前将各run云端元数据、指标、日志、项目列表与合并核对表保存到服务器和本地`outputs/raw-compact-c1-20261104/cleanup-discrete-runs-20261104/`。随后使用SwanLab官方API删除且仅删除上述四个重复run；删除后重新查询项目列表及训练曲线，实验数54→50，四个来源ID均消失，全部24个合并值仍在训练run。训练run仍为RUNNING，本地状态21900步、下一节点40000。当前模型、检查点、逐场视频/动作、原始评估文件及其他历史实验不变。

`merge-verification.json`记录逐指标来源与目标映射，`cleanup-request.json`和`cleanup-receipt.json`记录删除对象与API结果，`inventory-after.json`、`training-curves-after.json`、`verification-after.json`记录删除后复核。此次只清理监测数据的重复展示，没有新训练或评估，也未重跑代码测试。

### 其他历史 run 的同类合并与清理

2026-11-04 12:53 Asia/Shanghai，按用户“其它run也需要相同的处理”的要求，进一步核查项目50个run，将25个闭环/诊断run及3个重复监测run归入8个对应训练run，核验后删除28个来源云端run，项目剩22个。加上此前C1四个run的清理，累计从54减至22。其余训练阶段、不同模型路线和失败尝试保留。没有重新训练或评估。

| 保留的训练 run | 合入来源数 | 闭环与诊断的记录方式 |
|---|---:|---|
| [J5本体感知](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-proprio-3000-20261103) | 4 | 4000/5000/6000步开发验证；3003步工程验证单列 |
| [J5视觉确定性](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-deterministic-3000-20261103) | 3 | 4000步开发验证；half_corrected与corrected_half_holdout分别保留 |
| [J5解冻视觉](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-unfreeze-3000-20261103) | 8 | 4000步开发验证；七项权重干预按变体及初态协议分列 |
| [J5 LIBERO后端对照](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-libero-live-3652-20261102) | 4 | 1000/3000/6000步开发验证；旧训练监测历史核验去重 |
| [J5 ACT2](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-act2-normal-unified-20261101) | 6 | 1000/3000/6000步RAW开发验证；官方RGB基线独立指标；两条旧监测历史核验去重 |
| [ACT2工程恢复](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-act2-engineering-20261101) | 1 | 第7步工程闭环 |
| [动作读出解析校准](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-readout-fit-20261103) | 1 | readout_ridge诊断；横轴2224为拟合anchors数量，不是优化器更新 |
| [选择性动作专家](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-selective-expert-20261031) | 1 | best5000检查点中止评估，记录已执行过程及未完成状态 |

匹配依据为实际加载checkpoint及原始命令，不能直接相信复制配置中的旧run ID。完整开发验证使用`success/raw/*`并按checkpoint步数绘图；多次checkpoint形成曲线，只有一次完整评估的实验仍是单点。工程验证使用`success/engineering/*`；RGB使用`success/rgb/*`，动作验证使用`validation/rgb_reference/*`；权重干预按`success/diagnostics/development/*`或`success/diagnostics/holdout/*`保留变体名称。原始开发初态5–9与独立初态10–14没有拼成同一曲线。未完成闭环使用`closed_loop_partial/*`并明确`completed=0`，没有填造完整成功率。全部闭环展示归入“05 闭环评估”。

三个旧监测run的训练标量与保留run同step指标一致，规范化旧名称后复用已有历史，没有将旧MAE作为新MSE。各来源原始进度及系统遥测按`evaluation_history/<来源run>/<原指标>`保留原step，并隐藏这些明细图以免挤占闭环section。源日志和配置另外完整备份。此轮未修改已经停止的J5历史执行源码；当前C1自动闭环继续使用上节已修复的训练run续写入口。

本地及服务器均保存`outputs/swanlab-run-consolidation-20261104/`：36份来源/目标备份、迁移脚本、逐点映射`prepared.json`、上传记录、删除前后清单和核验收据。28个来源共23077个标量/系统指标点均有明确目标映射；实际新增15297点，包含依据完整等规模协议计算的总体统计。删除前后均从云端重新读取目标原有与新增数据，共50058点，全部存在且成功率精确一致。一项历史GPU显存百分比发生1 ULP的CSV/JSON序列化舍入（26.599349832857982→26.599349832857985）；原值保留在备份，差异在`verification.json`单列，其余点一致。

官方API批量删除返回成功；删除后28个来源均不在项目列表，8个目标全部存在。`configs/research/swanlab_runs.json`现只列22个保留run，[迁移清单](../../configs/research/swanlab_merges_20261104.json)保留全部28项源/目标对应关系与评价协议。当前C1 run仍为RUNNING，控制器PID673053存活，训练状态26700/40000；本次操作没有暂停训练。验证依据为云端完整指标API，未声称完成网页截图核验；本轮仅云端数据与索引整理，未新增代码测试。

最后的元数据复核发现SwanLab 0.10.3在`resume="must"`但省略`config`时会将云端配置替换为空对象。已从迁移前备份恢复上述8个目标run的全部原配置，并逐个云端读回核对配置值、名称与FINISHED状态；迁移脚本改为显式传入原配置。恢复记录为`metadata-verification.json`及`restore_metadata.py`。此问题只影响云端配置展示，原指标、服务器训练配置和检查点均保留；C1正式日志入口原本就显式传入配置，不受此问题影响。

### 新SSH入口恢复训练

2026-11-04 15:37 Asia/Shanghai，用户提供新SSH入口并要求继续训练。实查同一运行目录、共享数据及双RTX5090可用，旧训练/控制器进程均不存在；状态文件仍写68980/running，但这不是活动作业证据。训练日志末尾为68980，无80k阶段原生退出收据，本次不推断中断原因。

最新完整`training/latest.pt`实际为68000步，含model、EMA、296项Adam状态及两rank Torch/CUDA/Python/NumPy RNG。已保留该检查点硬链接和中断前状态/配置/指标/日志至`attempts/host-reconnect-20261104-153746/`，从68000恢复原控制器，PID1337，累计目标仍80000。68000之后未落入完整检查点的更新需要重算；旧日志及历史曲线保留，不宣称从68980精确恢复。

执行沿用服务器现有源码、数据拆分、模型/损失/采样/学习率日程和同一SwanLab训练run，不覆盖当前其他未提交源码修改。已实际推进69160步，越过中断前进度且69000完整检查点已保存；双卡计算正常，稳定更新约0.20秒、峰值allocated9.02GiB/卡。恢复命令/新SSH入口与收据保存在本地`outputs/raw-compact-c1-20261104/host-reconnect-20261104/`；服务器`latest-launch.json`指向当前控制器和独立启动日志。没有恢复已删除的定时研究跟进。

此次读取的完整开发闭环：10000步152/200、20000步157/200、40000步154/200、60000步147/200，均native0。当前选型仍为20000步157/200（78.5%），并非最新60000步；相应完整权重保留。原控制器继续80k训练及闭环，并按全量开发结果选型后运行预定五照明独立最终评价。此次只恢复运行，未更改算法或重复功能验收。

恢复后的SwanLab云端API已读回`objective/flow7`新记录69140步（timestamp1793778113000），对应本地状态69160/running。监测仍为同一训练run；读回收据在恢复归档的`swanlab-after-interruption-step.json`，没有以旧云端最大step冒充恢复后的新上传。

### 首轮完成、结果归档与官方横评

2026-11-04，训练累计80,000更新、全部预定开发及最终评价已完成，最后原生退出时间17:24:03 +08:00。无活动C1训练/评估进程；训练状态completed，全部完成阶段退出码0，原SwanLab训练run已FINISHED，定时研究跟进保持关闭。

六个等规模开发节点成功率依次为5000步64%、10000步76%、20000步78.5%、40000步77%、60000步73.5%、80000步72.5%，故按预定协议选择20000步EMA。相同200场20k→80k新增成功10、退化22；动作验证MSE降低约4.61%，未转化为闭环收益。

最终仅评选定模型，40任务×初态20–29×五照明，共1513/2000＝75.65%。按光照平均：Normal77.25%、Low79.75%、ExtremeLow66.50%、Over80.00%、ExtremeOver74.75%；按类别平均：Spatial84.2%、Object89.2%、Goal78.6%、LIBERO-10长任务50.6%。40初筛＋1200开发＋2000最终，共3240条唯一协议内逐场记录已重算；不将不同模型/协议拼作总体成功率。

按用户要求，以平均成功率为主整理光照、任务类别及其交叉横评，核对RawVLA论文v1主表。官方最强RawVLA＋π0.5总体74.44%，本轮数值差+1.21个百分点；SOG为84.00%对79.82%，长任务50.60%对58.28%。ExtremeLow差+27.78、Normal差−18.85个百分点。官方10000场，本轮2000场，未作逐场配对，不宣称显著超越。官网跨后端均值65.18%与论文68.82%不一致，统一取版本化主表，不混用附录或网站旧数值。

完整结果、40任务明细、四组PNG/SVG、失败视频样例、官方来源及复算脚本见[结果报告](raw_compact_c1_results_20261104.md)。本地/服务器产物根为`outputs/raw-compact-c1-20261104/final-results-20261104/`；选定EMA推理导出`policy-selected-020000-ema.pt`约195.5MB，来源完整`training/step-020000.pt`保留。末尾`policy.pt/latest.pt`是80k，不代替选定20k。SwanLab API已读回最终step20000的0.7565/2000和开发step80000的0.725。

本轮整理只重算原始结果、查阅官方报告、导出已选权重与写入文档，没有新增优化更新或仿真场次。长任务和正常光基本操作为当前平均成功率的主要提升方向；极低照明长任务32%仍是负结果。只有一个训练seed、pseudo-RAW仿真、同40已知任务，结论不外推实机或未见任务。
