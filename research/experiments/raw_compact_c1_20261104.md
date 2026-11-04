# C1 RAW 紧凑策略训练与多任务闭环

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
