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
