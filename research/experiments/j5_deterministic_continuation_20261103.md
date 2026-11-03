# J5：关闭视觉训练随机机制的3000→4000对照

2026-11-03用户授权修正并启动。使用原3000完整恢复状态，而非4000退化权重或6/20的半权重推理候选。原实验、所有原生闭环及半权重候选保持不动。

## 仅修改的训练因素

`adaptation.deterministic_raw_encoder=true`：关闭RAW编码器原SSL配置中的随机RoPE坐标shift/jitter/rescale以及drop-path。视觉仍全量可训练、保留train模式及梯度检查点，语言/动作基础权重仍冻结。改动此前在实际接受编码器与真实RAW输入上验证：重复train与train/eval前向maxabs0，响应与patch embedding梯度非零，见`j5_weight_audit_20261103.md`。此为数值/梯度证据，不是成功率收益。

继续同一个3000扩展检查点的权重、504项既有Adam状态、双rank RNG与选择状态；新视觉组与旧解冻实验一样从零Adam历史起步。可训练48,545,923参数。vision基准LR1e-6、bridge3e-5、language/expert_content5e-6、expert_qk1e-6；保留原以6000为终点的余弦曲线，不把终点改为4000以免改变学习率。双卡global256/micro64/accum2、Normal源拆分、官方flow loss与7D动作验证保持不变，不使用RGB训练。

控制器仅运行到累计4000步，再执行原四套件task0、Normal、初态5–9共20原生闭环并结束。对照原3000为5/20、旧解冻4000为3/20；半权重6/20是开发集选择候选，另列。无新增成功率门槛，不自动继续5000/6000或扩展照明。

本轮直接正式恢复，不额外在3003插入工程验证；普通3500/4000验证保留。此与前轮多一个3003验证计数有差别，已披露；原3000 best/bad_evals/lr_factor完整继承。

## 执行

配置`configs/experiments/j5_pi05_unfreeze_deterministic.json`。远端输出`/root/owac/repo/outputs/j5-deterministic-20261103`链接到数据盘`/root/autodl-tmp/owac/j5-deterministic-20261103`。独立`source/src`执行快照（起点实现提交5b5204b），原3000检查点硬链接为initial-003000.pt及初始best.pt，后续保存以原子替换创建新文件，不修改历史inode。

启动：runtime-j5-pi05环境，PYTHONPATH依次为独立source/src、官方openpi15a9616/src及openpi-client/src；`python -m owac.runtime.j5_action_curriculum --config outputs/j5-deterministic-20261103/production.json --output outputs/j5-deterministic-20261103 --resume outputs/j5-deterministic-20261103/normal/initial-003000.pt`。控制器PID605598。

监测：[OWAC修正版训练](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-deterministic-3000-20261103)，沿用四类section；4000闭环独立run `owac-j5-deterministic-dev-4000-20261103`。不回填其他实验曲线。

## 启动实测

实际完成3001更新并完整保存：checkpoint中deterministic_raw_encoder=true；715项Adam状态，RAW响应参数最大更新4.7684e-7、patch embedding最大更新5.8860e-7，确认关闭随机机制后视觉仍在学习。3001为7.571秒/更新、显存峰值18.041GiB/卡、容器RAM91.750GiB（匿名7.908、文件缓存82.409、共享.539），OOM0。这是启动时刻，不是长期资源上限保证。

SwanLab官方API独立读回RUNNING及3001的训练flow7=.09535874、vision实际LR5.8691e-7，排除仅本地日志更新。收据`startup-verification.json`。下降的首步训练loss不能当作成功率收益；本轮原生闭环尚未产生。

13项相关测试通过；make check仍被既有213项lint问题阻断。未修改无关工作区文件，未覆盖任何旧运行。
