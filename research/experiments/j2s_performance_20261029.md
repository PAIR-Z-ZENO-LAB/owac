# J2S 吞吐与内存诊断修复（2026-10-29）

## 实测结论

旧训练global32=2卡×micro2×accum8，每微批取回四个loss标量，GPU前后向被小批次与CPU同步切碎。恢复为global32=2卡×micro8×accum2，线程数每rank4→8，取消逐微批标量同步，保持原两窗口一组的裁剪/翻转随机序列。

|指标|修改前|修改后|
|---|---:|---:|
|真实双卡更新中位秒数|1.148676（37记录）|0.470592（22记录）|
|窗口/秒（32÷更新秒）|27.86|68.00|
|GPU0/1平均利用率|41.17% / 48.25%|78.50% / 79.71%|
|显存采样最大MiB|12906 / 12906|15616 / 16286|
|cgroup匿名内存GiB|5.171–5.179|5.219–5.470|
|文件页缓存GiB|109.920–110.033|109.468–109.802|
|各rank打开句柄首尾|131 / 131|131 / 131|

真实更新吞吐约2.44倍。GPU/内存采样分别12次/23.68秒与24次/53.49秒；不是全训练平均或长期峰值承诺。修改后CPU约4.91个核心等效占用。GPU利用率不要求人为达到100%，以实际吞吐和训练正确性为依据。

## 内存判断

AutoPanel的总内存包含文件页缓存。当前近110GiB来自RAW/RGB随机读取及模型保存/读取的文件页；匿名内存约5GiB、句柄不增长，cgroup oom/oom_kill为0。没有证据把原曲线认定为匿名内存泄漏。页缓存填满cgroup后会回收；采样期间存在直接回收开销，所以不能把它描述成完全没有成本。没有执行drop_caches、malloc_trim或全局清缓存，没有删除训练资产来伪造较低内存曲线。

恢复改为torch.load(mmap=True)，避免完整checkpoint先复制进匿名内存。实测发现Adam CPU step计数器会保留完整文件映射，使恢复后RSS约7.84–8.00GiB/进程，而匿名内存仍约5GiB总量。这是在本次恢复优化中发现的生命周期问题，不是原曲线匿名泄漏的证据。已在optimizer.load_state_dict后一次性clone CPU计数器，使它们与checkpoint存储解耦；不扫描/拷贝GPU动量，不在热路径反复处理。

SwanLab新增host/cgroup_memory_gib、anonymous_memory_gib、file_cache_gib、mapped_file_gib、oom_kills及worker RSS/open_fds。原GPU/磁盘指标保留，CPU内存各层不相加重复计算。

## 比较与恢复证据

- 1162步收到SIGUSR1，完整checkpoint完成、native0退出。旧launch/config/退出/源码收据保存到run/attempts/performance-pause-1162。
- 在同一1162状态、真实Adam驻留、后续8个rank0采样批上做零优化更新比较。micro2/4线程约1.05秒，micro8/4线程约0.426秒，micro8/8线程约0.411秒，micro16/8线程约0.463秒。因此选8，不用“更大一定更快”替代测量。单卡测量不冒称双卡吞吐。
- 从1162完整恢复，实际越过1500；531窗口验证正常，total loss0.265446，dense cosine loss0.195544，Gram0.010517，global cosine loss0.048867。性能修复不作为模型能力收益归因。
- 为解除checkpoint映射，1709再次完整保存、native0停止并从1709恢复；不重置学习率、优化器、RNG、采样位置或100k预算。收据位于run/attempts/mmap-pause-1709。
- global32、教师、模型、损失、数据拆分和增强分组不变。测试逐元素核对micro2和micro8产生相同增强数据与Python RNG末态。GPU微批变大导致浮点归约次序变化，不承诺参数位级一致，不新增人为数值门槛。

12项相关测试通过，包括增强/RNG保持、固定教师梯度边界、窗口reset、CPU计数器存储解耦及监控内存语义。修改文件ruff/format通过。make check仍有本轮之前诊断脚本的213项lint问题，完整日志已保留，不报告全库通过。

本地证据：outputs/j2s-performance-20261029/。远端原始采样与microbatch比较：outputs/j2s-raw-semantic-20261029/performance-*.jsonl、performance-summary.json、microbatch*-benchmark.json。SwanLab继续同一run，不新建或清空历史曲线。

## 最终恢复核实

1709恢复后实际超过1880步，两rank RSS约3.06GiB，匿名约2.32GiB/rank；/proc/maps中latest.pt映射均为0，证实CPU计数器解耦后大检查点映射已释放。原始记录mmap-release-verified.json已同步。本轮没有遗留额外诊断模型作业，正式训练继续。
