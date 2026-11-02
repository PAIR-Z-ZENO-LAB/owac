# J5-LIBERO：相同LIBERO后端初始化的RAW对照

## 问题与设计

2026-11-02用户授权保留J5视觉、继承已经会执行LIBERO任务的语言/动作后端，仅用RAW、指令和动作标签进行接口适配。负责人lzh/Codex；证据层为真实训练与LIBERO仿真闭环，不涉及真机。

旧ACT2已完成6000步/native0：1000/3000/6000三个正常光开发节点均0/20；6000步动作MSE=0.123421，执行前5步MSE=0.126209。参考官方RGB在相同初态5–9的四任务为20/20。旧实验权重、视频、日志完整保留，不续写旧运行。

新实验直接将`assets.pi05`指向RGB参考实际使用的`/root/autodl-pub/checkpoints/openpi/pi05_libero_pytorch/model.safetensors`。`fresh_official=true`通过原有严格加载路径初始化完整官方模型，随后移除RGB视觉塔，装入已接受的J5 12818 RAW编码器。新建桥接与零输出LoRA，不加载旧expert5000/ACT2适配器或优化器。保留的官方语言和动作基础权重冻结，J5冻结且eval；训练残差桥接、语言rank16、动作rank8 Q/K/V/O/MLP LoRA。

这与RGB参考共享后端初始化；与旧ACT2相比也清除了旧RAW适配器历史，因此不包装成仅改变一个权重文件的严格单变量消融。官方LIBERO后端有RGB训练历史，本实验不主张整套策略从RAW零起点学习。训练没有RGB图像、RGB特征、DINO或教师动作监督。

## 固定条件

- 配置：`configs/experiments/j5_pi05_libero_backend.json`；基础SHA `f6c0e6e`，分支`lzh/j5-libero-backend`。保留原工作区dirty修改；本次只新增配置、监测登记、记录及课程控制器的实验命名/正常光终止选项。
- J5：`outputs/j5-raw-relations-20261030/attempts/user-stop-12818/encoder.pt`；数据清单：`outputs/raw-distillation-all-bench-20261022/inventory.json`。
- 保持原轨迹拆分和Normal筛选：278训练轨迹、32验证轨迹；验证256 anchors/2272有效动作步。正常光验证覆盖23/40任务的限制仍在，本次不同时修改数据策略。
- 种子保持20261101；6000更新；双卡global256/micro128，每rank2个worker、prefetch1、逐帧读取。官方32D flow-matching训练损失；验证只统计7D有效动作并排除尾部补齐。
- 四组LR：桥接3e-5、语言5e-6、专家内容5e-6、专家Q/K 1e-6；warmup300、cosine、原plateau/选择策略不变。按验证动作MSE保留best；每500步验证，1000/3000/6000保留检查点并执行完整开发闭环。
- 开发闭环：四套件task0、每任务初态5–9、Normal、相同环境/扩散种子；10步去噪、每次执行5动作。每节点20场，保留失败及完整分母。不自动扩展五光照；完成三节点后正常结束，无新增成功率门槛。
- 成功判据采用LIBERO原生任务判定。损失下降、训练健康和启动通过不等于动作收益；20开发场不能外推全LIBERO。

## 运行与复现

服务器工作目录`/root/owac/repo`；输出根`outputs/j5-libero-backend-20261102`。运行环境沿用`/root/owac/runtime-j5-pi05/bin/python`、官方openpi固定版本`15a9616`及原LIBERO仿真环境；双RTX5090、主机容器116GiB、BF16计算、J5 FP32冻结。

```bash
export PYTHONPATH=/root/owac/repo/outputs/j5-libero-backend-20261102/source/src:/root/owac/openpi-official-15a9616/src:/root/owac/openpi-official-15a9616/packages/openpi-client/src
export JAX_PLATFORMS=cpu
/root/owac/runtime-j5-pi05/bin/python -m owac.runtime.j5_action_launch \
  --config outputs/j5-libero-backend-20261102/production.json \
  --output outputs/j5-libero-backend-20261102/normal \
  --steps 3 --initial-eval --no-evaluation
```

初始全验证和3次真实更新后，将完整状态续接到课程控制器，不重新初始化权重/优化器。正式控制器采用格式化后的独立`source-production/src`快照，配置保持相同配方；工程指标按原step回填同一SwanLab运行。

```bash
/root/owac/runtime-j5-pi05/bin/python -m owac.runtime.j5_action_curriculum \
  --config outputs/j5-libero-backend-20261102/production.json \
  --output outputs/j5-libero-backend-20261102 \
  --resume outputs/j5-libero-backend-20261102/normal/latest.pt
```

监测：[OWAC独立训练运行](https://swanlab.cn/@Jeong/OWAC/runs/owac-j5-libero-normal-20261102)。沿用统一section，动作MSE为主，补充执行前5步MSE、平移/旋转命令RMSE、夹爪符号错误率、优化器与分层内存。开发运行使用`owac-j5-libero-dev-{step}-20261102`，不会覆盖ACT2的评估记录。

## 实测与边界

- 初始化验证：step0动作MSE=0.3543558，step3=0.3539329；相同256 anchors/2272有效动作步。只是新接口的初始测量，不能推断最终动作效果。
- 3步执行native0，`attempts/initialization-3`保留完整检查点、504项优化器状态、双rank RNG及原始日志。初始化收据确认后端文件与RGB参考相同，旧适配器base为空；26359552训练参数中没有RAW编码器参数。
- 已实际从3恢复并完成step4。初次启动传入相对resume路径，课程控制器额外附带reset-patience；原bad_evals为0，未改变状态。仓库已将恢复路径标准化，避免未来同目录续训误判为跨阶段。
- step4：每更新7.25秒，主机83.96GiB（匿名8.13、文件缓存74.43、共享1.04），GPU峰值分配27.20GiB/卡，OOM0；一次采样双GPU利用率均100%，不冒充长期均值。正式SwanLab已连接，0/1/3步历史按原step回填；下一个开发闭环节点1000。
- 相关LoRA、动作指标、监测测试10项通过；修改文件Ruff/格式/编译通过。`make check`仍被其他文件既有213项lint问题阻断，没有修改无关文件。

整体训练/闭环尚未完成，不预报高成功率；总GPU时间、长期峰值资源和最终策略质量待测。原始收据本地副本位于`outputs/j5-libero-backend-20261102`，服务器保存完整权重、源码快照和配置。
