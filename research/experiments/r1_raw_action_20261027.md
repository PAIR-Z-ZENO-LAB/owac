# R1：真实动作监督的RAW视觉头（2026-10-27）

<!-- R1_LIVE_BEGIN -->
实际进度：**23,280/100,000更新**；状态：**running**。
最近训练batch flow loss：0.02481530（第23280步）。
最近验证：第23000步；完整验证flow loss 0.02978320。
<!-- R1_LIVE_END -->

R1正在训练，目标为100,000次实际优化器更新。尚未交付最终100k权重。用户本轮确认仅使用原拆分中的LIBERO，RoboTwin不参加本阶段。

## 数据与训练

复用原inventory.json，LIBERO train1421/validation177/test173源，35,693训练anchors。Normal/Low/ExtremeLow/Over/ExtremeOver全部保留；只读共享NPZ，沿用原源轨迹分区、双相机、RAW缩放、六帧burst、六端点反传、之前完整因果前缀和每源reset。训练端点从25开始、stride5；验证流式遍历全部验证端点，分别记录startup及eligible。test不读取动作或观测。

当前观测t监督actions[t:t+10]；episode末不足10步以真实动作有效掩码归一化损失。动作全部7维及8维robot_states沿用原生quantile归一化（q01/q99、原分母floor与clip），动作补零到32维；无额外夹爪转换。任务文本读取metadata_json.task_description。部署动作协议与历史LOWLIGHT保持。

从原π0.5 LIBERO的视觉骨干/位置编码/投影初始化，stem seed101、gate重新初始化。整个视觉头428,579,952参数训练，原语言/动作后端3,201,953,824参数冻结。实际路径不创建RAW教师、ISP或TargetStore，不读取蒸馏权重/目标。冻结后端保留输入梯度，使用原prepare_flow_targets、embed_action_suffix、forward_with_action_expert与decode_action_velocity；缺失右腕相机保留原masked slot。损失为有效未来动作上的32维原生速度MSE。

双卡global16；每卡microbatch2×4累积，按整个global batch真实有效动作数加权。AdamW lr1e-5、weight decay0.01、100更新warmup、原gradient clip1.0；视觉FP32权重/Adam、BF16前向。视觉块和冻结后端共享层使用activation checkpointing。验证每500更新及最终100k，固定每源flow抽样种子且恢复训练RNG；不按验证曲线提前停止或替换最终模型。

## 源码、配置、运行

- `src/owac/training/raw_action_adaptation.py`：独立动作数据路径、原生flow损失、100k计数、完整恢复、磁盘写入限制、验证及曲线。
- `src/owac/runtime/raw_action_adaptation.py`：训练/恢复入口。
- `configs/experiments/r1_raw_action_100k.json`：精确配置。
- 远端运行根：`/root/owac/repo/outputs/raw-distillation-all-bench-20261022/r1-raw-action-20261027/`。
- 启动环境：现成`/root/owac/runtime-full-vision/bin/python`，torchrun双rank；启动命令与原生退出码保存在`train-r3.launch.json`及完成后`train-r3.exit.json`。
- `updates.jsonl`逐更新记录损失/耗时/学习率/采样游标；`validation.jsonl`记录全部验证及按照明分组曲线；`curves.png/svg`在验证后更新。
- `latest.pt`保存视觉权重、Adam、scheduler、每rank CPU/CUDA/Python RNG、epoch/cursor和实际global_step。首更新、每250更新及每次验证后保存。最终保存`final100000.pt`；中断仅写paused，不冒充最终权重。
- 本地`outputs/r1-raw-action-20261027/`保留启动器、原有修改清单和运行证据。

首次两个启动在0更新时因评估runtime依赖失败：先是Transformers版本不匹配，再是缺少accelerate。改用服务器已有训练runtime；未下载或改装依赖，原始失败日志保留。第三次启动已实际推进11更新，首更新完整检查点5,143,546,139字节已保存。第11步loss0.07005237；这只是单batch训练损失，验证尚未到500步。

启动查看所属输出27,901,090,700字节＋旧临时缓存366,055,590,658字节，总计393,956,681,358字节（约366.90GiB）；380GiB上限剩余14,065,211,762字节。旧资产不删除。新阶段只保留一个完整latest（约4.79GiB），原子替换时同时保留新旧两个；最终视觉文件约1.60GiB。写入时按实际字节检查预算，只统计本阶段小目录，不重复扫描旧输出/缓存；预算错误直接失败并保留原latest。

## 验证

22项相关测试通过（动作/因果9项，研究台13项），覆盖动作对齐、末尾掩码、冻结后端输入梯度、原桥接路径固定noise/time损失一致性、检查点原子写入及磁盘失败保持原文件，以及既有batched因果/reset测试。`make check`通过。真实服务器已执行优化并保存首步完整状态；首个验证及最终100k结果待训练实际产生。

结果同步由本地常驻 `owac.runtime.raw_action_followup` 每300秒读取末条训练/验证记录，更新本报告与研究台；验证后同步曲线，进程结束后同步完整小型日志和原生退出收据。最终权重保留服务器，不额外传输旧资产或数据。同步进程收据位于 `outputs/r1-raw-action-20261027/followup-launch.json`，错误直接记录到 `followup.log`，不自动重训。

## SwanLab同步与命名

[R1实时曲线](https://swanlab.cn/@Jeong/OWAC/runs/owac-r1-action-100k-20261027) 已接入，名称为 `R1｜动作监督｜LIBERO五照明｜全视觉头｜100k｜20261027`。独立 `owac.runtime.raw_action_swanlab` 增量读取原始JSONL并回填此前指标，不重启训练。2026-10-27 17:39确认训练661步、验证500步已同步；500步完整验证flow loss为0.05722073。

OWAC现有5个run已按 `configs/research/swanlab_runs.json` 更新云端名称、说明、group及job_type并回读确认，原ID和结束状态/时间保留。清单tags为后续新建run的标签配置，历史run标签尚未回填。SDK resume接口不修改历史名称且会置为RUNNING，因此改用官方前端元数据接口；受影响历史状态已恢复并核对。

2026-10-27 18:00：核对云端，训练与 `validation/` 指标已上传；项目旧 `val/` 图表属于其他实验的指标命名。R1 同步器增补每约30秒的双卡利用率/显存与继承380 GiB预算占用（启动时基数加本阶段目录），从第1154步开始采集，之前没有资源曲线；已从云端回读确认。训练进程未中断。

2026-10-28 用户要求临时暂停并查看最新检查点闭环。SIGUSR1在29,071更新保存完整latest.pt后正常退出native0；尚未恢复训练。临时中期闭环使用该latest.pt及既有LOWLIGHT配置/catalog，Normal/Low/ExtremeLow各20场，共60场学生实验。远端输出 `outputs/raw-distillation-all-bench-20261022/r1-closed-loop-step29071-20261028/`，launch.json/exit.json保存原生运行收据；本次为中期检查，不代替最终100k实验。

中期闭环首次启动因LIBERO config路径误指向新输出目录，在首场reset前native1退出，0/60完成。保留原attempts及exit收据。修正为既有lowlight-20261027/libero-config/config.yaml后，重新启动至 `r1-closed-loop-step29071-20261028-attempt2`；失败启动不计入策略成功率。训练继续暂停在29,071步。

持续监测启动：attempt2因EGL vendor配置路径仍指向新输出目录，在首场reset时native1退出，完成0场。attempt3修正为既有runtime-gl/nvidia.json，保留两次失败启动收据；尚待真实执行证据。

已确认attempt3真正进入闭环：Normal/test-101100完成300环境步、60次重规划，未成功，耗时27.54秒，视频和动作NPZ已落盘；test-101101已启动。此次启动问题已解决。实际输出为 `r1-closed-loop-step29071-20261028-attempt3`。

## 29,071步中期闭环结果

60/60场已完成，三个照明进程均native0，无episode错误；全部运行至300步未成功。Normal 0/20、Low 0/20、ExtremeLow 0/20。相对历史蒸馏学生20/20、9/20、0/20，配对退化分别20、9、0，恢复均0；相对历史教师20/20、11/20、0/20，退化20、11、0，恢复均0。原始attempts.jsonl、动作NPZ、60视频与exit.json保存在远端 `outputs/raw-distillation-all-bench-20261022/r1-closed-loop-step29071-20261028-attempt3/`。所属输出/缓存总399,198,289,177字节，低于380GiB。训练仍暂停29,071步；该中期检查点尚未恢复闭环控制，失败具体原因待分析，不能凭较低验证flow loss推断控制能力。

梯度诊断：以29,071步检查点和真实训练源endpoint25执行单样本原生flow反传，无optimizer.step。442个视觉参数张量全部有非零梯度，Adam step全部29,071且exp_avg非零；冻结后端0个可训练参数张量、0个grad。token梯度L2=0.0010669093，gate=0.0001738446，projector=0.06208673。本次单样本loss=0.002470404，不作为全局梯度统计或闭环失败原因证明。原始逐参数结果与诊断源码在outputs/r1-raw-action-20261027/gradient-diagnostic/。

Adam更新诊断：用29,071步存储的一阶/二阶矩、bias correction、lr1e-5及wd0.01估算最近一步更新（衰减项使用已保存的更新后权重，因此是估计而非两检查点直接差分）。stem/gate/vision/projector的逐参数更新RMS分别1.887e-6/1.606e-6/1.764e-6/2.092e-6，相对模块权重L2分别0.03434%/0.03138%/0.002684%/0.007642%。数据梯度产生的自适应更新范数明显大于衰减项。该结果不支持简单用原始梯度小解释为几乎不更新，但不能证明当前步长足够；累计漂移和有效梯度方向仍未测量。原始结果见gradient-diagnostic/adam-update-probe.json。

累计权重比较：重建本次原始初始化，对比29,071步，stem/视觉骨干/投影相对L2变化28.73%/2.318%/7.475%；gate sigmoid均值由0.5变为0.499543（范围0.496054–0.503876）。详细模块及27层数据见outputs/r1-raw-action-20261027/gradient-diagnostic/parameter-comparison.md及parameter-delta.json。不是几乎无参数更新；是否适配充分仍不能仅由权重距离判断。

归一化排查：R1 normalize_window与固定StarVLA rawvla_npz_datasets的pi05_quantile分支一致（全7维动作及8维状态，quantile+clip）。闭环openpi/PI05后处理全7维反quantile，不加state、不翻转夹爪。训练分母max(range,1e-6)，部署逆变换range+1e-6存在微小差别。初始5个训练示范endpoint25动作往返各维MAE低于8.2e-7；进一步检查task6三个完整示范，越界动作因clip不可逆，最大绝对误差约0.0991，夹爪误差约0.000399且未反号。状态训练clip而部署不clip是不一致：60场首个决策状态均未越界，后续状态元素平均6.53%越界，最大超界0.826。因此不能据此认定归一化导致全失败，也不能排除它放大后续偏离；尚未做相同状态/噪声的clip与不clip动作对照。训练保持暂停，未修改协议。

状态clip因果对照：29,071权重，4个真实示范观测（3个task6），每对固定完整因果视觉tokens、语言、同一noise与10步flow，仅修改state是否clip。2个越界样本最大超界0.688637及0.033897，2个未越界对照；全部7维/10步生成动作逐元素差为0。源码确认prepare_language_state对LIBERO不序列化state，而OpenPI05ActionHead.embed_action_suffix不使用连续state参数，故该路径实际不消费状态。此前“clip差异可能放大闭环偏离”的判断在当前路径上撤回；state接入是否符合原始预训练协议是另一项待核对问题，不能据此直接归因0/60（历史成功蒸馏学生也沿用该路径）。原始数据及脚本在gradient-diagnostic/state-clip-action-probe.json和state_clip_probe.py。仅离线推理，无权重更新。

官方RGB基线与动作质量诊断已补充至 `research/experiments/r1_rgb_baseline_loss_audit_20261028.md`：官方LIBERO无状态输入是配置所设；官方动作quantile不clip，与R1所沿用StarVLA RAW分支存在差异。4样本flow分解确认总32维MSE显著小于真实7维误差，10步采样亦有明显旋转/夹爪偏差；尚不能建立失败因果或改用新的loss。
