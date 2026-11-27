2026-11-04 主分支与学术首页：按用户要求，将`v1.0`（970acfe）快进合入`main`，保留全部历史且不移动发布标签。README参考RAX的信息组织方式，改写为英文正式项目首页，包含当前C1方法图、逐照明/套件成绩、RawVLA官方横评、开发与复现入口、资产状态及软件引用；新增[复现指南](raw-compact-policy.md)和`CITATION.cff`。历史流水继续保留在本状态及实验记录，旧首页可从v1.0读取。未新增训练、闭环或更改模型；公共权重下载及项目级许可证未发布的事实在首页明确说明。

## 2026-11-27 论文复现发布

已按学界论文开源项目规范重写公开主页 `README.md` 并同步至 `origin/main`：补充居中标题与状态徽章、论文主视觉图（teaser、框架图、表征一致性与物体分离分析图、固定/切换光照曲线与柱状图，导出于 `docs/assets/`）、核心特点、方法公式与三阶段课程及策略结构规格、LIBERO 全量十方法主表与套件明细、配对消融表、真机 RAW10 三任务四光照结果表、Model Zoo、离线快速推理/评分/复现与真机部署入口、仓库结构及 BibTeX 引用；所有数值与归档 40,000 条主评测及 800 条消融记录核对一致。

`paper-v1.0.1` 修正文档中公开 J5 文件的 SHA-256。公开文件裁剪了训练元数据，212 个编码器张量及同一公开输入的 CLS、patch 输出与训练归档一致。公开工件从匿名下载重新验证。发布专项测试 107 项通过，另有 13 项依赖 ROS 2 接口，在工作站跳过。全库测试 3,133 项通过、53 项跳过、21 项失败；19 项在发布前主分支复查同样失败，两项监控测试单独重跑通过。全库 `make check` 仍报告九个历史 JEPA 文件的 213 条样式问题，本次变更的 Python 文件通过 Ruff 和格式检查。

`paper-v1.0` 包含从训练与评测服务器核对的 OWAC-Regression 10k EMA、OWAC-Flow 20k EMA、分阶段视觉训练配置、原仿真拆分、逐回合评分和真机源码。两个仿真权重以 Safetensors 单独发布。原始真机示范、真机策略权重及依赖示范拟合的现场参数保存在私有位置。

复核了归档与公开版本 292 和 298 个张量的逐元素一致性。在同一条公开 LIBERO 轨迹上，两种实现各自输出的 84 个动作分量完全一致。在 RTX 5090 上用小批量公开数据分别跑通 J3、J4、J5、回归、Flow 五个训练入口各一次参数更新；两份动作权重均完成验证、保存和导出。固定与切换照明的原生仿真分别跑通一回合，并从服务器保存的 40,000 条正式记录重新汇总主表。部署、训练和验证范围见[论文复现指南](paper-reproduction.md)与[真机指南](real-robot-reproduction.md)。


2026-11-04 v1.0成功里程碑：用户确认视觉编码器和动作训练成功，授权提交全部修改并推送v1.0标签。选定20,000步EMA，40任务五照明平均75.65%（1513/2000）；当前阶段结论及权重入口见[发布说明](releases/v1.0.md)。本地相关测试27通过/1可选依赖跳过；本轮模块Ruff/格式通过，全库make check仍为已有213项lint问题。定时研究跟进保持关闭。

2026-11-04 C1首轮结果已归档：80,000更新及全部预定闭环native0完成，最终结束17:24:03；当前无活动训练/评估进程。按开发157/200（78.5%）选择20,000步EMA，独立初态20–29、40任务、五照明最终1513/2000＝75.65%。光照平均Normal77.25%、Low79.75%、ExtremeLow66.50%、Over80.00%、ExtremeOver74.75%；类别平均Spatial84.2%、Object89.2%、Goal78.6%、Long50.6%。RawVLA论文主表最强RawVLA＋π0.5为74.44%，数值差+1.21个百分点，但Normal和Long仍落后；本轮2000场/官方10000场，未作同种子重现。选定EMA推理导出与四组PNG/SVG图、逐场证据、统计JSON已保存，完整历史检查点/失败保留；同一SwanLab run为FINISHED。80k末权重开发72.5%，不能替代选定20k策略。新SSH为183.249.99.226:27786，定时研究跟进保持关闭，未启动下一轮。见[完整结果与官方横评](../research/experiments/raw_compact_c1_results_20261104.md)。

2026-11-04 15:37 C1已在用户提供的新SSH入口恢复：旧进程已停止，日志至68980，最新完整状态为68000；保留中断证据后从68000恢复模型/EMA/Adam/两rank RNG，已真实推进69160并保存69000完整检查点，云端已读回69140新指标，目标仍80000。当前闭环最佳为20000步157/200（78.5%），40000/60000分别154/200、147/200；按最佳开发成绩选型，现有训练与最终评价控制器继续，同一SwanLab run，定时研究跟进保持关闭。见[实验记录](../research/experiments/raw_compact_c1_20261104.md)。

2026-11-04 其他历史闭环run已合并清理：25个闭环/诊断run及3个重复监测run合入8个对应训练run，项目实验数50→22；包含此前C1清理，累计54→22。来源23077个标量/系统指标均有保留映射，删除前后复核目标原有及合并数据共50058点；成功率完全一致，仅一项GPU遥测值有1 ULP序列化舍入并保留原值。开发曲线按checkpoint步数呈现，RGB、工程、权重干预、holdout及中止评估分列；单次评估仍保留单点。来源配置/指标/日志已双份备份，原始视频和权重保留。C1仍RUNNING，12:53实查26700步向40000推进。对应关系见[迁移清单](../configs/research/swanlab_merges_20261104.json)，详情见[实验记录](../research/experiments/raw_compact_c1_20261104.md)。

2026-11-04 C1重复闭环run已按用户要求清理：1000/5000/10000/20000四个run全部24个指标值逐项核对已合入训练run，备份云端配置/指标/日志后删除四个重复云端run；项目实验数54→50。删除后API复核24点仍完整，训练run仍RUNNING，实查21900步向40000推进；原始逐场证据与两份云端快照保留。审计目录`outputs/raw-compact-c1-20261104/cleanup-discrete-runs-20261104`，详见[实验记录](../research/experiments/raw_compact_c1_20261104.md)。

2026-11-04 C1闭环监测已接回[训练run](https://swanlab.cn/@Jeong/OWAC/runs/owac-raw-compact-j5-20261104)的“05 闭环评估”：横轴为checkpoint训练步数，云端读回5000/10000/20000步成功率64%/76%/78.5%及四套件完整序列。1000步40场初筛单列，独立最终评价另列；后续不再按checkpoint新建run。执行副本已更新，原控制器继续40000节点并实测超过20040步。日志专项本地7项、服务器7项通过，相关测试合计14通过/1可选环境跳过；全库make check仍为既有213项lint错误。详见[实验记录](../research/experiments/raw_compact_c1_20261104.md)。

2026-11-04 C1完成5000步Normal40任务200场闭环：128/200=64%，native0；空间39/50、物体42/50、目标34/50、长任务13/50。重合20初态从旧最佳6/20升至13/20（新增7、退化0），结构与数据覆盖同时改变，不能单因素归因。48,850,866参数、五照明1421源；已完整恢复并超过5120步，继续10000节点，后续八环境并行评估。当前是开发集pseudo-RAW仿真收益，非实机/全面高成功率；定时研究跟进已按用户要求删除；服务器现有80k训练与阶段闭环控制器继续运行。[方案](plans/RAW_COMPACT_POLICY_2026-11-04.md) · [实验](../research/experiments/raw_compact_c1_20261104.md)。

2026-11-03 J5本体感知已接通：RAW双视角+7关节角+双指位置+任务文字，经官方pi05离散状态提示进入后端。核实8维robot_states为末端位姿，改取仅机器人qpos；四套件传感器核对一致。3000→3003真实更新/完整保存/native0，13测试通过；220步工程闭环0/1，状态干预确实改变动作，不宣称收益。正式从3003恢复至6000，4000/5000/6000各20场Normal闭环，OWAC新run，OOM0。见 [本体输入记录](../research/experiments/j5_proprio_20261103.md)。

2026-11-03 J5原始权重与直接干预完成：372个基础张量与官方同dtype逐元素相等；新增140场原生闭环。3000与修正版4000混合开发6/20、新初态5/20，此前最佳分别6/20、4/20；收益有限。语言LoRA减半1/20、专家清零3/20、解析读出3/20。主要差异定位到RAW前端/策略接口，未证明单一根因。候选独立保存，全部实验结束。见 [权重干预记录](../research/experiments/j5_original_weight_interventions_20261103.md)。

2026-11-03 已按用户授权启动J5修正版：从同一3000完整状态恢复，仅关闭视觉随机RoPE/drop-path，视觉梯度和原LR/batch保留。3001更新/保存已核实，SwanLab云端已读回3001；显存18.04GiB/卡、OOM0。累计4000后执行相同20闭环并停止，收益待测。见 [修正版记录](../research/experiments/j5_deterministic_continuation_20261103.md)。

2026-11-03 J5权重实查完成：4000完整策略3/20，视觉单独更新1/20、撤回桥接2/20、3000/4000半权重6/20（原3000为5/20，新增成功3/退化2）。半权重推理候选独立保留，非可恢复Adam状态。发现解冻同时启用SSL随机RoPE缩放/drop-path，固定权重动作扰动已实测；增加opt-in关闭选项，未另训。4000完整保存/native0，后续训练及诊断服务已停止。见 [权重实查](../research/experiments/j5_weight_audit_20261103.md)。

2026-11-03 用户选择从J5-LIBERO 3000步解冻RAW视觉续训：继承权重/504项Adam/RNG，新增视觉LR1e-6；4855万可训练参数，global256/micro64。3000→3003真实更新/native0，视觉参数变化已核实，18.04GiB显存/卡、OOM0；正式控制器已启动，4000/5000/6000各20闭环。见 [实验记录](../research/experiments/j5_unfreeze_3000_20261103.md)。

2026-11-02 J5退化实查：六轨迹重放及348同观测动作精确复现；取罐6000失败、仅修夹爪仍失败、3000前55步接近抓取后交6000成功。验证取罐转折MSE .166→.205，均匀anchor仍改善；尚未定位具体参数模块。训练0，诊断服务已关闭。见 [实查记录](../research/experiments/j5_contact_regression_diagnosis_20261102.md)。

2026-11-02 SwanLab同步修复：旧J5-LIBERO run云端曲线停3000、本地已3652；完整保存/native0后保留历史，新建OWAC续写run `owac-j5-libero-live-3652-20261102`，原step全历史回填，模型/Adam/RNG/选择状态不重置。新run运行期间API已读到新更新3653及3500步MSE0.1060712；继续原6000节点，OOM0。旧run续写/云端可见性异常的内部根因未确定。见 [运行记录](../research/experiments/j5_libero_backend_control_20261102.md)。

2026-11-02 J5-LIBERO独立对照已启动：保留冻结J5 12818，改用RGB参考同一pi05_libero后端，清除旧动作适配器，新桥接及语言rank16/动作rank8 LoRA；Normal原数据/6000步配方。3步/native0及完整恢复至4已实测，双5090显存27.20GiB/卡、主机83.96GiB、OOM0，OWAC独立监测。1000/3000/6000各20开发闭环后结束，不自动扩光照；未宣称动作收益。旧ACT2已6000步/native0，三个节点均0/20。见 [对照记录](../research/experiments/j5_libero_backend_control_20261102.md)。

2026-11-01 J5-ACT2配对诊断：1000步Normal开发集RAW0/20、官方LIBERO RGB20/20，逐场物理初态/种子/光照/扩散噪声一致。256同源验证anchors动作MSE为0.17848/0.01842；指令干预响应RAW约为RGB三分之一，换图有明显响应。初始化不对等，不能归因于视觉编码器单独失效。1047步完整暂停后恢复，已完成1050更新；跳过已有1000评估、继续原3000节点。见 [诊断记录](../research/experiments/j5_act2_rgb_diagnosis_20261101.md)。

2026-11-01 SwanLab统一OWAC：全部运行归入OWAC，20项运行使用统一名称、说明和五类实验组；17个section合并为7个，保留286张图表，动作MSE置首。原生MOVE运行续写后网页仍读旧根，已保留迁移历史并从771步完整状态建立OWAC正式续训，回填原step指标；不重置训练权重/优化器/选择状态。训练及后续闭环统一写入OWAC。见 [迁移与监测记录](../research/experiments/j5_pi05_backend_adaptation_20261101.md)。

2026-11-01 J5监测V2：477步完整保存/native0后续训，当前已完成500步验证。动作训练迁入OWAC-Action-Training，四个固定分组并回填训练曲线；旧OWAC历史分组保留。32条Normal验证轨迹×8时刻、2272有效动作步，7维归一化采样动作MSE用于best/降LR；500步0.21244，best仍477步0.19956。辅助MAE、执行前5步MSE、分量RMSE、夹爪错误率；排除潜空间与轨迹尾部补齐。云端API核对通过，OOM0。见 [监测记录](../research/experiments/j5_pi05_backend_adaptation_20261101.md)。

2026-11-01 J5-ACT2：继承expert5000，冻结J5视觉，新增残差桥接、语言rank16及低LR专家Q/K；2636万可训练参数。双5090 micro128/global256完成3→5→7恢复/native0，完整300步闭环连通/native0但0/1成功。正式正常光已续接至10步，SwanLab在线；显存27.20GiB/卡、主机80.67GiB、OOM0。1000/3000/6000步开发闭环使用初态5..9，动作收益待测。旧评估已中断并保留：配对204例RAW0、RGB128。见 [后端适配记录](../research/experiments/j5_pi05_backend_adaptation_20261101.md)。

2026-11-01 J5执行诊断：Normal/ExtremeLow两条失败轨迹共14时刻复算及成功RGB首帧均复现原动作。已生成执行画面、固定PCA、目标query、动作专家attention；RAW并非完全不读视觉，但上下文query与指令attention明显不同于成功RGB参考。优先怀疑新模态后端适配不足，尚未因果定论；未训练。见 [执行诊断](../research/experiments/j5_pi05_execution_trace_20261101.md)。

2026-11-01 J5动作闭环：两阶段均6000步/native0，验证选择expert5000（MAE7=0.33939）。原自动评估被SwanLab resume参数错误阻断，已修复并实际启动。首个预定任务spatial0五照明各5初态，RAW0/25、官方RGB14/25；这是单任务初筛，不是40任务总体成功率。完整每角色1000场仍运行。见 [闭环记录](../research/experiments/j5_pi05_closed_loop_20261101.md)。

2026-10-31 J5-ACT 分阶段适配：新分支lzh/j5-action-staged，冻结J5 12818及官方π0.5，仅训练788480参数连接层；双5090 micro128/global256已从工程5完整恢复至正式6步，SwanLab在线。3→5完整恢复及177源动作采样/best保存通过，显存25.54GiB/卡、主机72.10GiB、OOM0。后续自动比较专家V/O/MLP rank8候选，再对验证胜出版本做RAW/RGB完整闭环；成功率尚未产生。见 [分阶段记录](../research/experiments/j5_pi05_staged_20261031.md)。

2026-10-31 J5-ACT LoRA：按用户要求在14041完整保存/native0停止，保留attempts/user-stop-14041。固定177源复核online/EMA均有退化；20例FP32采样误差未恢复，梯度能量81%集中动作专家LoRA。高学习率下后端共同适配失稳为主要假设，尚无因果对照；8k未保留是保存策略缺陷。未恢复训练，见 [停止与诊断](../research/experiments/j5_pi05_lora_diagnosis_20261031.md)。

2026-10-30 J5-ACT LoRA：全量训练112步完整保存/native0停止，继承online权重转为语言rank16/动作rank32 LoRA，RAW与动作接口继续训练。5861万可训练参数，双卡DDP micro64/累计batch256；正式续接已推进至140，约6.87秒/更新（约1.95倍加速），allocated18.23GiB/卡、OOM0。恢复/EMA验证/仿真连通通过，动作成功率待闭环。见 [LoRA记录](../research/experiments/j5_pi05_lora_20261030.md)。

2026-10-30 J5-ACT：用户授权将已保留的J5 12818视觉EMA接入官方π0.5-base，独立分支lzh/j5-pi05-full-finetune。29.62亿参数全量微调，LIBERO原分区/五照明、官方flow loss及batch256/30k配方；双5090已实际恢复并推进，SwanLab在线。RAW/RGB仿真连通检查通过；恢复/EMA切换问题已修复，4分钟内存峰值76.83GiB、OOM0。闭环成功率尚未完成，训练结束自动对照。见 [J5-ACT记录](../research/experiments/j5_pi05_full_20261030.md)。

2026-10-30：用户确认J5当前效果足够并要求停止；已在12818步完整保存/native0退出，独立encoder与完整恢复状态另存attempts/user-stop-12818。训练保持停止，未经新授权不恢复。见 research/experiments/j5_raw_relations_20261030.md。

2026-10-30 J5：继承J4最终纯RAW EMA继续30k；自身J3空间关系、实际token多样性、过去RAW软对应，无RGB/动作监督。双卡20→22恢复通过，内存约39GiB/OOM0；正式训练启动，SwanLab含J4对照图。质量未宣称成功，见 [J5记录](../research/experiments/j5_raw_relations_20261030.md)。

2026-10-30 J4：继承J3最终100k纯RAW EMA继续学习；新增同源曝光噪声配对与可见token一致性，无RGB/动作/外部权重。双卡工程20→22完整恢复通过，正式30k阶段已启动，SwanLab独立记录。保留逐帧读取及worker4/prefetch1。质量待实际图像评估，见 [J4记录](../research/experiments/j4_raw_exposure_20261030.md)。

2026-10-29 J3：用户授权保留DINO蒸馏并独立重建RAW自监督。新分支lzh/raw-self-supervised；旧框架保全086b77e与lzh/dino-distillation-preserved。无外部视觉权重、RGB目标或动作；辐射响应编码＋DINO/iBOT，双卡global192、100k计划已启动，SwanLab记录token/query。工程恢复通过不等于表征成功。见 research/experiments/j3_raw_ssl_20261029.md。

2026-10-29 用户要求暂停并保留：J2S已64366步完整保存、native0退出；latest.pt含完整恢复状态，encoder.pt为独立RAW编码器，均另存attempts/user-pause-64366。训练保持暂停，未经新授权不恢复。

2026-10-29 最新对比：固定48000完整ckpt与DINOv3全177源比较，RAW精确对应88.71%（33085为87.97%，DINO95.47%），检索862/1056；五照明空间指标均改善，训练继续原100k。见 [评估记录](../research/experiments/j2s_latest_comparison_20261029.md)。

2026-10-29 后续决策：用户授权自主评估；33085完整验证空间对应87.97%（DINO95.47%）、任务检索855/1056，近期验证仍改善，决定从33085完整恢复原100k。保留33085权重与原暂停收据。见 research/experiments/j2s_continue_decision_20261029.md。

2026-10-29：按用户要求，J2S已在33085步完整保存并暂停，训练native0；固定五照明RAW/DINO查询图已生成，未经新授权不恢复训练。见 research/experiments/j2s_pause_display_20261029.md。

2026-10-29 J2S性能修复：global32保持，micro2/accum8改为micro8/accum2、每rank8线程、移除逐微批标量同步；实测更新1.149→0.471秒（2.44倍），GPU均值约79%/80%。原内存增长主要是文件页缓存；新增分层监控并解除恢复时Adam CPU计数器持有的大checkpoint映射。1162/1709完整保存后恢复，100k预算不变。见 [性能记录](../research/experiments/j2s_performance_20261029.md)。

2026-10-29：用户授权自主选择完整RAW编码器路线；J2S采用训练期冻结DINOv3-L配对RGB监督，部署只输入RAW。427404532参数全部训练，双响应stem/局部因果门控/9-18-27层融合，LIBERO原拆分、双卡global32、目标100k。正式训练已产生完整step1并继续推进；未宣称达到DINOv3。J1保留82000停止，R1保留29071暂停。见 [J2S记录](../research/experiments/j2s_raw_semantic_20261029.md)。

2026-10-29：用户终止旧训练并要求完整RAW稠密语义encoder迭代。远端J1已82000步完整保存/native0暂停，R1保留29071暂停。全177验证源DINOv3-L对照已完成；第9层空间改善不代表语义改善。已按用户授权清理340.914GiB旧教师token缓存，关键权重/日志/视频全保留。空间修复候选已实现但未启动新训练；完整语义监督来源待明确。见 [J2记录](../research/experiments/j2_raw_encoder_20261029.md)。

2026-10-29 J1逐层诊断（81000步）：10条验证源上，无填充裁剪与零填充均显示block3/9改善、block15以后平均匹配误差增大；不是整个编码器均退化，也不是仅最终LN问题。详见research/experiments/j1_raw_jepa_20261028.md；未改训练配方。

2026-10-28 J1已授权并实际启动：公开π0.5视觉初始化＋新RAW stem/六帧因果门控＋JEPA式空间遮挡特征预测；无RAW教师、历史蒸馏权重或动作专家。LIBERO原1421/177/173拆分，100k更新目标，每500步验证，SwanLab全指标同步。R1保留暂停29,071。设计/监控入口见 [J1记录](../research/experiments/j1_raw_jepa_20261028.md)。

2026-10-27 R1已启动：按用户确认仅使用原LIBERO分区1421/177/173及五照明，35,693训练anchors；原始视觉初始化、重置stem/gate，428,579,952视觉参数仅真实动作flow-matching训练，3,201,953,824后端参数冻结。目标100,000优化器更新，双卡global16，实际更新及完整检查点已产生。源码/配置与实测见 `research/experiments/r1_raw_action_20261027.md`。

2026-10-27 R 子阶段名称代号已同步研究台、方案与提示词：R1 · ACT（纯动作学习）、R2 · LOOP（动作基线闭环）、R3 · CAUX（因果动作辅助）、R4 · DUAL（双路适应聚合）。

2026-10-27 最新路线为R：R1纯动作训练100k；R2闭环；R3因果flow-matching辅助头训练100k及闭环；R4快速细节/稳定证据适应性聚合训练100k及闭环。三次训练同源独立初始化，继承现有输入、评价协议及历史结果。GPT直接修改所有代码，不经过DeepSeek/Harness，此要求覆盖旧派工规则。当前R1待执行，路线见 `docs/plans/R_RAW_ACTION_ADAPTATION_2026-10-27.md`，提示词见 `docs/prompts/R/`。

2026-10-27 用户确认蒸馏技术路线走通，授权提交全部修改并创建“蒸馏成功”标签。已验收RAW因果学生→冻结后端闭环及正常光控制复刻；Low11/20教师、9/20学生和ExtremeLow双方0/20负结果保留。统筹台进入EXPLORE（下一阶段候选与工作单准备），不自动启动新训练/实验。交接见docs/plans/RAW_POST_DISTILLATION_EXPLORATION_2026-10-27.md。

2026-10-27 LOWLIGHT完整服务器评价：Normal教师/学生各20/20；Low教师11/20、学生9/20（退化2对、恢复0）；ExtremeLow均0/20。120场与6进程全部native0，无错误、缺场、重试或排除；60对原始数据及120原视频/60并排视频审计通过。现有学生含五照明训练，弱光控制保留不足，建议研究Normal RAW教师配对监督适配，但本轮训练0。最终所属输出/缓存约366.90GiB，380GiB约束保持。研究台与交接入口已更新，详见[完整报告](../research/experiments/raw_stem_siglip_20261020/lowlight-server-20261027.md)。

2026-10-27 12:08 LOWLIGHT服务器120场配对评价已启动；首6对Normal教师/学生各6/6，首对原始NPZ配对与执行动作检查通过，完整弱光结果待完成。启动所属占用约366.8GiB（含迁移依赖），380GiB约束不变；61相关测试及make check通过，训练0。

2026-10-27 LOWLIGHT新授权：服务器Normal/Low/ExtremeLow×20历史初态×教师/学生，共120场，仅评价。照明/角色隔离入口已实现并独立返工验证，56相关测试及make check通过；远端所属输出/缓存356.9GiB，依赖迁移计入380GiB上限；共享数据盘只读复用、不下载，当前尚未启动场次。见[服务器记录](../research/experiments/raw_stem_siglip_20261020/lowlight-server-20261027.md)。

2026-10-27 LOWLIGHT：完成资产、作者弱光参数与本机依赖只读核对；最小120场配对方案已交付，未授权运行、弱光闭环0、训练0。现有入口Normal硬编码及教师/学生混合加载需修改；远端所属输出/缓存占用未核清，380GiB约束保持。见[方案](../research/experiments/raw_stem_siglip_20261020/lowlight-plan-20261027.md)及configs/experiments/raw_distillation_lowlight_plan_v1.json。

2026-10-27：用户开启LOWLIGHT研究台阶段及复制入口。正常光阶段验收保持通过；当前为弱光方案与评价准备，尚未运行新实验。交接提示词见docs/prompts/RAW_DISTILLATION_LOWLIGHT.md。

2026-10-27 阶段验收：用户确认STEM（消息中的STEP）、ADAPT、CONTROL通过。全视觉头24遍/84768更新完成，最后学生与教师在正常光LIBERO goal/task6历史固定20初态均20/20；本阶段已基本复刻教师功能。ADAPT按实际全视觉头token蒸馏验收，未使用动作损失，语言/动作后端冻结。弱光待开展，未启动新训练或闭环。

2026-10-26：用户授权的最后学生84768权重仿真闭环已完成：LIBERO goal/task6正常光，复用历史固定20初态，教师20/20、学生20/20，配对退化0、恢复0；native0，40原始视频及20并排视频已保存。仅此单任务开发筛查，不外推全任务/弱光/实机；详见research/experiments/raw_stem_siglip_20261020/closed-loop-final84768-20261026.md。

2026-10-26：24遍/84768更新已native0完成，最后全验证token MSE0.270684。用户授权的最后权重离线动作对比已实际完成：LIBERO validation20源/80窗口，同RAW历史/语言/状态/噪声，共享冻结后端，教师重复差异0。正常光16窗口前5步专家MAE教师0.03835、学生0.05637；无闭环，不能宣称控制恢复。见[独立动作报告](../research/experiments/raw_stem_siglip_20261020/actions-final84768-20261026.md)。

# OWAC 当前状态

2026-10-25 15:09：用户授权继续扩大全量蒸馏，累计目标24遍/84,768更新。12遍42,384更新已native0完成，验证MSE0.3097147（8013窗口），末段仍下降。从42,384完整优化器442状态项/scheduler/RNG/采样游标恢复，实际推进42,390步/第13遍，SwanLab同步无错误。保持全视觉头、双卡global16、lr1e-5、原损失、每500步及epoch末验证，仅380GiB新增磁盘上限；无时间门槛、测试集或闭环。train-r12 PID68649/startticks14935363，执行源码733ff6a；上一6遍耗时11.94小时，新增12遍粗估24小时，不等于收敛保证。

2026-10-24 18:14：用户授权扩大全量蒸馏，累计目标由6遍扩至12遍（42,384更新）。前6遍21,192更新native0完成，验证MSE从18,500步0.381093降至21,192步0.367036；继续改善，不宣称收敛或控制恢复。从21,192完整Adam/scheduler/两rank RNG/采样游标恢复，已实际推进21,200步（第7遍）。保持全视觉头、global16=2×8、lr1e-5、原token损失、每500步及epoch末验证；仅380GiB新增磁盘上限。train-r11 PID29776/startticks7404486，源码8b5a7b2；SwanLab已同步21,200且无错误。未新增定时任务、测试集评估或闭环。

2026-10-22 19:25 当前实测：目标句柄修复后从250步完整Adam/scheduler/RNG/采样状态恢复，已实际推进390更新/6240呈现，越过此前300步故障位置。torchrun PID70886/PGID70886/startticks18696211及两rank存活，未产生native退出收据；两rank各107文件句柄，OOM计数仍1无新增。瞬时双卡利用率91%/100%，不代表长期平均。当前SwanLab：https://swanlab.cn/@Jeong/OWAC/runs/owac-all-bench-20261022-resume250 ，监控http://127.0.0.1:8880/。首遍验证尚未执行，控制未验证。仅380GiB新增磁盘上限；没有GPU时间/显存/利用率人为门槛，也未重建定时任务。


2026-10-22 19:04 后续故障：训练记录至300步后因目标mmap句柄未回收触发OSError24/native1，非OOM。已保留原始日志；完整状态在250步（含442参数Adam、2rank RNG/采样游标），不能称从300恢复。DeepSeek修复为目标行复制后立即关闭映射，不新增资源停止门槛；45相关功能测试及make check通过。将从250恢复，废弃分支260～300记录保留另列，尚待实际恢复日志。


2026-10-22 19:00 实际运行：全量RAW教师2163源/74129窗口已完成native0、测试目标0。学生在137完整状态重载后已实际推进210更新/3360呈现，双卡global16、全视觉头训练；只读RAW映射生效，无新增OOM。SwanLab实际同步210且无上传错误：https://swanlab.cn/@Jeong/OWAC/runs/owac-all-bench-20261022-attempt2 。训练仍在运行，首遍验证未完成，不宣称泛化/控制恢复。源码e658cd6，独立报告见research/experiments/raw_stem_siglip_20261020/all-bench-20261022-independent-review.md。仅380GiB磁盘上限，其他人为资源门槛移除；无定时任务重建。


2026-10-22 数据I/O进展：OOM修复后全视觉头已真实推进137更新并完整保存/native0暂停；同一DeepSeek实现未压缩NPZ内部RAW帧只读映射，避免整条episode匿名RAM复制，不改数据/模型/全局16/因果历史。Sol独立复核，累计43项相关测试及make check通过，按137优化器/采样器/RNG完整状态续训待实测；教师2163源/74129窗口复用，测试目标0。

2026-10-22 全量实验实时修订：教师2163源/74129窗口已全部生成native0。首次学生作业至少10更新后被容器116GiB配额OOM杀死，证据/初始模型/日志完整归档，不重算教师、不声称恢复未保存优化器。已由同一DeepSeek修复RAW缓存配置接线并改用每卡16源工作缓存，保持global16及完整历史，无额外内存/显存/时间停止门槛；第1实际更新即完整保存。Sol独立复核，累计42相关功能测试及make check通过，修复后作业从相同初始化重新启动待实测。

2026-10-22 最新用户授权：在现有SSH服务器新开全量RawVLA-Bench蒸馏实验，不改学生/教师/损失算法，训练整个学生视觉头。只保留新增硬盘380GiB上限；用户明确撤销GPU时间、显存、利用率等人为门槛及强制停止要求。全量只读共享数据已清点：2403轨迹（LIBERO1771、RoboTwin632）、408075帧、5照明；源轨迹拆分1923/240/240，先训练3遍训练集，global16双卡各8。旧实验/预算/结果保留，新实验不受旧3116更新上限约束；无动作损失/后端训练/闭环/实机授权。当前全量数据/双三相机接口由同一DeepSeek V41-Flash Max实现并经Sol独立复核，40项功能测试及make check通过；双卡教师目标已真实生成中，学生仍0更新，已配置目标完成后自动衔接3遍训练，无终止失败自动重试。监控http://127.0.0.1:8880/，SwanLab新run owac-all-bench-20261022。不能宣称充分训练或控制恢复。实验根outputs/raw-distillation-all-bench-20261022，临时tokens缓存/root/autodl-tmp/owac-all-bench-20261022。


2026-10-22 当前阶段已完成：全视觉头428579952参数累计3116新更新/49856anchors，完整checkpoint_final_003116，native0；此前阶段合计达到原5000更新等价上限。global16=双卡各8真并行，accum1；每步中位0.454秒，吞吐约旧串行4.5倍。当前蒸馏训练/教师缓存/状态重载/导出/配对评估链路的hash与无意义扫描/冻结审计/数值门槛已清理并同步服务器，123功能测试+makecheck通过。SwanLab私密OWAC已完整同步并结束：https://swanlab.cn/@Jeong/OWAC/runs/owac-fullvisual-shared-stage 。最终train loss0.07365，最佳val1.00053@1000、最终val1.00837，未作test/动作诊断/闭环，不宣称泛化或控制恢复。无预算扩展或定时任务重建。详见research/experiments/raw_stem_siglip_20261020/batch8-cleanup-20261022.md。

2026-10-22 当前运行已切换真batch8：从858完整状态恢复859并持续超过1250步，全局16=2rank×8、accum1，FP32权重/Adam+BF16 autocast。移除每步全参数/梯度审计及RAW范围/finite全扫描，status/GPU gather每10步。稳定host每步中位0.451秒（此前约2.04秒），25秒GPU均值85%/84%，不是收敛/控制结论。SwanLab私密OWAC已上线且持续回填/更新：https://swanlab.cn/@Jeong/OWAC/runs/owac-fullvisual-shared-stage 。导出/teacher-cache冷路径hash残留继续清理，本轮无新GPU benchmark/准入门槛。原3116新阶段/累计5000更新等价和资源预算不变，旧定时任务不恢复。

2026-10-22 用户明确要求清理蒸馏整条代码链的无意义防御性编程：取消Sol附加的数值阻断门槛，停止反复GPU对比测试；移除哈希、重复张量/参数/梯度扫描与冗余审计，不另造预检门槛。保持训练数学、因果/reset、完整恢复和资源上限。当前训练安全暂停于858更新，下一步使用双卡每rank8真batch、全局16和原BF16恢复；SwanLab私密OWAC项目监控并行接入。未授权其他研究阶段或预算扩展。

2026-10-22 最新批处理授权：保持全局batch16，优先双卡每rank8样本真正并行。旧串行路径已在858更新完整保存并native0退出；优化器、采样器、累计预算保留。DeepSeek V41-Flash Max已实现并通过8项工程测试和make check；真实双卡零更新数值/梯度/显存/吞吐验收进行中，未通过前不宣称切换完成。无新闭环、动作损失或预算扩展。

2026-10-22 哈希清理及性能修复已实测验收：当前训练执行路径不再计算/校验哈希，保留形状/有限值/相机/因果/reset及恢复计数检查。55步完整暂停后，双卡55→56更新/保存和重载验收均通过，连续训练已恢复并超过94步；采样器/优化器/预算未重置。RAW10源RAM缓存、静态packet每rank≤1.5GB缓存启用。稳定每步中位18秒→约2秒，两卡35秒平均利用率约8%/24%→44%/47%；不宣称持续满载。监控http://127.0.0.1:8879/。详见research/experiments/raw_stem_siglip_20261020/throughput-nohash-20261022.md。


2026-10-22 用户最新明确要求彻底移除本轮蒸馏链路中的所有哈希计算及校验。该明确指令覆盖历史资产/权重/状态/缓存/检查点及源码启动哈希要求；保留直接张量形状/有限值、相机/时间/分区/reset和优化器/采样器恢复检查。历史记录不重写，旧hash字段仅兼容读取而不校验。训练为性能修复已在新55步/880anchors完整保存并正常退出；从55恢复，不重算、不重置预算。实现由同一DeepSeek V41-Flash Max负责，Sol独立验收；未完成清理前不宣称已移除。


2026-10-22 15:10 远端整视觉头连续蒸馏已实际运行：从独立验收的全新step1完整状态恢复，实测已到新阶段4步/64训练anchors；Sol核对torchrun PID55499/startticks17190994/PGID55499及两rank进程，native exit仍为unknown（存活中）。严格双卡重载native0且0额外更新通过。仅共享RAW→新教师targets→全学生视觉头428,579,952参数，固定8/2/2、test不参与；教师/语言动作后端冻结。新监控http://127.0.0.1:8879/。最多3116新更新及剩余资源上限，既有定时自动化未恢复；未证明收敛或控制恢复。


2026-10-22 共享RAW整视觉头最新进展：远端直接只读共享task6 Normal12源，固定8/2/2；全新204行RAW教师目标已逐行及状态链独立验收，test0行。双RTX5090真实零更新梯度预检、1次全局16anchors优化更新及完整检查点独立验收通过；442视觉状态张量均实际更新，可训练428,579,952参数，教师与语言/动作后端冻结。重载检查进行中，尚未宣称长跑或收敛；新监控http://127.0.0.1:8879/，旧8878为停止历史。见research/experiments/raw_stem_siglip_20261020/full-visual-shared-20261022-independent-review.md。


2026-10-22 数据来源最新更正：用户明确要求“共享盘直接读RAW，在远端重新生成匹配的教师目标，再训练整个学生视觉头”。旧50条RAW及旧token缓存传输已停止，不作为新阶段训练数据；共享根目录/root/autodl-pub/datasets/rawvla按只读处理。task6正常光实有12条，采用固定源示范8/2/2拆分，仅train+validation生成新的RAW教师targets，test不参与训练或选型；不加入其他照明条件。整视觉头新阶段仍从0步开始，继承实际π0.5视觉权重、重置stem/gate；教师及动作后端冻结，资源总预算不扩展。工程和真实GPU预检进行中，尚无新优化更新。

2026-10-22 最新用户授权：本机stem-only蒸馏已按用户要求停止于1872步，定时跟进已删除。新阶段迁移至用户指定SSH实例101.71.76.104:27786，使用两张RTX5090；用户明确不从旧学生检查点续训，重新初始化RAW stem/gate，仍继承当前π0.5策略的位置编码、SigLIP与原投影，并训练整个学生视觉头（428,579,952参数）。教师与语言／动作后端冻结，仅token蒸馏，无动作损失／闭环／弱光／实机扩展。新建阶段与优化器，旧实验保留；当前工程实现、传输与预检进行中，不能称已启动或充分收敛。暂沿用原总预算剩余额度：最多3116次新全局更新／49856 anchors、140841.2632 aggregate GPU秒，双卡按两卡合计，新增训练输出剩余15396584412字节。实际资产／派工／授权记录见outputs/raw-distillation-full-vision-20261022/。本条取代历史stem-only当前范围，不改写其事实。

2026-10-21 最新授权：用户明确“可以开始充分蒸馏，并给我一个简单的监控页面来实时查看进度和效果。”本阶段仅C单臂RAW→RAW投影后视觉token蒸馏，从真实C50续训，累计最多5000更新／80000 anchors；新阶段最多64 GPU小时／20GiB。优化器、scheduler、RNG、sampler及累计50步必须恢复；只更新现有stem＋gate，继承视觉与后端冻结。按固定验证集token MSE＋0.1cosine选型，动作误差仅诊断；无B/D扩展、视觉解冻、闭环或弱光授权。当前C51真实首步已独立验收，连续蒸馏已启动并真实恢复到后续更新；实时进度见[本地监控](http://127.0.0.1:8878/)，每5秒刷新。验证token损失每250步检查；后台每20分钟核查，正常无异常时保持安静。闭环0，控制恢复未证明。派工与运行收据：`/home/jeong/zeno/owac/repo/outputs/raw-distillation-c-full-20261021-123313`。此条取代此前“暂不启动”的当前状态，历史结果不变。

2026-10-21 后续方向：只继续C单臂RAW→RAW视觉token充分蒸馏，不再默认开展B/D多轮对照；用户当前明确暂不启动、先审框架。教师／学生实际结构及候选预算见[框架说明](../docs/plans/RAW_TEACHER_STUDENT_FRAMEWORK_2026-10-21.md)。新训练authority未签发；投影后token选型尚待实现，既有pilot动作选型记录保留。



2026-10-21 研究统筹台全部 prompt 的 DeepSeek 编程默认改为完全权限（`danger-full-access`）与不询问审批（`never`）；可见 Web 会话、编程／闭环 key 分离继续沿用。新建／续接后由执行者核对实际权限并切换旧会话，不要求用户逐项批准；既定任务、预算与阶段状态保持。

2026-10-21 研究统筹台提示词统一改为可见 DeepSeek Harness Web 编程：工作单、回复、工具调用与返工均在用户可见专用对话中完成，不使用 headless 编程派工。编程／闭环 key 继续分离；任务专用 Web 凭据文件与共用人工会话隔离。此次只更新工作方式与提示词。
2026-10-21 **B/C50 pilot已完成并独立验收：token对齐改善，前5步动作误差回退，闭环未测**。两臂各50真实更新／800 anchors，只更新14,454,784个stem＋门控参数；继承视觉、教师与后端冻结。选型B1／C0，纯学生重载通过；模型作业约1.54 GPU小时、新输出约10.35GiB。C原监督终端143、原生子进程退出码不可得，真实50权重与日志已审计；未重训。当前全部本轮模型作业结束，停止于已用完的50上限；D／视觉解冻／闭环／弱光／实机未启动。见[本轮独立报告](../research/experiments/raw_stem_siglip_20261020/pilot-20261021-independent-review.md)。


2026-10-20 DeepSeek 凭据按用途分离：所有 prompt 明确编程使用 `DEEPSEEK_CODING_API_KEY`、在线闭环使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`；本机私有文件由按角色启动入口读取，编码 Harness 显式绑定 coding 并禁用托管凭据回退。只更新配置与提示词，当前阶段与实验预算不变。见[使用说明](deepseek-credentials.md)。
2026-10-20 **OWAC-D1.1 ALIGN工程与真实零更新预检通过；STEM待新授权**。因果6帧／相机独立状态、同源SigLIP继承、完整前缀缓存和原生动作梯度、纯学生checkpoint0独立重载均已实际检查；训练40源＋验证5源共969条教师缓存。164项针对性测试通过，2项需toy AdamW更新的旧测试排除、0跳过，make check通过。真实学生0更新／0闭环，控制未恢复。建议下一步B/C各≤50更新、合计≤4GPU小时／20GiB；当前未签发authority。见[独立报告](../research/experiments/raw_stem_siglip_20261020/align-independent-review.md)及[命令/资源](../research/experiments/raw_stem_siglip_20261020/ALIGN_RUNBOOK.md)。

2026-10-20 限定教师阶段已验收：[独立报告](../research/experiments/raw_stem_siglip_20261020/teacher-independent-review.md)。Normal task6共同20新初态：参照20/20、RAW教师20/20，仅开发筛查。选定前端初始化0；实际155前端更新／1240 anchors未带来已证明收益，未作为选定权重。验证9成功＋1人为中断保留完整N10；50回放／50策略场次已用完。学生0更新／0闭环，尚未恢复；旧N01／N02／N04仍暂停，不授权学生／弱光／实机扩展。


2026-10-20 授权前教师取得补查记录：官方6公开分支／main32历史提交与公开主页未找到训练后的RawVLA前端；完整源码已取得并独立合并配置，后端已恢复但不等于完整教师。已写[限定单任务教师复现提案](plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md)，拟仅训前端、累计≤2000更新，执行待明确授权；0优化、0新闭环。保留因果burst／递归，不固定单帧。


2026-10-20 授权前工程交付记录：每帧 RAW stem／SigLIP 蒸馏基线与受授权门控的训练、纯 artifact 评估入口已可审阅；68 项针对性测试通过，实际同源后端恢复及合成输入数值／梯度核对通过。新增优化更新0、闭环0，控制未恢复。完整 RAW 教师权重、真实配对数据和时序训练端到端尚未验收；不固定最终策略为单帧。见 [独立验收](../research/experiments/raw_stem_siglip_20261020/independent-review.md)。


2026-10-20 当前实施路线收敛到[RAW stem＋继承 SigLIP＋蒸馏](plans/RAW_STEM_SIGLIP_DISTILLATION_2026-10-20.md)。用户认可方案，执行 prompt 与采用决议已落库；统筹台默认展示此主线、完整提示词和可点击的分支／收敛历史。当前版本 OWAC-D1.1：主教师对外接收 RAW，内部可经 Neural ISP 成像；教师和学生不固定单帧，以控制效果、稳定性和成本选择因果历史／状态。学生部署直接 RAW→tokens，首轮后端冻结。工程实现进行中，本次修订未训练教师／学生或运行新闭环。作者模型实现已公开，训练后 RawVLA 前端／完整教师公开权重尚未找到，见[资源核查](../research/reviews/2026-10-20_rawvla_public_teacher_assets.md)。

2026-10-19 的[ConvNeXt 单任务视觉替换方案](../research/experiments/2026-10-19_visual_replacement_recovery.md)及其单帧前端转为诊断历史；旧后端 LoRA 配方不沿用。50 条真实示范、每适配臂累计最多 5,000 更新作为新方案首轮诊断规模，不能视为已执行或充分收敛。配套 RawVLA runtime、LIBERO 策略／统计／tokenizer、真实配对示范及初态仍须核验；DROID 权重不能替代。

2026-10-19 本地 worktree 清理完成：三个候选 checkout 均已移除，当前只保留 `/home/jeong/zeno/owac/repo/` 的 `main`，后续直接在此开发。清理前重新核验提交祖先关系、工作区、91,270 个证据文件与归档哈希；另保全 N02 本地第三方源码。历史分支、代码快照和主仓库证据归档保留，见[清理收据](../research/archives/2026-10-19_worktree_consolidation/worktree-cleanup.json)。

2026-10-19：用户暂停 N01、N02、N04，要求三个 worktree 合并回主线。远程服务器已不可达；本轮只合并、保全本地证据并运行本地代码检查。当前入口为 `repo/` 的本地 `main`，不自动恢复训练、诊断训练或下一阶段。

[合并与证据索引](../research/archives/2026-10-19_worktree_consolidation/README.md)记录原分支/提交、接口冲突处理、本地归档与远端缺口。四处原始状态快照分别保存在该目录的 `baseline/`、`n01/`、`n02/`、`n04/`，历史状态不作为当前作业状态。

| 方向 | 最后可查本地证据 | 当前结论 |
| --- | --- | --- |
| N01 v1.2 P3 | [停止记录](../research/experiments/n01_v1_2/P3_training_stopped_20261019.md)、[动作诊断](../research/experiments/n01_v1_2/P3_action_diagnosis_20261019.md)、[代码同步](../research/experiments/n01_v1_2/P3_code_sync_20261019.md) | RAW/RGB 各 0/12；正常光同场景作者 4/4、两臂各 0/4。专家观测动作已明显偏离，优先定位视觉—动作适配。未通过完整研究验收。 |
| N02 v1.2 P2/r17 | [r11 闭环收据](../research/experiments/n02_v1_2/closed_loop_r11_20261019/receipt.json)、[r17 结论](../research/experiments/n02_v1_2/residual_joint_r17_20261019/conclusions.json)、[r17 审计](../research/experiments/n02_v1_2/residual_joint_r17_20261019/) | r11 0/3；r17 新 Normal 19 条专家合格示范、423 窗口的小样本动作读出有冻结 RAW 收益，局部联合更新没有额外总体收益。有效 6144＋保留失败 1024＝实际 7168 更新；π 更新 0、学习策略闭环 0，不转为机器人成功。 |
| N04 v1.1 S2 | [v4.2 修复](../research/experiments/2026-10-16_n04_aux_repair_v4.md)、[v5 故障与修复](../research/experiments/2026-10-19_n04_stability_repair_v5.md)、[代码同步](../research/experiments/2026-10-19_n04_server_code_sync.md) | 历史开发闭环 0/100；v4.2 对应越界/目标消失、潜在量纲增长与 RoboTwin 退出保留。v5 短程零更新/优化/重载通过，只支持工程修复，长期稳定与任务收益未验收。 |

N04 v5 本地最后运行收据保存到 step10；这不是本轮确认的最终停止步。N01/N02 的已完成或停止回执同样以其各自记录时间为准。现在不能查询远端最终进程、日志、checkpoint 或存储状态，不能宣称远端独有权重已恢复或已丢失。

完整 Default π0.5 基线仍为 [2026-10-14 全量报告](../research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)：LIBERO 5820/10000（58.20%）、RoboTwin 508/3250（15.63%）；附加原条件 632 条另计。完整分母已取得，论文绝对数字未复现。原协议标签和历史负结果保持不动。

研究统筹台现从主 checkout 读取三条路线，当前节点为“暂停与本地证据”；旧提示词可阅读，但不会作为当前执行包发出。未来是否统一架构或恢复实验由新的研究决定确定。
