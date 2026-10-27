# LOWLIGHT 最小配对评价方案（2026-10-27）

状态：方案交付，新增弱光闭环0、训练0；尚无弱光成功率或视频。当前消息授权核对和方案准备，没有指定实际实验档位、场次数；以下120场为待确认的具体范围，不是启动收据。配置为研究协议，不能直接交给现有CLI执行。

## 当前资产与环境核对

基线源码HEAD：5afc74996d727f285f7db3b31c4e928b74a448ef；开始时git工作区干净。正常光原始报告为closed-loop-final84768-20261026.md：最后学生84768和完整RAW教师各20/20，配对退化/恢复均0。学生全量训练含ExtremeLow、Low、Normal、Over、ExtremeOver五照明，本轮测已有混合照明学生的弱光控制保留，不是仅正常光训练后的零样本迁移。

本机已核对学生student-final84768-eval.pt存在（约1.6GiB），教师前端teacher-validated-normal-task6-init0.pt存在（约3.2MiB）。完整教师还包含原配置中的继承视觉塔、投影和冻结动作后端；前端文件本身不等于完整模型。教师是本地作者架构init0，不是作者发布的训练后ISP。后端、tokenizer和统计路径沿用历史config.json；本轮未加载全模型或验证其张量内容。

历史manifest的20个test条目为test-101100至test-101119，states文件存在。runtime/bin/python在设置历史LIBERO_CONFIG_PATH、MUJOCO_GL=osmesa、PYOPENGL_PLATFORM=osmesa后，torch、mujoco、libero.libero、imageio、OpenGL均导入通过；不设置LIBERO路径会触发交互输入并EOF，不能遗漏该启动环境。nvidia-smi可见一张RTX5090（32607MiB）。本轮没有新建仿真环境、reset、渲染或模型前向，实际新闭环环境验收仍待执行。

磁盘只读核对：本机卷1.8TiB、已用约1.1TiB、可用约671GiB；outputs约87GiB；本机all-bench实验目录1,797,955,871字节。远端/root/autodl-tmp路径本机无权访问，远端持久输出及缓存当前占用未核对。87GiB不是本实验新增量，不能用380减87计算预算。运行前需合计实际所属持久输出、临时token缓存及新视频/NPZ，延续380GiB总约束，不重置额度；无GPU时间、显存、利用率门槛。不恢复旧作业、不下载。

## 最小建议范围与成像协议

建议同一goal/task6、历史20初态，Normal/Low/ExtremeLow三条件，各完整教师20场＋纯学生20场，共120场、60对。Normal重跑用于新隔离入口的一致性对照；历史20/20另列，不混入新分母。采用历史初态属于开发筛查，不是独立留出泛化测试。若授权范围调整，运行前固定清单，不能观察结果后挑初态或checkpoint。

作者固定源码experiments/simulation/libero/rawvla_bench/lighting.py定义Normal EV[-0.5,0.5]、Low[-4,-2.5]、ExtremeLow[-5.5,-4]，共同灯架common_agentview_key，基础强度0.02，弱光table_flood为0、sensor_saturation_ev为0。同episode使用相同两个均匀随机数：第一个映射各档EV范围，第二个sample_unit传入作者lighting_parameters_for_sample；两角色共享实际参数并记录，改变的是仿真照明，额外exposure_ev固定0。不能通过事后压暗视频或仅在配置写Low冒充弱光实验。

raw.py实际默认shot_noise=4e-4、read_noise=1e-5；方差为shot_noise×normalized_raw＋read_noise，后者是方差项，不是标准差。复用noise_seed=seed XOR 0x5A17及作者按frame_index/view_index生成的随机流。RGB8逆处理→RAW10噪声→uint8三通道pseudo-RAW→/255，非原生Bayer；不套用其他旧yaml的噪声值。保留现有固定逆处理增益，不做额外曝光补偿。作者噪声实现内部stable_uint32使用哈希构造随机seed，这是既有随机机制而非新增资产hash校验，本轮不改随机数学。

同任务、初态、语言、相机、渲染、观测处理、每次重规划noise完全配对。复用paired_noise(seed, episode_id, decision_step)，跨照明也共享。独立环境reset和每相机教师状态/学生递归状态清空；10步等待、6帧真实因果历史、horizon10/flow10/replan5、最多300策略步、原生quantile动作反归一化、check_success。首个RAW burst和机器人状态两角色一致；动作分叉后只有随机流/协议配对，不声称后续RAW数值相同。教师FP32、学生FP32权重/BF16前向差异如实披露。

## 源码复用与尚需实现

复用src/owac/runtime/raw_student_closed_loop.py、src/owac/simulation/raw_student_closed_loop.py、rawvla_teacher_controller.py及历史launch.py环境。现有LIBEROEnvAdapter._normal_lighting硬编码Normal；run没有读取弱光配置，还会同时加载两角色模型。因此当前入口不能直接作为本协议弱光或隔离纯学生入口。

运行前最小改动：显式注入照明domain及已固定参数；按role独立进程加载模型，学生进程不加载教师、不读取教师缓存或Normal观测；共享只读冻结动作后端权重。逐场保存实际照明、噪声参数、源初态、首burst、动作采样noise、动作chunk与执行动作、状态、步数、success、退出状态和视频。沿用attempts.jsonl/NPZ/MP4与并排视频账本，每个照明独立输出目录。主编码沿用现有可见DeepSeek Harness V41-Flash Max流程，负责人独立验收；本轮未派发编码或启动模型作业。

## 判读和完整交付

每条件分别报告教师/学生成功数/预定20、Wilson区间、完成/错误/中断数与完整场次清单；错误不可静默删分母，未完成时另列保守成功/20与已完成成功率。教师成功→学生失败为配对退化，教师失败→学生成功为配对恢复，另报双方成功/双方失败，汇总仅用完整配对，缺对单列。各角色另报Normal→弱光的丢失/恢复和完成步数。Normal均成功时可报弱光成功/Normal成功的条件保留比例；否则同时披露配对转移，避免仅比两个比例。

每场保留两段原视频和左教师右学生并排视频，全部60对可浏览，较短片停留末帧，不只挑代表片。Normal特权画面仅可用于事后人工分析且不得进入学生输入；预览提亮需标注、原视频保留。失败按可见行为记录定位/抓取/放置/超时及实现错误，信息丢失或后端不匹配只能列待验证原因，不由视频直接作因果结论。

若学生接近教师，只能说明当前单任务保留，不自动训练；若教师成功而学生失败，先核对隔离入口及成像/历史一致性，再提出适配。双方失败不支持学生架构更差；不同曝光/多帧新增测量带来的信息收益不得归因于架构。

## 仅在结果支持后制定适配

目前没有训练范围或更新数授权，不预填旧5000步。候选先限制stem、gate、时序聚合参数；若证据支持扩大到位置编码/SigLIP/归一化/投影，必须另列实际参数范围与训练容量，语言/动作后端始终冻结。正常光教师目标必须对同一时刻正常光RAW历史实际运行完整RAW教师生成；正常光特权仅限训练监督，不能把RGB视觉塔或学生token当成RAW教师目标。跨光照目标需同轨迹/同状态/相机/时间配对，不能把两条动作分叉的在线轨迹按步号硬配。

按源轨迹/场景/采集会话先拆train/validation/test，所有照明版本与相邻窗口同组；沿用已有1923/240/240源拆分时要核对清单，禁止测试目标参与训练或选型。当前20历史闭环初态保留为开发集，最终独立验证另定新初态，不宣称它未被反复使用。适配后固定checkpoint再做原学生/适配学生/教师同条件评价；纯学生测试禁止教师、教师缓存和正常光输入。

目前缺口：弱光接口、role隔离、运行前远端磁盘核算、新环境真实reset/渲染、全部弱光成功率/失败证据/视频。没有弱光实测，不能决定是否需要适配。
