# J4：继承 J3 的纯 RAW 曝光一致性续学

## 范围

用户授权迭代与服务器训练，并明确要求复用已有能力。分支 `lzh/raw-exposure-consistency`，基于 `c0179c8`。保留 J3 100000 步完整权重和全部历史，旧蒸馏路线不修改。J4 从 J3 最终 EMA 的 encoder 和两个原型头初始化 online/EMA；不是随机重训，不加载外部视觉权重。Adam 动量和学习率日程重新开始，以适应新目标；J4 内部断点则完整恢复 Adam、两 rank RNG、采样位置。没有 RGB 输入、RGB 重建/特征目标、动作或语言监督。

## 唯一新增学习任务

从同一个 Normal RAW 帧产生原观测与低曝光噪声观测，在原始分辨率增加噪声，然后使用完全相同的裁剪、翻转、缩放和温和增益。EMA 看原观测，online 看退化观测。保留 DINO/iBOT/KoLeo，对未 mask 的实际 backbone 输出 token 增加 `mean(1-cosine)`，权重 2。遮挡位置继续通过 iBOT 学习对应目标，不对它强加直接逐 token 重建。

无 slot、区域伪标签、Gram anchor、RGB 预训练网络、重建解码器或新增模型参数。最终部署仍然只需现有 RAW encoder。

## 数据与噪声

域仍均衡。50% 抽样选择对应域 Normal，另 50% 保持原五照明均衡，故期望 Normal 60%，其他各 10%。只有 Normal 生成变暗视图；真实 Low/ExtremeLow/Over/ExtremeOver 保持原值，继续原自监督。源划分不变，test 不读取。

曝光降低量均匀取 `[0, m]` EV，m 从 2 在前 3000 新更新内升至 6，随后固定；这是训练课程配置，不是质量通过门槛。8-bit LIBERO 保持量化；RoboTwin 保持浮点单位。

对已有噪声测量 x、增益 a=2^-EV，附加噪声方差为：

`a(1-a) * shot * max(x, 0) + (1-a²) * read_var`

来自 Var(x)=shot*s+read_var 下对缩放后已有噪声的补偿，避免简单再叠加全部噪声。仍是近似增强：源噪声相关性、截断偏差、量化和环境光变化均不等同真实重新曝光，不宣称合成图等价真实 ExtremeLow。

参数来源：
- LIBERO 作者 `experiments/simulation/libero/rawvla_bench/raw.py`：shot=4e-4，read_var=1e-5；读噪声字段是方差，非标准差。加载单位 uint8/255，上限 1。
- RoboTwin 现有 Normal RAW NPZ `metadata_json`：shot=2.5e-5、read_var=3.90625e-8、white_level=3.5。维持 three_channel_linear_float32 单位；不是相机实测标定。
- 训练前不读取配对 RGB。元数据中存在 RGB 路径不代表使用 RGB。

## 配置与计算

ViT-S/16、2×256 global、8×96 local、双 5090 global192、FP32 参数/BF16 autocast。新增阶段 30000 更新，lr5e-5、1000 warmup、余弦至2e-6；EMA .996→1，teacher temp .07，取消已经完成的原型冻结。旧训练 100000 步不计入新日程。

保留精确逐帧 pread、局部 fadvise、worker4/rank、prefetch1、线程8/rank，不读整条 episode、不创建教师缓存、不全局清缓存。配对图增加 CPU 预取体积，但不增加学生 forward 数；online 原有 global forward 改看对应退化视图。

## 测试与运行证据

- 7 项针对性测试通过，覆盖读取值一致、恢复采样、曝光噪声计算、配对几何与可复现性、实际 token 梯度和目标 stop-gradient。
- 新改动 ruff 通过；`make check` 仍有既有 213 项 lint 错误，日志在 `outputs/j4-raw-exposure-20261030/make-check.log`，未把失败说成通过。
- 服务器双卡真实 20 步 native0，完整恢复至22步 native0；工程输出独立保留，不计入正式曲线。
- 工程第20步约419.8源图/秒、每步0.457秒、allocated峰值22.75GiB/rank、cgroup总34.86GiB、共享6.51GiB、OOM0。这是短窗实测，不保证未来永不OOM。
- 正式阶段0评估的空间std0.0397782、图间cos0.3193171与J3最终相同，确认延续同一EMA表征。没有重新初始化编码器。

## 评估

沿用20固定验证源、两个域五照明的原始RAW/query/PCA图。增加4个Normal验证源的同源曝光探针（0/-2/-4/-6EV），固定噪声随机数；每次同时测特征余弦与去掉上/左一个patch后的位置偏移匹配。后者输出15×15个token，目标是原16×16网格偏移一个位置的坐标，不把相同位置匹配当作内容对应。它仍不是物体分割指标。

阶段0，每域2图：

| 域 | -2EV偏移匹配 | -4EV偏移匹配 | -6EV偏移匹配 |
|---|---:|---:|---:|
| LIBERO | 59.11% | 32.00% | 13.78% |
| RoboTwin | 70.00% | 46.00% | 13.33% |

每1000更新保存完整checkpoint、encoder、20张真实图与4张曝光探针图，上传SwanLab。PCA颜色不是质量评分，合成曝光匹配改善也不能替代真实弱光物体区分。

## 位置

- SwanLab: https://swanlab.cn/@Jeong/OWAC/runs/owac-j4-raw-exposure-20261030
- 远端：`/root/owac/repo/outputs/j4-raw-exposure-20261030/`
- 启动源码：该目录 `source/src`，独立于 J3 快照。
- 配置：`configs/experiments/j4_raw_exposure.json`

当前属于已启动续学，尚未宣称最终表征成功。不根据 loss 下降、GPU利用率或工程通过替代质量判断。
