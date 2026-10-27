# OWAC 蒸馏成功之后：探索准备

进入/home/jeong/zeno/owac/repo，读取AGENTS.md、README.md、docs/status.md，保留现有修改。

用户已验收蒸馏路线，标签为“蒸馏成功”。先读docs/plans/RAW_POST_DISTILLATION_EXPLORATION_2026-10-27.md和research/experiments/raw_stem_siglip_20261020/lowlight-server-20261027.md。学生84768已含五照明训练；Normal教师/学生20/20，Low11/20与9/20，ExtremeLow均0/20。不要重跑已完成120场或把弱光鲁棒性标为通过。

准备下一阶段候选比较和具体工作单：可以先复用现有证据作只读分析，不预设配对蒸馏或adapter为唯一方向；区分信息、表征、时序机制与动作读出证据。若考虑Normal RAW教师目标，必须实际经完整RAW教师生成，纯学生测试禁止教师/缓存/Normal特权输入。保持语言/动作后端冻结、源轨迹拆分和独立reset/因果历史。

本入口不是新目标生成、训练、闭环、下载或实机动作授权。按当前会话的具体授权执行；没有实验范围时先交付可审阅方案。新工作先核对现有占用，保留380GiB新增磁盘总约束，不恢复旧作业、旧5000步或GPU时间/显存/利用率门槛。主要编码继续沿用现有可见DeepSeek Harness V4.1-Flash/Max，负责人独立验收。
