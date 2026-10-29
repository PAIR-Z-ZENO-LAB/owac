# J2S 暂停与当前效果展示（2026-10-29）

用户要求暂停并按既定方案展示当前效果。SIGUSR1触发更新边界保存，33085步status=paused、checkpoint complete/encoder_complete=true、train native_exit=0。完整Adam/scheduler/双rank RNG/采样游标与独立encoder保存于远端outputs/j2s-raw-semantic-20261029。保持暂停，不自动恢复。

通过raw_semantic_visualize从33085导出独立encoder提取，使用预定每照明首个验证源、中间endpoint、camera0及固定查询(4,4)/(8,8)/(12,12)。五照明全保留；统一cosine[-1,1]，无逐图拉伸。RAW六帧224、DINO正常光RGB单帧256，均16×16网格；RAW预览仅gamma，与模型输入区分。不是物体标签查询，也不是开放域能力验收。

图和原始affinities.npz、manifest.json位于本地及远端outputs/j2s-query-paused-20261029。训练退出与checkpoint收据同步至本地该目录。可复现命令：PYTHONPATH=src /root/owac/runtime-full-vision/bin/python -m owac.runtime.raw_semantic_visualize --encoder outputs/j2s-raw-semantic-20261029/encoder.pt --output outputs/j2s-query-paused-20261029 --device cuda:0。

最近完整验证在33000步，177源531窗口/1062视图；dense cosine loss 0.0745064（平均cosine 0.925494），Gram MSE 0.00230419，global loss 0.0179947。正常光平均cosine 0.927558，极低光0.920147。这些是教师对齐指标，不是识别准确率。未重跑完整空间对应或跨源任务检索，本轮只作用户请求的固定展示。

正常光图的机器人、桌面整体相似度布局已接近教师；仍有局部扩散和边界差异。极低光样本也保留相近布局，但教师输入是正常光RGB，不能据此声称恢复不可观测信息或等同DINOv3开放域能力。
