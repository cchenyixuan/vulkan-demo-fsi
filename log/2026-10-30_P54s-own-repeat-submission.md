# 2026-10-30 P54s_own 同协议重复（P54s_own2）提交：54 m³ 的实现间噪声底

用户决定（2026-10-30）："加一个重复，提交到 mahuika"。

## 原来的代码

P54s_own 对 P54s_mean 的饥饿时间分数差 +6.5 点（`log/2026-10-30_P54s-results.md`），只有运行内子样本的离散
（两个时间半段、两组 lifeline，差在 6.1 到 7.1 点），没有独立实现。30 L 的噪声底来自 Da1k_C2（0.8 点）。

## 改成了什么

不改代码。P54s_own 原样重跑一次，作业名与输出前缀 P54s_own2：

    CODE_DIR=/nesi/project/uoa04509/sph/vulkan-demo-fsi_8cecf88 WALLS="--smooth-walls --baffles sheets" COUPLING=own \
        sbatch --export=ALL -J P54s_own2 job_P54.sh P54s_own2

Mahuika 作业 10328085（genoa，RTX PRO 6000，2 天上限），提交时 Priority 排队。设置与 P54s_own 完全相同
（30 mm，state_limited，τ 20 s，x_bio 10 g/kg，0.44 mol/s 从 t = 0，240 s，2 万条 lifeline 每 0.1 s，同一 lifeline 种子），
差别只来自 GPU 原子累加顺序带来的流场实现差异。输出 `/nesi/nobackup/uoa04509/sph/h54/P54s_own2_*`，预计约 17 h。

## 影响

完成后按 P54s 的流程分析（frozen 区域统计、p54s_fields.py、p54s_halves.py），给出 own 对 own2 的饥饿分数、LSL、
C_s 均值差，作为 54 m³ 上 +6.5 点的噪声底，并补入论文 3.4 的表与正文。

## 待确认

无。
