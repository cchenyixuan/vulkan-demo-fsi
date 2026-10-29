# P54s 提交（54 m³，30 L 生产壁面，state_limited own / mean）

日期：2026-10-29。用户打开 Mahuika 隧道后提交。

## 原来的代码

P54_own / P54_mean（`log/2026-10-26_P54-mahuika-submission.md`，结果 `log/2026-10-28_P54-results.md`）用的是
54 m³ 生成器原来的壁面：晶格圆柱壁和 90 mm 实体挡板。饥饿分数 14.9 %（own）对 9.4 %（mean）。

## 改成了什么

不改代码，只换壁面：`--smooth-walls --baffles sheets`（commit 8cecf88，`log/2026-10-29_haringa54-production-walls.md`），
其余设置与 P54 相同（30 mm，state_limited，τ 20 s，x_bio 10 g/kg，加料 0.44 mol/s 从 t = 0 起，240 s，
2 万条 lifeline 每 0.1 s，探针 0.5 s，快照 60 120 180 240 s，每小时检查点）。

- 代码目录 `/nesi/project/uoa04509/sph/vulkan-demo-fsi_8cecf88`（含预编译 spv），作业脚本 `cluster_jobs/mahuika/job_P54.sh`（WALLS 变量）。
- 提交命令：
  `CODE_DIR=.../vulkan-demo-fsi_8cecf88 WALLS="--smooth-walls --baffles sheets" COUPLING=own|mean sbatch --export=ALL -J P54s_own|P54s_mean job_P54.sh P54s_own|P54s_mean`
- 作业 10280346 `P54s_own`、10280347 `P54s_mean`（genoa，RTX PRO 6000，2 天上限），提交时 Priority 排队，fair-share 0.31。
- 输出 `/nesi/nobackup/uoa04509/sph/h54/P54s_{own,mean}_*`。

## 影响

Case 1 的核心对比改为新壁面版本。分析沿用 P54 的流程（regime 分析、`_integrate_state_limited_lifelines.py` 离线积分、库存与通量）。

## 待确认

1. 是否加一个同协议重复（例如 P54s_own 再跑一次）作为实现间的噪声底，判断 own 与 mean 的差是否超出噪声。
