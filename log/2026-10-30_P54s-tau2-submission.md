# 2026-10-30 54 m³ τ = 2 s 一对（P54s_t2_own / P54s_t2_mean）提交

用户决定（2026-10-30）："54 m³ τ = 2 s 上Mahuika"。

## 原来的代码

54 m³ 只有 τ = 20 s 一对（P54s，饥饿分数 16.7 对 10.2 %，`log/2026-10-30_P54s-results.md`），τ/τ_c ≈ 1（τ_c 20.9 ± 2.6 s）。
30 L 的 τ 扫描和极限算例（`log/2026-10-30_monod-limit-30L-results.md`）表明差别随 τ 减小而增大，但大罐没有小 τ/τ_c 的点。
`cluster_jobs/mahuika/job_P54.sh` 里 τ 写死为 20 s。

## 改成了什么

`job_P54.sh` 加环境变量 `TAU`（默认 20，τ_up = τ_down = TAU），不设时与原脚本相同。本地与 Mahuika 的副本在修改前逐字一致，修改后同步。

    CODE_DIR=/nesi/project/uoa04509/sph/vulkan-demo-fsi_8cecf88 WALLS="--smooth-walls --baffles sheets" COUPLING=own|mean TAU=2 \
        sbatch --export=ALL -J P54s_t2_own|P54s_t2_mean job_P54.sh P54s_t2_own|P54s_t2_mean

| 作业 | 算例 |
|---|---|
| 10341262 | P54s_t2_own |
| 10341263 | P54s_t2_mean |

genoa，RTX PRO 6000，2 天上限，提交时 Priority 排队。代码目录与 P54s 相同（8cecf88），除 τ 外设置与 P54s 完全相同：
30 mm，state_limited，x_bio 10 g/kg，0.44 mol/s 从 t = 0，240 s，2 万条 lifeline 每 0.1 s。输出 `/nesi/nobackup/uoa04509/sph/h54/P54s_t2_{own,mean}_*`，预计各约 17 h。

## 影响

补上 54 m³ 的 τ/τ_c ≈ 0.1 这一点，与 30 L τ 0.5 s（τ/τ_c 约 0.17，按估计的 τ_c ≈ 3 s）对应。
分析按 P54s 流程（frozen 区域统计、p54s_fields.py、p54s_halves.py，统计段 120 到 240 s）。结果进 3.4、表 tab:coupling-field、tab:cases 和判据图。

## 待确认

1. 作业开始后核对 `P54s_t2_*_case/case.yaml` 里的 τ 确为 2 s。
2. P54s 的统计段 120 s 对 τ 2 s 足够（τ 更小，细胞状态更快到统计稳态），不改。
