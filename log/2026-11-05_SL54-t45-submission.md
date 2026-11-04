# 54 m³ SL τ 45 s 一对（own、mean）提交：判据图上 54 m³ 的大 τ 点

日期：2026-11-05。判据图上 54 m³ 最大的 τ/τ_c 只有 0.94（SL τ20），30 L 有 2.1（W24_P1_t30，own − mean 已到 −1.4 点）。用户决定："这个也提交到mahuika吧，选一个合适的τ"。

## 原来的代码

- 54 m³ 的 state_limited 一对已有 τ 2 s、20 s（`cluster_jobs/mahuika/submit_prod_SL54.sh`）：54 m³，30 mm，G24 几何，Δ = 2.4 dx，q_max 8300 μmol/gdw/h，K_s 1e-5 mol/kg，x 10 g/kg，Y 90，维持 222 μmol/gdw/h，需求余量 0.1，μ 初值 6.94e-5 1/s，进料 0.44 mol/s 从 t = 0，240 s。
- τ_c（顶部注入到底部探针 C* = 0.05 的滞后乘 2）：30 L 14.4 s，54 m³ 21.2 s（`log/2026-11-04_30L-bottom-probe-tauc-submission.md`）。

## 改成了什么

不改求解器和生成器，代码 688e87a（到 HEAD 之间只改了 30 L 搅拌罐生成器和一个 30 L 分析脚本，不涉及本算例）。新增提交脚本 `cluster_jobs/mahuika/submit_prod_SL54_t45.sh`（仓库外，已上传到 `/nesi/project/uoa04509/sph/`），生成参数与 `submit_prod_SL54.sh` 逐字相同，只把 `--tau-up 2 --tau-down 2` 换成 `--tau-up 45 --tau-down 45`（diff 已核对）。

τ 的选择：

| 量 | 30 L W24_P1_t30 | 54 m³ G24_SL_t45 |
|---|---|---|
| τ / τ_c | 30 / 14.4 = 2.08 | 45 / 21.2 = 2.12 |
| 算例时长 / τ | 120 / 30 = 4.0 | 240 / 45 = 5.3 |
| 统计窗起点 / τ | 40 / 30 = 1.3 | 120 / 45 = 2.7 |

两罐在同一个 τ/τ_c 上各有一点，判据图上可以直接对比 own − mean 是否在 τ/τ_c ≈ 2 处都消失或反号。

| 作业 | 名称 | 含义 | 时限 |
|---|---|---|---|
| 10534757 | G24_SL_t45_own | 54 m³ state_limited，τ 45 s，局部双向 | 1 d 6 h |
| 10534758 | G24_SL_t45_mean | 54 m³ state_limited，τ 45 s，群体平均双向（每 0.5 s 更新） | 1 d 6 h |

提交时两者 Priority 排队；按 SL τ2 实测约 17 h 一个。时限合计 3,600 GPU·min，加上 FB-M、G24_monod 的剩余时限远小于每用户 20,160 GPU·min 的上限。

## 影响

- 跑完后用 SL τ2 / τ20 的口径（`cluster_jobs/mahuika/job_analyze_SL54.sh`，120 到 240 s）算 S / L / E 和 φ，加到判据图上。
- 与 G24_monod 的 S_0 一起，54 m³ 的归一化指标 (S_own − S_mean)/(S_0 − S_mean) 有 τ/τ_c 0.094、0.94、2.1 三点，和 30 L 的 0.035、0.21、2.1 对应。

## 待确认

- 大 τ 下 own 与 mean 的差可能落在噪声底以内（SL τ20 L′ 给出的重复差），到时按噪声底判断是否可算"消失"。
