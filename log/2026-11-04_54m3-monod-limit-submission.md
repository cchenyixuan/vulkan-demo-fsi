# 54 m³ 名义容量 Monod 单向算例（G24_monod）提交：判据图归一化需要的 τ → 0 极限

日期：2026-11-04。判据图上两罐的 own − mean 饥饿分数差不落在同一条曲线上（`log/2026-11-04_30L-bottom-probe-tauc-submission.md`）。讨论后建议的纵轴是 (S_own − S_mean) / (S_0 − S_mean)，S_0 为 τ → 0 极限的饥饿分数，用户决定："放到mahuika上跑吧"。

## 原来的代码

- 30 L 的 τ → 0 极限已有：W24_P1_monod（30 L，4 mm，无状态 Monod，q_max 8300 μmol/gdw/h，K_s 1e-5 mol/kg，x 10 g/kg，不生长，进料 2.5e-4 mol/s），40 到 120 s 的 S 20.6 %（`log/2026-11-03_production-analysis-results.md`）。
- 54 m³ 没有对应算例（`manuscript/open_items.md`：约 17 h，Mahuika，之前判断"要写这句才需要跑"）。
- 54 m³ 的 state_limited 算例 G24_SL_t2 / t20（own、mean）：54 m³，30 mm，G24 几何，Δ = 2.4 dx，q_max 8300，K_s 1e-5，x 10 g/kg，进料 0.44 mol/s 从 t = 0，240 s（`cluster_jobs/mahuika/submit_prod_SL54.sh`）。

## 改成了什么

不改求解器和生成器，代码 688e87a。新增提交脚本 `cluster_jobs/mahuika/submit_prod_monod54.sh`（仓库外，已上传到 `/nesi/project/uoa04509/sph/`）：

- 生成参数 = G24_SL_t2_own 的参数去掉 state_limited 一组（`--state-limited --growth-yield --maintenance-umol-per-g-h --demand-margin --tau-up --tau-down --growth-rate-initial --state-coupling`），只留 `--substrate --biomass 10 --q-max-umol-per-g-h 8300 --k-s 1e-5 --feed-rate 0.44 --feed-start 0`，与 30 L 的 W24_P1_monod 对应。
- 本地用同样参数生成（dx 0.030 m）：流体 1,939,254，dt 1.0119e-4 s；reaction 一行为 `{type: monod, q_max: 2.305556e-06, k_s: 1.0e-05, yield: 0.0}`（8300 μmol/gdw/h = 2.3056e-6 mol/g/s）。
- 运行：`job_prod.sh G24_monod 0.030 2371800 4941 988 60 120 180 240`，240 s，探针每 0.5 s，lifeline 2 万条从 t = 0 每 0.1 s（字段 substrate、uptake），快照 60、120、180、240 s，扭矩每 2000 步。
- Mahuika 作业 10473352（genoa，PRO 6000），时限 1 d 6 h，提交时 Priority 排队。按 SL τ2 own 实测 38 步/s 估计约 17 h。时限选择：MaxGRESRunMinsPerUser 20,160 GPU·min，OW-2、FB-L、FB-M 剩余时限合计约 9,860，余约 10,300，1,800 放得下。
- 输出 `/nesi/nobackup/uoa04509/sph/production/G24_monod_*`，结束时除检查点外复制到 `/nesi/project/uoa04509/sph/production/`。

## 影响

- 跑完后按 SL τ2 的口径（120 到 240 s，`_analyze_lifelines.py frozen --monod 1 1e-5` 等）算 S_0，两罐的判据图纵轴都可以写成 (S_own − S_mean) / (S_0 − S_mean)：τ → 0 时为 1，τ 很大时为 0。
- 30 L 现有的值（S_0 = 20.6 %）：τ/τ_c 0.035 为 0.62（own2 0.45），0.21 为 0.28，2.1 为 −0.09。own 重复算例的 S 差 3 点，折成这个指标约 0.17。
- 同时得到 54 m³ "两种单向汇夹住真值"的另一端（open_items 原来的用途）。

## 待确认

- 54 m³ 的 S_0 若与 mean 的 S 差很小（分母小），指标噪声会放大，到时再看是否可用。
