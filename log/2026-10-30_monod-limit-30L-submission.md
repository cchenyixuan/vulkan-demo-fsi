# 2026-10-30 30 L 名义容量 Monod 单向对照算例（P1_monod）提交：state_limited 在 τ → 0 的极限

用户决定（2026-10-30）："提交吧，跑这个对照算例，提交到paratera"。

## 原来的代码

30 L τ 扫描（`log/2026-10-30_tau-scan-30L-results.md`）：own 对 mean 的饥饿分数差 7.9 / 1.9 / 0.7 点（τ 0.5 / 3 / 30 s）。
τ → 0 时 state_limited 的细胞摄取等于供给 q_max C/(K_s + C)，即名义容量的无状态 Monod 汇。问题是 τ = 0.5 s 的双向场离这个极限有多远：
若仍明显不同，则在实际的 τ 范围内"按群体标定"和"名义容量"两种单向汇都不对；若已接近，则快适应的生物可以用名义容量的单向汇。

## 改成了什么

不改求解器。新增 `cluster_jobs/paratera/submit_monod_limit.sh`：30 L，4 mm 生产配置（与 P1、τ 扫描相同的 PROD 选项），
Monod 汇 q_max 8300 μmol/gdw/h，K_s 1e-5 mol/kg，x_bio 10 g/kg，不生长（growth-yield 0），进料 2.5e-4 mol/s 从 t = 0，初值 10 K_s，
120 s，1 万条 lifeline 每 0.03 s（字段 substrate、uptake），快照 40、80、120 s，统计 40 到 120 s。

- 代码 `~/run/sph/vulkan-demo-fsi_7d31c1e`，作业 1751659（P1_monod，gpu_5090,gpu_4090，排除 wqd10nbj04g3，时限 8 h），提交时 Priority 排队。
- 输出 `~/run/sph/knob/P1_monod_*`。预计用量约 3.5 卡时。

## 影响

先验估计：极限下摄取能力是 8300 × 10 g/kg = 2.3e-5 mol/kg/s，进料约 9e-6 mol/kg/s，所以平均饱和度约 0.4，平均 C_s 约 K_s 量级，
而 τ 0.5 s 的双向算例 lifeline 上平均 C_s 为 8.2e-5（约 8 K_s）。若此估计成立，τ = 0.5 s 离极限还很远。原因在模型本身：
μ_act = Y (min(供给, 需求) − m_s) ≤ μ + α μ_max，所以 μ 上升的速度不超过 α μ_max / τ，从平均值 0.25 升到 μ_max 0.73 1/h 约需 6.6 τ（τ 0.5 s 时约 3.3 s），
比细胞穿过羽流的时间长。论文 3.4 中"零点几秒内升到满容量"的说法据此改正。

分析按 τ 扫描的流程（frozen 区域统计、lifeline 上 C_s），与 τ 0.5 / 3 / 30 s 的 own 和 mean 并排。

## 待确认

无。
