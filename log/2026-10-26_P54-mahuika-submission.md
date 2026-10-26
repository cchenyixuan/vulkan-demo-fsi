# 2026-10-26 案例一 54 m³ 预试验提交到 Mahuika（P54_own、P54_mean）

## 原来的代码

`cluster_jobs/mahuika/job_54.sh` 固定 25 mm、3,796,000 步（320 s）、lifeline 每 1186 步、快照到 320 s，用 GENFLAGS 传生成参数，是 Tang 9-pool 算例（A54、C54、E54）的脚本。state_limited 的 54 m³ 生成选项在 80b4a2c 加入，尚无对应的作业脚本。

## 改成了什么

新增 `cluster_jobs/mahuika/job_P54.sh`（基于 `job_54.sh`），一对算例只差 COUPLING：

| 项 | 设定 |
|---|---|
| 几何 | haringa54，30 mm，`--free-slip-lid --sgs --shaft-stop-at-lid`，2,315,295 粒子，dt 1.0119e-4 s |
| 反应 | state_limited：q_max 8300 μmol/gdw/h，K_s 1e-5 mol/kg，Y 90 g/mol，m_s 222 μmol/gdw/h，α 0.1，τ_up = τ_down = 20 s，μ 初值 6.94e-5 1/s，x_bio 10 g/kg，进料 0.44 mol/s 从 t = 0 |
| 耦合 | P54_own：`--state-coupling own`；P54_mean：`--state-coupling mean` 加 `--population-mean-state-every 0.5`（每 0.5 s 写入 μ̄ = Σ m X μ / Σ m X） |
| 时长 | 240 s = 2,371,800 步，前 120 s 发展，后 120 s 统计 |
| 输出 | 探针每 4941 步（0.5 s）；lifeline 2 万条，每 988 步（0.1 s），字段 substrate growth_rate uptake；快照 60 120 180 240 s；扭矩每 2000 步；每小时检查点 |
| 机器 | Mahuika genoa，1 张 RTX PRO 6000，64 GB，时限 2 天；代码 `/nesi/project/uoa04509/sph/vulkan-demo-fsi_80b4a2c`（含预编译 spv） |

τ 取 20 s：A54 的循环时间 22.5 ± 1.9 s（循环时间 = 底部探针浓度升到 5 % 的滞后的 2 倍），τ/τ_circ ≈ 0.9。进料 0.44 mol/s 按 30 L 预试验的单细胞供给（2.5e-4 mol/s 对 x_bio 10 g/kg、约 17 kg 液体）乘以液体质量比放大。

提交：作业 10149873（P54_own）、10149874（P54_mean），排队中（Priority）。输出在 `/nesi/nobackup/uoa04509/sph/h54/P54_*`。

## 影响

- 不改求解器；只新增作业脚本。
- 预计每个约 14 到 20 h 运行（按 30 mm 的粒子数估计，未在 PRO 6000 上实测）。

## 判据（与 `log/2026-10-26_case1-pretest-results.md` 一致）

只比较底物场：饥饿 / 限制 / 过量时间分数（q = C_s/(C_s+K_s)，0.36 s 平均，S < 0.05，E > 0.95，统计 120 到 240 s 的完整片段）、LSL 次数与时长、C_s 分位数、羽流体积。饥饿分数差 ≥ 5 点算一阶效应，进入两个罐的 τ 扫描；否则重新考虑路线 B 的论据。

## 待确认

- 统计窗口只有 120 s，驻留时间会偏短 5 到 10 %（A54 的窗口检验）；两组同样偏，差值可比，但绝对值不与 LB-LES 对比。
- 54 m³ 的噪声底没有重复算例，暂借 30 L 的 0.8 点；若差值落在 2 到 5 点之间，需要一个重复。
