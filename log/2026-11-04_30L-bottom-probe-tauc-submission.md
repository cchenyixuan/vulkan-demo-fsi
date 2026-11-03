# 30 L 罐底探针示踪算例：按 54 m³ 的定义算 τ_c

日期：2026-11-04。用户决定判据图的 τ_c 两罐用同一定义："和54m³保持一致吧"。

## 原来的代码

- 稿件 Methods 的 τ_c 定义：顶部注入后，底部探针第一次达到 C* = 0.05 的滞后乘 2（Haringa 2023）。54 m³ 的探针 `probe_bottom` 在 `_demo_rushton_tank.py` 的 haringa54 预设里，r = 0.75 m（T/4），y = 0.25 m（T/12），方位与注入点相对；分析脚本 `_analyze_mixing_time_54m3.py`。OW-H（G24）得 21.2 ± 2.7 s，LB-LES 20.5 到 21.7 s，实测 19.3 s。
- 30 L 生成器 `_demo_stirred_tank_30l.py` 只有论文的两个探针 `PROBE_POINTS`（液面下 4 mm），没有改探针的选项。
- 30 L 的示踪算例 W24（`mix_3mm_sheets_W24`，3 mm，Δ = 2.4 dx，10 个脉冲 25 到 34 s，Paratera 作业 1746061）只有这两个探针，没有标量快照，算不出罐底滞后。

## 改成了什么

1. 生成器新增 `--probe NAME X Y Z`（可重复），追加在论文两个探针之后：

```python
extra_probes = tuple((name, (float(x), float(y), float(z))) for name, x, y, z in args.extra_probe)
for name, (x, y, z) in PROBE_POINTS + extra_probes:
```

不加 `--probe` 时 case.yaml 不变。

2. 30 L 罐底探针，按 54 m³ 的相对位置换算（T = 0.288 m）：

| 探针 | 位置 (x, y, z) m | 说明 |
|---|---|---|
| probe_bottom | (0, −0.029, −0.072) | 主探针：r = T/4 = 0.072 m；当地碟底 y = −0.0530 m，再往上 T/12 = 0.024 m；方位与注入点（+z，r 0.116 m）相对 |
| probe_bottom_b | (0, −0.029, 0.072) | 敏感性：同半径、同高度，注入点一侧 |
| probe_bottom_c | (−0.072, −0.029, 0) | 敏感性：同半径、同高度，与注入点相差 90° |

三个探针离挡板（r ≥ 0.114 m）、转子罩（r ≤ 0.021 m）和碟底（0.024 m，Shepard 半径 h = 9 mm）都有距离。

3. 新分析脚本 `experiment/v1/checks/_analyze_circulation_time_30l.py`：C* = C / (total / fluid_mass)（守恒的精确混匀值，与 54 m³ 相同）；τ_c = 2 × (C* 首次 ≥ 0.05 的时刻 − 脉冲开始时刻)，10 个脉冲分别算，报均值 ± 标准差；对每个 `probe_bottom*` 列各算一次。

4. 算例 `mix_3mm_sheets_W24_bp`：与 W24 相同（本地生成的 case.yaml 与 W24 的逐行比较，只多三个探针），只算到 55 s（838,000 步）。最后一个脉冲 34 s，之后还有 21 s。

- Paratera，`job_mix_pb.sh mix_3mm_sheets_W24_bp 0.003 838000 305 1 0`，`-t 08:00:00`，排除节点 wqd10nbj04g3。先提交到 gpu_5090（作业 1767469，分区没有空卡，排队）；用户说 hp_5090 价格相同、可以随意用，取消后改投 hp_5090 `-A hp5090`，作业 1767472，节点 wqd10nbj11g2，立即开始。
- 代码目录 `~/run/sph/vulkan-demo-fsi_688e87a_bp`：复制 688e87a，只换了生成器文件。
- `EXTRA_FLAGS` = 生产配置（`--skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2`）加 `--sgs-filter-width 2.4` 和三个 `--probe`。
- 预计按 W24 的速度（70 s 约 6 h）约 4.7 h，约 4.7 卡时。输出 `~/run/sph/mixing/mix_3mm_sheets_W24_bp_probes.csv`。

## 影响

- 只加探针，求解器和流场设置不变；两个顶部探针仍在，这个算例同时是 W24 的一次重复（τ95 可以再算一次）。
- 跑完后 30 L 的 τ_c 与 54 m³ 同一定义，判据图横轴两罐一致。

## 待确认

- 罐底探针的位置是按 54 m³ 的相对位置换算的，30 L 的参考数据里没有罐底探针。主探针之外的两个点给出位置敏感性；如果三个点相差很大，需要用户决定用哪个，或者取平均。
- 30 L 的流场有液面快通道（`log/2026-11-03_30L-circulation-time-fluent.md`），对罐底探针的影响要看结果。

## 结果（2026-11-04，算到 41.8 s 时的探针文件）

10 个脉冲在三个罐底探针上都已到 C* = 0.05（最后一个脉冲 34 s 开始，41.8 s 时已过 7.8 s），后续时间不改变首次到达时刻，所以 τ_c 已经定下来；算例继续跑到 55 s 只为 τ95 的重复。

| 探针 | τ_c（均值 ± 标准差，范围，n 10） | 滞后 |
|---|---|---|
| probe_bottom（主探针，与注入点相对） | 14.4 ± 1.3 s（12.1 到 16.0） | 7.2 s |
| probe_bottom_b（注入点一侧） | 14.0 ± 1.1 s（12.3 到 15.4） | 7.0 s |
| probe_bottom_c（相差 90°） | 14.2 ± 1.5 s（12.4 到 16.8） | 7.1 s |

三个位置相差不超过 0.4 s，小于脉冲间的标准差，罐底探针的方位不影响结果。30 L 的 τ_c 取主探针的 14.4 s。

对照：54 m³ 21.2 ± 2.7 s（OW-H，同一定义）；30 L 平均流主回路 V/Q 7.7 到 8.2 s，顶部探针 5 % 滞后 4.4 到 5.1 s（液面快通道），Fluent 顶部探针 8 到 10 s。

τ/τ_c（30 L 用 14.4 s，54 m³ 用 21.2 s）：

| 罐 | τ | τ/τ_c | own − mean 的 S |
|---|---|---|---|
| 30 L | 0.5 s | 0.035 | +10.9 点（own2 +7.9） |
| 30 L | 3 s | 0.21 | +4.7 点 |
| 30 L | 30 s | 2.1 | −1.4 点 |
| 54 m³ | 2 s | 0.094 | +15.1 点 |
| 54 m³ | 20 s | 0.94 | +7.1 点（own2 +8.3） |

同一 τ/τ_c 下 54 m³ 的差大于 30 L（τ/τ_c 约 0.1 到 0.2 时 54 m³ +15、30 L +5 到 +10；约 1 到 2 时 54 m³ +7 到 +8、30 L 约 −1），两罐不落在同一条曲线上。判据图的纵轴是否需要按罐内饥饿水平归一化（例如除以 mean 算例的 S），待和用户讨论。
