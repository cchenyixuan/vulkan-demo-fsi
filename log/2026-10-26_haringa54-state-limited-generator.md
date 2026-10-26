# 2026-10-26 54 m³ 生成器支持级别二模型 state_limited（案例一的大罐预试验）

## 原来的代码

`utils/geometry/_demo_rushton_tank.py` 只有 `--substrate`（Monod，4 个场）和 `--ninepool`（15 个场）；没有 `state_limited`。

## 改成了什么

照 30 L 生成器加入同一组选项：`--state-limited`、`--growth-yield`（g/mol）、`--maintenance-umol-per-g-h`、`--demand-margin`（α）、`--tau-up`、`--tau-down`（s）、`--product-rate P0 P1 P2`、`--growth-rate-initial`（1/s）、`--state-coupling own|mean`。`--substrate --state-limited` 时多写 growth_rate、product、maintenance 三个场（第二个 vec4，在底物 vec4 之后），反应行为 `type: state_limited`，mean 时加 `state_coupling: mean`。检查：`--state-limited` 需要 `--substrate`、`--growth-yield > 0`、不能与 `--ninepool` 同用；标量场上限按 7 个计。

## 验证

haringa54，30 mm，mean 模式本地生成：2,315,295 粒子，dt 1.0119e-4 s；本地 600 步冒烟：平均 μ 每 0.5 s（4941 步）写入一次，粒子无丢失，溢出 0，15.7 步/s（4070 Ti SUPER）。

## 影响

不加 `--state-limited` 时生成的算例与之前逐字节相同（Monod 和 ninepool 分支未改）。

## 预试验设定（P54_own / P54_mean，Mahuika）

参数与 30 L 预试验相同（`log/2026-10-25_state-limited-population-mean.md`），τ_up = τ_down = 20 s（接近 54 m³ 循环时间 22.5 s），x_bio 10 g/kg，进料 0.44 mol/s（每克细胞的供给与 30 L 相同），240 s（120 s 发展，120 s 统计），2 万条 lifeline 每 0.1 s，探针 0.5 s，快照 60 / 120 / 180 / 240 s；mean 模式每 0.5 s 写入平均 μ（τ = 20 s 时 0.5 s 内平均值几乎不变，省去每 0.05 s 读回的开销）。
