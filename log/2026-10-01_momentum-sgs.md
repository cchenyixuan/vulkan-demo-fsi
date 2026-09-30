# 2026-10-01 动量方程的 Smagorinsky 亚格子粘性（spec 77、78，`numerics.momentum_sgs`，默认关）

分支 `test`。接 `log/2026-09-30_tank-pb2000-resolution.md`。

## 背景

30 s 稳态的流场插值到网格（与 Fluent 网格尺度相当的滤波宽度）后，全槽平均速度仍比 Fluent 高：3 mm 单层 +8%，
3 mm 薄板 +23%，2 mm 薄板 +30%；动能高 30%，而 2 mm 薄板的输入功率与 Fluent 相当（总力矩 82 对 76 到 80）。
同样的输入、更多的动能，说明耗散比 LES 少。Fluent 和 M-Star 在动量方程里都有 Smagorinsky（C_s = 0.1）；
我们的 ν_t 只用于示踪剂扩散（`log/2026-09-27_scalar-transport.md`），动量方程只有分子粘度，3 × 10⁴ 的雷诺数下它几乎不起作用。

9 月 28 日曾估算"Smagorinsky 项只能耗散输入功率的 7%"而搁置（会话记录，未入日志）。那次按滤波宽度 Δ = dx 算，
而且当时 `own` 成对力每步凭空吃掉转子输入 37% 到 92% 的角动量，充当了耗散，滤波后的速度只比 Fluent 高 4%。
9 月 30 日成对力改成守恒形式后这个"假耗散"没了，速度差扩大到 23% 到 30%，缺口露出来。ν_t ∝ Δ²，Δ 取 h = 3 dx 比取 dx 大 9 倍，
所以两个都试。

## 合作者原来的代码

`force.comp` 的 Morris 粘性项每对粒子用材料粘度 ν：

```glsl
a_viscosity += morris_coefficient * viscosity_uniform * pair_volume_viscous
             * dot(self_velocity - neighbor_velocity, position_difference_neighbor_to_self)
             / (distance * distance + EPS_H_SQUARED) * kernel_gradient_pair;
```

`density.comp` 算 ν_t = (C_sΔ)²|S|（应变率由 KCG 修正的速度梯度得到），只在标量构建里编译，只给标量扩散用。

## 改成了什么

| 项目 | 内容 |
| --- | --- |
| `common.glsl` | spec 77 `USE_MOMENTUM_SGS`（默认 false），78 `MOMENTUM_SGS_LENGTH_SQUARED` = (C_sΔ)² |
| `density.comp` | 速度梯度和 ν_t 在 `USE_SCALAR_SGS || USE_MOMENTUM_SGS` 时计算；动量 SGS 开时用它自己的 (C_sΔ)² |
| `force.comp` | 流体与流体的粒子对：`pair_viscosity = ν + ½(ν_t,i + ν_t,j)`；流体与固体、薄板假想邻居仍用 ν |
| `correction.comp` | 流体标志（矩阵旁的 .w）在动量 SGS 开时也写入，`force.comp` 用它识别流体邻居 |
| `utils/sph/case.py` | `numerics.momentum_sgs`（bool）、`momentum_sgs_cs`（0.1）、`momentum_sgs_filter_width`（m，不设即 dx）；`momentum_sgs_length_squared` |
| `simulator_v1.py` | spec 项；`turbulent_viscosity` 缓冲和 density→force 的屏障在动量 SGS 开时也启用；`readback_turbulent_viscosity` 放开 |
| 生成器 | `--momentum-sgs CS --momentum-sgs-width {dx,2dx,h,米}` |

公式：

    ν_t,i = (C_s Δ)² |S_i|,   |S| = sqrt(2 S:S),   S = ½(∇v + ∇vᵀ)，∇v 用 KCG 修正的梯度，固体邻居按无滑移计入
    流体对的粘性项：   ν_pair = ν + ½(ν_t,i + ν_t,j)          两个粒子对称，成对力仍等大反向
    流体与固体的对：   ν_pair = ν                              近壁衰减的粗糙替代（真实 LES 里 ν_t 在壁面趋零）

ν_t 是本步 `density.comp` 写的，`force.comp` 在同一步读，两个核之间已有计算屏障。缓冲区不随 defrag 搬运，
但每步都被重写，读的都是本步的值。粘性时间步限制 h²/ν_t：Δ = h、|S| = 10³/s 时 ν_t ≈ 8 × 10⁻⁴，限值约 0.1 s，远大于时间步。

默认构建：spec 77 为 false 时 `density.comp`、`force.comp`、`correction.comp` 的新分支都是死代码。

## 本机验证（4 mm 薄板，0 g，p_b 2000，0.5 s）

| 0.3–0.5 s | Rushton | 转子输入 | 动能，J | 相对无 SGS | 最小密度 |
| --- | --- | --- | --- | --- | --- |
| 无 SGS | 59.9 | 103.9 | 0.355 | | 996.8 |
| Δ = dx | 55.7 | 101.9 | 0.336 | 动能 −5%，力矩 −4% | 996.0 |
| Δ = h | 55.0 | 98.7 | 0.268 | 动能 −25%，力矩 −7% | 996.0 |

稳定，无溢出，收支闭合（流体间项 −0.1）。速度慢约 20%（速度梯度累积和 ν_t 读取）。

## 并行科技算例（3 mm 薄板，6 s，与 `plates_true_pb2000_budget` 对照）

| 作业号 | 名称 | Δ |
| --- | --- | --- |
| 1642572 | plates_true_pb2000_sgs_dx_budget | dx = 3 mm |
| 1642573 | plates_true_pb2000_sgs_h_budget | h = 9 mm |

看三件事：网格滤波后的全槽平均速度（Fluent 0.157，M-Star 0.150）、动能、力矩和角动量收支；
另外用末态离线算 ν_t 的耗散 Σ m ν_t|S|² 占输入功率的份额。结果后补。

## 需要合作者确认

1. 是否同意动量方程加 Smagorinsky 项（默认关）；他之前把 ν_t 限制在标量上的理由。
2. Δ 取 dx 还是 h；C_s 是否沿用 0.1。
3. 流体与固体的对不加 ν_t 作为近壁衰减是否可接受，还是需要壁面距离函数。
