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

两个作业各 25.5 分钟（比无 SGS 慢 10%），合计 0.85 卡时。3–6 s 平均，力矩已除以 1.2187，收支含系数：

| | 无 SGS | Δ = dx | Δ = h |
| --- | --- | --- | --- |
| Rushton | 49.8 | 49.0 | 43.3 |
| PBT | 22.4 | 21.5 | 22.2 |
| 转子输入 | 88.1 | 86.0 | 79.8 |
| 壁面和挡板拿走 | 79.0 | 71.7 | 71.8 |
| 流体间项 | −0.05 | −0.07 | −0.06 |
| 动能，J | 0.831 | 0.817 | 0.752 |
| 全槽速度，粒子平均 / 滤波 2 dx（Fluent 0.157） | 0.203 / 0.188 | 0.204 / 0.190 | 0.196 / 0.186 |
| Rushton 区 / PBT 区 / 主体流速（Fluent 0.461 / 0.285 / 0.147） | 0.413 / 0.309 / 0.195 | 0.401 / 0.319 / 0.196 | 0.393 / 0.286 / 0.189 |

**结论：动量 SGS 不能把全槽速度降下来。** Δ = h 时 ν_t 达分子粘度的 16 到 50 倍，动能降 10%，全槽平均速度只降 1%，主体区降 3%；
它的作用集中在应变率最大的叶轮附近，等于给排出流加阻力，Rushton 力矩降 13%，方向与 Fluent 相反。Δ = dx 的效果在采样噪声内。
稳定，收支闭合。

对 9 月 28 日以来推断的修正：总耗散率由输入功率定死，SGS 只是把原来由数值机制承担的耗散换成显式的，大尺度的能量分配不变，
主体流速几乎不动。这正是 LES 里 SGS 模型的定位（耗掉传下来的能量，不决定大尺度流场）。主体偏快 20% 到 30% 的原因在大尺度的能量分配：
能量注入的位置和方向（Rushton 排出流）、壁面和挡板拿走的比例、或参照值的口径。单层 30 s 的滤波速度只比 Fluent 高 8%，薄板高 23%，
薄板的输入功率比单层高 10%，两者大致相称；真正没解释的是 2 mm 薄板：输入功率与 Fluent 相当，速度高 30%。

动量 SGS 保留为开关，默认关，不作为标准设置；Δ = dx 可作为混合时间的敏感性算例之一。
ν_t 耗散的份额没有单独算（需要末态的 ν_t 或离线重算速度梯度），动能变化已足以说明问题。

## 需要合作者确认

1. 是否同意动量方程加 Smagorinsky 项（默认关）；他之前把 ν_t 限制在标量上的理由。
2. Δ 取 dx 还是 h；C_s 是否沿用 0.1。
3. 流体与固体的对不加 ν_t 作为近壁衰减是否可接受，还是需要壁面距离函数。
