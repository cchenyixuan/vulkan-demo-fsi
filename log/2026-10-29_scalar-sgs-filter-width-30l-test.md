# 2026-10-29 标量 Smagorinsky 滤波宽度 Δ = 2.4 dx 的 30 L 小试验（并行科技）

## 原来的代码

标量亚格子扩散 D_t = ν_t / Sc_t，ν_t = (C_s Δ)² |S|，C_s 0.1，Sc_t 0.7，Δ 取粒子间距 dx（2026-09-27 实现时的默认值，`utils/sph/case.py` 的 `filter_width` 缺省为 dx；30 L 生成器没有改 Δ 的选项）。SPH 的对流没有数值扩散，粒子尺度以下的混合全部靠 D_t（2026-09-28：关掉后示踪剂不混匀）。Δ 与 C_s、Sc_t 从未做过敏感性。

## 改成了什么

- `utils/geometry/_demo_stirred_tank_30l.py`：新选项 `--sgs-filter-width FACTOR`，在 case.yaml 的 `scalars.sgs` 写 `filter_width: FACTOR × dx`（提交 c520a86）。不加此选项时输出不变。
- Δ 的依据：求解器看到的场是 Wendland C4 核（支撑半径 h = 3 dx）的加权平均，|S| 也由核梯度算出。按核的二阶矩，每个方向的标准差 σ = 0.2265 h = 0.68 dx，等效方盒滤波宽度 √12 σ = 0.78 h = 2.35 dx。取 Δ = 2.4 dx，D_t 约为原来的 (2.4)² = 5.8 倍。

## 算例

`mix_3mm_sheets_W24`：与 `mix_3mm_sheets_P1`（10 月 6 日，3 mm 生产配置，作业 1669084，τ95 30.1 ± 1.2 s）完全相同，只加 `--sgs-filter-width 2.4`。3 mm，1,066,000 步 = 70 s，10 个示踪脉冲（25 到 34 s 各 1 s），探针每 305 步（0.02 s）。并行科技 gpu_5090，`job_mix_pb.sh`，代码 `~/run/sph/vulkan-demo-fsi_c520a86`，排除节点 wqd10nbj04g3。预计 5090 上约 6 小时、约 6 卡时。

## 比较

| 量 | 对照 | 来源 |
|---|---|---|
| τ95（数据集算法与精确 C∞ 归一化） | 3 mm 基准 30.1 ± 1.2 s；M-Star 29.1 ± 3.9，Fluent 29.2 ± 4.4，实验 24.9 ± 2.7 | `_analyze_mixing_time.py` |
| 示踪剂到达探针时的峰值（相对终值） | 我们 3 到 7 倍（2 mm P1），M-Star 与 Fluent 1.2 到 2 倍 | `_compare_probe_fluctuations.py` |
| 0.5 到 2 Hz 频带的浓度脉动 | 2 mm P1 对 M-Star 1.5 倍、对 Fluent 14 倍 | 同上 |

判读：峰值向 LES 靠拢且 τ95 不变坏，则 Δ = 2.4 dx 有依据地用于 54 m³；峰值不动，则糖团稀释少不来自 Δ 的取法。

## 待确认

- 3 mm 基准是 10 月 6 日的代码（f91eedf）；本算例用 c520a86，此后对该算例的代码路径无改动（标量 vec4 掩码等均逐位不变），差别按实现间的随机差异处理。若需要更干净的对照，另跑一个 c520a86 的 Δ = dx 重复（约 6 卡时）。
