# 2026-10-29 30 L 涡结构三维对比页：Fluent LES 与 SPH 2 mm 的涡量（对数色标）

## 原来的代码

只有方位平均的子午流线、动能分区等二维对比（`log/2026-10-13_energy-by-region-2mm-vs-fluent.md`），没有三维涡结构的可视化。

## 改成了什么

| 文件 | 内容 |
|---|---|
| `experiment/v1/checks/_vorticity_grid_30l.py` | SPH 转储或 Fluent 插值文件 → 3 mm 规则网格上的 log10 \|ω\|。速度与样本数分别累加到网格，同一高斯核平滑后相除（归一化卷积）；尺度 σ、2σ、4σ、8σ，每格取核内至少 4 个样本的最细尺度。ω = ∇ × u（numpy.gradient）。流体区：SPH 取粒子分布（最细尺度样本数 ≥ 0.3 倍中位数，向内收一格）；Fluent 用 SPH 网格的流体区（`--mask-from`）。Fluent：y − 58.5 mm；Rushton 区旋流为负的快照镜像回来（z → −z）；转子区速度转回滑移网格 12° 的滞后 |
| `experiment/v1/checks/_vorticity_viewer_data.py` | 网格 → 灰度 PNG 图集（log10 \|ω\| 在 [−0.3, 2.7] 上量化到 1..255，0 为遮罩）与 manifest.json |
| 页面 `output/vorticity30l/viewer/index.html`（不入库） | three.js 0.128 的光线步进体渲染，颜色与不透明度按对数量级映射（inferno 色标），两栏同步相机，快照轮播，显示下限与不透明度滑块，剖切（完整 / 去前半 / 中心 24 mm 薄片），平滑尺度切换 |

数据：Fluent 细网格 25 到 34 s 十个快照（本机 Fluent 2025 R1 批处理导出速度，`fineNN.ip`，每个 3,408,405 / 3,408,218 个单元，26 到 34 s 识别为镜像）；SPH `s2_sheet2baf_smooth_fromrest20` 的 10、12、14、16、18、20 s 转储（并行科技 `~/run/sph/tank/`，在登录节点上算网格，`vort/sph2mm_s{3,6}_t*.npz`）。两种平滑：σ = 6 mm 两者统一（对比用），以及各自最细（SPH σ = 3 mm；Fluent 31 % 的格子 3 mm、69 % 为 6 mm，上部主体单元超过 10 mm）。

页面：https://claude.ai/artifact/UzFEyLZPWAtbCeXfusQJMm（私有）。

## |ω| 分位数（1/s，各帧平均）

| | p5 | p50 | p95 | p99.9 |
|---|---|---|---|---|
| Fluent，σ 6 mm | 1.0 | 5.8 | 24.1 | 59.5 |
| SPH 2 mm，σ 6 mm | | 7.8 | 24.4 | 59.2 |
| Fluent，各自最细 | 1.0 | 6.6 | 29.6 | 144.6 |
| SPH 2 mm，σ 3 mm | 2.4 | 11.1 | 40.7 | 143.8 |

（SPH σ 6 mm 的 p5 未单列；p50 为 6 帧的 7.3 到 8.2。）统一 6 mm 平滑时，两者的 p95 与 p99.9 几乎相同，SPH 的中位数高约 30 %（主体弱涡更强，与动能分区中非轴对称部分偏高一致）；各自最细时差别主要来自 Fluent 主体网格粗。

## 影响

- 只新增分析与可视化脚本，不改求解器。
- 快照间隔 1 到 2 s，相邻帧的涡结构互不相关，轮播只是不同瞬间的对比。

## 待确认

- 是否需要 SPH 的密帧（如每 0.02 s）动画来看涡的演化（需要新算例；Fluent 没有密帧数据）。
