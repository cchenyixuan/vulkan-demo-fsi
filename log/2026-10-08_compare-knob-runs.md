# 2026-10-08 旋钮算例比较脚本 `_compare_knob_runs.py`

## 目的

B1 和 54 m³ 的 A（单向）、C/D（局部双向）、E（群体平均双向）三类算例的统一比较，回答"局部耦合改变了什么"。

## 原来的代码

`_compare_lifeline_cells.py` 只比较两向与单向两组 lifeline 的分布和区域驻留，没有场快照和探针时间序列，也没有局部性的直接量。

## 改成了什么

新脚本 `experiment/v1/checks/_compare_knob_runs.py --run A=DIR --run C=DIR --run E=DIR [--one-pools A=OUT.npz] --ki K --start T [--plume 2] [--out DIR]`，每个算例目录取 `probes.csv`、`lifelines/`、`snapshots/`：

1. 探针：全罐平均 C_s、累计摄取、摄取率随时间（由 `total:uptake` 差分）。
2. 快照：C_s 均值、p95、最大值；羽流质量分数（C_s > 系数 × 均值）；X_gly 均值与 p95；**反馈偏差** d_i = f(X_gly,i)/f(X̄_gly) − 1，f = 1/(1 + X_gly/K_i)，报告 |d| > 10 % 的质量分数和羽流内外的平均 d。d 是群体平均耦合无法表达的量，是局部性的直接度量。
3. lifeline：C_s、X_gly、μ 的分布（单向的池来自离线积分器输出）、Haringa 2017 的区域分数和驻留时间（复用 `_compare_lifeline_cells.regimes`）。
4. `summary.json` 和（有 matplotlib 时）三联图。

顺带修了离线积分器在短于 10 s 的测试上统计为空而崩溃的问题。

## 验证

6 mm 罐三个算例各 6 s（进料 1 到 3 s）的端到端流水线：生成、运行、离线积分、比较全部跑通；三组在这个微小扰动下几乎相同（C_s 均值 1.5 %内，X_gly 偏差 ≤ 1.6 %），符合预期，只是流程测试。

## 待确认

- 区域阈值沿用 0.2/0.05（Haringa 2017），在 K_s 9.8e-6、C_s 稳态 9.7e-6 的条件下稳态本身就落在"过量"区，B1 分析时可能要按 Haringa 2018 的青霉菌定义调整。
- 羽流系数 2 倍均值是暂定值。
