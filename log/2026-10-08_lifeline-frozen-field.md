# 2026-10-08 lifeline 分析加 `frozen` 模式：检验 Haringa 的冻结底物场近似

## 动机

Haringa 2017、2018 的单向协议除了"细胞状态不进入汇"之外还有一层近似：底物场先在冻结的定态流场上解到定态，然后粒子在这个**固定的空间梯度**里走。我们的单向算例底物场是瞬态的。要评价他们的单向，这一层近似的代价要单独量出来；用我们自己的 lifeline 数据后处理就能做，不需要新算例。

## 原来的代码

`_analyze_lifelines.py` 有 integrity、passive、tracer、regime、selftest 五个子命令；regime 对记录的瞬时 C_s 做 Haringa 2017 的区域分析。没有空间平均场。

## 改成了什么

新增子命令 `frozen DIR [--field substrate] [--start T] [--cell 0.012] [--min-samples 20] [--monod QMAX KS] [--out NPZ]`：

1. 取 `--start` 之后的全部 lifeline 记录（位置、C_s），按 `--cell` 大小的直角网格分格，样本数 ≥ `--min-samples` 的格给出时间平均场 C̄_s(x)。lifeline 样本在罐内近似均匀，20,000 条 × 2,000 条记录足够填满 12 mm 的格。
2. 沿同一条轨迹取 C̄_s(x(t))，得到"冻结场版"lifeline，这就是 Haringa 的 parcel 会看到的信号（差别只剩 RANS 加随机游走对解析湍流）。
3. 输出瞬态与冻结两版的分布、方差分解（总方差 = 冻结场的空间方差 + 对平均场的时间脉动方差）和两版的区域分析（复用 `regime_analysis`）。

## 测试

- 6 mm 流水线算例（500 条、6 s）跑通。
- 4 mm 30 L 单向测试数据（`output/ninepool_test/one_lifelines`，x_bio 55 g/kg、连续进料、30 s，5 s 起）：瞬态 lifeline 有 2,360 个 SLS 饥饿事件，冻结场版几乎没有（S 分数 0 %），说明在这个条件下饥饿事件来自时间脉动而不是平均梯度，冻结场把它们全部抹掉。方差分解见运行输出。这是 D1one 和 A54 上要正式报告的量。

## 影响

只增加子命令，其他命令不变。

## 待确认

- 格大小 12 mm（3 个 2 mm 粒子间距、1 个 4 mm 间距的 3 倍）是暂定值，应对 6、12、24 mm 做一次敏感性。
- 平均场在实验室坐标系做，叶轮通过的周期性被平均掉，与 Haringa 的 MRF 定态场一致。
