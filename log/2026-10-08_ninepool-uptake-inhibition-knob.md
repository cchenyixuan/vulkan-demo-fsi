# 2026-10-08 9-pool 摄取抑制旋钮（数值实验）、群体平均耦合模式、周期进料

## 动机

Tang 2017 的摄取 v11 = k_E11 X_E11 C_s/(C_s + K_s) 只依赖 20 h 尺度的 E11，快池不反馈到摄取，所以在任何短于小时的算例里局部双向与单向（固定容量 Monod 汇）给出相同的底物场。Haringa 2018 明说：只有当胞内快速状态控制细胞与液相的交换项时（大肠杆菌 PTS，Chassagnole 2002，Lapin 2006），单向和他们的群体平均双向才失效。为了检验耦合方法在这种情形下的差别，加一个明示的、与 PTS 的 G6P 抑制同构的抑制项。**这不是青霉菌的生理**（Tang 原文：feast-famine 下葡萄糖亲和力稳定），是耦合方法的数值实验，默认关闭。

## 原来的代码

- `predict.comp` REACTION_MODE 3：`a11 = NP_P(4) e11 xbio_cmol dt_h / (Cs + NP_P(5))`，隐式求 dC_s。
- `case.py` `NINEPOOL_PARAMETER_ORDER` 45 个参数，`ReactionParameterBuffer` 一次上传不再改动。
- 0 维模型 `_model_ninepool.py`、离线积分器 `_integrate_ninepool_lifelines.py` 同一个 v11。
- 生成器：进料只有一个窗口（`--feed-start/--feed-stop`），9-pool 初值固定为 `NINEPOOL_INITIAL`，参数无命令行入口。

## 改成了什么

### 抑制项

    v11 = k_E11 X_E11 · C_s/(C_s + K_s) · 1/(1 + X_gly* / K_i)

X_gly* 是粒子自己的 X_gly（模式 own，局部双向）或主机写入的全罐平均 X_gly（模式 mean，Haringa 2018 的群体平均耦合）。同时把 k_E11 放大 (1 + X_gly,0/K_i)，X_gly,0 取初始（恒化器稳态）值，于是稳态下两因子相消，0 维恒化器稳态逐位不变，差别只出现在 X_gly 偏离稳态时。

| 处 | 改动 |
|---|---|
| `common.glsl` | spec 105 `NINEPOOL_UPTAKE_INHIBITION`，0 关（编译掉）、1 own、2 mean |
| `predict.comp` | `if (NINEPOOL_UPTAKE_INHIBITION != 0u) a11 *= 1/(1 + gly*/NP_P(45))`，gly* = 模式 2 取 NP_P(46)，否则 max(gly, 0) |
| `case.py` | 参数 45 `Ki11`（默认 inf，1/(1 + x/inf) 精确为 1）、46 `glyMean11`；`ScalarReactionConfig.uptake_inhibition: off/own/mean`，校验 Ki11 有限与模式一致；spec 映射 105 |
| `simulator_v1.py` | spec 105 条目；`update_reaction_parameter(index, value)` 在步间重传 64 个参数（`_staging_upload` 加 offset） |
| `_run_v1_headless.py` | `--population-mean-gly-every T`：模式 mean 的算例每 T s 读回流体粒子的质量加权平均 X_gly 写入 glyMean11（起步时写一次） |
| `_model_ninepool.py`、`_integrate_ninepool_lifelines.py` | 同一因子，`p.get("Ki11", inf)`；积分器 `--ki K` 自动做 k_E11 放大 |
| `_check_ninepool_box.py` | N4 = N1 的脉冲加 K_i = 2 X_gly,ss，GPU 对带旋钮的 0 维 RK4 |

### 生成器 `_demo_stirred_tank_30l.py`

| 选项 | 作用 |
|---|---|
| `--ninepool-ki K` | 写入 `ninepool: {Ki11, kE11（已放大）, glyMean11: X_gly0}` 和 `uptake_inhibition: own`；单向算例只写注释（离线积分器用 `--ki K`） |
| `--ninepool-ki-mean` | 模式 mean |
| `--ninepool-initial k=v,...` | 覆盖 gly aa sto paa e11 e32 e4 v33 Cs CPAA 的初值（`--paa-initial` 默认改为跟随） |
| `--ninepool-param k=v,...` | 任意参数覆盖 |
| `--feed-cycle PERIOD ON COUNT` | COUNT 个进料窗口，每个 ON s，间隔 PERIOD s，从 `--feed-start` 起，每个一条 `sources`（de Jonge 2011 的 360 s 协议） |

## 验证

| 检验 | 结果 |
|---|---|
| 默认逐位相同 | N1 盒子 66,667 步，新旧着色器的全部粒子场 `np.array_equal` 为真（spec 105 = 0 时分支被编译掉） |
| N4：K_i = 49.2，k_E11 0.3900，350 μmol/kg 脉冲 120 s | GPU 对 0 维 RK4 最大相对误差：C_s 5.6e-4，X_gly 6.5e-5，μ 7.6e-5，其余 ≤ 1e-4，粒子离散 0；通过 |
| 0 维恒化器稳态不变 | D = 0.05，K_i = 49.2 和 24.6，稳态 8 个池与 C_s 相对差 0.00e+00 |
| 模式 mean 冒烟 | 6 mm 罐 300 步，每 0.01 s 更新均值，94 步/s，无溢出 |
| 生成器 | 6 mm 两周期算例：两条 sources（0.5 到 36.5 s，360.5 到 396.5 s），`uptake_inhibition: mean`，`load_case` 通过 |

旋钮的量级（N4）：350 μmol/kg 脉冲后 X_gly 峰值从原模型的约 34 降到约 32，C_s 60 s 时 213 对 205 μmol/kg，即 K_i = 2 X_ss 是温和一档。

## 影响

- 不给 `uptake_inhibition` 的算例与之前逐位相同；D1 不受影响。
- B1 和 54 m³ 的 A（单向）、C（own）、D（own，K_i 更小）、E（mean）四类算例的全部选项已就位。

## 待确认

- K_i 两档取 2 倍和 1 倍稳态 X_gly（49.2、24.6）；抑制指数 n = 1。
- 模式 mean 的更新间隔：X_gly 时间常数约 7 s，0.05 s 够；每次更新一次标量回读，4 mm 约 70 MB。
- 一个盲点：在模式 mean 下 `glyMean11` 由主机更新，续算后的第一步用检查点里的旧值，随即被起步更新覆盖，无实质影响。
