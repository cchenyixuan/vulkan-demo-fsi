# 2026-10-25 级别二模型 `state_limited` 的群体平均耦合模式（spec 108），案例一预试验的准备

## 动机

论文案例一要回答"局部双向何时必要"。Tang 加旋钮的结果说明，摄取调制有上限时（旋钮折扣最多偏离群体平均约 40 %），只有局部双向才有的底物场反馈在 30 L 和 54 m³ 上都只有几个百分点。判据要求另一端：摄取能力本身是细胞状态、随时间常数 τ 适应、调制不封顶。`state_limited`（2026-10-06）就是这个结构：摄取 = min(供给, 需求)，需求 = ((μ + α μ_max)/Y + m_s) X，μ 向实际生长率弛豫（τ_up / τ_down）。刚饿过的细胞（μ ≈ 0）需求只有 α μ_max/Y + m_s，可低到 q_max 的一成。要比较局部双向和群体平均双向，缺的是群体平均模式。

## 原来的代码

`predict.comp` REACTION_MODE 2：

    demand = ((growth_rate + REACTION_DEMAND_MARGIN * growth_rate_maximum) / REACTION_YIELD + REACTION_MAINTENANCE) * biomass_time

growth_rate 是粒子自己的 μ。ReactionParameterBuffer（set 3 binding 14）只在 ninepool 时上传参数；运行器的群体平均钩子只认 ninepool 的 `uptake_inhibition: mean`。

## 改成了什么

| 文件 | 改动 |
|---|---|
| `common.glsl` | spec 108 `REACTION_STATE_COUPLING`：0 own（默认，原模型），1 mean |
| `predict.comp` | `demand_state = (REACTION_STATE_COUPLING == 1u) ? max(reaction_parameter[0], 0.0) : growth_rate`，需求用 demand_state；μ 的弛豫仍按粒子自己的实际生长率 |
| `case.py` | `ScalarReactionConfig.state_coupling: own / mean`（只属于 state_limited，否则报错）；`state_coupling_mode`；spec 映射 108 |
| `simulator_v1.py` | spec 条目 108；state_limited 且 mean 时上传 ReactionParameterBuffer（槽位全空，reaction_parameter[0] = growth_rate 场的初值），`update_reaction_parameter` 可用 |
| `_run_v1_headless.py` | `--population-mean-state-every T`：每 T s 读回流体粒子，生物量加权平均 μ = Σ m X μ / Σ m X 写入 reaction_parameter[0]（与 `--population-mean-gly-every` 同一钩子，加了生物量权重） |
| `_demo_stirred_tank_30l.py` | `--state-coupling own|mean`，mean 时在反应行写 `state_coupling: mean` |
| `_check_reaction_box.py` | S4：（a）reaction_parameter[0] 冻结为 0、C ≫ K_s 时需求固定为 α μ_max/Y，μ = α μ_max (1 − e^(−t/τ_up))；（b）均匀方箱每步写入平均值，必须与 own 模式逐位相同 |

公式：own 模式粒子 i 的摄取 U_i = min(S_i, D(μ_i))，mean 模式 U_i = min(S_i, D(μ̄))，μ̄ = Σ m X μ / Σ m X；S 为 Monod 供给（隐式线性化），D(μ) = ((μ + α μ_max)/Y + m_s) X dt。

## 验证（本机 4070 Ti SUPER，4 mm 方箱 16³）

| 检验 | 结果 |
|---|---|
| S1 到 S3 | 与 2026-10-06 的数字逐位相同（S1 差 0.00；S2 C 7.62e-5、X 3.26e-5、μ 1.30e-4；S3 up 3.33e-4、down 8.07e-5），默认路径未变 |
| S4（a）冻结平均值 | max \|μ − 解析\| / (α μ_max) = 7.0e-5 |
| S4（b）每步写入 | max \|μ_mean − μ_own\| / μ_max = 0.0 |
| 30 L 6 mm 冒烟 | mean 模式 3000 步，平均 μ 每 381 步（0.05 s）写入，187,440 粒子无丢失，溢出 0，165.7 步/s |

## 影响

- 不写 `state_coupling` 的算例与之前逐位相同；ninepool 和 monod 路径未动。
- 可以跑案例一的预试验：30 L，4 mm，局部（own）对群体平均（mean），比较底物场本身（区域分数、羽流体积、C_s 分布）。底物场的差就是只有局部双向才有的那一层；不比较 GPU 上的细胞池（mean 模式的细胞按平均状态吃，不是文献的评估方式）。

## 预试验设定（拟）

参数取大肠杆菌葡萄糖分批补料的文献量级，作为代表性演示参数：q_max 8300 μmol/gdw/h（约 1.5 g/g/h），K_s 1e-5 mol/kg，Y 90 g/mol（0.5 g/g），m_s 222 μmol/gdw/h（0.04 g/g/h），μ_max = Y (q_max − m_s) = 0.73 1/h，α 0.1，τ_up = τ_down = 3 s（约 30 L 的循环时间），x_bio 10 g/kg，进料 2.5e-4 mol/s 从 t = 0（对应 μ ≈ 0.25 1/h，q/q_max ≈ 0.36），初值 μ 6.94e-5 1/s。120 s，统计 40 到 120 s。

## 待确认

- 预试验的差别若只有几个百分点，说明对机制的估计有误，不进入 τ 扫描。
- 参数的生物学依据（Morchain 型适应模型；大肠杆菌饥饿与饱食下转运能力的调节）在写作前要逐项核对文献。
