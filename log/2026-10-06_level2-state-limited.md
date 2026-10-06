# 2026-10-06 阶段五第一步：级别二细胞模型 `state_limited`（`scalars.reactions` 新类型，默认无）

分支 `test`。设计见 `docs/stage5_pichia_level2_design_2026-10-02.md`（用户 10 月 6 日：重启功能等算法定了再做，先试阶段五）。
模型结构按设计文档实现，参数待过程数据；结构本身仍需用户确认（设计文档第 6 节）。

**结论先说：**

1. 新反应类型 `state_limited`（REACTION_MODE 2）：细胞带生理状态 μ（比生长率），摄取受 μ 限制，μ 向实际生长率弛豫（上调、下调两个时间常数），
   有维持消耗和产物生成。`monod` 路径一字未动，R1 到 R3 的数字与改动前相同；没有 `scalars:` 块的算例不受影响。
2. 验证 S1 到 S3（见下）。
3. 生成器新选项 `--state-limited` 及参数；默认输出逐字节不变（示踪剂算例 19 个文件比对相同）。

## 合作者原来的代码

`predict.comp` 的标量更新按 vec4 循环：读 `scalar_delta`（force.comp 的输运增量）、加连续源、级别一 Monod 摄取（三个场在同一个 vec4 里）、
Kahan 补偿累加、脉冲注入、写回。反应参数 spec 79、89 到 92。

## 改成了什么

### 模型（每粒子每步，C 为输运和加料之后的底物，X 生物量，μ 状态；`common.glsl` 注释同此）

    a       = q_max X dt / (K_s + C),   ΔC_env = C a / (1 + a)           供给（与级别一相同的线性化隐式）
    ΔC_dem  = ((μ + α μ_max) / Y + m_s) X dt                              状态允许的需求
    ΔC      = min(ΔC_env, ΔC_dem)                                         实际摄取
    ΔM      = min(ΔC, m_s X dt)                                           维持消耗
    ΔX      = Y (ΔC − ΔM),   μ_act = ΔX / (X dt)                          实际生长（不为负）
    μ      += (μ_act − μ) r / (1 + r),   r = dt / τ                        隐式弛豫，τ_up（μ_act > μ）或 τ_down
    ΔP      = max(p0 + p1 μ + p2 μ², 0) X dt                              产物，q_p(μ) 二次式占位，形式待数据
    μ_max   = max(Y (q_max − m_s), 0)

Y = 0 时不限需求（退化为纯供给）；α 很大或 τ → 0 时回到级别一。收支：Σ m (C + U − F) 守恒；X − X0 = Y (U − M)。

### 场的布局

七个场两个 vec4：C、X、U、F（与级别一相同，`REACTION_LAYOUT`）| μ、P、M（`REACTION_STATE_LAYOUT`，vec4 序号 | μ 分量 << 4 | P << 8 | M << 12，
分量 4 表示没有该场）。状态 vec4 的序号必须大于底物 vec4：`predict.comp` 在处理底物 vec4 时把状态 vec4 的增量写回 `scalar_delta`，
循环后一轮处理状态 vec4 时照常做 Kahan 累加，所以级别一的代码路径一行没改。每个表达式对任意输入有限（5090 规则）。

| 文件 | 改动 |
| --- | --- |
| `experiment/v1/shaders/common.glsl` | spec 96 `REACTION_STATE_LAYOUT`，97 `REACTION_MAINTENANCE`，98 `REACTION_DEMAND_MARGIN`，99 `REACTION_TAU_UP`，100 `REACTION_TAU_DOWN`，101 到 103 `REACTION_PRODUCT_P0..P2`；公式注释 |
| `experiment/v1/shaders/predict.comp` | `REACTION_MODE == 2` 的分支 |
| `utils/sph/case.py` | `ScalarReactionConfig` 新键 `growth_rate`、`product`、`maintenance`、`maintenance_rate`、`demand_margin`、`tau_up`、`tau_down`、`product_rate`；`mode`；`reaction_state_layout()`；校验（状态场同一 vec4 且在底物 vec4 之后）；映射 96 到 103 |
| `experiment/v1/utils/simulator_v1.py` | 同上的 spec 条目 |
| `experiment/v1/shaders/README.md`、`CLAUDE.md` | 常量表；空闲范围改为 104+ |
| `utils/geometry/_demo_stirred_tank_30l.py` | `--state-limited`、`--maintenance-umol-per-g-h`、`--demand-margin`（0.1）、`--tau-up`、`--tau-down`（60 s）、`--product-rate P0 P1 P2`、`--growth-rate-initial`（默认 μ_max）；`--substrate` 用 7 个场，`--tracers` ≤ 5 |
| `experiment/v1/checks/_check_reaction_box.py` | S1 到 S3；`--only` 默认加上它们 |

## 验证（本机，4 mm 静止方箱 16³，`_check_reaction_box.py`）

dt 1.2 × 10⁻⁴ s，无量纲参数。

| 试验 | 设置 | 结果 |
| --- | --- | --- |
| S1 级别一极限 | α = 10⁶，m_s = 0，p = 0，与 `monod` 同参数（q_max 2，K_s 0.2，Y 0.5），4 s | C、X 与 `monod` 逐位相同（差 0.00）；μ 末值 0.135 = 当时的实际生长率 |
| S2 批式全模型对 RK4 | q_max 2，K_s 0.2，Y 0.5，m_s 0.2，α 0.1，τ_up 0.5，τ_down 0.2，q_p = 0.1 + 0.5 μ，8 s | 最大差：C 7.6 × 10⁻⁵，X 3.3 × 10⁻⁵，μ 1.3 × 10⁻⁴，M 2.0 × 10⁻⁵，P 8.3 × 10⁻⁶；C + U 漂移 3.9 × 10⁻¹²；X − X0 − Y (U − M) 相对 X0 1.5 × 10⁻⁸ |
| S3 状态阶跃响应 | 上调：C = 10³ ≫ K_s，μ0 = 0，α 0.1，τ_up 0.5（拐点 4.5 s 前线性，之后指数）；下调：C = 0，μ0 = μ_max，τ_down 0.3 | 相对 μ_max 最大差：上调 3.3 × 10⁻⁴，下调 8.1 × 10⁻⁵（dt / τ = 4 × 10⁻⁴ 的一阶误差） |

S2 里能看到模型的行为：μ 从 0 线性爬升（需求受 α μ_max 限制），3 s 时底物耗尽前达到 0.54，之后随供给下降按 τ_down 回落，
维持消耗 M 在底物耗尽后停止，产物按 q_p(μ) 继续累积（p0 项）。

级别一回归：R1 到 R3 用新着色器重跑，所有数字与改动前相同（R1 实现差 1.52 × 10⁻⁷、离散差 1.67 × 10⁻³、漂移 4.2 × 10⁻¹²；
R2 1.30 × 10⁻⁴ / 6.52 × 10⁻⁵ / 5.6 × 10⁻¹²；R3 7.4 × 10⁻¹⁰ / 2.3 × 10⁻⁵ / 3.81 × 10⁻⁶），全部通过。

## 影响

- 没有 `scalars:` 块：着色器里整段被 `SCALAR_VEC4_COUNT = 0` 编译掉，与原来相同。
- 级别一（`monod`）和示踪剂算例：代码路径不变，R1 到 R3 与改动前相同；生成器示踪剂算例输出逐字节相同。
- `_analyze_lifelines.py regime` 现在按记录的底物或 q 场分区；级别二下 μ 可以用 `--lifeline-fields growth_rate` 记录，分区分析用 μ 的口径还没有定（设计文档第 5 步）。

## 待确认（用户、合作者）

1. 模型结构（摄取受当前 μ 限制、μ 有记忆、产物随 μ）是否符合对毕赤酵母的认识；q_p(μ) 的二次式只是占位。
2. 参数要过程数据：μ_max、K_s、Y、m_s、q_p(μ)；τ、α 若辨识不出按计划扫描。
3. 零维拟合工具（设计文档第 3 步）等数据到了再写。
