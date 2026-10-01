# 2026-10-01 阶段三第二步：Monod 摄取与连续加料（`scalars.reactions`、`scalars.sources`，默认无）

分支 `test`。接 `log/2026-10-01_lifeline-recorder.md`。设计见 `docs/stage3_lifeline_design_2026-10-01.md`。

**结论先说：**

1. 新增两个选项：`scalars.reactions`（Monod 摄取，生物量挂在流体粒子上）和 `scalars.sources`（球内连续加料）。
   都在 `predict.comp` 里逐粒子计算，没有邻居循环；没有这两个块的算例与原来完全相同。
2. 验证全部通过：批式摄取与求解器自身的递推一致到 1.5 × 10⁻⁷；带生长时与 RK4 一致到 10⁻⁴；加料箱的底物收支守恒到 7 × 10⁻¹⁰，
   加入量与 速率 × 时间 × 球内粒子份额 一致到 2 × 10⁻⁵；4 mm 搅拌槽 1 s 收支守恒到 1.6 × 10⁻¹⁰，底物处处非负。
3. 原有的示踪脉冲不受影响（标量方箱的结果与早上逐位相同，脉冲结束后总量不变）。
4. 生成器新增 `--substrate` 等选项，参数默认取 Haringa 2023。

## 合作者原来的代码

`predict.comp` 把 `force.comp` 算好的标量增量用 Kahan 补偿加到粒子上，然后处理示踪脉冲（球内的粒子直接置为给定值）。
脉冲由主机每步写入 4 个槽位（`ScalarInjectionSlot`：球心和半径平方、数值、启用标志与分量），再拷到显存。没有反应项，也没有连续源。

## 我们改成了什么

### 物理模型

每个流体粒子在同一个 vec4 里带四个场：底物 C（mol/kg）、生物量 X（g/kg）、累计摄取 U、累计加料 F。

摄取（每粒子每步，用输运增量和加料之后的底物）：

    a  = q_max X dt / (K_s + C)
    ΔC = C a / (1 + a)                 线性化的隐式更新，保证 ΔC ≤ C，C 不会变负
    C −= ΔC,   U += ΔC,   X += Y ΔC     Y = 0 时不生长

加料：球内每个流体粒子每步

    ΔC = 速率 · dt / (ρ₀ V),   V = 4/3 π R³,   F += ΔC

球内的粒子数随流动起伏，所以实际加入量与 速率 × 时间 有百分之几的差别；F 逐粒子记下真实加入量，收支因此是精确的：

    Σ m (C + U − F) = 常数（初始底物总量）

### 代码

`predict.comp`，在原来的 Kahan 加法之前把加料和摄取并入增量，U、F 因此也得到补偿累加：

```glsl
// Continuous sources: every FLUID particle inside the sphere at x_{n+1} gains value.x
if (slot.target.w == 1u && inside) { delta[c] += slot.value.x; delta[record] += slot.value.x; }
// Monod uptake on the substrate after the transport increment and the source
float substrate = max(value[s] + delta[s], 0.0);
float biomass   = max(value[x] + delta[x], 0.0);
float rate_factor = REACTION_Q_MAX * biomass * TIMESTEP / (REACTION_HALF_SATURATION + substrate);
float uptake = substrate * rate_factor / (1.0 + rate_factor);
delta[s] -= uptake;  delta[u] += uptake;  delta[x] += REACTION_YIELD * uptake;
```

原来的脉冲循环加了一个条件 `slot.target.w == 0u`，只处理脉冲。

| 文件 | 改动 |
| --- | --- |
| `experiment/v1/shaders/common.glsl` | spec 79 `REACTION_MODE`，89 `REACTION_LAYOUT`（vec4 序号和三个分量打包），90 `REACTION_Q_MAX`，91 `REACTION_HALF_SATURATION`，92 `REACTION_YIELD`；`ScalarInjectionSlot` 的 `target.w` 为模式（0 脉冲，1 连续源），`value.y` 为记录场的分量加一 |
| `experiment/v1/shaders/predict.comp` | 加料和摄取并入增量 |
| `utils/sph/case.py` | `ScalarReactionConfig`（case.yaml 键 `yield` 对应 `growth_yield`）、`ScalarSourceConfig`；校验：底物、生物量、摄取场必须在同一个 vec4，记录场与加料场同一个 vec4，至多一个反应，脉冲加连续源同时不超过 4 个；spec 映射 |
| `experiment/v1/utils/simulator_v1.py` | spec 项；`active_sources()`、`source_increment()`；槽位写入时区分模式 |
| `utils/geometry/_demo_stirred_tank_30l.py` | `--substrate`、`--q-max-umol-per-g-h`（1600）、`--k-s`（7.8e-6）、`--biomass`（55）、`--growth-yield`（0）、`--substrate-initial`（10 K_s）、`--substrate-diffusivity`（6e-10）、`--no-uptake`、`--feed-rate`（2.0e-4 mol/s）、`--feed-center`（数据集的注入点）、`--feed-radius`（0.02 m）、`--feed-start`、`--feed-stop`；可与 `--tracers` 共用（总数不超过 12 个场） |
| `experiment/v1/checks/_check_reaction_box.py` | 新文件，R1 到 R3 |

加料速率 2.0 × 10⁻⁴ mol/s 的来由：29.5 L 的摄取能力 q_max · X · M = 4.44 × 10⁻⁷ × 55 × 29.5 = 7.2 × 10⁻⁴ mol/s，取 0.28 倍，
使稳态下平均 q/q_max ≈ 0.28，与 Haringa 2023 的 54 m³ 罐相同（0.37 mol/s 对 1.32 mol/s）。

## 验证

### 方箱（16³ 个流体粒子，4 mm，静止，`_check_reaction_box.py`，共 1 分 35 秒）

| 编号 | 内容 | 结果 |
| --- | --- | --- |
| R1 | 批式，Haringa 参数，C0 = 10 K_s，不生长，6 s | GPU 与双精度递推之差 ≤ 1.5 × 10⁻⁷；C + U 守恒到 4 × 10⁻¹²；粒子之间无差异 |
| R2 | 批式带生长，无量纲参数（q_max 2，K_s 0.2，C0 1，X0 0.1，Y 0.5），8 s | 与 RK4 之差：C 1.3 × 10⁻⁴，X 6.5 × 10⁻⁵；X + Y C 守恒到 6 × 10⁻¹² |
| R3 | 加料箱：中心球半径 15 mm，速率 2 × 10⁻⁹ mol/s，摄取开，4 s | 收支漂移 7.4 × 10⁻¹⁰；加入量与预期之差 2.3 × 10⁻⁵；C 最小 3.8 × 10⁻⁶ |

R1 的离散误差（递推相对于精确解）在底物衰减约 15 个 e 倍后为 1.7 × 10⁻³。这是隐式线性化更新的一阶时间误差，
随 (λt)(λ dt)/2 增长，λ = q_max X / K_s = 3.1 /s；在 3 mm 槽的时间步下每步 2 × 10⁻⁴。对代谢区间的分类（0.05、0.95 两道门槛）没有影响，没有改用更高阶的格式。

### 回归

| 检查 | 结果 |
| --- | --- |
| `_check_scalar_box.py` T1、T5 | 与当天早上的结果逐位相同 |
| 4 mm 槽两个示踪脉冲，0.5 s | 脉冲结束后总量不变（6.186045868 × 10⁻³，漂移 10⁻¹² 量级） |

### 4 mm 搅拌槽冒烟（薄板，p_b 2000，标量 SGS 开，0.2 s 起加料，1 s，3 分 13 秒）

| 量 | 结果 |
| --- | --- |
| Σ m (C + U − F) | 2.786057564 × 10⁻³，全程不变（漂移 1.6 × 10⁻¹⁰） |
| 加入量（扣除质量系数） | 1.6103 × 10⁻⁴ mol，速率 × 时间 1.6129 × 10⁻⁴ mol |
| 底物 | 沿 5000 条 lifeline 在 5.6 × 10⁻⁵ 到 4.7 × 10⁻³ mol/kg 之间，无负值 |
| lifeline | 无丢失、无跳变；区间分析能跑通（1 s 内初始的 10 K_s 还没耗完，全部处于限制区，符合预期） |
| 速度 | 60.6 步/s |

### 零维补料分批极限（R4，用上面的冒烟数据）

槽平均底物与零维补料分批方程 dC/dt = dF/dt − q_max X C / (K_s + C) 比较（dF/dt 取算例实际加入量）：

| t，s | 槽平均 C | 零维方程 | 相对差 |
| --- | --- | --- | --- |
| 0.050 | 7.688246 × 10⁻⁵ | 7.688246 × 10⁻⁵ | 0 |
| 0.352 | 7.124974 × 10⁻⁵ | 7.124749 × 10⁻⁵ | +3.2 × 10⁻⁵ |
| 0.654 | 6.669485 × 10⁻⁵ | 6.667246 × 10⁻⁵ | +3.4 × 10⁻⁴ |
| 0.956 | 6.220395 × 10⁻⁵ | 6.213473 × 10⁻⁵ | +1.1 × 10⁻³ |

底物远高于 K_s 时槽平均回落到零维解，差别 1.3 × 10⁻³ 以内。偏差为正且随加料增大：加入的底物集中在加料点附近，那里摄取已饱和，
平均摄取率低于按平均浓度算的摄取率（q 对 C 是凹函数）。这正是 lifeline 研究要刻画的分区效应。

## 生产算例 C1（已准备，未提交）

30 L 槽 3 mm 薄板，不开重力，p_b 2000，标量 SGS 开；Haringa 动力学，X = 55 g/kg，初始 C = 10 K_s；20 s 起在数据集的注入点加料
2.0 × 10⁻⁴ mol/s（半径 2 cm 的球）；跑到 90 s（1,371,133 步）；从 30 s 起记录 2 万条 lifeline（位置、底物、累计摄取，每 10 次记录存速度和位移）；
作业末尾做完整性、被动粒子统计和代谢区间分析。被动示踪的统计就取自这批 lifeline 的轨迹。

| 集群 | 脚本 | 估计 |
| --- | --- | --- |
| 并行科技 | `~/run/sph/lifeline/job_lifeline_pb.sh`（已上传，求解器代码和着色器已部署，校验和一致） | 约 7 小时、7 卡时 |
| Mahuika | `job_lifeline.sh`（本机已准备，隧道恢复后上传） | 免费，排队一到两天 |

## 有什么影响

- 没有 `reactions` 和 `sources` 的算例不受影响：spec 79 为 0 时摄取的分支被消除；连续源的循环只在声明了脉冲或连续源的算例里编译。
- 生物量是流体粒子上不扩散的场，摄取在本粒子上直接扣除，不需要核权重分配，也没有原子累加带来的非确定性。

## 需要合作者确认

1. 摄取放在 `predict.comp` 里、与输运分步，并且用输运之后的底物计算速率，这个顺序他是否认可。
2. 连续源按名义体积 V 归一化，实际加入量随球内粒子数起伏，用记录场修正收支；他是否倾向按实际粒子数归一化（需要先数球内粒子，多一个归约步骤）。

## 文件

见上表。另有 `experiment/v1/checks/_check_reaction_box.py`。
