# 2026-10-13 流体与固体粒子对的粘度：`numerics.solid_pair_viscosity`（含 Werner 和 Wengle 壁面律）

分支 `test`。起因：光滑槽壳的 30 L 算例里槽底的粘性力矩只有 Fluent LES 的 40 %（1.0 对 2.46 mN·m，3 / 2 / 1.5 mm 为
1.09 / 0.99 / 0.94，不随分辨率改善；`log/2026-10-04_baffle-drag-diagnosis.md` 第四节）。第一层流体粒子离壁约半个间距，
y⁺ 约 30，粘性底层解析不了。合作者建议把壁面材料的粘度调大，流体与壁面粒子对取平均。本篇把它做成求解器选项，默认关，
默认构建不变；另加一个壁面律模式。标定与 20 s 对比见第四、五节。

## 一、原来的代码

`force.comp` 里流体粒子 i 与任何邻居 j 的 Morris 粘性项都用 i 自己材料的分子粘度 ν_f：

    a_i += 2 (d + 2) ν_f V_j (v_i − v_j)·x_ij / (|x_ij|² + 0.01 h²) B_i ∇W_ij

固体邻居（壁面、转子粒子，薄板的壁面假粒子）也是这个 ν_f。`materials.yaml` 里每种材料都有 `viscosity` 字段，但固体的值
不参与流体受力；生成器 `--viscosity` 把流体、壁面、转子三种材料一起改。固体的受力走 `solid_reaction_acceleration()`
（`numerics.solid_reaction_force`），用同一个 ν_f，所以固体读回的力是流体受力的负值，力矩读回依赖它。

## 二、改成了什么

### 选项与公式（`common.glsl` 特化常量 106、107）

`numerics.solid_pair_viscosity`，取值和流体与固体对的粘度 ν_ij（ν_f 流体材料、ν_s 固体粒子所属材料的 `viscosity`）：

| 取值 | ν_ij | 说明 |
| --- | --- | --- |
| `self`（默认） | ν_f | 原代码 |
| `harmonic` | 2 ν_f ν_s / (ν_f + ν_s) | 上限 2 ν_f，不能当旋钮 |
| `arithmetic` | (ν_f + ν_s) / 2 | |
| `solid` | ν_s | 壁面材料的粘度直接作用在流体与壁面对上 |
| `wall_law` | ν_eff,i（下式） | Werner 和 Wengle 显式幂律，`density.comp` 每步算一次，存入 set 0 binding 15 |

壁面律（`density.comp` 末尾，`WallViscosityBuffer`）。对每个流体粒子 i：

    d    = |x_i − x_w| − r_p            x_w 最近的壁面粒子（或薄板假粒子），r_p 粒子半径（SGS_WALL_DISTANCE_OFFSET）
    n    = Σ_j V_j ∇W_ij 的方向（j 取壁面邻居；和为零时取最近壁面粒子的方向）
    u_t  = |(v_i − v_w) − ((v_i − v_w)·n) n|          相对最近壁面粒子的切向速度
    Re_d = u_t d / ν_f
    Re_d ≤ 11.81²：u⁺ = y⁺，                 ν_eff = ν_f
    Re_d > 11.81²：u⁺ = 8.3 (y⁺)^{1/7}，     y⁺ = (Re_d / 8.3)^{7/8}，
                   ν_eff = u_τ² d / u_t = ν_f y⁺ / u⁺ = ν_f (Re_d / 8.3)^{3/4} / 8.3   （≥ ν_f，在 11.81 处连续）

推导：u_t / u_τ = 8.3 (u_τ d / ν_f)^{1/7} 解出 u_τ；SPH 对的粘性应力 μ_eff u_t / d 等于壁面律的 ρ u_τ²。
d 有下限（10⁻³ r_p），法向由构造是单位向量，pow 的底数为正：任何输入都有限（RTX 5090 的分支投机问题，
`log/2026-09-30_thin-plates-mirror.md`）。

壁面律里哪些固体算"壁"：`kind: boundary` 且不是 `free_slip` 的粒子、静止薄板的假粒子。转子粒子和转子薄板默认不算
（与转子的对保持 ν_f），`numerics.wall_law_rotor: true` 时算。常数模式（harmonic / arithmetic / solid）对转子对用转子材料
的 ν_s，所以转子材料保持分子值时叶片的粘性力矩不变。无剪切壁（`free_slip`）在任何模式下都没有粘性力。

### 代码

- `common.glsl`：常量 106 `SOLID_PAIR_VISCOSITY_MODE`、107 `WALL_LAW_INCLUDES_ROTOR`；set 0 binding 15 `WallViscosityBuffer`
  （非 wall_law 模式 16 B 占位）；`ThinPlate.flags.z` 存薄板粒子的材料组（假粒子的 ν_s 用它）。
- `density.comp`：wall_law 模式下累计最近壁面粒子的距离、速度、方向和 Σ V_j ∇W_ij，循环后按上式写 ν_eff。
- `force.comp`：函数 `solid_pair_viscosity()`；流体循环里固体邻居（含薄板假粒子）把 `pair_viscosity` 换成 ν_ij；
  薄板假粒子的反作用记录也用同一个 ν_ij；`solid_reaction_acceleration()` 用同一个 ν_ij（流体粒子的 ν_eff 从缓冲读）。
  模式 0 时所有新读取都被特化常量消掉。
- `simulator_v1.py`：常量、缓冲、`density → force` 的计算屏障（wall_law 时）、`readback_wall_viscosity()`、薄板材料组打包。
- `utils/sph/case.py`：`NumericsConfig.solid_pair_viscosity`（校验）、`wall_law_rotor`，`_SPEC_CONSTANT_MAPPING` 106、107。
- 生成器 `_demo_stirred_tank_30l.py`：`--wall-viscosity NU`（只改 `tank_wall`）、`--rotor-viscosity NU`（只改 `impeller`，
  默认等于 `--viscosity`）、`--solid-pair-viscosity MODE`、`--wall-law-rotor`。默认输出除 `materials.yaml` 两行注释外逐字节相同。
- 检查脚本：`_check_tank_energy.py` 的 dump 在 wall_law 模式下多存 `wall_viscosity`；`_analyze_wall_viscous_torque.py` 加
  `--pair-viscosity NU`（solid 模式的重建）、`--dump-viscosity`（用 dump 里的 ν_eff）、`--json`；
  `_check_buoyancy_force.py`、`_check_hydrostatic_drift.py` 加 `--solid-pair-viscosity`、`--wall-viscosity`。
- 集群脚本（仓库外 `cluster_jobs/paratera/`）：`job_wall_calib_shm.sh`（同一算例 Fluent 场起算 0.7 s 加 0.22 s 静止，
  再从 dump 重建粘性力矩）、`submit_1013_wall.sh`。

## 三、回归

本机 RTX 4070 Ti SUPER，改动前的代码在同一提交的独立 worktree 里编译运行。

### 默认构建不变

- 生成器：3 mm 生产配置（`--skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates
  --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2`）改动前后 19 个输出文件里 18 个逐字节相同，
  `materials.yaml` 只差两行注释（数值相同）。
- 二维方腔 `cases/lid_driven_cavity_2d`（1,046,529 粒子，600 步，`--dump`）：改动前两次运行之间排序后的位置差
  最大 1.008、平均 1.43e-3；改动前对改动后最大 1.011、平均 1.47e-3 和 1.26e-3。改动后的差在改动前自身的运行间噪声之内
  （体素接收列表的原子操作顺序不可复现）。alive 1,046,529 相同，溢出计数全零。`_test_common.comp` 手工编译通过。
- `self` 模式就是模式 0，与不写选项走同一段代码。

### Couette（`cases/taylor_couette_2d`，ν = 10⁻³，内筒转子 2 rad/s，外筒静止，dx 1 mm，h/dx 5）

分档剖面在 0.2 / 0.4 / 0.7 / 1.0 / 1.5 s 与改动前比较（最大 |Δu|，mm/s；轮辋速度 100 mm/s），以及
`_check_effective_viscosity.py` 的起动拟合（有效粘度 / 材料粘度）：

| 算例 | 时间步 | 最大 \|Δu\| 对改动前，0.2 … 1.5 s | 1.5 s 剖面对解析解的均方根 | 起动拟合 |
| --- | --- | --- | --- | --- |
| 改动前 self | CFL 0.15 | 0 | 0.26 | 0.910 |
| self | 0.15 | 0.00 0.00 0.01 0.02 0.03 | 0.26 | 0.910 |
| harmonic，ν_s = ν | 0.15 | 0.00 0.01 0.01 0.05 0.03 | 0.26 | 0.910 |
| arithmetic，ν_s = ν | 0.15 | 0.00 0.03 0.05 0.04 0.01 | 0.27 | 0.910 |
| solid，ν_s = ν | 0.15 | 0.00 0.01 0.01 0.02 0.01 | 0.26 | 0.910 |
| wall_law | 0.15 | 0.00 0.01 0.01 0.02 0.01 | 0.26 | 0.910 |
| wall_law，`wall_law_rotor` | 0.15 | 0.00 0.03 0.02 0.03 0.07 | 0.26 | 0.910 |
| self，材料 ν_s = 10 ν（证明 self 不读固体粘度） | 0.15 | 0.00 0.02 0.03 0.09 0.10 | 0.25 | 0.910 |

ν_s = ν 时五种模式与改动前的差都在运行间噪声（约 0.1 mm/s）之内；壁面律在层流 Couette 里 Re_d = u_t d / ν
最大约 0.1 × 0.0005 / 10⁻³ = 0.05 ≪ 139.5，退化为 ν_f，与 self 相同（这正是设计要求）。

ν_s = 10 ν。这个算例的 ν 已经是 10⁻³，对粒子对取 10⁻² 时 CFL 0.15 的时间步（1.9e-4 s）超过显式粘性稳定限
（约 0.125 dx² / ν_ij = 1.3e-5 s）：壁和转子都取 10 ν 的 solid 模式在 CFL 0.15 下发散（溢出 6006 / 118134，KCG 回退 1.28 M）。
下面把 CFL 降到 0.03（dt 3.75e-5）和 0.006（dt 7.5e-6）各算一遍，两档结果一致到 0.05 mm/s，说明已经不受时间步影响：

| 算例（CFL 0.006） | 最大 \|Δu\| 对 self，0.2 … 1.5 s | 1.5 s 均方根 | 内筒第一档 u（解析 95.9） | 外筒最后一档 u（解析 1.69） | 起动拟合 |
| --- | --- | --- | --- | --- | --- |
| self | 0 | 0.30 | 95.7 | 1.78 | 0.910 |
| solid，只有壁 10 ν（转子 ν） | 0.11 0.93 1.74 2.03 2.16 | 1.52 | 95.7 | 0.20 | 0.880 |
| solid，壁和转子都 10 ν | 12.4 9.9 8.8 8.3 8.2 | 3.84 | 101.7 | 0.22 | 1.290 |
| harmonic，都 10 ν（ν_ij = 1.82 ν） | 4.7 3.7 3.3 3.2 3.1 | 1.34 | 98.4 | 1.07 | 1.040 |
| arithmetic，都 10 ν（ν_ij = 5.5 ν） | 10.2 8.2 7.3 6.9 6.8 | 3.14 | 101.1 | 0.40 | 1.220 |
| solid，都 3 ν（CFL 0.03） | 7.5 6.0 5.3 5.1 5.0 | 2.28 | 100.0 | 0.70 | 1.130 |

- 30 L 槽里 ν_w = 100 ν = 10⁻⁴、dx 3 mm 的粘性稳定限约 1.1e-2 s，远大于 dt 6.6e-5 s，不受影响。
- 只把静止外壁的粘度调到 10 ν：外壁第一层流体从 1.78 降到 0.20 mm/s（解析 1.69），即壁面对被过度制动、剖面在壁旁偏离解析解
  约 1.5 mm/s，起动拟合略慢（0.88）；内筒不变。壁面旁的 Morris 对项不是精确的剪切应力，常数倍数是旋钮而不是壁面模型。
- 转子也调到 10 ν 时起动明显变快（拟合 1.29 对 0.91），但内筒第一档 u 超过解析值（101.7 对 95.9，甚至超过轮辋的 100）：
  大粘度的 Morris 对项把第一层流体拖得比刚体转动还快，是沿 x_ij 方向的对力对曲面转动壁的已知近似。这就是转子材料
  默认保持分子值的原因。
- 稳态剖面"不变"只在壁外两三档以外成立；壁旁一两档随 ν_s 变化，幅度 2 到 6 mm/s（轮辋速度的 2 到 6 %）。

### 浮力闭合（`_check_buoyancy_force.py --hdx 3 --reaction --solid-pressure accumulate`，6000 步）

F_y 读回 / (ρ₀ g N dx³)：改动前 1.210，改动后 self 1.213，wall_law（壁 10 ν）1.210，solid（壁 10 ν）1.214；
样本标准差 0.0003 到 0.0011；质量因子 1.2187。全部等于 V_p / dx³ 的 1.21（`log/2026-09-29_torque-readback-mass-factor.md`），
壁面加流体加块体的垂向力都等于流体重量（−20.25 对 −20.27 N）。静止时 u_t = 0，壁面律退化为 ν_f。

### 静止方箱（`_check_hydrostatic_drift.py --dx 0.004 --background 2000`，8000 步 = 0.94 s，p_b 2000 Pa 下重力）

各层粒子数占格子数的百分比、密度范围、最大流速、平均位移在改动前、改动后 self、wall_law、solid（壁 10 ν）四个算例里
都在同一水平（底层 95 到 97 %，顶层 92 到 94 %，密度 999.1 到 1011.3，最大速度 0.27 到 0.30 m/s 的点阵重排噪声），
溢出和 KCG 回退为零。

## 四、标定（并行科技，3 mm，Fluent 25 s 流场起算 0.7 s）

CALIBRATION_PLACEHOLDER

## 五、20 s 从静止

TWENTY_PLACEHOLDER

## 六、结论

CONCLUSION_PLACEHOLDER

## 待合作者确认

CONFIRM_PLACEHOLDER
