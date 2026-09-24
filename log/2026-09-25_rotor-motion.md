# 2026-09-25 转子运动：MATERIAL_ROTOR 粒子的指定刚体旋转

本条记录每一步是怎么做的、为什么这样做、怎么验证的，供合作者核对正确性。
所有改动在 `test` 分支。

## 0. 设计要点（先定的约束）

1. **单向耦合**：转子运动由指定角速度决定，流体不反作用于转子。与 M-Star 一致。
2. **参考位置法**：每步 `x_{n+1} = pivot + R(θ(t_{n+1})) (x_ref − pivot)`，θ 在 CPU 上用 float64 累积，
   不做 GPU 上 float32 的逐步小角度累加（后者几十万步后半径漂移到毫米级）。
3. **命令缓冲预录制**：每步变化的量（cos θ、sin θ、当前 ω）不能走推送常量，放进一个
   host-visible 的小缓冲由 CPU 每步提交前写入。轴向和轴心不变，走特化常量。
4. **参考位置放在 `extension_fields.xyz`**：这个 set 0 缓冲已经存在、已在 defrag 的 9 个拷贝字段里，
   所以重排池时参考位置和粒子一起搬，不需要改 defrag。代价是占用了原本预留给标量的 xyz，
   `.w` 仍空着，底物浓度以后用 `.w` 或新缓冲。
5. **转子在 density.comp 中按壁面处理**：存静止密度、跳过固体-固体邻居对。压力仍来自本步含相对速度
   散度项的 new_density，所以流体逼近叶片时压力上升，机制与静止壁面相同。
6. **力矩从 force.comp 已有输出读回**：force 对 ROTOR 不早退（只对 BOUNDARY、INLET 早退），
   每个转子粒子的加速度已经是流体作用力除以质量，CPU 读回后 Σ m (x − pivot) × (a − g) 沿轴分量即力矩。
   转子-转子对的力反对称，在总和里抵消。

## 1. 改动清单

| 文件 | 改动 |
| --- | --- |
| `experiment/v1/shaders/common.glsl` | 特化常量 56–61：`ROTOR_AXIS_XYZ`、`ROTOR_PIVOT_XYZ`；set 3 binding 9 `RotorStateBuffer`（cos θ、sin θ、ω、t）；`extension_fields` 注释更新 |
| `experiment/v1/shaders/helpers.glsl` | `is_solid_kind(kind)`（BOUNDARY 或 ROTOR）、`rotate_about_axis(v, axis, cos, sin)`（Rodrigues） |
| `experiment/v1/shaders/predict.comp` | 不再对 ROTOR 早退；新增 `is_rotor` 分支：kick/drift 之后用刚体位置和速度覆盖，跨体素检测与流体相同 |
| `experiment/v1/shaders/density.comp` | 两处 `kind == MATERIAL_BOUNDARY` 改为 `is_solid_kind(kind)`（跳过固体对、存静止密度） |
| `utils/sph/case.py` | `RotorConfig`（axis、pivot、ramp_time）；case.yaml 可选 `rotor:` 块；`Case.rotor`、`Case.rotor_angular_velocity`；校验"有 rotor 材料必须有 rotor 块，反之亦然"；`_SPEC_CONSTANT_MAPPING` 加 56–61 |
| `experiment/v1/utils/simulator_v1.py` | `SPEC_ID_ROTOR_*`；`_BufferSpec.host_visible`；`rotor_state` 缓冲（host-visible、coherent、常驻映射）；初始上传 `extension_fields = 初始位置`；`rotor_angle_and_rate(t)`、`_rotor_write_state(t_next)`；`step()` 提交前写 t_{n+1} 的状态（`wait=False` 时先 `vkQueueWaitIdle`）；`readback_velocity_mass / acceleration / material / extension_fields`、`readback_rotor_torque()`；`destroy()` 先 unmap |
| `experiment/v1/_run_v1_headless.py` | `--torque-every N`、`--torque-log PATH`；`--dump` 多存 material 与 velocity_mass |
| `utils/geometry/_demo_stirred_tank_30l.py` | 写 `rotor:` 块（轴 +y、轴心原点、`--ramp-time` 默认 0.05 s）；角速度改为带符号 −20.944 rad/s |
| `utils/geometry/_demo_taylor_couette_2d.py` | 新增：二维环形 Couette 验证算例生成器 |

## 2. 符号约定

- 旋转按右手定则绕 `axis`；`rotor_angular_velocity` 带符号。
- M-Star 的"−200 rpm 绕 +y"对应 ω = −20.944 rad/s，叶片从 +x 转向 +z（θ 增大方向）。
  PBT 叶片 y 随 +θ 下降，所以前缘在下，向下泵送，符合标准 PBT 用法。
- 斜坡：t < T 时 ω(t) = w t/T、θ = w t²/(2T)；t ≥ T 时 ω = w、θ = w (t − T/2)。启动 T 内 GPU 读到的
  ω 与 θ 严格一致，不是分别近似。

## 3. 每步的数据流

1. CPU：`step()` 计算 t_{n+1} = t_n + dt，得 (θ, ω)，把 (cos θ, sin θ, ω, t_{n+1}) 写进映射的 `rotor_state`。
2. GPU predict：ROTOR 粒子读 `extension_fields.xyz` 作 x_ref，Rodrigues 旋转得新位置，速度 = ω · axis × (x − pivot)，
   然后与流体一样算新体素、必要时原子追加到 incoming 列表。
3. update_voxel、correction 不变。density 对 ROTOR 存 ρ₀、压力用本步 new_density。force 对 ROTOR 算加速度（读回用）和 PST 位移（predict 忽略）。
4. 每 N 步 CPU 读回位置、加速度、质量、材料，算力矩。

## 4. 验证

（以下随做随记）

### 4.1 回归：无转子算例不变

- 二维方腔 1,046,529 粒子 300 步：211 步/s，溢出 0。三维方腔 1,295,029 粒子 100 步：34.6 步/s，溢出 0。
  （改动前分别为 232 与 37.6 步/s，差异在运行间波动范围内，predict 多了一个分支判断。）

### 4.2 刚体检查：30 L 槽 3 mm 档，转子转动 2000 步

设置：ω = −20.944 rad/s，斜坡 0.05 s，dt = 1.313×10⁻⁴ s，2000 步 = 0.2626 s，θ = −4.976 rad（0.79 圈）。
中途在第 1000 和 2000 步各做了一次 defrag（cadence 1000），所以这个检查同时覆盖了"参考位置随 defrag 搬运"。

分析脚本：把初始 `rotor.obj` 绕 +y 旋转 θ，与导出的转子粒子做最近邻匹配。

| 检查项 | 结果 |
| --- | --- |
| 粒子数 | 7785 → 7785 |
| 与旋转后初始位置的最近邻距离 | 最大 1.9×10⁻⁵ m，均值 8.7×10⁻⁶ m（对应打印角度只有 3 位小数的截断，Δθ ≈ 4×10⁻⁴ rad） |
| 匹配是否一一对应 | 是（7785 个唯一索引） |
| 到轴距离多重集 | 排序后逐项最大差 3.1×10⁻⁹ m |
| 高度 y 多重集 | 排序后逐项最大差 1.5×10⁻⁸ m |
| 速度与 ω·axis×(x−pivot) 的差 | 最大 7.9×10⁻⁸ m/s |
| 溢出计数 | 全 0，粒子数守恒 1,241,257 |

结论：参考位置法在 defrag 跨越后仍精确，半径和高度不漂移，速度场与刚体运动一致。

力矩读回（每 400 步一次）：

| 步 | t (s) | 轴向力矩 (N·m) | 轴向力 F_y (N) |
| --- | --- | --- | --- |
| 400 | 0.0525 | 0.0999 | −0.69 |
| 800 | 0.1050 | 0.0919 | −0.35 |
| 1200 | 0.1575 | 0.0858 | −0.17 |
| 1600 | 0.2101 | 0.0901 | −0.41 |
| 2000 | 0.2626 | 0.0874 | −0.36 |

量级检查：P = τω ≈ 0.09 × 20.94 ≈ 1.9 W，ρN³D⁵ = 998 × 3.333³ × 0.096⁵ = 0.300，Np ≈ 6.3。
Rushton 约 5 加 PBT 约 1.3 的和在 6 左右，量级合理。但这只是启动后不到一圈的瞬时值，流场未发展，
正式的功率数要跑到统计定常再时间平均，并做分辨率收敛。F_y 为负表示 PBT 向下泵送时流体对转子的反作用向下，方向合理。

速度：带转子 300 步 76.6 步/s（静止时 80）。带 `--torque-every 400` 时降到 43 步/s，
是每次读回四个整缓冲（各约 20 MB）经 staging 拷贝的开销，正式运行时把间隔放大到几千步即可。

### 4.3 Taylor–Couette 二维层流：与解析解对比

算例 `cases/taylor_couette_2d`：内筒 r_i = 0.05 m 为 ROTOR，Ω = 2 rad/s（斜坡 0.1 s），外筒 r_o = 0.10 m 静止，
ν = 10⁻³ m²/s，Re = 5，dx = 1 mm，h/dx = 5，c₀ = 4 m/s，dt = 1.875×10⁻⁴ s。
流体 23,096、壁面 3,504、转子 1,660 粒子。跑 40,000 步 = 7.5 s = 3 个粘性时间 gap²/ν，2386 步/s。

解析解 u_θ = A r + B/r，A = Ω r_i²/(r_i² − r_o²)，B = −A r_o²。
把流体粒子按半径分 25 箱取切向速度平均：

- 最大偏差 0.25% × u_内筒（0.1 m/s），均方根 0.13%。
- 偏差符号：内筒附近略低（−0.25%），中部 −0.16%，外壁附近 +0.1%。剖面整体略平，与核平滑和壁面附近核截断一致。
- 箱内标准差 1×10⁻³ m/s（1% u_内筒），径向速度残差 2×10⁻⁴ m/s。
- 溢出 0，粒子守恒。图：`cases/taylor_couette_2d/validation_profile.png`。

结论：转子的无滑移条件通过粘性项正确传给流体，稳态剖面与解析解一致到 0.3% 以内。

## 5. 尚未做 / 供合作者核对的点

1. `density.comp` 对 ROTOR 与 BOUNDARY 同样处理这一决定：ROTOR 的压力来自本步 new_density（含相对速度散度项），
   是否与作者对运动壁面的设想一致，请核对。若改为 Adami 式压力外推，改动点只在 density.comp。
2. force.comp 对 ROTOR 仍计算 PST 位移（predict 忽略），只是浪费算力；若想省掉，在 force 里对 ROTOR 跳过 PST 段即可。
3. `extension_fields.xyz` 被占用为参考位置。后续标量输运请用 `.w` 或新建 set 0 缓冲并加进 defrag 拷贝列表。
4. 力矩包含转子-壁面（轴穿过顶盖、底板处）的邻居对贡献，量小但非零；需要时在 force 或读回后剔除。
5. 三维搅拌槽的功率数验证（长时间运行、时间平均、分辨率收敛）还没做，是下一步。
6. 多个转子（不同轴或角速度）不支持：一个 case 只有一组轴/轴心/角速度。

## 6. 复现验证的命令

```
python utils/geometry/_demo_stirred_tank_30l.py --dx 0.003 --hdx 3 --out cases/stirred_tank_30l_3mm --no-preview
python experiment/v1/_run_v1_headless.py cases/stirred_tank_30l_3mm/case.yaml --max-steps 2000 --torque-every 400 --dump tank.npz
python experiment/v1/checks/_check_rotor_rigid.py tank.npz cases/stirred_tank_30l_3mm/rotor.obj 2 <angle printed at step 2000> -20.94395

python utils/geometry/_demo_taylor_couette_2d.py --dx 0.001
python experiment/v1/_run_v1_headless.py cases/taylor_couette_2d/case.yaml --max-steps 40000 --dump couette.npz
python experiment/v1/checks/_check_taylor_couette.py couette.npz couette.png
```

转子材料的 group id 是该材料在 case.yaml `particles` 列表中首次出现的顺序号（搅拌槽算例为 2）。
