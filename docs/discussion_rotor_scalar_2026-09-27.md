# 转子与标量输运：实现、遇到的问题、需要合作者确认的事项

日期：2026-09-27　分支：`test`　代码状态：76d273b（标量输运）及之前的转子相关提交

这份文档用于和合作者讨论。内容按"怎么实现的、遇到了哪些坑、哪些地方需要他确认"组织，
每个公式后面紧跟对应的代码（取自当前 test 分支），方便逐条核对。第 5 节的汇总表可以直接当作讨论提纲。

## 0. 阅读说明

### 0.1 结构

| 节 | 内容 |
| --- | --- |
| 1 | 转子的实现：刚体运动、固体处理、力矩与功率数，公式与代码对照 |
| 2 | 转子相关的坑，按发现顺序：现象、原因、处理、影响 |
| 3 | 标量输运的实现：扩散、归一化、SGS、Kahan、注入、探针，公式与代码对照 |
| 4 | 标量输运开发中遇到的问题 |
| 5 | 需要合作者确认的事项，含汇总表 |
| 6 | 下一步 |

### 0.2 详细记录

本文只做摘要。数字和推导的出处如下，都在 `log/` 目录：

| 主题 | 日志 | 提交 |
| --- | --- | --- |
| 管线创建随机失败 | 2026-09-24_pipeline-stage-keepalive.md | 2f1340a |
| 30 L 槽生成器 | 2026-09-25_stirred-tank-30l-case.md | 3363ece |
| 转子刚体旋转与力矩 | 2026-09-25_rotor-motion.md | 907b632 |
| 功率数算例（渗漏、声速、残留槽位、分辨率） | 2026-09-25_np-campaign.md | 01d9286 起 |
| defrag 间隔 | 2026-09-25_defrag-cadence.md | cf1216f、f99cbfa |
| 转向修正 | 2026-09-26_rotor-direction-fix.md | f0505d4 |
| 分桨力矩 | 2026-09-26_per-impeller-torque.md | ce086b1 |
| 碟形槽底 | 2026-09-26_dished-bottom.md | 5abaed8 |
| 转子力矩的组成、固体压力 | 2026-09-27_rotor-torque-pair-forces.md | 21025dc |
| 标量输运 | 2026-09-27_scalar-transport.md | 76d273b |
| 标量输运审核清单（原代码、改动、影响、代码对比） | 2026-09-27_scalar-transport-review.md | 与本文同一提交 |

### 0.3 符号

| 符号 | 含义 | 代码中的名字 |
| --- | --- | --- |
| h | 核支撑半径（不是光滑长度），搅拌槽 h = 3 dx | `physics.h`、`SMOOTHING_LENGTH` |
| dx | 粒子间距 | `2 * particle_radius` |
| d | 维数 | `DIMENSION` |
| x_ij | x_i − x_j | `position_difference_neighbor_to_self` |
| ∇W_ij | 对 x_i 求导的原始核梯度 | `kernel_gradient_raw`、`raw_gradient` |
| M_i | KCG 矩阵 Σ_j V_j (x_j − x_i) ⊗ ∇W_ij | 只在 correction.comp 里出现 |
| (M_i + ξI)⁻¹ | correction.comp 存储的逆，ξ = 0.1 | `self_correction_matrix`（名字是 matrix，内容是逆） |
| ∇W̃_ij | 修正后的核梯度 (M_i + ξI)⁻¹ ∇W_ij | `kernel_gradient_corrected`、`corrected_gradient` |
| k、p | 转轴单位向量、轴心 | `ROTOR_AXIS_*`、`ROTOR_PIVOT_*` |
| θ、ω | 转角、角速度（右手定则绕 k） | `rotor_cos_theta`、`rotor_angular_velocity_now` |

代码注释里有时把存储的逆也写作 M_i，本文统一写成 (M_i + ξI)⁻¹，以免混淆。
代码位置按文件和函数给出，行号对应 76d273b。

## 1. 转子的实现

### 1.1 设计要点

1. **单向耦合**：转子按给定角速度转动，流体不反作用于转子，与数据集 M-Star 的设置相同。
2. **参考位置法**：每步都从初始位置直接旋转到当前角度，角度在 CPU 上用 float64 计算。
   不在 GPU 上用 float32 逐步累加小角度，因为那样几十万步后转子半径会漂移到毫米级。
3. **随时间变化的量放在 host-visible 小缓冲里**：步进命令缓冲是预录制的，不能每步改推送常量。
   轴向和轴心不变，作为特化常量；cos θ、sin θ、ω 由 CPU 每步提交前写入。
4. **参考位置存在 `extension_fields.xyz`**：这个 set 0 缓冲原本就在 defrag 的搬运列表里，所以不用改 defrag。
5. **density.comp 把转子当作壁面**：存静止密度，跳过固体与固体的粒子对。
6. **force.comp 不对转子早退**：转子粒子的加速度就是邻居作用力除以质量再加重力，CPU 读回后求力矩。
7. **一个算例只有一个转子**：所有 rotor 类材料共用一组轴、轴心和角速度。

### 1.2 输入

case.yaml 的 `rotor:` 块和材料库中 rotor 类材料的角速度（搅拌槽生成器的输出）：

```yaml
# case.yaml
rotor:
  axis: [0.0, 1.0, 0.0]
  pivot: [0.0, 0.0, 0.0]
  ramp_time: 0.0500      # s, linear spin-up (M-Star used 0.0092 s)

# materials.yaml
impeller:
  kind: rotor
  rest_density: 998.0
  viscosity: 1.0e-6
  rotor_angular_velocity: 20.94395   # rad/s，右手定则绕 +y，即 M-Star 的 -200 rpm（从上往下看逆时针）
```

- `utils/sph/case.py` 的 `RotorConfig` 读 `rotor:` 块，并校验"有 rotor 材料就必须有 rotor 块，反之亦然"。
- 轴和轴心作为特化常量 56–61 传入（common.glsl 的 `ROTOR_AXIS_X` 到 `ROTOR_PIVOT_Z`）。
- 转向约定见 1.8 节，这是我们踩过的坑之一。

### 1.3 转角和角速度：线性斜坡启动，CPU 上 float64 计算

公式（w 为材料的带符号角速度，T 为 `ramp_time`）：

```
t <  T:   ω(t) = w t / T,    θ(t) = w t² / (2T)
t >= T:   ω(t) = w,          θ(t) = w (t − T/2)
```

θ 是 ω 的精确积分，所以启动阶段 GPU 读到的角度和角速度严格对应。

代码（`simulator_v1.py`，`rotor_angle_and_rate` 第 1224 行、`_rotor_write_state` 第 1239 行）：

```python
def rotor_angle_and_rate(self, time: float) -> tuple[float, float]:
    if self.case.rotor is None:
        return 0.0, 0.0
    w = float(self.case.rotor_angular_velocity)
    T = float(self.case.rotor.ramp_time)
    if T <= 0.0 or time >= T:
        return w * (time - 0.5 * T), w
    return w * time * time / (2.0 * T), w * time / T

def _rotor_write_state(self, time_next: float) -> None:
    if "rotor_state" not in self._mapped:
        return
    theta, omega = self.rotor_angle_and_rate(time_next)
    payload = struct.pack("ffff", math.cos(theta), math.sin(theta), omega, time_next)
    self._mapped["rotor_state"][:16] = payload
    self.rotor_angle = theta
```

每步提交前写入 t_{n+1} 时刻的状态（`step()`）。异步提交时先等上一步结束，避免 GPU 还在读的时候被改写：

```python
def step(self, *, wait: bool = True) -> None:
    cmd = self.step_cmd
    if self.case.rotor is not None or self._injection_staging is not None:
        if not wait:
            vkQueueWaitIdle(self.ctx.compute_queue)
        time_next = self.simulation_time + self.case.timestep
        if self.case.rotor is not None:
            self._rotor_write_state(time_next)
        ...
```

GPU 端的缓冲（common.glsl，set 3 binding 9，host-visible、coherent、常驻映射）：

```glsl
layout(std430, set = 3, binding = 9) buffer RotorStateBuffer {
    float rotor_cos_theta;
    float rotor_sin_theta;
    float rotor_angular_velocity_now;
    float rotor_time;
};
```

### 1.4 predict.comp：刚体位置和速度

公式（Rodrigues 旋转）：

```
x_{n+1} = p + R(θ_{n+1}) (x_ref − p)
R(θ) a  = a cos θ + (k × a) sin θ + k (k·a)(1 − cos θ)
v_{n+1} = ω_{n+1} k × (x_{n+1} − p)
```

x_ref 是粒子的初始位置，上传时写进 `extension_fields.xyz`（`simulator_v1.py` 的上传部分）：

```python
# ---- extension_fields.xyz = reference (initial) position --------------
reference = positions.copy()
reference[:, 3] = 0.0
data["extension_fields"] = reference.tobytes()
```

旋转函数（helpers.glsl 第 286 行）：

```glsl
vec3 rotate_about_axis(vec3 vector, vec3 axis, float cos_theta, float sin_theta) {
    return vector * cos_theta
         + cross(axis, vector) * sin_theta
         + axis * dot(axis, vector) * (1.0 - cos_theta);
}
```

predict.comp：BOUNDARY 和 INLET 早退；ROTOR 先照常做 kick 和 drift，再被刚体结果覆盖（第 124–132 行）：

```glsl
if (self_params.kind == MATERIAL_BOUNDARY) return;   // static walls
if (self_params.kind == MATERIAL_INLET)    return;   // emitter templates
bool is_rotor = (self_params.kind == MATERIAL_ROTOR);
...
vec3 new_velocity = self_velocity_half + self_acceleration * TIMESTEP;           // 流体：v_{n+1/2}
vec3 new_position = self_position + new_velocity * TIMESTEP + self_shift;        // 流体：x_{n+1}
if (is_rotor) {
    vec3 rotor_axis  = normalize(vec3(ROTOR_AXIS_X, ROTOR_AXIS_Y, ROTOR_AXIS_Z));
    vec3 rotor_pivot = vec3(ROTOR_PIVOT_X, ROTOR_PIVOT_Y, ROTOR_PIVOT_Z);
    vec3 reference_arm = extension_fields[self_particle_id].xyz - rotor_pivot;
    vec3 rotated_arm   = rotate_about_axis(reference_arm, rotor_axis,
                                           rotor_cos_theta, rotor_sin_theta);
    new_position = rotor_pivot + rotated_arm;
    new_velocity = rotor_angular_velocity_now * cross(rotor_axis, rotated_arm);
}
// 之后的跨体素检测、原子追加与流体完全相同
```

- 转子的 acceleration 和 shift 都不影响它的运动。
- 刚体速度写进 `velocity_mass.xyz`，所以 density 的连续方程和 force 的粘性项自动看到运动壁面，无滑移条件不需要额外处理。
- 时间层的细节：转子存的是 t_{n+1} 的刚体速度，流体存的是 v_{n+1/2}。两者方向差 ω dt/2，
  3 mm 档 c₀ = 20 U_tip 时约 7×10⁻⁴ rad，可以忽略。

### 1.5 density.comp：转子按固体处理

公式。固体粒子 s（壁面和转子）每步把存储的密度重置为 ρ₀，压力只由本步的增量决定：

```
存储：ρ_s ← ρ₀
压力：P_s = B [ ((ρ₀ + dt · (Dρ/dt)_s) / ρ₀)^γ − 1 ],   B = ρ₀ c₀² / γ,  γ = 7
(Dρ/dt)_s = Σ_{j∈流体} [ −ρ₀ V_j (v_j − v_s)·∇W̃_sj
                        + δ h c₀ · 2 (ρ_j − ρ₀) (x_js·∇W̃_sj) / (r² + ε_h²) · V_j ]
```

求和只包括流体邻居，因为固体与固体的粒子对被跳过。对转子来说 v_s 是刚体速度，
所以流体逼近叶片时漂移项为正，叶片前方的压力升高，机制和静止壁面相同。

代码（density.comp 第 177–180 行和第 300–303 行）。这两处原来写的是 `kind == MATERIAL_BOUNDARY`，转子提交改成了 `is_solid_kind`：

```glsl
bool is_solid_kind(uint kind) {                       // helpers.glsl 第 280 行
    return kind == MATERIAL_BOUNDARY || kind == MATERIAL_ROTOR;
}

// 邻居循环内：固体自身跳过固体邻居
if (is_solid_kind(self_params.kind)) {
    uint neighbor_group_id = material[neighbor_particle_id];
    if (is_solid_kind(material_parameters[neighbor_group_id].kind)) continue;
}

// 循环后：固体存 ρ₀，压力用本步的 new_density
float density_to_store = is_solid_kind(self_params.kind)
    ? self_params.rest_density
    : new_density;
density_pressure_scratch[self_particle_id] = vec2(density_to_store, new_pressure);
```

这个固体压力模型是合作者原有的设计（density.comp 的注释称之为 legacy 的"property buffer = ρ₀、runtime buffer = 每步压力"），
转子只是沿用。它对功率数和渗漏的影响见 5.1 节第 1 条。

### 1.6 force.comp：转子粒子照常计算加速度

force.comp 只对 BOUNDARY 和 INLET 早退（第 124–126 行），转子粒子和流体走同一段代码，对所有邻居（流体、壁面、其他转子粒子）求和：

```glsl
if (self_params.kind == MATERIAL_BOUNDARY) return;
if (self_params.kind == MATERIAL_INLET)    return;
...
// 邻居循环内（第 315–328 行），neighbor_volume_dynamic = V_j = m_i / ρ_j：
float pressure_combined =
    (self_pressure > 0.0 || near_surface)
        ? (neighbor_pressure + self_pressure)        // P_i > 0 或 kernel_sum < 0.75：对称形式
        : (neighbor_pressure - self_pressure);       // 否则：反对称形式（TIC）
a_pressure -= neighbor_volume_dynamic * pressure_combined / self_density
            * kernel_gradient_corrected;
a_viscosity += morris_coefficient * viscosity_uniform * neighbor_volume_dynamic
             * dot(self_velocity - neighbor_velocity, position_difference_neighbor_to_self)
             / (distance * distance + EPS_H_SQUARED) * kernel_gradient_corrected;
...
vec3 total_acceleration = a_pressure + a_viscosity + gravity_vector;   // 转子粒子也加了重力
acceleration[self_particle_id] = vec4(total_acceleration, vorticity_z);
shift[self_particle_id]        = vec4(particle_shift, 0.0);            // 转子的 PST 位移算了，但 predict 不用
```

对应的公式：

```
a_i = − Σ_j V_j F_ij / ρ_i ∇W̃_ij  +  Σ_j 2(d+2) ν V_j (v_i − v_j)·x_ij / (r² + ε_h²) ∇W̃_ij  +  g
F_ij = P_i + P_j   （P_i > 0，或 kernel_sum_i < 0.75）
F_ij = P_j − P_i   （其他情况）
V_j = m_i / ρ_j,   ∇W̃_ij = (M_i + ξI)⁻¹ ∇W_ij,   ε_h² = EPS_H_SQUARED = 0.01 h²
```

### 1.7 力矩和功率数

公式。粒子 i 受到的邻居作用力为 f_i = m_i (a_i − g)，转子受到的合力和绕轴力矩为：

```
F  = Σ_{i∈转子} m_i (a_i − g)
τ  = k · Σ_{i∈转子} (x_i − p) × m_i (a_i − g)
Np = |τ| ω / (ρ N³ D⁵),   ρ = 998 kg/m³,  N = 200/60 rev/s,  D = 0.096 m（Rushton 直径）
```

代码（`simulator_v1.py` 的 `readback_rotor_torque`，第 1404 行；Np 在 `experiment/v1/checks/_analyze_power_number.py` 里算）：

```python
groups = np.asarray(self.rotor_group_ids(), dtype=np.uint32)
material = self.readback_material()
is_rotor = np.isin(material, groups)
is_rotor[0] = False                                  # 0 号槽位是哨兵
positions = self.readback_positions()
alive = self.live_slot_mask(positions)               # 去掉 defrag 后的残留槽位，见 2.6 节
sel = is_rotor & alive
x = positions[sel, :3].astype(np.float64)
a = self.readback_acceleration()[sel, :3].astype(np.float64)
m = self.readback_velocity_mass()[sel, 3].astype(np.float64)
g = np.asarray(self.case.physics.gravity, dtype=np.float64)
force_per_particle = (a - g) * m[:, None]
arm = x - pivot
torque_per_particle = np.cross(arm, force_per_particle)
torque = torque_per_particle.sum(axis=0)
result = {"force": force_per_particle.sum(axis=0), "torque": torque,
          "torque_axis": float(np.dot(torque, axis)), ...}
```

- 这个函数的文档字符串写着"转子与转子粒子对的力反对称，求和时抵消"。严格说这不成立，
  因为 ∇W̃ 只用自身的 (M_i + ξI)⁻¹，而且 TIC 按自身压力的符号切换形式。实测影响见 5.1 节第 2 条。
- **分桨统计**（ce086b1）：给出 `split_height` 和 `shaft_radius` 时，按到轴的距离 r 和沿轴高度分组：
  r < 7 mm 算轴；其余粒子高度 < 0.1 m 算下桨（Rushton），≥ 0.1 m 算上桨（PBT）。三组之和严格等于总力矩。
  7 mm 和 0.1 m 都落在零件之间的空隙里（轴半径 4 mm，轮毂半径 10.2 和 10.9 mm；Rushton 在 y = 0.021–0.048 m，PBT 在 y = 0.180–0.209 m）。

```python
if split_height is not None and shaft_radius is not None:
    height = arm @ axis
    radial = np.linalg.norm(arm - np.outer(height, axis), axis=1)
    axial_torque = torque_per_particle @ axis
    is_shaft = radial < shaft_radius
    is_lower = ~is_shaft & (height < split_height)
    is_upper = ~is_shaft & (height >= split_height)
```

### 1.8 转向约定

- 数据集的 M-Star 输入文件 input.xml 写的是 `freq = -200`（rpm），`rotationAxis = (0, 1, 0)`，重力 `(0, -9.81, 0)`。
- M-Star 的转速符号不是右手定则。它的文档（Moving Bodies 一节）规定，正转速表示沿重力方向看过去是顺时针。
  所以 −200 rpm 是从上往下看逆时针，也就是右手定则下绕 +y 的**正**角速度 +20.944 rad/s。
- CAD 中 6 片 PBT 叶片都是"沿 +θ 方向 y 降低"。按上面的转向，叶片的前缘是高边，所以 PBT 向下泵，这是标准的下压式用法。
- 修正后 PBT 平面核心区的轴向速度为 −0.28 m/s；修正前为 +0.24 m/s，即向上泵。

### 1.9 验证和主要结果

| 检查 | 条件 | 结果 |
| --- | --- | --- |
| 刚体运动 | 3 mm 平底槽，2000 步，0.79 圈，中间跨过 2 次 defrag | 7785 个转子粒子一一对应；到轴距离排序后逐项最大差 3.1×10⁻⁹ m，高度 1.5×10⁻⁸ m，速度与 ωk×(x−p) 最大差 7.9×10⁻⁸ m/s |
| 二维 Taylor–Couette | r_i = 0.05 m 内筒为转子，Ω = 2 rad/s，Re = 5，dx = 1 mm，h/dx = 5，7.5 s | 切向速度与解析解 u = Ar + B/r 的最大偏差为内筒速度的 0.25%，均方根 0.13% |
| 无转子算例回归 | 二维、三维方腔 | 活粒子数、溢出计数不变 |

Couette 的稳态剖面与粘度无关，所以它验证的是无滑移条件的传递，不能验证粘性项的量值（见 5.2 节第 3 条）。

功率数（碟底几何、正确转向、c₀ = 20 U_tip、h/dx = 3、30 s，取 20–30 s 平均，都按 D = 0.096 m 计算）：

| 算例 | 叶片厚度 | 总 Np | Rushton | PBT |
| --- | --- | --- | --- | --- |
| 4 mm | 12 mm | 6.24 ± 0.27 | 2.46 | 3.78 |
| 3 mm | 9 mm | 5.25 ± 0.17 | 2.47 | 2.78 |
| 3 mm，叶片 2 层 | 6 mm | 5.53 ± 0.16 | 2.61 | 2.92 |
| 2 mm | 6 mm | 5.45 ± 0.17 | 2.92 | 2.53 |
| 3 mm，只有 Rushton | 9 mm | 2.76 ± 0.11 | 2.76 | — |
| 3 mm，只有 PBT | 9 mm | 3.12 ± 0.11 | — | 3.12 |
| 数据集的 LES | 实际厚度 2.2–2.5 mm | M-Star 4.3–4.5，Fluent 5.0–5.5 | — | — |

- 总 Np 在 3 mm 和 2 mm 之间相差不到 4%，已收敛，落在 Fluent 的范围内，比 M-Star 高约 20%。
- 分桨的功率分配没有收敛：从 3 mm 到 2 mm，Rushton +12%，PBT −13%，两者正好抵消。见 2.10 节。
- 表中的数字直接来自力矩读回，没有修正。读回比守恒的估计高 4–5%（5.1 节第 2 条），修正后 3 mm 约 5.0，2 mm 约 5.2。

## 2. 转子相关的坑（按发现顺序）

### 2.1 管线创建随机返回 VK_ERROR_UNKNOWN（与转子无关，但影响所有机器）

- **现象**：在 4070 Ti SUPER（驱动 616.92，python-vulkan 1.3.275.1）上，构造 SphSimulatorV1 时 `vkCreateComputePipelines`
  随机失败，每次失败的 kernel 不同；二维方腔偶发，三维方腔几乎必现。SPIR-V 能通过 spirv-val。
- **原因**：`_build_compute_pipelines` 在循环里反复给局部变量 `stage` 赋值。`pName="main"` 对应的 C 字符串由 Python 的 stage 对象持有，
  对象被回收后指针悬空，驱动读到的是被复用的内存。旁证：循环里最后一个管线和单独创建的 defrag 管线从来不失败，因为它们的 stage 对象还活着。
- **处理**（2f1340a，`simulator_v1.py` 第 878 行起）：

```python
create_infos = []
stage_keepalive = []                     # 新增
for name in SHADER_NAMES_HOT:
    stage = VkPipelineShaderStageCreateInfo(
        stage=VK_SHADER_STAGE_COMPUTE_BIT,
        module=self.shader_modules[self._module_name_for_pipeline(name)],
        pName="main",
        pSpecializationInfo=self.spec_info_global,
    )
    stage_keepalive.append(stage)        # 新增：保活到 vkCreateComputePipelines 返回之后
    create_infos.append(VkComputePipelineCreateInfo(stage=stage, layout=self.pipeline_layout))
result = vkCreateComputePipelines(self.ctx.device, VK_NULL_HANDLE,
                                  len(create_infos), create_infos, None)
```

- **影响**：合作者其他分支里如果有同样的循环写法，换驱动或换机器后也可能随机失败。

### 2.2 流动发展后越跑越慢：defrag 间隔从 1000 改为 10

- **现象**：3 mm 槽在 4070 Ti SUPER 上静止时 80 步/s，流场发展后降到 23 步/s。correction、density、force 三个邻居循环同时变慢；
  predict 的原子操作不到 3%，不是原因。显卡频率和温度正常。
- **原因**：defrag 之后编号相邻的粒子在同一个体素，一组线程读同一批邻居，访存合并。粒子跨体素时只更新名单、不搬数据，
  一组线程逐渐分属不同体素，各自随机访存。桨叶附近的粒子 1000 步要移动约 40 个间距。
- **实验**（6000 步，每 1000 步测一次，步/s）：

| 步 | 间隔 1000 | 间隔 100 | 间隔 10 | 间隔 1 |
| --- | --- | --- | --- | --- |
| 1000 | 47.7 | 71.9 | 72.7 | 55.0 |
| 3000 | 40.0 | 68.8 | 70.1 | 54.3 |
| 6000 | 29.3 | 62.9 | 70.1 | 53.4 |

- **处理**（f99cbfa）：搅拌槽生成器写 `defrag_cadence: 10`。defrag 只重排存储，不改变任何粒子量。一次 defrag 约 4 ms，每 10 步一次的开销约 4%。
- **影响**：所有全域运动的算例都适用，流动发展后速度约为原来的 2.4 倍。可以考虑作为独立的 PR 提给主线。

### 2.3 薄件要加厚到至少 3 层粒子

- 叶片、圆盘、挡板实际厚 2.2–2.65 mm，比粒子间距还薄。生成器把它们加厚到至少 `--thin-layers`（默认 3）个间距：
  3 mm 档为 9 mm，2 mm 档为 6 mm。只有一两层粒子时挡不住穿透。这一点要写进论文的限制。
- 叶片厚度的影响（碟底，3 mm）：从 9 mm 减到 6 mm，两个桨的 Np 都升 5–6%，小于分辨率的影响（见 2.10 节）。
- 壁面层数取 h/dx 层，刚好覆盖核支撑。h/dx = 4 时渗漏反而严重得多（c₀ = 10 U_tip 下丢 461 个粒子，2810 个越过顶盖），
  说明"壁厚等于核半径"在更宽的核下更脆弱。这也是采用 h/dx = 3 的理由之一。

### 2.4 声速 c₀ = 10 U_tip 不够：顶盖渗漏，Np 偏低约 20%

- **现象**（c₀ = 10 U_tip，3 mm，重力关）：30 s 内约 260 个流体粒子（0.03%）穿过 3 层顶盖，集中在轴周围 r = 18–36 mm 的环带，
  也就是顶盖下方的涡核区。顶盖下最后一层流体粒子比内部层多 25%，流体被压向顶盖。
- **机理（推断）**：重力关时没有背景静压，涡核压力为负。TIC 只在自身压力为负时切换到反对称形式；
  流体压力为正、顶盖粒子压力为负时，P_i + P_j 可以为负，流体被吸向壁面。固体压力的模型见 5.1 节第 1 条。
- **声速序列**（3 mm，当时是平底和反转向，但趋势可用）：

| c₀ / U_tip | Mach | Np | 渗漏 |
| --- | --- | --- | --- |
| 10 | 0.10 | 4.02 ± 0.12（重复一次 3.96 ± 0.11） | 有 |
| 20 | 0.05 | 5.29 ± 0.16 | 无 |
| 40 | 0.025 | 5.04 ± 0.21 | 无 |

- **处理**：30 L 算例统一用 c₀ = 20 U_tip，时间步减半。加厚顶盖和开重力两条路都不再走。
- **注意**：生成器 `--c0-factor` 的默认值仍然是 10，我们所有的算例都显式传 20。是否把默认值改成 20，可以讨论。

### 2.5 开重力的算例在 23.5 s 发散（未解决）

- 4 mm、c₀ = 20 U_tip、g = 9.81 m/s²，没有做静压初始化。前 23.5 s 正常（瞬时 Np 中位数约 4.4，轴向力 −1.8 N，与转子排开体积的浮力同量级），
  之后发散：活粒子从 557,622 掉到 154,180，overflow_inside、overflow_incoming、correction_fallback 都达到百万以上。
- 原因不明，力矩日志里看不到前兆。候选：顶盖附近仍有拉伸态；重力使底部压缩，体素占用逼近 max_per_voxel = 64；incoming 上限 32。
- 目前重力一律关闭。闭口单相槽里重力只增加静水压，但关掉重力也就没有了能压住负压的背景压力。

### 2.6 defrag 后的残留槽位让力矩变成 NaN

- **现象**：上面开重力的算例发散后，力矩变成 NaN。
- **原因**：defrag 只把存活粒子打包到槽位 [1, alive_count] 并拷回这一段，尾部保留 defrag 之前的旧数据（voxel_id 非零，
  但不在任何体素列表里，kernel 看不到）。一旦有粒子被杀死，用 `voxel_id > 0` 判断存活就会把旧的转子副本也算进来。
- **处理**（da9c96d）：`live_slot_mask()` 按 GlobalStatus 的 alive_particle_count 截断：

```python
alive_count = int(self.readback_global_status()["alive_particle_count"])
mask = np.zeros(positions.shape[0], dtype=bool)
mask[1:alive_count + 1] = True
mask &= positions[:, 3] > 0
```

- **影响**：没有丢粒子的算例不受影响，可用"转子粒子数等于初始值"来判断。任何按槽位统计的读回都应该用这个掩码。

### 2.7 集群上 NVIDIA 着色器缓存的文件锁

- **现象**：两个作业同时启动时，其中一个一小时不占 GPU，也没有输出。
- **原因（判断）**：家目录在 NFS 上，NVIDIA 着色器磁盘缓存的文件锁冲突。
- **处理**：作业脚本里加 `export __GL_SHADER_DISK_CACHE_PATH=/tmp/nvcache_$SLURM_JOB_ID`。
- 同一集群上另外两个容易误判的现象：存储服务器时钟比计算节点慢约 35 分钟，不能用文件修改时间判断作业是否停滞；
  Slurm 的 cgroup 让每个作业都把自己的卡看成 0 号。

### 2.8 转向反了：M-Star 的转速符号不是右手定则

- **现象**：用户指出 PBT 应当向下压液体。检查发现 2026-09-25 到 26 的所有搅拌槽算例都在向上泵。
- **原因**：见 1.8 节。生成器按右手定则理解 −200 rpm，给了 ω = −20.944 rad/s。
- **处理**（f0505d4）：ω = +2π·200/60，生成器和 materials.yaml 的注释写明了推导。
- **影响**：之前两轮 Np 序列只能作为分辨率、核半径、声速的趋势参考。同为 2 mm 时，上泵比下泵的 Np 高 6%。
- 仓库里提交的 3 mm 算例 case.yaml 顶部注释当时没有随之更新，仍写着旧的转向解释（materials.yaml 的符号是对的）。本次提交一并改正。

### 2.9 槽底建错：平底应为碟形封头

- **现象**：检查剖面图时发现 Rushton 离底只有约 4 cm，而论文图 1 标注为 h₂ = d₂ = 0.096 m。
- **原因**：第一次从 StaticBody.stl 量尺寸时，把 y = 0 当成了槽底，漏掉了下面的碟形封头。
- **处理**（5abaed8）：按 STL 测出的 18 点底面轮廓 `FLOOR_PROFILE` 建碟底（球冠加过渡圆角），加上中心轴承座，挡板改为从 y = 0.010 m 起。
- **结果**：液体体积 27.8 → 30.6 L，Rushton 离底 0.40 D → 1.05 D，3 mm 档粒子数 1,241,257 → 1,360,038；Rushton 的 Np 升了 28%。
- **影响**：之前所有 Np 数字只对平底几何成立，不能和数据集比较。泄漏检查脚本里"流体低于 y = 0"一项也要改成按底面轮廓判断。

### 2.10 分桨功率没有收敛

- 总 Np 已经收敛，但两个桨各自的功率一个偏低一个偏高。单独运转时，Rushton 为 2.76，约为文献估计 4.4–4.8 的 60%；
  PBT 按自身直径为 2.78，约为文献值 1.6–1.7 的 1.7 倍。
- 两桨相互作用：双桨时两个桨都比单独运转低约 11%，与文献 0–10% 同量级，可以排除"相互作用使 Rushton 减半"。
- 可能原因：Rushton 叶片高 19 mm，3 mm 档只有约 2 个核半径，叶片后方的尾涡解析不足；叶片被加厚到 9 mm。
  从 3 mm 加密到 2 mm 时，两个桨都朝文献值移动（Rushton +12%，PBT −13%），支持"分辨率不足"的解释。
- 要让分桨功率单独收敛，估计需要 1 mm（单卡约 72 小时，或上多卡）。混合时间和 lifeline 预计对此不敏感。

### 2.11 力矩读回的前提不成立：转子内部粒子对并不抵消

- **现象**：写这份文档时核对 `readback_rotor_torque` 的文档字符串，它说转子与转子粒子对的力反对称、求和抵消。
  force.comp 只用自身的修正矩阵，TIC 又按自身压力切换，所以这个前提不成立。
- **测量**（21025dc，`experiment/v1/checks/_check_rotor_torque_split.py`）：在 CPU 上用 float64 重算转子粒子的力，
  与 GPU 的加速度相差不到 4×10⁻⁷，再按邻居类型拆分。转子内部粒子对占读回力矩的 35–44%，几乎全部来自 TIC；
  但读回相对守恒参照只高 4.2–4.9%。详见 5.1 节第 2 条。
- **处理**：文档字符串已改正；力矩的算法没有改，怎么改需要和合作者商量。

## 3. 标量输运的实现

### 3.1 数据和开关

所有功能都由算例的 `scalars:` 块开启。没有这个块时，force 用不含标量代码的构建，其余 kernel 中的标量路径由特化常量关闭。

```yaml
scalars:
  fields:                         # 1 到 12 个场，第 k 个场放在 vec4 k//4 的分量 k%4
    - {name: tracer_01, diffusivity: 1.0e-9, turbulent: true, initial: 0.0}
  sgs: {enabled: true, smagorinsky_cs: 0.1, filter_width: null, turbulent_schmidt: 0.7}   # Δ 默认 = dx
  injections:                     # 同一时刻最多 4 个脉冲
    - {field: tracer_01, center: [0.0, 0.4055, 0.1164], radius: 0.0106, start: 25.0, duration: 1.0, value: 1.0}
  probes:
    radius: 0.009
    points:
      - {name: probe_1, position: [-0.127, 0.4015, -0.018]}
  shift_correction: false         # 默认 false
  compensated_sum: true           # 默认 true
  bounds_limiter: true            # 默认 true，只在 shift_correction 打开时起作用
```

- 浓度按"单位质量"定义：对流由粒子运动完成，不改变粒子上的值；守恒量为 Σ m_i C_i。
- 固体粒子的标量恒为 0，且不参与任何标量求和，所以壁面是零通量边界。

| 位置 | 名称 | 内容 | 随 defrag 搬运 |
| --- | --- | --- | --- |
| set 0 binding 10 | particle_uid | uint，上传时的 pid，永久编号 | 是 |
| set 0 binding 11 | scalar | vec4 × SCALAR_VEC4_COUNT，下标 pid·NV + v | 是 |
| set 0 binding 12 | scalar_compensation | Kahan 补偿项 | 是 |
| set 0 binding 13 | scalar_delta | force 算出、下一步 predict 施加的增量 | 是 |
| set 0 binding 14 | turbulent_viscosity | ν_t | 是（只为读回对齐） |
| set 3 binding 10 | scalar_parameters | 每组 2 个 vec4：D_m 和 SGS 权重 w（0 或 1） | — |
| set 3 binding 11 | scalar_injection | 4 个注入槽，每个 48 B | — |
| correction_inverse[2i+1].z | 原先写 0 | d / tr M（未正则化） | 是 |
| correction_inverse[2i+1].w | 原先写 0 | 流体标记（流体 1，其他 0，只在开启标量时写） | 是 |

特化常量 63–71：SCALAR_VEC4_COUNT、USE_SCALAR_SGS、SGS_LENGTH_SQUARED = (C_s Δ)²、INVERSE_TURBULENT_SCHMIDT、
USE_SCALAR_SHIFT_CORRECTION、USE_SCALAR_COMPENSATED_SUM、USE_SCALAR_INJECTION、USE_SCALAR_BOUNDS_LIMITER、SCALAR_FIELD_COUNT。
common.glsl、`simulator_v1.py` 的 `_global_spec_entries` 和 `case.py` 的 `_SPEC_CONSTANT_MAPPING` 三处同步。

### 3.2 扩散项（force.comp 邻居循环内）

目标方程为 dC/dt = (1/ρ)∇·(ρD∇C)。离散采用 Brookshaw / Morris 形式，D 取两粒子的算术平均：

```
(dC/dt)_i = Σ_{j∈流体} 2 m_j (D_i + D_j) / (ρ_i + ρ_j) · λ_ij · (C_i − C_j) · (x_ij·∇W_ij) / (r² + η²)

D_i  = D_m + w · ν_t,i / Sc_t          （w 为该场的 SGS 权重，0 或 1）
λ_ij = ( d / tr M_i + d / tr M_j ) / 2   （tr M 为未正则化的迹，见 3.3 节）
η²   = 0.01 dx²
```

x_ij·∇W_ij = r W'(r) < 0，所以局部极大值会衰减。D_i = D_j、λ = 1、η = 0 时，就是标准的 SPH 拉普拉斯 Σ_j V_j 2D (C_i − C_j) x_ij·∇W_ij / r²。

代码。循环前取本粒子的量（第 168–212 行）：

```glsl
bool do_scalar = (SCALAR_VEC4_COUNT > 0u) && (self_params.kind == MATERIAL_FLUID);
float scalar_laplacian_eps_squared = 0.04 * self_params.radius * self_params.radius;   // η² = 0.01 dx²（radius = dx/2）
float self_inverse_trace = do_scalar
    ? scalar_normalisation_and_fluid_flag(self_particle_id).x : 1.0;                    // d / tr M_i
...
self_diffusivity[vec4_index] = scalar_parameters[2u * vec4_index]                      // D_m
                             + scalar_parameters[2u * vec4_index + 1u]                 // w
                               * (self_turbulent_viscosity * INVERSE_TURBULENT_SCHMIDT);   // ν_t / Sc_t
```

邻居循环内（第 379–427 行，只列扩散部分）：

```glsl
if (do_scalar) {
    vec2  neighbor_normalisation_and_flag = scalar_normalisation_and_fluid_flag(neighbor_particle_id);
    float fluid_weight  = neighbor_normalisation_and_flag.y;          // 流体 1，固体 0
    float neighbor_mass = velocity_mass[neighbor_particle_id].w;
    float neighbor_turbulent_viscosity =
        USE_SCALAR_SGS ? turbulent_viscosity[neighbor_particle_id] : 0.0;
    float pair_normalisation = 0.5 * (self_inverse_trace + neighbor_normalisation_and_flag.x);   // λ_ij
    float laplacian_factor = fluid_weight * 2.0 * neighbor_mass * pair_normalisation
        * dot(position_difference_neighbor_to_self, kernel_gradient_raw)
        / ((self_density + neighbor_density)
           * (distance * distance + scalar_laplacian_eps_squared));
    for (uint vec4_index = 0u; vec4_index < SCALAR_VEC4_COUNT; vec4_index++) {
        vec4 neighbor_scalar = scalar[scalar_index(neighbor_particle_id, vec4_index)]
                             * scalar_component_mask(vec4_index);
        vec4 neighbor_diffusivity = scalar_parameters[2u * vec4_index]
            + scalar_parameters[2u * vec4_index + 1u]
              * (neighbor_turbulent_viscosity * INVERSE_TURBULENT_SCHMIDT);
        vec4 pair_diffusivity = self_diffusivity[vec4_index] + neighbor_diffusivity;   // D_i + D_j
        vec4 scalar_difference = self_scalar[vec4_index] - neighbor_scalar;          // C_i − C_j
        scalar_rate[vec4_index] += (laplacian_factor * pair_diffusivity) * scalar_difference;
        ...
    }
}
```

公式与代码的对应：

| 公式中的项 | 代码 |
| --- | --- |
| 2 m_j / (ρ_i + ρ_j) | `2.0 * neighbor_mass / (self_density + neighbor_density)` |
| λ_ij | `pair_normalisation` |
| x_ij·∇W_ij（原始核梯度） | `dot(position_difference_neighbor_to_self, kernel_gradient_raw)` |
| r² + η² | `distance * distance + scalar_laplacian_eps_squared` |
| D_i + D_j | `pair_diffusivity` |
| C_i − C_j | `scalar_difference` |
| 只对流体邻居求和 | 乘以 `fluid_weight`，不用分支（原因见 4.8 节） |

- **为什么用算术平均**：Cleary & Monaghan（1999）的调和平均在 D 突变处（例如两种材料的界面）更好，但每个邻居每个分量要做一次除法。
  这里固体邻居已经显式排除，D 在流体内连续变化，算术平均是 Morris 等（1997）粘性项的标准做法。D_i = D_j 时两者相同。
- **分量掩码**：`scalar_component_mask` 由 SCALAR_FIELD_COUNT 决定，管线特化后多余分量是编译期常量 0，相应运算被删掉。

### 3.3 归一化 λ_ij 和 η²

**推导。** 取 C = |x − x_i|²，精确值为 (1/ρ)∇·(ρD∇C) = 2dD。离散算子中 C_i − C_j = −r²，于是（ρ、D 均匀，先取 η = 0）

```
L_i = Σ_j V_j 2D λ (−r²) (x_ij·∇W_ij) / r² = 2Dλ Σ_j V_j (−x_ij·∇W_ij) = 2Dλ tr M_i
其中 tr M_i = Σ_j V_j (x_j − x_i)·∇W_ij = −Σ_j V_j x_ij·∇W_ij
```

取 λ = d / tr M_i 时 L_i = 2dD，对 r² 精确。若 M 各向同性，即 M = (tr M / d) I（规则晶格上成立），
再利用立方对称性，对任意二次函数都精确。η² > 0 时每一项乘以 r²/(r² + η²)，η² = 0.01 dx² 时整体只少 0.56%。

在测试晶格（简单立方，h/dx = 3，按 Σ_{j≠i} V W = 1 校准的体积 V = 1.219 dx³）上，对 r² 直接计算离散算子，精确值为 6：

| 写法 | L[r²] | 误差 | T1 实测方差增长比 |
| --- | --- | --- | --- |
| 教科书 Brookshaw，原始 ∇W，η² = 0.01 h² | 6.916 | +15.3% | 1.149 |
| 同上，η² = 0 | 7.263 | +21.0% | — |
| 存储的 (M + ξI)⁻¹ 取两粒子平均，η² = 0.01 dx² | 5.511 | −8.2% | 0.918 |
| 精确的 M⁻¹，η² = 0.01 h² | 5.714 | −4.8% | — |
| λ_ij（未正则化的迹），η² = 0.01 dx²，**采用** | 5.966 | −0.56% | 0.993 |

- 原始算子偏大，是因为校准后的体积使 tr M = 3.63，而不是 d = 3。
- 不直接用存储的逆，是因为 correction.comp 在求逆前给对角线加了 ξ = 0.1，这个晶格上 M 的对角元约为 1.21，
  存储的逆比精确的逆小 7.6%（GPU 读回 0.76308，(1.2105 + 0.1)⁻¹ = 0.76307，精确值 0.82610）。
- η² 不用 EPS_H_SQUARED：它等于 0.01 h² = 0.09 dx²，使最近邻的贡献少 8%，整体少约 5%。η 只需要保证重合的粒子不发散。

代码（correction.comp 第 166–174 行，在加 ξ 之前取迹，写进原来写 0 的槽位）：

```glsl
float correction_matrix_trace = correction_matrix[0][0] + correction_matrix[1][1]
    + ((DIMENSION == 3u) ? correction_matrix[2][2] : 0.0);
float scalar_laplacian_normalisation =
    (correction_matrix_trace > 1e-6) ? float(DIMENSION) / correction_matrix_trace : 1.0;
float scalar_fluid_flag = 0.0;
if (SCALAR_VEC4_COUNT > 0u) {
    scalar_fluid_flag =
        (material_parameters[material[self_particle_id]].kind == MATERIAL_FLUID) ? 1.0 : 0.0;
}

correction_matrix[0][0] += REGULARIZATION_XI;        // 合作者原有的正则化，未改动
...
correction_inverse[self_particle_id * 2u + 1u] = vec4(
    correction_matrix_inverse[0][2],
    correction_matrix_inverse[1][2],
    scalar_laplacian_normalisation,   // d / tr(M)，未正则化（原来是 0.0）
    scalar_fluid_flag);               // 流体标记（原来是 0.0）
```

读取（helpers.glsl 第 201 行）：

```glsl
vec2 scalar_normalisation_and_fluid_flag(uint particle_id) {
    return correction_inverse[particle_id * 2u + 1u].zw;
}
```

### 3.4 守恒和单调

- **守恒**：把 (dC/dt)_i 乘以 m_i，成对系数为 2 m_i m_j (D_i + D_j) λ_ij (x_ij·∇W_ij) / ((ρ_i + ρ_j)(r² + η²))，
  关于 (i, j) 对称，而 (C_i − C_j) 反对称，所以每一对对 Σ_i m_i C_i 的贡献正好抵消，总量守恒到舍入误差（T2：2 s 内相对变化小于 1.3×10⁻⁹）。
- **单调**：写成 (dC/dt)_i = Σ_j a_ij (C_j − C_i)，所有 a_ij > 0。显式 Euler 在 dt Σ_j a_ij < 1 时单调。
  规则晶格上 Σ_j a_ij = 3.4 D/dx²（D_i = D_j = D）。桨叶区 ν_t 的 95 分位约 2×10⁻⁵ m²/s（T7），
  对应 dt Σ_j a_ij 约 5×10⁻⁴（4 mm）到 7×10⁻⁴（3 mm），远小于 1，所以扩散本身不需要限制器。
- **壁面**：固体邻居的贡献被 `fluid_weight` 乘掉，壁面法向通量为零（Neumann 边界）。

### 3.5 Smagorinsky ν_t（density.comp，只用于标量）

公式：

```
G   = ∇v_i = [ Σ_j V_j (v_j − v_i) ⊗ ∇W_ij ] M_i⁻¹      （未正则化的 M⁻¹，一阶一致）
S   = (G + Gᵀ) / 2,    |S| = sqrt(2 S:S)
ν_t = (C_s Δ)² |S|                                    （SGS_LENGTH_SQUARED = (C_s Δ)²，Δ 默认为 dx）
```

固体邻居按其壁面或转子速度计入，相当于无滑移。**ν_t 只进标量的扩散系数，不进动量方程。**
放在 density.comp，是因为它已经读取了邻居的速度、体积和核梯度；force.comp 在同一步读取 ν_t。

代码（density.comp，邻居循环内累加原始梯度，循环后乘一次未正则化的逆）：

```glsl
if (USE_SCALAR_SGS && self_is_fluid) {                 // 邻居循环内
    velocity_gradient += neighbor_volume
        * outerProduct(neighbor_velocity - self_velocity, raw_gradient);
}
...
if (USE_SCALAR_SGS && self_is_fluid) {                 // 循环后
    if (USE_KCG_CORRECTION) {
        velocity_gradient = velocity_gradient
                          * unregularized_correction_inverse(self_correction_matrix);
    }
    mat3  strain_rate = 0.5 * (velocity_gradient + transpose(velocity_gradient));
    float strain_rate_contracted = dot(strain_rate[0], strain_rate[0])
                                 + dot(strain_rate[1], strain_rate[1])
                                 + dot(strain_rate[2], strain_rate[2]);   // S:S
    turbulent_viscosity[self_particle_id] = SGS_LENGTH_SQUARED * sqrt(2.0 * strain_rate_contracted);
}
```

从存储的逆恢复未正则化的逆（helpers.glsl 第 212 行）：M = stored⁻¹ − ξI，再求逆。二维时 z 行列是单位阵、没有加 ξ，所以只改 xx、yy；
M 接近奇异时退回存储值：

```glsl
mat3 unregularized_correction_inverse(mat3 regularized_inverse) {
    mat3 matrix = inverse(regularized_inverse);          // = M + ξI
    matrix[0][0] -= REGULARIZATION_XI;
    matrix[1][1] -= REGULARIZATION_XI;
    if (DIMENSION == 3u) {
        matrix[2][2] -= REGULARIZATION_XI;
    }
    if (abs(determinant(matrix)) <= REGULARIZATION_DETERMINANT_THRESHOLD) {
        return regularized_inverse;
    }
    return inverse(matrix);                              // = M⁻¹
}
```

- 原有的"计算→传输→计算"屏障链只声明了传输访问，所以开启 SGS 时在 density 和 force 之间额外插入一个"计算→计算"屏障。
- 相对存储的逆，未正则化的逆使一个粒子多做两次 3×3 求逆，只在开启 SGS 时发生（density 多 0.8 ms，见 3.12 节）。

### 3.6 Kahan 补偿累加（predict.comp）

公式（c 为补偿项，真实值为 C − c，主机端 `scalar_snapshot()` 按此恢复）：

```
y = ΔC − c;   t = C + y;   c = (t − C) − y;   C = t
```

**必要性。** 3 mm 槽的时间步约 6.6×10⁻⁵ s。主体区 D_t 约 10⁻⁷ m²/s 时，每步增量约为 2.5×10⁻⁶ 乘以邻居间的浓度差。
到混合后期，浓度差约 10⁻³，增量低于 C ≈ 0.5 处半个 ulp（3×10⁻⁸），普通 float32 累加会把每一步都舍入掉。

代码（predict.comp 第 176–211 行）。`precise` 禁止编译器重排这四行，否则补偿项会被代数化简掉：

```glsl
if (SCALAR_VEC4_COUNT > 0u && !is_rotor) {             // 边界已早退，这里只剩流体
    for (uint vec4_index = 0u; vec4_index < SCALAR_VEC4_COUNT; vec4_index++) {
        uint index = scalar_index(self_particle_id, vec4_index);
        vec4 delta = scalar_delta[index];
        precise vec4 value = scalar[index];
        precise vec4 compensation = vec4(0.0);
        if (USE_SCALAR_COMPENSATED_SUM) {
            compensation = scalar_compensation[index];
            precise vec4 corrected = delta - compensation;       // y = ΔC − c
            precise vec4 sum = value + corrected;                // t = C + y
            compensation = (sum - value) - corrected;            // c = (t − C) − y
            value = sum;                                         // C = t
        } else {
            value = value + delta;
        }
        if (USE_SCALAR_INJECTION) { ... 见 3.8 节 ... }
        scalar[index] = value;
        if (USE_SCALAR_COMPENSATED_SUM) {
            scalar_compensation[index] = compensation;
        }
    }
}
```

这段放在越界杀死粒子和体素登记之后、写回位置之前，所以被杀死的粒子不会更新标量。

### 3.7 位移修正和限制器（可选，默认关）

公式：

```
∇C_i  = M_i⁻¹ Σ_{j∈流体} V_j (C_j − C_i) ∇W_ij         （未正则化的 M⁻¹）
ΔC_i  = dt (dC/dt)_i + shift_i · ∇C_i                  （shift_i 与 predict 加到位置上的是同一个量）
限制器：ΔC_i ← clamp(ΔC_i, min_j C_j − C_i, max_j C_j − C_i)   （范围取自本粒子和流体邻居）
```

代码（force.comp 第 465–495 行）。利用 M⁻¹ 对称，shift·(M⁻¹g) = (M⁻¹shift)·g，所以循环内只累加原始梯度 g：

```glsl
if (do_scalar) {
    for (uint vec4_index = 0u; vec4_index < SCALAR_VEC4_COUNT; vec4_index++) {
        vec4 delta = TIMESTEP * scalar_rate[vec4_index];
        if (do_scalar_gradient) {                                 // 只在 shift_correction: true 时
            vec3 corrected_shift = scalar_gradient_matrix * particle_shift;   // 未正则化的 M⁻¹
            delta += corrected_shift.x * scalar_gradient_x[vec4_index]
                   + corrected_shift.y * scalar_gradient_y[vec4_index]
                   + corrected_shift.z * scalar_gradient_z[vec4_index];
        }
        if (USE_SCALAR_BOUNDS_LIMITER && do_scalar_gradient) {
            delta = clamp(delta,
                          scalar_minimum[vec4_index] - self_scalar[vec4_index],
                          scalar_maximum[vec4_index] - self_scalar[vec4_index]);
        }
        scalar_delta[scalar_index(self_particle_id, vec4_index)] = delta;
    }
}
```

默认关闭的理由见 4.6 节和 5.3 节第 3 条。梯度只对流体邻居求和，而 M 包含全部邻居，所以壁面附近的法向梯度偏小，相当于零梯度边界。

### 3.8 注入和探针

**注入**：脉冲在 start ≤ t_{n+1} < start + duration 期间有效，predict 在新位置 x_{n+1} 判断粒子是否在球内，
在球内就把目标分量设为给定值，并把补偿项清零：

```
|x_{n+1} − c|² < R²   ⇒   C_k ← value,   c_k ← 0
```

```glsl
if (USE_SCALAR_INJECTION) {
    for (uint slot_index = 0u; slot_index < MAX_INJECTION_SLOTS; slot_index++) {
        ScalarInjectionSlot slot = scalar_injection[slot_index];
        if (slot.target.x == 0u || slot.target.y != vec4_index) continue;   // 未启用，或不是这个 vec4
        vec3 offset = new_position - slot.center_radius_squared.xyz;
        if (dot(offset, offset) < slot.center_radius_squared.w) {
            value[slot.target.z] = slot.value.x;
            compensation[slot.target.z] = 0.0;
        }
    }
}
```

注入槽由主机每步写进一个 host-visible 暂存缓冲（`_injection_write_state`），步进命令在 predict 之前把它拷进 device-local 缓冲，
避免每个流体粒子线程都跨 PCIe 读主机内存。转子状态只有转子粒子读，所以沿用 host-visible 的方式。

**探针**：主机端对探针点周围的活流体粒子做 Shepard 核加权平均，核形状与求解器相同（Wendland C4），支撑半径 R 默认为 h：

```
C(p) = Σ_j C_j W(|p − x_j|; R) / Σ_j W(|p − x_j|; R)
```

```python
q = np.sqrt(distance_squared[inside]) / radius
weight = (1.0 - q) ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0)
result[p] = weight @ values[inside] / weight.sum()
```

同时记录每个场的守恒总量 Σ m_i C_i（`scalar_totals()`）。

### 3.9 defrag 和永久粒子编号

- `particle_uid`（set 0 binding 10，原来这里只有一行注释，为 GlobalIdBuffer 预留）：上传时等于 pid，随 defrag 搬运。
  lifeline 追踪单个粒子要用它，本次也用它把 defrag 前后的粒子对应起来。
- defrag.comp 新增的搬运：

```glsl
dst_particle_uid[new_id] = particle_uid[old_id];
for (uint vec4_index = 0u; vec4_index < SCALAR_VEC4_COUNT; vec4_index++) {
    dst_scalar[...]              = scalar[...];
    dst_scalar_compensation[...] = scalar_compensation[...];
    dst_scalar_delta[...]        = scalar_delta[...];
}
if (USE_SCALAR_SGS) {
    dst_turbulent_viscosity[new_id] = turbulent_viscosity[old_id];
}
```

- scalar_delta 必须搬运：defrag 发生在写出增量的 force 和施加增量的下一步 predict 之间。
- ν_t 每步重算，搬运它只是为了让主机读回的值与粒子对齐（见 4.5 节）。
- set 4 的布局和回拷列表 `DEFRAG_SET4_BINDINGS` 同步加入 10–14。

### 3.10 force.comp 编译成两个版本

最初只用特化常量关闭标量代码，但无标量时 force 仍多花 0.23 ms（约 5%），交替测量稳定存在。
用预处理器把标量代码整段去掉后恢复到 4.33 ms，说明驱动在管线特化时没有删干净。所以 force.comp 编译两次：

```glsl
// force.comp
#ifndef FORCE_WITH_SCALARS
#define FORCE_WITH_SCALARS 0      // force.comp.spv：标量代码被删掉，与改动前的 kernel 相同
#endif                            // force_scalar.comp.spv：编译时加 -DFORCE_WITH_SCALARS=1
...
#if FORCE_WITH_SCALARS
    ... 三段标量代码 ...
#endif
```

```python
# compile_shaders_v1.py
SOURCE_VARIANTS = {
    "force.comp": [("force_scalar.comp", ["-DFORCE_WITH_SCALARS=1"])],
}

# simulator_v1.py
def _module_name_for_pipeline(self, pipeline_name: str) -> str:
    if pipeline_name == "force" and self.case.scalar_vec4_count > 0:
        return SHADER_NAME_FORCE_WITH_SCALARS          # "force_scalar"
    return pipeline_name
```

### 3.11 每步执行顺序（开启标量时）

```
predict       : 位置、速度更新；流体 C ← C + ΔC_n（Kahan）；有效脉冲覆盖注入球内的值
update_voxel  : 不变
correction    : 不变；另存 d / tr M（未正则化）和流体标记到 correction_inverse[2i+1].zw
density       : 不变；开启 SGS 时计算并写 ν_t
（仅 SGS：计算→计算屏障）→ density 暂存区拷回
force         : 不变；另算扩散项（可选 ∇C 和限制器），写 ΔC_{n+1} 到 scalar_delta
defrag（每 10 步）: 另搬运 binding 10–14
```

启动阶段：初始浓度用 `write_scalars()` 在 bootstrap 之前写入，bootstrap 的 force 为第一步准备 ΔC_0。

### 3.12 验证和性能

检查脚本在 `experiment/v1/checks/`：`_check_scalar_box.py`（T1–T5）、`_check_scalar_sgs_couette.py`（T6）、
`_check_scalar_tank_pulse.py`（T7）、`_check_shift_dispersion.py`（位移漂移）、`_check_operator_consistency.py`（晶格上的算子精度）。

| 测试 | 内容 | 结果 |
| --- | --- | --- |
| 回归 | 无标量的 3 mm 槽 100 步逐粒子比较；二维方腔 300 步 | 与改动前的差异在运行间噪声以内 |
| T1 | 三维高斯扩散，D = 3×10⁻⁵ m²/s，dx = 2 mm，每 10 步 defrag | 方差增长 / 2Dt = 0.993–0.994，均方根误差 / 峰值 < 1.1×10⁻³ |
| T2 | 同一次运行的守恒 | 2 s 内 Σ m C 相对变化 < 1.3×10⁻⁹ |
| T3 | D = 0、不接收 ν_t 的场 | 每个粒子的值不变（按 uid 比较，最大变化 0） |
| T4 | 小增量（每步 2.9×10⁻⁸，低于半个 ulp），100000 步 | Kahan：峰值衰减 0.9714（解析 0.9672），方差增长 0.985；普通 float32 累加完全不扩散 |
| T5 | 位移修正对线性场 C = 1 + 10x 的正确性 | 残差 / 预期变化 = 6.6×10⁻⁵（用存储的逆时为 7.7%） |
| T6 | 二维 Taylor–Couette 中的 ν_t | ν_t / 解析值 = 0.997（各径向分箱 0.993–1.001） |
| T7 | 4 mm 碟底槽，两个示踪剂脉冲，SGS 开，转子转动 | 无溢出、无 NaN；不修正时总量变化 7.8×10⁻¹¹，值域 [0, 1] |

性能（4070 Ti SUPER，3 mm 平底槽 124 万粒子，每步各 kernel 的时间，毫秒）：

| 配置 | predict | correction | density | force | 合计 |
| --- | --- | --- | --- | --- | --- |
| 改动前 | 0.24 | 3.23 | 3.69 | 4.33 | 11.61 |
| 新版，无标量 | 0.24 | 3.23 | 3.70 | 4.34 | 11.64 |
| 1 个场 | 0.37 | 3.22 | 3.67 | 6.03 | 13.45 |
| 1 个场 + SGS | 0.37 | 3.22 | 4.47 | 6.16 | 14.40 |
| 12 个场 + SGS | 0.65 | 3.23 | 4.48 | 10.52 | 19.05 |

1 个场每步多 16%，12 个场多 64%。force 的邻居循环受访存延迟限制，每个流体邻居多读两次数据（浓度，以及 d/tr M 和流体标记）影响明显。

## 4. 标量输运开发中遇到的问题（按时间顺序）

### 4.1 教科书 Brookshaw 算子扩散快 15%

- **现象**：T1 方差增长比 1.149。
- **原因**：校准后的体积使 tr M = 3.63 而不是 3（见 3.3 节）；η² = 0.01 h² 又抵消了一部分。在同一晶格上直接计算离散算子得 +15.3%，与 T1 吻合。
- **修改前的写法**（示意）：

```glsl
float laplacian_factor = 2.0 * neighbor_mass
    * dot(position_difference_neighbor_to_self, kernel_gradient_raw)
    / ((self_density + neighbor_density) * (distance * distance + EPS_H_SQUARED));
```

- **处理**：先改为用修正矩阵归一化，于是遇到下一个问题。

### 4.2 KCG 矩阵的 Tikhonov 偏差 7.6%

- **现象**：改用存储的逆（两粒子平均）后，扩散反而慢 8%（0.918）。
- **原因**：读回 GPU 上的矩阵，对角元为 0.76308，正好等于 (1.2105 + 0.1)⁻¹，也就是正则化后的值；精确的逆应为 0.82610。
- **处理**：correction.comp 在加 ξ 之前把 d / tr M 存进 correction_inverse[2i+1].z（代码见 3.3 节），标量用它归一化，T1 变为 0.993。
- **影响**：这个偏差同样存在于合作者的压力梯度、粘性项和 δ 扩散中，本次没有改动，见 5.2 节第 1 条。

### 4.3 η² 取得偏大

EPS_H_SQUARED = 0.01 h²，而本代码的 h 是支撑半径，所以 η² = 0.09 dx²，使拉普拉斯算子整体少约 5%。标量改用 0.01 dx²。见 5.2 节第 2 条。

### 4.4 位移修正的正确性（T5）

用存储的逆计算浓度梯度时，线性场的残差为 7.7%；改用未正则化的逆后降到 6.6×10⁻⁵，与 4.2 节的原因相同。

### 4.5 读回的 ν_t 与粒子错位（T6）

- **现象**：T6 读回的 ν_t 沿半径几乎是平的。
- **原因**：ν_t 最初当作临时量，不随 defrag 搬运，而 T6 的最后一步正好触发了 defrag，读回的值和粒子对不上。
  在主机上按同一公式重算，结果与 GPU 完全一致（比值 1.0000），说明模拟本身没有问题。
- **处理**：binding 14 也随 defrag 搬运（代码见 3.9 节）。另外，速度梯度最初用存储的逆，ν_t 偏小 8%，改用未正则化的逆后比值为 0.997。

### 4.6 位移修正既不守恒也不有界

- **现象**（T7，4 mm 槽，脉冲结束后 0.53 s）：

| 处理方式 | 总量变化 | 值域 |
| --- | --- | --- |
| 不修正（现在的默认） | 7.8×10⁻¹¹ | [0, 1] |
| 泰勒修正 | +0.51% | [−0.30, 1.20] |
| 泰勒修正加限制器 | +2.6% | [0, 1] |

- **原因**：扩散项按粒子对计算，i 得到的正好是 j 失去的；而 Σ_i m_i shift_i·∇C_i 没有成对抵消的结构，只在场光滑、位移均匀时才近似为零。
  注入球边缘的浓度在一个粒子间距内从 0 跳到 1，一阶插值会冲出范围。限制器把下冲截到 0 等于凭空加量，上冲截到 1 等于减量，两者不对称，净效果是总量增加。
- **代价**：不修正时，粒子被 PST 带离流体质点，浓度随粒子一起走。测得的漂移（4 mm 碟底槽，跟踪 2 万个粒子）：

| 时间间隔 | 累计偏离的均方根 | 与"每步同向"累加之比 | 等效扩散系数 |
| --- | --- | --- | --- |
| 1 步 | 7.2 µm（0.0018 dx） | 0.99 | — |
| 0.0036 s | 0.29 mm | 0.97 | 3.9×10⁻⁶ m²/s |
| 0.031 s | 2.0 mm | 0.79 | 2.2×10⁻⁵ m²/s |
| 0.26 s | 11.7 mm | 0.53 | 8.6×10⁻⁵ m²/s（仍在增长） |

  折合相对流体的速度约 8 cm/s（均方根），等效扩散约 10⁻⁴ m²/s，比 Smagorinsky 的 D_t 大（桨叶区中位数约 1×10⁻⁵，主体约 5×10⁻⁸）。
- **处理**：默认不修正，与速度、密度的处理一致（predict 也只移动位置，不修正速度和密度）。
  对槽尺度混合（L²/D 约 900 s，远长于混合时间 25 s）影响有限；混合时间验证时计划修正开、关各跑一次。

### 4.7 限制器的舍入问题（T4 回归）

- **现象**：某一轮回归中 T4 完全不扩散。
- **原因**：先形成 C_i + ΔC_i 再截断。ΔC_i 小于 C_i 的半个 ulp 时，C_i + ΔC_i 在 float32 中就等于 C_i，增量被抹成 0，Kahan 也救不回来。
- **修改前后**：

```glsl
// 修改前（示意）：
delta = clamp(self_scalar[v] + delta, scalar_minimum[v], scalar_maximum[v]) - self_scalar[v];
// 修改后：直接截断增量，而且只在位移修正打开时启用
if (USE_SCALAR_BOUNDS_LIMITER && do_scalar_gradient) {
    delta = clamp(delta, scalar_minimum[v] - self_scalar[v], scalar_maximum[v] - self_scalar[v]);
}
```

### 4.8 性能：force 最初慢一倍

开启 1 个场时，force 从 4.6 ms 变成 9.5 ms。依次做了四项优化：

| 步骤 | force（ms） |
| --- | --- |
| 最初：读邻居材料 → 查材料参数 → 按类型分支 → 分支内读浓度 | 9.5 |
| 用 correction 预存的流体标记加乘法，代替两次相互依赖的查表 | 7.3 |
| 每个邻居每个分量的除法改为每对一次 | 7.2 |
| 去掉依赖邻居类型的分支，所有读取无条件发出 | 6.6 |
| 按场数给分量加掩码 | 6.1 |

```glsl
// 修改前（示意）：三次读取串成依赖链
uint neighbor_kind = material_parameters[material[neighbor_particle_id]].kind;
if (neighbor_kind == MATERIAL_FLUID) {
    vec4 neighbor_scalar = scalar[scalar_index(neighbor_particle_id, v)];
    ...
}
// 修改后：标记和 d/tr M 在同一个 vec4 里，浓度的读取与之并行，固体邻居靠乘 0 排除
vec2 neighbor_normalisation_and_flag = scalar_normalisation_and_fluid_flag(neighbor_particle_id);
float fluid_weight = neighbor_normalisation_and_flag.y;
...
float laplacian_factor = fluid_weight * 2.0 * neighbor_mass * ...;
```

另外把 force 编译成带标量和不带标量两个版本（3.10 节），消除了无标量时 0.23 ms 的开销。

## 5. 需要合作者确认的事项

"影响已有结果"一栏说明这件事是否影响他代码中已有的算例（方腔、Couette、功率数），还是只影响新加的标量部分。

### 5.1 转子和固体边界

1. **固体的压力模型。** 固体（壁面和转子）的压力只由一步增量决定：P_s = EOS(ρ₀ + dt·(Dρ/dt)_s)（1.5 节）。
   由于 dt = CFL·h/c₀，漂移部分给出的压力与 c₀ 成正比，更像一个刚度随 c₀ 变化的罚函数，而不是把流体压力外推到壁面。
   实测（碟底槽 4 mm 和 3 mm，启动阶段 t ≤ 1.05 s，详见 `log/2026-09-27_rotor-torque-pair-forces.md`）：
   - 用前一步的密度按上式重算，与 GPU 读回的固体压力相差不超过 0.08 Pa，即 ρ₀ 附近 float32 的分辨率。所以上式就是 GPU 实际算的东西。
   - 固体压力主要来自漂移项：转子上漂移部分的均方根为 147–160 Pa，扩散部分只有 18–19 Pa。
   - 它与相邻流体压力（Shepard 平均 P̄_f）只弱相关。转子上 P_s ≈ 0.37–0.46 P̄_f，相关系数约 0.5；
     在 |P̄_f| 最大的四分之一粒子中，两者之比的中位数为 0.28–0.35。壁面上比例为 0.05–0.4，相关系数不超过 0.4。
   - 漂移部分在形式上与 c₀ 成正比（Δρ ∝ dt ∝ 1/c₀，P = c₀²Δρ）。扩散部分与 c₀ 无关，约为相邻流体压力的 7%（转子）或 2–4%（壁面）。
   请他确认：这是不是有意的设计？是否考虑改成 Adami 等（2012）的压力外推：

   ```
   P_s = [ Σ_f P_f W_sf + (g − a_s) · Σ_f ρ_f (x_s − x_f) W_sf ] / Σ_f W_sf
   ```

   对转子，a_s 取刚体加速度（向心加速度 −ω² r_⊥）。改动点只在 density.comp 的固体分支。
   这可能与 2.4 节的声速敏感性（c₀ 从 10 到 20 U_tip，Np 变 31%）和顶盖渗漏有关，但这一点还没有验证。
2. **力矩中转子内部粒子对的贡献。** 转子粒子之间的压力和粘性力不严格反对称（∇W̃ 只用自身的 (M_i + ξI)⁻¹，TIC 按自身压力切换），
   所以读回的力矩里有一部分来自转子内部。
   实测（同上，另用一个独立编写的短脚本复算过，结果一致）：
   - 转子与转子粒子对贡献了读回力矩的 35–44%，几乎全部来自 TIC：约一半转子粒子压力 ≤ 0，走反对称分支 P_j − P_i。
   - 这部分不能直接扣掉。反对称分支多出的 2P_i Σ_j V_j ∇W̃_ij / ρ_i 对全部邻居求和接近 0，只是在流体邻居和转子邻居之间一正一负：
     流体一列因此少了 31–40%，内部一列多了 34–44%。只取流体一列会把力矩低估 35–44%。
   - 与守恒参照（平均矩阵、P_i + P_j，转子与流体之间严格满足作用力等于反作用力）相比，读回的力矩高 4.2–4.9%，
     其中 TIC 的净残差 3.6–4.2%，矩阵不对称 0.4–0.5%。从 4 mm 到 3 mm 没有减小。从流体一侧用对称写法算的反作用，也是读回的 95.6%。
   - 如果发展后的流场中比例相同，碟底槽的总 Np 应从 5.25 降到约 5.0（3 mm），从 5.45 降到约 5.2（2 mm）。

   请他确认：转子粒子的加速度只用于力矩，不影响流场（predict 用刚体运动覆盖它），
   是否可以让转子粒子在 force.comp 里始终用 P_i + P_j？这样偏差只剩约 0.5%，但已有的 Np 结果会整体变小约 4%。
   另一种做法是维持现状，把约 5% 作为方法不确定度。文档字符串里"转子粒子对抵消"的说法已经改正（21025dc）。
3. **力矩中转子与壁面粒子对的贡献。** 轴穿过顶盖和轴承座附近时，转子粒子有壁面邻居。
   实测：不超过读回力矩的 0.006%，可以忽略。
4. **重力和背景压力。** 开重力的算例在 23.5 s 发散（2.5 节），目前重力一律关闭，涡核处可以出现负压。
   是否加背景压力 p_b 或做静压初始化？TIC 在负压下的切换是否需要调整？
5. **薄运动构件的分辨率。** Rushton 叶片只有约 2 个核半径高，叶片被加厚到 3 dx，分桨功率不收敛（2.10 节）。
   他在 δ⁺-SPH 里处理薄运动构件有没有经验，比如更小的 h/dx、局部加密、或别的边界处理？
6. **小问题，确认即可**：
   - force 对转子粒子仍计算 PST 位移，predict 不用，只是浪费；是否在 force 里对转子跳过 PST 段？
   - `extension_fields.xyz` 被转子的参考位置占用，他的其他分支是否另有用途？标量没有用 `.w`，而是新开了 binding 11–14。
   - 一个算例只支持一个转子。
   - 生成器 `--c0-factor` 默认值是否改为 20（2.4 节）。

### 5.2 求解器共性的发现（影响已有结果）

这几条是在做标量时发现的，都在合作者原有的代码里，本次**没有改动**。逐处的原代码和计算见 `log/2026-09-27_scalar-transport-review.md` 第 2–4 节。

1. **KCG 正则化 ξ = 0.1 是绝对值。** 规则晶格上 M 的对角元约 1.13–1.21，所以存储的逆偏小 7.6%（3D，h/dx = 3）或 8.1%（2D，h/dx = 5）。
   压力梯度、粘性项、δ 扩散都用这个修正梯度，因此都整体偏小。是否有意？是否改为相对正则化，例如：

   ```glsl
   float regularization = REGULARIZATION_XI * correction_matrix_trace / float(DIMENSION);   // xi 取 1e-3 量级
   correction_matrix[0][0] += regularization;   // 其余对角元同理
   ```

   这会改变所有算例的动量方程，需要重跑方腔、Couette 和功率数。
2. **EPS_H_SQUARED = 0.01 h² 中的 h 是支撑半径。** 文献中 0.01 h² 的 h 通常指光滑长度（支撑半径的一半），所以这里的 η² 是文献值的 4 倍，
   使最近邻的贡献少 8%。是否有意？
3. **Morris 粘性项在规则晶格上偏小 16–22%。** 用 `_check_operator_consistency.py` 在理想晶格上计算（与精确值之比）：

   | 晶格 | 现状：(M + ξI)⁻¹，η² = 0.01 h² | M⁻¹，η² = 0.01 h² | M⁻¹，η² = 0.01 dx² | 原始 ∇W，η² = 0.01 h² |
   | --- | --- | --- | --- | --- |
   | 3D，h/dx = 3（搅拌槽） | 0.783 | 0.848 | 0.876 | 1.026 |
   | 2D，h/dx = 5（方腔、Couette） | 0.840 | 0.915 | 0.972 | 1.033 |

   等效粘度偏小，等效雷诺数偏高约 19–28%。这是否影响他之前与 Ghia 基准的对比？另外，三维 h/dx = 3 时，即使用精确的 M⁻¹、η² 取 0，仍偏小 12%，
   不乘修正矩阵反而只偏大 3%，所以"乘修正矩阵"对这个向量形式的粘性项未必是合适的归一化。
4. **PST 让粒子相对流体持续漂移。** 每步 0.0018 dx，但方向持续，折合约 8 cm/s（均方根），0.26 s 累计 11.7 mm（4.6 节）。
   对 pst_main = 0.1、pst_anti = 0.0005、CFL = 0.15 的设置，这个量级是否正常？他在其他算例里见过类似的量吗？
5. **位移后速度和密度不做泰勒修正。** predict 把粒子移到新位置后，仍带着原来的速度和密度。这是 δ⁺-SPH 常见的简化，是否有意？

### 5.3 标量输运本身（只影响标量）

1. **correction_inverse[2i+1] 的 z、w 两个槽位**原来写 0，现在存 d / tr M 和流体标记。他的其他分支里这两个槽位是否另有用途？
   test 分支上只有 `unpack_correction_inverse` 读这个 vec4，而它只用 x、y。
2. **从存储的逆反推未正则化的逆**（3.5 节）在二维、以及 correction.comp 退回单位阵或做 Frobenius 截断的粒子上并不准确，
   但这些情况只在粒子明显缺邻居时出现。是否可以接受？
3. **标量的位移处理**：不修正（现在的默认，守恒有界，但有约 10⁻⁴ m²/s 的额外数值扩散）、泰勒修正（不守恒、会越界），
   还是 δ-ALE-SPH（Antuono 等 2021）的守恒形式？后者把位移写成粒子对之间的通量，但要同时改连续性方程和动量方程。
4. **标量和动量用不同的归一化和 η²**：标量用 λ_ij 和 0.01 dx²，动量仍用 (M + ξI)⁻¹ 和 0.01 h²。他能否接受，还是希望动量这边也统一改？
5. **λ 是各向同性的归一化**：每个粒子一个数，只纠正整体尺度，不纠正方向相关的误差。更完整的做法是对称地使用两粒子的完整未正则化矩阵，
   但每个邻居要多读 24 B 或多做两次矩阵求逆。他是否认可现在的做法？
6. **force 编译两个版本**（3.10 节）的做法是否可以接受？

### 5.4 汇总（讨论提纲）

| 编号 | 事项 | 本文位置 | 影响已有结果 | 我们的建议 |
| --- | --- | --- | --- | --- |
| R1 | 固体压力只由一步增量决定，以漂移项为主，只弱相关于流体压力 | 5.1-1 | 是：所有固体边界，功率数 | 确认意图；若改为压力外推，需重跑方腔、Couette、功率数 |
| R2 | 读回的力矩比守恒参照高 4.2–4.9%（TIC 作用在转子粒子上） | 5.1-2 | 是：功率数约 −4% | 转子粒子始终用 P_i + P_j（不影响流场），或作为方法不确定度 |
| R3 | 力矩中转子与壁面粒子对 | 5.1-3 | 否：≤ 0.006% | 不需处理 |
| R4 | 重力、背景压力、负压下的 TIC | 5.1-4 | 是：有负压的算例 | 先讨论方案，再重跑开重力的算例 |
| R5 | 薄运动构件的分辨率 | 5.1-5 | 是：分桨功率 | 听取他的经验 |
| S1 | ξ = 0.1 为绝对值 | 5.2-1 | 是：所有算例 | 改为相对正则化后重跑方腔、Couette、功率数 |
| S2 | EPS_H_SQUARED 的 h 取支撑半径 | 5.2-2 | 是：粘性项、δ 扩散 | 确认意图 |
| S3 | 粘性项在晶格上偏小 16–22% | 5.2-3 | 是：方腔的等效雷诺数 | 与 S1、S2 一起决定 |
| S4 | PST 漂移约 8 cm/s | 5.2-4 | 是：速度、密度、标量 | 确认量级是否正常 |
| S5 | 位移后不修正速度、密度 | 5.2-5 | 是 | 确认意图 |
| C1 | zw 槽位的新用途 | 5.3-1 | 否 | 确认其他分支没有冲突 |
| C2 | 反推未正则化的逆 | 5.3-2 | 否 | 可接受 |
| C3 | 标量的位移处理 | 5.3-3 | 否 | 暂用不修正；混合时间做开关对照 |
| C4 | 标量与动量的归一化不同 | 5.3-4 | 否 | 取决于 S1–S3 的决定 |
| C5 | 各向同性的 λ | 5.3-5 | 否 | 可接受 |
| C6 | force 编译两个版本 | 5.3-6 | 否 | 可接受 |

## 6. 下一步

1. **等合作者对 S1–S3、R1–R2 的决定。** 如果改动量方程或固体压力，需要重跑二维方腔（对 Ghia）、Couette 和碟底功率数序列。
2. **混合时间验证**：3 mm 碟底槽，10 个示踪剂，25 s 起每秒一次脉冲，至少 60 s，与实验 t95 ≈ 25 s 和两套 LES 的约 29 s 比较；
   同时做位移修正开和关、SGS 开和关的对照。估算：3 mm 碟底槽在 5090 上无标量时约 120 步/s，10 个示踪剂加 SGS 约慢 60%，
   60 s 约 91 万步，每个变体约 3.5 小时。在并行科技上跑需要先确认费用。
3. **lifeline**：细胞挂在流体粒子上（用 particle_uid 追踪），反应项加在 predict.comp 的标量更新处。
