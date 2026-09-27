# 2026-09-27 标量输运：给合作者的审核清单（原代码、改动、影响、需要确认）

本文件配合 `2026-09-27_scalar-transport.md` 使用，专门给合作者审核。主日志按功能组织；本文件按合作者的代码组织：
每一处先写原代码，再写改成什么、有什么影响、需要他确认什么，并附代码对比。

- 对比的两个提交：649fed6（标量输运之前）和 76d273b（标量输运）。完整差异可用下面的命令查看：
  `git diff 649fed6 76d273b -- experiment/v1/shaders experiment/v1/utils utils/sph`
- 第 2、3、4 节讨论的物理项（KCG 正则化、压力梯度、粘性项、δ 密度扩散、predict 里的位移），都来自合作者 2026-05-07 的
  提交 9b6a41b。本次提交**没有修改**这些项。第 2、3、4 节报告的是在它们上面测出来的现象，以及标量部分为此采取的处理。
- test 分支更早的提交也改过这几个 kernel，包括转子（2026-09-25）和邻居循环的平方距离比较（8ee9e9d）。它们各有自己的 log，
  这里不重复。
- 与合作者讨论用的总文档是 `docs/discussion_rotor_scalar_2026-09-27.md`，与本文件同一提交。它覆盖转子和标量输运两部分，
  本文件只覆盖标量输运对合作者代码的改动。本提交的其他内容见第 7 节。

## 1. 总览：合作者代码中被改动的地方

"无标量时生效"一列的意思是：算例没有 `scalars:` 块时，这处改动是否真的被执行。

| 文件 | 原来 | 现在 | 无标量时生效 |
| --- | --- | --- | --- |
| correction.comp | correction_inverse[2i+1] 的 z、w 写 0 | z 写 d / tr M（未正则化），w 写流体标记（仅开启标量时，否则写 0） | z 生效，但没有动量项读它 |
| density.comp | 无 | Smagorinsky ν_t，受 USE_SCALAR_SGS 控制 | 否 |
| force.comp | 无 | 三段标量代码，放在 `#if FORCE_WITH_SCALARS` 里，另编译为 force_scalar.comp.spv | 否，force.comp.spv 与原来相同 |
| predict.comp | 无 | 标量增量的 Kahan 累加和注入，受 SCALAR_VEC4_COUNT 控制 | 否 |
| defrag.comp | 搬运 set 0 binding 0–9 | 另搬运 binding 10–14 | uid 这一项生效，每次多拷 4 B/粒子 |
| common.glsl | set 0 binding 10 只有一行注释，为 GlobalIdBuffer 预留 | 新增 set 0 binding 10–14、set 3 binding 10–11、特化常量 63–71 | 只是声明 |
| helpers.glsl | 无 | scalar_index、scalar_component_mask、scalar_normalisation_and_fluid_flag、unregularized_correction_inverse | 否，不被调用 |
| simulator_v1.py | 27 个缓冲 | 新增 7 个，共 34 个；particle_uid 按池大小分配，其余在无标量时为 16 B 占位 | uid 分配和 defrag 回拷生效 |
| case.py | 无 | ScalarsConfig 等配置类；特化常量表增加 63–71 | 否 |
| compile_shaders_v1.py | 每个 .comp 编译一次 | force.comp 额外编译一次，带 -DFORCE_WITH_SCALARS=1 | 只影响编译 |
| _run_v1_headless.py | 力矩采样循环 | 力矩和探针共用一个采样循环；dump 里多了 particle_uid | 是，dump 多一个数组 |

回归证据（详见主日志 6.1 和 7 节）：
- 3 mm 槽不做 defrag，100 步后逐粒子比较：与改动前的差异在运行间噪声以内，活粒子数和材料都一致。
- 二维方腔 300 步，速度与改动前一致。
- 3 mm 槽的各 kernel 耗时，与改动前的 worktree 交替测量：force 4.34 ms 对 4.33 ms，各 kernel 合计 11.64 ms 对 11.61 ms。

## 2. 发现一：KCG 正则化 ξ = 0.1 使所有修正梯度偏小

### 2.1 原代码

correction.comp（9b6a41b，本次未改动）：

```glsl
    correction_matrix[0][0] += REGULARIZATION_XI;          // REGULARIZATION_XI = numerics.regularization.xi = 0.1
    correction_matrix[1][1] += REGULARIZATION_XI;
    if (DIMENSION == 2u) {
        ... z 行列置为单位阵 ...
    } else {
        correction_matrix[2][2] += REGULARIZATION_XI;
    }
    ...
    correction_matrix_inverse = inverse(correction_matrix);   // 存的是 (M + ξ I)⁻¹
    ...
    correction_inverse[self_particle_id * 2u + 1u] = vec4(m02, m12, 0.0, 0.0);
```

其中 M = Σ_j V_j (x_j − x_i) ⊗ ∇W_ij。存下的 (M + ξI)⁻¹ 被下列各项当作"修正梯度" ∇̃W = (M + ξI)⁻¹ ∇W 使用：

```glsl
// force.comp
vec3 kernel_gradient_corrected = self_correction_matrix * kernel_gradient_raw;
a_pressure  -= neighbor_volume_dynamic * pressure_combined / self_density * kernel_gradient_corrected;
a_viscosity += morris_coefficient * viscosity_uniform * neighbor_volume_dynamic
             * dot(self_velocity - neighbor_velocity, position_difference_neighbor_to_self)
             / (distance * distance + EPS_H_SQUARED) * kernel_gradient_corrected;
// 另外 PST 位移和涡量也用 kernel_gradient_corrected

// density.comp
vec3 corrected_gradient = self_correction_matrix * raw_gradient;
density_drift += -self_density * neighbor_volume * dot(neighbor_velocity - self_velocity, corrected_gradient);
float d_ij = 2.0 * psi_ij * dot(position_difference_self_to_neighbor, corrected_gradient)
           / (distance * distance + EPS_H_SQUARED) * neighbor_volume;
```

### 2.2 本次改动

正则化本身和存储的逆都没动。correction.comp 只是在加 ξ 之前多算一个迹，写进原本写 0 的两个槽位：

```diff
+    float correction_matrix_trace = correction_matrix[0][0] + correction_matrix[1][1]
+        + ((DIMENSION == 3u) ? correction_matrix[2][2] : 0.0);
+    float scalar_laplacian_normalisation =
+        (correction_matrix_trace > 1e-6) ? float(DIMENSION) / correction_matrix_trace : 1.0;
+    float scalar_fluid_flag = 0.0;
+    if (SCALAR_VEC4_COUNT > 0u) {
+        scalar_fluid_flag =
+            (material_parameters[material[self_particle_id]].kind == MATERIAL_FLUID) ? 1.0 : 0.0;
+    }
+
     correction_matrix[0][0] += REGULARIZATION_XI;
     ...
     correction_inverse[self_particle_id * 2u + 1u] = vec4(
         correction_matrix_inverse[0][2],
         correction_matrix_inverse[1][2],
-        0.0,
-        0.0);
+        scalar_laplacian_normalisation,   // d / tr(M), unregularized (2026-09-27)
+        scalar_fluid_flag);               // 1 = FLUID (2026-09-27, scalars only)
```

helpers.glsl 新增一个函数，从存储的逆反推未正则化的逆。它只被标量代码调用，用于 force 的浓度梯度和 density 的 ν_t：

```glsl
mat3 unregularized_correction_inverse(mat3 regularized_inverse) {
    mat3 matrix = inverse(regularized_inverse);          // = M + ξ I
    matrix[0][0] -= REGULARIZATION_XI;
    matrix[1][1] -= REGULARIZATION_XI;
    if (DIMENSION == 3u) {
        matrix[2][2] -= REGULARIZATION_XI;               // 2D 的 z 行列是单位阵，没加 ξ
    }
    if (abs(determinant(matrix)) <= REGULARIZATION_DETERMINANT_THRESHOLD) {
        return regularized_inverse;                      // M 本身接近奇异时退回存储值
    }
    return inverse(matrix);                              // = M⁻¹
}
```

### 2.3 影响

修正矩阵的目的是让梯度对线性场精确，即 Σ_j V_j (x_j − x_i) ⊗ ∇̃W_ij = I。加了 ξ 后，这个和等于 M (M + ξI)⁻¹。
若 M = m I，比值就是 m / (m + ξ)。ξ 是绝对值 0.1，而规则晶格上的 m 只比 1 大一点，所以偏差并不小。

在理想规则晶格上，用每个算子作用于已知的场，与精确值相比（`python experiment/v1/checks/_check_operator_consistency.py`）：

| 晶格 | M 对角元 | 存储的逆 / 精确的逆 | 压力梯度，P = x | Morris 粘性项，v = (y², 0, 0) | δ 扩散形式，f = r² |
| --- | --- | --- | --- | --- | --- |
| 3D，h/dx = 3（搅拌槽） | 1.2105 | 0.924 | 0.924 | 0.783 | 0.880 |
| 2D，h/dx = 5（方腔、Couette） | 1.1290 | 0.919 | 0.919 | 0.840 | 0.854 |

- 表里后三列都是"合作者现在的写法"，即 (M + ξI)⁻¹ 加上 η² = 0.01 h²。粘性项和 δ 扩散还叠加了第 4 节 η 的影响。
- 3D 晶格上存储的逆，GPU 读回为 0.76308，与 (1.2105 + 0.1)⁻¹ = 0.76307 一致。2D Couette 的主机计算中，
  正则化的逆为 0.813，未正则化的为 0.886。
- **压力梯度**整体偏小 7.6%（3D）或 8.1%（2D）。在弱可压格式中，这大致相当于把压力对密度的响应整体调软。
- **粘性项**在方腔晶格上偏小 16%，等效粘度约为设定值的 0.84 倍，等效雷诺数高约 19%。
- **Couette 验证测不出来**：稳态解 u = Ar + B/r 与粘度无关，所以 2026-09-25 那次 0.25% 的吻合，不能说明粘性项的量值是对的。
- **对功率数、方腔与 Ghia 对比的影响方向，不能简单推断**：压力场会自行调整来平衡。需要改掉 ξ 后重跑，才能知道。
- **本次提交的处理**：动量和密度不变；只有标量扩散、浓度梯度和 ν_t 使用未正则化的量（第 4 节）。

### 2.4 需要合作者确认

1. ξ = 0.1 用绝对值，是否是有意为之？当初的目的，比如是否为了处理自由面附近 M 接近奇异的情况？
2. 是否同意改为相对正则化，或者只在 M 接近奇异时才正则化？下面是一种可能的写法，**本次未实施**：
   ```glsl
   float regularization = REGULARIZATION_XI * correction_matrix_trace / float(DIMENSION);   // 例如 xi = 1e-3
   correction_matrix[0][0] += regularization;   // 其余对角元同理
   ```
   这会改变所有算例的动量方程，需要重跑方腔（对 Ghia）、Couette 和功率数。
3. correction_inverse[2i+1] 的 z、w 两个槽位，他的其他分支里有没有别的用途？在 test 分支上，只有 unpack_correction_inverse 读取，
   而它只用 x、y。多卡分支的 ghost 交换会搬运整个 vec4，不受影响。
4. unregularized_correction_inverse 的反推方式，在二维和"退回单位阵"或"Frobenius 截断"两种特殊情况下是否可以接受？
   目前这两种情况下反推结果并不准确，但它们只在粒子明显缺邻居时出现。

计算方法：`experiment/v1/checks/_check_operator_consistency.py` 在理想晶格的内部粒子上逐邻居求和，
体积用 case.py 的校准值，M 用未正则化的和，梯度分别取 (M + ξI)⁻¹∇W、M⁻¹∇W 和原始 ∇W，
η² 分别取 0.01 h²、0.01 dx² 和 0，并与已知的精确结果相比。不需要任何算例文件，几秒钟跑完。

## 3. 发现二：粒子位移修正让粒子相对流体持续漂移

### 3.1 原代码

predict.comp（本次未改动）：

```glsl
    vec3 new_velocity = self_velocity_half + self_acceleration * TIMESTEP;
    vec3 new_position = self_position + new_velocity * TIMESTEP + self_shift;   // shift 来自上一步 force 的 PST
    ...
    position_voxel_id[self_particle_id] = vec4(new_position, float(new_voxel_id));
    velocity_mass[self_particle_id]     = vec4(new_velocity, self_mass);         // 速度不插值到位移后的位置
```

- density_pressure 在位移后也不修正。也就是说，粒子被 shift 推到新位置后，仍然带着原来的速度和密度。
- 这是 δ⁺-SPH 常见的简化：每步的位移很小，泰勒修正项通常被省略。

### 3.2 本次改动

位移修正本身、位置更新、速度和密度都没动。新加的浓度场提供两种处理方式，默认用和速度、密度一致的方式：

```
默认（shift_correction: false）：  C_{n+1} = C_n + dt · (扩散项)
可选（shift_correction: true）：   C_{n+1} = C_n + dt · (扩散项) + shift_n · ∇C_n      一阶泰勒插值
可选限制器（bounds_limiter）：      把增量截断到 [邻居最小值 − C_i, 邻居最大值 − C_i]，只在修正打开时起作用
```

对应代码在 force.comp 的第三段 `#if FORCE_WITH_SCALARS` 中，见 5.3 节。浓度梯度用未正则化的 M⁻¹（第 2 节）。

### 3.3 影响

**位移造成的漂移**：在 4 mm 碟底搅拌槽中，转子启动 0.875 s 后，按 uid 跟踪 2 万个流体粒子，累加每一步实际施加的 shift，
得到粒子相对流体的累计偏离（脚本 `checks/_check_shift_dispersion.py`）：

| 时间间隔 | 累计偏离均方根 | 与"每步同向"累加之比 | 等效扩散系数估计 |
| --- | --- | --- | --- |
| 1 步（8.75×10⁻⁵ s） | 7.2 µm，即 0.0018 dx | 0.99 | — |
| 0.0036 s | 0.29 mm | 0.97 | 3.9×10⁻⁶ m²/s |
| 0.031 s | 2.0 mm | 0.79 | 2.2×10⁻⁵ m²/s |
| 0.26 s | 11.7 mm，约 3 dx | 0.53 | 8.6×10⁻⁵ m²/s，仍在增长 |

- 单步很小，但方向持续很久：折合相对流体的速度约 8 cm/s（均方根），量级与"剪切率 × dx"相当。
  可以理解为：剪切不断扭曲粒子排布，PST 不断把它推回均匀。
- **这对合作者代码中的速度和密度同样成立**：它们也不修正，所以同样被这个位移带离流体质点。本次只是第一次把这个量测出来。

**标量的两种处理，在脉冲测试中的表现**（4 mm 槽，示踪剂在 0.3–0.5 s 注入，看之后 0.53 s）：

| 处理方式 | 总量变化 | 浓度范围 | 人为输运 |
| --- | --- | --- | --- |
| 不修正（默认） | 7.8×10⁻¹¹ | [0, 1] | 有，即上表的漂移 |
| 泰勒修正 | +0.51% | [−0.30, 1.20] | 光滑区基本消除 |
| 泰勒修正 + 限制器 | +2.6% | [0, 1] | 光滑区基本消除 |

- **泰勒项为什么不守恒**：扩散项按粒子对计算，i 得到的正好是 j 失去的，总量严格不变。
  而 Σ_i m_i shift_i·∇C_i 没有成对抵消的结构，只有在场光滑、位移场均匀时才近似为零。
- **陡峭前沿会越界**：注入球边缘的浓度在一个粒子间距内从 0 跳到 1，一阶插值会冲出范围。
- **限制器让守恒更差**：把下冲截断到 0 等于凭空加量，上冲截断到 1 等于减量，两者不对称，净效果是总量增加。

### 3.4 需要合作者确认

1. 每步 0.0018 dx、方向持续、折合约 8 cm/s 的漂移，对 pst_main = 0.1、pst_anti = 0.0005、CFL = 0.15 的设置来说是否正常？
   他在其他算例里见过类似的量吗？
2. 速度和密度在位移后不做泰勒修正，是否是有意的？
3. 标量采用哪种处理：
   - 不修正，现在的默认；
   - 泰勒修正；
   - δ-ALE-SPH（Antuono 等 2021）的守恒形式。它把位移当作额外的输运速度，写成粒子对之间的通量，一方得失等于另一方失得，
     但需要在连续性方程和动量方程中加入相应的项，具体形式以原文为准。

## 4. 发现三：扩散类算子的归一化与 η²

### 4.1 原代码

合作者代码中原来没有标量扩散，但有两个同类算子，见 2.1 节：force.comp 的 Morris 粘性项，以及 density.comp 的 δ 密度扩散。
两者分母中的防除零项相同：

```glsl
// common.glsl
// Used in δ-SPH density diffusion and artificial-viscosity expressions where a
// 1/(r² + ε_h²) term would otherwise blow up when two particles approach each
// other. Typical value: 0.01 · H² (Antuono et al. δ-SPH).
layout(constant_id = 40) const float EPS_H_SQUARED = 8.1e-7;  // 0.01 · 0.009²
```

```python
# utils/sph/case.py
def eps_h_squared(self) -> float:
    """Antuono δ-SPH division-by-zero guard for 1/(r² + ε_h²) terms."""
    h = self.physics.h
    return 0.01 * h * h
```

在本代码中，physics.h 是**核支撑半径**，h/dx = 3 时 h = 3 dx，所以 η² = 0.09 dx²。
文献中 0.01 h² 的 h 通常指光滑长度，对应的支撑半径是 2h；按本代码的定义换算，这个 η² 是文献值的 4 倍。
它让最近邻（r = dx）那一项减少 8%。

### 4.2 本次改动

合作者的粘性项和 δ 扩散都没动。标量扩散是新写的，用了自己的一套归一化和 η²：

```glsl
// force.comp（FORCE_WITH_SCALARS 构建），每个流体邻居：
float scalar_laplacian_eps_squared = 0.04 * self_params.radius * self_params.radius;   // η² = 0.01 dx²
float pair_normalisation = 0.5 * (self_inverse_trace + neighbor_normalisation_and_flag.x);   // λ_ij
float laplacian_factor = fluid_weight * 2.0 * neighbor_mass * pair_normalisation
    * dot(position_difference_neighbor_to_self, kernel_gradient_raw)
    / ((self_density + neighbor_density) * (distance * distance + scalar_laplacian_eps_squared));
...
scalar_rate[vec4_index] += (laplacian_factor * pair_diffusivity) * scalar_difference;   // (D_i + D_j)(C_i − C_j)
```

公式：

```
(dC/dt)_i = Σ_j 2 m_j/(ρ_i + ρ_j) · (D_i + D_j) · λ_ij · (C_i − C_j) · x_ij·∇W_ij / (r² + η²)
λ_ij = (d / tr M_i + d / tr M_j) / 2        tr M 未正则化，由 correction.comp 存入
η²   = 0.01 dx²
```

### 4.3 影响

**标量扩散本身**：在 h/dx = 3 的晶格上，对 f = r² 直接计算离散算子，精确值为 6：

| 写法 | 结果 | 误差 | T1 实测方差增长比 |
| --- | --- | --- | --- |
| 教科书 Brookshaw，原始 ∇W，η² = 0.01 h² | 6.916 | +15.3% | 1.149 |
| 同上，η² = 0 | 7.263 | +21.0% | — |
| 用存储的 (M + ξI)⁻¹ 取两粒子平均，η² = 0.01 dx² | 5.511 | −8.2% | 0.918 |
| 用精确的 M⁻¹，η² = 0.01 h² | 5.714 | −4.8% | — |
| λ_ij（未正则化迹），η² = 0.01 dx²，**采用** | 5.966 | −0.56% | 0.993 |

**对合作者现有算子的含义**（`_check_operator_consistency.py`，数值为与精确值之比）：

| 晶格 | 算子 | (M + ξI)⁻¹，η² = 0.01 h²（现状） | M⁻¹，η² = 0.01 h² | M⁻¹，η² = 0.01 dx² | 原始 ∇W，η² = 0.01 h² | 原始 ∇W，η² = 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 3D，h/dx = 3 | Morris 粘性项 | 0.783 | 0.848 | 0.876 | 1.026 | 1.065 |
| 3D，h/dx = 3 | δ 扩散形式 | 0.880 | 0.952 | 0.994 | 1.153 | 1.210 |
| 2D，h/dx = 5 | Morris 粘性项 | 0.840 | 0.915 | 0.972 | 1.033 | 1.100 |
| 2D，h/dx = 5 | δ 扩散形式 | 0.854 | 0.930 | 0.997 | 1.050 | 1.129 |

- δ 扩散形式和标量扩散是同一类算子：先改用精确的 M⁻¹，再把 η² 取为 0.01 dx²，就能接近精确。
- **Morris 粘性项不同**：在 3D h/dx = 3 的晶格上，即使用精确的 M⁻¹、η² 取 0，仍偏小 12%；不用修正矩阵反而只偏大 6.5%。
  2D h/dx = 5 的晶格上，这个差别小得多（精确 M⁻¹ 为 0.975）。原因是这个向量形式的系数 2(d+2) 来自连续介质推导，
  对应的离散矩与 M 所归一化的二阶矩不同，核支撑内粒子越少，差别越大。
  所以对粘性项来说，"乘修正矩阵"未必是合适的归一化方式，至少在 h/dx = 3 的三维晶格上不是。
- 以上都是理想规则晶格上的算子性质。流动中的粒子分布偏离规则晶格，但 PST 会让它保持接近规则。
- 标量的 λ 是各向同性的归一化：每个粒子一个数，所有方向同比例缩放。修正矩阵则是 3×3 的，还能纠正方向相关的误差。
  两者在规则晶格上等价；在粒子排布各向异性的地方，比如壁面附近和强剪切区，λ 只纠正平均大小。

### 4.4 需要合作者确认

1. EPS_H_SQUARED = 0.01 h² 中的 h 按支撑半径取，是否是有意为之？
2. 粘性项偏小 16–22%（取决于晶格），方腔的等效雷诺数会因此偏高。这是否影响他之前与 Ghia 基准的对比？是否需要修正？
3. 标量用和动量不同的 λ 和 η²，他能否接受，还是希望动量这边也统一改？
4. λ 采用各向同性归一化，他是否认可？更完整的做法，是对称地使用两粒子的完整未正则化矩阵，
   但每个邻居要多读 24 B，或者多做两次矩阵求逆。

## 5. 其余改动：新增功能，不改变已有物理

### 5.1 common.glsl

新增的声明：

| 位置 | 名称 | 说明 |
| --- | --- | --- |
| 特化常量 63 | SCALAR_VEC4_COUNT | 0 表示关闭标量 |
| 64–66 | USE_SCALAR_SGS、SGS_LENGTH_SQUARED、INVERSE_TURBULENT_SCHMIDT | Smagorinsky 参数 |
| 67、68、70 | USE_SCALAR_SHIFT_CORRECTION、USE_SCALAR_COMPENSATED_SUM、USE_SCALAR_BOUNDS_LIMITER | 数值开关 |
| 69 | USE_SCALAR_INJECTION | 注入 |
| 71 | SCALAR_FIELD_COUNT | 分量掩码 |
| set 0 binding 10 | ParticleUidBuffer | 原来这里只有一行注释，为 GlobalIdBuffer 预留 |
| set 0 binding 11–14 | Scalar、ScalarCompensation、ScalarDelta、TurbulentViscosity | 标量数据 |
| set 3 binding 10–11 | ScalarParameters、ScalarInjection | 参数和注入槽 |

CorrectionInverseBuffer 的注释改为说明 z、w 两个槽位的新用途。

### 5.2 density.comp：ν_t，仅在 USE_SCALAR_SGS 时存在

```diff
+    bool self_is_fluid = (self_params.kind == MATERIAL_FLUID);
+    mat3 velocity_gradient = mat3(0.0);
     ...
             vec3 corrected_gradient  = self_correction_matrix * raw_gradient;
+            if (USE_SCALAR_SGS && self_is_fluid) {
+                velocity_gradient += neighbor_volume
+                    * outerProduct(neighbor_velocity - self_velocity, raw_gradient);
+            }
     ...
     density_pressure_scratch[self_particle_id] = vec2(density_to_store, new_pressure);
+    if (USE_SCALAR_SGS && self_is_fluid) {
+        if (USE_KCG_CORRECTION) {
+            velocity_gradient = velocity_gradient
+                              * unregularized_correction_inverse(self_correction_matrix);
+        }
+        mat3  strain_rate = 0.5 * (velocity_gradient + transpose(velocity_gradient));
+        float strain_rate_contracted = dot(strain_rate[0], strain_rate[0])
+                                     + dot(strain_rate[1], strain_rate[1])
+                                     + dot(strain_rate[2], strain_rate[2]);   // S:S
+        turbulent_viscosity[self_particle_id] =
+            SGS_LENGTH_SQUARED * sqrt(2.0 * strain_rate_contracted);
+    }
```

ν_t 只用于标量扩散，不进入 force 的粘性项。开启 SGS 时，模拟器在 density 与 force 之间额外插入一个"计算→计算"内存屏障。

### 5.3 force.comp：三段标量代码，只在 FORCE_WITH_SCALARS 构建中存在

```glsl
#ifndef FORCE_WITH_SCALARS
#define FORCE_WITH_SCALARS 0      // force.comp.spv：标量代码被预处理器删掉，kernel 与原来相同
#endif                            // force_scalar.comp.spv：编译时加 -DFORCE_WITH_SCALARS=1
```

- **第一段，循环前**：读本粒子的浓度、ν_t 和 d/tr M，初始化累加器。
- **第二段，邻居循环内**：扩散项，代码见 4.2；可选的浓度梯度和限制器范围。固体邻居通过 fluid_weight = 0 排除，不用分支。
- **第三段，循环后**：写出下一步的增量：

```glsl
    if (do_scalar) {
        for (uint vec4_index = 0u; vec4_index < SCALAR_VEC4_COUNT; vec4_index++) {
            vec4 delta = TIMESTEP * scalar_rate[vec4_index];
            if (do_scalar_gradient) {                                   // 仅 shift_correction: true
                vec3 corrected_shift = scalar_gradient_matrix * particle_shift;   // 未正则化 M⁻¹
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

为什么编译两个版本：最初只用特化常量关闭标量代码，但无标量时 force 仍多花 0.23 ms，约 5%，交替测量稳定存在；
用预处理器把这部分代码删掉后，就恢复到原来的 4.33 ms。说明驱动在管线特化时没有把这些代码删干净。

### 5.4 predict.comp：施加增量和注入，仅在 SCALAR_VEC4_COUNT > 0 时存在

```diff
+    if (SCALAR_VEC4_COUNT > 0u && !is_rotor) {
+        for (uint vec4_index = 0u; vec4_index < SCALAR_VEC4_COUNT; vec4_index++) {
+            uint index = scalar_index(self_particle_id, vec4_index);
+            vec4 delta = scalar_delta[index];
+            precise vec4 value = scalar[index];
+            precise vec4 compensation = vec4(0.0);
+            if (USE_SCALAR_COMPENSATED_SUM) {                 // Kahan 补偿累加
+                compensation = scalar_compensation[index];
+                precise vec4 corrected = delta - compensation;
+                precise vec4 sum = value + corrected;
+                compensation = (sum - value) - corrected;
+                value = sum;
+            } else {
+                value = value + delta;
+            }
+            if (USE_SCALAR_INJECTION) { ... 球内的流体粒子设为脉冲值，并清零补偿项 ... }
+            scalar[index] = value;
+            if (USE_SCALAR_COMPENSATED_SUM) {
+                scalar_compensation[index] = compensation;
+            }
+        }
+    }
     // Write updated state (only Cases A and B reach here).
```

插入位置在越界杀死粒子和体素登记之后、写回位置之前，所以被杀死的粒子不会更新标量。

### 5.5 defrag.comp

```diff
+        dst_particle_uid[new_id]                 = particle_uid[old_id];
+        for (uint vec4_index = 0u; vec4_index < SCALAR_VEC4_COUNT; vec4_index++) {
+            dst_scalar[...]              = scalar[...];
+            dst_scalar_compensation[...] = scalar_compensation[...];
+            dst_scalar_delta[...]        = scalar_delta[...];
+        }
+        if (USE_SCALAR_SGS) {
+            dst_turbulent_viscosity[new_id] = turbulent_viscosity[old_id];
+        }
```

- scalar_delta 必须搬运：defrag 发生在写出增量的 force 与使用增量的下一步 predict 之间。
- ν_t 其实每步都会重算，搬运它只是为了让主机读回的值与粒子对得上。这是 T6 调试中发现的问题。
- set 4 的布局和回拷列表 DEFRAG_SET4_BINDINGS 同步加入 10–14。

### 5.6 主机侧

- **simulator_v1.py**：
  - 新缓冲：particle_uid 按池大小分配，其余在无标量时是 16 B 占位。
  - 注入槽：主机每步写入一个 host-visible 暂存缓冲，step 命令在 predict 之前把它拷进 device-local 缓冲。
    这样可以避免每个流体粒子线程都跨 PCIe 读主机内存。
  - 算例有标量时，force 管线改用 force_scalar 模块。
  - 新增读回和探针接口。
- **case.py**：`scalars:` 块的配置类和校验；特化常量表加入 63–71，与 common.glsl 和 simulator_v1.py 三处同步。

## 6. 需要合作者确认的事项汇总

| 编号 | 事项 | 所在章节 | 是否影响他已有的结果 |
| --- | --- | --- | --- |
| 1 | ξ = 0.1 用绝对值是否有意；是否改为相对正则化 | 2.4 | 是，影响所有算例 |
| 2 | correction_inverse[2i+1].zw 两个槽位在其他分支是否另有用途 | 2.4 | 否 |
| 3 | 反推未正则化逆的方式是否可接受 | 2.4 | 否，只影响标量 |
| 4 | 位移漂移的量级是否正常 | 3.4 | 是，速度和密度也受影响 |
| 5 | 速度和密度不做位移修正是否有意 | 3.4 | 是 |
| 6 | 标量的位移处理：不修正、泰勒修正或 δ-ALE | 3.4 | 否，只影响标量 |
| 7 | EPS_H_SQUARED 的 h 按支撑半径取是否有意 | 4.4 | 是，影响粘性项和 δ 扩散 |
| 8 | 粘性项偏小 16–22% 是否影响方腔对比 | 4.4 | 是 |
| 9 | 标量与动量使用不同的 λ、η² 是否可以接受 | 4.4 | 否 |
| 10 | 各向同性的 λ 是否可以接受 | 4.4 | 否，只影响标量 |
| 11 | force 编译两个版本的做法是否可以接受 | 5.3 | 否 |

## 7. 本提交的其他内容

本文件同时作为本次提交的 log。除本审核清单外，本次提交还包括：

| 文件 | 内容 |
| --- | --- |
| `experiment/v1/checks/_check_operator_consistency.py` | 新增。在理想晶格的内部粒子上计算压力梯度、Morris 粘性项、δ 扩散形式对已知场的结果，与精确值比较。第 2.3、4.3 节的表格由它生成，几秒钟跑完，不需要算例文件 |
| `docs/discussion_rotor_scalar_2026-09-27.md` | 新增。与合作者讨论用：转子和标量输运的实现、遇到的问题、需要确认的事项，公式与代码对照。转子力矩和固体压力的实测数字来自上一个提交 21025dc（`2026-09-27_rotor-torque-pair-forces.md`） |
| `log/2026-09-27_scalar-transport.md` | 开头和第 8 节加了指向本文件和讨论文档的说明；第 4.5 节"每步增量约为 4×10⁻⁶ ΔC"改为 2.5×10⁻⁶（按规则晶格上 Σ_j a_ij = 3.4 D/dx² 重算，结论不变）；第 6.1 节表头里 \|dx\|、\|dv\| 的竖线加了转义，原来会把表格拆错列 |
| `cases/stirred_tank_30l_3mm/case.yaml` | 只改注释。顶部仍是 2026-09-26 转向修正之前的旧解释（称 M-Star 的 −200 rpm 是右手定则下的负角速度），与 materials.yaml 中已改正的 +20.944 rad/s 矛盾，现已改正。另注明这个入库算例是 2026-09-25 的平底、c₀ = 10 U_tip 基准，目前的算例都用生成器重新生成（碟底，`--c0-factor 20`） |
| `utils/geometry/_demo_stirred_tank_30l.py` | 只改生成的 case.yaml 注释：原来固定写"10 * 桨尖速度"，现在写实际的 `--c0-factor`。已用 8 mm 生成一次验证，输出为 "20 * 1.028 m/s" |

验证：`_check_operator_consistency.py` 的输出与第 2.3、4.3 节的表格一致；生成器改动后在 8 mm 下完整运行一次，case.yaml 正常。
其余都是文档和注释，不影响任何计算。

