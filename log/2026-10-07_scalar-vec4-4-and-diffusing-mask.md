# 2026-10-07 九池模型第 1 步：标量上限 16 个场，不扩散的 vec4 跳过邻居循环（spec 104）

分支 `test`。九池模型（`docs/stage5_pichia_level2_design_2026-10-02.md` 第 8 节）要每粒子 15 个场，原上限 12；
细胞携带的场不扩散，不该进邻居循环。

## 合作者原来的代码

`common.glsl` 的 `MAX_SCALAR_VEC4 = 3`（12 个场），`force.comp` 的寄存器数组按它开，循环到 `SCALAR_VEC4_COUNT`；
邻居循环里对每个 vec4 都读邻居的标量、算扩散项，不管扩散系数是否为 0（扩散系数 0 时增量恰为 0，但读取和乘加照做）。

## 改成了什么

| 文件 | 改动 |
| --- | --- |
| `experiment/v1/shaders/common.glsl` | `MAX_SCALAR_VEC4` 3 改 4；spec 104 `SCALAR_DIFFUSING_VEC4_MASK`（默认 0xF，即全部参与，和原来一样） |
| `experiment/v1/shaders/force.comp` | 邻居循环里 `if (((SCALAR_DIFFUSING_VEC4_MASK >> v) & 1u) == 0u) continue;`，spec 常量展开后整个 vec4 的读取和乘加被编译掉 |
| `utils/sph/case.py` | `MAX_SCALAR_FIELDS` 16；`ScalarsConfig.diffusing_vec4_mask()`：某 vec4 里有场的分子扩散系数 > 0 或（SGS 开且 turbulent）则置位；映射表 104 |
| `experiment/v1/utils/simulator_v1.py` | `MAX_SCALAR_VEC4` 4，spec 104 条目 |
| `utils/geometry/_demo_stirred_tank_30l.py` | 场数上限 16（`--tracers` ≤ 12，带底物 ≤ 9） |
| `experiment/v1/shaders/README.md`、`CLAUDE.md` | 常量表；空闲范围改为 105+ |

跳过的 vec4 的输运增量本来就是 0（扩散系数 0 乘任何有限数），位移修正默认关，所以结果逐位不变；只有当 `shift_correction: true` 时
那个 vec4 的梯度项也会被跳过，这是行为变化，但不扩散的场本来就不该随位移插值（细胞随粒子走），记在这里。

## 验证（本机 4070 Ti SUPER）

| 项 | 改前 | 改后 |
| --- | --- | --- |
| 3 mm 生产配置（无标量），400 步 | 44.9 步/s | 44.7、44.3 步/s |
| 4 mm 10 个示踪剂（3 个 vec4 全扩散），400 步 | 64.6 | 64.5、64.4 |
| R1 到 R3、S1 到 S3 | 全部数字逐位相同（R1 1.52e-7 / 1.67e-3 / 4.2e-12，R2 1.30e-4 / 6.52e-5 / 5.6e-12，R3 7.4e-10 / 2.3e-5 / 3.81e-6，S1 0 / 0，S2 7.62e-5 … 1.5e-8，S3 3.33e-4 / 8.07e-5），全部通过 |
| 4 mm 7 个场 `state_limited`（2 个 vec4），400 步 | | 68.6、68.5 步/s |
| 4 mm 16 个场（4 个 vec4 全扩散，mask 0xF），400 步 | 不可能（上限 12） | 57.0、57.0 步/s |
| 4 mm 16 个场，只有 vec4 0 扩散（mask 0x1） | | 67.1、67.3 步/s |

跳过后 16 个场的算例只比 7 个场慢 2%，九池算例的标量开销基本只剩胞外那一个 vec4；全扩散时每多一个 vec4 约 −13%（比先前估的 21% 小）。
（第一轮计时曾读到 4 到 11 步/s，重跑三次都是上表的值，当作偶发，记在这里。）

## 影响

- 没有 `scalars:` 块的算例：着色器里标量段被 `SCALAR_VEC4_COUNT = 0` 编译掉，步速不变（上表）。
- 现有标量算例（示踪剂、级别一、级别二）：所有 vec4 都有扩散场或增量恰为 0，结果不变。
- 寄存器数组从 3 组变 4 组：默认算例步速没有可测变化，说明未用的数组被优化掉了。

## 待确认

- 位移修正开启时不扩散的 vec4 不再做梯度插值，是否接受（细胞随粒子走，本来就不该插值）。
