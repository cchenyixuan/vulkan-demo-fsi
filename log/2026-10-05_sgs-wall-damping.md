# 2026-10-05 动量 SGS 的近壁阻尼（spec 94、95，`numerics.momentum_sgs_wall_damping`，默认关）

分支 `test`。接 `log/2026-10-05_paratera-2mm-sheet-baffles.md` 第六到八节：多出的脉动堆在核尺度上方一到三倍的地方，
Δ = dx 的 SGS 太弱；Δ = h 时（2026-10-03）Rushton 力矩被压 15%。用户：选项 1（近壁阻尼）和 Δ = h 配 C_s 0.1，试一下，提交并行科技。

## 合作者原来的代码

动量方程没有 SGS 项（`log/2026-10-01_momentum-sgs.md`）。我们 10 月 1 日加的动量 SGS 里，流体与流体的粒子对用
ν_pair = ν + ½(ν_t,i + ν_t,j)，ν_t = (C_sΔ)²|S|，不管离壁多近都是全额；只有流体与固体的粒子对保持分子粘度。
薄板叶片两侧一个核半径内的流体粒子对也是全额 ν_t。标准 Smagorinsky 实现（包括 Fluent 的 Smagorinsky-Lilly）都有近壁限制。

## 改成了什么

`density.comp` 算 ν_t 的地方（公式写在 `common.glsl` spec 94 的注释里）：

    L_s  = min(κ d, C_s Δ),   κ = 0.41
    ν_t  = L_s² |S|
    d    = max(到最近固体邻居的距离 − r_p, 0)

- "固体邻居"= 核支撑内的普通壁面或转子粒子，以及薄板的壁面虚粒子（`WITH_THIN_PLATES` 构建里 `neighbor_is_wall_dummy`）。
  支撑内没有固体时不限制。
- r_p = 粒子半径（`physics.particle_radius`，spec 95 `SGS_WALL_DISTANCE_OFFSET`）：固体粒子中心在表面以内半个间距，减掉它得到到表面的距离。
- 邻居循环里多读一次邻居的材料种类，只在选项打开时编译进去。

```glsl
if (USE_MOMENTUM_SGS && USE_SGS_WALL_DAMPING && self_is_fluid) {
    bool neighbor_is_solid = is_solid_kind(material_parameters[material[neighbor_particle_id]].kind);
#if WITH_THIN_PLATES
    neighbor_is_solid = neighbor_is_solid || neighbor_is_wall_dummy;
#endif
    if (neighbor_is_solid) nearest_wall_distance = min(nearest_wall_distance, distance);
}
...
float wall_length = 0.41 * max(nearest_wall_distance - SGS_WALL_DISTANCE_OFFSET, 0.0);
mixing_length_squared = min(mixing_length_squared, wall_length * wall_length);
```

| 文件 | 改动 |
| --- | --- |
| `experiment/v1/shaders/common.glsl` | spec 94 `USE_SGS_WALL_DAMPING`（默认 false），95 `SGS_WALL_DISTANCE_OFFSET` |
| `experiment/v1/shaders/density.comp` | 最近固体邻居的距离；ν_t 的限制 |
| `utils/sph/case.py` | `NumericsConfig.momentum_sgs_wall_damping`；spec 映射 94、95 |
| `experiment/v1/utils/simulator_v1.py` | spec 常量 |
| `utils/geometry/_demo_stirred_tank_30l.py` | `--momentum-sgs-wall-damping`，写 `momentum_sgs_wall_damping: true` |
| `experiment/v1/shaders/README.md`、`CLAUDE.md` | spec 表 |

## 影响

- 选项关闭（默认）时 spec 94 为 false，新分支是死代码；默认构建冒烟测试（3 mm 基准，Fluent 流场起算 0.03 s）正常，
  力矩与之前同量级（Rushton 29.1、全部壁面 69.1 mN·m，这是起算瞬态，不是稳态值）。
- 生成器不加选项时输出不变（两个新算例的 fluid.obj 与基准逐字节相同，case.yaml 只多 SGS 那几行）。
- 打开后：Δ = h、C_s 0.1，0.03 s 冒烟：不加阻尼 Rushton 28.4，加阻尼 28.8，方向符合预期（阻尼减少叶片附近的涡粘度），
  不丢粒子，overflow 0。

## 并行科技算例

3 mm 基准，从静止 20 s，各一张 5090（见 `log/2026-10-05_paratera-2mm-sheet-baffles.md` 第九节的结果）：

| 作业名 | 设置 |
| --- | --- |
| `sgsh01` | `--momentum-sgs 0.1 --momentum-sgs-width h` |
| `sgsh01wd` | 同上加 `--momentum-sgs-wall-damping` |

判据：核尺度上方一到三倍（9 到 30 mm）的脉动能量是否向 Fluent 降，Rushton 力矩和子午环流是否保住。

## 需要合作者确认

1. κ = 0.41 和"到最近固体粒子的距离减粒子半径"这个壁距定义；薄板的虚粒子距离是到虚粒子而不是到板面（最多差半个间距）。
2. 流体与固体粒子对仍用分子粘度；加了近壁限制后这一条可以考虑改成 ν_pair = ν + ν_t,i（现在两种近壁处理叠在一起）。
