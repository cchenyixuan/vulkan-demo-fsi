# 2026-10-08 54 m³ Rushton 生成器加入 9-pool 选项

## 原来的代码

`utils/geometry/_demo_rushton_tank.py` 的 `scalars_block` 只有示踪剂（`--tracers`）和 Haringa 2023 的 Monod 底物设置（`--substrate`，容量固定 1600 μmol/g/h，进料一个窗口、无 stop）。9-pool 细胞模型只在 30 L 生成器里。

## 改成了什么

- 新增 `feed_source_lines(args, injection)`：进料窗口写成 `sources` 条目，支持 `--feed-stop` 和 `--feed-cycle PERIOD ON COUNT`，位置和半径取预设的注入球（haringa54：(0.8, 7.35, 0) m，半径 0.2 m）。
- `scalars_block` 加 `--ninepool` 分支：与 30 L 生成器相同的 15 个场（单向时 biomass 与 paa_ext 交换位置以满足 Monod 汇同一 vec4 的要求）、相同的 reactions 行（含 `ninepool:` 覆盖和 `uptake_inhibition`）。常量和辅助函数（`NINEPOOL_INITIAL`、`NINEPOOL_FIELDS`、`parse_assignments`、`ninepool_knob`）直接从 30 L 生成器导入（导入耗时 0.15 s，无副作用），两处只有一份定义。
- 新选项：`--feed-stop`、`--feed-cycle`、`--q-max-umol-per-g-h`（默认仍 1600，`--substrate` 的输出不变）、`--ninepool`、`--ninepool-one-way`、`--paa-initial`、`--ninepool-ki`、`--ninepool-ki-mean`、`--ninepool-initial`、`--ninepool-param`。`--ninepool` 隐含 `--substrate`，最多再带 1 个示踪剂。

## 验证

80 mm 粗分辨率 haringa54 预设、`--ninepool --ninepool-ki 49.2` 加 D = 0.05 稳态初值：15 个场，`uptake_inhibition: own`，Ki11 49.2，k_E11 0.3900，`load_case` 通过，扩散掩码 0x1（只有第一个 vec4 进邻居循环）。dt 2.70e-4 s（U_tip 6.67 m/s，c0 133 m/s）；25 mm 时按比例约 8.4e-5 s。

## 影响

不带新选项的输出逐字不变（Monod 的 q_max 仍取 1600 默认值）。54 m³ 的 A54 到 E54 算例可以直接生成。

## 待确认

- haringa54 的 p_b 是 2 ρ U_tip² ≈ 89 kPa（Rushton 生成器默认系数 2，与 30 L 的 2000 Pa 对应），H2/H3 的其他未决项（挡板厚度、分辨率）不在本次范围。
