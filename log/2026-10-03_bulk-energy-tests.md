# 2026-10-03 主体动能偏高的排查：夜间对照算例，生成器支持叶轮和挡板分别选表示

分支 `test`。起因：30 L 槽主体区的动能是 Fluent 的 1.2 到 1.6 倍，原因还没找到（`log/2026-10-03_fluent-and-wall-torque.md`）。
有两个嫌疑：背景压力驱动的粒子噪声，和薄板的边。用户要求两个嫌疑的对照都跑一遍，明早讨论。

已知的情况：

- 只在主体区。叶轮区的速度和动能与 Fluent 相当。
- 不是旋流。主体切向速度比 Fluent 低，|u| 却高 20% 到 36%。
- 全槽动能 5 s 后就稳定，到 30 s 不变。
- 同样的功率，我们的耗散慢 25% 到 40%：KE/P 0.51 到 0.57 s，Fluent 0.41 s。
- 3 mm 时，叶片和挡板都换成片状粒子，全槽动能少 18%（0.726 对 0.884 J）。

## 一、生成器：叶轮和挡板可以分别选表示

### 原来的代码

`utils/geometry/_demo_stirred_tank_30l.py` 的 `--thin-plates` 把叶片、Rushton 圆盘和挡板全部做成薄板，不能与片状粒子的选项同用：

    if args.thin_plates:
        if args.conformal_blades or args.conformal_baffles:
            parser.error("--thin-plates replaces --conformal-blades and --conformal-baffles")

所以已有的对照只有"全薄板"和"全片状"两种，分不出是叶轮还是挡板的作用。

### 改成了什么

`--thin-plates` 可以与其中一个片状选项同用：

| 选项 | 叶片 | Rushton 圆盘 | 挡板 |
| --- | --- | --- | --- |
| `--thin-plates` | 薄板 | 薄板 | 薄板 |
| `--thin-plates --conformal-baffles`（新） | 薄板 | 薄板 | 片状 |
| `--thin-plates --conformal-blades`（新） | 片状 | 格点实心 | 薄板 |
| `--conformal-blades --conformal-baffles` | 片状 | 格点实心 | 片状 |

两个片状选项都加时，`--thin-plates` 仍报错。"片状"即单层普通粒子，铺在叶片或挡板自身坐标系的平面网格上。

    impeller_plates = args.thin_plates and not args.conformal_blades
    baffle_plates   = args.thin_plates and not args.conformal_baffles
    build_solids(..., with_blades=not (args.conformal_blades or impeller_plates),
                      with_baffle_plates=not (args.conformal_baffles or baffle_plates),
                      with_disk=not impeller_plates)
    thin_plate_sheets(..., with_impeller=impeller_plates, with_baffles=baffle_plates)

`thin_plate_sheets()` 新增 `with_impeller`、`with_baffles` 两个参数，默认 True。为 False 时跳过叶片和圆盘，或跳过挡板。

### 影响

- 不用新组合时，输出逐字节不变。4 mm 的全薄板和全片状算例重新生成后，所有 `.obj`、`case.yaml`、`materials.yaml` 与改前相同。
- 3 mm 加 `--legacy-pbt` 重新生成的全薄板算例，粒子数与 9 月 30 日以来的生产算例相同：流体 1,092,430，壁面 262,556，转子 5,423。
- 两个新组合 4 mm 试跑 500 步正常。粒子数：薄板叶轮加片状挡板，流体 458,935；片状叶轮加薄板挡板，流体 458,949。分别与全薄板、全片状相同。

### 需要确认

无。只是诊断用的组合，默认行为不变。

## 二、新脚本 `experiment/v1/checks/_check_tank_energy.py`

每 `--every` 步写一行 CSV：

- 转子力矩，分上下桨；
- 各壁面部件的力矩：圆柱面、顶盖、底、各挡板旁的普通壁面粒子（支架，或片状挡板本身）、各挡板薄板，以及 `_check_blade_leak_tracking.py` 的分类；
- 分区动能：Rushton 区、PBT 区（Fluent 的 rt_rotorbox、pbt_rotorbox）、主体。质量按 ρ0 dx³，与 Fluent 可比；
- 主体动能按径向、切向、轴向分量，各区平均速度，主体平均切向速度，流体角动量；
- 密度均值和范围，负压粒子比例，1% 压力分位数，溢出计数。

结束时写最终快照，格式与生产算例相同，已有的分析脚本都能读。`--rest` 把转子转速设为 0。

## 三、夜间算例

本机 RTX 4070 Ti SUPER，依次运行。驱动脚本 `output/kecause/run_all.sh`，结果在 `output/kecause/`，都在 git 外。
公共参数 `--hdx 3 --c0-factor 20 --thin-layers 1 --skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse`。

| 算例 | 分辨率 | 状态，时长 | 叶轮 | 挡板 | p_b，Pa | 回答什么 |
| --- | --- | --- | --- | --- | --- | --- |
| rest4_lattice_pb2000 | 4 mm | 静止，3 s | 格点三层 | 格点三层 | 2000 | 背景压力驱动的粒子运动是衰减还是维持 |
| rest4_lattice_pb1000 | 4 mm | 静止，3 s | 格点三层 | 格点三层 | 1000 | 它与 p_b 的关系 |
| rest4_lattice_pb4000 | 4 mm | 静止，3 s | 格点三层 | 格点三层 | 4000 | 同上 |
| rest4_plates_pb2000 | 4 mm | 静止，3 s | 薄板 | 薄板 | 2000 | 薄板是否额外维持运动 |
| rest4_sheets_pb2000 | 4 mm | 静止，3 s | 片状 | 片状 | 2000 | 片状是否额外维持运动 |
| s3_plates_pb2000 | 3 mm | 搅拌，6 s | 薄板 | 薄板 | 2000 | 基准，与生产算例对照 |
| s3_plates_pb1000 | 3 mm | 搅拌，6 s | 薄板 | 薄板 | 1000 | 主体动能是否随 p_b 变 |
| s3_plates_pb4000 | 3 mm | 搅拌，6 s | 薄板 | 薄板 | 4000 | 同上 |
| s3_plateimp_sheetbaf_pb2000 | 3 mm | 搅拌，6 s | 薄板 | 片状 | 2000 | 只换挡板 |
| s3_sheetimp_platebaf_pb2000 | 3 mm | 搅拌，6 s | 片状 | 薄板 | 2000 | 只换叶轮 |
| s3_sheets_pb2000 | 3 mm | 搅拌，6 s | 片状 | 片状 | 2000 | 全片状基准 |

3 mm 都加 `--legacy-pbt`，几何与生产算例相同，便于对照已有的 30 s 结果。全槽动能 5 s 后就稳定，6 s 够用。
估计用时：4 mm 静止每个约 8 分钟，3 mm 搅拌每个约 30 分钟，合计约 3.5 小时。

判读方法：

- 静止槽的运动若在 3 s 内衰减到很小，粒子噪声不是持续的能量来源。若维持，且随 p_b 增大，就是。
- 搅拌算例的主体动能若随 p_b 单调变化，背景压力是原因之一。
- "只换挡板"和"只换叶轮"两个算例，分别给出挡板和叶轮的表示对主体动能的贡献。按对称性，三块挡板的平均载荷应当相同，也一并检查。

## 结果

明早补。
