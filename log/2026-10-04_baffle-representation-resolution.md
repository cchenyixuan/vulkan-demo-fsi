# 2026-10-04 挡板的表示和分辨率：同一个 Fluent 流场里比较

分支 `test`。用户：研究一下（接 `log/2026-10-04_baffle-drag-diagnosis.md` 第五节第 1 条）。
问题：光滑槽壳的算例里，薄板挡板在 Fluent 的流场里只刹出 Fluent 的六成。这是薄板模型的问题，还是 3 mm 不够细？

**结论先说：**

1. 薄板挡板随加密上升，但很慢：3 / 2 / 1.5 mm 为 45.0 / 54.0 / 60.4 mN·m，是 Fluent 69.7 的 65% / 77% / 87%。
   照这个趋势，要到 1 mm 以下才够，粒子数是 3 mm 的几十倍。
2. 换成普通固体粒子的贴合片，3 mm 就有 63.0（90%），比 1.5 mm 的薄板还高，粒子数只有它的 13%。
   三层片（9 mm 厚）是 63.1，与双层片（6 mm 厚）相同：多出来的阻力来自表示方式，不是挡板变厚。
3. 贴合片加密收敛得快：2 mm 双层片 67.8（97%），全部壁面 73.7（Fluent 77.8）；修正槽壳后重算 66.8 / 73.1，不变。
4. 同时得到：Rushton 的薄板叶片在同一流场里 46.0 / 55.9 / 61.4 mN·m，1.5 mm 与 Fluent 的 60.7 一致。
   槽底的粘性摩擦三个分辨率都约 1.0（Fluent 2.46），加密没有用。PBT 18.3 / 18.9 / 20.4，加密后离 Fluent 的 15.0 更远。
5. 3 mm 下挡板改用双层贴合片。已排队：光滑槽壳加双层片挡板，从静止算 20 s。
6. 2 mm 的光滑槽壳在轴承座和挡板支架旁边留了缝，2 mm 两个算例丢了粒子（最多万分之三）。生成器已修正，3 mm 输出不变。

## 一、做法

同一个流场：把 Fluent 细网格 25 s 的速度场插值到各算例的流体粒子上（`_map_fluent_velocity.py`），算 0.7 s，取 0.3 到 0.7 s 的平均。
每个算例另从静止算 0.2 s，取 0.02 到 0.21 s 的平均，作为离散壁面在 p_b 下的静态力矩。两者相减，得到流动产生的力矩：

    流动产生的力矩 = 搅拌算例 0.3 到 0.7 s 的平均 − 静止算例 0.02 到 0.21 s 的平均

静态力矩在各类壁面之间会互相抵消（诊断篇第一节：薄板 −20 到 −32，板旁的格子支架 +29），全部壁面的合计在静止时只有 −2 到 +3 mN·m。
所以"全部壁面"这一列不受静态部分怎么分配的影响，最可靠。

公共设置：光滑槽壳（`--smooth-walls`）、改正后的 PBT、STL 轮毂、p_b 2000 Pa、h/dx 3、c₀ = 20 U_tip，叶轮（Rushton 叶片和圆盘、PBT 叶片）都是薄板。
只改挡板的表示和粒子间距。真实挡板宽 24 mm（r 114.35 到 138.36 mm）、厚 2.6 mm，离槽壁 5.6 mm。

| 算例 | 间距 | 挡板 | 模型里挡板的厚度 | 粒子数（其中流体） | 静止 0.2 s | Fluent 起算 0.7 s |
| --- | --- | --- | --- | --- | --- | --- |
| `s3_plates_pb2000_newpbt_smooth` | 3 mm | 薄板 | 约 1 个间距 | 1,300,643（1,092,574） | 已有 | 已有（10 s 算例的前 0.7 s） |
| `s2_plates_smooth` | 2 mm | 薄板 | 约 1 个间距 | 4,187,321（3,722,186） | 6.1 min | 21.7 min |
| `s15_plates_smooth` | 1.5 mm | 薄板 | 约 1 个间距 | 9,700,881（8,872,724） | 17.5 min | 62.4 min |
| `s3_sheet2baf_smooth` | 3 mm | 双层贴合片 | 6 mm | 1,301,220（1,089,407） | 1.3 min | 4.6 min |
| `s3_sheet3baf_smooth` | 3 mm | 三层贴合片 | 9 mm | 1,301,198（1,085,647） | 1.3 min | 4.7 min |
| `s2_sheet2baf_smooth` | 2 mm | 双层贴合片 | 4 mm | 4,188,125（3,714,886） | 5.7 min | 19.1 min |

贴合片是挡板自己坐标系里的平面网格（`--conformal-baffles --thin-layers N`），普通固体粒子，壁面压力用累积方式，与槽壳相同。
`--thin-layers` 在这些算例里只影响挡板：叶片和圆盘是薄板，支架和探头有自己的尺寸。各算例的 case.yaml 除数值外只差挡板那三块薄板。

生成（仓库根目录，求解器环境）：

    BASE="--hdx 3 --c0-factor 20 --skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse \
          --no-preview --thin-plates --background-pressure 2000 --smooth-walls"
    python utils/geometry/_demo_stirred_tank_30l.py $BASE --thin-layers 1 --dx 0.002  --out output/kecause/cases/s2_plates_smooth
    python utils/geometry/_demo_stirred_tank_30l.py $BASE --thin-layers 1 --dx 0.0015 --out output/kecause/cases/s15_plates_smooth
    python utils/geometry/_demo_stirred_tank_30l.py $BASE --thin-layers 2 --conformal-baffles --dx 0.003 --out output/kecause/cases/s3_sheet2baf_smooth
    python utils/geometry/_demo_stirred_tank_30l.py $BASE --thin-layers 3 --conformal-baffles --dx 0.003 --out output/kecause/cases/s3_sheet3baf_smooth
    python utils/geometry/_demo_stirred_tank_30l.py $BASE --thin-layers 2 --conformal-baffles --dx 0.002 --out output/kecause/cases/s2_sheet2baf_smooth

运行：`output/kecause/run_baffle_study.sh`（双层片 3 mm、薄板 2 mm、1.5 mm）和 `run_baffle_study2.sh`（三层片 3 mm、双层片 2 mm），
每个算例 `_check_tank_energy.py --rest --time 0.2` 和 `--time 0.7 --initial-velocity fluent_fine25_on_<算例>.npy --dump-times 0.5`，
都加 `--split-height 0.025`，约每 0.02 s 一行。本机 RTX 4070 Ti SUPER，overflow 为 0，KCG 回退只在 2 mm 的旧算例里出现（见第五节）。
3 mm 和 1.5 mm 没有丢粒子。**2 mm 的两个算例丢了粒子**（静止 0.2 s 54 / 154 个，Fluent 起算 0.7 s 91 / 979 个，薄板 / 双层片），
原因是光滑槽壳生成器的一个漏洞，已修正，见第五节。丢的量最多是流体的万分之三，不影响本篇的力矩结论；2 mm 双层片用修正后的几何重算了一遍。
汇总：`experiment/v1/checks/_compare_wall_torques.py`（新，见第四节），图 `docs/figures/tank_baffle_drag_resolution.png`。

## 二、结果

流动产生的力矩，mN·m，制动为正；叶轮是流体得到的力矩。± 是 0.1 s 分块平均的标准误差。

| 算例 | 挡板 | ± | 其中板 | 板旁壁面 | 圆柱面 | 槽底 | 全部壁面 | ± | 静止时全部壁面 | y < 25 mm | Rushton | PBT | 角动量，g m²/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 mm 薄板 | 45.0 | 2.1 | 44.9 | 0.1 | 4.7 | 1.09 | 50.0 | 2.3 | −2.0 | 12.6 | 46.0 | 18.3 | 236 |
| 2 mm 薄板 | 54.0 | 2.2 | 49.6 | 4.4 | 4.4 | 0.99 | 58.8 | 2.6 | +0.3 | 14.8 | 55.9 | 18.9 | 237 |
| 1.5 mm 薄板 | 60.4 | 0.8 | 55.7 | 4.7 | 4.3 | 0.94 | 65.5 | 0.8 | −0.4 | 19.0 | 61.4 | 20.4 | 238 |
| 3 mm 双层片 | 63.0 | 1.5 | | 63.0 | 5.7 | 0.87 | 69.3 | 1.4 | −0.8 | 16.4 | 47.8 | 18.1 | 222 |
| 3 mm 三层片 | 63.1 | 1.4 | | 63.1 | 5.3 | 1.06 | 68.9 | 1.5 | −0.3 | 16.8 | 46.2 | 18.4 | 221 |
| 2 mm 双层片 | 67.8 | 3.0 | | 67.8 | 5.4 | 0.99 | 73.7 | 2.9 | +0.9 | 18.3 | 55.0 | 20.0 | 227 |
| Fluent 细网格，25 到 34 s | 69.7 | | | | 5.6 | 2.46 | 77.8 | | | | 60.7 | 15.0 | 229 |

- "挡板" = 板 + 挡板方位 ±20°、100 < r < 139.5 mm 的普通壁面粒子（底部支架、两根探头；贴合片本身也在这一类）。
- Fluent 的挡板 69.7 是十个快照的平均，快照之间的标准差 9.3；粗网格 67.6（`log/2026-10-03_fluent-and-wall-torque.md`），对网格不敏感。
- 角动量是窗口内的平均。Fluent 场的初值约 229，薄板算例在窗口里上升（输入大于制动），贴合片算例下降。

全部壁面按 0.1 s 分块（mN·m，已减静止值）：

| 算例 | 0 到 0.1 | 0.1 到 0.2 | 0.2 到 0.3 | 0.3 到 0.4 | 0.4 到 0.5 | 0.5 到 0.6 | 0.6 到 0.7 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 mm 薄板 | 43.0 | 46.4 | 51.5 | 54.6 | 52.9 | 44.9 | 47.6 |
| 2 mm 薄板 | 56.2 | 61.8 | 59.3 | 57.6 | 53.0 | 59.3 | 65.4 |
| 1.5 mm 薄板 | 56.2 | 69.8 | 76.5 | 67.1 | 66.5 | 64.4 | 64.0 |
| 3 mm 双层片 | 72.8 | 85.2 | 76.7 | 72.0 | 68.6 | 65.8 | 70.9 |
| 3 mm 三层片 | 70.3 | 83.2 | 80.0 | 73.3 | 68.1 | 66.4 | 67.7 |
| 2 mm 双层片 | 75.4 | 85.8 | 84.5 | 81.7 | 70.3 | 68.9 | 74.1 |

开头 0.3 s 是挡板附近的流动从 Fluent 场调整到各自模型的过程（多数算例在 0.1 到 0.3 s 有一个峰），所以窗口取 0.3 到 0.7 s。

## 三、分析

### 1. 薄板随加密上升，但收敛很慢

| 间距 | 3 mm | 2 mm | 1.5 mm |
| --- | --- | --- | --- |
| 挡板，占 Fluent 的比例 | 65% | 77% | 87% |
| 每加密一级增加 | | +9.0 | +6.4 |
| 全部壁面，占 Fluent 77.8 的比例 | 64% | 76% | 84% |

- 两次增量之比 9.0 / 6.4 = 1.41。按 D = D₀ − k dx^p 拟合，p 接近 0；三个点各有 1 到 2 mN·m 的误差，阶数定不准，只能说比一阶慢。
  按一阶外推，要约 0.8 mm 才到 69.7，粒子数是 3 mm 的约 50 倍。
- 与 `log/2026-09-30_thin-plates-mirror.md` 测试 3、4 一致：薄板的力矩在 4 / 2 / 1 mm 都没有收敛，一直随加密上升。

### 2. 贴合片：同样间距下刹得多，厚度不重要

- 3 mm 双层片 63.0，比 3 mm 薄板高 18，比 1.5 mm 薄板还高 2.6。算 0.7 s 用 4.6 min，1.5 mm 薄板用 62.4 min。
- 三层片 63.1，与双层片相同。模型里的挡板从 6 mm 加到 9 mm 厚，阻力不变。多出来的阻力来自普通固体粒子的表示，
  不是挡板变厚带来的阻塞，也不是两侧流体隔着片的核函数耦合：快照里离中面最近的流体，双层片 4.2 mm、三层片 5.6 mm，
  两侧相距 2.8 和 3.7 个间距，核半径是 3 个间距，双层片两侧之间的核函数值已接近零，三层片完全隔开，结果一样。
- 2 mm 双层片（模型里 4 mm 厚）67.8，比 3 mm 双层片高 4.8，到 Fluent 的 97%；全部壁面 73.7，到 95%。同样从 3 mm 加密到 2 mm，薄板增加 9.0，贴合片增加 4.8：贴合片离收敛近得多。

为什么同样的间距下分侧镜像的薄板比固体片刹得少，本篇没有分开，下面两条是推测（待确认）：

- 诊断篇第三节：贴板第一层粒子的压力两侧都偏高，尾流一侧比来流一侧还高。分侧镜像的虚粒子不参与密度扩散，近壁的阻尼少一半（薄板 log 已知问题 1）。
- 板的内缘：连线不穿过板轮廓的邻居照常相互作用，内缘附近两侧的压力可以绕过板边互通，板的有效宽度可能偏小。
  固体片的边是 2 到 3 个间距厚的实体，绕行的路径更长。

### 3. 同时得到的其他部件

- **Rushton**：同一流场里薄板叶片 46.0 / 55.9 / 61.4，增量之比 1.8（阶数约 0.8），1.5 mm 与 Fluent 的 60.7 一致。
  3 mm 低 24%，2 mm 低 8%，与 `log/2026-10-03_impeller-nearfield.md` 的结论一致。
- **槽底**：1.09 / 0.99 / 0.94，Fluent 2.46，不随加密改善。全部是粘性力。1.5 mm 时第一层粒子离壁 0.75 mm，y⁺ 仍约 15，
  粘性底层解析不了。Fluent 的 LES 在这里用壁面律。要补的是壁面模型，不是分辨率。
- **圆柱面**：4.7 / 4.4 / 4.3，Fluent 5.6。
- **PBT**：18.3 / 18.9 / 20.4，Fluent 15.0，加密后偏差变大（1.5 mm 高 36%）。原因未查，另案。
- **y < 25 mm 的壁面**（槽底、圆柱面下段、挡板下端和支架）：12.6 / 14.8 / 19.0，双层片 16.4；Fluent 这一段是 17.2（`log/2026-10-03_impeller-nearfield.md` 第十一节）。

### 4. 对角动量收支的意义

在 Fluent 的流场里（L 约 222 到 238 g m²/s）：

| | 输入（Rushton + PBT） | 全部壁面 | 差 | 窗口里 L 的走向 |
| --- | --- | --- | --- | --- |
| 3 mm 薄板 | 64.3 | 50.0 | +14.3 | 上升 |
| 3 mm 双层片 | 65.9 | 69.3 | −3.4 | 下降 |
| Fluent | 75.7 | 77.8 | 平衡 | |

3 mm 薄板挡板制动不够，旋流会一直涨到制动追上输入（从静止起算的 20 s 算例停在 L 292 到 297，比 Fluent 高 28%）。
双层片在 Fluent 的角动量下已经制动略多于输入，稳定的 L 应在 Fluent 附近或略低：3 mm 的 Rushton 力矩还低 24%，输入偏小。

## 四、新增脚本 `experiment/v1/checks/_compare_wall_torques.py`

原来的代码：没有。诊断篇用 scratchpad 里的一次性脚本手算。

改成了什么：新的分析脚本，只读 `_check_tank_energy.py` 的 CSV。每个 `--case 系列 间距 静止CSV 搅拌CSV`：

    流动产生的力矩（每一类） = 搅拌 CSV 在 --window 内的平均 − 静止 CSV 在 --rest-window 内的平均
    挡板 = Σ plate_baffle_k + Σ wall_at_baffle_k     （各自减静止值）
    全部壁面 = csv_walls（所有静止边界粒子与静止薄板的合计）
    标准误差 = （0.1 s 分块平均的样本标准差）/ √（块数）

打印表格；`--figure` 画挡板和全部壁面对粒子间距的图，同一系列连线，Fluent 画虚线。本篇的表和图：

    K=output/kecause
    python experiment/v1/checks/_compare_wall_torques.py \n      --case "thin plates" 3 $K/rest_s3_plates_pb2000_newpbt_smooth.csv $K/s3_plates_pb2000_newpbt_smooth_fluentinit.csv \n      --case "thin plates" 2 $K/rest_s2_plates_smooth.csv $K/s2_plates_smooth_fluentinit.csv \n      --case "thin plates" 1.5 $K/rest_s15_plates_smooth.csv $K/s15_plates_smooth_fluentinit.csv \n      --case "2-layer sheets" 3 $K/rest_s3_sheet2baf_smooth.csv $K/s3_sheet2baf_smooth_fluentinit.csv \n      --case "2-layer sheets" 2 $K/rest_s2_sheet2baf_smooth.csv $K/s2_sheet2baf_smooth_fluentinit.csv \n      --case "3-layer sheets" 3 $K/rest_s3_sheet3baf_smooth.csv $K/s3_sheet3baf_smooth_fluentinit.csv \n      --figure docs/figures/tank_baffle_drag_resolution.png

影响：不改求解器、生成器和已有脚本，已有结果不变。

待确认：分块标准误差假设各 0.1 s 块相互独立，窗口只有 0.4 s，误差是粗估；静态力矩取静止算例 0.02 到 0.2 s 的平均，
这段时间里格子在 p_b 下还在调整（`sph-at-rest-and-wall-torque`），各类的静态值有不确定性，但全部壁面的合计不受影响。

## 五、2 mm 丢粒子：光滑槽壳在格子固体旁边留了缝（生成器已修正）

现象：2 mm 的两个算例丢粒子，3 mm 和 1.5 mm 不丢。

| 算例 | 静止 0.2 s | Fluent 起算 0.7 s |
| --- | --- | --- |
| 2 mm 薄板 | 54 | 91 |
| 2 mm 双层片 | 154 | 979 |

位置（快照里活着的流体粒子，按求解器的 `live_slot_mask` 规则取，快照里排在活粒子数之后的槽位是旧的副本）：

- 任何时刻都有约 20 个流体粒子在槽底以下：r 11.6 到 14.6 mm、y −72 到 −63 mm，紧贴轴承座（格子固体，r 14.5 mm）外侧。
  转动的钟形罩（转子，r 约 18 mm 以内，下沿 y 约 −58 mm）和槽底之间有一层流体，0.5 s 时那里的粒子速度中位数 0.34 m/s，
  与钟形罩在 r 16 mm 处的线速度 0.35 m/s 相当。这层流体从轴承座和槽底壳之间的缝往下走，出了计算域就被删掉。
  背景压力 2000 Pa 把粒子往没有流体的地方推，所以静止时也丢。
- 另有 14 到 15 个流体粒子停在 r 145 mm（槽壁第一层壳粒子的半径）、y 2 到 18 mm、方位 55.8° 和 307.2°：
  两个挡板支架（宽 30.6 mm，贴着槽壁，y 1 到 20.7 mm）侧面和槽壁壳之间的缺口里。2 mm 和 1.5 mm 都有，3 mm 没有。它们出不去。

原来的代码（`utils/geometry/_demo_stirred_tank_30l.py`，`--smooth-walls`，2026-10-03 加的，默认关）：

    shell_points = smooth_tank_shell(dx, border)
    shell_points = shell_points[(wall_solid.signed_distance(shell_points) > 0.0)
                                & (rotor_region.signed_distance(shell_points) > 0.0)]

落在格子固体（轴承座、挡板支架）里面的壳粒子全部丢掉，固体保留。格子固体的表面是台阶，最外面的格点离真实表面最多约 1 个间距，
台阶的缺口和壳之间就留了缝。2 mm 时轴承座外侧的缝正好能过一个粒子。

改成了什么：

    depth = wall_solid.signed_distance(shell_points)          # 负 = 在固体里面
    fills_notch = (depth > -dx) & ~points_closer_than(shell_points, sites[is_wall_solid], 0.6 * dx)
    outside_rotor = rotor_region.signed_distance(shell_points) > 0.0
    shell_points = shell_points[((depth > 0.0) | fills_notch) & outside_rotor]

固体里的壳粒子，离固体表面不到 1 个间距、且 0.6 dx 以内没有该固体的格点，就保留，用来填台阶的缺口。转子里的照旧全部丢掉。
生成时打印填了多少个（"... of them in notches of lattice solids"）。

影响：

- 默认生成（不加 `--smooth-walls`）逐字节不变：4 mm 默认参数，新旧生成器的 6 个输出文件逐一比较。
- 3 mm `--smooth-walls`：没有要填的缺口，薄板和双层片两个算例的输出逐字节不变。本篇和此前所有 3 mm 光滑槽壳的结果不受影响，
  正在跑的 20 s 算例也一样。
- 2 mm：补 137 个壳粒子，112 个在轴承座外侧（r 14.1 mm，槽底下三层），25 个在支架侧面的槽壁第一层（r 145 mm，方位 56°、67°、176°、307°）。
- 1.5 mm：补 102 个，8 个在轴承座，94 个在支架侧面。
- 两个分辨率的流体粒子都不变（fluid.obj 逐字节相同），所以 Fluent 流场的插值文件可以沿用。
- 验证（`output/kecause/run_smoothfix_test.sh`，算例 `s2_sheet2baf_smoothfix`，与 20 s 算例同时在 GPU 上跑）：

  | 2 mm 双层片 | 修正前 | 修正后 |
  | --- | --- | --- |
  | 静止 0.2 s 丢的粒子 | 154 | 0 |
  | 静止 0.2 s 末，槽底以下的流体粒子 | 24 | 0 |
  | 静止 0.2 s 末，r > 145 mm 的流体粒子 | 16 | 0 |
  | 静止时全部壁面的静态力矩，mN·m | +0.9 | +0.3 |
  | Fluent 起算 0.7 s 丢的粒子 | 979 | 0 |
  | Fluent 起算 0.7 s 末，槽底以下 / r > 145 mm 的流体粒子 | 21 / 15 | 0 / 0 |
  | KCG 回退计数，静止 / Fluent 起算 | 9262 / 9687 | 0 / 0 |
  | 流动产生的挡板力矩 / 全部壁面，mN·m | 67.8 ± 3.0 / 73.7 ± 2.9 | 66.8 ± 2.9 / 73.1 ± 2.7 |
  | Rushton / PBT，mN·m | 55.0 / 20.0 | 56.0 / 19.7 |

  修正后不再丢粒子，KCG 回退也没有了；力矩在误差范围内不变，第二节的 2 mm 双层片结论成立。
  2 mm 薄板的旧算例 KCG 回退 11584 / 41048，其余分辨率都是 0。推测是漏进缝里和槽底以下的粒子邻居不全，修正矩阵退化。

待确认：填缺口的阈值（1 个间距深、0.6 dx 间隔）是按这两处缺口定的。换几何或换间距时，用静止 0.2 s 检查粒子数，
并看流体粒子是否出现在 r > R + 0.5 dx 或槽底以下。

## 六、结论和下一步

- 3 mm 下挡板改用双层贴合片：同一流场里刹出 Fluent 的 90%，算量不变，静止时也没有薄板挡板的密度上漂
  （`log/2026-10-03_bulk-energy-tests.md` 4.1；本篇 0.2 s 静止：薄板平均密度 998.41 → 998.83，双层片 998.30 → 998.23）。
  三层没有更多好处。
- 叶轮继续用薄板：Rushton 的薄板叶片 1.5 mm 已到 Fluent 的值；2026-10-03 格子槽壳的对照里片状叶轮的 Rushton 力矩更低（43 对 52 mN·m，4 到 6 s）。
- 剩下的偏差：3 mm 的 Rushton 力矩低 24%（薄板叶片的分辨率），槽底粘性摩擦（要壁面模型），PBT 偏高。
- 在跑（`output/kecause/run_sheet_rest20.sh`，17:31 开始，约 2.5 小时）：`s3_sheet2baf_smooth` 从静止起算 20 s，
  快照 10、12、14、16、18、20 s，与薄板的 20 s 算例（`log/2026-10-04_smooth-from-rest-20s.md`）对比旋流、主体动能、Rushton 力矩和槽底流型。

## 文件

- 新：`experiment/v1/checks/_compare_wall_torques.py`，`docs/figures/tank_baffle_drag_resolution.png`，本 log。
- 改：`utils/geometry/_demo_stirred_tank_30l.py`（`--smooth-walls` 填格子固体的台阶缺口，第五节）。
- 不入库：`output/kecause/cases/{s2_plates_smooth, s15_plates_smooth, s3_sheet2baf_smooth, s3_sheet3baf_smooth, s2_sheet2baf_smooth}`，
  修正后重新生成的 `{s2_sheet2baf, s2_plates, s15_plates, s3_sheet2baf, s3_plates}_smoothfix`，
  `output/kecause/{rest_,}<算例>{,_fluentinit}.csv / .log / _final.npz / _t0.500.npz`，`fluent_fine25_on_<算例>.npy`，
  驱动 `run_baffle_study.sh`、`run_baffle_study2.sh`、`run_sheet_rest20.sh`、`run_smoothfix_test.sh`。
