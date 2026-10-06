# 2026-10-06 生成器选项 `--solid-disk LAYERS`：Rushton 圆盘改成格子实体，叶片仍是薄板（射流倾角试验）

## 起因

子午环流比 Fluent 弱 20% 的差额全在 Rushton 高度和壁面附近（方位平均轴向动能 Fluent 104 mJ 对 2 mm 82 mJ，差的 22 mJ 中
13 mJ 在 y 19 到 58 mm 的 r > 110 mm，5 mJ 在两桨之间的壁面）。原因是 Rushton 射流的倾角：Fluent 向下斜 35°（u_r 0.262，u_y −0.185），
我们 15 到 17°（0.33 到 0.34，−0.09 到 −0.11）；射流更水平地打到壁面后分成两股，约 0.5 N D³ 进碟底小环流（0.60 对 Fluent 0.34），
主环流在 y 60 到 120 弱 18%（1.13 对 1.38 N D³）。y 150 以上、PBT 泵送（0.67 对 0.66）、Rushton 出流（0.96 对 1.00）都与 Fluent 一致。

倾角在 Rushton 处丢，不是远场拖的：3 mm 从 Fluent 流场起算，倾角 0.3 s 40°，0.7 s 24°，1 s 20°；同时圆盘下方叶片间隙的轴向速度
从 −0.10 m/s（Fluent −0.096）1 s 内到 0，之后反转为 +0.14，从下面的进流 0.02 升到 0.27 N D³，圆盘下方的射流 0.57 降到 0.26，
圆盘高度的出流 0.108（Fluent 0.037）。Fluent 里从圆盘边缘外绕下、穿过下半叶片间隙继续向下的那股流，在我们这里到圆盘高度就径向出去。

猜测：圆盘是垂直于轴的薄板环，边缘 r 32 mm；与叶尖一样，边缘一个核半径内上下两侧直接互通（薄板规则只切断轮廓以内的穿越），
圆盘下方的低压拉不住上方的来流。验证：圆盘改成格子实体粒子，叶片不动，从 Fluent 流场算几秒看倾角是否保持。

## 改动（`utils/geometry/_demo_stirred_tank_30l.py`）

原来的代码：`--thin-plates` 时圆盘是薄板环（`thin_plate_sheets` 的 `with_impeller` 同时管叶片和圆盘），`build_solids(with_disk=not impeller_plates)`
不生成格子圆盘，格子圆盘厚度取 max(真实 2.5 mm, thin_layers × dx)。

改成了什么：

- `thin_plate_sheets(..., with_disk=True)`：`with_disk=False` 时不生成圆盘薄板。
- `build_solids(..., disk_thickness=None)`：给定时圆盘厚度用它，不用 max(真实, thin)。
- 新选项 `--solid-disk LAYERS`：需要 `--thin-plates` 且没有 `--conformal-blades`；圆盘为 LAYERS × dx 厚的格子转子圆柱，叶片薄板不变。
  叶片薄板穿过圆盘平面处，离薄板粒子 0.6 dx 以内的格子圆盘粒子按原规则被丢掉（薄板本身挡住那条缝）。

影响：不加选项时 3 mm 基准（光滑壳、双层片挡板、薄板叶轮）的 19 个输出文件逐字节相同（已比对）。
加 `--solid-disk 1` 的 3 mm 算例：薄板 13 块变 12 块（984 个薄板粒子），格子转子粒子 3,961 变 4,267（圆盘一层 306 个），流体数不变。

## 算例

并行科技 gpu_5090，`job_impeller_parts_shm.sh`，3 mm，从 Fluent 25 s 流场算 4 s，窗口 0.3 到 0.7、1 到 2、2 到 4 s，
dump 0.5、1、2、3、4 s：`s3_sheet2baf_smooth_soliddisk1_fluentinit`（实体圆盘）和 `s3_sheet2baf_smooth_fluentinit4s`（薄板圆盘基准）。
判据：1 到 4 s 的射流倾角、圆盘下方叶片间隙的轴向速度、从下面的进流、碟底小环流（`mean_flow_regions.py`、`_analyze_rushton_lower_flow.py`）。

## 待确认

- 实体圆盘一层（3 mm 厚，真实 2.5）两面共用一个压力，与挡板片状相同的近似；若它保住了倾角，说明是圆盘边缘的互通，
  而不是厚度。
- 圆盘的假力矩（薄板 3 mm 时 9.1 mN·m）在实体圆盘下应消失，部件表里看。
