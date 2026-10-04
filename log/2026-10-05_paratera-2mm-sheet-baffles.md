# 2026-10-05 并行科技：2 mm 光滑槽壳加双层片挡板，从静止起算 20 s

分支 `test`。用户：2 mm 加双层片挡板算一下吧，提交并行科技。

为什么是这个算例：3 mm 的基准（`log/2026-10-04_sheet-baffles-from-rest-20s.md`）旋流已接近 Fluent，但 Rushton 力矩低 22%，
子午循环弱 36%。在 Fluent 的流场里 2 mm 的 Rushton 到 Fluent 的 92%，双层片挡板到 97%
（`log/2026-10-04_baffle-representation-resolution.md`）。PBT 叶尖多出约 5 mN·m，加密不变
（`log/2026-10-04_fluctuation-scales-and-pbt-bands.md`），这个算例里也会有。

## 一、算例

与 3 mm 基准相同，只是间距 2 mm：光滑槽壳（含 2026-10-04 的缺口修正）、双层贴合片挡板、薄板叶轮、改正后的 PBT、STL 轮毂、
p_b 2000 Pa、c₀ = 20 U_tip、h/dx 3。节点上生成：fluid 3,714,886，wall 456,618，rotor 16,758，合计 4,188,262（填缺口 137 个），
与本机的 `s2_sheet2baf_smoothfix` 逐项相同。dt = 4.376e-5 s，20 s 共 457,040 步。

    sbatch -p gpu_5090 -J s2sh20 -t 07:00:00 job_tank_energy_shm.sh s2_sheet2baf_smooth_fromrest20 0.002 \
        "--time 20 --every 900 --split-height 0.025 --dump-times 10 12 14 16 18" \
        --skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates \
        --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2

（作业脚本给生成器加 `--dx 0.002 --hdx 3 --c0-factor 20 --no-preview`。）每 900 步一行（约 0.039 s，3 mm 算例是每 300 步 0.020 s：
2 mm 一次记录要读回 480 万个槽位、约 3 s，每 450 步记一行要多花约 0.8 小时）。

## 二、部署

- 代码：提交 c1e23db 的 `git archive` 加本机编译的 14 个 SPIR-V，解到并行科技 `~/run/sph/vulkan-demo-fsi_c1e23db`（新目录，旧的
  `vulkan-demo-fsi_pb` 不动）；`force_plates.comp.spv` 的 md5 两边相同（54d08ccc…）。
- 新作业脚本 `~/run/sph/tank/job_tank_energy_shm.sh`（本机副本 `SPH lifeline/cluster_jobs/paratera/job_tank_energy_shm.sh`）：
  在节点的 /dev/shm 生成算例、运行 `_check_tank_energy.py`（经 `run_check_pb.py`，用预编译的 spv），CSV 和快照写在 /dev/shm，
  结束时复制到 `~/run/sph/tank`；到时限被杀（SIGTERM）时也先复制；每 10 分钟把 CSV 复制回家目录（`NAME.csv.partial`）。
  `CODE_DIR` 可换代码目录。Slurm 输出 `/data/run01/scxm138/logs/tank_<作业名>_<作业号>.out`。
- 验证：作业 1661284（`t2tank`，同样设置算 0.01 s）67 s 跑完，生成 37 s，粒子数同上，CSV 和最终快照都复制回来了。
  按每 50 步一行时 10.7 步/s，其中每次读回约 3 s；推得纯推进约 29 步/s。

## 三、运行

作业 1661323，gpu_5090 一张 RTX 5090（节点 wqd10naf02g2），北京时间 2026-10-04 20:17 开始。按每 900 步约 34 s 估计约 4.8 小时，
时限 7 小时，约 5 卡时（另有验证作业约 0.02 卡时）。

结果：（运行结束后补在这里）

## 四、结束后要做的

- 下载 CSV 和快照（6 个，每个 231 MB）到 `output/kecause/`。
- 与 3 mm 基准、薄板 20 s 和 Fluent 对比，用 3 mm 那篇的同一套：`sheet20_windows.py` 的窗口表、`_analyze_rushton_lower_flow.py`、
  `_analyze_axisymmetric_energy.py`、`_plot_energy_maps.py`、`_plot_meridional_streamlines.py`、`_analyze_fluctuation_scales.py`、`_analyze_pbt_bands.py`（只看流动）。
- 重点：Rushton 力矩和子午循环是否随 2 mm 补上，角动量、主体动能、非轴对称部分，以及 PBT 叶尖。
