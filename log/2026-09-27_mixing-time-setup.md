# 2026-09-27 混合时间算例：验证数据、τ95 的算法、运行和分析脚本、本机流程测试

## 目的

按 Rautenbach 等（2026，Comput. Chem. Eng. 210, 109615）的做法算 30 L 碟底槽的混合时间 τ95，
与数据集 DARUS-5523（https://doi.org/10.18419/DARUS-5523）中的实验和两套 LES 比较。
本次提交准备代码和流程；正式算例在并行科技上运行，结果另记。

## 验证数据（DARUS-5523）

数据集比论文正文给的多：实验的电导率原始记录（11 次重复）、M-Star 每次重跑的两个探针时间曲线、
Fluent 每次混合模拟的探针曲线、每个实现的 τ95 汇总表，以及作者自己的后处理脚本都公开了。

τ95 的参考分布（`05_final_results_collections/`，平均值和标准差取自
`mixing_time_statistics_collection_paper_200rpm.tab`，个数和范围由 `longest_mixing_times_mstar_exper_high_low_res_all_mstar.tab` 统计）：

| 来源 | 个数 | 平均 (s) | 标准差 (s) | 范围 (s) |
| --- | --- | --- | --- | --- |
| 实验 | 11 | 24.91 | 2.71 | 19–30 |
| M-Star LBM-LES，LX400（论文采用的网格，约 9×10⁷ 格点） | 50 | 29.10 | 3.86 | 20.4–37.5 |
| Fluent FV-LES | 50 | 29.17 | 4.42 | 19.9–36.7 |
| M-Star 最粗网格 LX080（约 70 万格点） | 60 | 23.73 | 1.84 | 19.4–28.6 |

- 单次实现的 τ95 波动很大（标准差 3–4 s，范围约 20–40 s），所以必须用多个实现的分布比较。
- M-Star 的网格序列：LX080 为 23.7 s，LX150 为 26.1 s，LX400 为 29.1 s，LX500 为 28.2 s。粗网格混合更快，说明结果依赖分辨率。
- 论文表 2：25–75 s 内全槽平均速度的滚动平均，M-Star 0.1495 m/s，Fluent 0.1566 m/s。这个量不需要示踪剂就能比较。

两个软件的模拟方式（从数据集的原始记录核对）：

| 软件 | 每次模拟 | 示踪剂 | 重跑 | 总模拟时间 | τ95 个数 |
| --- | --- | --- | --- | --- | --- |
| M-Star | 0 到 64–68 s | 10 个，1 s 脉冲，从 25 s 起每秒一个 | 5 次 | 约 330 s | 50 |
| Fluent | 种子算例 0–34 s，再从 25、26 … 34 s 各跑 60 s | 每次 1 个 | 10 个起点 × 5 次 | 约 3030 s | 50 |

M-Star 的探针记录间隔为 0.02 s。论文认为一次模拟里多次注入可以部分代替独立重跑（结论 3）。

## τ95 的算法，以及与数据集的核对

照搬数据集脚本 `03_codes_for_post/mixing_times_in_simulation_10dyes_control_vol_filer_for_longest.py`：

```
C*(t) = (C(t) − C(第一个采样)) / (C(最后一个采样) − C(第一个采样))
在 t ≥ t0（脉冲开始）的采样中，找最后一个 C* 落在 [0.95, 1.05] 之外的采样；
其后第一个"本身和后两个采样都在带内"的采样时刻为 t_s（搜索到记录结束前 3 个采样为止），τ95 = t_s − t0
一个脉冲报告两个探针中较大的值（"longest"）；从不稳定下来的探针不参加比较
```

`experiment/v1/checks/_analyze_mixing_time.py --mstar PROBE_1 PROBE_2` 用同一算法处理 M-Star 的原始探针文件。
对 LX400 的 trial_02 和 trial_03（`02_simulation_results/01_mixing_time_results/00_raw_txt/grid_study_200rpm_LX400/`），
算出的 20 个值与汇总表 LX400 一列的第 11–30 行逐个相同（到小数点后两位）：

| 示踪剂 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_02，我们的脚本 | 28.16 | 29.90 | 26.32 | 25.36 | 27.72 | 31.50 | 32.36 | 31.40 | 29.86 | 28.88 |
| trial_02，数据集 | 28.16 | 29.90 | 26.32 | 25.36 | 27.72 | 31.50 | 32.36 | 31.40 | 29.86 | 28.88 |
| trial_03，我们的脚本 | 33.90 | 32.98 | 32.48 | 31.62 | 32.82 | 31.82 | 28.72 | 29.88 | 28.88 | 27.88 |
| trial_03，数据集 | 33.90 | 32.98 | 32.48 | 31.62 | 32.82 | 31.82 | 28.72 | 29.88 | 28.88 | 27.88 |

所以我们的 τ95 与论文的数字可以直接比较。另外脚本还用守恒总量给出精确的混匀值 C∞ = Σ m C / Σ m，
作为第二种归一化；以及全槽指标（见下）。

## 代码改动

| 文件 | 内容 |
| --- | --- |
| `experiment/v1/_run_v1_headless.py` | 探针 CSV 在原有列之后加 `fluid_mass`、`mean_speed`（流体粒子速度大小的平均），以及每个场的 `cov:`（质量加权的变异系数）和 `mixed5:`（浓度在均值 ±5% 以内的质量占比）；新选项 `--scalar-snapshot-times T ...`、`--scalar-snapshot-dir`、`--scalar-snapshot-fields`，在指定时刻保存流体粒子的位置、质量、uid 和浓度 |
| `experiment/v1/utils/simulator_v1.py` | `scalar_snapshot()` 多返回流体粒子的速度（本来就读了 velocity_mass，只是没返回） |
| `utils/geometry/_demo_stirred_tank_30l.py` | 新选项 `--shift-correction`，在 `scalars:` 块写 `shift_correction: true`；限制器保持默认（开） |
| `experiment/v1/checks/_analyze_mixing_time.py` | 新增：按数据集算法算 τ95，另给精确 C∞ 归一化的 τ95、全槽 τ95（`mixed5` 此后一直 ≥ 0.95 的时刻）、25 s 以后的平均速度和功率数；画探针曲线、与参考分布的对比、全槽指标和切片图；`--mstar` 模式处理数据集原始文件 |

原有的 CSV 列顺序不变，新列都加在后面。

## 本机流程测试（4070 Ti SUPER）

**单示踪剂**：4 mm 碟底槽，c₀ = 20 U_tip，SGS 开，一个脉冲 0.3–0.5 s，跑 34,300 步（3.0 s），每 229 步（0.02 s）采样一次，
在 0.3、0.4、0.6、1、2、3 s 保存快照。

- 103 步/s，5.5 分钟；溢出为 0，活粒子数不变。
- 脉冲结束后示踪剂总量的相对变化为 6.5×10⁻¹⁰。
- 3 s 内示踪剂还没到探针：转子刚启动，液面附近几乎不动，注入团基本停在原地。所以 τ95 为 NaN，这是预期结果。
  正式算例在 25 s 流场发展后才注入。
- 分析脚本读 CSV、画曲线、画切片图都正常。图在本地 `output/mixing_trial/analysis/`。

**变体的启动检查**（4 mm，一个示踪剂，2000 步）：

| 变体 | 步/s | 溢出 |
| --- | --- | --- |
| SGS 开，位移修正开 | 97.0 | 0 |
| SGS 关，位移修正关 | 115.5 | 0 |
| SGS 关，位移修正开 | 112.1 | 0 |

**正式配置的速度**：3 mm、10 个示踪剂、SGS 开，3000 步，36.3 步/s（含每 305 步一次的采样）；
显存 486 MB 加 defrag 暂存 436 MB。10 个脉冲按时注入（测试时把间隔压到 0.02 s）。
按 3 mm 碟底槽无标量时 5090 与本机的速度比（约 1.85 倍）估计，5090 上约 67 步/s。

## 正式算例（并行科技 gpu_5090，每个 1 张卡）

共同参数：3 mm 碟底槽，h/dx = 3，c₀ = 20 U_tip，dt = 6.564×10⁻⁵ s；10 个示踪剂，1 s 脉冲，从 25 s 起每秒一个，
注入点和 5 mL 注入球按论文表 1；探针为论文的两个测点，Shepard 半径 h = 9 mm；每 305 步（0.020 s）采样一次；
1,142,700 步 = 75.0 s（覆盖论文表 2 的 25–75 s 窗口，最后一个脉冲之后还有 41 s）；每 1000 步记录分桨力矩。

| 名称 | SGS | 位移修正 | 快照 |
| --- | --- | --- | --- |
| mix_3mm_base | 开 | 关 | tracer_01 在 25.5、26、27、29、31、35、40、50 s |
| mix_3mm_shift | 开 | 开 | 无 |
| mix_3mm_nosgs | 关 | 关 | 无 |
| mix_3mm_shift_nosgs | 关 | 开 | 无 |

预计每个 4.5–5 小时，合计约 19 卡时。作业脚本 `~/run/sph/mixing/job_mix.sh`（全文见下），代码为本次提交，
以 `git archive` 加上本地编译的 SPIR-V 打包上传（集群上没有 glslc）。

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 10:00:00
#SBATCH -o /data/run01/scxm138/logs/mix_%x_%j.out
# usage: sbatch -p gpu_5090 -J <name> job_mix.sh NAME DX STEPS PROBE_EVERY SGS(0|1) SHIFT(0|1) [SNAPSHOT_FIELD]
NAME=$1; DX=$2; STEPS=$3; PROBE_EVERY=$4; SGS=$5; SHIFT=$6; SNAPSHOT_FIELD=${7:-}
source ~/run/tools/env.sh
export PYTHONPATH=$HOME/run/sph/vulkan-demo-fsi
export __GL_SHADER_DISK_CACHE_PATH=/tmp/nvcache_$SLURM_JOB_ID
cd ~/run/sph/vulkan-demo-fsi
OUT=~/run/sph/mixing
CASE=/dev/shm/scxm138/cases/$NAME
FLAGS="--tracers 10 --injection-start 25 --injection-interval 1 --injection-duration 1"
[ "$SGS" = 1 ] && FLAGS="$FLAGS --sgs";  [ "$SHIFT" = 1 ] && FLAGS="$FLAGS --shift-correction"
python utils/geometry/_demo_stirred_tank_30l.py --dx $DX --hdx 3 --c0-factor 20 $FLAGS --out "$CASE" --no-preview
sed -i "s#material_library: materials.yaml#material_library: $CASE/materials.yaml#" "$CASE/case.yaml"
python ~/run/sph/bench_v1.py "$CASE/case.yaml" --max-steps $STEPS --probe-every $PROBE_EVERY \
    --probe-log "$OUT/${NAME}_probes.csv" --torque-every 1000 --torque-log "$OUT/${NAME}_torque.csv" \
    --torque-split-height 0.1 --torque-shaft-radius 0.007 \
    [--scalar-snapshot-times 25.5 26 27 29 31 35 40 50 --scalar-snapshot-dir ... --scalar-snapshot-fields $SNAPSHOT_FIELD]
```

（上面省略了日志和计时行；实际脚本另外把 case.yaml 复制到输出目录。）

### 提交记录（2026-09-27 14:36，北京时间）

- 代码：7387966，打包为 `~/run/sph/fsi_code_20260927.tar.gz` 并解到 `~/run/sph/vulkan-demo-fsi`，
  旧目录改名为 `vulkan-demo-fsi.bak_20260927`。
- 先在 gpu_5090 上用 srun 跑了约 2 分钟的冒烟测试（作业 1630509）：4 mm、一个示踪剂、SGS 和位移修正都开、
  0.1 s 时存快照，2000 步，126 步/s，溢出为 0，探针记录和快照都正常。
- 四个作业都用 `sbatch -p gpu_5090 -t 12:00:00`（按实际用时计费，时限只是上限）：

| 作业号 | 名称 | 开跑后约 10 分钟时的速度 | 预计还需 |
| --- | --- | --- | --- |
| 1630511 | mix_3mm_base | 约 54 步/s | 约 5.7 小时 |
| 1630512 | mix_3mm_shift | 约 44 步/s | 约 7.0 小时 |
| 1630513 | mix_3mm_nosgs | 约 64 步/s | 约 4.8 小时 |
| 1630514 | mix_3mm_shift_nosgs | 约 54 步/s | 约 5.7 小时 |

  速度按 90 s 内的步数差估算，受采样间隔影响只是粗略值。合计约 24 卡时，比事先估计的 19 卡时多：
  带标量时 5090 与本机的速度比没有无标量时那么大。
- 日志在 `/data/run01/scxm138/logs/mix_<名称>_<作业号>.out`，输出在 `~/run/sph/mixing/`。

## 结果出来后的分析

```
python experiment/v1/checks/_analyze_mixing_time.py mix_3mm_base_probes.csv --case mix_3mm_base_case.yaml \
    --torque-log mix_3mm_base_torque.csv --snapshots mix_3mm_base_snapshots --out-dir output/mixing --label base
```

## 与论文做法的差别

- **注入**：我们在 1 s 内把注入球内流体粒子的浓度设为 1；M-Star 用 1 s 的源项。被标记的液体量不同，
  但 τ95 按终值归一化，对注入量不敏感；两者都在 1 s 内沿当地流动形成一条示踪带。
- **探针**：我们对探针点周围 9 mm 内的流体粒子做 Shepard 平均；M-Star 用探针杆下方的一个小控制体。探针杆本身都作为固体建模。
- **实验**：注入的是 4 M 氢氧化钠溶液，有密度差和注入动量，论文认为这使实验的 τ95 偏短；模拟都是中性的被动标量。
