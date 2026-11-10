# OW-15 提交：30 L 单向分辨率序列加 1.5 mm，看扭矩是否收敛

日期：2026-11-11。用户："现在 30 L 的力矩不收敛，速度场也偏高，到 paratera 提交一个 1.5 mm 的算例，看看力矩能不能收敛"。

算例含义：OW = 单向（Monod 汇，细胞不反馈状态）；OW-4、OW-3、OW-2 是 30 L 罐 4、3、2 mm 的分辨率序列（`log/2026-11-02_production-OW-submission.md`、`2026-11-04_OW2-results.md`），OW15 是新加的 1.5 mm。

## 原来的代码

- 不改求解器。代码 688e87a（与 OW-2、OW-3、OW-4 相同），14 个 spv，`force_scalar.comp.spv` md5 502dc896。
- 已有扭矩（`output/production/resolution_series/OW*_torque.csv`，t ≥ 20 s 平均，除以质量因子 1.2187，mN·m）：

| 算例 | dx mm | 总扭矩 | Rushton（下） | PBT（上） | 每 10 s 窗的总扭矩，0 到 90 s |
|---|---|---|---|---|---|
| OW-4 | 4 | 61.6 | 43.0 | 18.4 | 65.3 62.6 61.5 61.9 62.4 61.6 61.8 60.4 61.6 |
| OW-3 | 3 | 66.3 | 48.0 | 18.2 | 70.0 69.7 66.1 65.8 67.6 66.5 66.5 67.0 64.3 |
| OW-2 | 2 | 80.0 | 60.2 | 19.8 | 82.1 79.6 79.6 77.7 81.5 78.6 82.2 79.7 80.8 |

  参考 Fluent LES：Rushton 60.7 ± 2.9，PBT 15.0 ± 1.7 mN·m。总扭矩 4 到 3 mm 升 8 %，3 到 2 mm 升 21 %，没有收敛的迹象。

## 改成了什么

1. **Paratera 运行目录重建**。11-08 存储迁移后原 `~/run`（/data_old/run01/scxm138）改名 `~/run_old`，新的 `~/run`（/data/run01/scxm138）是空的。为了让原作业脚本不改就能用，在新 `~/run` 下重建：`tools/`（env.sh、v5env_py313.tar.gz、pywrap，自 run_old 复制，md5 一致）、`sph/vulkan-demo-fsi_688e87a`（复制，diff -rq 无差异）、`sph/run_check_pb.py`、`sph/job_prod_pb.sh`、`sph/production/`、`sph/ckpt/`、`logs/`。旧数据仍在 `~/run_old/sph`，未动。
2. **提交脚本** `cluster_jobs/paratera/submit_prod_OW15.sh`（仓库外，同 submit_prod_OW.sh 的 OW-3 一行，只改 dx 和步数）：

   `job_prod_pb.sh OW15 0.0015 914100 3047 914 10 20 30`，GENFLAGS = 生产流动参数（`--hdx 3 --c0-factor 20 --skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2 --sgs --sgs-filter-width 2.4`）加 Monod 汇（1110 μmol/(g·h)，K_s 9.8e-6，55 g/kg，进料 2e-4 mol/s 从 0 s），无示踪；RUNFLAGS `--torque-every 1000 --torque-split-height 0.1 --torque-shaft-radius 0.007`；lifeline 2 万条每 914 步（0.03 s），探针每 3047 步（0.1 s），快照 10、20、30 s，每小时检查点。
3. **作业 1797424**，gpu_5090，节点 wqd10naf02g2，11-11 03:59 CST 开始，时限 96 h。
   - 生成：流体 8,859,434，壁 805,906，转子 36,517，共 9,701,857，pool 11,157,248；dt 3.282e-5 s；流体体积 29.90 L；薄板 13 块。本地先用同一命令生成过一次，输出一致。
   - 运行：38 个缓冲区 2.79 GB 加 defrag 2.09 GB，nvidia-smi 5.0 GB；27 格扫描（`use_neighbor_list` 为 false，没有超过 4 GB 的单个缓冲区）；bootstrap 存活 9,701,857。
   - 步速：6000 步 682 s，8.8 步/s。914,100 步约 29 h，预计 11-12 09:00 CST（14:00 NZDT）前后结束，约 29 卡时。

## 影响

- **为什么 30 s**：三个已有算例 10 s 之后各 10 s 窗的均值离散在 ±2.5 mN·m 以内（OW-2：77.7 到 82.2），而分辨率之间差 5 到 14 mN·m，所以 20 到 30 s 一个窗（与序列的 t ≥ 20 s 口径一致）足以判断 1.5 mm 是停在 80 附近还是继续升。lifeline 的 30 到 90 s 统计窗这次拿不到。
- 要补到 90 s（S、φ、τ95 的第四个点）：同一提交行加 `RESUME=latest`、步数改 2,742,300，再约 58 h。
- 结束后的分析：扭矩三项（总、下、上）对 dx；平均流速和泵送用 `prod_analysis.sh` 的 passive 和快照口径，与 OW-2 对比（OW-4 0.178，OW-2 0.188 m/s）。Paratera 上分析脚本里的 `~/run/sph/production` 现在指向新目录，旧三个算例在 `~/run_old/sph/production`。

## 待确认

- 扭矩若在 1.5 mm 仍上升，是否再往下（1.2 mm 约 19 M 粒子，显存约 10 GB，放得下；步速约 4 步/s，30 s 约 80 h）。
- 是否把 OW15 续到 90 s。
