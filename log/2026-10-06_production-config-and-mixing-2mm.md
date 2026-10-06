# 2026-10-06 生产配置定稿，2 mm 混合时间算例 P1（用户：那就定生产配置，跑 2 mm 的混合时间）

## 生产配置

10 月 2 日以来的研究脉络（`log/2026-10-06_paddle-refinement-faces.md`、`2026-10-06_solid-disk-option.md`、
`2026-10-05_paratera-2mm-sheet-baffles.md`）收口：平均流、角动量收支、各部件制动在 2 mm 都已对上 Fluent，
剩下的差别（非轴对称动能 +28%，Rushton 射流倾角 15 到 17° 对 35° 带来的主环流下半段 −18%，PBT 叶尖力矩偏高 4 到 6 mN·m，
薄板叶轮力矩随分辨率不收敛）都已定位、记录，对所有试过的表示和数值旋钮不敏感，不再追。

| 项 | 取值 | 依据 |
| --- | --- | --- |
| 间距 | 2 mm，h = 3 dx，c₀ = 20 U_tip | 10-05：Rushton / 挡板 / 壁面 / 角动量与 Fluent 一致 |
| 槽壁 | 光滑壳（`--smooth-walls`，含台阶缺口填充） | 10-03：单环流型 |
| 挡板 | 双层贴合片（`--conformal-baffles --thin-layers 2`） | 10-04：制动 90% 到 97% |
| 叶轮 | 薄板（`--thin-plates`），圆盘薄板环 | 10-06：实体圆盘无改善 |
| 压力 | 背景压力 2000 Pa，不开重力 | 09-30 |
| 固体 | `--skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse` | 09-29 |
| 位移修正 | PST 0.1 / 0.0005，`shift_transport none` | PST 0.05 和 `shift_transport` 记为备选（后者 09-30 已否，见 `log/2026-09-30_shift-transport.md`） |
| 动量 SGS | 关 | Δ = h 加近壁阻尼记为备选，由混合时间对实验判 |
| 示踪剂 | 10 个，25 到 34 s 每秒一个 1 s 脉冲，标量 SGS C_s 0.1 Δ = dx Sc_t 0.7，位移修正关 | 与 A1 到 A3 相同 |

生成命令（并行科技 `job_mix_pb.sh`，`CODE_DIR` 指向 `vulkan-demo-fsi_f91eedf`）：
`--dx 0.002 --hdx 3 --c0-factor 20 --tracers 10 --injection-start 25 --injection-interval 1 --injection-duration 1 --sgs`
加 `EXTRA_FLAGS="--skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2"`。

## 算例 P1

作业 1668830，gpu_4090（gpu_5090 满到 10 月 8 日；gpu_4090 每卡限 6 核，`-c 6`），`mix_2mm_sheets_P1`，
1,600,000 步 = 70 s（dt 4.376e-5 s），探针每 457 步 = 0.02 s，力矩每 1000 步，末态 dump。
`job_mix_pb.sh` 加了 `CODE_DIR` 环境变量（原来固定 `vulkan-demo-fsi_pb`）。
预计 4090 上约 10 步/s，40 到 45 小时，约 45 卡时，是迄今最大的单个作业。

预期（10-06 估计）：τ95 29 到 33 s，比 3 mm 旧配置的 23.9 s 长，与两套 LES（29.1 / 29.2）同档，高于实验 24.9 ± 2.7。
原因是子午环流比 Fluent 弱 20%（射流倾角），旧配置的 23.9 里有环流偏快的补偿。
2 mm 从静止 20 s 时角动量还在缓慢上升（10-05），25 到 34 s 的脉冲落在尚未完全定常的流场上，分析时看 25 到 70 s 的力矩曲线。

## 待确认

- 混合时间高于实验时，用 PST 0.05 和 SGS Δ = h 加阻尼各做一个 2 mm 敏感性算例（各约 45 卡时），还是只做 3 mm 的？
- 实验比三套 CFD 都快 4 到 8 s，注入方式和探头响应要在报告里单独讨论。

## 改到 Mahuika（用户：P1 改到去 mahuika 排队吧）

并行科技作业 1668830 在 4090 上实测约 13 步/s（15:07 起算，7.5 分钟到 6000 步），160 万步要 34 小时、34 卡时；
用户决定改到 Mahuika 排队（免费）。1668830 已取消（跑了约 20 分钟，约 0.3 卡时）。

Mahuika 当时 GPU 全满：PRO 6000 8 张在用、排队 2 个作业各要 2 张；H100 8 张在用、排 7 个；L4 16 张在用、排 9 个；
A100 22 张在用、无排队（但 A100 上这个算例约 6 步/s，要 75 小时）。
PRO 6000 上 2 mm 加 10 个示踪剂实测 14 步/s（10 月 2 日），160 万步约 32 小时。

代码：本机 `git archive` HEAD（6a89cf2，求解器和生成器与 f91eedf 相同）加本机编译的 spv 打成 `fsi_f91eedf.tar.gz`，
解到 `/nesi/project/uoa04509/sph/vulkan-demo-fsi_f91eedf`（`force_plates.comp.spv` 的 md5 与本机一致）。
作业 9554591，`job_mix.sh`（已有 `CODE_DIR` 支持），`--gres=gpu:pro_6000:1 --mem=96G -t 48:00:00`，参数与并行科技的相同
（`mix_2mm_sheets_P1 0.002 1600000 457 1 0`，`EXTRA_FLAGS` 为生产配置的生成器选项）。输出 `/nesi/nobackup/uoa04509/sph/mixing/`，
日志 `.../sph/logs/mix_mix2mmP1_9554591.out`。
