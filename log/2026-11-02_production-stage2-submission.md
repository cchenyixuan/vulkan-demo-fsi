# 阶段 2 其余算例提交

日期：2026-11-02。用户要求："再提交两个到mahuika吧，然后同一个阶段的小于10h的任务提交paratera"。"同一个阶段"按阶段 2 理解：OW-3、OW-4 已提交，剩下的长算例 SL τ2 L、M 正好两个，不到 10 h 的是 30 L 系列。SL τ0.5 L′ 是规划 A 线新增的噪声重复算例，只要约 3.3 h，一并提交。

## 原来的代码

求解器不变，代码 688e87a。`job_prod.sh`、`job_prod_pb.sh` 的 lifeline 和快照字段写死为 substrate、uptake，state_limited 还要记录 growth_rate。

## 改成了什么

脚本加环境变量 `FIELDS`，默认值不变。提交脚本 `cluster_jobs/mahuika/submit_prod_SL54.sh`、`cluster_jobs/paratera/submit_prod_SL30.sh`（仓库外）。

| 算例 | 机器 | 作业 | 设置 | 模拟 | 时限 |
|---|---|---|---|---|---|
| G24_SL_t2_own | Mahuika，PRO 6000 | 10384030，排队 | 54 m³，30 mm，τ 2 s，局部 | 240 s | 3 d |
| G24_SL_t2_mean | Mahuika，PRO 6000 | 10384031，排队 | 同上，群体平均，每 0.5 s | 240 s | 3 d |
| W24_P1_t3_own | Paratera | 1760334 | 30 L，4 mm，τ 3 s，局部 | 120 s | 7 h |
| W24_P1_t3_mean | Paratera | 1760335 | 同上，群体平均，每 0.05 s | 120 s | 7 h |
| W24_P1_t30_own | Paratera | 1760336 | τ 30 s，局部 | 120 s | 7 h |
| W24_P1_t30_mean | Paratera | 1760337 | τ 30 s，群体平均 | 120 s | 7 h |
| W24_P1_t0p5_own2 | Paratera | 1760338 | τ 0.5 s，局部，重复 | 120 s | 7 h |
| W24_P1_monod | Paratera | 1760339 | Monod 8300，K_s 1e-5，x 10 g/kg | 120 s | 7 h |

设置与阶段 1 的 W24_P1_t0p5、G24_SL_t20 相同，只差 τ；另多记录扭矩。本地用同样的参数生成 τ 3 s mean 和 τ 2 s mean，与阶段 1 的 case.yaml 对比，都只有 reaction 一行不同，54 m³ 的 fluid、wall、rotor 三个 obj md5 相同。

原始数据位置同 `log/2026-11-02_production-OW-submission.md`。

## 影响

- SL τ2 提交在阶段门之前，是用户的决定。若阶段门判 54 m³ 塌缩，这一对仍是 Δ = 2.4 dx 的 τ 点，可保留为结果，不需要取消。
- 提交时 8 张 PRO 6000 全部在用，这两个作业在排队，暂无预计开始时间。
- 阶段 1 的 own 与这里的 τ 3、30 s 用不同代码目录（486162c、688e87a），两者之间求解器和 30 L 生成器无改动。

- 用户随后要求"不要排4090"：六个 Paratera 排队作业用 `scontrol update` 改到 gpu_5090，提交脚本同步改掉。OW-3 原作业 1760075 已在 gpu_4090 上运行，用户要求重排：运行约 15 min 时取消，未到第一个检查点，部分输出已删除，重新提交为 1760361（只排 gpu_5090）。

## 待确认

无。
