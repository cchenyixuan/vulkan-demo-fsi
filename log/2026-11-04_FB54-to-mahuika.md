# FB-L、FB-M 从 ws8 改到 Mahuika

日期：2026-11-04。用户要求："FB-L（FB54_own），FB-M（FB54_mean）这两个换到mahuika上排队"。

## 原来的代码

- 求解器和生成器不变，代码 688e87a。
- 两个算例排在 ws8 跳板机的 queue_C1（GPU 1，接 G24_SL_t20_own2）和 queue_C2（GPU 0，接 OW54）之后，设置见 `log/2026-11-03_ws8-C54-queue.md`，case 由 `cluster_jobs/ws8gpu/gen_C54.sh` 生成。
- 改动时（NZDT 00:32）两个前序算例都还在跑，FB54 未开始，没有输出。

## 改成了什么

ws8：把 `runs\FB54_own\case`、`runs\FB54_mean\case` 改名为 `case_moved_to_mahuika`。队列脚本 `Run-Case` 找不到 `case\case.yaml` 时记 "no case, skipped" 并跳过，所以 G24_SL_t20_own2、OW54 跑完后两个队列直接结束。没有删除文件，没有停进程。

Mahuika：新提交脚本 `cluster_jobs/mahuika/submit_prod_FB54.sh`（仓库外），用 `job_prod.sh`（md5 9fd8466c，与本地相同）：

| 算例 | 作业 | 时限 | 与 ws8 的差别 |
|---|---|---|---|
| FB54_own | 10442544，排队（Priority） | 1-16:00:00 | 无 |
| FB54_mean | 10442545，排队（Priority） | 1-16:00:00 | 无 |

- 生成器参数与 `gen_C54.sh` 相同：`--preset haringa54 --dx 0.030 --free-slip-lid --sgs --shaft-stop-at-lid --smooth-walls --baffles sheets --shaft-to-floor --sparger-ring --sgs-filter-width 2.4 --ninepool --biomass 55 --ninepool-initial <D = 0.028 稳态> --feed-rate 0.3605 --feed-start 0 --ninepool-ki 35.6`，mean 另加 `--ninepool-ki-mean`。
- 运行参数与 queue_C1、C2 相同：2,371,800 步，探针每 4941 步，扭矩每 2000 步，lifeline 2 万条从 t = 0 每 988 步，快照 60、120、180、240 s，每小时检查点；mean 加 `--population-mean-gly-every 0.05`。
- 一处不同：`job_prod.sh` 的快照字段与 lifeline 字段相同（substrate、gly、growth_rate、sto、uptake），ws8 的快照不含 sto。只多存一个字段。
- 提交前在 Mahuika 的 688e87a 源码里确认了生成器和运行器的所有新选项都存在。

## 影响

- 时限按 SL τ2 own 在 PRO 6000 上的 17.1 h 和九池对 SL 的步速比约 0.7 估，约 24 h，取 40 h 余量，超时可用 `RESUME=latest` 续跑。
- Mahuika 的 MaxGRESRunMinsPerUser（20,160 GPU·min）：提交时 OW-2、OW-25 的剩余时限约 6,800 和 3,900 min，加两个 2,400 min 共约 15,500，在限额内。
- 提交时 PRO 6000 有 1 张空闲（g03），两个作业排在优先级更高的作业后面，开始时间未知。OW-25 预计 11-04 08:30 左右结束，会再空出 1 张。
- ws8 两张卡在 G24_SL_t20_own2、OW54 跑完后空闲。
- 原始数据和检查点按 `job_prod.sh` 的规则留在 Mahuika（nobackup 运行目录，结束时复制到 project 并校验 md5）。

## 待确认

无。
