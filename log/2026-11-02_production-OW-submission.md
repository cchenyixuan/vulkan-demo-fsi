# 生产算例 OW-2、OW-3、OW-4、OW-25 提交

日期：2026-11-02。用户要求："把 OW-2,OW-25 提交到 mahuika 排队，OW-4 和 OW-3 提交到 paratera，因为这个可能是发表案例了，所以原始数据保留好，先放在服务器上，回头我会拉回来"。这四个算例不依赖阶段门。

## 原来的代码

- 求解器不变。代码 688e87a，含 14 个 compute spv，本地编译，两台机器上 `force_scalar.comp.spv` md5 均为 502dc896。
- 旧分辨率序列 Δ = dx，Monod 1354，进料从 20 s 开始，lifeline 从 30 s 开始。
- 旧作业脚本 `job_knob_pb.sh` 重跑时会 `rm -rf` 同名输出。

## 改成了什么

| 算例 | 机器 | 作业 | 设置 | 模拟 | 步数 | 时限 |
|---|---|---|---|---|---|---|
| OW-2 | Mahuika g01，PRO 6000 | 10383502 | 30 L，2 mm，Monod 1110 / 9.8e-6，10 个示踪脉冲 | 90 s | 2,056,600 | 6 d |
| OW-25 | Mahuika g03，PRO 6000 | 10383503 | 54 m³，25 mm，Monod 1600 / 7.8e-6，8 个示踪脉冲 | 240 s | 2,846,160 | 4 d |
| OW-4 | Paratera gpu_5090, gpu_4090 | 1760074，排队 | 30 L，4 mm，Monod 1110 / 9.8e-6，10 个示踪脉冲 | 90 s | 1,028,340 | 10 h |
| OW-3 | Paratera gpu_5090, gpu_4090 | 1760075，排队 | 30 L，3 mm，Monod 1110 / 9.8e-6，兼作 9P-OW | 90 s | 1,371,133 | 20 h |

共同设置：Δ = 2.4 dx，55 g/kg，lifeline 2 万条，从 t = 0 记录，字段 substrate、uptake，30 L 每 0.03 s，54 m³ 每 0.1 s；每小时检查点；Paratera 排除节点 wqd10nbj04g3。

- 30 L 进料 2e-4 mol/s。探针 OW-2、OW-4 每 0.02 s，OW-3 每 0.1 s。快照 30、60、90 s。
- OW-25 探针每 0.5 s，快照 60、120、180、240 s。

规划之外的三处决定：

1. **进料从 t = 0 开始**，不是旧序列的 20 s。这与 Methods 的写法和其他新算例一致，而且 lifeline 从 t = 0 记录时，细胞不会在进料前先饿约 10 s，OW-3 才能用于九池离线积分。
2. **OW-25 进料 0.362 mol/s**，不是 0.3605。25 mm 流体体积 52.57 m³，30 mm 为 52.36 m³，按 0.37 × V / 53.73 保持单位质量进料，分辨率对照才公平。
3. **OW-3 用普通 Monod**，不用 `--ninepool-one-way`。C_s 只取决于汇，九池沿 lifeline 离线积分，与生成器的 one-way 模式等价。

短测（本地 RTX 4070 Ti SUPER，4 mm，3,430 步，脉冲时间压缩到 0.02 到 0.2 s）：存活 577,837 不变，四个溢出计数 0，共 14 个场（上限 16），10 个示踪脉冲全部释放，释放后总量守恒到 9 位有效数字。

原始数据：

| 机器 | 目录 | 说明 |
|---|---|---|
| Mahuika | `/nesi/nobackup/uoa04509/sph/production/` | 运行目录，含检查点 |
| Mahuika | `/nesi/project/uoa04509/sph/production/` | 作业结束时除检查点外全部复制，并按 md5 校验 |
| Paratera | `~/run/sph/production/` | 全部输出 |
| Paratera | `/data/run01/scxm138/sph/ckpt/NAME_ckpt` | 检查点 |

新脚本 `cluster_jobs/mahuika/job_prod.sh`、`submit_prod_OW.sh`，`cluster_jobs/paratera/job_prod_pb.sh`、`submit_prod_OW.sh`（在仓库外）。脚本不删除任何输出，同名输出已存在时拒绝启动，结束时写 `NAME_manifest.txt`（提交、作业号、生成参数、全部文件 md5）。

## 影响

- 实测步速（Mahuika，启动后 37.5 min 的平均，含生成算例）：OW-2 11.5 步/s，OW-25 20.3 步/s。最初按 240 s 窗估的 OW-2 8 步/s 偏低，1000 步的日志粒度太粗。
- OW-2 预计约 50 h，比规划的 44 h 多约 14 %。OW-25 预计约 39 h，与规划的 41 h 一致。

## 待确认

- Paratera 的 `~/run` 所在 JuiceFS 300 GB 已用 265 GB，还有 36 GB。OW-3、OW-4 约需 8 GB，够用，后续往这里放数据要先看余量。
- Paratera 两个作业提交时 gpu_5090 无空闲卡，在排队。
