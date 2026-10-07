# 2026-10-08 断点续算（checkpoint / resume）

## 动机

54 m³ 的算例单个要 40 到 70 h，集群单作业有墙钟上限，而求解器此前只能从头跑。需要能在任意时刻把完整状态存盘、下次从那里接着算，并且 lifeline、探针日志、场快照在接续处不重、不漏。

## 原来的代码

- `SphSimulatorV1` 的状态全部在设备缓冲区里（`self.buffers`，37 个），主机侧只有 `simulation_time`、`step_count`、`rotor_angle` 三个数。启动只有一条路：构造函数上传初始粒子，`bootstrap()` 做体素化、第一次核函数求和、反向半步踢（`v_{-1/2} = v_0 − ½ a_0 dt`）和一次 defrag。
- `_run_v1_headless.py` 的采样循环（扭矩、探针、快照、lifeline、流动统计）只能从第 0 步开始；探针和扭矩日志以追加方式打开。
- `LifelineRecorder` 的样本（uid 列表）在第一次记录时随机抽取，之后按 10 s 一块写 `lifeline_NNNN.npz`。

## 改成了什么

### simulator_v1.py：`write_checkpoint` / `restore_checkpoint`

检查点 = 每个设备缓冲区的原始字节 + 三个主机数 + 调用方的附加 JSON。

```python
def checkpoint_arrays(self):           # {'buffer:<name>': uint8 数组}，host_visible 的从映射内存读，其余 staging 回读
def write_checkpoint(self, path, extra) # np.savez 到 path.tmp.npz 再 os.replace，中断的写不会毁掉上一个好文件
def restore_checkpoint(self, path)     # 校验格式、dt、每个缓冲区的字节数；vkQueueWaitIdle 后逐个上传；恢复三个主机数；返回 extra
```

为什么存全部缓冲区而不是只存粒子：蛙跳格式的状态是 `(x_n, v_{n-1/2}, a_n, ρ_n)`，再加标量及其 Kahan 补偿、uid、体素列表、alive 计数、薄板反力等。全部原样恢复后，下一步 `step()` 读到的和中断前完全一样，不需要再做体素化，更不能再做半步踢（速度已经是半步的）。所以续算时**不调用** `bootstrap()`，构造函数照常上传初始状态，然后 `restore_checkpoint` 覆盖掉。defrag 的暂存集（set 4）和注入 staging 每步重写，不存。

文件大小 = 缓冲区总字节数：6 mm 测试罐 84 MB（0.4 s），2 mm 30 L 罐约 1.7 GB，54 m³ 25 mm 约 2 GB。

### `_run_v1_headless.py`

| 选项 | 作用 |
|---|---|
| `--checkpoint-dir DIR` | 写到 `DIR/checkpoint_latest.npz`，上一个改名为 `checkpoint_previous.npz`；运行结束时总写一次 |
| `--checkpoint-wall-minutes M` | 每 M 分钟墙钟时间写一次（用于墙钟上限） |
| `--checkpoint-every T` | 每 T s 流动时间写一次 |
| `--resume PATH` 或 `--resume latest` | 从检查点接着算；`--max-steps` 是总步数；其余采样选项与中断前相同 |

续算时的一致性处理：

- lifeline：检查点写入前先 `flush()`，使磁盘上的块恰好到检查点；`extra['lifeline']` 保存 uid 列表与计数器；恢复时删掉编号 ≥ 保存的 `chunk_index` 的块（崩溃前多写的）。
- 探针、扭矩 CSV：恢复时删掉时间列 > 检查点时间的行（`truncate_csv_after`）。
- 场快照：时间 ≤ 检查点的从待办列表里去掉。
- 检查点里存了 case.yaml 的 sha1，换了算例文件会拒绝。
- `--flow-statistics` 的累加器没有做保存，与 `--resume` 同用时直接报错。
- 步速统计改为按本次运行的步数算。

### lifeline_recorder.py

新增 `checkpoint_state()`（先 flush，返回 uid 与计数器）和 `restore(state)`（恢复并清理多余块）。

## 验证（6 mm 两向罐，187,440 粒子，本机 4070 Ti SUPER）

A 跑 400 步并写检查点，B 从 A 的检查点续到 800 步；C、D 各自不中断跑 800 步作为运行间噪声参照。选项都带探针（每 50 步）、500 条 lifeline（每 20 步，0.05 s 一块）、两个快照时刻。

| 量 | 续算 B 对 C | 噪声 D 对 C |
|---|---|---|
| 终态位置最大差 | 4.0e-2 m | 5.7e-2 m |
| 终态位置平均差 | 3.7e-4 m | 4.0e-4 m |
| 终态标量最大差（C_s） | 7.9e-6 | 3.6e-5 |
| 探针 CSV | 16 行，步号严格递增，无重复 | 同 |
| lifeline | 31 条记录，步号 200..800 唯一递增，4 块 | 同 |
| 快照 | 305、571 两个，与 C 相同步号 | 同 |

崩溃路径：从倒数第二个检查点（第 381 步）恢复一个已经跑到 400 步的目录，正确删掉了检查点之后写出的 1 个 lifeline 块和 1 行探针，跳过已存快照，续到 800 步后探针 16 行、lifeline 31 条均唯一连续。

续算与不中断的差别在求解器本身的运行间非确定性（体素 incoming 列表的原子追加顺序）以内，没有系统性偏差。

## 影响

- 默认行为不变：不给 `--checkpoint-dir` 和 `--resume` 时代码路径与之前相同。
- 集群作业脚本可以用 `--checkpoint-wall-minutes 60 --checkpoint-dir ...`，被墙钟杀掉后用 `--resume latest` 重新提交即可接上，最多损失一小时。
- 检查点里包含完整缓冲区，也可以用来把已完成的算例延长（例如 320 s 续到 420 s）。

## 待确认

- 随机样本的 uid 依赖于槽位顺序，同一种子两次独立运行抽到的粒子不同（原有行为，与续算无关）；续算保持的是中断前抽到的那组。
- `--flow-statistics` 的续算未实现（H 系列流场统计算例若需要再加）。
- 2 GB 级检查点在 Mahuika 的 nobackup 上每小时写一次是否合适，待第一次 54 m³ 作业观察。
