# 2026-09-25 邻居循环两项实验：平方距离比较、每步建邻居表

一次提交，两项实验。第一项保留并默认启用，第二项结果为负，代码保留为可选开关、默认关闭。
测试机：RTX 4070 Ti SUPER，3 mm 搅拌槽（1,241,257 个活粒子，h/dx = 3，defrag 每 10 步），
分 kernel 计时脚本对每个 kernel 单独提交 30 次取平均，"静止"指 t = 0，"300 步"指转子启动后 0.04 s。

## 起点：静止时每步 12.6–12.9 ms（两次测量），三个邻居 kernel 占 97%

| kernel | ms | 占比 |
| --- | --- | --- |
| predict | 0.28 | 2% |
| update_voxel | 0.18 | 1% |
| correction | 3.8 | 30% |
| density | 3.9 | 31% |
| force | 4.6 | 36% |

每个粒子扫 27 个体素约 730 个候选，其中约 113 个在核半径内。

## 实验 1：先比较距离平方，只对通过的候选开方

### 改动

`correction.comp` / `density.comp` / `force.comp` 循环内：

```
旧: float distance = length(x_ij);  if (distance >= h || distance < 1e-12) continue;
新: float d2 = dot(x_ij, x_ij);      if (d2 >= h*h || d2 < 1e-24) continue;  float distance = sqrt(d2);
```

`helpers.glsl` 新增 `smoothing_length_squared()`（glslang 不允许用特化常量初始化全局 const，所以写成函数，驱动在建管线时折叠），
以及 `evaluate_kernel_unguarded()` / `evaluate_kernel_gradient_unguarded()`：去掉核函数内部的 q ≥ 1 和 r < 1e-12 保护，
只在循环内调用（循环已保证 1e-24 ≤ r² < h²）。`force.comp` 循环外的 `evaluate_kernel(delta_x)` 仍用带保护版本。

### 过程中的一个反直觉结果

只改比较、不改核函数时，反而慢了 6%（13.5–13.7 ms，两个变体各测两次，基线用 git stash 回退重测两次确认 12.7–12.9 ms）。
原因：原来循环判断和核函数内部保护判断的是同一个 `distance`，编译器能识别重复并删掉核函数里的判断；
改成平方比较后两处判断的量不同，编译器识别不了，113 个命中路径上每次多出两组比较和分支。
换成无保护核函数后恢复并略优于基线。

### 结果

| 版本 | 静止 ms/步 | 有 defrag 200 步 steps/s | 无 defrag 100 步 steps/s |
| --- | --- | --- | --- |
| 基线 | 12.7–12.9 | 76.6 / 76.7 | 34.8 / 35.0 |
| 平方比较 + 带保护核函数 | 13.5–13.7 | 72.4 | — |
| 平方比较 + 无保护核函数（采用） | 12.4–12.6 | 78.6（300 步） | 41.0 |

静止时快 2–3%，粒子未排序（无 defrag）时快 17%。

### 验证

无 defrag 100 步后逐粒子对比（无 defrag 时粒子编号稳定；开 defrag 后原子操作决定顺序，编号会变，不能逐粒子比）：

| 对比 | max dx (m) | max dv (m/s) |
| --- | --- | --- |
| 基线 vs 基线（运行间噪声） | 5.4e-7 | 2.2e-6 |
| 基线 vs 新版 | 7.2e-7 | 3.2e-6 |

同一量级，无系统偏差；活粒子数、动能一致，材料一致，溢出计数全 0。

## 实验 2：每步建一次邻居表（结果为负，默认关闭）

### 改动

- 新 kernel `build_neighbor_list.comp`：update_voxel 之后（启动时 initialize_voxelization 之后）每粒子扫一次 27 个体素，
  把通过距离测试的邻居编号按扫描顺序写入 `neighbor_list`，个数写入 `neighbor_count`（set 1 绑定 5、6）。
  表是转置布局 `list[k * stride + pid]`，同一轮循环里一个 warp 的连续粒子读连续地址。
- `correction` / `density` / `force` 改成统一的两层循环，由特化常量 `USE_NEIGHBOR_LIST`（id 47）选候选来源：
  表模式一个"格"就是本粒子的表，不再做距离测试；体素模式 27 个格按原来 z 最慢、x 最快的顺序扫，做距离测试。
  两种模式访问同一批粒子对、同一顺序，结果只差舍入。
- 容量 `MAX_NEIGHBORS`（id 62）= `capacities.max_neighbors`，默认 0 表示取支撑球内最密堆积上界
  （3D：√2·4/3·π·(h/dx)³，h/dx = 3 时 160；2D 腔体 91），与 `max_per_voxel` 的校验逻辑相同。
  溢出计入 `GlobalStatusBuffer.overflow_neighbor_count`（缓冲区从 64 B 扩到 80 B），启动时溢出报错，结束时打印。
- 开关 `numerics.use_neighbor_list`，默认 false；关闭时表缓冲区只分配 1 项/粒子，kernel 不读。

### 结果（静止，3 mm 槽）

| kernel | 体素扫描 ms | 邻居表 ms |
| --- | --- | --- |
| build_neighbor_list | — | 8.2 |
| correction | 3.5 | 4.1 |
| density | 3.9 | 4.4 |
| force | 4.7 | 3.0 |
| 合计 | 12.5 | 20.2 |

有 defrag 200 步：49.9 steps/s（体素扫描 76–79）。2D 腔体 1M：129 steps/s（体素扫描 247–256）。
无 defrag（粒子未排序）100 步：41.6 对 42.6，持平。

### 为什么慢

体素扫描里一个 warp 的 32 个粒子在同一体素，同一轮循环读的候选 j 是同一个，硬件一次取回广播给 32 个线程；
730 个候选里被丢掉的 617 个每个只花一次广播读和十几条指令，很便宜。
邻居表里每个粒子有自己的表，同一轮循环 32 个线程取的是 32 个不同邻居，位置、速度、密度每项都是 32 次分散访问，
L1 事务数约为原来的 9 倍，抵消了迭代次数减少 6.5 倍的好处；只有循环体最重的 force 略有收益。
建表 kernel 本身 8 ms 是因为写入分散：每个线程按自己的计数写各自的行，4 字节写散落在不同缓存行。
即使把写入合并到理想带宽（约 1 ms），合计仍约 16 ms，不如 12.5 ms。

结论：粒子按体素排好序时，体素扫描的广播读已经是高效的访存模式，每粒子邻居表在这块卡上不划算。
真正的空间在"命中"路径的分散取数（把 27 个体素的数据先搬进共享内存），不在候选筛选。

### 验证

无 defrag 100 步逐粒子对比：基线 vs 表模式 max dx 6.9e-7、dv 2.8e-6；基线 vs 新二进制的体素模式 7.3e-7、3.7e-6；
表模式 vs 体素模式 5.1e-7、4.0e-6，均在运行间噪声（5.4e-7、2.2e-6）量级。
2D 腔体两种模式 300 步动能一致到 7 位（6.218229），活粒子数一致，`overflow_neighbor` 全 0。

## 对其他部分的影响

- `readback_global_status()` 多两个字段；`_run_v1_headless.py` 打印 `overflow_neighbor`。
- 旧 case.yaml 不用改：`max_neighbors` 与 `use_neighbor_list` 都有默认值。
- 特化常量 47、62 已占用，`CLAUDE.md` 的空闲区间已更新。
- 缓冲区数量 25 → 27（表关闭时多占 2 × 4 × pool 字节，3 mm 槽约 11 MB）。
