# 阶段 1 提交：Δ = 2.4 dx 试点（跳板机 ws8）

日期：2026-11-02。用户决定启动阶段 1（"启动阶段一吧"），此时 G10 还在 Mahuika 上跑（119 s / 240 s），所以"Δ 是主因"这个前提尚未验证，这四个算例是先行投入。

## 原来的代码

求解器不变。现有 state_limited 算例都用 Δ = dx：54 m³ 的 P54s（无圆环，轴到 0.785 m），30 L 的 P1_t0p5。跳板机上没有部署过任何东西，`run_sph.py` 只会选第一张名字匹配的卡，而这台机器两张卡同名。

## 改成了什么

不改求解器。四个算例，串行，只用 GPU 1：

| 顺序 | 算例 | 与基线的差别 | 时长 / 统计窗 | 预计 |
|---|---|---|---|---|
| 1 | W24_P1_t0p5_own | P1_t0p5_own 加 `filter_width: 9.6e-3`（2.4 × 4 mm） | 120 s / 40 到 120 s | 约 3.5 h |
| 2 | W24_P1_t0p5_mean | P1_t0p5_mean 同上，μ̄ 每 0.05 s | 同上 | 约 3.5 h |
| 3 | G24_SL_t20_own | P54s_own 换 G24 几何（圆环、轴到底）加 Δ = 2.4 dx，进料仍 0.44 mol/s | 240 s / 120 到 240 s | 约 19.5 h |
| 4 | G24_SL_t20_mean | P54s_mean 同上，μ̄ 每 0.5 s | 同上 | 约 19.5 h |

30 L 排在前面，因为它快，也是最可能垮的一端（判断见 `待确认`）。

生成命令在本地执行，算例整目录上传，因为圆环和轴到底的选项还在工作区、未提交（`_demo_rushton_tank.py` md5 b0bb3f44，`_demo_stirred_tank_30l.py` md5 d0de8eb7）：

```
python utils/geometry/_demo_rushton_tank.py --preset haringa54 --dx 0.030 --free-slip-lid --sgs --shaft-stop-at-lid --smooth-walls --baffles sheets --shaft-to-floor --sparger-ring --sgs-filter-width 2.4 --substrate --state-limited --biomass 10 --q-max-umol-per-g-h 8300 --k-s 1e-5 --growth-yield 90 --maintenance-umol-per-g-h 222 --demand-margin 0.1 --tau-up 20 --tau-down 20 --growth-rate-initial 6.94e-5 --feed-rate 0.44 --feed-start 0 --state-coupling own|mean
python utils/geometry/_demo_stirred_tank_30l.py --dx 0.004 --hdx 3 --c0-factor 20 --skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2 --sgs --sgs-filter-width 2.4 --substrate --state-limited --biomass 10 --q-max-umol-per-g-h 8300 --k-s 1e-5 --growth-yield 90 --maintenance-umol-per-g-h 222 --demand-margin 0.1 --growth-rate-initial 6.94e-5 --feed-rate 2.5e-4 --feed-start 0 --tau-up 0.5 --tau-down 0.5 --state-coupling own|mean --no-preview
```

生成后的核对：

- 54 m³：fluid / wall / rotor.obj 与 `cases/haringa54_30mm_G24` md5 相同。
- 30 L：case.yaml 与 Paratera 上 `P1_t0p5_own_case` 相比只多 `filter_width` 一行（`material_library` 路径除外）；三个 obj 去掉 CR 后 md5 相同。
- 每对 own 和 mean 的 case.yaml 只差 `state_coupling: mean`。

部署在 `C:\Users\WS8\Documents\sph\`：代码 `lifeline\vulkan-demo-fsi_486162c`（8cecf88 到 486162c 求解器无改动，与 G24、G10、P54s 同一求解器，14 个 spv 本地编译），`lifeline\venv`，`lifeline\run_sph_jump.py`，队列 `queue_S1.ps1`，输出 `runs\<算例名>\`，日志 `runs\queue_S1.log`。脚本源文件在仓库外的 `cluster_jobs/ws8gpu/`。重跑队列会跳过已完成的算例并从每小时检查点续算。

短测（6000 步，30 L）：存活 577,837 不变，四个溢出计数为 0，130.8 步/s。

## 影响

- 不改仓库代码，现有结果不受影响。
- 这台机器 Vulkan 序号与 nvidia-smi 相反：`SPH_DEVICE_MATCH=0` 是 nvidia-smi GPU 1，`=1` 是 GPU 0（接显示器）。短测第一次按 `=1` 跑，在 GPU 0 上满载约 40 s，发现后改为 0；启动后实测 GPU 1 98 %，GPU 0 0 %。
- 控制台有用户 cchen 登录且活跃，所以只用 GPU 1，串行总耗时约 46 h。

## 待确认

- 阶段门判据：54 m³ 的 own 减 mean 的饥饿份额明显高于噪声（重复 0.3 到 0.5 个百分点，P54s 为 6.5）且 30 L 趋势保持则继续阶段 2；30 L 饥饿份额接近 0 则停，重新设计 30 L 工况；54 m³ 塌缩则停，讨论主线。
- G10 出结果后才能判断 Δ 是不是主因；如果不是，这四个算例作为敏感性结果保留。
- 工作区里的生成器选项需要提交，否则这四个算例在仓库 HEAD 上不可复现；该改动来自另一个会话，本次未提交。
