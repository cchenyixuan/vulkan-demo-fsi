# 2026-10-07 九池细胞模型进求解器（`scalars.reactions` 类型 `ninepool`，REACTION_MODE 3，默认无）

分支 `test`。接 `log/2026-10-07_scalar-vec4-4-and-diffusing-mask.md`（第 1 步）。设计见
`docs/stage5_pichia_level2_design_2026-10-02.md` 第 8 节；用户 10 月 7 日批准菌种改青霉菌、大罐参照改 54 m³、副产物为胞内贮藏物，
并要求一直做到第 5 步（30 L 的 D1 提交 Mahuika）。

**结论先说：**

1. Tang 等 2017 的九池模型按 Haringa 2018 补充材料 A 的形式（含 ATP 代数补丁）放进了 `predict.comp`；参数走新的缓冲区，不用 spec 常量。
2. 零维参照模型有一个 bug（化学计量矩阵的 ATP 行和 PAA 行调换），这次顺带修了；修后 X_PAA 恒化器值 2.71（原文约 3），其它量不变。
3. 验证 N1 到 N3 见下。
4. 生成器 `--ninepool`（双向）和 `--ninepool-one-way`（单向，Monod 汇）；示踪剂算例输出逐字节不变。
5. D1（2 mm 生产配置，双向、单向各一个，90 s，2 万条 lifeline）提交 Mahuika，见文末。

## 合作者原来的代码

`predict.comp` 的标量更新按 vec4 循环：输运增量、连续源、反应（级别一 Monod 在一个 vec4 内）、Kahan 累加、脉冲。
反应参数只有 spec 常量；缓冲区 set 3 用到 binding 13。

## 改成了什么

### 数据

每粒子 15 个场，4 个 vec4（生成器 `NINEPOOL_FIELDS` 的顺序）：

| vec4 | 场 | 说明 |
| --- | --- | --- |
| 0 | substrate（C_s）、paa_ext（C_PAA）、uptake（累计摄取）、feed（累计加料） | 前两个扩散并带标量 SGS，后两个为记录 |
| 1 | gly、aa、sto、paa_pool | 代谢物池，μmol/gdw，不扩散 |
| 2 | e11、e32、e4、pen_capacity（v33） | 酶池和青霉素合成容量，不扩散 |
| 3 | biomass（x_bio，gdw/kg）、product（累计青霉素，mol/kg）、growth_rate（当步 μ，覆盖写） | 不扩散 |

单向算例把 biomass 和 paa_ext 对调（Monod 汇要求底物、生物量、摄取在同一个 vec4）。

新缓冲区 `ReactionParameterBuffer`（set 3 binding 14，320 B）：`reaction_field_slot[16]`（角色到场序号，缺省 0xFFFFFFFF）加
`reaction_parameter[64]`（`case.py` 的 `NINEPOOL_PARAMETER_ORDER` 顺序，45 个在用），启动时上传一次。

### 每步（`predict.comp`，REACTION_MODE 3）

在底物所在 vec4 的那一轮：读齐全部 vec4 的值和增量，算：

    ATP   = A X_gly³ / (X_gly³ + B³)                                   代数补丁，A 8.5，B 10.5
    a11   = k_E11 X_E11 (x_bio / M_w) dt_h / (C_s + K_s)，ΔC_s = C_s a11 / (1 + a11)   线性化隐式，ΔC_s ≤ C_s
    v11   = ΔC_s / ((x_bio / M_w) dt_h)                                 与 ΔC_s 一致的摄取速率
    v12 … v42 按表 A2（Hill 项用乘法展开，不用 pow 的负底）
    池：ΔX_i = 10⁶/M_w (S_i · v) dt_h − μ X_i dt_h                      表 A1 化学计量，dt_h = dt / 3600
    酶池：E11 的 Hill 合成，E32、E4 的 α + βμ 合成，v33 受 X_gly 抑制，各减降解与稀释
    x_bio += (μ − v_d) x_bio dt_h；C_PAA += (v32 − v31)(x_bio / M_w) dt_h（≥ −C_PAA）；U += ΔC_s；P += v33 (x_bio / M_w) dt_h

其它 vec4 的增量写回 `scalar_delta`，由后面几轮照常做 Kahan 累加；`case.py` 校验所有角色的 vec4 序号不小于底物的。
所有量读入时 max(·, 0)；每个表达式对任意输入有限。前向 Euler，与 Haringa 2018 相同。

| 文件 | 改动 |
| --- | --- |
| `experiment/v1/shaders/common.glsl` | `ReactionParameterBuffer`；REACTION_MODE 注释 |
| `experiment/v1/shaders/predict.comp` | REACTION_MODE 3 分支 |
| `utils/sph/case.py` | `NINEPOOL_PARAMETER_ORDER`、`NINEPOOL_DEFAULTS`（含三处订正）、`NINEPOOL_ROLES`；`ScalarReactionConfig` 的角色键和 `ninepool:` 覆盖字典；校验；`reaction_slots()`；`mode` 3；Monod 的"同一 vec4"校验不再用于 ninepool |
| `experiment/v1/utils/simulator_v1.py` | 缓冲区规格与初始上传 |
| `utils/geometry/_demo_stirred_tank_30l.py` | `--ninepool`、`--ninepool-one-way`、`--paa-initial`；初值常量 `NINEPOOL_INITIAL`（零维 μ ≈ 0.035 的稳态）；场数上限 16 |
| `experiment/v1/checks/_model_ninepool.py` | 修 ATP / PAA 行调换；scipy 改为延迟导入（求解器环境没有 scipy） |
| `experiment/v1/checks/_check_ninepool_box.py` | 新：N1 到 N3 |

## 验证（本机，4 mm 静止方箱 16³，c₀ 2 m/s 使 dt 9 × 10⁻⁴ s）

| 试验 | 设置 | 结果 |
| --- | --- | --- |
| N1 葡萄糖脉冲 | 池取零维 D 0.05 稳态，C_s 350 μmol/kg，360 s（40 万步），对零维模型的 RK4（自写，步长 0.01 s） | 相对差：C_s 1.9 × 10⁻⁵，X_gly 6.1 × 10⁻⁵，X_sto 9 × 10⁻⁷，X_PAA 1.9 × 10⁻⁶，q_p 2 × 10⁻⁷，x_bio 9 × 10⁻⁸，C_PAA 4 × 10⁻⁸，μ 1.0 × 10⁻⁴；粒子之间差 0 |
| N2 收支 | N1 的序列 | Σm(C_s + U) 漂移 3.8 × 10⁻¹³；Σm(C_PAA + x_bio X_PAA 10⁻⁶ + P) 漂移 3.0 × 10⁻⁶ |
| N3 加料球 | 球半径 15 mm，2 × 10⁻⁸ mol/s，0.5 s 起，60 s | Σm(C_s + U − F) 漂移 4.6 × 10⁻¹¹；加入量对预期差 8.4 × 10⁻⁶；C_s、各池非负 |

N1 的曲线就是一个饱饥循环：GLY 从 24.6 升到 37（120 s），糖耗尽后 40 s 内掉到 8、再慢慢到 1.7；μ 从 0.055 升到 0.091 再回 0；
贮藏物在饱段积累 60 μmol/gdw。与零维一致到 10⁻⁴，是前向 Euler 对 10 s 时间尺度的一阶误差量级。

## 影响

- 没有 `scalars:` 块：不变。级别一、级别二、示踪剂：代码路径不变（REACTION_MODE 1、2 的分支一字未动），生成器示踪剂算例逐字节相同。
- 新缓冲区对所有算例都分配（320 B），没有反应时不读。

## D1 提交（Mahuika，`cluster_jobs/mahuika/submit_D1.sh`）

代码 `fsi_138281c.tar.gz`（`git archive` 加本机编译的 spv，`predict.comp.spv` md5 与本机一致）解到
`/nesi/project/uoa04509/sph/vulkan-demo-fsi_138281c`。两个作业，PRO 6000，各 `--mem=96G -t 60:00:00`，`job_lifeline.sh`：

| 作业 | 名称 | 生成器选项（生产配置之外） | 内容 |
| --- | --- | --- | --- |
| 9569343 | D1two | `--ninepool --sgs --feed-start 20` | 双向：九池在粒子上，摄取从粒子自己的 C_s 扣 |
| 9569344 | D1one | `--ninepool --ninepool-one-way --q-max-umol-per-g-h 1354 --k-s 9.8e-6 --sgs --feed-start 20` | 单向：Monod 汇（容量 = 双向稳态 k_E11 X_E11），池沿 lifeline 离线积分 |

共同：2 mm，90 s（2,056,600 步），加料 2 × 10⁻⁴ mol/s 从 20 s，2 万条 lifeline 从 30 s，记录 substrate、gly、growth_rate、pen_capacity、sto，
探针每 457 步，`REGIME_MONOD="3.761e-7 9.8e-6"`（分区按 C_s 的 Monod 比）。预计每个约 40 小时；队列前面有 P1（预计 10 月 10 日开始）。
输出 `/nesi/nobackup/uoa04509/sph/lifeline/D1_2mm_ninepool_{two,one}_way_*`，日志 `sph/logs/life_D1two_9569343.out`、`life_D1one_9569344.out`。

跑完后要做的分析：两套 lifeline 的 X_gly、μ、q_p 分布和区间停留时间；单向的池用 `_model_ninepool.py` 沿记录的 C_s 积分（脚本待写）；
全罐 q_p 相对理想混合的损失。

## 待确认

- 单向协议里 Monod 汇的容量取双向稳态的 k_E11 X_E11（1354 μmol/gdw/h）而不是 Haringa 的固定 1600，为的是两种协议的群体平均摄取一致。
- 饥饿段 ATP 代数补丁的低估（Haringa 自己写明）原样保留。
- 2 mm 从静止 20 s 时流场还在缓慢变化（10-05），加料 20 s 起、lifeline 30 s 起，和 C1 一样。
