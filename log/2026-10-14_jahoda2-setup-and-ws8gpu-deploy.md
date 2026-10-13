# 2026-10-14 双 Rushton 槽（Haringa、Vandewijer、Mudde 2018）的算例准备与 ws8-gpu 部署

分支 `test`。目的：复现 Haringa 等 2018 CERD 136:870 的双 Rushton 槽（Jahoda 2007 几何，唯一在两桨之间有 LDA 测量的几何），
用来检验 30 L 罐里"两桨之间和上部脉动偏高"（`log/2026-10-13_energy-by-region-2mm-vs-fluent.md`）是湍流、宏观不稳定还是噪声。

**结论先说：**

1. 生成器加了预设 `jahoda2`（T = 0.29 m，H = 2T，两个 Rushton D = T/3，下桨 C = T/3，桨间距 T，轴 20 mm，N = 5 s⁻¹，顶面无剪切），
   `--flow-statistics` 对每个叶轮各写一组排出流剖面线和叶尖圆柱，另写论文的 LDA 测点（两桨盘面 r = 50 到 102 mm 六点、
   中面 7 × 11 格点、中面全截面极坐标网格算 Q_ax）；新增 `--point-series`（77 × 4 个中面点的速度时间序列，间隔等于统计采样间隔 0.02 s）、
   `--probe NAME X Y Z`（覆盖预设探针）。单桨预设 `jahoda` 的全部输出逐字节不变（4 mm 算例 8 个文件 cmp 相同）。
2. 分析脚本 `_analyze_flow_statistics.py` 支持多叶轮（每个叶轮的剖面、泵送数；力矩按 `--torque-split-height` 的分列给每桨 Po）、
   LDA 出流点表、中面 7 × 11 表（U_r、U_y、脉动、k）和 Q_ax；新建 `_analyze_point_series.py`（自相关的阻尼余弦拟合，
   van Maanen 1999，论文式 (3)；噪声、周期、湍流三部分；k_t 含与不含宏观不稳定；主频 f/N；谱积分法作对照；合成信号自检通过）。
3. 本机 3 mm 冒烟（300 步含统计和点序列；60 步含 5 个示踪剂和探针）：粒子数守恒，四个溢出计数为 0。
4. 部署到合作者的工作站 ws8-gpu（RTX 5090）：见第四节；队列脚本依次跑 J2-S3、J2-M3、J2-S2，见第五节。

## 一、几何与测量（论文 3.1、3.2 节和图 1）

| 项目 | 论文 CFD 几何（我们照此） | 论文实验槽 |
| --- | --- | --- |
| 槽径 T | 0.29 m | 0.26 m |
| 液高 H | 2T = 0.58 m | 2T |
| 挡板 | 4 块全高，宽 T/10 = 29 mm，位于 θ = 0、90、180、270° | 同 |
| 叶轮 | 两个 6 叶 Rushton，D = T/3 = 96.67 mm，标准比例（叶长 D/4，叶高 D/5，盘径 3D/4） | 同 |
| 下桨离底 C | T/3 = 96.67 mm（3 mm 格子对齐到 97.5 mm，2 mm 到 97 mm） | 同 |
| 桨间距 ΔC | T，上桨 4T/3 = 386.67 mm（3 mm 对齐到 385.5 mm，2 mm 到 387 mm） | 同 |
| 轴径 | 20 mm | 20 mm |
| 转速 N | 5 s⁻¹（300 rpm），U_tip = πDN = 1.5184 m/s，Re = ND²/ν = 4.67 × 10⁴ | 5.78 s⁻¹，Re = 4.34 × 10⁴ |
| 顶面 | 无剪切（`--free-slip-lid`） | 自由面 |
| 两桨叶片相对角 | 论文未给，两桨都取 15°（一片叶在 θ = 15°） | |

测量点（都在挡板面内，我们在四个挡板面各取一套，分析时取平均）：

| 测量 | 位置 | 我们的采样集 |
| --- | --- | --- |
| 叶轮出流（图 2） | 两桨盘面高度，r = 50、60、70、77、90、102 mm | `lda_outflow_<lower/upper>_<θ>`，6 点，带叶片相位分档 |
| 中面格点（图 1、图 3） | y_c = 5T/6 = 241.67 mm，r/R = 0.200、0.358、0.435、0.512、0.566、0.694、0.765，y/T = 0、±0.019、±0.038、±0.077、±0.154、±0.231 | `midplane_<θ>`，7 × 11 = 77 点，均值与二阶矩 |
| 轴向交换流量 Q_ax | 中面上向下流量的面积分 | `midplane_disk`，极坐标网格（环距 2 dx，环上约 2 dx），Q_ax = −Σ min(⟨u_y⟩, 0) dA |
| 点时间序列 | 中面 77 点 × 4 个挡板面 | `--point-series`，Shepard 核平均（半径 2 dx）的三分量，每 0.02 s 一个样本 |
| 混合时间（Jahoda 2007） | 示踪剂球半径 12.5 mm，高 0.551 m（1.9T），r = T/4 = 72.5 mm，挡板面 θ = 0；探针在挡板之间 θ = 45°，离壁 T/20（r = 130.5 mm），底部 y = T/4 = 72.5 mm，顶部 y = 1.25T = 362.5 mm | 预设的 `injection` 和 `probes`（`probe_bottom`、`probe_top`，Shepard 半径 h） |

另保留 H1 的集：每桨四条排出流剖面线（挡板之间，ξ = (r − r_tip)/(R − r_tip) 0 到 1 共 41 点，带相位和速度梯度），每桨叶尖圆柱（泵送数），两张 r-y 平面（流型）。

## 二、代码

### 原来的代码

- `utils/geometry/_demo_rushton_tank.py`：预设 `jahoda`（单桨）和 `haringa54`；`--flow-statistics` 只允许单桨（`parser.error`），采样集写死在 `flow_statistics_sets()`；探针位置写死在预设里。
- `experiment/v1/utils/flow_statistics.py`：按点累积一、二阶矩，不存时间序列。
- `experiment/v1/checks/_analyze_flow_statistics.py`：只读第一个叶轮，集名 `discharge_<θ>`、`tip_cylinder`、`plane_<θ>`。
- `experiment/v1/checks/_analyze_mixing_time.py`：探针名写死 `probe_1`、`probe_2`（30 L 罐）。

### 改成了什么

| 文件 | 改动 |
| --- | --- |
| `utils/geometry/_demo_rushton_tank.py` | 预设 `jahoda2`（含 `lda` 表：出流半径、中面高度、r/R 和 y/T 列表）；`flow_statistics_sets()` 按叶轮循环，多叶轮时集名带叶轮名并记 `impeller`；`lda` 预设加 LDA 出流点、中面格点、中面极坐标网格；`point_series_sets()` 写 `series`；`--point-series`；`--probe NAME X Y Z`（可重复）；`--flow-statistics` 允许多叶轮（要求叶片数和叶片相位相同，因为采样器的相位角只有一个）；`tank.json` 加 `lda` |
| `experiment/v1/utils/flow_statistics.py` | `PointSeries`：每次采样对 `series` 的点做 Shepard 插值（与 `PointSet` 同一核、同一半径，抽成 `shepard_velocity()`），存内存，每次 flush 和结束时写 `<stem>_series.npz`（`time`、`rotor_angle`、`<name>/velocity` (samples, points, 3) float32、`meta`）；配置里没有 `series` 时不执行任何新代码，`stats.npz` 内容不变 |
| `experiment/v1/checks/_analyze_flow_statistics.py` | 多叶轮：剖面和泵送数按叶轮；Po 按力矩 CSV 的分列（total、lower、upper、shaft）；LDA 出流点表 `<label>_lda_outflow_<impeller>.csv`；中面表 `<label>_midplane.csv` 和图 `_midplane.png`（图 3 形式）；Q_ax；单叶轮的输出文件名和表格不变 |
| `experiment/v1/checks/_analyze_point_series.py` | 新建，见下 |
| `experiment/v1/checks/_analyze_mixing_time.py` | `--probes BOTTOM TOP` 覆盖探针名 |

### 点时间序列的分析（`_analyze_point_series.py`）

对每个点（四个挡板面平均）和径向、轴向分量（切向也算，但论文没测）：

    均值 U，方差 u'²（核平均信号，采样间隔 Δt）
    自相关 ρ(τ) = ⟨u'(t) u'(t+τ)⟩ / u'²，有偏估计，FFT，τ 到 --max-lag（默认 20 s）
    拟合（van Maanen 1999，论文式 (3)），只用 τ > 0：
        ρ_fit(τ) = b + c₀ e^(−a₀τ) + Σ_{i=1,2} Σ_{j=1,2} c_ij e^(−a_ij τ) cos(2π j f_i τ)
        两个基频 f₁、f₂（边界 0.005 N 到 0.12 N）和各自的一次谐波，阻尼 a_ij ∈ [0, 0.33 s⁻¹]（论文的约束值），c ≥ 0，
        25 组 (f₁, f₂) 初值加该点谱的两个最高峰作初值，取残差最小者
    噪声份额 ε = 1 − ρ_fit(0) = 1 − (b + c₀ + Σ c_ij)        （有白噪声时 ρ(0) = 1 高于拟合外推）
    周期份额 P_fit = Σ c_ij                                     （论文式 (5)）
    谱积分法 P_SI = E(f < 0.1 N) / E_total（Welch 谱，Hann 窗）    （论文式 (6)，论文称其高估，因为不区分 MI 和其他大尺度运动）
    主频：振幅最大的余弦的 f/N；第二频率：与主频相差 20 % 以上、振幅次大的余弦
        （基频和谐波在拟合里可以互换：f₁ = 0.01 N 而振幅全在 0.02 N 的谐波上与 f₁ = 0.02 N 等价，所以按振幅报频率）

能量（论文的两分量各向同性估计，u'² 为原始方差，单位 1000 k/U_tip²）：

    k*   = 3/4 (u'²_rad + u'²_ax)(1 − ε)                 去噪声的总脉动能
    k_t  = 3/4 Σ_c u'²_c (1 − ε_c − P_c)                  湍动能（去宏观不稳定）
    k_MI = k* − k_t                                       宏观不稳定能量
    P 取 P_fit 得 k_t,fit、k_MI,fit（表 5 形式），取 P_SI 得 k_t,SI、k_MI,SI（表 6 形式）；k3 用三分量 1/2 Σ

自检（`--selftest`）：AR(1) 湍流（积分时间 0.3 s）加 f/N = 0.02（带相位游走）和 0.055 的两个余弦加白噪声，2 × 2 点 × 4 面，6000 样本：
8 个拟合全部找到主频 0.020、周期份额 0.19 到 0.31（期望 0.219）、噪声 0.04 到 0.09（期望 0.065）；第二频率 0.055 在 8 个里找到 7 个
（1 个落到 0.027 的局部极小），通过标准是主频、份额全对且第二频率 ≥ 3/4。P_SI 为 0.52：AR(1) 的低频能量也被计入，正如论文所说谱积分法高估。

### 冒烟（本机 RTX 4070 Ti SUPER，3 mm）

| 算例 | 粒子 | 步数 | 结果 |
| --- | --- | --- | --- |
| jahoda2 3 mm，统计每 10 步从 0 s，点序列，力矩分列 | 流体 1,388,377，壁 227,267，盖 24,384，转子 7,757，池 1,895,040 | 300 | 存活数相符，溢出全 0；`stats.npz` 25 集 10,979 点，`stats_series.npz` (30, 308, 3) 无 NaN；分析脚本两个都跑通（数值接近零，只验流程） |
| 同上加 5 个示踪剂（25 s 起每 25 s）、SGS、探针每 20 步 | 同上 | 60 | 存活数相符，溢出全 0；探针 CSV 列名 `probe_bottom:tracer_01` 等 |

注：3 mm 时 H/dx = 193.33 不是整数（H1 的 3 mm 同样），顶盖不在流体层上方半格处；2 mm 时为 290，整数。

## 三、算例配置

| 算例 | dx | 流时 | 生成器参数 | 运行参数 | 用途 |
| --- | --- | --- | --- | --- | --- |
| J2-S3 | 3 mm | 100 s（2,249,500 步） | `--preset jahoda2 --dx 0.003 --free-slip-lid --flow-statistics --point-series` | 统计从 20 s 每 0.02 s（450 步），力矩每 1000 步并按高度 0.2417 m、轴半径 0.010 m 分列，末态 `--dump` | 出流剖面、中面格点、Q_ax、点序列（80 s，4000 样本） |
| J2-M3 | 3 mm | 150 s（3,374,250 步） | `--preset jahoda2 --dx 0.003 --free-slip-lid --tracers 5 --injection-start 25 --injection-interval 25 --injection-duration 0.05 --sgs` | 探针每 0.02 s（450 步），力矩每 1000 步 | 混合时间 θ95（Jahoda 2007 双桨 92 到 104，即 τ95 18.4 到 20.8 s） |
| J2-S2 | 2 mm | 60 s（2,024,600 步） | `--preset jahoda2 --dx 0.002 --free-slip-lid --flow-statistics --point-series` | 统计从 20 s 每 0.02 s（675 步），力矩分列，末态 `--dump` | 分辨率检查（40 s，2000 样本） |

数值设置照 H1：h/dx = 3，c₀ = 20 U_tip = 30.37 m/s，p_b = 2 ρ U_tip² = 4611 Pa，CFL 0.15（3 mm dt = 4.445 × 10⁻⁵ s，2 mm dt = 2.963 × 10⁻⁵ s），
实心三层挡板（3 mm 时 9 mm 厚，2 mm 时 6 mm），叶片和圆盘薄板，无重力，转子 0.1 s 加速。

## 四、部署（ws8-gpu）

目标机：合作者的 Windows 11 工作站（ssh 别名 `ws8-gpu`，经跳板 `ws8`；主机名 DESKTOP-9IN3NPD，账户 Administrator），RTX 5090 32 GB，驱动 591.86，
Vulkan 设备列表 [0] AMD Radeon 核显、[1] RTX 5090（运行加 `--device 1`）；系统 Python 3.13.11，无 conda、git、glslc、cmake；控制台是日文 cmd（代码页 cp932）。
`D:\sph` 原来已有合作者自己的工程（camera.py、Components、venv 等，2026-02-02），所以我们的东西全放在子目录 `D:\sph\lifeline`，
venv 也建在 `D:\sph\lifeline\venv` 而不是任务里写的 `D:\sph\venv`（那个已存在，是别人的）。

| 步骤 | 命令（本机 Git Bash，除注明外） |
| --- | --- |
| 编译着色器 | `VULKAN_SDK=D:/Coding/Anaconda3/envs/sph/Library python experiment/v1/compile_shaders_v1.py`（16 个） |
| 打包 | `git archive --format=tar --prefix=vulkan-demo-fsi/ HEAD -o code.tar`；`tar --force-local -rf code.tar --transform 's,^,vulkan-demo-fsi/,' experiment/v1/shaders/spv`；`gzip` → `fsi_code_d47bec0.tar.gz`（21.3 MB，330 项） |
| 上传 | `ssh ws8-gpu 'mkdir D:\sph\lifeline & mkdir D:\sph\runs'`；`scp fsi_code_d47bec0.tar.gz cluster_jobs/ws8gpu/run_sph.py ws8-gpu:D:/sph/lifeline/`；`scp cluster_jobs/ws8gpu/queue_J2.ps1 ws8-gpu:D:/sph/` |
| 解压、建环境 | 目标机：`cd /d D:\sph\lifeline && tar -xzf fsi_code_d47bec0.tar.gz && "C:\Program Files\Python313\python.exe" -m venv D:\sph\lifeline\venv && D:\sph\lifeline\venv\Scripts\python.exe -m pip install vulkan numpy pyyaml pyrr scipy`（vulkan 1.3.275.1，scipy 1.18.1，PyYAML 6.0.3） |
| 包装脚本 | `D:\sph\lifeline\run_sph.py`（仓库外副本 `cluster_jobs/ws8gpu/run_sph.py`）：把 `compile_shaders_v1.compile_v1_shaders` 换成只打印的函数，用随包上传的 14 个 spv；把 stdout/stderr 改成 UTF-8（cp932 编不了力矩行里的 "N·m"，第一次基准因此 UnicodeEncodeError）；再以 `runpy` 跑给定脚本。用法 `venv\Scripts\python.exe D:\sph\lifeline\run_sph.py experiment/v1/_run_v1_headless.py CASE\case.yaml --device 1 ...` |

基准（目标机，`--preset jahoda --dx 0.003 --free-slip-lid`，单桨 851,902 粒子（流体 699,929），池 979,712，薄板内核，500 步，力矩每 250 步）：

| 量 | 值 | 定义 |
| --- | --- | --- |
| 步速 | 111.9 步/s | `_run_v1_headless.py` 末行：(500 步) / (bootstrap 后到第 500 步的墙钟时间)，含每 250 步一次力矩回读 |
| 设备 | [1] NVIDIA GeForce RTX 5090，api 1.4.325 | `VulkanContext selected` 行 |
| 溢出 | inside、incoming、fallback、neighbor 全 0 | `readback_global_status()` |
| 力矩 | 第 500 步 −1.867 × 10⁻² N·m（0.022 s，仍在 0.1 s 加速段） | `readback_rotor_torque()`，未除质量因子 |

对照：同一算例本机 4070 Ti SUPER 约 60 步/s（记忆里的 3 mm 槽 80 步/s 乘薄板内核的 0.75），Paratera 的 5090 跑 30 L 3 mm 薄板槽 75 步/s（136 万粒子）。
按粒子数线性外推，J2 的 3 mm 算例（1,647,785 粒子）约 58 步/s，2 mm（约 5.5 M 粒子）约 17 步/s，带 5 个示踪剂加标量 SGS 的 M3 约为 S3 的 0.5 到 0.6 倍（记忆：10 示踪剂加 SGS 时 40 对 75）。
日志：`D:\sph\runs\bench\generate.log`、`run_500.log`。

## 五、队列

目标机没有调度器，排队由 `D:\sph\queue_J2.ps1`（仓库外副本 `cluster_jobs/ws8gpu/queue_J2.ps1`）自己做：依次跑 J2-S3、J2-M3、J2-S2，
每个算例在 `D:\sph\runs\<名字>\` 下先用生成器生成 `case\`（`generate.log`），再经包装脚本跑求解器（`run.log`、`command.txt` 记完整命令行），
力矩 `torque.csv`，探针 `probes.csv`，统计 `stats\stats.npz`（含 `stats_series.npz`，每 250 个样本 flush 一次并留副本），末态 `final.npz`；
结束时写 `DONE`（退出码和用时），队列日志 `D:\sph\runs\queue_J2.log`。有 `DONE` 的算例在重启队列时跳过；M3 每 60 分钟写检查点并在重启时 `--resume latest`
（S 算例带流动统计，不能恢复，所以不写检查点）。求解器子进程由 `cmd /c` 启动并把 stdout 和 stderr 追加到 `run.log`（`PYTHONUNBUFFERED=1`、`PYTHONUTF8=1`）。

每个算例的命令行（目标机，`$P = D:\sph\lifeline\venv\Scripts\python.exe`，`$W = D:\sph\lifeline\run_sph.py`，`$R = D:\sph\runs`）：

```
# J2-S3
$P D:\sph\lifeline\vulkan-demo-fsi\utils\geometry\_demo_rushton_tank.py --preset jahoda2 --dx 0.003 --free-slip-lid --flow-statistics --point-series --out $R\J2-S3\case
$P $W experiment/v1/_run_v1_headless.py $R\J2-S3\case\case.yaml --device 1 --max-steps 2249500 --torque-log $R\J2-S3\torque.csv --dump $R\J2-S3\final.npz --flow-statistics-start 20 --torque-every 1000 --torque-split-height 0.2417 --torque-shaft-radius 0.0105 --flow-statistics $R\J2-S3\case\flow_statistics.json --flow-statistics-out $R\J2-S3\stats\stats.npz
# J2-M3
$P ..._demo_rushton_tank.py --preset jahoda2 --dx 0.003 --free-slip-lid --tracers 5 --injection-start 25 --injection-interval 25 --injection-duration 0.05 --sgs --out $R\J2-M3\case
$P $W experiment/v1/_run_v1_headless.py $R\J2-M3\case\case.yaml --device 1 --max-steps 3374250 --torque-log $R\J2-M3\torque.csv --dump $R\J2-M3\final.npz --probe-every 450 --torque-every 1000 --probe-log $R\J2-M3\probes.csv --checkpoint-dir $R\J2-M3\checkpoint --checkpoint-wall-minutes 60
# J2-S2
$P ..._demo_rushton_tank.py --preset jahoda2 --dx 0.002 --free-slip-lid --flow-statistics --point-series --out $R\J2-S2\case
$P $W experiment/v1/_run_v1_headless.py $R\J2-S2\case\case.yaml --device 1 --max-steps 2024600 --torque-log $R\J2-S2\torque.csv --dump $R\J2-S2\final.npz --flow-statistics-start 20 --torque-every 1000 --torque-split-height 0.2417 --torque-shaft-radius 0.0105 --flow-statistics $R\J2-S2\case\flow_statistics.json --flow-statistics-out $R\J2-S2\stats\stats.npz
```

流动统计的采样间隔不传，用运行器的默认 round(0.02 s / dt)：3 mm 450 步，2 mm 675 步。力矩分列：高度 0.2417 m（两桨中间）以下归下桨、以上归上桨，r < 10.5 mm 归轴
（`readback_rotor_torque` 用 r < shaft_radius，轴半径 10 mm，圆盘最内环在 11.5 mm 以外）。

**分离启动：`Start-Process` 不行，要用 WMI。** 第一次用 `Start-Process powershell.exe ... -WindowStyle Hidden` 启动，ssh 会话一结束，
Windows OpenSSH 把会话的整棵进程树（队列的 PowerShell 和求解器的 python）一起杀掉：J2-S3 在第 9000 步停住，队列日志没有"finished"行，GPU 占用 0 %，
留下的日志改名为 `D:\sph\runs\J2-S3\run_killed_by_ssh_exit.log`。改用 `Invoke-CimMethod Win32_Process Create`：进程由 WMI 服务创建（会话 0，账户仍是 Administrator），
不属于 ssh 会话，会话退出后继续跑；在会话 0 里 Vulkan 照常选到 [1] RTX 5090，GPU 占用 97 %。队列脚本在启动求解器前把已有的 `run.log` 改名为 `run_<时间>.log`。

启动、查看进度、停止（ssh 的远端 shell 是 cmd，引号层数多，命令复杂时用 `powershell -EncodedCommand <base64 的 UTF-16LE>`，部署时就是这么做的）：

```
ssh ws8-gpu "powershell -NoProfile -Command \"Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\sph\queue_J2.ps1'; CurrentDirectory='D:\sph'}\""
ssh ws8-gpu "powershell -NoProfile -Command \"Get-Content D:\sph\runs\queue_J2.log -Tail 5; Get-Content D:\sph\runs\J2-S3\run.log -Tail 3; Get-Process python | Select Id,StartTime,CPU,SessionId\""
ssh ws8-gpu "tasklist /FI \"IMAGENAME eq python.exe\""        # 停止前先看清是不是我们的进程，再 taskkill /PID <pid> /F
```

跳板机对短时间内的多次 ssh 连接会重置（Connection reset by 100.102.201.91，也会掐断已建立的会话）：本次部署里连续 4 次连接后被重置，隔 75 s 和 4 min 再连仍被拒，
隔 10 min 恢复。推测是 OpenSSH 9.8 起默认开启的 PerSourcePenalties（每个来源地址按异常断开累计罚时，上限 10 min），未在跳板上确认。
对策：一次连接里把事情串起来做（打补丁、启动、在会话里 Start-Sleep 后读日志都放进同一条 EncodedCommand），两次连接之间隔 10 min 以上。
分析（本机，结果拷回后）：

```
python experiment/v1/checks/_analyze_flow_statistics.py stats.npz --tank case/tank.json --torque torque.csv --torque-from 20 --out-dir analysis --label J2-S3
python experiment/v1/checks/_analyze_point_series.py stats_series.npz --tank case/tank.json --out-dir analysis --label J2-S3
python experiment/v1/checks/_analyze_mixing_time.py probes.csv --case case/case.yaml --probes probe_bottom probe_top --out-dir analysis --label J2-M3
```

队列状态（目标机本地时间 2026-10-13，比本机早；以下时间都是目标机的）：

| 时间 | 事件 |
| --- | --- |
| 19:43:39 | 第一次启动（Start-Process），J2-S3 生成 6 s 后开跑 |
| 约 19:47 | ssh 会话结束，进程树被杀，停在第 9000 步（0.40 s 流时） |
| 19:59:36 | 第二次启动（WMI，队列 pid 18116，求解器 pid 3028，会话 0），J2-S3 重新从头跑 |
| 20:02 | 启动用的 ssh 会话退出 |
| 20:12:25 | 检查：进程仍在，第 49,000 步（2.18 s 流时），90 s 内从 49,000 到 55,000 步；GPU 占用 98 %，显存 1,906 MiB，80 °C，484 W |

| 量 | 值 | 定义 |
| --- | --- | --- |
| J2-S3 步速 | 约 67 步/s | 两种算法一致：(55,000 − 49,000) / 90.0 s = 66.7（力矩行每 1000 步一条，分辨率 ±11）；49,000 步 / (20:12:25 − 19:59:36 − 约 45 s 的生成加载和 bootstrap) = 67.7 |
| J2-S3 预计用时 | 2,249,500 / 67 = 33,600 s ≈ 9.3 h，约 10-14 05:20 结束 | |
| J2-M3 预计 | 带 5 个示踪剂加标量 SGS 按 S3 步速的 0.55 计约 37 步/s，3,374,250 步 ≈ 25 h | 系数来自记忆里 30 L 3 mm 槽加 10 示踪剂加 SGS 时 40 对 75 步/s |
| J2-S2 预计 | 粒子数约 1.65 M × (3/2)³ ≈ 5.6 M，步速按粒子数反比约 20 步/s，2,024,600 步 ≈ 28 h | 显存约 2.2 GB（3 mm 用 330 MB，按池大小放大） |

三个算例串行约 2.6 天。进度查看命令见上；S3 结束后 `D:\sph\runs\J2-S3\DONE` 出现，队列自动转 M3。

## 有什么影响

- 求解器不变。采样器只在配置含 `series` 时多做插值，默认输出逐字节不变。
- `jahoda` 预设和 H1 分析的输出不变（文件名、表格）。

## 需要合作者确认

1. 两桨叶片相对角取 0°（都是 15°）；论文没给。
2. M3 的示踪剂用了标量 SGS（Smagorinsky，Sc_t 0.7）；H1 没有标量场所以无此问题。
3. 3 mm 时 H/dx 不是整数（顶盖离最上一层流体不是半格）；H1 的 3 mm 相同。
