# τ20 一对的 480 s 续算从 ws8 移到 Mahuika：检查点经 Paratera 中转

日期：2026-11-07。背景：`log/2026-11-07_relaxation-fraction-criterion.md` 的续算表把 54 m³ SL τ45 一对放在 Mahuika（作业 10620385 / 10620386），τ20 一对（G24_SL_t20_own / _mean，11 月 2 到 3 日在跳板机 ws8 上跑的 stage 1 算例）放在 ws8 续算。用户取消了 ws8 上的续算，要求把 240 s 末检查点拉回本机再传到 Mahuika 续算。

## 原来的代码

- 不改求解器。续算用 `_run_v1_headless.py --resume latest`：从检查点恢复全部缓冲区，核对 case.yaml 的 sha1 与检查点记录一致，lifeline 记录器从检查点恢复块号，probe / torque 日志以追加方式写。
- `cluster_jobs/mahuika/submit_prod_SL54_t20_extend.sh`（另一个对话写的，未提交）：`job_prod.sh` 加 `RESUME=latest`，步数 4,743,600（480 s），快照 300、360、420、480 s，mean 加 `--population-mean-state-every 0.5`，CODE_DIR 写的是 486162c。

## 改成了什么

1. **传输路径**。本机到 ws8 的 Tailscale 走东京中继（ws8 侧是对称 NAT，无 UPnP，直连建不起来），scp 一个 54 MB 文件 5 分钟只到 5.8 MB 就停住。ws8 自己的校园网到 Paratera 和 NeSI lander 的 22 端口都通。在 ws8 上生成专用密钥 `C:\Users\WS8\.ssh\id_paratera`（ed25519），用户把公钥登记进 Paratera 的 authorized_keys，然后：
   - ws8 → Paratera `~/run/sph/xfer/`：脚本 `C:\Users\WS8\Documents\sph\push_t20.ps1` 以 WMI 分离运行，own 813 MB 用 121 s，mean 297 s（2.4 到 6.7 MB/s）；
   - Paratera → 本机 `output/xfer/`：scp；
   - 本机 → Mahuika `/nesi/nobackup/uoa04509/sph/production/G24_SL_t20_{own,mean}_ckpt/checkpoint_latest.npz`：各约 15 min。
   每一跳都核 MD5：own 1bc758e0d408a2f5a2a896f9a546280b，mean 0b233b0e5640370fba201557e0cf3894，与 ws8 源文件一致。
2. **case 目录没走 ws8**。stage 1 的 case 是 11 月 2 日在本机生成后上传 ws8 的，本地副本 `output/stage1/cases/G24_SL_t20_{own,mean}/` 的 case.yaml sha1（own 663711cf…，mean 5178c726…）与检查点 `extra.case_sha1` 一致，own 和 mean 的 obj、materials.yaml 逐文件 MD5 相同（只有 case.yaml 多一项 `state_coupling: mean`）。打包 case.yaml、materials.yaml、tank.json、21 个 obj，加上本地 `output/stage1/results/` 里的 `lifeline_meta.json`（20 块、2400 条）和 0 到 240 s 的 probes.csv、torque.csv，上传后按 `job_prod.sh` 的布局放好；case.yaml 里 `material_library: materials.yaml` 和 obj 路径都是相对路径，不需要改。lifeline 第 0 到 19 块留在本地，续算从第 20 块起写。
3. **提交脚本改一处**：CODE_DIR 从 Mahuika 上没有的 486162c 改为已部署的 688e87a。`git diff --stat 486162c 688e87a -- experiment/v1 utils/sph materials` 为空，两者只差 54 m³ 生成器（圆环、轴到底、滤波宽度选项），续算不跑生成器。脚本头部补记了输入来源。
4. 提交：Mahuika 作业 10620767（SLt20oX）、10620768（SLt20mX），PRO 6000，时限 28 h。own 07:46 NZDT 在 g04 开始，日志确认 `restored checkpoint: step=2371800 t=240.001349 s alive=2,282,450`、`resumed at step 2371800`、`[lifeline] restored: 20,000 particles, 2400 records in 20 chunks`，前 40 min 约 38 步/s；mean 排队中（Priority，空卡被 planned 给更高优先级的作业）。

## 影响

- 四个 54 m³ 续算（τ20、τ45 各一对）都在 Mahuika，ws8 不再承担算例。ws8 上 `G24_SL_t20_own\DONE` 仍叫 `DONE_240s`，如果以后在 ws8 重启 `launch_X1.ps1` 会再续算一次，不要启。
- ws8 上的大文件以后走 ws8 → Paratera（密钥已登记），不走 Tailscale。
- 续算结果的分析（240 到 480 s 的 S、20 s 子窗、I 的新值，F_mu 重算）等 τ45 和 τ20 都完成后一起做；own 预计 11-08 凌晨完成，mean 看排队。

## 待确认

- 两个 own 续算各占 PRO 6000 一张，mean 排队多久不确定；若超过半天可考虑改投 H100（`job_prod.sh` 的 `--gres` 要改）。
- Paratera `~/run/sph/xfer/` 里的两个检查点和一个 lifeline 测试文件（共 1.7 GB）用完后可删；本机 `output/xfer/` 的副本同样。
