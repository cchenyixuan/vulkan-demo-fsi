# 2026-10-26 工作文件夹移到 D:\Project\SPH lifeline，去掉写死的绝对路径

## 原来的代码

整个 `SPH lifeline` 原在 `D:\Uni of Auckland Dropbox\Beichen Zhao\CFD Projects\SPH lifeline`（Dropbox 内）。四个脚本写死了这个路径：

| 文件 | 写死的内容 |
|---|---|
| `cluster_jobs/paratera/stitch_Da30_local.sh` | `cd` 到仓库根目录 |
| `experiment/v1/checks/_analyze_baffle_wake_and_lid.py` | `sys.path` 加 checks 目录 |
| `experiment/v1/checks/_analyze_scale_bands.py` | `sys.path` 加 checks 目录；参考快照 REF |
| `experiment/v1/checks/_plot_large_scale_regions.py` | 同上（已另有一行按 `__file__` 加 checks 目录） |

## 改成了什么

用户把 `SPH lifeline` 整体移到 `D:\Project\SPH lifeline`（移出 Dropbox）。四个脚本改为按自身位置推算：

- shell：`cd "$(dirname "$0")/../.."`
- Python：checks 目录为 `Path(__file__).resolve().parent`，仓库根为 `Path(__file__).resolve().parents[3]`，REF = 根 / `output/kecause/s3_plates_pb2000_newpbt_fluentinit_dense_t0.300.npz`；`_plot_large_scale_regions.py` 删掉重复的那一行。

## 影响

- 只改路径，不改计算。语法检查通过，REF 所在目录在新位置可解析。
- 以后再移动文件夹，这四个脚本不需要改。
- 集群上的代码副本不受影响（Mahuika `vulkan-demo-fsi_80b4a2c`，Paratera `vulkan-demo-fsi_pb`，ws8 `D:\sph\lifeline\vulkan-demo-fsi`）。
- Claude 的项目记忆按路径存放，已复制到新路径对应的目录。

## 待确认

- `output/`（63 GB，git 忽略）不再在 Dropbox 里，没有云端副本；需要的话另做备份。
