# 2026-09-24 查看器速度着色改为 jet 配色

## 改动

- `experiment/v1/shaders/render/particle.vert`
  - 新增 `colormap_jet(t)`：深蓝 → 蓝 → 青 → 绿 → 黄 → 红 → 深红，输入 [0, 1]。
  - 颜色模式 0（速度）和 1（加速度）改用 `colormap_jet`，原 `colormap_viridis` 保留未删。
  - 其他颜色模式（密度偏差、体素、核函数和、涡量）不变。

## 原因

viridis 在默认速度色标（1 m/s 饱和）下，空腔内部低速区几乎全是深紫色，看不出结构。
jet 是 CFD 后处理的常见配色，低速深蓝、高速红，读图习惯一致。

## 验证

- `_run_v1_viewer.py --point-size 4` 重新编译 shader 后运行正常，1M 方腔算例渲染正确。
- 求解器 kernel 未改动，headless 结果不受影响。

## 环境备注（不进仓库）

本机无 Vulkan SDK，glslc 来自 conda 环境 `sph` 的 shaderc 包，`VULKAN_SDK` 指向该环境的 `Library` 目录。
本机 NVIDIA 着色器磁盘缓存会让 `vkCreateComputePipelines` 随机返回 VK_ERROR_UNKNOWN，运行时设 `__GL_SHADER_DISK_CACHE=0` 规避。
