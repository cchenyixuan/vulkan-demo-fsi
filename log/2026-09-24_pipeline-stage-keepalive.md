# 2026-09-24 修复 vkCreateComputePipelines 随机返回 VK_ERROR_UNKNOWN

## 现象

在 RTX 4070 Ti SUPER（驱动 616.92，python-vulkan 1.3.275.1，Python 3.12）上，
`SphSimulatorV1` 构造时 `vkCreateComputePipelines` 随机失败，失败的 kernel 每次不同
（predict、update_voxel、correction、density、force 都出现过），二维方腔偶发，三维方腔几乎必现。
SPIR-V 通过 spirv-val，与 glslc 优化级别、隐式层无关。关闭 NVIDIA 着色器磁盘缓存能降低二维的失败率，
但三维仍失败，说明缓存只是改变了内存复用模式，不是根因。

## 根因

`_build_compute_pipelines` 在循环里反复给局部变量 `stage` 赋新的
`VkPipelineShaderStageCreateInfo`。该结构体按值拷进 `VkComputePipelineCreateInfo`，
但 `pName="main"` 对应的 C 字符串由 Python 侧的 stage 对象持有，对象被回收后指针悬空。
到真正调用 `vkCreateComputePipelines` 时，驱动读到的入口点名是被复用的内存，返回 VK_ERROR_UNKNOWN。
旁证：循环最后一个 `bootstrap_half_kick` 和单独创建的 `defrag` 从未失败，因为它们的 stage 对象仍然存活。

## 改动

- `experiment/v1/utils/simulator_v1.py`：`_build_compute_pipelines` 增加 `stage_keepalive` 列表，
  所有 stage 对象保活到 `vkCreateComputePipelines` 返回之后。
- `renderer_v1.py` 的图形管线本来就把两个 stage 放在列表里，无需改动。

## 验证

- 逐个创建管线的诊断脚本，驱动缓存开启，三维方腔连续 8 次全部 8 个管线成功（修复前 3 次里每次失败 3 到 4 个）。
- headless 三维方腔 1,295,029 粒子 300 步：37.6 步/s，溢出计数 0。
- headless 二维方腔 1,046,529 粒子 300 步：231.9 步/s，溢出计数 0。
