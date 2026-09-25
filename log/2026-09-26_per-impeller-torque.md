# 2026-09-26 分桨力矩输出

## 改动

- `SphSimulatorV1.readback_rotor_torque(split_height, shaft_radius)`：两个参数都是可选的。
  给出时，转子粒子按到转轴的距离 r 和沿轴的高度 h 分成三组，r 和 h 都从 pivot 起算：
  r < shaft_radius 算"轴"；其余粒子中 h < split_height 算"下桨"，h ≥ split_height 算"上桨"。
  返回值多出 `torque_axis_lower / _upper / _shaft` 和三组粒子数。三段相加严格等于 `torque_axis`。
- `_run_v1_headless.py` 新增 `--torque-split-height H` 和 `--torque-shaft-radius R` 两个参数。
  给出后，力矩 CSV 多三列 `torque_lower, torque_upper, torque_shaft`。第一次采样时会打印三组粒子数。
  原有七列不变，旧 CSV 和旧分析脚本照常可用。
- `checks/_analyze_power_number.py`：检测到分桨列时，额外打印 Rushton、PBT、轴各自的 Np。
  三者都按 D = 0.096 m 计算，因此相加等于总 Np。另给出按 PBT 自身直径 0.0982 m 计算的 PBT Np，
  用来和单桨关联式比较。
- 3 mm 算例的 materials.yaml 转向修正（见 2026-09-26_rotor-direction-fix.md）一起提交。

搅拌槽参数取 H = 0.1 m，R = 7 mm。轴半径 4 mm，Rushton 和 PBT 轮毂半径分别是 10.2 mm 和 10.9 mm。
Rushton 位于 y = 0.021–0.048，PBT 位于 y = 0.180–0.209，所以 7 mm 和 0.1 m 都落在两类零件之间的空隙里。

## 验证

4 mm 算例，正确转向，6000 步（约 1.05 s），每 1000 步采样一次：

- 三组粒子数：下桨 1,640，上桨 1,970，轴 1,022，合计 4,632，等于转子粒子总数。
- 三段之和与总力矩的相对差最大 5.4e-7，属于 float64 求和的舍入误差。
- 1 s 时下桨约 43%，上桨约 57%，轴不到 0.1%。流场还在启动阶段，而且 4 mm 档桨叶厚 12 mm，
  是实际 2.5 mm 的 5 倍，这个比例只是初步读数，要看 20–30 s 窗口的结果。
