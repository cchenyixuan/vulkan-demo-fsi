# 54 m³ 生成器：标准几何和 Δ 选项入库

日期：2026-11-02。用户决定修正几何为 54 m³ 标准配置，这些选项必须入库，否则生产算例不可复现。选项由另一个会话写好，G24、G10、阶段 1 都用这个版本生成（md5 b0bb3f44），这里只是提交，内容没有改。

## 原来的代码

`utils/geometry/_demo_rushton_tank.py` 无圆环，转轴只到最低叶片下缘，没有 `--sgs-filter-width`。

## 改成了什么

- `--sparger-ring`：preset haringa54 的静止实心圆环，外径 1.3 m，截面直径 0.13 m，中心高 0.35 m。尺寸从 Haringa 2016 图 1、2023 图 3 读取，截面按圆形假设。
- `--shaft-to-floor`：转轴延伸到罐底。
- `--sgs-filter-width FACTOR`：`scalars.sgs.filter_width = FACTOR × dx`。

## 影响

不加这三个选项时输出不变。生产用法的完整命令见 `log/2026-11-02_production-rerun-plan.md` 和后续提交日志。

## 待确认

无。
