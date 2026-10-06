# 2026-10-06 级别二模型改为 Tang 等 2017 九池模型；零维复现通过（三处参数订正）

用户指出 Lei 2001 的记忆是小时级，对 lifeline 没有意义；重新按"循环时间尺度的记忆"筛选，改为 Tang 2017 九池模型
（Haringa 2018 补充材料 A 的形式），菌种青霉菌，大罐参照 54 m³ 罐。比较、复现结果、三处与打印表不符的参数和实现要点记在
`docs/stage5_pichia_level2_design_2026-10-02.md` 第 8 节。

本次提交：`experiment/v1/checks/_model_ninepool.py`（零维模型）、`docs/figures/ninepool_reference_check.png`。没有改求解器。

待确认（用户）：菌种改青霉菌、参照改 54 m³；三处参数订正是否可接受（或向作者核对）；副产物是胞内贮藏物而非排出物。
