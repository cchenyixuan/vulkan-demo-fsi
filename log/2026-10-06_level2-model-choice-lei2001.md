# 2026-10-06 级别二模型选择：改用 Lei 等 2001 的酿酒酵母结构模型，零维复现通过

用户要求：不限定毕赤酵母，用现成的文献模型；必须有记忆效应，底物高或低时有副产物。
比较八篇（`reference/biology model/`），结论和实现要点记在 `docs/stage5_pichia_level2_design_2026-10-02.md` 第 7 节。

本次提交：
- `experiment/v1/checks/_model_lei2001.py`：Lei 2001 的零维模型（表 5 速率、表 6 收支、表 7 参数），scipy LSODA。
- `docs/figures/lei2001_reference_check.png`：恒化器扫描（S_f 15 g/L）和批式（15 g/L）。
  D_crit 0.38 h⁻¹、冲出约 0.48、乙醇 D 0.45 时 4.3 g/L、批式乙醇峰 5.4 g/L 在 18.9 h，与原文图 3、图 5 一致。

没有改求解器。`state_limited`（de8a343）保留为简化选项。

待确认（用户）：菌种改为酿酒酵母、大罐参照改为 Haringa 2017 的 22 m³ 算例；参数缓冲区的做法要告知合作者。
