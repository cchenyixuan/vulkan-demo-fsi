# 2026-09-30 成对力修正改成三选一：自身矩阵、平均矩阵、反向修正（spec 48）

分支 `test`。上一个提交（`log/2026-09-30_symmetric-pair-correction.md`）加了"平均矩阵"的开关。
这次把开关改成模式，增加文献里的"反向修正"，并用同样的 6 s 收支算例对比三种形式。

## 我们上一版的代码

spec 48 是布尔量 `USE_SYMMETRIC_PAIR_CORRECTION`，配置项 `numerics.symmetric_pair_correction`。
打开后流体与流体的粒子对用两个修正矩阵的平均：

    压力项：(P_i + P_j) · ½ (B_i + B_j) ∇W_ij          B = M⁻¹

## 改成了什么

spec 48 改成无符号整数 `PAIR_CORRECTION_MODE`，配置项 `numerics.pair_correction`：

| 取值 | 名称 | 压力项（正压，对称形式） | 负压区（TIC 形式） | 粘性项 |
| --- | --- | --- | --- | --- |
| 0 | `own` | (P_i + P_j) B_i ∇W | (P_j − P_i) B_i ∇W | B_i ∇W，邻居体积 V_j |
| 1 | `mean` | (P_i + P_j) ½(B_i + B_j) ∇W | (P_j − P_i) ½(B_i + B_j) ∇W | ½(B_i + B_j) ∇W，体积 2m/(ρ_i + ρ_j) |
| 2 | `reverse` | (P_i B_j + P_j B_i) ∇W | (P_j − P_i) B_i ∇W | 同 `mean` |

`own` 是合作者原来的代码，默认值，行为不变。

反向修正出自 Zhang、Adams、Hu 2025（CMAME 433:117484）。公式从引用它的论文里核对过
（arXiv 2503.11292 式 18：∇ψ_i = −Σ_j (ψ_i B_j + ψ_j B_i) ∇W_ij V_j）。把它拆开：

    (P_i B_j + P_j B_i) ∇W = P_i (B_i + B_j) ∇W + (P_j − P_i) B_i ∇W

- 第二部分是粒子 i 自己的一阶精确梯度。
- 第一部分与绝对压力成正比，是残差，要靠粒子排列来消除。
- 负压区的 TIC 形式去掉的正是第一部分，所以在 `reverse` 模式下 TIC 粒子对保持用 B_i。

`force.comp` 里的实现：

```glsl
vec3 pressure_gradient_pair = pressure_combined * kernel_gradient_pair;      // own 和 mean
if (PAIR_CORRECTION_MODE == 2u && pair_is_corrected) {
    // reverse correction: P_i B_j + P_j B_i; the TIC form keeps B_i
    pressure_gradient_pair = symmetric_form
        ? (self_pressure * kernel_gradient_neighbor + neighbor_pressure * kernel_gradient_corrected)
        : (pressure_combined * kernel_gradient_corrected);
}
a_pressure -= neighbor_volume_dynamic / self_density * pressure_gradient_pair;
```

其中 `kernel_gradient_corrected` = B_i ∇W，`kernel_gradient_neighbor` = B_j ∇W，
`pair_is_corrected` 表示这一对是流体与流体、两个矩阵都可用。

配套改动：

| 文件 | 改动 |
| --- | --- |
| `experiment/v1/shaders/common.glsl` | spec 48 改为 `PAIR_CORRECTION_MODE`，注释里写明三种形式 |
| `experiment/v1/shaders/force.comp` | 上面的分支 |
| `experiment/v1/shaders/correction.comp` | 写入流体标志的条件改为 `PAIR_CORRECTION_MODE != 0u` |
| `utils/sph/case.py` | `PAIR_CORRECTION_MODES`，`NumericsConfig.pair_correction`；旧的 `symmetric_pair_correction: true` 仍可用，等同于 `mean` |
| `experiment/v1/utils/simulator_v1.py` | spec 常量 |
| `utils/geometry/_demo_stirred_tank_30l.py` | 新增 `--pair-correction {own,mean,reverse}`；旧的 `--symmetric-pair-correction` 保留 |
| `experiment/v1/shaders/README.md`、`CLAUDE.md` | spec 常量表 |

## 验证

### 二维 Couette（本机，40,000 步）

流体之间成对力的净力矩与解析力矩之比：

| 设置 | `own` | `mean` | `reverse` |
| --- | --- | --- | --- |
| 背景压力 20 Pa，位移修正打开 | −0.23% | −0.01% | −0.005% |
| 位移修正关闭（粒子排列变乱） | −28% | −6% | −4.5% |
| 不加背景压力，位移修正打开 | +0.4% | 没有单独跑 | +0.5% |

转子的读回力矩与解析值之比 1.027–1.033，三种形式相同。用新配置项写的 `mean` 重现了上一版的结果。

### 本机搅拌槽短测试

`reverse`，重力 1 倍，4000 步：粒子数 1,360,100 不变，overflow 为 0，收支各项与 `mean` 的前 4000 步接近。

### 并行科技的 6 s 收支算例

| 作业号 | 名称 | 重力 | 用时 |
| --- | --- | --- | --- |
| 1638507 | blade_conf1_acc_g03_rev_budget | 0.3 倍 | 19 分 58 秒 |
| 1638508 | blade_conf1_acc_g1_rev_budget | 1 倍 | 19 分 52 秒 |

设置与上一版日志的算例相同（3 mm，单层贴体叶片，h/dx = 3，c₀ = 20 U_tip，累积壁面，旧几何），
`case.yaml` 只差修正模式一行。粒子数不变，overflow 和 KCG 回退为 0。合计 0.66 卡时。

**角动量收支**，4–6 s 平均，读回值（含质量系数 1.2187），mN·m：

| 算例 | 来自转子 | 来自壁面 | 流体之间的成对力 | 占转子输入 | 位移修正 | 实测的 dL/dt |
| --- | --- | --- | --- | --- | --- | --- |
| 重力 0.3 倍，`own` | +65.3 | −22.0 | −24.1 | −37.0% | −4.6 | +16.4 |
| 重力 0.3 倍，`mean` | +67.9 | −44.5 | +4.6 | +6.8% | −5.3 | +24.3 |
| 重力 0.3 倍，`reverse` | +65.8 | −46.6 | +4.1 | +6.2% | −5.1 | +16.8 |
| 重力 1 倍，`own` | +73.5 | −2.7 | −67.9 | −92.4% | −4.3 | −2.4 |
| 重力 1 倍，`mean` | +80.2 | −59.7 | −0.1 | −0.2% | −6.1 | +13.3 |
| 重力 1 倍，`reverse` | +80.1 | −55.6 | −0.2 | −0.3% | −6.3 | +18.4 |

**力矩**，4–6 s 平均，已除以 1.2187，mN·m：

| 算例 | Rushton | PBT | 总计 | 壁面合计 | 其中挡板 |
| --- | --- | --- | --- | --- | --- |
| 重力 0.3 倍，`own` | 36.6 | 16.9 | 53.6 | 18.1 | 11.8 |
| 重力 0.3 倍，`mean` | 38.7 | 17.0 | 55.7 | 36.5 | 19.3 |
| 重力 0.3 倍，`reverse` | 37.5 | 16.5 | 54.0 | 38.2 | 21.2 |
| 重力 1 倍，`own` | 40.9 | 19.3 | 60.3 | 2.2 | 13.9 |
| 重力 1 倍，`mean` | 45.1 | 20.6 | 65.8 | 49.0 | 35.0 |
| 重力 1 倍，`reverse` | 45.2 | 20.5 | 65.7 | 45.7 | 31.4 |

4–6 s 内总力矩各采样的标准差为 1.4–1.9 mN·m。

**全槽的量和各区速度**（角动量和动能为 4–6 s 平均，速度为 t = 6 s 单个时刻的粒子平均）：

| 量 | 0.3 倍 `mean` | 0.3 倍 `reverse` | 1 倍 `mean` | 1 倍 `reverse` |
| --- | --- | --- | --- | --- |
| 角动量，10⁻³ kg·m²/s | 225 | 223 | 231 | 241 |
| 动能，J | 0.91 | 0.91 | 1.03 | 1.06 |
| Rushton 区 \|v\|，m/s | 0.395 | 0.384 | 0.414 | 0.412 |
| PBT 区 \|v\|，m/s | 0.302 | 0.299 | 0.296 | 0.294 |
| 全槽 \|v\|，m/s | 0.188 | 0.184 | 0.202 | 0.206 |
| 全槽 u_t，m/s | 0.079 | 0.077 | 0.079 | 0.083 |
| 上部 y > 250 mm \|v\|，m/s | 0.077 | 0.077 | 0.108 | 0.109 |
| 壁面竖直力，N（重量 × 质量系数：0.3 倍 106.0，1 倍 353.2） | 106.4 | 106.4 | 352.9 | 353.0 |

## 结论

1. `mean` 和 `reverse` 在这组算例里没有可分辨的差别：力矩相差不超过 3%，在采样波动之内；
   成对力的角动量误差都降到转子输入的 0.2%–7%。
2. 两种守恒形式都把角动量交给了壁面，流场不再被压力水平压低。这是"成对守恒"带来的，与具体形式无关。
3. `reverse` 有已发表的文献依据，一阶部分用的是自身矩阵。后面的算例建议用 `reverse`。
4. 6 s 时流动还在建立。稳态的力矩和流场需要 30 s 算例，几何也要换成新几何（挡板 24 mm 加钟形罩）。

## 没有解释的地方

- 重力 0.3 倍时成对力一项是 +4 到 +6 mN·m，两种守恒形式都有，重力 1 倍时接近零。原因没有查。
- Zhang 等要求粒子按修正矩阵做松弛。我们的位移修正不是按这个条件设计的，残差没有单独测过。

## 需要合作者确认

1. 默认值保持 `own`。是否同意有重力或背景压力的算例改用守恒形式。
2. `mean` 和 `reverse` 选哪一个作为推荐值，或者有没有第三种更合适的形式。

## 输出（不入库）

`output/budget/paratera_sym/blade_conf1_acc_g{03,1}_rev_budget_*`：`budget.csv`、`status.csv`、`torque.csv`、`case.yaml`、`final.npz`。
