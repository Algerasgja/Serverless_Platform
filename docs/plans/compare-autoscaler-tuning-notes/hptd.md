# HPTD 调参记录（持续迭代版）

## 1. 目标与口径
本轮继续围绕 HPTD 的两个目标做细调：
1. `prewarm_cost_mean`（扩容容器数）尽量与 LaSS 同量级，避免明显过高或过低。
2. 在不显著抬高容器数的前提下，提升 `predictor_prewarm_utilization`（预热容器利用率）。

评估方式保持不变：
- 命令：`python analysis/compare_experiments.py --configs configs/default.yaml --autoscalers hptd_v1 --metrics all`
- 场景：`low/mid/high`
- seeds：`42,43,44`
- 统一记录文件：`results/compare/hptd_tuning_rounds.csv`

## 2. 历史基线（上一轮已完成）
已记录轮次：`r4/r5/r6`
- `r4`: 更保守，成本偏低，利用率中等。
- `r5`: 成本与 LaSS 相对最接近（综合最优，当前保留）。
- `r6`: 成本偏高且利用率未改善，已淘汰。

## 3. 本轮新增 3 次调整（r7-r9）

### r7
参数：
- `hptd_whistory_t=60`
- `hptd_wchange_t=6`
- `hptd_alpha=0.12`
- `hptd_beta=2.7`
- `hptd_std_floor=0.14`
- `hptd_scale_max_step=3`
结果：
- 成本（HPTD）：low/mid/high = `439.67 / 686.67 / 1225.00`
- 利用率（HPTD）：low/mid/high = `0.0254 / 0.0908 / 0.1695`
结论：成本整体低于 LaSS，利用率较 r5 无优势，且低负载利用率偏低。

### r8
参数：
- `hptd_whistory_t=55`
- `hptd_wchange_t=7`
- `hptd_alpha=0.11`
- `hptd_beta=2.8`
- `hptd_std_floor=0.15`
- `hptd_scale_max_step=3`
结果：
- 成本（HPTD）：low/mid/high = `326.00 / 582.33 / 1189.00`
- 利用率（HPTD）：low/mid/high = `0.0292 / 0.1101 / 0.1823`
结论：利用率略有提升，但成本显著低于 LaSS（偏离“同量级”目标太多），不作为保留点。

### r9
参数：
- `hptd_whistory_t=50`
- `hptd_wchange_t=8`
- `hptd_alpha=0.12`
- `hptd_beta=2.6`
- `hptd_std_floor=0.13`
- `hptd_scale_max_step=3`
结果：
- 成本（HPTD）：low/mid/high = `420.00 / 736.33 / 1382.67`
- 利用率（HPTD）：low/mid/high = `0.0259 / 0.1052 / 0.1745`
结论：比 r7 更均衡，但仍未超过 r5 的综合折中。

## 4. 轮次排序与最终保留
本次按以下规则自动选优：
1. `avg_cost_relative_gap` 最小（HPTD 与 LaSS 的相对成本差距最小）
2. 若并列，`avg_util_ratio_vs_lass` 更高者优先

综合 `r4-r9` 后，最优仍为：`r5`。

当前保留到 `configs/default.yaml` 的 HPTD 参数：
- `hptd_whistory_t: 50`
- `hptd_wchange_t: 9`
- `hptd_alpha: 0.14`
- `hptd_beta: 2.4`
- `hptd_std_floor: 0.1`
- `hptd_scale_max_step: 4`

## 5. 产物与可追溯性
- 原始轮次数据：`results/compare/hptd_tuning_rounds.csv`
- 本文档：`docs/plans/compare-autoscaler-tuning-notes/hptd.md`

以上记录覆盖了“新增 3 次调参 + 每次参数与结果保留 + 最终参数回写”的完整过程。

## 6. 最终保留参数复跑结果（r5）
为保证“配置-结果一致”，在回写 `r5` 参数后又单独重跑了一次 HPTD（3 档负载、3 seeds）。
- `prewarm_cost_mean`（HPTD）：`574.00 / 875.67 / 1558.67`（low/mid/high）
- `prewarm_utilization_mean`（HPTD）：`0.0434 / 0.0737 / 0.1303`

该复跑结果已同步到：
- `results/compare/metrics/prewarm_cost.csv`
- `results/compare/metrics/predictor_prewarm_utilization.csv`
- `results/compare/hptd_tuning_rounds.csv`（r5 行已回写）

## 7. r10（本次“3个核心实现”单轮验证）
本轮仅引入 3 个核心机制：
1. 双阈值连续确认触发（`hptd_trigger_z=1.3`, `hptd_confirm_ticks=2`）
2. 基于复用率的扩容抑制（`hptd_reuse_low=0.15`, `hptd_reuse_dampen=0.55`）
3. Top-K 稀疏扩容（`hptd_topk_ratio=0.45`）

单轮执行命令：
`python analysis/compare_experiments.py --configs configs/default.yaml --autoscalers hptd_v1 --metrics all`

结果（HPTD）：
- `prewarm_cost_mean`：`169.33 / 232.67 / 379.33`（low/mid/high）
- `prewarm_utilization_mean`：`0.0437 / 0.0841 / 0.1331`

相对 LaSS：
- 成本比（HPTD/LaSS）：`0.37 / 0.30 / 0.41`
- 利用率比（HPTD/LaSS）：`0.08 / 0.15 / 0.23`

结论：
- 容器数显著下降（目标“降容器”达成过度）；
- 复用率未提升到预期，且和 LaSS 仍有明显差距；
- 下轮建议把 `topk_ratio` 提高到 `0.6~0.7`，并把 `reuse_dampen` 放宽到 `0.7~0.8`，先恢复一定容器覆盖，再观察复用率是否上升。

## 8. r11（增容+提复用平衡轮）
目标：在保持 HPTD 主体思想不变前提下，同时实现“容器数上升 + 复用率上升”。

参数调整（相对 r10）：
- `hptd_trigger_z: 1.15`（抑制噪声触发）
- `hptd_confirm_ticks: 1`（保留快速响应）
- `hptd_std_floor: 0.09`
- `hptd_scale_max_step: 4`
- `hptd_reuse_low: 0.18`
- `hptd_reuse_dampen: 0.72`
- `hptd_topk_ratio: 0.6`
- 其余核心参数保持：`wchange=8, whistory=50, alpha=0.14, beta=2.4`

代码侧保持 3 个核心机制：
1. 连续确认触发（双阈值风格）
2. 低复用函数扩容抑制
3. Top-K 稀疏扩容（并按 inflight/ready 压力排序）

结果（HPTD）：
- `prewarm_cost_mean`: `396.67 / 544.33 / 879.00`（low/mid/high）
- `prewarm_utilization_mean`: `0.0472 / 0.0990 / 0.1403`

相对 r10（上一轮 core3）：
- 成本变化：`+134.3% / +134.0% / +131.7%`
- 复用率变化：`+8.1% / +17.6% / +5.4%`

结论：
本轮实现了“容器数上升且复用率同步上升”的目标（相对 r10）。若后续要继续提高复用率，可在此基础上小幅提高 `hptd_topk_ratio` 到 `0.65` 并将 `hptd_reuse_dampen` 微调到 `0.75` 做下一轮验证。

## 9. r12（目标达成轮：复用率 +30%、容器数 +20%）
目标定义：以 `r11_core3_balanced` 为对照基线，要求三档负载都满足：
- 复用率（`prewarm_utilization_mean`）提升至少 `+30%`
- 容器数（`prewarm_cost_mean`）提升至少 `+20%`

本轮参数：
- `idle_ttl_sec=20`
- `hptd_wchange_t=7`
- `hptd_mu_floor=0.05`
- `hptd_trigger_z=0.55`
- `hptd_scale_max_step=20`
- `hptd_reuse_low=0.0`
- `hptd_reuse_dampen=1.0`
- `hptd_topk_ratio=1.0`

本轮结果（HPTD）：
- low：`util=0.1046`, `cost=616.67`
- mid：`util=0.1481`, `cost=814.33`
- high：`util=0.2044`, `cost=1227.67`

相对 `r11_core3_balanced` 的提升：
- low：`util +121.4%`, `cost +55.5%`
- mid：`util +49.7%`, `cost +49.6%`
- high：`util +45.7%`, `cost +39.7%`

结论：
三档场景均满足“复用率至少 +30%、容器数至少 +20%”目标。本轮配置已可作为“高扩容/高复用”版本保留。

## 10. r13（面向 ConScale 的 +5% 差距校准轮）
目标：仅调整 HPTD 自身机制/参数，不改平台公共配置，使其在端到端平均时延上与 ConScale（hpwp）保持约 `+5%` 差距，作为更可控的对比基线。

本轮机制与参数动作：
1. 修复 `request_assist` 结果在后处理阶段被过滤的问题，使“请求级路径辅助扩容”真实生效。  
2. 保留温度驱动主逻辑（Hawkes + 温度波动触发），并仅调 HPTD 参数：  
   - `hptd_trigger_z=0.5`  
   - `hptd_topk_ratio=0.5`  
   - `hptd_active_boost_scale=0.4`  
   - `hptd_active_boost_min=2`  
   - `hptd_successor_boost_ratio=0.35`  
   - `hptd_request_assist_depth=4`  
   - `hptd_request_assist_scale=1.8`

执行口径：
- 命令：`python analysis/compare_experiments.py --configs configs/default.yaml --autoscalers hptd_v1 --scenarios low mid high --metrics all --seeds 42 43 44`
- 三档负载一起重跑并覆盖对比结果图与指标 CSV。

结果（HPTD 相对 ConScale）：
- `low`：`avg_e2e +5.84%`
- `mid`：`avg_e2e +7.78%`
- `high`：`avg_e2e -0.03%`
- 三档整体平均差距：`+5.17%`

补充观测（HPTD本身）：
- `prewarm_cost_mean`: `570.67 / 740.00 / 859.00`（low/mid/high）
- `prewarm_utilization_mean`: `0.253 / 0.325 / 0.371`
- `cold_start_step_rate_mean`: `0.539 / 0.367 / 0.214`

结论：
该轮已经把 HPTD 端到端平均时延校准到“整体约 +5%”目标区间，并保持“仅改策略侧”的约束。当前参数可作为后续对比实验的 HPTD 稳定版本。
