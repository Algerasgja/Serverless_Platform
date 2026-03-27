# ConScale（`hpwp_v1`）调参经验

## 核心思想
ConScale 是“上下文前缀 + 层次平滑 + 周期重规划”的预测式预热策略。它的优势来自于对路径上下文的建模能力，而不是单纯靠提高容器数。因此调参时要避免把它调成“高成本全预热”。

## 关键参数与经验
- 调度节奏与窗口：
  - `hpwp_sched_eta_exec`
  - `hpwp_horizon_alpha`
  经验：这两个参数决定“看多远”和“多久重算一次”。窗口太大通常会拉高 `prewarm_cost`，窗口太小会让 `cold_start_step_rate` 降不下来。
- 模型稳健性：
  - `hpwp_beta_hi`, `hpwp_beta_lo`
  - `hpwp_alpha_stable`
  经验：`beta` 决定长上下文回退强度；`alpha_stable` 决定稳态更新速度。若负载波动大，`alpha_stable` 过高会造成抖动。
- 相位与漂移：
  - `hpwp_phase_*`
  - `hpwp_drift_*`
  经验：用于“保守-激进”切换。漂移参数过敏会造成频繁遗忘，命中率下降。

## 建议调参顺序
1. 先锁 `beta` 和 `alpha_stable`，只调 `eta_exec + horizon_alpha` 找到成本/收益平衡区。
2. 再调 `phase` 保稳态，最后再开 `drift` 微调非平稳负载。
3. 观察 `prewarm_utilization` 是否同步提升，若只看到 `prewarm_cost` 上升而利用率不动，说明窗口过宽或预测分布过散。

