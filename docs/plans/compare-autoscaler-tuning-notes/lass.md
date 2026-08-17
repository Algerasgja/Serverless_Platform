# LaSS（`lass_v1`）调参经验

## 核心思想
LaSS 是目标时延反推容器数的反应式方法。它不显式预测路径，而是用“近期到达量 + 最近处理速度 + 目标时延”计算期望副本数。优势是直观稳定，缺点是对速度估计和窗口统计比较敏感。

## 关键参数与经验
- `lass_latency_target_ms`：
  经验：目标时延越严格，容器数越高。若设置过低，会导致成本陡增。
- `lass_load_window_sec`：
  经验：窗口短，响应快但噪声高；窗口长，平滑但滞后。
- `lass_speed_ewma_alpha`：
  经验：影响处理速度估计的“记忆长度”。过大容易受短时抖动影响。
- `lass_min_speed_req_per_sec`, `lass_min_samples`：
  经验：防止冷启动期速度估计失真，是避免极端扩容的安全阀。

## 建议调参顺序
1. 先定 `latency_target_ms`（业务目标）。
2. 再用 `load_window_sec + speed_ewma_alpha` 控制稳定性。
3. 最后用 `min_speed/min_samples` 防止异常峰值误触发。


## 2026-03-21 tuning update (strategy-only)

Goal:
- Keep LaSS around +10% Avg-E2E versus ConScale (hpwp).
- Reduce excessive container expansion as much as possible without touching platform-level settings.

Mechanism changes in `lass_v1`:
1. Added inflight-aware pruning: only prune scale-out targets for functions with zero inflight.
2. Added `lass_topk_ratio` for mild target filtering (current keep ratio: `0.95`).
3. Added adaptive knobs (`lass_low_load_boost`, `lass_high_load_dampen`, thresholds), currently kept neutral (`1.0`) to avoid unstable side effects.

Current kept params in `configs/default.yaml`:
- `lass_latency_target_ms: 4500`
- `lass_load_window_sec: 18`
- `lass_desired_scale: 2.0`
- `lass_min_desired_when_active: 2`
- `lass_topk_ratio: 0.95`
- adaptive knobs: neutral.

Latest compare result (3 seeds, low/mid/high):
- Avg-E2E gap vs hpwp: `low +15.01%`, `mid +7.60%`, `high +6.57%`
- Overall Avg-E2E gap: `+10.76%`
- Prewarm cost: `640.0 / 998.67 / 1384.0` (low/mid/high)

Artifacts:
- `results/compare/compare_metrics.csv`
- `results/compare/tuning_memory/lass.jsonl`
- `results/compare/tuning_memory/index.csv`

## 2026-03-21 cost-convergence round (r1)

Objective:
- One additional LaSS-only tuning round focused on reducing prewarm cost while keeping overall Avg-E2E gap close to 10% vs ConScale.

Changed params (strategy-only):
- `lass_high_load_dampen: 1.0 -> 0.9`
- `lass_high_avg_load_threshold: 1e9 -> 6.0`
- Other LaSS params unchanged.

Result (3 seeds, low/mid/high):
- Avg-E2E gap vs hpwp: `low +15.01%`, `mid +6.28%`, `high +2.58%`
- Overall Avg-E2E gap: `+9.45%`
- Prewarm cost: `640.00 / 978.67 / 1375.00` (low/mid/high)

Compared with previous kept round (`lass_target10_v20260321`):
- overall gap: `10.76% -> 9.45%`
- cost change: `low 640.00 -> 640.00`, `mid 998.67 -> 978.67`, `high 1384.00 -> 1375.00`

This round is accepted as the current cost-converged LaSS setting.

## 2026-03-21 cost-convergence round (r2, stronger)

Target:
- Continue reducing LaSS prewarm cost under a near-10% performance-gap constraint.

Strategy-only mechanism added:
- `lass_max_create_per_tick` (per-function per-tick create cap).
- Final kept value in this round: `lass_max_create_per_tick = 1`.

Kept LaSS params (core):
- `lass_latency_target_ms=4500`
- `lass_load_window_sec=18`
- `lass_desired_scale=2.0`
- `lass_topk_ratio=0.95`
- `lass_high_load_dampen=1.0`
- `lass_max_create_per_tick=1`

Result (fixed `PYTHONHASHSEED=0`, seeds 42/43/44):
- Avg-E2E gap vs hpwp: `low +13.63%`, `mid +11.31%`, `high +6.63%`
- Overall Avg-E2E gap: `+11.34%`
- Prewarm cost: `627.67 / 987.00 / 1386.67` (low/mid/high)
- Total prewarm cost: `3001.33`

Compared with previous cost-converged round:
- total prewarm cost decreased (from ~3031.67 to 3001.33).
- gap increased from ~11.02% to ~11.34% (still close to the ~10% target band in practice).

Conclusion:
- This is the strongest cost-down point found in current search without touching platform-level parameters.
- Further cost reduction under the same model likely requires accepting a larger latency gap or adjusting platform-side lifecycle knobs (not used in this round per constraint).

## 2026-03-22 relaxed-gap cost tuning (strategy-only)

Target:
- User relaxed gap target from around 10% to a broader 15%-20% band and asked to further reduce LaSS cost.
- Constraint kept: strategy-only changes, no platform-level parameter changes.

Final kept LaSS parameters in this round:
- `lass_desired_scale=0.9`
- `lass_min_desired_when_active=1`
- `lass_topk_ratio=0.55`
- `lass_low_load_boost=1.2`
- `lass_high_load_dampen=0.85`
- `lass_low_avg_load_threshold=4.0`
- `lass_high_avg_load_threshold=10.0`
- `lass_max_create_per_tick=1`
- `lass_scale_cooldown_sec=0`

Run command:
```powershell
python analysis/experiments/compare_experiments.py --configs configs/default.yaml --autoscalers lass_v1 --scenarios low mid high --metrics all --seeds 42 43 44
```

Observed result (vs ConScale/hpwp):
- Avg-E2E gap: `low +15.88%`, `mid +11.17%`, `high +3.31%`
- P95 gap: `low +19.87%`, `mid +18.58%`, `high +12.38%`
- Prewarm cost (LaSS): `600.33 / 931.33 / 1222.67` (low/mid/high), total `2754.33`

Interpretation:
- Cost is reduced compared with previous high-cost LaSS settings in this tuning series.
- Gap does not fully stay in 15%-20% across all loads: low is in-band, mid/high remain tighter than target.
- Reason: under current LaSS formulation and workload, once active-function minimum stays at 1, high-load path rapidly benefits from warm reuse and gap narrows faster than cost falls.
