# 对比实验设计（统一脚本版）

## 1. 实验目标
在统一执行语义（`least_load + 无队列 + 单容器单任务`）下，对比不同扩缩容策略在以下维度的表现：
- 端到端时延（Avg/P95/P99）
- 冷启动率
- 预热利用率
- 扩容成本（扩容容器数）

## 2. 当前对比方法集合
统一对比脚本默认方法：
- `conscale`（`hpwp_v1`）
- `xunadu`（映射到 `xanadu_opt_v1`）
- `oracle`（`oracle_future_v1`）
- `dbw`（`depth_breadth_v1`）
- `kraken_vomm`（`kraken_vomm_v1`）
- `kpa`（`kpa_v1`）

暂不纳入当前对比：
- `xanadu_v1`（旧版，代码保留）
- `hist_keepalive_prewarm_v1`
- `no_autoscale_v1`

## 3. 场景与重复设置
- 负载场景：`low/mid/high = 0.5x / 1.0x / 2.0x`
- 默认 seeds：`42,43,44`
- 聚合口径：每个 `场景-策略` 输出 `mean/std`

## 4. 指标定义
- `e2e_bundle`：`avg_e2e_ms / p95_ms / p99_ms`
- `cold_start_step_rate`：`cold_start_steps / (cold_start_steps + warm_reuse_steps)`
- `predictor_prewarm_utilization`：`prewarm_consumed / prewarm_created`
- `prewarm_cost`：`prewarm_created`（扩容容器数）

## 5. 统一脚本与局部更新
脚本：`analysis/experiments/compare_experiments.py`

支持按子集局部重跑：
- `--metrics`：只更新指定指标
- `--scenarios`：只重跑指定负载档位
- `--autoscalers`：只重跑指定策略

局部更新规则：
- 仅重跑你指定的子集组合；
- 新结果按 `(scenario,strategy)` 键合并到历史快照；
- 未命中的历史结果保留，不会被全量清空。

## 6. 输出目录（统一）
- 汇总快照：`results/compare/compare_metrics.csv`
- 指标数据：`results/compare/metrics/*.csv`
- 图像输出：`results/compare/figures/*.png`

## 7. 常用命令
```powershell
# 全量（默认方法 + 全场景 + 全指标）
python analysis/experiments/compare_experiments.py --configs configs/default.yaml

# 局部：只更新 low 场景的 ConScale/KPA 的扩容成本
python analysis/experiments/compare_experiments.py --configs configs/default.yaml --scenarios low --autoscalers hpwp_v1 kpa_v1 --metrics prewarm_cost

# 仅重绘（不重跑仿真）
python analysis/plotting/plot_metrics_from_csv.py
python analysis/plotting/plot_metrics_from_csv.py --metrics e2e_bundle cold_start_step_rate
```
