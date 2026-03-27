# 对比算法参数调整经验（按算法分类）

本目录用于沉淀当前对比实验中各扩缩容算法的调参经验，目标是支持两类工作：

1. 快速复现实验时，按算法读取“先调哪些参数、再看哪些指标、如何判断是否过度”。
2. 需要做定向优化时，减少盲目网格搜索，把调参动作约束到最关键的少量参数。

当前覆盖算法（与 `analysis/compare_experiments.py` 默认对比集合一致）：

- `ConScale (hpwp_v1)`：[hpwp-conscale.md](./hpwp-conscale.md)
- `Xunadu (xanadu_opt_v1)`：[xanadu-opt.md](./xanadu-opt.md)
- `Kraken (kraken_vomm_v1)`：[kraken-vomm.md](./kraken-vomm.md)
- `LaSS (lass_v1)`：[lass.md](./lass.md)
- `QLAS (rl_q_v1)`：[rl-q.md](./rl-q.md)
- `HPTD (hptd_v1)`：[hptd.md](./hptd.md)
- `DBW (depth_breadth_v1)`：[depth-breadth.md](./depth-breadth.md)
- `KPA (kpa_v1)`：[kpa.md](./kpa.md)
- `Oracle (oracle_future_v1)`：[oracle.md](./oracle.md)

统一建议的调参流程：

1. 先固定负载场景（low/mid/high）和随机种子组（如 42/43/44）。
2. 每轮只改同一算法 1~2 个参数，避免归因混淆。
3. 主看四个指标：`avg_e2e_ms_mean`、`p95_ms_mean`、`prewarm_cost_mean`、`prewarm_utilization_mean`。
4. 若目标是“降低冷启动”，先看 `cold_start_step_rate_mean` 是否同步下降，避免仅靠增加容器数“硬压”延迟。
5. 每次改动后只局部重跑该算法并覆盖快照，再决定下一步。

