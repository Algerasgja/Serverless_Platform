# KPA（`kpa_v1`）调参经验

## 核心思想
KPA 是典型并发目标驱动策略，分稳态窗口与 panic 窗口两级控制。它不依赖 DAG 上下文，主要通过并发观测值与目标并发比值决定扩容速度，属于可解释、工程化成熟的通用基线。

## 关键参数与经验
- `kpa_target_concurrency`：
  经验：目标并发越低，期望副本越多，延迟更稳但成本更高。
- `kpa_stable_window_sec`, `kpa_panic_window_sec`：
  经验：稳态窗口越长越平滑，panic 窗口越短越激进。
- `kpa_panic_threshold`：
  经验：阈值越低越容易进入 panic 扩容。

## 建议调参顺序
1. 先用 `target_concurrency` 粗定成本档位；
2. 再调窗口长度平衡“响应速度 vs 抖动”；
3. 最后调 panic 阈值控制突发时的敏感性。

