# DBW（`depth_breadth_v1`）调参经验

## 核心思想
DBW 是深度受限的广度扩容：从当前函数向后枚举窗口内可达节点，不看概率，按可达集合聚合期望容器数。优点是覆盖稳健、实现直接；缺点是容易在分叉 DAG 上过配。

## 关键参数与经验
- `dbw_horizon_boost`：
  经验：扩大可达窗口会显著增加成本，尤其在高分叉 DAG。
- `dbw_desired_scale`：
  经验：整体预算倍率。通常用于全局拉高/拉低成本曲线。
- `dbw_misallocation_ratio`：
  经验：用于模拟错配，建议在正式对比中保持较低，否则会放大无效扩容。

## 建议调参顺序
1. 先锁 `misallocation_ratio`，避免和算法本体混淆。
2. 用 `horizon_boost` 调覆盖范围，用 `desired_scale` 做细粒度修正。
3. 若 low 负载下成本过高，优先缩窗口而非直接降倍率。

