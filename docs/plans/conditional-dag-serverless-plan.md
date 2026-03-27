# 条件 DAG Serverless 平台执行方案（当前版本）

## 1. 研究目标与问题定义
本平台面向条件 DAG 无服务器场景，核心研究问题为：在真实数据驱动负载下，不同扩缩容策略对端到端时延、成功率与扩容成本的影响。

平台采用“物理节点 + 容器即时调度 + 无队列失败语义”执行模型，强调与生产无服务器系统一致的关键约束：
- 请求按 DAG 依赖执行，前驱未完成（含传输）则后继不可启动。
- 容器单任务并发，优先复用，无法复用时冷启动。
- 无等待队列；资源不足即失败（`capacity_exhausted`）。

## 2. 执行架构

### 2.1 控制面
- 调度器：`least_load`
- 扩缩容策略（可插拔）：`hpwp_v1`、`kpa_v1`、`xanadu_v1`、`hist_keepalive_prewarm_v1`、`no_autoscale_v1`
- DAG 路径策略：`mode_prefix_coupled_v1`

### 2.2 数据面
- 物理节点：同构节点，包含 CPU、内存、容器集合。
- 容器状态：`COLD_STARTING / RUNNING / IDLE`。
- 容器并发：固定为单容器单任务。
- 资源释放：容器空闲超过 `idle_ttl_sec` 后回收。

## 3. 请求生命周期与调度语义

### 3.1 步骤级调度流程
对于请求路径中的每个函数步骤，系统按以下顺序处理：
1. 查找同函数的空闲容器，若存在则直接复用（`warm_reuse`）。
2. 若无空闲容器，则在最低负载物理节点冷启动容器（`cold_start`）。
3. 若无可用槽位或资源不足，立即失败（`capacity_exhausted`）。

### 3.2 DAG 依赖约束
- 一个请求仅执行一条 `root -> leaf` 路径。
- 后继函数只能在前驱函数“执行完成且输出传输完成”后启动。

## 4. 时延模型与失败模型

### 4.1 请求时延分解
每个步骤时延由三部分组成：
- 冷启动时延（仅冷启动步骤计入）
- 数据传输时延（仅跨节点传输计入）
- 执行时延（由函数计算量与分配 CPU 决定）

请求级时延累计规则：
- `total_latency_ms = cold_start_latency_ms + data_transfer_latency_ms + execution_latency_ms`

### 4.2 失败语义
- `capacity_exhausted`：步骤调度时资源不足，立即失败。
- `timeout`：请求总运行时间超过 `request_timeout_ms`。

## 5. 可复现性与产物

### 5.1 可复现机制
- 全局随机种子：`experiment.random_seed`
- 子模块偏移种子：负载、路径耦合、函数画像、带宽矩阵均使用固定 seed offset。
- 配置快照持久化：每次运行写入 `config.snapshot.yaml`。

### 5.2 运行产物
每次实验目录：`runs/<timestamp>_<scheduler>_<autoscaler>/`

必备文件：
- `config.snapshot.yaml`
- `scheduler_decisions.csv`
- `autoscaler_decisions.csv`
- `node_metrics.csv`
- `request_paths.csv`
- `summary.json`

## 6. 当前默认基线配置（摘要）
- 路径上下文：默认开启漂移（`context_regime=drifting`）
- 物理节点：`12` 节点、每节点 `8` 容器槽
- 空闲回收：`idle_ttl_sec=20`
- 全局实例上限：`max_total_instances=96`

上述默认值用于提高实验对扩缩容策略差异的敏感度，并减少“高冗余容量掩盖策略差异”的风险。
