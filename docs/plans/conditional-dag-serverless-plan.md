# 条件 DAG Serverless 平台当前执行方案（无队列 + 物理节点）

## 摘要

- 当前平台采用“物理节点 + 容器即时调度”执行模型。
- 请求到达后不排队，只允许两种行为：复用空闲容器或冷启动新容器。
- 容量不足时立即失败（`capacity_exhausted`）。
- 单请求始终沿 DAG 执行一条 `root -> leaf` 路径。

## 执行模型

- 调度：
  - 策略为 `least_load`。
  - 优先复用同函数空闲容器；无空闲时在最低负载物理节点冷启动。
- 容器：
  - 单容器单任务并发（并发固定为 1）。
  - 状态：`COLD_STARTING / RUNNING / IDLE`。
  - 空闲超过 `idle_ttl_sec` 回收。
- 物理节点：
  - 同构节点，包含 CPU、内存与容器集合。
  - 资源不足时拒绝分配新容器。

## 时延口径

- 每个步骤时延由三部分组成：
  - 冷启动时延（仅冷启动步骤）
  - 数据传输时延（前驱输出跨节点传输）
  - 执行时延（按分配 CPU 与计算量推进）
- 请求总时延按步骤累计：
  - `total = cold_start + data_transfer + execution`
- 若失败则记录失败原因，不走排队补偿。

## 数据与路径

- DAG 结构来源：`data/raw/filtered_tasks.csv`（唯一结构 Top-K，默认 20）。
- 负载来源：`Real-World-App-Emulation` 的 `invokesCDF + CVs`。
- 路径生成：
  - 规则：`mode_prefix_coupled_v1`
  - 上下文模式：`fixed`（默认）或 `drifting`（可切换）
  - 请求内前缀分支会影响后续分支选择。

## 产物与追踪

- 每轮实验输出目录：
  - `runs/<timestamp>_<scheduler>_<autoscaler>/`
- 关键文件：
  - `config.snapshot.yaml`
  - `scheduler_decisions.csv`
  - `node_metrics.csv`
  - `request_paths.csv`
  - `summary.json`
- `summary.json` 包含：
  - `dag_selection`
  - `workload_replay_profile`
  - `path_model`
