# 实验设置（当前模式）

## 1. 负载产生

- 负载模式：`workload.mode = replay`
- 输入数据：
  - `data/real-world-emulation/CDFs/invokesCDF.csv`
  - `data/real-world-emulation/CDFs/CVs.csv`
- 生成流程：
  1. 每个 DAG 独立采样 `(avg_iat, cv)`。
  2. 以 `lognormal` 生成 IAT 序列，毫秒级推进到达事件。
  3. 按 `rate_multiplier` 缩放强度：
     - `effective_iat_ms = base_iat_ms / rate_multiplier`
- 可复现性：
  - 每个 DAG 使用独立子种子：
    - `experiment.random_seed + workload.realworld_seed_offset + dag_index`

## 2. DAG 结构来源

- 数据文件：`data/raw/filtered_tasks.csv`
- 解析规则：
  - 按 `job_name` 聚合 job 级 DAG。
  - 从 `task_name` 抽取数字任务号与依赖号，脏格式行可跳过并计数。
- 结构选择：
  - 唯一结构去重（同构签名）。
  - 排序：`node_count DESC -> edge_count DESC -> support_count DESC -> signature ASC`
  - 取 Top-K，默认 `dataset.dag_top_k = 20`
- 模板命名：`dag_0001 ... dag_0020`

## 3. 请求路径与上下文模式

- 路径规则：`dag_policy.path_rule = mode_prefix_coupled_v1`
- 请求语义：每个请求仅生成并执行一条 `root -> leaf` 路径。
- 路径打分：
  - `score = log(base_prob + eps) + mode_strength * theta + prefix_strength * prefix_term`
  - `prefix_term` 来自“同一请求”已选前缀分支，按 `prefix_decay` 衰减，窗口 `prefix_window`

### 3.1 fixed（当前默认）

- `dag_policy.context_regime = fixed`
- 每个 DAG 的模式先验 `pi_fixed` 在整次运行中保持不变。
- 体现“同 DAG 调用模式稳定”，同时保持请求内前缀影响后缀。

### 3.2 drifting（可切换）

- `dag_policy.context_regime = drifting`
- 每个 DAG 维护 `pi_cur` 与 `pi_target`：
  - 按 `drifting_interval_sec` 刷新目标分布
  - 按 `drifting_strength` 平滑逼近目标
- 体现“调用模式随运行逐步变化”。

## 4. 当前关键配置

`configs/default.yaml`：

```yaml
dataset:
  dag_tasks_file: data/raw/filtered_tasks.csv
  dag_top_k: 20

workload:
  mode: replay
  invokes_cdf_file: data/real-world-emulation/CDFs/invokesCDF.csv
  cvs_cdf_file: data/real-world-emulation/CDFs/CVs.csv
  rate_multiplier: 0.0001
  realworld_seed_offset: 303

dag_policy:
  path_rule: mode_prefix_coupled_v1
  context_regime: fixed
  mode_count: 3
  mode_strength: 1.0
  prefix_strength: 1.2
  prefix_decay: 0.85
  prefix_window: 3
```

## 5. 结果产物

- 输出目录：`runs/<timestamp>_<scheduler>_<autoscaler>/`
- 当前重点文件：
  - `request_paths.csv`（最终执行路径与时延）
  - `summary.json`（含 `dag_selection`、`workload_replay_profile`、`path_model`）
