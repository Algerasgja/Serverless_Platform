# 条件路径上下文生成与模式设置说明

## 1. 文档目的
本文档用于说明当前平台中“上下文（context）”的来源、生成机制、作用路径，以及当前默认模式与配置行为。重点回答两个问题：
1. 上下文是如何被构造并影响路径选择的。
2. 当前系统默认采用哪种上下文模式，配置项如何生效。

## 2. 上下文的定义与建模边界

### 2.1 当前实现中的上下文定义
在 `path_rule = mode_prefix_coupled_v1` 下，上下文由两层组成：
- 全局模式上下文（mode-level）：同一 DAG 模板在当前时刻的模式分布 `pi`。
- 请求内前缀上下文（in-request prefix）：当前请求此前分叉决策序列对后续分叉的耦合影响。

### 2.2 不属于当前上下文的部分
- 在 `mode_prefix_coupled_v1` 下，路径采样不使用 `session_id` 的历史边计数。
- 旧的 `critical_path` 路径规则才会使用会话级 `edge_counts`（历史转移偏好）。

换言之，当前主路径规则的上下文是“模式 + 请求内前缀”，不是“跨请求会话记忆”。

## 3. 上下文生成总流程（mode_prefix_coupled_v1）

### 3.1 离线初始化：为每个 DAG 模板构建路径模型
对每个 UM（DAG 模板）构建 `DagPathModel`，核心对象包括：
- `split_nodes`：所有分叉节点（出度 > 1，按拓扑顺序）
- `branches_by_split`：每个分叉的候选后继
- `base_probs_by_split`：来自 DAG 转移概率的基础概率
- `theta[m][s][b]`：模式 `m` 在分叉 `s` 对分支 `b` 的偏好项
- `coupling[(i,j)][bi][bj]`：前缀分叉选择对后续分叉选择的耦合项
- `pi_fixed`：模式先验分布（Dirichlet 采样）
- `pi_cur/pi_target`：漂移模式的当前分布与目标分布

### 3.2 在线采样：单请求路径生成
每次请求到达后，路径按以下顺序生成：
1. 采样模式 `m`（来自 `pi_fixed` 或 `pi_cur`）。
2. 从 `__start__` 开始遍历 DAG。
3. 对非分叉节点：确定性前进。
4. 对分叉节点：对每个候选分支计算打分并经 softmax 采样。
5. 到达叶子结束，得到一条 `root -> leaf` 路径。

保证：单请求始终只产生一条执行路径。

## 4. 分叉打分机制（上下文如何起作用）
在分叉节点 `s_j` 对候选分支 `b` 的打分为：

`score = log(base_prob + eps) + mode_strength * theta + prefix_strength * prefix_term`

其中：
- `base_prob`：DAG 中该分支的基础概率。
- `theta`：模式对该分支的偏好强度。
- `prefix_term`：请求内前缀影响项。

`prefix_term` 由已选前缀分叉累加得到，仅取最近 `prefix_window` 个分叉，并按分叉间距 `gap` 做衰减：

`prefix_term = Σ (prefix_decay^gap * coupling)`

然后对所有候选分支做 softmax（温度为 `temperature`）得到采样概率。

## 5. 上下文模式（fixed vs drifting）

### 5.1 fixed 模式
- 模式分布：`pi_dag(t) = pi_fixed`（运行全程不变）
- 特征：调用模式稳定，主要随机性来自请求内前缀耦合和分支采样。

### 5.2 drifting 模式
- 模式分布：`pi_dag(t) = pi_cur(t)`，随时间平滑变化。
- 刷新机制：
  - 当 `now_sec >= next_refresh_sec` 时，生成新 `pi_target`：
    `pi_target ~ Dirichlet(drifting_concentration * pi_cur + drifting_floor)`
- 平滑逼近：
  - `dt = max(1, now_sec - last_update_sec)`
  - `blend = 1 - (1 - drifting_strength)^dt`
  - `pi_cur = normalize((1-blend)*pi_cur + blend*pi_target)`

该设计用于模拟真实业务中上下文模式随时间缓慢迁移的现象，而非突变。

## 6. 当前模式设置（代码与配置）

### 6.1 当前默认设置
当前平台默认已开启漂移：
- `dag_policy.path_rule = mode_prefix_coupled_v1`
- `dag_policy.context_drifting_enabled = true`
- `dag_policy.context_regime = drifting`

默认参数（`configs/default.yaml`）：
- `mode_count = 3`
- `mode_prior_concentration = 2.0`
- `mode_strength = 1.0`
- `prefix_strength = 1.2`
- `prefix_decay = 0.85`
- `prefix_window = 3`
- `temperature = 1.0`
- `drifting_interval_sec = 30`
- `drifting_strength = 0.08`
- `drifting_concentration = 200.0`
- `drifting_floor = 0.001`
- `eps = 1e-9`

### 6.2 配置生效优先级
配置加载层对 `context_regime` 与 `context_drifting_enabled` 做一致化处理：
- 若显式设置 `context_drifting_enabled`：以该开关为准并覆盖 `context_regime`。
- 若未设置该开关：根据 `context_regime` 推导开关值。
- 非法 `context_regime` 会被强制回退为 `fixed`。

### 6.3 路径规则兼容说明
- 当 `path_rule=mode_prefix_coupled_v1` 时，`session_continue_prob/context_alpha` 不参与路径生成（会给出忽略告警）。
- 仅在 `critical_path` 规则下，会话历史边计数才参与下一跳偏置。

## 7. 可观测性与结果追踪
运行结束后，`summary.json.path_model` 会输出：
- 当前 `path_rule` 与 `context_regime`
- 上下文参数快照
- 每个 UM 的模型摘要（如 `split_count`、`pi_fixed`）
- 在 drifting 模式下还包含 `pi_initial` 与 `pi_current`

该输出用于验证“上下文是否按配置启用、是否发生漂移、漂移幅度是否合理”。

## 8. 设计含义与使用建议
- 若研究“稳定业务模式”下算法表现，使用 `fixed`。
- 若研究“概念漂移/模式迁移”下鲁棒性，使用 `drifting`（当前默认）。
- 若需要跨请求长期记忆影响路径，请切换或扩展 `critical_path` 路线；当前主路径规则不包含跨请求会话历史。

---
对应实现参考：
- `simulator/dag/engine.py`
- `simulator/config.py`
- `configs/default.yaml`
