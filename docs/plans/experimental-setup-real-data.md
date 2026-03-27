# 真实数据实验设置（DAG + 负载 + 条件路径）

## 1. 数据来源与口径

### 1.1 DAG 结构数据
- 文件：`data/raw/filtered_tasks.csv`
- 来源语义：Cluster-trace-v2018 任务级依赖记录
- 解析粒度：`job_name` 级 DAG

### 1.2 负载统计数据
- 文件：
  - `data/real-world-emulation/CDFs/invokesCDF.csv`
  - `data/real-world-emulation/CDFs/CVs.csv`
- 来源语义：ServerlessBench Real-World-App-Emulation 的调用频率与变异系数统计

### 1.3 缺失数据行为
当 `workload.mode=replay` 时，上述数据缺失将直接报错，不回退为合成数据。

## 2. DAG 接入与筛选流程

### 2.1 task_name 依赖解析
平台从 `task_name` 中提取“任务编号 + 依赖编号”，仅数字部分参与依赖构建；支持如下脏格式：
- `M5_3_4`
- `R4_2_Stg9`
- 尾部冗余下划线

非法格式行将记录并跳过，不中断整体构建。

### 2.2 唯一结构去重
同一 job 的节点和边先去重，再基于规范化签名做“唯一结构”聚合：
- 节点重映射（消除原始编号偏差）
- 边集排序后构建结构签名

### 2.3 结构排序与采样
支持两种模式：
- `topk_unique`：按复杂度优先选择
- `random_unique`：在唯一结构集合中按固定种子随机抽取

当前默认：
- `dag_selection_mode = random_unique`
- `dag_top_k = 20`
- `dag_selection_seed = 42`

## 3. replay 负载生成

### 3.1 先验参数采样
对每个 DAG 独立采样 `(avg_iat, cv)`：
- `avg_iat` 来自 invokes CDF
- `cv` 来自 CV CDF

### 3.2 到达过程
每个 DAG 使用独立续更新过程（renewal process）生成请求到达：
- IAT 分布：lognormal（由 `mean + cv` 推导）
- 时间精度：毫秒级
- 最终多 DAG 到达流按时间戳归并

### 3.3 负载缩放
`rate_multiplier` 直接作用于 IAT：
- `effective_iat_ms = base_iat_ms / rate_multiplier`
- `rate_multiplier <= 0` 时不产生请求

## 4. 条件路径生成与上下文模式

### 4.1 路径规则
- `dag_policy.path_rule = mode_prefix_coupled_v1`
- 单请求仅生成并执行一条路径

### 4.2 前缀影响后缀
同一请求内，前序分支选择会通过耦合项影响后续分支概率，形成上下文相关路径。

### 4.3 上下文模式
- `fixed`：模式分布全程固定
- `drifting`：模式分布随时间平滑漂移

当前默认：
- `context_drifting_enabled = true`
- `context_regime = drifting`

## 5. 可复现性约束
- 主随机种子：`experiment.random_seed`
- 子种子偏移：
  - `workload.realworld_seed_offset`
  - `dag_policy.coupling_seed_offset`
  - 其他资源模型 seed offset（函数画像/带宽）

同一配置与种子下，DAG 选择、请求到达与路径采样应保持可复现。

## 6. 运行产物与追踪字段
关键追踪文件：
- `request_paths.csv`：路径与时延分解
- `scheduler_decisions.csv`：步骤调度与冷启/复用决策
- `summary.json`：
  - `dag_selection`
  - `workload_replay_profile`
  - `path_model`

该设计用于保证“数据来源可追溯、生成过程可解释、实验结果可复现”。
