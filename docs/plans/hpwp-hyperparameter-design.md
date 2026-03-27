# HPWP 超参数实验设计（论文式说明）

## 1. 研究目的
本实验用于识别 HPWP 在当前平台上的“高影响参数区域”，并输出可直接复用的最佳配置快照，为后续消融与对比实验提供固定参数基线。

## 2. 参数选择原则
参数筛选遵循“影响大、可解释、可稳定复现”三项原则，最终选择以下四类关键参数：
- 调度周期缩放：`hpwp_sched_eta_exec`
- 紧迫规划视野：`hpwp_horizon_alpha`
- 层次贝叶斯回退强度：`(hpwp_beta_hi, hpwp_beta_lo)`
- 稳态更新强度：`hpwp_alpha_stable`

## 3. 对比基线与候选空间

### 3.1 默认基线点（固定纳入）
`HPWP_DEFAULT_POINT`：
- `hpwp_sched_eta_exec = 0.5`
- `hpwp_horizon_alpha = 2.8`
- `hpwp_beta_hi = 60.0`
- `hpwp_beta_lo = 10.0`
- `hpwp_alpha_stable = 0.08`

此外固定纳入一个“历史最优锚点”（用于局部精调）：
- `hpwp_sched_eta_exec = 0.4`
- `hpwp_horizon_alpha = 1.5`
- `hpwp_beta_hi = 160.0`
- `hpwp_beta_lo = 24.0`
- `hpwp_alpha_stable = 0.08`

### 3.2 候选集合
- `hpwp_sched_eta_exec`: `[0.3, 0.35, 0.4, 0.45, 0.5, 0.6]`
- `hpwp_horizon_alpha`: `[1.2, 1.5, 1.8, 2.2, 2.8]`
- `(hpwp_beta_hi, hpwp_beta_lo)`: `[(60,10), (80,12), (120,18), (160,24), (200,30)]`
- `hpwp_alpha_stable`: `[0.05, 0.08, 0.10, 0.12]`

全组合规模：`6 × 5 × 5 × 4 = 600`。

## 4. 采样与复现策略

### 4.1 抽样规则
- 默认 `trial_count = 24`
- 固定包含 2 个锚点（当前基线 + 历史最优）
- 剩余点采用“局部精调 + 全局探索”混合抽样：
  - 约 50% 来自历史最优附近（局部池）
  - 约 50% 来自全局候选（全局池）
  - 若某池不足，名额自动溢出到另一池
- 采样种子：`sample_seed = 20260313`

### 4.2 重复运行
- 每个参数点运行 seeds：`42, 43, 44`
- 输出 `mean/std` 聚合结果

该设计在计算预算与覆盖广度之间取得平衡，同时保证可复现。

## 5. 排序与最优点定义

### 5.1 排序规则
按以下顺序排序：
1. `avg_e2e_ms_mean`（主目标，越小越好）
2. `p95_ms_mean`（次目标，越小越好）

### 5.2 最优点输出
排序第一的配置写入 `results/hparam_best.yaml`，并用于后续消融实验中的关键参数冻结。

## 6. 指标与可视化

### 6.1 指标文件
- `results/hparam_metrics.csv`
- 每行包含：聚合性能指标 + 对应超参数 + 运行状态

### 6.2 主图
- `results/hparam_tradeoff.png`
- 图意：`AE(Avg E2E)` 与 `P95` 的性能权衡关系
- 用途：观察参数点在性能空间中的分布、识别稳定优区与异常区

### 6.3 主图颜色与形状编码（实现口径）
当前主图由 `analysis/experiment_common.py` 中 `plot_tradeoff(...)` 生成，视觉编码规则如下：
- 横轴：`AE = avg_e2e_ms_mean`（越小越好）
- 纵轴：`P95 = p95_ms_mean`（越小越好）
- 点颜色（Color）：映射 `AS = hpwp_alpha_stable`
  - 颜色盘：`viridis`
  - 归一化：`Normalize(vmin=min(AS), vmax=max(AS))`
  - 右侧颜色条标签：`AS`
- 点形状（Marker）：映射 `B = (hpwp_beta_hi, hpwp_beta_lo)`
  - 先对所有 `B` 组合排序，再按顺序分配 marker（`o,s,^,D,P,X,v,<,>,h,8,p`）
- 点大小（Size）：映射 `CSS = cold_start_share_mean`
  - 线性缩放到 `[80, 260]`（样本内最小 CSS -> 80，最大 CSS -> 260）
- 特殊标记：
  - `BEST`：红色实心五角星（主次排序最优点）
  - `PF`：灰色虚线 Pareto 前沿

### 6.4 点名规则与右上角图例含义
#### 点名规则（你提到的“点名字函数”）
点名由 `plot_tradeoff(...)` 内部按当前输入行顺序生成：
- 规则：`T01, T02, ..., T24`（格式 `T{idx:02d}`）
- 该顺序来自 `hparam_experiments.py` 里对结果按 `(avg_e2e_ms_mean, p95_ms_mean)` 排序后的顺序。
- 图中仅显示短标签 `Txx`，完整参数仍在 `results/hparam_metrics.csv` 中查表。

#### 右上角图例（Legend）解释
右上角图例标题是 `B / Marks`，表示“`B` 参数组与点形状的映射关系”：
- `B40/8`、`B60/10`、`B80/12`、`B160/24`：
  - 前半段（如 `B40`、`B60`）对应 `beta_hi=40/60`；
  - 斜杠后数字（如 `/8`、`/10`）对应 `beta_lo=8/10`；
  - 整体是一个 `(beta_hi, beta_lo)` 组合，并映射到一种点形状。
- `BEST`：全体候选里排序第一点（最优点）。
- `PF`：Pareto Front（在 AE 与 P95 双目标下不可支配前沿）。

#### 为什么要用不同形状标注点
使用不同形状是为了把“同一图中颜色与大小之外的第三个维度”编码出来，避免参数信息重叠：
- 颜色已用于编码 `AS`；
- 大小已用于编码 `CSS`；
- 形状用于编码 `B=(beta_hi,beta_lo)`，能直接观察不同贝叶斯回退强度组合在 `AE-P95` 空间中的分布差异与聚类趋势。

## 7. 结果输出与用途
本实验产出三类文件：
- 参数评估明细：`hparam_metrics.csv`
- 权衡主图：`hparam_tradeoff.png`
- 最佳配置快照：`hparam_best.yaml`

其中 `hparam_best.yaml` 为后续消融实验的参数冻结来源，是实验链路的关键衔接文件。

## 8. 复现实验命令
```powershell
python analysis/hparam_experiments.py --config configs/default.yaml
```

可选：
```powershell
python analysis/hparam_experiments.py --config configs/default.yaml --trial-count 24 --sample-seed 20260313
python analysis/hparam_experiments.py --config configs/default.yaml --seeds 42 43 44
```
