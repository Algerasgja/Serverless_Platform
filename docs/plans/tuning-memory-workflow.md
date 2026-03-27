# 调参与机制变更记忆流程（快速迭代版）

## 目标
为避免每轮从零开始试错，建立统一的“可检索调参记忆”：
1. 记录每一轮的策略参数、机制改动、结果快照。
2. 保留平台配置指纹，确保可复现和可比。
3. 下一轮直接按目标检索历史最优候选，缩短迭代时间。

## 工具
新增脚本：`analysis/tuning_memory.py`

支持两个命令：
1. `record`：记录当前轮次。
2. `suggest`：按目标从历史轮次中推荐 Top-N。

## 记录一轮（record）
示例（记录 Kraken 当前轮次）：

```powershell
python analysis/tuning_memory.py record `
  --strategy kraken_vomm `
  --round-id kraken_mech_r1 `
  --config configs/default.yaml `
  --compare-csv results/compare/compare_metrics.csv `
  --mechanism-change "uniform_mix + pressure_gate + active_gate" `
  --notes "目标: Kraken整体比ConScale慢约10%"
```

输出：
1. `results/compare/tuning_memory/kraken_vomm.jsonl`
2. `results/compare/tuning_memory/index.csv`

每条记录包含：
1. `strategy_params`（当前策略参数快照）
2. `metrics_by_scenario`（low/mid/high 指标）
3. `aggregate_metrics`（跨场景均值）
4. `gap_to_hpwp`（相对 ConScale 的差距）
5. `platform_hash`（平台配置指纹，防止跨平台误比）
6. `mechanism_change / notes`（机制和调参意图）

## 历史推荐（suggest）
示例 1：找“最接近比 ConScale 慢 10%”的历史轮次

```powershell
python analysis/tuning_memory.py suggest `
  --strategy kraken_vomm `
  --objective near_gap_10 `
  --target-gap 10 `
  --top 5
```

示例 2：找延迟最小轮次

```powershell
python analysis/tuning_memory.py suggest `
  --strategy kraken_vomm `
  --objective min_avg_e2e `
  --top 5
```

示例 3：找冷启动率最低轮次

```powershell
python analysis/tuning_memory.py suggest `
  --strategy kraken_vomm `
  --objective min_cold_rate `
  --top 5
```

## 推荐执行规范
1. 每次改动后都先跑完整三档负载（`low/mid/high`）。
2. 跑完立刻 `record`，确保“参数-机制-结果”一一对应。
3. 下一轮调参前先 `suggest`，优先从历史近邻点出发。
4. 若 `platform_hash` 变化，视为不同实验域，不直接横向比较。
5. 每轮只改一个策略，避免归因混淆。

## 迭代收益
使用该流程后，可把“盲调”转成“有记忆的局部搜索”：
1. 减少重复尝试相同参数区间。
2. 降低历史信息丢失造成的回退成本。
3. 提高“定向目标（如 +10% 劣化）”的命中速度。

