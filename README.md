# 条件 DAG Serverless 仿真平台

本项目是一个本地 Python 仿真平台，用于 Serverless 扩缩容实验，核心能力包括：
- 条件 DAG 单路径执行
- 条件路径生成（`fixed`/`drifting` 上下文模式）
- 最低负载调度（`least_load`）
- 无队列执行语义（容量不足立即失败）
- 实验结果持久化，支持多策略对比

## 快速开始

1. 安装依赖

```powershell
python -m pip install -e .
```

2. 运行仿真

```powershell
python -m simulator.main --config configs/default.yaml
```

3. 查看结果目录

```text
runs/<timestamp>_<scheduler>_<autoscaler>/
```

4. 生成图表

```powershell
python analysis/plotting/plot.py timeseries --run-dir runs/<timestamp>_least_load_kpa_v1
python analysis/plotting/plot.py latency-breakdown --run-dirs runs/<timestamp>_least_load_kpa_v1
python analysis/plotting/plot.py e2e-compare --run-dirs runs/<runA> runs/<runB> --out e2e_compare.png
python analysis/experiments/compare_experiments.py --configs configs/default.yaml
python analysis/experiments/ablation_experiments.py --config configs/default.yaml
python analysis/experiments/hparam_experiments.py --config configs/default.yaml
```

实验脚本按实验类型输出到 `results/` 下的分类目录，并直接覆盖同名结果文件。
当前默认路径：
- 对比实验：`results/compare/`
- 超参数实验：`results/hparam/latest/`
- 消融实验：`results/ablation/latest/`
其中对比实验已统一为单套并支持按指标/场景/方法做局部更新：
- 默认方法：`conscale(hpwp)/xunadu(=xanadu_opt_v1)/oracle/dbw/kraken_vomm/kpa`
- 已移出当前对比：`xanadu_v1`（旧版）、`hist`、`na`
- 不传 `--metrics`：执行全部指标并输出到 `results/compare/*`
- 传 `--metrics e2e_bundle cold_start_step_rate`：仅更新对应指标数据和图
- 传 `--scenarios low` 或 `--autoscalers hpwp_v1 kpa_v1`：仅重跑局部设置，并合并更新结果快照

示例：
```powershell
python analysis/experiments/compare_experiments.py --configs configs/default.yaml --metrics e2e_bundle
python analysis/experiments/compare_experiments.py --configs configs/default.yaml --scenarios low --autoscalers hpwp_v1 kpa_v1 --metrics prewarm_cost
# 仅基于已生成的 metrics CSV 重绘图像（不重新跑实验）
python analysis/plotting/plot_metrics_from_csv.py
python analysis/plotting/plot_metrics_from_csv.py --metrics e2e_bundle prewarm_cost
```

## 真实数据接入

`replay` 模式默认接入真实数据：
- DAG：`data/raw/filtered_tasks.csv`
- 频率/CV：`data/real-world-emulation/CDFs/invokesCDF.csv`、`data/real-world-emulation/CDFs/CVs.csv`

`replay` 模式下如果这些文件缺失会直接报错（不回退合成数据）。
`generative` 模式仍可在无真实数据时使用内置回退数据运行。

实验设置说明：
- `docs/plans/experimental-setup-real-data.md`
- `docs/plans/hpwp-experiment-suite.md`

默认 DAG 选择：
- `dataset.dag_selection_mode = random_unique`（在唯一结构集合中随机抽取）
- `dataset.dag_top_k = 20`
- `dag_policy.context_regime = fixed | drifting`（是否开启上下文漂移）

## 输出产物

每轮实验必含：
- `config.snapshot.yaml`
- `scheduler_decisions.csv`
- `node_metrics.csv`
- `request_paths.csv`
- `summary.json`

## 测试

```powershell
python -m unittest discover -s tests -v
```

