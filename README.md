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
python analysis/plot.py timeseries --run-dir runs/<timestamp>_least_load_hpa_v1
python analysis/plot.py latency-breakdown --run-dirs runs/<timestamp>_least_load_hpa_v1
```

## 真实数据接入

`replay` 模式默认接入真实数据：
- DAG：`data/raw/filtered_tasks.csv`
- 频率/CV：`data/real-world-emulation/CDFs/invokesCDF.csv`、`data/real-world-emulation/CDFs/CVs.csv`

`replay` 模式下如果这些文件缺失会直接报错（不回退合成数据）。
`generative` 模式仍可在无真实数据时使用内置回退数据运行。

实验设置说明：
- `docs/plans/experimental-setup-real-data.md`

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

