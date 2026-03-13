# Raw Dataset Inputs (Current Mode)

当前 `replay` 模式仅依赖以下原始数据：

- `filtered_tasks.csv`
  - 路径：`data/raw/filtered_tasks.csv`
  - 用途：构建 DAG 结构并做唯一结构 Top-K 选择

- `invokesCDF.csv`
  - 路径：`data/real-world-emulation/CDFs/invokesCDF.csv`
  - 用途：采样平均调用间隔（IAT）

- `CVs.csv`
  - 路径：`data/real-world-emulation/CDFs/CVs.csv`
  - 用途：采样调用间隔变异系数（CV）

说明：

- 旧的 `MSCallGraph/MSRTQps/Node` 数据文件在当前模式下不再使用。
- `workload.mode = replay` 时，如果上述 3 个文件任一缺失，程序会直接报错。
