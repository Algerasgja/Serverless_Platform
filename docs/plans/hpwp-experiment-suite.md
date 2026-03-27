# HPWP 实验套件总览（对比 + 消融 + 超参数）

## 1. 文档结构
为保证论文写作与复现实验的一致性，HPWP 实验套件拆分为三类独立设计文档：
- 对比实验设计：`docs/plans/compare-experiment-design.md`
- 消融实验设计：`docs/plans/hpwp-ablation-design.md`
- 超参数实验设计：`docs/plans/hpwp-hyperparameter-design.md`

本文件仅作为总览入口与执行导航，不再承载详细实验细节。

## 2. 实验类型与目标
- 对比实验：统一单套基线
  - `conscale(hpwp)/xunadu(=xanadu_opt)/oracle/dbw/kraken_vomm/kpa`
  - 当前不纳入：`xanadu_v1`、`hist`、`no_as`
- 消融实验：验证 HPWP 各组成机制对性能增益的边际贡献。
- 超参数实验：在受控参数空间内识别稳定有效的参数组合，并生成最佳配置快照。

## 3. 脚本入口
- 对比：`analysis/compare_experiments.py`
- 消融：`analysis/ablation_experiments.py`
- 超参数：`analysis/hparam_experiments.py`

## 4. 默认输出目录
所有实验脚本统一输出到仓库根目录 `results/`（覆盖同名文件）：
- 对比：
  - 统一目录：`results/compare/metrics/*` 与 `results/compare/figures/*`
  - 汇总快照：`results/compare/compare_metrics.csv`
- 消融：`results/ablation_metrics.csv`、`results/ablation_pairwise.csv`、`results/ablation_e2e.png`、`results/ablation_gain.png`
- 超参数：`results/hparam_metrics.csv`、`results/hparam_tradeoff.png`、`results/hparam_best.yaml`

## 5. 推荐执行顺序
1. 先运行超参数实验，获得 `hparam_best.yaml`。
2. 再运行消融实验，冻结关键参数后分析机制贡献。
3. 最后运行对比实验，输出主结论图与基线比较结果。

## 6. 复现声明
上述三个实验共享同一平台执行语义与真实数据接入口径；详细的数据来源、路径生成、资源模型定义请参考：
- `docs/plans/experimental-setup-real-data.md`
- `docs/plans/conditional-dag-serverless-plan.md`
