# QLAS（`rl_q_v1`）调参经验

## 核心思想
QLAS 使用简化 Q-learning 在线学习“状态-动作”映射，动作集合为 `{-1,0,+1}`，并通过奖励函数（吞吐/成本）逐步调整扩容行为。它适合做可学习基线，但在短实验窗口下对随机性较敏感。

## 关键参数与经验
- 学习相关：
  - `rl_learning_rate`
  - `rl_discount_factor`
  - `rl_epsilon_init`, `rl_epsilon_decay`, `rl_epsilon_min`
  经验：探索率过高会导致前期表现不稳定；衰减过快又会早收敛到次优策略。
- 动作相关：
  - `rl_step_size`
  - `rl_util_threshold`
  - `rl_inhibit_token_max`
  经验：步长太大易振荡，抑制 token 太高会错过补容时机。
- 奖励相关：
  - `rl_reward_tolerance`
  - `rl_scalability_alpha`
  经验：容忍带太宽会让学习信号变弱，太窄则容易过拟合噪声。

## 建议调参顺序
1. 先调探索策略（epsilon 三件套）让曲线可重复。
2. 再调动作步长与阈值，最后才调奖励细节。
3. 评价时至少看 3 seeds，避免单次结果误导。

