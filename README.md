#  Using reinforcement learning to teach robots how to walk
This project explores the impact of different learning rate strategies on the convergence of reinforcement learning models across various environments. In RL, finding an optimal policy involves training an agent to act within an environment to maximize cumulative rewards. We study the effect that the learning rate hyperparameter has on a model's rate of convergence.

We investigate fixed, dynamic, and adaptive learning rate strategies to determine their effectiveness in training RL models. Dynamic rates adjust over time, while adaptive rates modify based on the model's performance, in this context, the reward loss. Our study uses simple RL algorithms, such as DQNs and PPOs, across increasingly complex environments to assess how these strategies affect training efficiency and model performance.

The aim is to identify which learning rate strategy most effectively improves model performance and speeds up convergence to an optimal policy.

Keywords: Reinforcement Learning, Learning Rate, Optimization, Convergence

Technologies: Python, Stable-baselines3, Gymnasium, Optuna
