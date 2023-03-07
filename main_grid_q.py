from algorithms.q_learning import QLearning
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
    plot_learning_curve,
)

# Gird World
env = GridWorld()

num_episodes: int = 500

q = QLearning(env, alpha=0.3, eps=0.1)
q.train(num_episodes)
plot_gridworld_trajectory(q)
plot_gridworld_policy(q)
plot_learning_curve(q)
