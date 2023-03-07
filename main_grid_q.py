from algorithms.q_learning import QLearning
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
    plot_learning_curve,
)

# Gird World
env = GridWorld()

num_episodes: int = 100

q = QLearning(env)
q.train(num_episodes)
plot_gridworld_trajectory(q)
plot_gridworld_policy(q)
plot_learning_curve(q)
