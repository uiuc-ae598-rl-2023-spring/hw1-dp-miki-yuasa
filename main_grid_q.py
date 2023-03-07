from matplotlib import pyplot as plt
from algorithms.q_learning import QLearning
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
    plot_learning_curve,
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.subplot.bottom"] = 0.15
# Gird World
env = GridWorld()

num_episodes: int = 5000

q = QLearning(env)
q.train(num_episodes)
plot_gridworld_trajectory(q)
plot_gridworld_policy(q)
plot_learning_curve(q, 100)
