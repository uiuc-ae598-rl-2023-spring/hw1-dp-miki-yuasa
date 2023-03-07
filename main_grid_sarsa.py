from matplotlib import pyplot as plt
from algorithms.sarsa import Sarsa
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
    plot_learning_curve,
)

env = GridWorld()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.left"] = 0.15


num_episodes: int = 5000
sarsa = Sarsa(env)
sarsa.train(num_episodes)
plot_gridworld_trajectory(sarsa)
plot_gridworld_policy(sarsa)
plot_learning_curve(sarsa, 100)
