from matplotlib import pyplot as plt
from algorithms.sarsa import Sarsa
from algorithms.q_learning import QLearning
from discrete_pendulum import Pendulum
from utils.plot import (
    plot_gridworld_policy,
    plot_learning_curve,
    plot_pendulum_trajectory,
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.left"] = 0.15
# Gird World
env = Pendulum()

num_episodes: int = 200
sarsa = Sarsa(env)
sarsa.train(num_episodes)
plot_pendulum_trajectory(sarsa)
plot_learning_curve(sarsa, 5)
plot_gridworld_policy(sarsa)