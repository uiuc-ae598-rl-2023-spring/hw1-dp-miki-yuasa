from matplotlib import pyplot as plt
from algorithms.q_learning import QLearning
from discrete_pendulum import Pendulum
from utils.plot import (
    plot_gridworld_policy,
    plot_learning_curve,
    plot_pendulum_trajectory,
)

# Gird World
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.left"] = 0.18
env = Pendulum()

num_episodes: int = 200

q = QLearning(env)
q.train(num_episodes)
plot_pendulum_trajectory(q)
plot_learning_curve(q, 5)
plot_gridworld_policy(q)
