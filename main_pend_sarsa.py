from matplotlib import pyplot as plt
from algorithms.sarsa import Sarsa
from algorithms.q_learning import QLearning
from discrete_pendulum import Pendulum
from utils.plot import (
    plot_learning_curve,
    plot_pendulum_trajectory,
)

plt.rcParams["font.family"] = "Times New Roman"

# Gird World
env = Pendulum()

num_episodes: int = 200
sarsa = Sarsa(env)
sarsa.train(num_episodes)
plot_pendulum_trajectory(sarsa)
plot_learning_curve(sarsa)
