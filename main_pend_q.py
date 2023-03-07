from matplotlib import pyplot as plt
from algorithms.q_learning import QLearning
from discrete_pendulum import Pendulum
from utils.plot import (
    plot_learning_curve,
    plot_pendulum_trajectory,
)

# Gird World
plt.rcParams["font.family"] = "Times New Roman"

env = Pendulum()

num_episodes: int = 200
alphas = [0.25, 0.5, 0.75]

q = QLearning(env)
q.train(num_episodes)
plot_pendulum_trajectory(q)
plot_learning_curve(q)
