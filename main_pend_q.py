from algorithms.q_learning import QLearning
from discrete_pendulum import Pendulum
from utils.plot import (
    plot_learning_curve,
    plot_pendulum_trajectory,
)

# Gird World
env = Pendulum()

num_episodes: int = 200

q = QLearning(env, alpha=0.1, eps=0.05)
q.train(num_episodes)
plot_pendulum_trajectory(q)
plot_learning_curve(q)
