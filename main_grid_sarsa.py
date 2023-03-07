from algorithms.sarsa import Sarsa
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
    plot_learning_curve,
)

env = GridWorld()

num_episodes: int = 100
sarsa = Sarsa(env)
sarsa.train(num_episodes)
plot_gridworld_trajectory(sarsa)
plot_gridworld_policy(sarsa)
plot_learning_curve(sarsa)
