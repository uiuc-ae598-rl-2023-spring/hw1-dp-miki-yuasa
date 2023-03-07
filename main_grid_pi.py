from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration
from algorithms.sarsa import Sarsa
from algorithms.q_learning import QLearning
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
)

# Gird World
env = GridWorld()

pi = PolicyIteration(env, gamma=0.95)
pi.train()
plot_gridworld_trajectory(pi)
plot_gridworld_policy(pi)
