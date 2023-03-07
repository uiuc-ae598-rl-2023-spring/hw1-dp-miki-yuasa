from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration
from algorithms.sarsa import Sarsa
from algorithms.q_learning import QLearning
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
    plot_learning_curve,
)

# Gird World
env = GridWorld()


vi = ValueIteration(env)
vi.train()
plot_gridworld_trajectory(vi)
plot_gridworld_policy(vi)
