from matplotlib import pyplot as plt
from algorithms.policy_iteration import PolicyIteration
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
)

plt.rcParams["font.family"] = "Times New Roman"
# Gird World
env = GridWorld()

pi = PolicyIteration(env, gamma=0.95)
pi.train()
plot_gridworld_trajectory(pi)
plot_gridworld_policy(pi)
