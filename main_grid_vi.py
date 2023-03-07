from matplotlib import pyplot as plt
from algorithms.value_iteration import ValueIteration
from gridworld import GridWorld
from utils.plot import (
    plot_gridworld_policy,
    plot_gridworld_trajectory,
)

plt.rcParams["font.family"] = "Times New Roman"

# Gird World
env = GridWorld()


vi = ValueIteration(env)
vi.train()
plot_gridworld_trajectory(vi)
plot_gridworld_policy(vi)
