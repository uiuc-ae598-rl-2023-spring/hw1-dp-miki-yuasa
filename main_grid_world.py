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

pi = PolicyIteration(env)
pi.train()
plot_gridworld_trajectory(pi)
plot_gridworld_policy(pi)


vi = ValueIteration(env)
vi.train()
plot_gridworld_trajectory(vi)
plot_gridworld_policy(vi)

num_episodes: int = 100
sarsa = Sarsa(env)
sarsa.train(num_episodes)
plot_gridworld_trajectory(sarsa)
plot_gridworld_policy(sarsa)
plot_learning_curve(sarsa)

q = QLearning(env)
q.train(num_episodes)
plot_gridworld_trajectory(q)
plot_gridworld_policy(q)
plot_learning_curve(q)
