from numbers import Real
import random
import numpy as np
from algorithms.td0 import td0

from discrete_pendulum import Pendulum
from gridworld import GridWorld
from utils.eps_greedy import act


class Sarsa:
    def __init__(
        self,
        env: GridWorld | Pendulum,
        alpha: float = 0.5,
        gamma: float = 0.95,
        eps: float = 0.1,
    ) -> None:
        self.env: GridWorld = env

        self.alpha: float = alpha
        self.gamma: float = gamma
        self.eps: float = eps

        self.num_states: int = env.num_states
        self.num_actions: int = env.num_actions
        self.max_num_steps: int = env.max_num_steps

    def train(self, num_episodes: int):
        Q = np.zeros([self.num_states, self.num_actions])
        alpha: float = self.alpha
        gamma: float = self.gamma

        returns: list[float] = []

        for episode in range(num_episodes):
            s: int = self.env.reset()
            a: int = act(s, Q, self.num_actions, self.eps)

            episode_return: float = 0

            for _ in range(self.max_num_steps):
                s1: int
                r: Real
                done: bool
                s1, r, done = self.env.step(a)
                a1 = act(s1, Q, self.num_actions, self.eps)
                Q[s, a] += alpha * (r + gamma * Q[s1, a1] - Q[s, a])

                s = s1
                a = a1

                episode_return += r

                if done:
                    break
                else:
                    pass

            returns.append(episode_return)

        V = td0(self.env, 1000, Q, alpha, gamma)

        self.Q = Q
        self.V = V
        self.returns = returns
        self.policy = np.array(
            [act(s, self.Q, self.num_actions, self.eps) for s in range(self.num_states)]
        )

        return Q, V, returns
