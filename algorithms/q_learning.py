from numbers import Real

import numpy as np

from algorithms.td0 import td0
from discrete_pendulum import Pendulum
from gridworld import GridWorld
from utils.eps_greedy import act


class QLearning:
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

            episode_return: float = 0

            done: bool = False

            while not done:
                s1: int
                r: Real
                a: int = act(s, Q, self.num_actions, self.eps)
                s1, r, done = self.env.step(a)
                Q[s, a] += alpha * (
                    r
                    + gamma * np.max([Q[s1, a] for a in range(self.num_actions)])
                    - Q[s, a]
                )

                s = s1

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
