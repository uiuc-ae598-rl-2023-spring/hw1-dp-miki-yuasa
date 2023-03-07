import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from gridworld import GridWorld

Action = Literal[0, 1, 2, 3]


class ValueIteration:
    def __init__(
        self, env: GridWorld, gamma: float = 0.95, theta: float = 1e-3
    ) -> None:
        self.env: GridWorld = env
        self.gamma: float = gamma
        self.theta: float = theta

        self.hard_version: bool = env.hard_version
        self.num_states: int = env.num_states
        self.num_actions: int = env.num_actions
        self.last_action: Action = env.last_action
        self.max_num_steps: int = env.max_num_steps

    def train(self, plot_learning_curve: bool = True):
        p = self.env.p
        r = self.env.r
        gamma: float = self.gamma

        V = np.zeros(self.num_states)

        V_history: list[float] = [np.mean(V)]

        while True:
            Delta: float = 0
            for s in range(self.num_states):
                v: float = V[s]
                V[s] = np.max(
                    [
                        np.sum(
                            [
                                p(s1, s, a) * (r(s, a) + gamma * V[s1])
                                for s1 in range(self.num_states)
                            ]
                        )
                        for a in range(self.num_actions)
                    ]
                )
                Delta = max(Delta, abs(v - V[s]))

            V_history.append(np.mean(V))

            if Delta < self.theta:
                break
            else:
                pass

        policy = np.zeros(self.num_states)

        for s in range(self.num_states):
            policy[s] = np.argmax(
                [
                    np.sum(
                        [
                            p(s1, s, a) * (r(s, a) + gamma * V[s1])
                            for s1 in range(self.num_states)
                        ]
                    )
                    for a in range(self.num_actions)
                ]
            )

        if plot_learning_curve:
            fig, ax = plt.subplots()
            ax.plot(V_history)
            ax.set_xlabel("Value Iteration [-]")
            ax.set_ylabel("Mean Value [-]")
            ax.grid(True)
            plt.title("Mean Value vs Iterations")
            plt.savefig("figures/gridworld/learning_curve_vi.png", dpi=600)
        else:
            pass

        self.V = V
        self.policy = policy
        self.V_history = V_history

        return V, policy
