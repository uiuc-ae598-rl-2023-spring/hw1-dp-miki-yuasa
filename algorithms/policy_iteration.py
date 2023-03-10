from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld

Action = Literal[0, 1, 2, 3]


class PolicyIteration:
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

        # 1 Initialization
        V = np.random.rand(self.num_states)
        policy = np.random.randint(self.num_actions, size=self.num_states)

        V_history: list[float] = [np.mean(V)]

        iter_ind: int = 0
        while True:
            # 2 Policy Evaluation

            print("Policy Evaluation")
            while True:
                for s in range(self.num_states):
                    Delta: float = 0
                    v: float = V[s]
                    V[s] = np.sum(
                        [
                            p(s1, s, policy[s]) * (r(s, policy[s]) + gamma * V[s1])
                            for s1 in range(self.num_states)
                        ]
                    )
                    Delta = max(Delta, abs(v - V[s]))
                    print(
                        "Delta: {}, abs:{}, v:{}, V[s]:{}".format(
                            Delta, abs(v - V[s]), v, V[s]
                        )
                    )

                if Delta < self.theta:
                    break
                else:
                    pass

            print("Policy Improvement")
            # 3 Policy Improvement
            policy_stable: bool = True
            for s in range(self.num_states):
                old_action: Action = policy[s]
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
                print("old: {}, new: {}".format(old_action, policy[s]))
                if old_action != policy[s]:
                    policy_stable = False
                else:
                    pass

            V_history.append(np.mean(V))
            print(iter_ind)
            iter_ind += 1

            if policy_stable:
                break
            else:
                pass

        if plot_learning_curve:
            fig, ax = plt.subplots()
            ax.plot(V_history)
            ax.set_xlabel("Value Iteration [-]")
            ax.set_ylabel("Mean Value [-]")
            ax.grid(True)
            plt.savefig("figures/gridworld/learning_curve_pi.png", dpi=600)
        else:
            pass

        self.V = V
        self.policy = policy
        self.V_hisotry = V_history

        return V, policy
