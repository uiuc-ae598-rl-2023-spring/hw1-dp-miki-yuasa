import numpy as np
from discrete_pendulum import Pendulum
from gridworld import GridWorld


def td0(
    env: GridWorld | Pendulum,
    num_episodes: int,
    Q: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    V = np.zeros(env.num_states)

    for _ in range(num_episodes):
        s: int = env.reset()

        for _ in range(env.max_num_steps):
            a: int = np.argmax(Q[s])
            s1, r, done = env.step(a)
            V[s] += alpha * (r + gamma * V[s1] - V[s])
            s = s1

            if done:
                break
            else:
                pass

    return V
