import random
import numpy as np


def act(s: int, Q: np.ndarray, num_actions: int, eps: float) -> int:
    action: int = (
        random.randint(0, num_actions-1) if random.random() < eps else np.argmax(Q[s])
    )

    return action
