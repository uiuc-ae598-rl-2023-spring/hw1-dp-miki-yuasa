import itertools
import os
import pickle
from multiprocessing import Pool

from matplotlib import pyplot as plt
from typing import Literal
from algorithms.q_learning import QLearning
from algorithms.sarsa import Sarsa
from gridworld import GridWorld
from discrete_pendulum import Pendulum
from utils.plot import plot_batch_lc

model_name: Literal["q", "sarsa"] = "sarsa"
env = GridWorld()
num_episodes: int = 500
alphas: list[float] = [0.7, 0.5, 0.3]
epss: list[float] = [0.15, 0.1, 0.05]
hyperparam_pairs: list[tuple[float, float]] = list(itertools.product(alphas, epss))

plt.rcParams["font.family"] = "Times New Roman"
env_name: str = "gridworld" if isinstance(env, GridWorld) else "pendulum"
models_file: str = "./models/models_{}_{}.pickle".format(env_name, model_name)


models: list[QLearning | Sarsa] = []

if os.path.isfile(models_file):
    with open(models_file, "rb") as f:
        models = pickle.load(f)

else:
    for alpha, eps in hyperparam_pairs:
        model = (
            QLearning(env, alpha=alpha, eps=eps)
            if model_name == "q"
            else Sarsa(env, alpha=alpha, eps=eps)
        )
        model.train(num_episodes)
        models.append(model)

    with open(models_file, "wb") as f:
        pickle.dump(models, f)

model_groups: list[list[QLearning | Sarsa]] = [models[0:3], models[3:6], models[6:]]

for i, models in enumerate(model_groups):
    plot_batch_lc(models, model_name, i)
