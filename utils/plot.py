import random
from typing import Callable

from matplotlib import pyplot as plt
from numpy import ndarray
from algorithms.policy_iteration import PolicyIteration
from algorithms.sarsa import Sarsa
from algorithms.value_iteration import ValueIteration
from algorithms.q_learning import QLearning
from gridworld import GridWorld


def plot_gridworld_trajectory(
    model: PolicyIteration | ValueIteration | Sarsa | QLearning,
) -> None:
    policy: ndarray = model.policy

    title: str = (
        "SARSA"
        if isinstance(model, Sarsa)
        else "Q-learning"
        if isinstance(model, QLearning)
        else "Policy Iteration"
        if isinstance(model, PolicyIteration)
        else "Value Iteration"
    )

    tag: str = (
        "pi"
        if isinstance(model, PolicyIteration)
        else "vi"
        if isinstance(model, ValueIteration)
        else "sarsa"
        if isinstance(model, Sarsa)
        else "q_learning"
    )

    s = model.env.reset()
    done = False
    log: dict[str, list[int] | list[float]] = {
        "t": [0],
        "s": [s],
        "a": [],
        "r": [],
    }
    while not done:
        a = policy[s]
        (s, r, done) = model.env.step(a)
        log["t"].append(log["t"][-1] + 1)
        log["s"].append(s)
        log["a"].append(a)
        log["r"].append(r)

    t = log["t"]
    s = log["s"]
    a = log["a"]
    r = log["r"]
    fig, axs = plt.subplots(nrows=3, ncols=1)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axs[0].plot(t, s, label="s")
    axs[0].set_ylabel("State [-]")
    axs[1].plot(t[:-1], a, label="a")
    axs[1].set_ylabel("Action [-]")
    axs[2].plot(t[:-1], r, label="r")
    axs[2].set_ylabel("Reward [-]")
    axs[2].set_xlabel("Timestep")
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    plt.savefig("figures/gridworld/gridworld_trajectory_{}.png".format(tag), dpi=600)


def plot_gridworld_policy(
    model: PolicyIteration | ValueIteration | Sarsa | QLearning,
) -> None:
    tag: str = (
        "pi"
        if isinstance(model, PolicyIteration)
        else "vi"
        if isinstance(model, ValueIteration)
        else "sarsa"
        if isinstance(model, Sarsa)
        else "q_learning"
    )
    policy = model.policy
    V = model.V
    # Plot state-value function
    fig, axs = plt.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    axs[0].plot(V)
    axs[0].set_ylabel("State-value")
    axs[0].set_title("State-value function")
    # Plot policy
    axs[1].bar(range(model.env.num_states), policy)
    axs[1].set_xlabel("State")
    axs[1].set_ylabel("Action")
    axs[1].set_title("Policy")
    plt.savefig("figures/gridworld/policy_statevalue_{}.png".format(tag), dpi=600)


def plot_learning_curve(model: Sarsa | QLearning):
    directory: str = "gridworld" if isinstance(model.env, GridWorld) else "pendulum"
    title: str = "SARSA" if isinstance(model, Sarsa) else "Q-learning"

    fig, ax = plt.subplots()
    ax.plot(model.returns)
    ax.set_xlabel("Episode [-]")
    ax.set_ylabel("Return [-]")
    ax.grid(True)
    plt.savefig(
        "figures/{}/learning_curve_{}.png".format(directory, tag(model)), dpi=600
    )


def plot_pendulum_trajectory(model: Sarsa | QLearning) -> None:
    # Initialize simulation
    env = model.env
    s = model.env.reset()

    # Create log to store data from simulation
    log = {
        "t": [0],
        "s": [s],
        "a": [],
        "r": [],
        "theta": [
            env.x[0]
        ],  # agent does not have access to this, but helpful for display
        "thetadot": [
            env.x[1]
        ],  # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(env.num_actions)
        (s, r, done) = env.step(a)
        log["t"].append(log["t"][-1] + 1)
        log["s"].append(s)
        log["a"].append(a)
        log["r"].append(r)
        log["theta"].append(env.x[0])
        log["thetadot"].append(env.x[1])

    t = log["t"]
    s = log["s"]
    a = log["a"]
    r = log["r"]
    # Plot data and save to png file
    title: str = "SARSA" if isinstance(model, Sarsa) else "Q-learning"
    fig, ax = plt.subplots(4, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    ax[0].plot(t, s, label="s")
    ax[0].set_ylabel("State [-]")
    ax[1].plot(t[:-1], a, label="a")
    ax[1].set_ylabel("Action [-]")
    ax[2].plot(t[:-1], r, label="r")
    ax[2].set_ylabel("Reward [-]")
    ax[3].plot(log["t"], log["theta"])
    ax[3].plot(log["t"], log["thetadot"])
    ax[3].legend(["theta", "thetadot"])
    ax[3].set_xlabel("Timestep [-]")
    plt.savefig(
        "figures/pendulum/pendulum_trajectory_{}.png".format(tag(model)), dpi=600
    )


def tag(model: PolicyIteration | ValueIteration | Sarsa | QLearning) -> str:
    tag: str = (
        "pi"
        if isinstance(model, PolicyIteration)
        else "vi"
        if isinstance(model, ValueIteration)
        else "sarsa"
        if isinstance(model, Sarsa)
        else "q_learning"
    )
    return tag


def plot_batch_lc(models: list[QLearning | Sarsa], model_name, ind):
    directory: str = "gridworld" if isinstance(models[0].env, GridWorld) else "pendulum"

    fig, ax = plt.subplots()
    for model in models:
        ax.plot(
            model.returns,
            label=r"$\alpha$={}, $\epsilon$={}".format(model.alpha, model.eps),
        )
    ax.set_xlabel("Episode [-]")
    ax.set_ylabel("Return [-]")
    ax.set_ylim(0, 180)
    ax.grid(True)
    ax.legend(loc="lower center", ncol=3)
    plt.savefig(
        "figures/{}/learning_curve_barch_{}_{}.png".format(directory, model_name, ind),
        dpi=600,
    )
