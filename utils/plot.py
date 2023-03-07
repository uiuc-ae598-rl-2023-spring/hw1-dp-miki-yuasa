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
    fig, ax = plt.subplots()
    ax.plot(t, s, label="s")
    ax.plot(t[:-1], a, label="a")
    ax.plot(t[:-1], r, label="r")
    ax.legend()
    ax.grid(True)
    plt.title("{} Trajectory".format(title))
    plt.savefig("figures/gridworld/gridworld_trajectory_{}.png".format(tag))


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
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    axs[0].plot(V)
    axs[0].set_xlabel("State")
    axs[0].set_ylabel("State-value")
    axs[0].set_title("State-value function")
    # Plot policy
    axs[1].bar(range(model.env.num_states), policy)
    axs[1].set_xlabel("State")
    axs[1].set_ylabel("Action")
    axs[1].set_title("Policy")
    plt.savefig("figures/gridworld/policy_statevalue_{}.png".format(tag))


def plot_learning_curve(model: Sarsa | QLearning):
    directory: str = "gridworld" if isinstance(model.env, GridWorld) else "pendulum"
    title: str = "SARSA" if isinstance(model, Sarsa) else "Q-learning"

    fig, ax = plt.subplots()
    ax.plot(model.returns)
    ax.grid(True)
    plt.title("{} Learning Curve".format(title))
    plt.savefig("figures/{}/learning_curve_{}.png".format(directory, tag(model)))


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

    # Plot data and save to png file
    title: str = "SARSA" if isinstance(model, Sarsa) else "Q-learning"
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(log["t"], log["s"])
    ax[0].plot(log["t"][:-1], log["a"])
    ax[0].plot(log["t"][:-1], log["r"])
    ax[0].legend(["s", "a", "r"])
    ax[1].plot(log["t"], log["theta"])
    ax[1].plot(log["t"], log["thetadot"])
    ax[1].legend(["theta", "thetadot"])
    plt.savefig("figures/pendulum/pendulum_trajectory_{}.png".format(tag(model)))


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
