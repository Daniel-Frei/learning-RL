# src/rl/rollout.py
from __future__ import annotations

import numpy as np
from .env import Gridworld
from .policy import TabularSoftmaxPolicy


def run_episode(
    environment: Gridworld,
    policy: TabularSoftmaxPolicy,
    random_generator: np.random.Generator,
):
    """
    Generate ONE full episode (trajectory).

    What is an episode?
        A sequence of:
            state -> action -> reward -> next_state
        until we hit a terminal condition.

    Returns:
        states_visited
        actions_taken
        rewards_received
    """

    # Reset environment to starting position
    current_state_index = environment.reset()

    # Lists to store the trajectory
    states_visited = [current_state_index]  # include initial state
    actions_taken = []
    rewards_received = []

    episode_finished = False

    while not episode_finished:

        # 1. Policy chooses an action
        chosen_action_index = policy.sample_action(
            current_state_index,
            random_generator,
        )

        # 2. Environment applies the action
        next_state_index, reward, episode_finished = environment.step(
            chosen_action_index
        )

        # 3. Store transition
        actions_taken.append(chosen_action_index)
        rewards_received.append(float(reward))
        states_visited.append(next_state_index)

        # 4. Move forward in time
        current_state_index = next_state_index

    return states_visited, actions_taken, rewards_received


def returns_from_rewards(
    rewards_received: list[float],
    discount_factor: float,
):
    """
    Compute Monte Carlo returns.

    Return definition:

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    We compute this BACKWARDS because it is much faster.
    """

    number_of_timesteps = len(rewards_received)

    # Placeholder for returns
    returns = [0.0] * number_of_timesteps

    discounted_future_return = 0.0

    # Walk backward through the episode
    for timestep in reversed(range(number_of_timesteps)):

        reward_at_timestep = rewards_received[timestep]

        discounted_future_return = (
            reward_at_timestep
            + discount_factor * discounted_future_return
        )

        returns[timestep] = discounted_future_return

    return returns
