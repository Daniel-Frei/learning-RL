# src/rl/reinforce.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .policy import TabularSoftmaxPolicy
from .rollout import returns_from_rewards


@dataclass
class ReinforceConfig:
    """
    Configuration for one REINFORCE update.
    """
    learning_rate: float = 0.10
    discount_factor: float = 0.99
    use_baseline: bool = True


def reinforce_update(
    policy: TabularSoftmaxPolicy,
    states_visited: list[int],      # length T+1
    actions_taken: list[int],       # length T
    rewards_received: list[float],  # length T
    cfg: ReinforceConfig,
):
    """
    Perform ONE Monte Carlo REINFORCE (policy gradient) update using a single episode.

    We want to maximize expected return J(theta).

    REINFORCE update (gradient ascent):
        theta += lr * sum_t [  (∇_theta log π_theta(a_t | s_t)) * advantage_t ]

    For a tabular softmax policy, the gradient has a closed form:
        ∇_theta[s_t, :] log π(a_t | s_t) = one_hot(a_t) - π(. | s_t)

    Advantage (variance reduction):
        advantage_t = G_t - baseline
        baseline can be mean(G) for the episode (simple choice).

    Returns:
        dict with episode stats to plot learning curves in the UI.
    """

    # ------------------------------------------------------------
    # 1) Compute Monte Carlo returns G_t from rewards r_t
    # ------------------------------------------------------------
    # returns[t] = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    discounted_returns = np.array(
        returns_from_rewards(rewards_received, cfg.discount_factor),
        dtype=np.float64,
    )  # shape: (T,)

    # ------------------------------------------------------------
    # 2) Compute baseline + advantage
    # ------------------------------------------------------------
    if cfg.use_baseline:
        baseline_value = float(np.mean(discounted_returns))
        advantages = discounted_returns - baseline_value
    else:
        baseline_value = 0.0
        advantages = discounted_returns

    # ------------------------------------------------------------
    # 3) Accumulate policy gradient over the episode
    # ------------------------------------------------------------
    # The gradient has the same shape as the parameter table:
    # policy.policy_parameters[state_index, action_index]
    policy_gradient = np.zeros_like(policy.policy_parameters, dtype=np.float64)

    # There are T actions and T rewards, but T+1 states.
    # We use states_visited[t] with actions_taken[t] and advantages[t].
    for timestep in range(len(actions_taken)):
        state_index = states_visited[timestep]
        action_index = actions_taken[timestep]
        advantage_at_timestep = float(advantages[timestep])

        # π(.|s_t): probabilities for each action in this state
        action_probabilities = policy.action_probabilities(state_index)  # shape: (num_actions,)

        # one_hot(a_t): vector with 1 at chosen action, 0 elsewhere
        one_hot_action = np.zeros(policy.number_of_actions, dtype=np.float64)
        one_hot_action[action_index] = 1.0

        # For tabular softmax:
        # grad_log_pi = one_hot(a_t) - π(.|s_t)
        gradient_log_probability = one_hot_action - action_probabilities

        # Weight by advantage
        # grad[state_index, :] += grad_log_pi * advantage_t
        policy_gradient[state_index, :] += gradient_log_probability * advantage_at_timestep

    # ------------------------------------------------------------
    # 4) Gradient ASCENT step (we are maximizing expected return)
    # ------------------------------------------------------------
    policy.policy_parameters += cfg.learning_rate * policy_gradient

    # ------------------------------------------------------------
    # 5) Return useful stats for debugging / UI
    # ------------------------------------------------------------
    episode_return = float(np.sum(rewards_received))
    episode_length = int(len(rewards_received))

    return {
        "return": episode_return,
        "len": episode_length,
        "baseline": float(baseline_value),
    }
