# src/rl/policy.py
from __future__ import annotations

import numpy as np


def softmax(action_logits: np.ndarray) -> np.ndarray:
    """
    Convert raw scores ("logits") into probabilities.

    Why subtract the max?
        Numerical stability.
        Without it, exp() can overflow for large numbers.

    Softmax formula:

        pi(a|s) = exp(logit_a) / sum(exp(logit_all_actions))

    Output:
        A probability distribution over actions that sums to 1.
    """

    # Stabilize exponentials
    stabilized_logits = action_logits - np.max(action_logits)

    exponentials = np.exp(stabilized_logits)

    action_probabilities = exponentials / (np.sum(exponentials) + 1e-12)

    return action_probabilities


class TabularSoftmaxPolicy:
    """
    A tabular policy with softmax.

    "Tabular" means:
        We store parameters directly for each state-action pair.

    Conceptually:

        policy_parameters[state, action] -> preference for that action

    These preferences are turned into probabilities via softmax.

    No neural networks yet — just a table.
    Perfect for learning RL fundamentals.
    """

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        random_seed: int = 0,
    ):
        """
        Create the policy parameter table.

        Shape:
            (number_of_states, number_of_actions)

        Each row corresponds to a state.
        Each column corresponds to an action.
        """

        random_generator = np.random.default_rng(random_seed)

        # Small random initialization is important:
        # If all logits were identical, the policy would be perfectly uniform.
        # Tiny noise breaks symmetry.
        self.policy_parameters = 0.01 * random_generator.standard_normal(
            (number_of_states, number_of_actions)
        )

        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

    # -------------------------------------------------
    # Core policy operations
    # -------------------------------------------------

    def action_probabilities(self, state_index: int) -> np.ndarray:
        """
        Return π(. | state).

        Steps:
            1. Look up parameter row for this state.
            2. Convert logits -> probabilities via softmax.
        """

        action_logits_for_state = self.policy_parameters[state_index]

        probabilities = softmax(action_logits_for_state)

        return probabilities

    def sample_action(
        self,
        state_index: int,
        random_generator: np.random.Generator,
    ) -> int:
        """
        Sample an action according to the policy distribution.

        This is what makes the policy STOCHASTIC.

        Example:
            probs = [0.7, 0.1, 0.1, 0.1]

        The agent will pick action 0 about 70% of the time.
        """

        probabilities = self.action_probabilities(state_index)

        chosen_action_index = random_generator.choice(
            self.number_of_actions,
            p=probabilities,
        )

        return int(chosen_action_index)
