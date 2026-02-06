# src/rl/env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

# We represent actions as integers: 0,1,2,3
# And map them to "Up/Down/Left/Right" movement on the grid.
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

# action_index -> (delta_row, delta_column)
ACTION_TO_DELTA = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}

# A grid position is a pair: (row, column)
Position = Tuple[int, int]


@dataclass
class Gridworld:
    """
    A tiny grid-based environment (an MDP) for reinforcement learning.

    - The agent starts at `start_position`.
    - The goal is to reach `goal_position`.
    - Some cells may be blocked by `wall_positions`.
    - Each step gives `step_reward` (usually negative).
    - Reaching the goal gives `goal_reward`.
    - The episode ends when the goal is reached OR when we hit `max_steps_per_episode`.
    """

    grid_height: int = 6
    grid_width: int = 8

    start_position: Position = (0, 0)
    goal_position: Position = (5, 7)

    wall_positions: Optional[Set[Position]] = None

    step_reward: float = -0.04
    goal_reward: float = 1.0

    max_steps_per_episode: int = 20

    def __post_init__(self) -> None:
        # If the caller didn't provide walls, use an empty set.
        if self.wall_positions is None:
            self.wall_positions = set()

        # Initialize internal state (agent position, step counter).
        self.reset()

    # -----------------------
    # "Size" helper properties
    # -----------------------

    @property
    def number_of_states(self) -> int:
        """Total states in the grid = height * width."""
        return self.grid_height * self.grid_width

    @property
    def number_of_actions(self) -> int:
        """We have 4 discrete actions: up/down/left/right."""
        return 4

    # -----------------------
    # Episode control
    # -----------------------

    def reset(self) -> int:
        """
        Start a new episode.
        Returns the initial state index (an integer).
        """
        self.agent_position: Position = self.start_position
        self.steps_taken_in_episode: int = 0
        return self._current_state_index()

    # -----------------------
    # State representation
    # -----------------------

    def _current_state_index(self) -> int:
        """
        Convert the agent's (row, col) into a single integer state index.

        Example for width=8:
          (row=0, col=0) -> 0
          (row=0, col=1) -> 1
          (row=1, col=0) -> 8
        """
        row, column = self.agent_position
        return row * self.grid_width + column

    def state_index_to_position(self, state_index: int) -> Position:
        """
        Convert an integer state index back to (row, col).
        """
        row = state_index // self.grid_width
        column = state_index % self.grid_width
        return (row, column)

    # -----------------------
    # Terminal / transition
    # -----------------------

    def is_terminal_position(self, position: Position) -> bool:
        """The episode ends if we are at the goal position."""
        return position == self.goal_position

    def step(self, action_index: int):
        """
        Apply one action.

        Returns:
          next_state_index (int),
          reward (float),
          done (bool)
        """
        self.steps_taken_in_episode += 1

        # Current position
        current_row, current_column = self.agent_position

        # Proposed movement for this action
        delta_row, delta_column = ACTION_TO_DELTA[action_index]
        proposed_row = current_row + delta_row
        proposed_column = current_column + delta_column

        # If proposed move is out of bounds, stay in place.
        out_of_bounds = (
            proposed_row < 0
            or proposed_row >= self.grid_height
            or proposed_column < 0
            or proposed_column >= self.grid_width
        )
        if out_of_bounds:
            proposed_row, proposed_column = current_row, current_column

        # If proposed move hits a wall, stay in place.
        proposed_position = (proposed_row, proposed_column)
        if proposed_position in self.wall_positions:
            proposed_row, proposed_column = current_row, current_column
            proposed_position = (proposed_row, proposed_column)

        # Commit the move
        self.agent_position = proposed_position

        # Determine if the episode is done
        reached_goal = self.is_terminal_position(self.agent_position)
        exceeded_time_limit = self.steps_taken_in_episode >= self.max_steps_per_episode
        done = reached_goal or exceeded_time_limit

        # Reward: goal reward if goal reached, else step reward
        reward = self.goal_reward if reached_goal else self.step_reward

        return self._current_state_index(), reward, done
