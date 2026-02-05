# src/rl/env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple

ACTIONS = ["U", "D", "L", "R"]
A2D = {
    0: (-1, 0),  # U
    1: (1, 0),   # D
    2: (0, -1),  # L
    3: (0, 1),   # R
}

Pos = Tuple[int, int]

@dataclass
class Gridworld:
    H: int = 6
    W: int = 8
    start: Pos = (0, 0)
    goal: Pos = (5, 7)
    walls: Set[Pos] = None
    step_reward: float = -0.04
    goal_reward: float = 1.0
    max_steps: int = 200

    def __post_init__(self):
        if self.walls is None:
            self.walls = set()
        self.reset()

    @property
    def nS(self) -> int:
        return self.H * self.W

    @property
    def nA(self) -> int:
        return 4

    def reset(self) -> int:
        self.pos = self.start
        self.t = 0
        return self._s()

    def _s(self) -> int:
        r, c = self.pos
        return r * self.W + c

    def state_to_pos(self, s: int) -> Pos:
        return (s // self.W, s % self.W)

    def is_terminal(self, pos: Pos) -> bool:
        return pos == self.goal

    def step(self, a: int):
        self.t += 1
        dr, dc = A2D[a]
        r, c = self.pos
        nr, nc = r + dr, c + dc

        # bounds
        if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
            nr, nc = r, c

        # walls
        if (nr, nc) in self.walls:
            nr, nc = r, c

        self.pos = (nr, nc)

        done = self.is_terminal(self.pos) or (self.t >= self.max_steps)
        reward = self.goal_reward if self.is_terminal(self.pos) else self.step_reward
        return self._s(), reward, done
