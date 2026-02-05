# src/rl/policy.py
from __future__ import annotations

import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

class TabularSoftmaxPolicy:
    def __init__(self, nS: int, nA: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.theta = 0.01 * rng.standard_normal((nS, nA))

    def probs(self, s: int) -> np.ndarray:
        return softmax(self.theta[s])

    def sample(self, s: int, rng: np.random.Generator) -> int:
        p = self.probs(s)
        return int(rng.choice(len(p), p=p))
