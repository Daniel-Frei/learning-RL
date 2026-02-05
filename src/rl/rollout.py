# src/rl/rollout.py
from __future__ import annotations

import numpy as np
from .env import Gridworld
from .policy import TabularSoftmaxPolicy

def run_episode(env: Gridworld, policy: TabularSoftmaxPolicy, rng: np.random.Generator):
    s = env.reset()
    S, A, R = [s], [], []
    done = False
    while not done:
        a = policy.sample(s, rng)
        s2, r, done = env.step(a)
        A.append(a)
        R.append(float(r))
        S.append(s2)
        s = s2
    return S, A, R

def returns_from_rewards(R, gamma: float):
    G = [0.0] * len(R)
    g = 0.0
    for t in reversed(range(len(R))):
        g = R[t] + gamma * g
        G[t] = g
    return G
