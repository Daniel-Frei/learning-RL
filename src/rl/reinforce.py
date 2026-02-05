# src/rl/reinforce.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .policy import TabularSoftmaxPolicy
from .rollout import returns_from_rewards

@dataclass
class ReinforceConfig:
    lr: float = 0.10
    gamma: float = 0.99
    use_baseline: bool = True

def reinforce_update(policy: TabularSoftmaxPolicy, S, A, R, cfg: ReinforceConfig):
    """
    Homework:
      Implement ONE Monte Carlo REINFORCE gradient-ascent step.

    Inputs:
      S: list of states length T+1
      A: list of actions length T
      R: list of rewards length T

    Returns:
      dict with debug info, e.g. {"return": ..., "len": ...}

    Notes:
      - Use cfg.gamma for returns
      - If cfg.use_baseline, subtract baseline (e.g. mean(G)) to reduce variance
      - policy.theta is your parameter table.
      - policy.probs(s) gives pi(.|s)

    Leave everything else unchanged.
    """
    G = returns_from_rewards(R, cfg.gamma)

    # TODO baseline
    if cfg.use_baseline:
        # b = ?
        # adv = ?
        raise NotImplementedError("Implement baseline + advantage")
    else:
        # adv = G
        raise NotImplementedError("Implement advantage without baseline")

    # TODO compute grad and update theta
    # grad[s,:] += (one_hot(a) - pi(.|s)) * adv_t
    # policy.theta += cfg.lr * grad

    # Return episode stats (so UI can plot learning curve)
    return {"return": float(sum(R)), "len": int(len(R))}
