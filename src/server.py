# src/server.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.rl.env import Gridworld
from src.rl.policy import TabularSoftmaxPolicy
from src.rl.rollout import run_episode
from src.rl.reinforce import reinforce_update, ReinforceConfig

app = FastAPI()

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Global single-session state (fine for learning).
rng = np.random.default_rng(42)

env = Gridworld(
    grid_height=6,
    grid_width=8,
    start_position=(0, 0),
    goal_position=(5, 7),
    wall_positions={(1, 2), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (2, 5)},
    step_reward=-0.04,
    goal_reward=1.0,
    max_steps_per_episode=20,  # or 20, depending on what you want
)

policy = TabularSoftmaxPolicy(env.number_of_states, env.number_of_actions, random_seed=1)

history_returns: list[float] = []  # for plotting learning curve


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
def get_state():
    return {
        "H": env.grid_height,
        "W": env.grid_width,
        "start": list(env.start_position),
        "goal": list(env.goal_position),
        "walls": [list(pos) for pos in sorted(env.wall_positions)],
        "step_reward": env.step_reward,
        "goal_reward": env.goal_reward,
        "max_steps": env.max_steps_per_episode,
    }


@app.post("/api/reset")
def reset():
    env.reset()
    return {"ok": True}


@app.get("/api/policy")
def get_policy():
    probs = []
    for state_index in range(env.number_of_states):
        probs.append(policy.action_probabilities(state_index).tolist())
    return {"probs": probs}


@app.post("/api/episode")
def sample_episode():
    states, actions, rewards = run_episode(env, policy, rng)

    trajectory_positions = [list(env.state_index_to_position(s)) for s in states]

    return {
        "trajectory": trajectory_positions,  # list of [row, col]
        "actions": actions,                  # list of ints
        "rewards": rewards,                  # list of floats
        "return": float(sum(rewards)),
        "len": int(len(rewards)),
        "history_returns": history_returns[-200:],
    }


@app.post("/api/update")
def do_update(lr: float = 0.10, gamma: float = 0.99, baseline: bool = True):
    states, actions, rewards = run_episode(env, policy, rng)
    cfg = ReinforceConfig(lr=lr, gamma=gamma, use_baseline=baseline)

    try:
        info = reinforce_update(policy, states, actions, rewards, cfg)
    except NotImplementedError as e:
        return JSONResponse(
            status_code=501,
            content={"error": str(e), "hint": "Implement src/rl/reinforce.py"},
        )

    history_returns.append(float(info["return"]))

    trajectory_positions = [list(env.state_index_to_position(s)) for s in states]

    return {
        "trajectory": trajectory_positions,
        "return": float(info["return"]),
        "len": int(info["len"]),
        "history_returns": history_returns[-200:],
    }
