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
    H=6, W=8,
    start=(0, 0),
    goal=(5, 7),
    walls={(1,2), (2,2), (3,2), (4,2), (4,3), (4,4), (2,5)},
    step_reward=-0.04,
    goal_reward=1.0,
    max_steps=200
)

policy = TabularSoftmaxPolicy(env.nS, env.nA, seed=1)

history_returns = []  # for plotting learning curve

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/api/state")
def get_state():
    return {
        "H": env.H,
        "W": env.W,
        "start": list(env.start),
        "goal": list(env.goal),
        "walls": [list(x) for x in sorted(env.walls)],
        "step_reward": env.step_reward,
        "goal_reward": env.goal_reward,
        "max_steps": env.max_steps,
    }

@app.post("/api/reset")
def reset():
    env.reset()
    return {"ok": True}

@app.get("/api/policy")
def get_policy():
    # Return action probs for each state (for later arrows/heatmap)
    probs = []
    for s in range(env.nS):
        probs.append(policy.probs(s).tolist())
    return {"probs": probs}

@app.post("/api/episode")
def sample_episode():
    S, A, R = run_episode(env, policy, rng)
    traj = [list(env.state_to_pos(s)) for s in S]
    return {
        "trajectory": traj,   # list of [r,c]
        "actions": A,         # list of ints
        "rewards": R,         # list of floats
        "return": float(sum(R)),
        "len": int(len(R)),
        "history_returns": history_returns[-200:],
    }

@app.post("/api/update")
def do_update(lr: float = 0.10, gamma: float = 0.99, baseline: bool = True):
    S, A, R = run_episode(env, policy, rng)
    cfg = ReinforceConfig(lr=lr, gamma=gamma, use_baseline=baseline)
    try:
        info = reinforce_update(policy, S, A, R, cfg)
    except NotImplementedError as e:
        return JSONResponse(
            status_code=501,
            content={"error": str(e), "hint": "Implement src/rl/reinforce.py"},
        )

    history_returns.append(info["return"])
    traj = [list(env.state_to_pos(s)) for s in S]
    return {
        "trajectory": traj,
        "return": info["return"],
        "len": info["len"],
        "history_returns": history_returns[-200:],
    }
