# src/server.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rl.env import Gridworld
from src.rl.policy import TabularSoftmaxPolicy
from src.rl.rollout import run_episode
from src.rl.reinforce import reinforce_update, ReinforceConfig

app = FastAPI()

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Global single-session state (fine for learning).
rng = np.random.default_rng(42)

class EnvConfig(BaseModel):
    grid_height: int
    grid_width: int
    start_position: list[int]
    goal_position: list[int]
    wall_positions: list[list[int]]
    step_reward: float
    goal_reward: float
    max_steps_per_episode: int


def _validate_and_build_env(cfg: EnvConfig) -> Gridworld:
    if cfg.grid_height <= 0 or cfg.grid_width <= 0:
        raise HTTPException(status_code=400, detail="Grid height/width must be > 0")

    if len(cfg.start_position) != 2 or len(cfg.goal_position) != 2:
        raise HTTPException(status_code=400, detail="Start/goal must be length-2 lists")

    start = tuple(cfg.start_position)
    goal = tuple(cfg.goal_position)

    def in_bounds(pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < cfg.grid_height and 0 <= pos[1] < cfg.grid_width

    if not in_bounds(start):
        raise HTTPException(status_code=400, detail="Start position out of bounds")
    if not in_bounds(goal):
        raise HTTPException(status_code=400, detail="Goal position out of bounds")

    walls: set[tuple[int, int]] = set()
    for wall in cfg.wall_positions:
        if len(wall) != 2:
            raise HTTPException(status_code=400, detail="Wall positions must be length-2 lists")
        pos = (int(wall[0]), int(wall[1]))
        if not in_bounds(pos):
            raise HTTPException(status_code=400, detail=f"Wall out of bounds: {pos}")
        walls.add(pos)

    if start in walls:
        raise HTTPException(status_code=400, detail="Start position cannot be a wall")
    if goal in walls:
        raise HTTPException(status_code=400, detail="Goal position cannot be a wall")

    return Gridworld(
        grid_height=cfg.grid_height,
        grid_width=cfg.grid_width,
        start_position=start,
        goal_position=goal,
        wall_positions=walls,
        step_reward=cfg.step_reward,
        goal_reward=cfg.goal_reward,
        max_steps_per_episode=cfg.max_steps_per_episode,
    )


env = _validate_and_build_env(
    EnvConfig(
        grid_height=6,
        grid_width=8,
        start_position=[0, 0],
        goal_position=[5, 7],
        wall_positions=[[1, 2], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4], [2, 5]],
        step_reward=-0.04,
        goal_reward=1.0,
        max_steps_per_episode=100,
    )
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
        "number_of_states": env.number_of_states,
        "number_of_actions": env.number_of_actions,
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


@app.post("/api/config")
def update_config(cfg: EnvConfig):
    global env, policy, history_returns
    env = _validate_and_build_env(cfg)
    policy = TabularSoftmaxPolicy(env.number_of_states, env.number_of_actions, random_seed=1)
    history_returns = []
    return get_state()


@app.get("/api/policy")
def get_policy():
    probs = []
    for state_index in range(env.number_of_states):
        probs.append(policy.action_probabilities(state_index).tolist())
    return {"probs": probs}


@app.get("/api/parameters")
def get_parameters():
    return {"parameters": policy.policy_parameters.tolist()}


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
    cfg = ReinforceConfig(
        learning_rate=lr,
        discount_factor=gamma,
        use_baseline=baseline,
    )

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
