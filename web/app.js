const grid = document.getElementById("grid");
const gctx = grid.getContext("2d");

const curve = document.getElementById("curve");
const cctx = curve.getContext("2d");

const logEl = document.getElementById("log");

const btnEpisode = document.getElementById("btnEpisode");
const btnUpdate = document.getElementById("btnUpdate");
const btnUpdateN = document.getElementById("btnUpdateN");

const lrEl = document.getElementById("lr");
const gammaEl = document.getElementById("gamma");
const nEl = document.getElementById("n");
const baselineEl = document.getElementById("baseline");
const gridHEl = document.getElementById("gridH");
const gridWEl = document.getElementById("gridW");
const startPosEl = document.getElementById("startPos");
const goalPosEl = document.getElementById("goalPos");
const wallsEl = document.getElementById("walls");
const stepRewardEl = document.getElementById("stepReward");
const goalRewardEl = document.getElementById("goalReward");
const maxStepsEl = document.getElementById("maxSteps");
const btnApplyEnv = document.getElementById("btnApplyEnv");

const tabButtons = document.querySelectorAll(".tab-button");
const tabGrid = document.getElementById("tab-grid");
const tabParams = document.getElementById("tab-params");
const paramsTableEl = document.getElementById("paramsTable");
const btnRefreshParams = document.getElementById("btnRefreshParams");

let env = null;
let lastTraj = null;
let hist = [];
let activeTab = "grid";

function log(msg) {
  logEl.textContent = msg + "\n" + logEl.textContent;
}

function cellSize() {
  const cw = grid.width / env.W;
  const ch = grid.height / env.H;
  return { cw, ch };
}

function drawGrid() {
  if (!env) return;
  const { cw, ch } = cellSize();

  // background
  gctx.clearRect(0, 0, grid.width, grid.height);

  // cells
  for (let r = 0; r < env.H; r++) {
    for (let c = 0; c < env.W; c++) {
      gctx.fillStyle = "#151518";
      gctx.fillRect(c * cw, r * ch, cw, ch);
      gctx.strokeStyle = "#222";
      gctx.strokeRect(c * cw, r * ch, cw, ch);
    }
  }

  // state indices (top-right of each cell)
  gctx.fillStyle = "#7a7a7a";
  gctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, \"Liberation Mono\", monospace";
  gctx.textAlign = "right";
  gctx.textBaseline = "top";
  for (let r = 0; r < env.H; r++) {
    for (let c = 0; c < env.W; c++) {
      const stateIndex = r * env.W + c;
      const x = (c + 1) * cw - 4;
      const y = r * ch + 3;
      gctx.fillText(String(stateIndex), x, y);
    }
  }

  // walls
  for (const [r, c] of env.walls) {
    gctx.fillStyle = "#444";
    gctx.fillRect(c * cw, r * ch, cw, ch);
  }

  // start
  gctx.fillStyle = "#2b6";
  gctx.fillRect(env.start[1] * cw, env.start[0] * ch, cw, ch);

  // goal
  gctx.fillStyle = "#ea5";
  gctx.fillRect(env.goal[1] * cw, env.goal[0] * ch, cw, ch);

  // trajectory
  if (lastTraj && lastTraj.length > 1) {
    gctx.strokeStyle = "#7af";
    gctx.lineWidth = 3;
    gctx.beginPath();
    for (let i = 0; i < lastTraj.length; i++) {
      const [rr, cc] = lastTraj[i];
      const x = cc * cw + cw / 2;
      const y = rr * ch + ch / 2;
      if (i === 0) gctx.moveTo(x, y);
      else gctx.lineTo(x, y);
    }
    gctx.stroke();

    // dots
    gctx.fillStyle = "#7af";
    for (const [rr, cc] of lastTraj) {
      const x = cc * cw + cw / 2;
      const y = rr * ch + ch / 2;
      gctx.beginPath();
      gctx.arc(x, y, Math.min(cw, ch) * 0.12, 0, Math.PI * 2);
      gctx.fill();
    }
  }
}

function drawCurve() {
  cctx.clearRect(0, 0, curve.width, curve.height);
  if (!hist || hist.length < 2) return;

  const min = Math.min(...hist);
  const max = Math.max(...hist);
  const range = (max - min) || 1;

  cctx.strokeStyle = "#7af";
  cctx.lineWidth = 2;
  cctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const x = (i / (hist.length - 1)) * curve.width;
    const y = curve.height - ((hist[i] - min) / range) * curve.height;
    if (i === 0) cctx.moveTo(x, y);
    else cctx.lineTo(x, y);
  }
  cctx.stroke();
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw { status: res.status, data };
  }
  return data;
}

function setActiveTab(name) {
  activeTab = name;
  tabButtons.forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.tab === name);
  });
  tabGrid.classList.toggle("is-active", name === "grid");
  tabParams.classList.toggle("is-active", name === "params");
  if (name === "params") {
    loadParams();
  }
}

function renderParamsTable(parameters) {
  if (!parameters || parameters.length === 0) {
    paramsTableEl.textContent = "No parameters.";
    return;
  }

  const actions = parameters[0].length;
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");

  const thState = document.createElement("th");
  thState.textContent = "state";
  headRow.appendChild(thState);

  for (let a = 0; a < actions; a++) {
    const th = document.createElement("th");
    th.textContent = `a${a}`;
    headRow.appendChild(th);
  }

  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (let s = 0; s < parameters.length; s++) {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.textContent = `s${s}`;
    row.appendChild(label);

    const rowParams = parameters[s];
    for (let a = 0; a < rowParams.length; a++) {
      const td = document.createElement("td");
      td.textContent = rowParams[a].toFixed(4);
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }

  table.appendChild(tbody);
  paramsTableEl.innerHTML = "";
  paramsTableEl.appendChild(table);
}

function parsePair(text) {
  const parts = text.split(",").map((p) => p.trim()).filter((p) => p.length > 0);
  if (parts.length !== 2) {
    throw new Error(`Expected 'r,c' but got '${text}'`);
  }
  const r = parseInt(parts[0], 10);
  const c = parseInt(parts[1], 10);
  if (Number.isNaN(r) || Number.isNaN(c)) {
    throw new Error(`Invalid numbers in '${text}'`);
  }
  return [r, c];
}

function parseWalls(text) {
  const walls = [];
  const tokens = text.split(/[;\n]/);
  for (const raw of tokens) {
    const t = raw.trim();
    if (!t) continue;
    walls.push(parsePair(t));
  }
  return walls;
}

function formatPair(pair) {
  return `${pair[0]},${pair[1]}`;
}

function ensureNumber(value, name) {
  if (Number.isNaN(value)) {
    throw new Error(`${name} must be a number`);
  }
  return value;
}

function populateEnvInputs(state) {
  gridHEl.value = state.H;
  gridWEl.value = state.W;
  startPosEl.value = formatPair(state.start);
  goalPosEl.value = formatPair(state.goal);
  wallsEl.value = state.walls.map(formatPair).join("; ");
  stepRewardEl.value = state.step_reward;
  goalRewardEl.value = state.goal_reward;
  maxStepsEl.value = state.max_steps;
}

async function loadParams() {
  if (!paramsTableEl) return;
  try {
    const d = await api("/api/parameters");
    renderParamsTable(d.parameters);
  } catch (e) {
    paramsTableEl.textContent = `Error loading parameters: ${e.status}`;
  }
}

async function init() {
  env = await api("/api/state");
  populateEnvInputs(env);
  drawGrid();
  log("Loaded environment.");
}

btnEpisode.onclick = async () => {
  const d = await api("/api/episode", { method: "POST" });
  lastTraj = d.trajectory;
  hist = d.history_returns || hist;
  drawGrid();
  drawCurve();
  log(`Episode: return=${d.return.toFixed(3)} len=${d.len}`);
};

btnUpdate.onclick = async () => {
  const lr = parseFloat(lrEl.value);
  const gamma = parseFloat(gammaEl.value);
  const baseline = baselineEl.checked;

  try {
    const d = await api(`/api/update?lr=${lr}&gamma=${gamma}&baseline=${baseline}`, { method: "POST" });
    lastTraj = d.trajectory;
    hist = d.history_returns || hist;
    drawGrid();
    drawCurve();
    log(`Update: return=${d.return.toFixed(3)} len=${d.len}`);
    if (activeTab === "params") {
      loadParams();
    }
  } catch (e) {
    if (e.status === 501) {
      log(`Update not implemented: ${e.data.error}`);
    } else {
      log(`Error: ${e.status} ${JSON.stringify(e.data)}`);
    }
  }
};

btnUpdateN.onclick = async () => {
  const N = parseInt(nEl.value, 10);
  for (let i = 0; i < N; i++) {
    await btnUpdate.onclick();
  }
};

btnApplyEnv.onclick = async () => {
  try {
    const payload = {
      grid_height: ensureNumber(parseInt(gridHEl.value, 10), "grid_height"),
      grid_width: ensureNumber(parseInt(gridWEl.value, 10), "grid_width"),
      start_position: parsePair(startPosEl.value),
      goal_position: parsePair(goalPosEl.value),
      wall_positions: parseWalls(wallsEl.value),
      step_reward: ensureNumber(parseFloat(stepRewardEl.value), "step_reward"),
      goal_reward: ensureNumber(parseFloat(goalRewardEl.value), "goal_reward"),
      max_steps_per_episode: ensureNumber(parseInt(maxStepsEl.value, 10), "max_steps"),
    };

    const d = await api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    env = d;
    lastTraj = null;
    hist = [];
    populateEnvInputs(env);
    drawGrid();
    drawCurve();
    if (activeTab === "params") {
      loadParams();
    }
    log("Updated environment.");
  } catch (e) {
    const msg = e?.data?.detail || e?.message || "Unknown error";
    log(`Config error: ${msg}`);
  }
};

tabButtons.forEach((btn) => {
  btn.onclick = () => setActiveTab(btn.dataset.tab);
});

btnRefreshParams.onclick = () => loadParams();

init();
