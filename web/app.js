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

let env = null;
let lastTraj = null;
let hist = [];

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

async function init() {
  env = await api("/api/state");
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

init();
