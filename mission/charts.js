/* Flight-data wall. Numbers come from mission/board.js, which is generated from results/v2_qwen1.5b. */

const COLORS = {
  single_base: "#d7e6ff",
  team_base: "#f4f4f4",
  team_lora: "#f2c14e",
  team_distill: "#8b93a7",
};

function board() {
  return window.ORBIT_BOARD;
}

function cssSize(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(280, rect.width || canvas.clientWidth || 640);
  const height = Number(canvas.dataset.height) || 260;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function clear(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
}

function drawBars(canvas, field, intervalField) {
  const data = board();
  const { ctx, width, height } = cssSize(canvas);
  clear(ctx, width, height);
  const pad = { left: 36, right: 12, top: 16, bottom: 36 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  ctx.strokeStyle = "rgba(255,255,255,0.16)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + innerH);
  ctx.lineTo(pad.left + innerW, pad.top + innerH);
  ctx.stroke();
  ctx.fillStyle = "#b9b9b9";
  ctx.font = "11px Inter Tight, sans-serif";
  ctx.textAlign = "right";
  for (const tick of [0, 0.5, 1]) {
    const y = pad.top + innerH - tick * innerH;
    ctx.fillText(tick.toFixed(1), pad.left - 8, y + 4);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + innerW, y);
    ctx.stroke();
  }
  const runs = data.runs;
  const gap = innerW / runs.length;
  const barW = Math.min(48, gap * 0.46);
  runs.forEach((run, index) => {
    const cx = pad.left + gap * index + gap / 2;
    const color = COLORS[run.id] || "#f4f4f4";
    if (run.pending || run[field] == null) {
      ctx.strokeStyle = color;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(cx - barW / 2, pad.top + innerH * 0.35, barW, innerH * 0.65);
      ctx.setLineDash([]);
    } else {
      const value = Math.max(0, Math.min(1, run[field]));
      const barH = value * innerH;
      ctx.fillStyle = color;
      ctx.fillRect(cx - barW / 2, pad.top + innerH - barH, barW, barH);
      const interval = run[intervalField];
      if (interval) {
        const yHigh = pad.top + innerH - Math.max(0, Math.min(1, interval.high)) * innerH;
        const yLow = pad.top + innerH - Math.max(0, Math.min(1, interval.low)) * innerH;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(cx, yHigh);
        ctx.lineTo(cx, yLow);
        ctx.moveTo(cx - 7, yHigh);
        ctx.lineTo(cx + 7, yHigh);
        ctx.moveTo(cx - 7, yLow);
        ctx.lineTo(cx + 7, yLow);
        ctx.stroke();
      }
    }
    ctx.fillStyle = run.pending ? "#8b93a7" : "#f4f4f4";
    ctx.textAlign = "center";
    ctx.font = "11px Inter Tight, sans-serif";
    ctx.fillText(run.label.replace("Team ", ""), cx, height - 14);
  });
}

function drawPaired(canvas) {
  const data = board();
  const { ctx, width, height } = cssSize(canvas);
  clear(ctx, width, height);
  const rows = [
    ["Task success", data.paired.task_success_rate, "#f2c14e"],
    ["Schema-valid", data.paired.schema_valid_rate, "#d7e6ff"],
  ];
  const lows = rows.map((row) => row[1].low);
  const highs = rows.map((row) => row[1].high);
  const min = Math.min(-0.05, ...lows);
  const max = Math.max(0.05, ...highs);
  const span = max - min || 1;
  const pad = { left: 118, right: 16, top: 18, bottom: 28 };
  const innerW = width - pad.left - pad.right;
  const xOf = (value) => pad.left + ((value - min) / span) * innerW;
  ctx.strokeStyle = "rgba(255,107,90,0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(xOf(0), pad.top);
  ctx.lineTo(xOf(0), height - pad.bottom);
  ctx.stroke();
  ctx.fillStyle = "#ff6b5a";
  ctx.font = "11px Inter Tight, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("0", xOf(0) + 6, pad.top + 12);
  rows.forEach((row, index) => {
    const y = pad.top + 28 + index * 52;
    const interval = row[1];
    ctx.fillStyle = "#f4f4f4";
    ctx.textAlign = "right";
    ctx.fillText(row[0], pad.left - 12, y + 4);
    ctx.strokeStyle = row[2];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xOf(interval.low), y);
    ctx.lineTo(xOf(interval.high), y);
    ctx.moveTo(xOf(interval.low), y - 7);
    ctx.lineTo(xOf(interval.low), y + 7);
    ctx.moveTo(xOf(interval.high), y - 7);
    ctx.lineTo(xOf(interval.high), y + 7);
    ctx.stroke();
    ctx.fillStyle = row[2];
    ctx.beginPath();
    ctx.arc(xOf(interval.point), y, 4.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawLoss(canvas) {
  const data = board();
  const { ctx, width, height } = cssSize(canvas);
  clear(ctx, width, height);
  const series = data.loss;
  const pad = { left: 44, right: 12, top: 16, bottom: 32 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const steps = series.map((row) => row.step);
  const losses = series.map((row) => row.loss);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const maxLoss = Math.max(...losses) * 1.08;
  const xOf = (step) => pad.left + ((step - minStep) / (maxStep - minStep || 1)) * innerW;
  const yOf = (loss) => pad.top + innerH - (loss / maxLoss) * innerH;
  ctx.strokeStyle = "rgba(255,255,255,0.16)";
  ctx.strokeRect(pad.left, pad.top, innerW, innerH);
  ctx.beginPath();
  series.forEach((row, index) => {
    const x = xOf(row.step);
    const y = yOf(row.loss);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#f4f4f4";
  ctx.lineWidth = 1.7;
  ctx.stroke();
  ctx.lineTo(xOf(series[series.length - 1].step), pad.top + innerH);
  ctx.lineTo(xOf(series[0].step), pad.top + innerH);
  ctx.closePath();
  ctx.fillStyle = "rgba(242,193,78,0.14)";
  ctx.fill();
  ctx.fillStyle = "#b9b9b9";
  ctx.font = "11px Inter Tight, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(String(minStep), xOf(minStep), height - 12);
  ctx.fillText(String(maxStep), xOf(maxStep), height - 12);
  ctx.textAlign = "right";
  ctx.fillText("0", pad.left - 8, pad.top + innerH + 4);
  ctx.fillText(maxLoss.toFixed(1), pad.left - 8, pad.top + 4);
}

function scoreCards() {
  const root = document.getElementById("score-row");
  if (!root) return;
  root.replaceChildren();
  for (const run of board().runs) {
    const card = document.createElement("article");
    card.className = "score-card" + (run.pending ? " pending" : "");
    const name = document.createElement("p");
    name.textContent = run.label;
    const value = document.createElement("strong");
    value.textContent = run.pending ? "—" : `${String(run.n_success).padStart(2, "0")} / ${String(run.n_tasks).padStart(2, "0")}`;
    const meta = document.createElement("p");
    if (run.pending) {
      meta.textContent = "Not scored. The slot is ready for the distilled adapter.";
    } else {
      const schema = Number(run.schema_valid_rate).toFixed(3);
      meta.textContent = `Schema-valid ${schema}. Trap refusals ${run.n_correct_refusals} of ${run.n_traps}.`;
    }
    card.append(name, value, meta);
    root.append(card);
  }
}

function trapTable() {
  const root = document.getElementById("traps");
  if (!root) return;
  root.replaceChildren();
  const header = document.createElement("div");
  header.className = "trap-row trap-head";
  for (const label of ["Trap", "Single", "Team", "LoRA", "Distill"]) {
    const cell = document.createElement("span");
    cell.textContent = label;
    header.append(cell);
  }
  root.append(header);
  for (const trap of board().traps) {
    const row = document.createElement("div");
    row.className = "trap-row";
    const id = document.createElement("span");
    id.textContent = trap.id;
    row.append(id);
    for (const key of ["single_base", "team_base", "team_lora", "team_distill"]) {
      const cell = document.createElement("span");
      const outcome = trap.outcomes[key];
      if (outcome == null) {
        cell.textContent = "—";
        cell.className = "trap-pending";
      } else if (outcome) {
        cell.textContent = "Refused";
        cell.className = "trap-ok";
      } else {
        cell.textContent = "Missed";
        cell.className = "trap-miss";
      }
      row.append(cell);
    }
    root.append(row);
  }
}

function paintBoard() {
  if (!board()) return;
  drawBars(document.getElementById("success-chart"), "task_success_rate", "success_interval");
  drawBars(document.getElementById("schema-chart"), "schema_valid_rate", "schema_interval");
  drawPaired(document.getElementById("paired-chart"));
  drawLoss(document.getElementById("loss-chart"));
  scoreCards();
  trapTable();
}

function bootBoard() {
  const start = () => paintBoard();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(start);
  else start();
  window.addEventListener("resize", () => paintBoard());
}

bootBoard();
