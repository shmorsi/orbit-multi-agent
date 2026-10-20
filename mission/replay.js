const params = new URLSearchParams(location.search);
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const CLOCK_STEP = 4;

const state = {
  events: [],
  mission: null,
  cursor: 0,
  playing: false,
  timer: null,
  stepMs: Number(params.get("stepms")) || 2200,
  live: false,
};

const seats = {
  planner: document.getElementById("seat-planner"),
  worker: document.getElementById("seat-worker"),
  critic: document.getElementById("seat-critic"),
};

if (params.get("shot")) document.body.classList.add("shot");
if (params.get("record")) document.body.classList.add("record");

function pad(value) {
  return String(value).padStart(2, "0");
}

function clock(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  return `T+${pad(Math.floor(whole / 60))}:${pad(whole % 60)}`;
}

function setBadge(kind) {
  const badge = document.getElementById("badge");
  badge.className = "badge " + kind;
  badge.textContent = kind === "success" ? "Mission success" : kind === "abort" ? "Abort" : "In flight";
  document.body.classList.toggle("success", kind === "success");
  document.body.classList.toggle("abort", kind === "abort");
  const mark = document.getElementById("mark");
  if (mark) mark.textContent = kind === "success" ? "Mission success" : kind === "abort" ? "Abort" : "Orbit";
}

function isScripted(mission, backend) {
  const harness = mission && mission.harness;
  return harness === "scripted_harness" || backend === "scripted";
}

function applyHarness(mission, backend) {
  const scripted = isScripted(mission, backend);
  document.getElementById("harness").hidden = !scripted;
  document.getElementById("title-harness").hidden = !scripted;
}

let titleTimer = null;

function showTitle(mission) {
  const card = document.getElementById("titlecard");
  document.getElementById("title-question").textContent = mission && mission.question ? mission.question : "";
  applyHarness(mission);
  card.classList.add("show");
}

function hideTitle() {
  if (titleTimer) clearTimeout(titleTimer);
  titleTimer = null;
  document.getElementById("titlecard").classList.remove("show");
}

function setNotice(text) {
  document.getElementById("notice").textContent = text || "";
}

function summarize(event) {
  const payload = event.payload;
  if (payload && Array.isArray(payload.steps)) {
    const tools = payload.steps.map((step) => step.tool).filter(Boolean);
    return tools.length ? tools.join(" · ") : "Empty plan";
  }
  if (payload && payload.action === "tool") return payload.tool;
  if (payload && payload.action === "final") return payload.answer || payload.status;
  if (payload && payload.verdict) return `${payload.verdict}: ${payload.reason || ""}`.trim();
  if (event.kind === "result") return `${event.tool} ${event.call}`;
  if (event.kind === "revise") return event.reason || "Revise";
  if (event.kind === "invalid") return event.error || "Invalid";
  if (typeof event.content === "string") return event.content.replace(/\s+/g, " ").slice(0, 140);
  return event.kind;
}

function route(event) {
  if (event.agent === "planner") return "Planner → Worker";
  if (event.agent === "tool") return "Tool → Worker";
  if (event.agent === "critic") return "Critic → Worker";
  if (event.agent === "worker") {
    const action = event.payload && event.payload.action;
    return action === "final" ? "Worker → Critic" : "Worker → Tool";
  }
  return "Bus";
}

function renderRules(rules) {
  const root = document.getElementById("rules");
  root.replaceChildren();
  if (!rules) return;
  const cards = [
    ["Planner", rules.planner],
    ["Worker", rules.worker],
    ["Critic", rules.critic],
    ["Limits", `At most ${rules.max_turns} turns, ${rules.max_critic_retries} revise passes, and ${rules.max_schema_tries} schema tries on a turn.`],
  ];
  for (const [title, body] of cards) {
    if (!body) continue;
    const article = document.createElement("article");
    const heading = document.createElement("h3");
    heading.textContent = title;
    const copy = document.createElement("p");
    copy.textContent = body;
    article.append(heading, copy);
    root.append(article);
  }
}

function center(node) {
  const stage = document.getElementById("stage").getBoundingClientRect();
  const box = node.getBoundingClientRect();
  return {
    x: box.left + box.width / 2 - stage.left,
    y: box.top + box.height / 2 - stage.top,
  };
}

function moveDot(dot, start, end, quiet, delay) {
  dot.classList.add("on");
  dot.style.transition = "none";
  dot.style.left = `${start.x}px`;
  dot.style.top = `${start.y}px`;
  if (quiet) {
    dot.style.left = `${end.x}px`;
    dot.style.top = `${end.y}px`;
    return;
  }
  requestAnimationFrame(() => {
    setTimeout(() => {
      dot.style.transition = "left 900ms cubic-bezier(0.4, 0, 0.2, 1), top 900ms cubic-bezier(0.4, 0, 0.2, 1)";
      dot.style.left = `${end.x}px`;
      dot.style.top = `${end.y}px`;
    }, delay);
  });
}

function pulse(fromName, toName) {
  const from = fromName === "tool" ? document.getElementById("dock") : seats[fromName];
  const to = toName === "tool" ? document.getElementById("dock") : seats[toName];
  if (!from || !to) return;
  const start = center(from);
  const end = center(to);
  const quiet = reduced || document.body.classList.contains("shot");
  moveDot(document.getElementById("pulse"), start, end, quiet, 0);
  const trail = document.getElementById("pulse-trail");
  if (trail) moveDot(trail, start, end, quiet, 90);
}

function focusAgent(event) {
  for (const seat of Object.values(seats)) seat.classList.remove("lit");
  const name = event.agent === "tool" ? "worker" : event.agent;
  if (seats[name]) seats[name].classList.add("lit");
  if (event.kind === "message") return;
  if (event.agent === "planner") pulse("planner", "worker");
  if (event.agent === "worker") {
    const action = event.payload && event.payload.action;
    pulse("worker", action === "final" ? "critic" : "tool");
  }
  if (event.agent === "tool") pulse("tool", "worker");
  if (event.agent === "critic") pulse("critic", "worker");
}

function render() {
  const shown = state.events.slice(0, state.cursor);
  const steps = shown.reduce((max, event) => Math.max(max, event.step || 0), 0);
  const retries = shown.filter((event) => event.kind === "revise").length;
  const tools = shown.filter((event) => event.kind === "result").length;
  document.getElementById("steps").textContent = pad(steps);
  document.getElementById("retries").textContent = pad(retries);
  document.getElementById("tools").textContent = pad(tools);
  const scrub = document.getElementById("scrub");
  scrub.max = String(state.events.length);
  scrub.value = String(state.cursor);
  const done = state.cursor >= state.events.length && state.events.length > 0 && state.mission;
  if (done && state.mission.status === "answer") setBadge("success");
  else if (done && state.mission.status === "refuse") setBadge("abort");
  else setBadge("inflight");
  const log = document.getElementById("log");
  log.replaceChildren();
  for (const event of shown) {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = clock(event.t_sec == null ? event.index * CLOCK_STEP : event.t_sec);
    const body = document.createElement("div");
    const routeLine = document.createElement("div");
    routeLine.className = "route";
    routeLine.textContent = route(event);
    const detail = document.createElement("div");
    detail.className = "detail";
    detail.textContent = summarize(event);
    body.append(routeLine, detail);
    item.append(time, body);
    log.append(item);
  }
  const last = shown[shown.length - 1];
  if (last) {
    focusAgent(last);
    const dock = document.getElementById("dock");
    const toolEvent = [...shown].reverse().find((event) => event.kind === "result");
    if (toolEvent) {
      dock.textContent = `${toolEvent.tool} · ${toolEvent.call}`;
      dock.classList.add("on");
    }
    if (!document.body.classList.contains("shot")) {
      const row = log.lastElementChild;
      if (row) row.scrollIntoView({ block: "nearest" });
    }
  }
}

function stopTimer() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
}

function pause() {
  state.playing = false;
  document.getElementById("play").setAttribute("aria-pressed", "false");
  stopTimer();
}

function schedule() {
  stopTimer();
  const speed = Number(document.getElementById("speed").value) || 1;
  state.timer = setTimeout(advance, state.stepMs / speed);
}

function advance() {
  if (!state.playing) return;
  if (state.cursor >= state.events.length) {
    pause();
    render();
    return;
  }
  state.cursor += 1;
  render();
  schedule();
}

function play() {
  hideTitle();
  if (!state.events.length) return;
  if (state.cursor >= state.events.length) state.cursor = 0;
  state.playing = true;
  document.getElementById("play").setAttribute("aria-pressed", "true");
  if (state.cursor === 0) {
    state.cursor = 1;
    render();
  }
  schedule();
}

function loadMission(mission, events) {
  pause();
  state.mission = mission;
  state.events = events;
  state.cursor = 0;
  state.live = false;
  document.getElementById("question").textContent = mission && mission.question ? mission.question : "Saved trace";
  applyHarness(mission);
  renderRules(mission && mission.flight_rules);
  document.getElementById("dock").classList.remove("on");
  render();
}

async function loadTrace(file) {
  const response = await fetch(`traces/${file}`);
  if (!response.ok) throw new Error(`missing trace ${file}`);
  const lines = (await response.text()).split("\n").filter((line) => line.trim());
  const rows = lines.map((line) => JSON.parse(line));
  const mission = rows.find((row) => row.kind === "mission");
  const events = rows.filter((row) => row.kind !== "mission");
  loadMission(mission, events);
  const shot = Number(params.get("shot"));
  if (shot) {
    hideTitle();
    state.cursor = Math.min(events.length, Math.max(0, shot));
    render();
    return;
  }
  showTitle(mission);
  if (params.get("autoplay") === "1") {
    const titleMs = Number(params.get("titlems")) || 2800;
    titleTimer = setTimeout(() => {
      hideTitle();
      state.cursor = Math.min(1, events.length);
      render();
      if (state.cursor < events.length) play();
    }, titleMs);
  }
}

async function loadIndex() {
  const response = await fetch("traces/index.json");
  const index = await response.json();
  const select = document.getElementById("trace");
  select.replaceChildren();
  for (const item of index) {
    const option = document.createElement("option");
    option.value = item.file;
    option.textContent = `${item.task_id} · ${item.status}`;
    select.append(option);
  }
  const requested = params.get("trace");
  const choice = index.some((item) => item.file === requested) ? requested : "revise.jsonl";
  if ([...select.options].some((option) => option.value === choice)) select.value = choice;
  await loadTrace(select.value);
}

function markLive() {
  const select = document.getElementById("trace");
  let option = [...select.options].find((item) => item.value === "live");
  if (!option) {
    option = document.createElement("option");
    option.value = "live";
    option.textContent = "Live";
    select.append(option);
  }
  select.value = "live";
}

async function runLive(payload) {
  pause();
  state.live = true;
  state.events = [];
  state.mission = null;
  state.cursor = 0;
  markLive();
  applyHarness(null, payload.backend);
  showTitle({ question: document.getElementById("question").textContent, harness: payload.backend === "scripted" ? "scripted_harness" : payload.backend });
  titleTimer = setTimeout(hideTitle, Number(params.get("titlems")) || 1600);
  document.getElementById("dock").classList.remove("on");
  setBadge("inflight");
  setNotice("");
  render();
  let response;
  try {
    response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    setNotice("This page could not reach the local server. Saved traces still replay. Start it with make demo.");
    return;
  }
  if (!response.ok || !response.body) {
    setNotice("Live runs need the local server. Saved traces still replay on this page.");
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const row = JSON.parse(line);
      if (row.kind === "error") {
        setNotice(row.error);
        continue;
      }
      if (row.kind === "mission") {
        state.mission = row;
        applyHarness(row);
        document.getElementById("question").textContent = row.question || document.getElementById("question").textContent;
        renderRules(row.flight_rules);
        state.cursor = state.events.length;
        render();
        continue;
      }
      state.events.push(row);
      state.cursor = state.events.length;
      render();
    }
  }
}

document.getElementById("play").addEventListener("click", play);
document.getElementById("pause").addEventListener("click", pause);
document.getElementById("scrub").addEventListener("input", (event) => {
  pause();
  hideTitle();
  state.cursor = Number(event.target.value);
  render();
});
document.getElementById("trace").addEventListener("change", (event) => {
  if (event.target.value === "live") return;
  params.delete("autoplay");
  loadTrace(event.target.value).catch((error) => setNotice(error.message));
});
document.getElementById("live").addEventListener("submit", (event) => {
  event.preventDefault();
  const backend = document.getElementById("backend").value;
  const taskId = document.getElementById("task-id").value.trim();
  const question = document.getElementById("ask").value.trim();
  const model = document.getElementById("model").value.trim();
  const payload = { backend, model };
  if (taskId) payload.task_id = taskId;
  else payload.question = question;
  document.getElementById("question").textContent = question || `Running ${taskId || "a task"}`;
  runLive(payload);
});
document.addEventListener("keydown", (event) => {
  if (event.code !== "Space") return;
  const tag = event.target && event.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  event.preventDefault();
  if (state.playing) pause();
  else play();
});

loadIndex().catch((error) => setNotice(error.message));
