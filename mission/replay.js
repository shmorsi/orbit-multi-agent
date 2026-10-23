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

const V1_SEATS = {
  planner: document.getElementById("seat-planner"),
  worker: document.getElementById("seat-worker"),
  critic: document.getElementById("seat-critic"),
};
let seats = V1_SEATS;
let team = "v1";

// Team v2: six seats around the ring, clockwise in the order work flows.
const V2_ROLES = [
  ["commander", "Commander", "Assign"],
  ["scout", "Scout", "Read"],
  ["engineer", "Engineer", "Code"],
  ["analyst", "Analyst", "Draft"],
  ["critic", "Critic", "Check"],
  ["auditor", "Auditor", "Audit"],
];

const PHASE_LABELS = {
  v1: { planner: "01 Planner", worker: "02 Worker", tool: "03 Tool", critic: "04 Critic" },
  v2: {
    planner: "01 Commander",
    worker: "02 Specialist",
    tool: "03 Tool",
    critic: "04 Critic",
    auditor: "05 Auditor",
  },
};

function teamOf(mission, events) {
  const rules = (mission && mission.flight_rules) || {};
  if ((mission && mission.team) === "v2" || rules.team === "v2") return "v2";
  return events.some((event) => event.kind === "assign" || event.kind === "audit") ? "v2" : "v1";
}

function setPhases(next) {
  const labels = PHASE_LABELS[next];
  document.querySelectorAll("#phases li").forEach((item) => {
    const label = labels[item.dataset.phase];
    item.hidden = !label;
    if (label) item.textContent = label;
  });
}

function setTeam(next) {
  team = next;
  document.body.dataset.team = next;
  setPhases(next);
  const stage = document.getElementById("stage");
  stage.querySelectorAll(".seat.v2").forEach((node) => node.remove());
  for (const seat of Object.values(V1_SEATS)) seat.hidden = next === "v2";
  if (next !== "v2") {
    seats = V1_SEATS;
    return;
  }
  seats = {};
  const anchor = document.getElementById("pulse-trail");
  const reach = stage.clientWidth < 460 ? 34 : 38;
  V2_ROLES.forEach(([key, label, verb], index) => {
    const angle = -Math.PI / 2 + (index * Math.PI) / 3;
    const seat = document.createElement("div");
    seat.className = `seat v2 ${key}`;
    seat.id = `seat-v2-${key}`;
    seat.style.left = `${50 + Math.cos(angle) * reach}%`;
    seat.style.top = `${50 + Math.sin(angle) * reach}%`;
    seat.append(label);
    const small = document.createElement("small");
    small.textContent = verb;
    seat.append(small);
    stage.insertBefore(seat, anchor);
    seats[key] = seat;
  });
}

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

function finding(item) {
  const name = item.claim != null ? item.claim : `number ${item.number}`;
  return `${item.ok ? "ok" : "no"} ${name}${item.citation ? ` ${item.citation}` : ""}`;
}

function summarize(event) {
  const payload = event.payload;
  if (event.kind === "assign") return `Step ${event.step_number} to ${event.role}: ${event.goal}${event.tool ? ` (${event.tool})` : ""}`;
  if (event.kind === "draft") return `Draft: ${event.answer || event.status}`;
  if (event.kind === "audit") {
    const items = (event.findings || []).map(finding).join(" · ");
    return `${event.verdict === "pass" ? "Pass" : "Refuse"}: ${items}`;
  }
  if (event.kind === "final") return `${event.status === "refuse" ? "Refused" : "Final"}: ${event.answer || ""}`;
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

function routeV2(event) {
  const name = (key) => (key === "tool" ? "Tool" : key.charAt(0).toUpperCase() + key.slice(1));
  const action = event.payload && event.payload.action;
  const verdict = event.payload && event.payload.verdict;
  if (event.kind === "assign") return `Commander → ${name(event.role)}`;
  if (event.kind === "result") return `Tool → ${name(event.by || "analyst")}`;
  if (event.kind === "draft") return "Analyst → Critic";
  if (event.kind === "revise") return "Critic → Analyst";
  if (event.kind === "audit") return "Auditor · claim check";
  if (event.kind === "final") return "Auditor → Trace";
  if (event.agent === "critic") return verdict === "accept" ? "Critic → Auditor" : "Critic → Analyst";
  if (event.agent === "commander") return "Commander · plan";
  if (action === "tool") return `${name(event.agent)} → Tool`;
  if (action === "final") return `${name(event.agent)} → Critic`;
  return event.agent ? name(event.agent) : "Bus";
}

function route(event) {
  if (team === "v2") return routeV2(event);
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
  const v2 = rules.team === "v2";
  const tools = (key) => {
    const list = (rules.permissions && rules.permissions[key]) || [];
    return list.length ? ` Tools: ${list.join(", ")}.` : "";
  };
  const roleCards = v2
    ? V2_ROLES.map(([key, label]) => [label, rules[key] ? `${rules[key]}${tools(key)}` : ""])
    : [["Planner", rules.planner], ["Worker", rules.worker], ["Critic", rules.critic]];
  const cards = [
    ...roleCards,
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

function phaseOf(event) {
  if (!event) return null;
  if (team === "v2") {
    if (event.agent === "auditor" || event.kind === "audit" || event.kind === "final") return "auditor";
    if (event.agent === "critic" || event.kind === "revise") return "critic";
    if (event.agent === "tool" || (event.payload && event.payload.action === "tool")) return "tool";
    if (event.agent === "commander" || event.kind === "assign") return "planner";
    if (event.agent === "scout" || event.agent === "engineer" || event.agent === "analyst" || event.kind === "draft") return "worker";
    return null;
  }
  if (event.agent === "tool" || (event.payload && event.payload.action === "tool")) return "tool";
  if (event.agent === "critic" || event.kind === "revise") return "critic";
  if (event.agent === "planner") return "planner";
  if (event.agent === "worker") return "worker";
  return null;
}

function telemetryText(event) {
  if (!event) return "Awaiting uplink";
  if (event.kind === "audit") {
    const lines = (event.findings || []).map(finding);
    return `AUDIT  ${event.verdict}\n${lines.join("\n")}`;
  }
  if (event.kind === "final") return `FINAL  ${event.status}\n${event.answer || ""}`;
  if (event.kind === "assign") return `ASSIGN  ${event.role}\n${event.goal || ""}`;
  if (event.kind === "draft") return `DRAFT  ${event.status}\n${event.answer || ""}`;
  if (event.kind === "result") {
    const result = event.result || {};
    const body = result.stdout || result.error || JSON.stringify(result);
    return `${event.call}  ${event.tool}\n${String(body).trim()}`;
  }
  if (event.payload && event.payload.action === "tool") {
    const args = event.payload.arguments || {};
    const body = args.code || args.expression || args.query || args.path || "";
    return `CALL  ${event.payload.tool}\n${String(body).slice(0, 320)}`;
  }
  if (event.payload && event.payload.action === "final") {
    return `FINAL  ${event.payload.status}\n${event.payload.answer || ""}`;
  }
  if (event.payload && event.payload.verdict) {
    return `CRITIC  ${event.payload.verdict}\n${event.payload.reason || ""}`;
  }
  if (event.payload && Array.isArray(event.payload.steps)) {
    if (!event.payload.steps.length) return "PLAN  empty\nThe corpus cannot answer.";
    return event.payload.steps.map((step, index) => `${index + 1}. ${step.tool}  ${step.goal}`).join("\n");
  }
  if (event.kind === "revise") return `REVISE\n${event.reason || ""}`;
  if (event.kind === "invalid") return `INVALID\n${event.error || ""}`;
  if (typeof event.content === "string") return event.content.slice(0, 320);
  return event.kind;
}

function paintPhase(shown) {
  const last = shown[shown.length - 1];
  const phase = phaseOf(last);
  const items = document.querySelectorAll("#phases li");
  items.forEach((item) => item.classList.toggle("on", item.dataset.phase === phase));
  const box = document.getElementById("telemetry");
  if (box) box.textContent = telemetryText(last);
}

function focusAgentV2(event) {
  const verdict = event.payload && event.payload.verdict;
  const action = event.payload && event.payload.action;
  const lit = event.agent === "tool" ? event.by : event.agent;
  if (seats[lit]) seats[lit].classList.add("lit");
  if (event.kind === "message") return;
  if (event.kind === "assign") pulse("commander", event.role);
  else if (event.kind === "result") pulse("tool", event.by || "analyst");
  else if (action === "tool") pulse(event.agent, "tool");
  else if (event.kind === "draft") pulse("analyst", "critic");
  else if (event.kind === "revise" || verdict === "revise") pulse("critic", "analyst");
  else if (verdict === "accept") pulse("critic", "auditor");
  const auditor = seats.auditor;
  if (event.kind === "audit" && auditor) auditor.dataset.verdict = event.verdict;
}

function focusAgent(event) {
  for (const seat of Object.values(seats)) seat.classList.remove("lit");
  if (team === "v2") {
    focusAgentV2(event);
    return;
  }
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
  if (seats.auditor && !shown.some((event) => event.kind === "audit")) delete seats.auditor.dataset.verdict;
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
  paintPhase(shown);
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
  setTeam(teamOf(mission, events));
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
  // Team v2 traces live in traces/v2/. A missing index leaves the v1 list as it was.
  try {
    const v2Response = await fetch("traces/v2/index.json");
    if (v2Response.ok) {
      const v2Index = await v2Response.json();
      const v1Group = document.createElement("optgroup");
      v1Group.label = "Team v1, three roles";
      v1Group.append(...select.children);
      const v2Group = document.createElement("optgroup");
      v2Group.label = "Team v2, six roles";
      for (const item of v2Index) {
        const option = document.createElement("option");
        option.value = `v2/${item.file}`;
        option.textContent = `${item.task_id} · ${item.outcome || item.status}`;
        v2Group.append(option);
        index.push({ file: option.value });
      }
      select.append(v1Group, v2Group);
    }
  } catch (error) {
    setNotice("");
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
  setTeam("v1");
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
        setTeam(teamOf(row, state.events));
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
