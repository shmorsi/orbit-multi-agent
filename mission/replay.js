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

const V3_ROLES = [
  ["coordinator", "Coordinator", "Open"],
  ["commander", "Commander", "Assign"],
  ["researcher", "Researcher", "Read"],
  ["scout", "Scout", "Search"],
  ["engineer", "Engineer", "Code"],
  ["analyst", "Analyst", "Draft"],
  ["critic", "Critic", "Check"],
  ["redteam", "Red team", "Attack"],
  ["verifier", "Verifier", "Verify"],
  ["auditor", "Auditor", "Gate"],
];

const CODE_ROLES = [
  ["planner", "Planner", "Plan"],
  ["coder", "Coder", "Edit"],
  ["test_runner", "Tests", "Run"],
  ["critic", "Critic", "Check"],
  ["reviewer", "Reviewer", "Ship"],
];

const BOARD_ROLES = [
  ["ceo", "CEO", "Frame"],
  ["cfo", "CFO", "Finance"],
  ["cto", "CTO", "Product"],
  ["coo", "COO", "Ops"],
  ["auditor", "Auditor", "Gate"],
];

const ROLE_LINES = {
  planner: "Writes the plan. Does not call tools.",
  worker: "Calls a tool, or drafts the answer.",
  critic: "Accepts the draft or sends it back.",
  commander: "Assigns each step to the seat that owns that tool.",
  scout: "Reads the corpus. Cannot run code.",
  engineer: "Runs code or writes a file. Cannot answer.",
  analyst: "Drafts the only answer the auditor scores.",
  auditor: "Refuses a claim or a number the tools do not support.",
  coordinator: "Opens the table. No tools and no answer.",
  researcher: "Reads the corpus. Cannot run code or answer.",
  redteam: "Proposes one claim the tools do not support.",
  verifier: "The code gate re-checks that claim.",
  coder: "Edits source in a temp workspace. Tests are off limits.",
  test_runner: "Runs pytest with a timeout and no network.",
  reviewer: "Rejects a patch whose tests fail or whose diff touches the tests.",
  ceo: "Frames the decision. Does not invent a figure.",
  cfo: "Cites the financials. A figure the file does not contain is refused.",
  cto: "Cites the product metrics.",
  coo: "Cites the launch memo.",
};

const PHASE_LABELS = {
  v1: { planner: "01 Planner", worker: "02 Worker", tool: "03 Tool", critic: "04 Critic" },
  v2: {
    planner: "01 Commander",
    worker: "02 Specialist",
    tool: "03 Tool",
    critic: "04 Critic",
    auditor: "05 Auditor",
  },
  v3: {
    planner: "01 Coordinator",
    worker: "02 Crew",
    tool: "03 Tool",
    critic: "04 Verifier",
    auditor: "05 Auditor",
  },
  code: {
    planner: "01 Planner",
    worker: "02 Coder",
    tool: "03 Tests",
    critic: "04 Critic",
    auditor: "05 Reviewer",
  },
  boardroom: {
    planner: "01 CEO",
    worker: "02 Officers",
    tool: "03 Files",
    critic: "04 Check",
    auditor: "05 Auditor",
  },
};

function teamOf(mission, events) {
  const rules = (mission && mission.flight_rules) || {};
  const named = (mission && mission.team) || rules.team;
  if (named === "code" || named === "boardroom" || named === "v3" || named === "v2" || named === "v1") return named;
  if (events.some((event) => event.agent === "coordinator" || event.agent === "redteam" || event.agent === "verifier")) return "v3";
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

function clearRoundTable() {
  const stage = document.getElementById("stage");
  stage.querySelectorAll(".seat.v2, .seat.v3, .seat.code, .seat.boardroom").forEach((node) => node.remove());
  const links = document.getElementById("links");
  if (links) links.remove();
}

function placeSeats(roles, className, reach) {
  const stage = document.getElementById("stage");
  const anchor = document.getElementById("pulse-trail");
  const placed = {};
  roles.forEach(([key, label, verb], index) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / roles.length;
    const seat = document.createElement("div");
    seat.className = `seat ${className} ${key}`;
    seat.id = `seat-${className}-${key}`;
    seat.style.left = `${50 + Math.cos(angle) * reach}%`;
    seat.style.top = `${50 + Math.sin(angle) * reach}%`;
    seat.append(label);
    const small = document.createElement("small");
    small.textContent = verb;
    seat.append(small);
    stage.insertBefore(seat, anchor);
    placed[key] = seat;
  });
  return placed;
}

function setTeam(next) {
  team = next;
  document.body.dataset.team = next;
  setPhases(next);
  const stage = document.getElementById("stage");
  clearRoundTable();
  for (const seat of Object.values(V1_SEATS)) seat.hidden = next !== "v1";
  const work = document.getElementById("codework");
  if (work) work.hidden = next !== "code" && next !== "boardroom";
  const lanes = document.getElementById("lanes");
  if (lanes) lanes.hidden = next !== "code" && next !== "boardroom";
  if (next === "v1") {
    seats = V1_SEATS;
    return;
  }
  if (next === "code" || next === "boardroom") {
    const narrow = stage.clientWidth < 460;
    const reach = narrow ? 34 : 38;
    seats = placeSeats(next === "code" ? CODE_ROLES : BOARD_ROLES, next, reach);
    return;
  }
  const narrow = stage.clientWidth < 460;
  const reach = next === "v3" ? (narrow ? 36 : 40) : narrow ? 34 : 38;
  seats = placeSeats(next === "v3" ? V3_ROLES : V2_ROLES, next, reach);
  if (next !== "v3") return;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.id = "links";
  svg.setAttribute("aria-hidden", "true");
  stage.insertBefore(svg, document.getElementById("pulse-trail"));
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
  const names = { code: "Orbit Code", boardroom: "Orbit Boardroom", incident: "Incident commander" };
  const scenarioName = mission && names[mission.scenario];
  document.body.dataset.scenario = scenarioName ? mission.scenario : "";
  const scenario = document.getElementById("scenario");
  if (scenario) {
    scenario.hidden = !scenarioName;
    scenario.textContent = scenarioName || "";
  }
  document.querySelectorAll(".scenarios button").forEach((button) => {
    const on = Boolean(mission && button.dataset.scenario === mission.scenario);
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.classList.toggle("on", on);
  });
  const note = document.getElementById("scenario-note");
  if (note) {
    note.hidden = !(mission && mission.scenario_note);
    note.textContent = mission && mission.scenario_note ? mission.scenario_note : "";
  }
  const kicker = document.querySelector("#titlecard .kicker");
  if (kicker) kicker.textContent = scenarioName || "Orbit";
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
  if (payload && payload.why && payload.claim) return payload.why;
  if (event.kind === "result") return `${event.tool} ${event.call}`;
  if (event.kind === "edit") return "Edit";
  if (event.kind === "test") return event.ok ? "Tests passed" : "Tests failed";
  if (event.kind === "review") return `${event.verdict}: ${event.reason || ""}`.trim();
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
  if (team === "code") {
    if (event.agent === "planner") return "Planner → Coder";
    if (event.kind === "edit") return "Coder → Test runner";
    if (event.kind === "test") return event.ok ? "Tests pass" : "Tests fail";
    if (event.kind === "review") return event.verdict === "reject" ? "Reviewer rejects" : "Reviewer accepts";
    if (event.kind === "revise") return "Critic → Coder";
    if (event.agent === "critic") return "Critic → Reviewer";
    if (event.kind === "gate") return "Grounding gate";
    if (event.kind === "final") return "Reviewer → Trace";
    return event.agent || "Bus";
  }
  if (team === "boardroom") {
    if (event.agent === "ceo") return "CEO · frame";
    if (event.agent === "cfo") return "CFO → Auditor";
    if (event.agent === "cto") return "CTO → Auditor";
    if (event.agent === "coo") return "COO → Auditor";
    if (event.kind === "audit" || event.kind === "gate") return event.verdict === "refuse" || event.code_verdict === "refuse" ? "Auditor refuses" : "Auditor · check";
    if (event.kind === "final") return "Auditor → Memo";
    return event.agent || "Bus";
  }
  if (team === "v3") {
    if (event.kind === "gate") return "Red team → Verifier";
    if (event.agent === "coordinator") return "Coordinator · open";
    if (event.agent === "researcher") return "Researcher → Tool";
    if (event.agent === "redteam") return "Red team · claim";
    if (event.agent === "verifier") return "Verifier · model";
    return routeV2(event);
  }
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
  const roster = rules.team === "code"
    ? CODE_ROLES
    : rules.team === "boardroom"
      ? BOARD_ROLES
      : rules.team === "v3"
        ? V3_ROLES
        : rules.team === "v2"
          ? V2_ROLES
          : null;
  const tools = (key) => {
    const list = (rules.permissions && rules.permissions[key]) || [];
    return list.length ? ` Tools: ${list.join(", ")}.` : "";
  };
  const roleCards = roster
    ? roster.map(([key, label]) => [label, rules[key] ? `${rules[key]}${tools(key)}` : ""])
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
  if (team === "code") {
    if (event.kind === "review" || event.kind === "gate" || event.kind === "final" || event.agent === "reviewer") return "auditor";
    if (event.agent === "critic" || event.kind === "revise") return "critic";
    if (event.kind === "test" || event.agent === "test_runner") return "tool";
    if (event.kind === "edit" || event.agent === "coder") return "worker";
    if (event.agent === "planner") return "planner";
    return null;
  }
  if (team === "boardroom") {
    if (event.agent === "auditor" || event.kind === "audit" || event.kind === "gate" || event.kind === "final") return "auditor";
    if (event.agent === "ceo") return "planner";
    if (event.agent === "cfo" || event.agent === "cto" || event.agent === "coo") return "worker";
    return null;
  }
  if (team === "v3") {
    if (event.agent === "auditor" || event.kind === "audit" || event.kind === "final") return "auditor";
    if (event.kind === "gate" || event.agent === "verifier" || event.agent === "redteam") return "critic";
    if (event.agent === "critic" || event.kind === "revise") return "critic";
    if (event.agent === "tool" || (event.payload && event.payload.action === "tool")) return "tool";
    if (event.agent === "coordinator" || event.agent === "commander" || event.kind === "assign") return "planner";
    return "worker";
  }
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
  if (event.kind === "edit") return `EDIT\n${String(event.diff || "").slice(0, 360)}`;
  if (event.kind === "test") return `${event.ok ? "PASS" : "FAIL"}  ${event.summary || ""}`;
  if (event.kind === "review") return `REVIEW  ${event.verdict}\n${event.reason || ""}`;
  if (event.kind === "gate") return `GATE  ${event.code_verdict}\n${event.reason || ""}`;
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
    return event.payload.steps.map((step, index) => `${index + 1}. ${step.tool ? `${step.tool}  ` : ""}${step.goal}`).join("\n");
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
  if (event.kind === "gate") pulse("redteam", "verifier");
  if (event.kind === "audit" && auditor) auditor.dataset.verdict = event.verdict;
}

function focusCode(event) {
  const name = event.agent;
  if (seats[name]) seats[name].classList.add("lit");
  if (event.agent === "planner") pulse("planner", "coder");
  else if (event.kind === "edit") pulse("coder", "test_runner");
  else if (event.kind === "test") pulse("test_runner", "reviewer");
  else if (event.kind === "review" && event.verdict === "reject") pulse("reviewer", "critic");
  else if (event.kind === "review") pulse("critic", "reviewer");
  else if (event.kind === "revise") pulse("critic", "coder");
  else if (event.agent === "critic") pulse("critic", "reviewer");
  else if (event.agent === "ceo") pulse("ceo", "cfo");
  else if (event.agent === "cfo") pulse("cfo", "cto");
  else if (event.agent === "cto") pulse("cto", "coo");
  else if (event.agent === "coo") pulse("coo", "auditor");
  else if (event.kind === "audit" || event.kind === "gate") {
    if (seats.auditor) seats.auditor.dataset.verdict = event.verdict || event.code_verdict;
  }
  const reviewer = seats.reviewer;
  if (reviewer && event.kind === "review") reviewer.dataset.verdict = event.verdict === "reject" ? "refuse" : "pass";
}

function focusAgent(event) {
  for (const seat of Object.values(seats)) seat.classList.remove("lit");
  if (team === "code" || team === "boardroom") {
    focusCode(event);
    return;
  }
  if (team === "v2" || team === "v3") {
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

function drawLinks(shown) {
  const svg = document.getElementById("links");
  const stage = document.getElementById("stage");
  if (!svg) return;
  svg.setAttribute("viewBox", `0 0 ${stage.clientWidth} ${stage.clientHeight}`);
  svg.replaceChildren();
  const points = [];
  for (const event of shown) {
    const name = event.kind === "assign" ? event.role : event.agent === "tool" ? event.by : event.agent;
    if (!seats[name]) continue;
    const point = center(seats[name]);
    const previous = points[points.length - 1];
    if (!previous || previous.name !== name) points.push({ name, x: point.x, y: point.y });
  }
  for (let index = 1; index < points.length; index += 1) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(points[index - 1].x));
    line.setAttribute("y1", String(points[index - 1].y));
    line.setAttribute("x2", String(points[index].x));
    line.setAttribute("y2", String(points[index].y));
    if (index === points.length - 1) line.setAttribute("class", "live");
    svg.append(line);
  }
}

function paintDiff(text) {
  const pre = document.getElementById("diffview");
  if (!pre) return;
  pre.replaceChildren();
  const source = text || "";
  if (!source.trim()) {
    pre.hidden = true;
    return;
  }
  pre.hidden = false;
  const keywords = /^(def|return|if|else|for|in|class|None|True|False|import|not|and|or)$/;
  for (const line of source.split("\n")) {
    const row = document.createElement("div");
    let body = line;
    const marker = (line.startsWith("+") || line.startsWith("-")) && !line.startsWith("+++") && !line.startsWith("---");
    if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("@@")) row.className = "meta";
    else if (line.startsWith("+")) row.className = "add";
    else if (line.startsWith("-")) row.className = "del";
    if (marker) {
      body = line.slice(1);
      row.append(line.slice(0, 1));
    }
    const parts = body.split(/(\b(?:def|return|if|else|for|in|class|None|True|False|import|not|and|or)\b|"[^"]*"|'[^']*')/g);
    for (const part of parts) {
      if (!part) continue;
      const span = document.createElement("span");
      if (keywords.test(part)) span.className = "kw";
      else if (part.startsWith('"') || part.startsWith("'")) span.className = "str";
      span.textContent = part;
      row.append(span);
    }
    if (!line) row.textContent = " ";
    pre.append(row);
  }
}

function paintLanes(shown) {
  const root = document.getElementById("lanes");
  if (!root) return;
  const roles = team === "code" ? CODE_ROLES : team === "boardroom" ? BOARD_ROLES : [];
  root.replaceChildren();
  if (!roles.length) return;
  for (const [key, label] of roles) {
    const row = document.createElement("div");
    row.className = "lane";
    const name = document.createElement("span");
    name.textContent = label;
    const ticks = document.createElement("div");
    ticks.className = "ticks";
    for (const event of shown) {
      const mark = document.createElement("i");
      const hit = event.agent === key || (key === "reviewer" && event.kind === "review") || (key === "auditor" && (event.kind === "audit" || event.kind === "gate"));
      if (hit) {
        const bad = event.verdict === "reject" || event.verdict === "refuse" || event.code_verdict === "refuse" || event.ok === false;
        const good = event.verdict === "accept" || event.verdict === "pass" || event.code_verdict === "pass" || event.ok === true;
        mark.className = bad ? "on bad" : good ? "on good" : "on";
      }
      ticks.append(mark);
    }
    row.append(name, ticks);
    root.append(row);
  }
}

function paintTests(shown) {
  const lamp = document.getElementById("testlamp");
  if (!lamp) return;
  const event = [...shown].reverse().find((item) => item.kind === "test");
  if (!event || team !== "code") {
    lamp.hidden = true;
    return;
  }
  lamp.hidden = false;
  lamp.className = "testlamp " + (event.ok ? "pass" : "fail");
  lamp.textContent = event.ok ? "Tests passed" : "Tests failed";
}

function paintWork(shown) {
  paintLanes(shown);
  paintTests(shown);
  if (team === "code") {
    const edit = [...shown].reverse().find((item) => item.kind === "edit");
    paintDiff(edit ? edit.diff : "");
    return;
  }
  if (team === "boardroom") {
    const audit = [...shown].reverse().find((item) => item.kind === "audit");
    if (!audit) {
      paintDiff("");
      return;
    }
    const lines = (audit.findings || []).map((finding) => {
      const mark = finding.ok ? "keep" : "refuse";
      return `${mark}  ${finding.claim}  ${finding.source || ""}`;
    });
    paintDiff(lines.join("\n"));
    const pre = document.getElementById("diffview");
    if (pre) {
      [...pre.children].forEach((row) => {
        if (row.textContent.startsWith("refuse")) row.classList.add("del");
        if (row.textContent.startsWith("keep")) row.classList.add("add");
      });
    }
  }
}

function refusalText(shown, mission) {
  const scenario = mission && mission.scenario;
  if (scenario === "code") {
    const review = [...shown].reverse().find((event) => event.kind === "review");
    if (review && review.verdict === "reject") return review.reason || "The reviewer rejected the patch.";
    const gate = [...shown].reverse().find((event) => event.kind === "gate" && event.code_verdict === "refuse");
    return gate ? gate.reason || "" : "";
  }
  if (scenario === "boardroom") {
    const audit = [...shown].reverse().find((event) => event.kind === "audit" && event.verdict === "refuse");
    if (!audit) return "";
    const missed = (audit.findings || []).filter((item) => !item.ok).map((item) => item.claim).filter(Boolean);
    const names = missed.length ? missed.join(", ") : "a claim";
    return `The auditor refused ${names}. The cited file does not support that figure.`;
  }
  const bits = [];
  const gate = [...shown].reverse().find((event) => event.kind === "gate" && event.code_verdict === "refuse");
  if (gate) {
    const blame = gate.claim && gate.claim.name === "inventory_errors";
    bits.push(
      blame
        ? "The red team blamed the inventory service. The cited tool result does not support that root cause. The code gate refused it. That claim is not the team's answer."
        : "The red team proposed a count the tool results do not contain. The code gate refused it. That claim is not the team's answer.",
    );
  }
  const audit = [...shown].reverse().find((event) => event.kind === "audit" && event.verdict === "refuse");
  if (audit) bits.push(audit.reason || "The auditor refused the draft because a claim or a number is not in the cited tool results.");
  const done = shown.length > 0 && shown.length === state.events.length && mission && mission.status === "refuse";
  if (done && !audit && mission.answer) bits.push(mission.answer);
  return bits.join(" ");
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
    if ((event.kind === "review" && event.verdict === "reject") || (event.kind === "audit" && event.verdict === "refuse") || (event.kind === "gate" && event.code_verdict === "refuse")) {
      item.classList.add("reject");
    }
    if (event.kind === "test") item.classList.add(event.ok ? "pass" : "fail");
    item.append(time, body);
    log.append(item);
  }
  paintWork(shown);
  const last = shown[shown.length - 1];
  paintPhase(shown);
  if (team === "v3") drawLinks(shown);
  const stage = document.getElementById("stage");
  let verdict = null;
  for (const event of shown) {
    if (event.kind === "review") verdict = event.verdict === "reject" ? "refuse" : "pass";
    if (event.kind === "gate") verdict = event.code_verdict;
    if (event.kind === "audit") verdict = event.verdict;
  }
  const flash = verdict === "refuse" ? "flash-refuse" : verdict === "pass" ? "flash-pass" : "";
  if (flash) {
    if (!stage.classList.contains(flash)) {
      stage.classList.remove("flash-pass", "flash-refuse");
      stage.classList.add(flash);
    }
  } else {
    stage.classList.remove("flash-pass", "flash-refuse");
  }
  const crew = document.getElementById("crew");
  if (crew) {
    const name = last ? (last.agent === "tool" ? last.by : last.kind === "assign" ? last.role : last.agent) : "";
    crew.textContent = ROLE_LINES[name] || "";
  }
  const why = document.getElementById("why");
  const whyBody = document.getElementById("why-text");
  if (why && whyBody) {
    const text = refusalText(shown, state.mission);
    why.hidden = !text;
    whyBody.textContent = text;
  }
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
  try {
    const v3Response = await fetch("traces/v3/index.json");
    if (v3Response.ok) {
      const v3Index = await v3Response.json();
      const v3Group = document.createElement("optgroup");
      v3Group.label = "Round table";
      for (const item of v3Index) {
        const option = document.createElement("option");
        option.value = `v3/${item.file}`;
        option.textContent = `${item.task_id} · ${item.outcome || item.status}`;
        v3Group.append(option);
        index.push({ file: option.value });
      }
      select.append(v3Group);
    }
  } catch (error) {
    setNotice("");
  }
  try {
    const incidentResponse = await fetch("traces/incident/index.json");
    if (incidentResponse.ok) {
      const incidentIndex = await incidentResponse.json();
      const incidentGroup = document.createElement("optgroup");
      incidentGroup.label = "Incident commander";
      for (const item of incidentIndex) {
        const option = document.createElement("option");
        option.value = `incident/${item.file}`;
        option.textContent = `${item.task_id} · ${item.status}`;
        incidentGroup.append(option);
        index.push({ file: option.value });
      }
      select.prepend(incidentGroup);
    }
  } catch (error) {
    setNotice("");
  }
  const requested = params.get("trace");
  try {
    const codeResponse = await fetch("traces/code/index.json");
    if (codeResponse.ok) {
      const codeIndex = await codeResponse.json();
      const codeGroup = document.createElement("optgroup");
      codeGroup.label = "Orbit Code";
      for (const item of codeIndex) {
        const option = document.createElement("option");
        option.value = `code/${item.file}`;
        option.textContent = `${item.task_id} · ${item.outcome || item.status}`;
        codeGroup.append(option);
        index.push({ file: option.value });
      }
      select.prepend(codeGroup);
    }
  } catch (error) {
    setNotice("");
  }
  try {
    const boardResponse = await fetch("traces/boardroom/index.json");
    if (boardResponse.ok) {
      const boardIndex = await boardResponse.json();
      const boardGroup = document.createElement("optgroup");
      boardGroup.label = "Orbit Boardroom";
      for (const item of boardIndex) {
        const option = document.createElement("option");
        option.value = `boardroom/${item.file}`;
        option.textContent = `${item.task_id} · ${item.status}`;
        boardGroup.append(option);
        index.push({ file: option.value });
      }
      const codeGroup = select.querySelector("optgroup");
      if (codeGroup) codeGroup.after(boardGroup);
      else select.prepend(boardGroup);
    }
  } catch (error) {
    setNotice("");
  }
  const featured = "code/repair.jsonl";
  const choice = index.some((item) => item.file === requested)
    ? requested
    : index.some((item) => item.file === featured)
      ? featured
      : "revise.jsonl";
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
const SCENARIO_TRACE = {
  code: "code/repair.jsonl",
  boardroom: "boardroom/memo.jsonl",
  incident: "incident/report.jsonl",
};

document.querySelectorAll(".scenarios button").forEach((button) => {
  button.addEventListener("click", () => {
    const preferred = SCENARIO_TRACE[button.dataset.scenario];
    const prefix = `${button.dataset.scenario}/`;
    const options = [...document.getElementById("trace").options];
    const option = options.find((item) => item.value === preferred) || options.find((item) => item.value.startsWith(prefix));
    if (!option) return;
    document.getElementById("trace").value = option.value;
    params.delete("autoplay");
    loadTrace(option.value).catch((error) => setNotice(error.message));
  });
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
