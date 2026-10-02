"use strict";

const $ = (id) => document.getElementById(id);
const TERMINAL = new Set(["completed", "failed", "interrupted", "cancelled"]);
const RESUMABLE = new Set(["failed", "interrupted"]);
const AGENTS = ["supervisor", "search", "analyst", "writer"];

const state = {
  source: null,
  session: null,
  runs: [],
  currentRun: null,
  reportText: "",
  receivedTerminal: false,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const raw = await response.text();
  let body = null;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    body = raw;
  }
  if (!response.ok) {
    const detail = body && typeof body === "object" ? body.detail : body;
    throw new Error(detail || `请求失败（HTTP ${response.status}）`);
  }
  return body;
}

function setNotice(message = "", kind = "") {
  const notice = $("notice");
  notice.textContent = message;
  notice.className = `notice${message ? " visible" : ""}${kind ? ` ${kind}` : ""}`;
}

function setRunStatus(status) {
  const badge = $("run-status");
  badge.textContent = status || "draft";
  badge.className = `badge ${status || ""}`;
}

function rememberSession(sessionId) {
  if (!sessionId) return;
  localStorage.setItem("multi-agent:last-session", sessionId);
  const url = new URL(window.location.href);
  url.searchParams.set("session_id", sessionId);
  history.replaceState(null, "", url);
}

function shortId(value) {
  if (!value) return "—";
  return value.length > 22 ? `${value.slice(0, 10)}…${value.slice(-8)}` : value;
}

function closeStream() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
}

function resetPipeline() {
  renderSections([]);
  for (const name of AGENTS) $("agent-" + name).className = "agent";
  $("meta-supervisor").textContent = "任务分解与路由决策";
  $("meta-search").textContent = "knowledge-service · Tavily";
  $("meta-analyst").textContent = "证据完整性审查";
  $("meta-writer").textContent = "结构化报告生成";
  $("event-log").replaceChildren();
  $("event-cursor").textContent = "event #0";
}

function renderSections(sections = []) {
  $("sections-panel").classList.toggle("hidden", sections.length === 0);
  const labels = { pending: "待研究", researching: "研究中", drafted: "待审校", complete: "已审校", limited: "存在局限" };
  const nodes = sections.map((section) => {
    const details = document.createElement("details");
    details.className = "section-artifact";
    const summary = document.createElement("summary");
    summary.textContent = `${section.title} · ${labels[section.status] || section.status} · 第 ${section.revision} 版`;
    const content = document.createElement("div");
    content.style.whiteSpace = "pre-wrap";
    const sources = (section.sources || []).map((source, index) => {
      const meta = source.metadata || {};
      return `[来源${index + 1}] ${meta.source || meta.title || ""} ${meta.page > 0 ? `p.${meta.page}` : ""} ${meta.url || ""}`;
    });
    content.textContent = `${section.question}\n\n${section.draft || "尚无草稿"}\n\n${(section.limitations || []).join("\n")}\n\n${sources.join("\n")}`;
    details.append(summary, content);
    return details;
  });
  $("sections-list").replaceChildren(...nodes);
}

function setAgent(name, status, detail) {
  const node = $("agent-" + name);
  if (!node) return;
  node.className = `agent ${status || ""}`;
  if (detail) $("meta-" + name).textContent = detail;
}

function appendLog(type, message, eventId) {
  const row = document.createElement("div");
  row.className = "event-line";
  const time = document.createElement("span");
  time.className = "event-time";
  time.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  const text = document.createElement("span");
  text.textContent = `${type}: ${message}`;
  row.append(time, text);
  $("event-log").append(row);
  $("event-log").scrollTop = $("event-log").scrollHeight;
  if (eventId) $("event-cursor").textContent = `event #${eventId}`;
}

function statusLabel(status) {
  return {
    created: "待启动",
    running: "执行中",
    completed: "已完成",
    failed: "失败",
    interrupted: "已中断",
    cancelled: "已取消",
  }[status] || status;
}

function renderRunTree() {
  const root = $("run-tree");
  root.replaceChildren();
  $("run-count").textContent = `${state.runs.length} runs`;
  if (!state.runs.length) {
    const empty = document.createElement("div");
    empty.className = "tree-empty";
    empty.textContent = "此 Session 还没有 Run。";
    root.append(empty);
    return;
  }

  const ids = new Set(state.runs.map((run) => run.run_id));
  const children = new Map();
  for (const run of state.runs) {
    const parent = run.parent_run_id && ids.has(run.parent_run_id)
      ? run.parent_run_id
      : null;
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(run);
  }

  const appendNode = (run, depth) => {
    const node = document.createElement("article");
    node.className = `run-node${state.currentRun?.run_id === run.run_id ? " selected" : ""}`;
    node.style.setProperty("--depth", String(depth));
    node.dataset.runId = run.run_id;

    const top = document.createElement("div");
    top.className = "run-node-top";
    const id = document.createElement("span");
    id.className = "run-id";
    id.textContent = shortId(run.run_id);
    id.title = run.run_id;
    const badge = document.createElement("span");
    badge.className = `badge ${run.status}`;
    badge.textContent = statusLabel(run.status);
    top.append(id, badge);

    const question = document.createElement("p");
    question.className = "run-question";
    question.textContent = run.question;
    question.title = run.question;
    node.append(top, question);
    node.addEventListener("click", () => openRun(run.run_id));
    root.append(node);
    for (const child of children.get(run.run_id) || []) appendNode(child, depth + 1);
  };

  for (const run of children.get(null) || []) appendNode(run, 0);
}

async function loadSession(sessionId = $("session-id").value.trim(), options = {}) {
  if (!sessionId) {
    setNotice("请输入 Session ID。", "error");
    return;
  }
  try {
    const timeline = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    state.session = timeline.session;
    state.runs = timeline.runs;
    $("session-id").value = timeline.session.session_id;
    $("session-title").textContent = timeline.session.title;
    rememberSession(timeline.session.session_id);
    renderRunTree();
    if (!options.quiet) setNotice(`已加载 ${timeline.runs.length} 个 Run。`, "success");
  } catch (error) {
    if (!options.quiet) setNotice(error.message, "error");
    throw error;
  }
}

async function createSession() {
  const title = $("question").value.trim().slice(0, 200) || "新的研究 Session";
  try {
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title }),
    });
    closeStream();
    state.session = session;
    state.runs = [];
    state.currentRun = null;
    $("session-id").value = session.session_id;
    $("session-title").textContent = session.title;
    $("run-id").value = "";
    $("parent-run-id").value = "";
    rememberSession(session.session_id);
    renderRunTree();
    updateRunActions(null);
    clearReport();
    setNotice("已创建空白 Session，可以开始第一个研究任务。", "success");
  } catch (error) {
    setNotice(error.message, "error");
  }
}

function updateRunActions(run) {
  const completed = run?.status === "completed";
  const resumable = run && RESUMABLE.has(run.status);
  $("start-btn").classList.toggle("hidden", Boolean(run));
  $("start-existing-btn").classList.toggle("hidden", run?.status !== "created");
  $("continue-btn").classList.toggle("hidden", !completed);
  $("use-as-parent-btn").classList.toggle("hidden", !completed);
  $("resume-btn").classList.toggle("hidden", !resumable);
  $("copy-btn").classList.toggle("hidden", !state.reportText);
  setRunStatus(run?.status || "draft");
}

function clearReport() {
  state.reportText = "";
  $("report-content").replaceChildren();
  $("report-content").classList.add("hidden");
  $("report-empty").classList.remove("hidden");
  $("stats").classList.add("hidden");
  $("copy-btn").classList.add("hidden");
  $("report-title").textContent = "研究报告";
  $("report-subtitle").textContent = "选择历史 Run 或启动新任务。";
}

function sanitizeMarkup(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  template.content.querySelectorAll("script, iframe, object, embed, form").forEach((node) => node.remove());
  template.content.querySelectorAll("*").forEach((node) => {
    for (const attribute of [...node.attributes]) {
      const name = attribute.name.toLowerCase();
      const value = attribute.value.trim().toLowerCase();
      if (name.startsWith("on") || ((name === "href" || name === "src") && value.startsWith("javascript:"))) {
        node.removeAttribute(attribute.name);
      }
    }
  });
  return template.innerHTML;
}

function renderReport(report, metrics = {}) {
  state.reportText = report || "";
  if (!state.reportText) return;
  const content = $("report-content");
  if (window.marked?.parse) {
    content.innerHTML = sanitizeMarkup(window.marked.parse(state.reportText));
  } else {
    content.textContent = state.reportText;
    content.style.whiteSpace = "pre-wrap";
  }
  $("report-empty").classList.add("hidden");
  content.classList.remove("hidden");
  $("stats").classList.remove("hidden");
  $("copy-btn").classList.remove("hidden");

  const reportBody = state.reportText.split(/##\s*(?:参考来源|参考资料|References)/i)[0];
  const citationIds = new Set([...reportBody.matchAll(/\[来源\s*(\d+)\]/g)].map((match) => match[1]));
  $("stat-citations").textContent = citationIds.size;
  $("stat-chars").textContent = (metrics.char_count ?? state.reportText.length).toLocaleString();
  if (metrics.total_iterations != null) $("stat-iterations").textContent = metrics.total_iterations;
  if (metrics.total_results != null) $("stat-results").textContent = metrics.total_results;
  if (metrics.token_budget_used != null) $("stat-tokens").textContent = metrics.token_budget_used.toLocaleString();
}

async function openRun(runId) {
  closeStream();
  setNotice();
  resetPipeline();
  try {
    const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
    state.currentRun = run;
    $("run-id").value = run.run_id;
    $("session-id").value = run.session_id || "";
    $("parent-run-id").value = run.parent_run_id || "";
    $("question").value = run.question;
    state.reportText = "";
    if (run.final_report) renderReport(run.final_report);
    else clearReport();
    renderSections(run.sections || []);
    $("report-title").textContent = `研究报告 · ${shortId(run.run_id)}`;
    $("report-subtitle").textContent = `${statusLabel(run.status)} · 创建于 ${new Date(run.created_at).toLocaleString("zh-CN")}`;
    updateRunActions(run);
    renderRunTree();
    if (run.status !== "created") subscribeRun(run);
  } catch (error) {
    setNotice(error.message, "error");
  }
}

function eventMessage(type, data) {
  switch (type) {
    case "section_plan": return `章节计划已生成，共 ${data.sections?.length || 0} 章`;
    case "section_snapshot": return "已恢复章节草稿与研究进度";
    case "section_progress": return `${data.stage} · 已完成 ${(data.sections || []).filter(s => ["complete", "limited"].includes(s.status)).length}/${data.sections?.length || 0} 章`;
    case "run_created": return `任务已创建${data.parent_run_id ? `，父 Run ${shortId(data.parent_run_id)}` : ""}`;
    case "run_started": return "后台执行已启动";
    case "run_resumed": return "已从原 Checkpoint 恢复";
    case "start": return data.resumed ? "LangGraph 恢复执行" : "LangGraph 开始执行";
    case "supervisor_decision": return `第 ${data.iteration} 轮 → ${data.next || "结束"}`;
    case "search_complete": return `新增 ${data.new_count} 条，累计 ${data.total_count} 条`;
    case "analyst_verdict": return `${data.verdict} · 置信度 ${Math.round((data.confidence || 0) * 100)}%`;
    case "report_ready": return `报告生成完成，共 ${data.char_count || 0} 字`;
    case "done": return "研究任务已完成";
    case "run_interrupted": return `任务中断：${data.reason || "未知原因"}`;
    case "error": return data.message || "任务失败";
    default: return "收到状态更新";
  }
}

function handleRunEvent(type, event) {
  let data = {};
  try { data = JSON.parse(event.data); } catch { data = { message: event.data }; }
  appendLog(type, eventMessage(type, data), event.lastEventId);

  if (data.sections) {
    renderSections(data.sections);
    if (state.currentRun) state.currentRun.sections = data.sections;
  }
  if (type.startsWith("section_")) {
    const target = { plan_sections: "supervisor", section_search: "search", section_analyze: "analyst", section_write: "writer", section_review: "analyst" }[data.stage];
    if (target) setAgent(target, "done", eventMessage(type, data));
  }

  if (type === "run_started" || type === "run_resumed" || type === "start") {
    setRunStatus("running");
    setAgent("supervisor", "active", type === "run_resumed" ? "读取原 Run Checkpoint" : "等待首次路由决策");
  } else if (type === "supervisor_decision") {
    setAgent("supervisor", "done", `第 ${data.iteration} 轮 → ${data.next || "结束"}`);
    const target = { search_agent: "search", analyst_agent: "analyst", writer_agent: "writer" }[data.next];
    if (target) setAgent(target, "active", data.reason || "正在执行");
  } else if (type === "search_complete") {
    setAgent("search", "done", `+${data.new_count} new · ${data.total_count} total`);
  } else if (type === "analyst_verdict") {
    setAgent("analyst", "done", `${data.verdict} · ${Math.round((data.confidence || 0) * 100)}% · ${data.gaps?.length || 0} gaps`);
  } else if (type === "report_ready") {
    setAgent("writer", "done", `${data.char_count || 0} 字 · ${data.writer_status || "完成"}`);
  } else if (type === "done") {
    state.receivedTerminal = true;
    setRunStatus("completed");
    setAgent("writer", "done", `${data.char_count || 0} 字 · 完成`);
    renderReport(data.report || state.reportText, data);
    if (state.currentRun) state.currentRun.status = "completed";
    updateRunActions(state.currentRun);
    closeStream();
    setNotice(data.report_quality === "limited"
      ? "报告已装配，部分章节存在未解决的证据或审校缺口，请阅读局限说明。"
      : "任务已完成；断开页面不会影响后台 Run。", data.report_quality === "limited" ? "" : "success");
    window.setTimeout(() => loadSession($("session-id").value.trim(), { quiet: true }).catch(() => {}), 350);
  } else if (type === "error" || type === "run_interrupted") {
    state.receivedTerminal = true;
    const status = type === "error" ? "failed" : "interrupted";
    setRunStatus(status);
    for (const name of AGENTS) {
      if ($("agent-" + name).classList.contains("active")) setAgent(name, "error", eventMessage(type, data));
    }
    if (state.currentRun) state.currentRun.status = status;
    updateRunActions(state.currentRun);
    closeStream();
    setNotice(eventMessage(type, data), "error");
    loadSession($("session-id").value.trim(), { quiet: true }).catch(() => {});
  }
}

function subscribeRun(run) {
  closeStream();
  state.receivedTerminal = false;
  const source = new EventSource(`/api/runs/${encodeURIComponent(run.run_id)}/stream`);
  state.source = source;
  const eventTypes = [
    "run_created", "run_started", "run_resumed", "start", "supervisor_decision",
    "search_complete", "analyst_verdict", "report_ready", "done", "run_interrupted", "error",
    "section_plan", "section_progress", "section_snapshot",
  ];
  for (const type of eventTypes) source.addEventListener(type, (event) => handleRunEvent(type, event));
  source.onerror = () => {
    if (state.receivedTerminal || TERMINAL.has(state.currentRun?.status)) {
      closeStream();
      return;
    }
    appendLog("connection", "事件流暂时断开，浏览器将自动重连", "");
  };
}

async function startResearch() {
  const question = $("question").value.trim();
  if (question.length < 5) {
    setNotice("研究问题至少需要 5 个字符。", "error");
    return;
  }
  const button = $("start-btn");
  button.disabled = true;
  closeStream();
  resetPipeline();
  clearReport();
  setNotice("正在创建 Run…");
  try {
    const payload = { question };
    const sessionId = $("session-id").value.trim();
    const parentRunId = $("parent-run-id").value.trim();
    if (sessionId) payload.session_id = sessionId;
    if (parentRunId) payload.parent_run_id = parentRunId;

    const created = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
    state.currentRun = created;
    $("run-id").value = created.run_id;
    $("session-id").value = created.session_id;
    rememberSession(created.session_id);
    setRunStatus("created");
    updateRunActions(created);
    await loadSession(created.session_id, { quiet: true });

    await startExistingRun();
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function startExistingRun() {
  const run = state.currentRun;
  if (!run || run.status !== "created") return;
  $("start-existing-btn").disabled = true;
  try {
    const running = await api(`/api/runs/${encodeURIComponent(run.run_id)}/start`, { method: "POST" });
    state.currentRun = running;
    setRunStatus("running");
    updateRunActions(running);
    renderRunTree();
    subscribeRun(running);
    setNotice("Run 已在后台执行。关闭或刷新页面不会取消任务。", "success");
  } catch (error) {
    setNotice(error.message, "error");
    throw error;
  } finally {
    $("start-existing-btn").disabled = false;
  }
}

function prepareNewRun() {
  closeStream();
  state.currentRun = null;
  $("run-id").value = "";
  $("parent-run-id").value = "";
  $("question").value = "";
  resetPipeline();
  clearReport();
  updateRunActions(null);
  renderRunTree();
  setNotice("已进入空白 Run 草稿；仍保留当前 Session，正在运行的后台任务不会被取消。", "success");
  $("question").focus();
}

function prepareContinuation() {
  const run = state.currentRun;
  if (!run || run.status !== "completed") return;
  $("parent-run-id").value = run.run_id;
  $("run-id").value = "";
  $("question").value = "";
  state.currentRun = null;
  updateRunActions(null);
  renderRunTree();
  setNotice("已选定父 Run。输入新问题后会创建一个独立的子 Run，不会覆写父 Run。", "success");
  $("question").focus();
}

async function resumeCurrentRun() {
  const run = state.currentRun;
  if (!run || !RESUMABLE.has(run.status)) return;
  $("resume-btn").disabled = true;
  closeStream();
  resetPipeline();
  try {
    const running = await api(`/api/runs/${encodeURIComponent(run.run_id)}/resume`, { method: "POST" });
    state.currentRun = running;
    setRunStatus("running");
    updateRunActions(running);
    subscribeRun(running);
    setNotice("正在恢复同一 Run：run_id 不变，并从已有 Checkpoint 继续。", "success");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    $("resume-btn").disabled = false;
  }
}

async function copyReport() {
  if (!state.reportText) return;
  try {
    await navigator.clipboard.writeText(state.reportText);
    setNotice("报告已复制到剪贴板。", "success");
  } catch {
    setNotice("浏览器未授予剪贴板权限。", "error");
  }
}

async function checkHealth() {
  const node = $("health");
  try {
    const health = await api("/api/health");
    const ok = health.status === "ok" && health.run_store === "ok" && health.checkpointer?.postgres === "ok";
    node.className = `health ${ok ? "ok" : "error"}`;
    $("health-text").textContent = ok ? "服务与存储正常" : "部分依赖异常";
  } catch {
    node.className = "health error";
    $("health-text").textContent = "服务不可用";
  }
}

$("load-session-btn").addEventListener("click", () => loadSession().catch(() => {}));
$("session-id").addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadSession().catch(() => {});
});
$("new-session-btn").addEventListener("click", createSession);
$("new-run-btn").addEventListener("click", prepareNewRun);
$("start-btn").addEventListener("click", startResearch);
$("start-existing-btn").addEventListener("click", () => startExistingRun().catch(() => {}));
$("continue-btn").addEventListener("click", prepareContinuation);
$("use-as-parent-btn").addEventListener("click", prepareContinuation);
$("resume-btn").addEventListener("click", resumeCurrentRun);
$("copy-btn").addEventListener("click", copyReport);
$("clear-parent-btn").addEventListener("click", () => { $("parent-run-id").value = ""; });
window.addEventListener("beforeunload", closeStream);

checkHealth();
const initialSession = new URLSearchParams(location.search).get("session_id")
  || localStorage.getItem("multi-agent:last-session");
if (initialSession) {
  $("session-id").value = initialSession;
  loadSession(initialSession, { quiet: true }).catch(() => {
    setNotice("上次使用的 Session 已不存在或暂时无法加载。", "error");
  });
}
