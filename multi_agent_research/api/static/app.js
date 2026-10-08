"use strict";

const $ = (id) => document.getElementById(id);
const TERMINAL = new Set(["completed", "failed", "interrupted", "cancelled", "paused", "budget_limited"]);
const RESUMABLE = new Set(["failed", "interrupted", "paused", "budget_limited"]);
const AGENTS = ["supervisor", "search", "analyst", "writer"];

const state = {
  source: null,
  session: null,
  runs: [],
  currentRun: null,
  reportText: "",
  receivedTerminal: false,
  cursor: 0,
  viewRequest: 0,
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
  renderReportReview(null);
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
  const labels = { pending: "待研究", stale: "依赖变更，待重建（下方是旧稿）", researching: "研究中", drafted: "待审校", complete: "已审校", limited: "存在局限", evidence_ready: "证据已补充，正文待更新（下方为旧稿）", waiting_evidence: "待补证据（下方为旧稿）" };
  const nodes = sections.map((section) => {
    const details = document.createElement("details");
    details.className = "section-artifact";
    const summary = document.createElement("summary");
    summary.textContent = `${section.title} · ${section.status === "claims_pending" ? "结论关联待修复（未完成核验）" : labels[section.status] || section.status} · 第 ${section.revision} 版`;
    const content = document.createElement("div");
    content.style.whiteSpace = "pre-wrap";
    const sources = (section.sources || []).map((source, index) => {
      const meta = source.metadata || {};
      return `[来源${index + 1}] ${meta.source || meta.title || ""} ${meta.page > 0 ? `p.${meta.page}` : ""} ${meta.url || ""}\nchunk: ${meta.chunk_id || "—"} · 检索时间: ${meta.retrieved_at || "未知"}${meta.inherited_from_run ? ` · 继承自 ${meta.inherited_from_run}，非本轮新检索` : ""}`;
    });
    const assessments = { supported: "模型判断有支持", uncertain: "不确定", unsupported: "缺少支持" };
    const relations = { supports: "支持", contradicts: "反对", context: "背景" };
    const claims = (section.claims || []).map(claim => `${claim.claim_id} · ${assessments[claim.assessment]}\n${claim.statement}\n${claim.caveat || ""}\n${(claim.evidence || []).map(link => `${relations[link.relation]} [来源${link.source_number}]：${link.quote}\n证据 ID：${link.evidence_id}`).join("\n")}`);
    content.textContent = `${section.question}\n依赖：${(section.depends_on || []).join(", ") || "无"}\n历史版本：${(section.previous_drafts || []).length}\n\n${section.draft || "尚无草稿"}\n\n${(section.limitations || []).join("\n")}\n\n${claims.join("\n\n")}\n\n${sources.join("\n")}`;
    details.append(summary, content);
    if (section.claim_work?.pending?.length || section.claim_work?.batch_errors?.length) {
      const pending = document.createElement("details");
      const title = document.createElement("summary");
      title.textContent = `已保存 ${section.claims?.length || 0} 条通过校验的关联；${section.claim_work.pending?.length || 0} 条待修复`;
      const body = document.createElement("div");
      body.style.whiteSpace = "pre-wrap";
      body.textContent = (section.claim_work.pending || []).map(p =>
        `Claim ${p.slot}（未通过校验）：${p.candidate?.statement || "字段格式待修复"}\n${p.errors.map(e => e.message || e.type).join("\n")}`).join("\n\n") || "批次格式未通过，尚不能定位具体 Claim。";
      pending.append(title, body);
      details.append(pending);
    }
    if (state.currentRun?.status === "completed" && sections.every(s => ["complete", "limited"].includes(s.status))) {
      const runId = state.currentRun.run_id;
      const input = document.createElement("textarea");
      input.placeholder = "本章需要补充哪些证据或修改哪些结论？（至少 5 字）";
      input.setAttribute("aria-label", `${section.title}修订要求`);
      input.maxLength = 2000;
      const button = document.createElement("button");
      button.className = "btn btn-ghost";
      button.textContent = "创建本章修订 Run";
      button.onclick = async () => {
        const instruction = input.value.trim();
        if (instruction.length < 5) { setNotice("请填写至少 5 字的修订要求。", "error"); return; }
        button.disabled = true;
        try {
          const created = await api(`/api/runs/${encodeURIComponent(runId)}/sections/${encodeURIComponent(section.section_id)}/revisions`, {
            method: "POST", body: JSON.stringify({ instruction }),
          });
          await loadSession(created.session_id, { quiet: true });
          await openRun(created.run_id);
          setNotice("已创建修订 Run；核对待重建章节后点击启动。原报告保持不变。", "success");
        } catch (error) { setNotice(error.message, "error"); }
        finally { button.disabled = false; }
      };
      details.append(input, button);
    }
    if (["completed", "paused", "failed", "interrupted", "budget_limited"].includes(state.currentRun?.status)) {
      const runId = state.currentRun.run_id;
      const panel = document.createElement("div");
      const description = document.createElement("p");
      description.textContent = `局部操作创建子 Run，共用累计预算，不改原 Run。依赖本章的章节可能需更新。候选证据：${(section.results || []).length} 条。`;
      const instruction = document.createElement("textarea");
      instruction.maxLength = 2000;
      instruction.placeholder = "说明本次继续/补证据/刷新要求（至少 5 字）";
      instruction.setAttribute("aria-label", `${section.title}局部操作要求`);
      panel.append(description, instruction);
      const modes = [["continue", "选择本章继续"], ["supplement", "仅补证据，不改正文"], ["refresh", "刷新来源并修订"]];
      for (const [mode, label] of modes) {
        if (mode === "continue" && ["complete", "limited"].includes(section.status)) continue;
        const button = document.createElement("button");
        button.className = "btn btn-ghost";
        button.textContent = label;
        button.onclick = async () => {
          const text = instruction.value.trim();
          if (text.length < 5) { setNotice("请填写至少 5 字的操作要求。", "error"); return; }
          button.disabled = true;
          try {
            const created = await api(`/api/runs/${encodeURIComponent(runId)}/sections/${encodeURIComponent(section.section_id)}/operations`, {
              method: "POST", body: JSON.stringify({ mode, instruction: text }),
            });
            await loadSession(created.session_id, { quiet: true });
            await openRun(created.run_id);
            const op = created.parent_context?.section_operation;
            setNotice(`已创建局部操作，尚未执行。待执行：${op?.work_ids?.join(", ") || section.section_id}；受影响：${op?.affected_ids?.join(", ") || "无"}。核对后点击启动；预算不清零。`, "success");
          } catch (error) { setNotice(error.message, "error"); }
          finally { button.disabled = false; }
        };
        panel.append(button);
      }
      const candidates = document.createElement("details");
      const heading = document.createElement("summary");
      heading.textContent = "候选证据（不等于已支持正文）";
      const excerpts = document.createElement("div");
      excerpts.style.whiteSpace = "pre-wrap";
      excerpts.textContent = (section.results || []).map((item, index) =>
        `${index + 1}. ${item.metadata?.source || item.metadata?.url || "来源未知"} · 检索 ${item.metadata?.retrieved_at || "时间未知"}\n${item.content || ""}`).join("\n\n");
      candidates.append(heading, excerpts);
      panel.append(candidates);
      details.append(panel);
    }
    return details;
  });
  $("sections-list").replaceChildren(...nodes);
}

function renderReportReview(review) {
  const node = $("report-review");
  node.classList.toggle("hidden", !review);
  node.style.whiteSpace = "pre-wrap";
  node.textContent = review ? `全篇一致性审校：${review.verdict === "pass" ? "未发现冲突（模型判断）" : "存在待解决问题"}\n${review.summary || ""}\n${(review.issues || []).map(issue => `${issue.section_ids.join(", ")} · ${issue.kind}: ${issue.detail}`).join("\n")}` : "";
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
    paused: "已暂停",
    budget_limited: "预算不足",
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
    badge.textContent = run.status === "completed" && run.parent_context?.section_operation &&
      (run.sections || []).some(s => !["complete", "limited"].includes(s.status)) ? "局部操作完成" : statusLabel(run.status);
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
    const selected = timeline.runs.find(run => run.run_id === state.currentRun?.run_id);
    if (selected && selected.execution_id !== state.currentRun.execution_id) {
      // A different tab may have resumed the Run; reconcile against authoritative state.
      await openRun(selected.run_id);
    }
    // A delayed Session request must not overwrite the current Run's newer SSE state.
    syncRunTree();
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
  renderBudget(run?.budget);
  const completed = run?.status === "completed" && (run.sections || []).every(s => ["complete", "limited"].includes(s.status));
  const resumable = run && RESUMABLE.has(run.status);
  const legacy = run && (!run.budget_id || run.budget?.version !== 2);
  $("start-btn").classList.toggle("hidden", Boolean(run));
  $("start-existing-btn").classList.toggle("hidden", run?.status !== "created");
  $("continue-btn").classList.toggle("hidden", !completed);
  $("use-as-parent-btn").classList.toggle("hidden", !completed);
  $("resume-btn").classList.toggle("hidden", !resumable);
  const policy = run?.budget?.policy;
  const exhausted = policy && (run.budget.charged_tokens >= policy.tokens ||
    run.budget.model_calls >= policy.model_calls);
  $("resume-btn").disabled = Boolean(legacy || exhausted || run?.status === "budget_limited");
  $("start-existing-btn").disabled = Boolean(legacy || exhausted);
  $("pause-btn").classList.toggle("hidden", run?.status !== "running");
  $("pause-btn").disabled = Boolean(run?.pause_requested);
  $("pause-btn").textContent = run?.pause_requested ? "正在暂停…" : "暂停研究";
  $("migrate-budget-btn").classList.toggle("hidden", !legacy || run?.status === "running");
  $("increase-budget-btn").classList.toggle("hidden", !run || legacy || run.status === "running");
  if (run?.budget_id) $("run-budget").textContent += ` 预算账户：${run.budget_id}（同研究子 Run 共用）。`;
  if (run?.status === "budget_limited" || exhausted) $("run-budget").textContent += " 预算不足，普通继续不会增加额度。";
  if (policy && run.budget.retrieval_calls >= policy.retrieval_calls) $("run-budget").textContent += " 检索额度已满，只能复用已保存结果，不能发起新检索。";
  $("copy-btn").classList.toggle("hidden", !state.reportText);
  setRunStatus(run?.status || "draft");
}

function renderBudget(budget) {
  const node = $("run-budget");
  if (!budget?.policy) { node.textContent = ""; return; }
  const policy = budget.policy;
  const unknown = budget.unknown_model_calls ?? Object.values(budget.reservations || {})
    .filter(entry => entry.kind === "model" && entry.status === "unknown").length;
  node.textContent = `已保存预算：模型 ${budget.model_calls}/${policy.model_calls} 次 · 检索 ${budget.retrieval_calls}/${policy.retrieval_calls} 次 · Token 占用 ${budget.charged_tokens}/${policy.tokens}（已知 ${budget.known_tokens}，${unknown} 次用量待确认）`;
  node.textContent += budget.version === 2
    ? ` · 每次执行最多 ${Math.round(policy.wall_seconds / 60)} 分钟；跨天继续重新计时，累计费用不清零。`
    : " · 旧版绝对截止策略：需明确确认迁移后才能继续。";
  if (budget.legacy_history_incomplete) node.textContent += " 历史调用统计不完整。";
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
  if (metrics.model_usage?.attempts) {
    $("stat-tokens").textContent = metrics.model_usage.tokens.toLocaleString();
    $("stat-tokens").title = `${metrics.model_usage.attempts} 次调用尝试（含格式纠正/失败）；${metrics.model_usage.unknown || 0} 次缺少用量信息`;
  }
}

function syncRunTree() {
  if (state.currentRun) {
    state.runs = state.runs.map(run => run.run_id === state.currentRun.run_id ? { ...state.currentRun } : run);
    const run = state.currentRun;
    $("report-subtitle").textContent = statusLabel(run.status)
      + (run.created_at ? ` · 创建于 ${new Date(run.created_at).toLocaleString("zh-CN")}` : "");
  }
  renderRunTree();
}

async function openRun(runId) {
  const request = ++state.viewRequest;
  closeStream();
  setNotice();
  resetPipeline();
  try {
    const snapshot = await api(`/api/runs/${encodeURIComponent(runId)}/snapshot`);
    if (request !== state.viewRequest) return;
    const run = snapshot.run;
    state.currentRun = run;
    state.cursor = snapshot.cursor;
    $("run-id").value = run.run_id;
    $("session-id").value = run.session_id || "";
    $("parent-run-id").value = run.parent_run_id || "";
    $("question").value = run.question;
    state.reportText = "";
    if (run.final_report) renderReport(run.final_report, { model_usage: run.model_usage });
    else clearReport();
    renderSections(run.sections?.length ? run.sections : (run.parent_context?.revision_sections || []));
    renderReportReview(run.report_review);
    const partial = run.parent_context?.section_operation && (run.sections || []).some(s => !["complete", "limited"].includes(s.status));
    $("report-title").textContent = `${partial ? "章节阶段产物" : "研究报告"} · ${shortId(run.run_id)}`;
    updateRunActions(run);
    syncRunTree();
    $("event-cursor").textContent = `snapshot #${snapshot.cursor}`;
    if (run.error_message) setNotice(run.error_message, "error");
    if (run.status === "running") subscribeRun(run, snapshot.cursor);
  } catch (error) {
    if (request === state.viewRequest) setNotice(error.message, "error");
  }
}

function eventMessage(type, data) {
  switch (type) {
    case "section_plan": return `章节计划已生成，共 ${data.sections?.length || 0} 章`;
    case "section_snapshot": return "已恢复章节草稿与研究进度";
    case "report_review": return "全篇一致性审校完成";
    case "section_progress": return `${data.stage} · 已完成 ${(data.sections || []).filter(s => ["complete", "limited"].includes(s.status)).length}/${data.sections?.length || 0} 章`;
    case "run_created": return `任务已创建${data.parent_run_id ? `，父 Run ${shortId(data.parent_run_id)}` : ""}`;
    case "run_started": return "后台执行已启动";
    case "run_resumed": return "已从原 Checkpoint 恢复";
    case "start": return data.reinitialized ? "启动前中断且无执行产物，重新初始化研究"
      : data.resumed ? "LangGraph 恢复执行" : "LangGraph 开始执行";
    case "supervisor_decision": return `第 ${data.iteration} 轮 → ${data.next || "结束"}`;
    case "search_complete": return `新增 ${data.new_count} 条，累计 ${data.total_count} 条`;
    case "analyst_verdict": return `${data.verdict} · 置信度 ${Math.round((data.confidence || 0) * 100)}%`;
    case "report_ready": return `报告生成完成，共 ${data.char_count || 0} 字`;
    case "done": return "研究任务已完成";
    case "run_interrupted": return `任务中断：${data.reason || "未知原因"}`;
    case "retrieval_progress": return `${data.provider} · ${data.reused ? "复用已保存结果" : data.status} · 累计尝试 ${data.attempts}/${data.max_attempts} · ${data.result_count} 条`;
    case "run_paused": return data.message || "已暂停，章节和累计预算保留";
    case "budget_limited": return data.message || "累计预算不足，继续不会重置";
    case "pause_requested": return "正在暂停，等待当前操作保存";
    case "error": return data.message || "任务失败";
    default: return "收到状态更新";
  }
}

function handleRunEvent(type, event) {
  let data = {};
  try { data = JSON.parse(event.data); } catch { data = { message: event.data }; }
  if (data.run_id && data.run_id !== state.currentRun?.run_id) return;
  const sequence = Number(event.lastEventId || 0);
  if (sequence && sequence <= state.cursor) return;
  if (sequence) state.cursor = sequence;
  const startsExecution = type === "run_started" || type === "run_resumed";
  if (startsExecution && state.currentRun) {
    if (state.currentRun.execution_id && data.execution_id !== state.currentRun.execution_id) {
      // UUIDs are not ordered; only the server snapshot decides which execution is current.
      openRun(state.currentRun.run_id);
      return;
    }
    state.currentRun.execution_id = data.execution_id;
    state.currentRun.status = "running";
    state.currentRun.error_message = null;
  } else if (state.currentRun?.execution_id && data.execution_id !== state.currentRun.execution_id) {
    appendLog("history", "忽略旧执行批次的事件", event.lastEventId);
    return;
  }
  appendLog(type, eventMessage(type, data), event.lastEventId);

  if (data.budget && state.currentRun) {
    state.currentRun.budget = data.budget;
    renderBudget(data.budget);
  }

  if (data.sections) {
    renderSections(data.sections);
    if (state.currentRun) state.currentRun.sections = data.sections;
  }
  if ("report_review" in data) {
    renderReportReview(data.report_review);
    if (state.currentRun) state.currentRun.report_review = data.report_review;
  }
  if (type.startsWith("section_")) {
    const target = { plan_sections: "supervisor", section_search: "search", section_analyze: "analyst", section_write: "writer", section_review: "analyst", section_claims: "analyst" }[data.stage];
    if (target && target !== "supervisor") setAgent("supervisor", "done", "章节协调中");
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
    if (state.currentRun) {
      state.currentRun.status = "completed";
      state.currentRun.final_report = data.report;
      state.currentRun.error_message = null;
      state.currentRun.model_usage = data.model_usage;
    }
    renderSections(data.sections || state.currentRun?.sections || []);
    if (data.report_quality === "partial") $("report-title").textContent = `章节阶段产物 · ${shortId(state.currentRun?.run_id)}`;
    updateRunActions(state.currentRun);
    closeStream();
    setNotice(data.report_quality === "partial" ? "局部操作已结束，仍有待完成/待补证据章节；这不是完整报告。" : data.report_quality === "limited"
      ? "报告已装配，部分章节存在未解决的证据或审校缺口，请阅读局限说明。"
      : "任务已完成；断开页面不会影响后台 Run。", data.report_quality === "limited" ? "" : "success");
    window.setTimeout(() => loadSession($("session-id").value.trim(), { quiet: true }).catch(() => {}), 350);
  } else if (type === "pause_requested") {
    if (state.currentRun) state.currentRun.pause_requested = true;
    updateRunActions(state.currentRun);
    setNotice(eventMessage(type, data));
  } else if (["error", "run_interrupted", "run_paused", "budget_limited"].includes(type)) {
    state.receivedTerminal = true;
    const status = ({error: "failed", run_interrupted: "interrupted", run_paused: "paused", budget_limited: "budget_limited"})[type];
    setRunStatus(status);
    const failedAgent = ({section_claims:"analyst", section_review:"analyst", section_analyze:"analyst", section_search:"search", section_write:"writer"})[data.stage];
    if (failedAgent) {
      setAgent("supervisor", "done", "已保存进度");
      setAgent(failedAgent, "error", data.reason === "claims_pending" ? `${data.section_id}：结论关联待修复，展开章节查看详情` : `${data.section_id || ""}：步骤停止，见错误详情`);
    } else {
      for (const name of AGENTS) {
        if ($("agent-" + name).classList.contains("active")) setAgent(name, "error", "执行停止，详见下方错误信息");
      }
    }
    if (state.currentRun) {
      state.currentRun.status = status;
      state.currentRun.error_message = eventMessage(type, data);
      state.currentRun.model_usage = data.model_usage;
    }
    updateRunActions(state.currentRun);
    closeStream();
    renderSections(state.currentRun?.sections || []);
    setNotice(eventMessage(type, data), "error");
    loadSession($("session-id").value.trim(), { quiet: true }).catch(() => {});
  }
  syncRunTree();
}

function subscribeRun(run, cursor) {
  closeStream();
  state.receivedTerminal = false;
  const source = new EventSource(`/api/runs/${encodeURIComponent(run.run_id)}/stream?after=${cursor}`);
  state.source = source;
  const eventTypes = [
    "run_created", "run_started", "run_resumed", "start", "supervisor_decision",
    "search_complete", "analyst_verdict", "report_ready", "done", "run_interrupted", "error",
    "section_plan", "section_progress", "section_snapshot", "report_review",
    "pause_requested", "run_paused", "budget_limited", "budget_migrated", "budget_increased", "retrieval_progress",
  ];
  for (const type of eventTypes) source.addEventListener(type, (event) => {
    if (state.source === source) handleRunEvent(type, event);
  });
  source.onerror = () => {
    if (state.source !== source) return;
    if (state.receivedTerminal || TERMINAL.has(state.currentRun?.status)) {
      closeStream();
      return;
    }
    appendLog("connection", "事件流暂时断开，浏览器将自动重连", "");
    // The server may have reached a terminal state without an event (old deployments,
    // abrupt restarts). Reconcile from a fresh snapshot, never restart replay at zero.
    window.setTimeout(() => {
      if (state.source === source && state.currentRun?.run_id === run.run_id) openRun(run.run_id);
    }, 1500);
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
    await openRun(running.run_id);
    if (state.currentRun?.status === "running") setNotice("Run 已在后台执行。关闭或刷新页面不会取消任务。", "success");
  } catch (error) {
    setNotice(error.message, "error");
    throw error;
  } finally {
    $("start-existing-btn").disabled = false;
  }
}

function prepareNewRun() {
  state.viewRequest += 1;
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
  setNotice("已选定父 Run。子 Run 不覆写父报告，但共用该研究的累计预算；独立研究请使用空白 Run。", "success");
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
    await openRun(running.run_id);
    if (state.currentRun?.status === "running") setNotice("正在恢复同一 Run：run_id 不变，并从已有 Checkpoint 继续。", "success");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    updateRunActions(state.currentRun);
  }
}

async function pauseCurrentRun() {
  const run = state.currentRun;
  if (!run) return;
  try {
    state.currentRun = await api(`/api/runs/${encodeURIComponent(run.run_id)}/pause`, {method: "POST"});
    updateRunActions(state.currentRun);
    setNotice("正在暂停：不再发起新请求，当前调用可能继续完成并计入预算。");
  } catch (error) { setNotice(error.message, "error"); }
}

async function migrateCurrentBudget() {
  const run = state.currentRun;
  if (!run || !window.confirm("确认将此历史 Run 迁移为每次执行独立计时？历史消耗、未知预留和额度均保留，不自动执行；历史父子 Run 不自动合账。")) return;
  try {
    await api(`/api/runs/${encodeURIComponent(run.run_id)}/budget/migrate`, {
      method: "POST", body: JSON.stringify({confirm: true, reason: "用户在页面明确确认迁移旧版绝对时间策略"}),
    });
    await openRun(run.run_id);
    setNotice("预算策略已迁移；历史费用保留。可以单独选择继续。", "success");
  } catch (error) { setNotice(error.message, "error"); }
}

async function increaseCurrentBudget() {
  const run = state.currentRun;
  if (!run?.budget_id || run.status === "running") return;
  const expected = run.budget.policy.tokens;
  const value = window.prompt(`当前共享上限 ${expected}，已占用 ${run.budget.charged_tokens}。请输入新的总上限（不是追加量）：`);
  if (value === null) return;
  const total = Number(value);
  if (!Number.isSafeInteger(total) || total <= expected) {
    setNotice("新总上限必须是大于当前上限的整数。", "error"); return;
  }
  const reason = window.prompt("请填写追加原因（至少 5 字，将保存在审计记录中）：");
  if (reason === null) return;
  if (reason.trim().length < 5 || reason.length > 500) { setNotice("原因需 5–500 字。", "error"); return; }
  if (!window.confirm(`确认同研究共享额度 ${expected} → ${total}（增加 ${total - expected}）？消耗与未知预留不变，不自动恢复。`)) return;
  const key = JSON.stringify([run.run_id, expected, total, reason.trim()]);
  // Retain the same key after an uncertain network response; CAS also blocks stale retries.
  if (state.budgetIncrease?.key !== key) state.budgetIncrease = {key, id: crypto.randomUUID()};
  $("increase-budget-btn").disabled = true;
  try {
    await api(`/api/runs/${encodeURIComponent(run.run_id)}/budget/increase`, {
      method:"POST", body:JSON.stringify({confirm:true, request_id:state.budgetIncrease.id,
        expected_tokens:expected, new_tokens:total, reason:reason.trim()}),
    });
    if (state.currentRun?.run_id === run.run_id) await openRun(run.run_id);
    setNotice("共享 Token 上限已追加，历史消耗不变。尚未启动，请单独选择恢复。", "success");
  } catch (error) { setNotice(error.message, "error"); }
  finally { $("increase-budget-btn").disabled = false; }
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
    const ok = health.status === "ok" && health.run_store === "ok"
      && health.checkpointer?.status === "ok" && health.checkpointer?.persistent === true
      && health.runtime?.ownership === "ok" && health.runtime?.monitor === "ok";
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
$("pause-btn").addEventListener("click", pauseCurrentRun);
$("migrate-budget-btn").addEventListener("click", migrateCurrentBudget);
$("increase-budget-btn").addEventListener("click", increaseCurrentBudget);
$("copy-btn").addEventListener("click", copyReport);
$("clear-parent-btn").addEventListener("click", () => { $("parent-run-id").value = ""; });
window.addEventListener("beforeunload", closeStream);

async function pollHealth() {
  await checkHealth();
  window.setTimeout(pollHealth, 15000);
}
pollHealth();
const initialSession = new URLSearchParams(location.search).get("session_id")
  || localStorage.getItem("multi-agent:last-session");
if (initialSession) {
  $("session-id").value = initialSession;
  loadSession(initialSession, { quiet: true }).catch(() => {
    setNotice("上次使用的 Session 已不存在或暂时无法加载。", "error");
  });
}
