// Offline DOM-contract tests; does not replace a real browser visual check.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

test('pipeline distinguishes five role agents from Claim processor and retrieval service', () => {
  const html = readFileSync(join(__dirname, '../multi_agent_research/api/static/index.html'), 'utf8');
  for (const id of ['planner', 'evidence-research', 'section-writer', 'section-reviewer', 'report-reviewer']) {
    assert.match(html, new RegExp(`id="agent-${id}"`));
  }
  assert.match(html, /id="processor-claim-binding"/);
  assert.doesNotMatch(html, /id="agent-claim-extractor"/);
  assert.doesNotMatch(html, /id="agent-(?:supervisor|search|analyst|writer)"/);
  assert.match(html, /非 Agent：Claim Binding Processor/);
  assert.match(html, /非 Agent：Retrieval Service/);
  assert.match(html, /id="internal-audit"/);
  assert.match(html, /不属于正式报告正文/);
});

function setup() {
  class Element {
    constructor(tag = 'div') {
      this.tag = tag;
      this.children = [];
      this.style = { setProperty() {} };
      this.dataset = {};
      this.value = '';
      this.textContent = '';
      this.classList = { toggle() {}, add() {}, remove() {}, contains() { return false; } };
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener() {}
    setAttribute() {}
  }
  const nodes = new Map();
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, new Element());
      return nodes.get(id);
    },
    createElement(tag) { return new Element(tag); },
  };
  const context = vm.createContext({
    document, URLSearchParams, location: { search: '' },
    localStorage: { getItem() { return null; } },
    window: { addEventListener() {}, setTimeout() {}, clearTimeout() {} },
    fetch: async () => ({ ok: true, text: async () => '{"status":"ok"}' }),
    EventSource: class {
      constructor(url) { this.url = url; this.handlers = {}; this.closed = false; }
      addEventListener(type, fn) { this.handlers[type] = fn; }
      close() { this.closed = true; }
    },
  });
  vm.runInContext(readFileSync(join(__dirname, '../multi_agent_research/api/static/app.js'), 'utf8'), context);
  return { context, nodes };
}

const section = {
  section_id: 'section_1', title: '成本', question: '成本优势如何', status: 'complete', revision: 2,
  draft: '<script>not executable</script>[来源1]',
  depends_on: [], sources: [{ content: '成本下降', metadata: { source: '报告.pdf', page: 3, chunk_id: 'c1' } }],
  claims: [{ claim_id: 'section_1:v2:c1', statement: '成本下降', assessment: 'supported', evidence: [
    { source_number: 1, quote: '成本下降', evidence_id: 'sha256-id', relation: 'supports' },
  ] }],
};

test('budget increase confirms total, retries same ID and never resumes automatically', async () => {
  const { context } = setup();
  context.crypto = {randomUUID: () => 'test-budget-increase-id'};
  context.answers = ['600000', '用户同意追加预算'];
  context.calls = [];
  vm.runInContext(`window.prompt=()=>answers.shift(); window.confirm=()=>true;
    state.currentRun={run_id:'r',budget_id:'b',status:'budget_limited',budget:{policy:{tokens:500000},charged_tokens:435910}};
    api=async (url,options)=>{ calls.push({url,body:JSON.parse(options.body)}); throw new Error('network'); };`, context);
  await vm.runInContext('increaseCurrentBudget()', context);
  context.answers = ['600000', '用户同意追加预算'];
  await vm.runInContext('increaseCurrentBudget()', context);
  assert.equal(context.calls.length, 2);
  assert.equal(context.calls[0].url, '/api/runs/r/budget/increase');
  assert.equal(context.calls[0].body.new_tokens, 600000);
  assert.equal(context.calls[0].body.expected_tokens, 500000);
  assert.equal(context.calls[0].body.request_id, context.calls[1].body.request_id);
  context.answers = ['700000', '再次追加研究预算'];
  vm.runInContext('window.confirm=()=>false', context);
  await vm.runInContext('increaseCurrentBudget()', context);
  assert.equal(context.calls.length, 2);
});

test('pending claims show saved and unresolved items without executing their text', () => {
  const { context, nodes } = setup();
  context.partialSection = {...section, status:'claims_pending', claim_work:{pending:[{
    slot:8, candidate:{statement:'<script>bad()</script>'}, errors:[{message:'draft quote mismatch'}],
  }], batch_errors:[]}};
  vm.runInContext('renderSections([partialSection])', context);
  const flatten = el => [el.textContent, ...el.children.flatMap(flatten)];
  const text = flatten(nodes.get('sections-list')).join('\n');
  assert.match(text, /未完成核验/);
  assert.match(text, /已保存 1 条通过校验/);
  assert.match(text, /1 条待修复/);
  assert.match(text, /<script>bad\(\)<\/script>/);
  assert.match(text, /draft quote mismatch/);
});

test('health accepts durable SQLite and rejects a degraded runtime', async () => {
  const { context, nodes } = setup();
  vm.runInContext(`api = async () => ({status:'ok', run_store:'ok',
    checkpointer:{status:'ok', backend:'sqlite', persistent:true},
    runtime:{ownership:'ok', monitor:'ok'}});`, context);
  await vm.runInContext('checkHealth()', context);
  assert.equal(nodes.get('health').className, 'health ok');
  vm.runInContext(`api = async () => { throw new Error('503'); };`, context);
  await vm.runInContext('checkHealth()', context);
  assert.equal(nodes.get('health').className, 'health error');
});

test('budget view distinguishes reservations from known usage and clears on a new draft', () => {
  const { context, nodes } = setup();
  vm.runInContext(`renderBudget({policy:{model_calls:10,retrieval_calls:8,tokens:10000},
    model_calls:3,retrieval_calls:2,charged_tokens:9000,known_tokens:1000,deadline:2000000000,
    unknown_model_calls:1,legacy_history_incomplete:true});`, context);
  assert.match(nodes.get('run-budget').textContent, /模型 3\/10/);
  assert.match(nodes.get('run-budget').textContent, /Token 9000\/10000（已知 1000，1 次/);
  assert.match(nodes.get('run-budget').textContent, /需明确确认迁移/);
  assert.match(nodes.get('run-budget').textContent, /历史调用统计不完整/);
  vm.runInContext('updateRunActions(null)', context);
  assert.equal(nodes.get('run-budget').textContent, '');
});

test('usage view separates confirmed usage from independent budget occupancy', () => {
  const { context, nodes } = setup();
  context.usage = {
    run_id: 'current', budget_scope: 'run', stage_attribution_complete: true,
    history_incomplete: false,
    current: {model_calls: 9, retrieval_calls: 8, known_tokens: 81450, charged_tokens: 81450},
    budget: {model_calls: 9, retrieval_calls: 8, known_tokens: 81450, charged_tokens: 81450},
    stages: [
      {kind:'model',stage:'section_write',scope:'section_1',calls:2,known_tokens:25208,unknown_calls:0},
      {kind:'retrieval',stage:'retrieval',scope:'knowledge',calls:4,known_tokens:0,unknown_calls:0},
    ],
  };
  vm.runInContext('renderUsage(usage)', context);
  assert.equal(nodes.get('usage-current-tokens').textContent, '81,450 Token');
  assert.match(nodes.get('usage-current-calls').textContent, /模型 9 次 · 检索 8 次/);
  assert.equal(nodes.get('usage-budget-tokens').textContent, '81,450 Token');
  assert.match(nodes.get('usage-budget-calls').textContent, /模型 9 次 · 检索 8 次/);
  assert.equal(nodes.get('usage-stage-list').children.length, 2);
  assert.match(nodes.get('usage-stage-list').children[0].children[0].textContent, /Section Writer.*section_1/);
  assert.match(nodes.get('usage-stage-list').children[0].children[1].textContent, /25,208 Token · 2 次/);
});

test('v2 pause/resume controls preserve cumulative quota and explain migration', () => {
  const { context, nodes } = setup();
  vm.runInContext(`state.currentRun = {run_id:'r',budget_id:'b',status:'paused',
    budget:{version:2,policy:{model_calls:10,retrieval_calls:10,tokens:1000,wall_seconds:3600},
    model_calls:2,retrieval_calls:1,known_tokens:50,charged_tokens:100}};
    updateRunActions(state.currentRun);`, context);
  assert.equal(nodes.get('resume-btn').disabled, false);
  assert.match(nodes.get('run-budget').textContent, /累计费用不清零/);
  assert.match(nodes.get('run-budget').textContent, /账户：b/);
  vm.runInContext(`state.currentRun.status='running'; state.currentRun.pause_requested=true;
    updateRunActions(state.currentRun);`, context);
  assert.equal(nodes.get('pause-btn').disabled, true);
  assert.match(nodes.get('pause-btn').textContent, /正在暂停/);
  vm.runInContext(`state.currentRun.status='budget_limited'; updateRunActions(state.currentRun);`, context);
  assert.equal(nodes.get('resume-btn').disabled, true);
  vm.runInContext(`state.currentRun.status='failed'; state.currentRun.budget_id=null;
    state.currentRun.budget.version=1; updateRunActions(state.currentRun);`, context);
  assert.equal(nodes.get('resume-btn').disabled, true);
  assert.match(nodes.get('run-budget').textContent, /需明确确认迁移/);
});

test('retrieval receipt events show replay and a full retrieval quota permits cached resume', () => {
  const { context, nodes } = setup();
  vm.runInContext(`state.currentRun = {run_id:'r',budget_id:'b',status:'paused',execution_id:'e',
    budget:{version:2,policy:{model_calls:10,retrieval_calls:2,tokens:1000,wall_seconds:3600},
    model_calls:1,retrieval_calls:2,known_tokens:50,charged_tokens:50}};
    updateRunActions(state.currentRun); subscribeRun(state.currentRun, 10);`, context);
  assert.equal(nodes.get('resume-btn').disabled, false);
  assert.match(nodes.get('run-budget').textContent, /只能复用已保存结果/);
  const source = vm.runInContext('state.source', context);
  assert.equal(typeof source.handlers.retrieval_progress, 'function');
  context.receipt = {provider:'knowledge',reused:true,status:'succeeded',attempts:1,max_attempts:4,result_count:3};
  assert.match(vm.runInContext('eventMessage("retrieval_progress", receipt)', context), /复用已保存结果.*1\/4.*3 条/);
});

test('chapter UI uses text and shows exact evidence locator and revision form', () => {
  const { context, nodes } = setup();
  context.fixture = section;
  vm.runInContext('state.currentRun = {run_id: "parent", status: "completed"}; renderSections([fixture]);', context);
  const details = nodes.get('sections-list').children[0];
  assert.match(details.children[1].textContent, /成本下降/);
  assert.match(details.children[1].textContent, /sha256-id/);
  assert.match(details.children[1].textContent, /chunk: c1/);
  assert.match(details.children[1].textContent, /<script>not executable<\/script>/);
  assert.equal(details.children[1].innerHTML, undefined);
  assert.equal(details.children[2].tag, 'textarea');
  assert.equal(details.children[3].textContent, '创建本章修订 Run');
});

test('paused chapter operations create scoped children without starting them', async () => {
  const { context, nodes } = setup();
  context.fixture = {...section, status:'evidence_ready', results:[{content:'<script>source</script>',metadata:{retrieved_at:'2026-10-08'}}]};
  context.calls = [];
  vm.runInContext(`state.currentRun={run_id:'paused-parent',status:'paused'};
    api=async (path,options)=>{calls.push({path,options});return {run_id:'child',session_id:'s',parent_context:{section_operation:{work_ids:['section_1'],affected_ids:['section_1','section_3']}}};};
    loadSession=async()=>{};openRun=async(id)=>{calls.push({opened:id});};renderSections([fixture]);`, context);
  const details = nodes.get('sections-list').children[0];
  assert.match(details.children[0].textContent,/正文待更新/);
  const panel = details.children[2];
  panel.children[1].value='仅补充最新的反面证据';
  assert.equal(panel.children[2].textContent,'选择本章继续');
  assert.equal(panel.children[3].textContent,'仅补证据，不改正文');
  await panel.children[3].onclick();
  assert.equal(context.calls[0].path,'/api/runs/paused-parent/sections/section_1/operations');
  assert.equal(JSON.parse(context.calls[0].options.body).mode,'supplement');
  assert.equal(context.calls.length,2);
  assert.match(panel.children[5].children[1].textContent,/<script>source<\/script>/);
  assert.equal(panel.children[5].children[1].innerHTML,undefined);
});

test('revision form creates a new Run without starting or overwriting parent', async () => {
  const { context, nodes } = setup();
  context.fixture = section;
  context.calls = [];
  vm.runInContext(`
    state.currentRun = {run_id: 'parent', status: 'completed'};
    api = async (path, options) => { calls.push({path, options}); return {run_id: 'child', session_id: 'session'}; };
    loadSession = async () => {};
    openRun = async (id) => { calls.push({opened: id}); };
    renderSections([fixture]);
  `, context);
  const details = nodes.get('sections-list').children[0];
  details.children[2].value = '补充最新年度成本证据';
  await details.children[3].onclick();
  assert.equal(context.calls[0].path, '/api/runs/parent/sections/section_1/revisions');
  assert.equal(JSON.parse(context.calls[0].options.body).instruction, '补充最新年度成本证据');
  assert.equal(context.calls[1].opened, 'child');
  assert.equal(context.calls.length, 2);
  assert.equal(vm.runInContext('state.currentRun.run_id', context), 'parent');
});

test('SSE done enables revision and renders final review; running snapshots do not', () => {
  const { context, nodes } = setup();
  context.fixture = section;
  vm.runInContext('state.currentRun = {run_id: "parent", status: "running"}; renderSections([fixture]);', context);
  assert.equal(nodes.get('sections-list').children[0].children.length, 2);
  context.event = { data: JSON.stringify({
    sections: [section], report: '测试报告', report_quality: 'limited',
    report_review: { verdict: 'revise', summary: '请核对日期', issues: [
      { section_ids: ['section_1'], kind: 'scope', detail: '年份口径冲突' },
    ] },
  }), lastEventId: '123' };
  vm.runInContext('handleRunEvent("done", event)', context);
  assert.equal(nodes.get('sections-list').children[0].children.length, 5);
  assert.match(nodes.get('report-review').textContent, /年份口径冲突/);
  assert.match(nodes.get('report-review').textContent, /内部全篇一致性审校/);
});

test('completed report can be deterministically reassembled without starting research', async () => {
  const { context, nodes } = setup();
  context.fixture = section;
  context.calls = [];
  vm.runInContext(`state.currentRun={run_id:'done',status:'completed',sections:[fixture],report_review:{verdict:'pass'}};
    api=async (path,options)=>{calls.push({path,options});return {...state.currentRun,final_report:'# 已整理'};};
    syncRunTree=()=>{};`, context);
  await vm.runInContext('reassembleCurrentReport()', context);
  assert.equal(context.calls[0].path, '/api/runs/done/report/reassemble');
  assert.equal(context.calls[0].options.method, 'POST');
  assert.match(nodes.get('notice').textContent, /未调用模型或检索/);
});

test('resume starts after snapshot cursor and ignores historical terminal events', () => {
  const { context } = setup();
  vm.runInContext(`
    state.currentRun = {run_id: 'run1', status: 'running', execution_id: 'new'};
    state.runs = [state.currentRun]; state.cursor = 118;
    subscribeRun(state.currentRun, 118);
  `, context);
  const source = vm.runInContext('state.source', context);
  assert.equal(source.url, '/api/runs/run1/stream?after=118');
  const event = (id, execution, report) => ({lastEventId: String(id), data: JSON.stringify({
    run_id: 'run1', execution_id: execution, message: 'previous failure', report,
  })});
  source.handlers.error(event(117, 'old'));
  assert.equal(source.closed, false);
  source.handlers.error(event(119, 'old'));
  assert.equal(source.closed, false);
  assert.equal(vm.runInContext('state.currentRun.status', context), 'running');
  source.handlers.done(event(120, 'new', 'recovered report'));
  assert.equal(source.closed, true);
  assert.equal(vm.runInContext('state.currentRun.status', context), 'completed');
  assert.equal(vm.runInContext('state.runs[0].status', context), 'completed');
});

test('a second failure updates the tree and duplicate/closed-source events are harmless', () => {
  const { context, nodes } = setup();
  vm.runInContext(`
    state.currentRun = {run_id: 'run1', status: 'running', execution_id: 'new'};
    state.runs = [{...state.currentRun}]; state.cursor = 118;
    subscribeRun(state.currentRun, 118);
  `, context);
  const oldSource = vm.runInContext('state.source', context);
  vm.runInContext('subscribeRun(state.currentRun, 118)', context);
  const source = vm.runInContext('state.source', context);
  const event = {lastEventId: '121', data: JSON.stringify({run_id:'run1', execution_id:'new', message:'new error'})};
  oldSource.handlers.error(event);
  assert.equal(source.closed, false);
  source.handlers.error(event);
  assert.equal(vm.runInContext('state.runs[0].status', context), 'failed');
  assert.equal(vm.runInContext('state.currentRun.error_message', context), 'new error');
  assert.match(nodes.get('report-subtitle').textContent, /失败/);
});

test('opening a running Run uses its atomic snapshot cursor rather than replaying zero', async () => {
  const { context } = setup();
  context.calls = [];
  vm.runInContext(`
    api = async path => {
      calls.push(path);
      return {cursor: 120, run: {run_id: 'r', session_id: 's', question: 'question',
        status: 'running', execution_id: 'execution2', created_at: '2026-10-02', sections: []}};
    };
  `, context);
  await vm.runInContext('openRun("r")', context);
  assert.equal(context.calls[0], '/api/runs/r/snapshot');
  assert.equal(vm.runInContext('state.source.url', context), '/api/runs/r/stream?after=120');
});

test('terminal snapshot shows saved error and never opens a historical event stream', async () => {
  const { context, nodes } = setup();
  vm.runInContext(`api = async () => ({cursor: 121, run: {run_id:'r', status:'failed',
    execution_id:'e2', error_message:'search_queries: at most 2', sections:[], created_at:'2026-10-02'}});`, context);
  await vm.runInContext('openRun("r")', context);
  assert.equal(vm.runInContext('state.source', context), null);
  assert.match(nodes.get('notice').textContent, /search_queries/);
});

test('a delayed snapshot from a previous selection cannot switch the current Run back', async () => {
  const { context } = setup();
  vm.runInContext(`
    var resolveOld;
    api = path => path.includes('/old/') ? new Promise(resolve => { resolveOld = resolve; })
      : Promise.resolve({cursor:200, run:{run_id:'new', status:'created', sections:[], created_at:'2026-10-02'}});
  `, context);
  const first = vm.runInContext('openRun("old")', context);
  await vm.runInContext('openRun("new")', context);
  vm.runInContext(`resolveOld({cursor:100, run:{run_id:'old', status:'running'}})`, context);
  await first;
  assert.equal(vm.runInContext('state.currentRun.run_id', context), 'new');
  assert.equal(vm.runInContext('state.cursor', context), 200);
});
