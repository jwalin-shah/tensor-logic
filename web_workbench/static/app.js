const sourceEl = document.getElementById("source");
const resultEl = document.getElementById("result");
const jsonEl = document.getElementById("json-view");
const proofEl = document.getElementById("proof-view");
const relationEl = document.getElementById("relation");
const arg1El = document.getElementById("arg1");
const arg2El = document.getElementById("arg2");
const recursiveEl = document.getElementById("recursive");
const presetEl = document.getElementById("preset");
const relationOptionsEl = document.getElementById("relation-options");
const argumentOptionsEl = document.getElementById("argument-options");
const programIndexEl = document.getElementById("program-index");
const programSummaryEl = document.getElementById("program-summary");
const lineCountEl = document.getElementById("line-count");
const historyEl = document.getElementById("history");
const answerStripEl = document.getElementById("answer-strip");
const answerTitleEl = document.getElementById("answer-title");
const durationEl = document.getElementById("duration");
const exitCodeEl = document.getElementById("exit-code");
const activePresetEl = document.getElementById("active-preset");
const resetSourceEl = document.getElementById("reset-source");
const clearHistoryEl = document.getElementById("clear-history");
const ingestPathEl = document.getElementById("ingest-path");
const loadRepoEl = document.getElementById("load-repo");
const ingestSummaryEl = document.getElementById("ingest-summary");
const impactModuleEl = document.getElementById("impact-module");
const analyzeImpactEl = document.getElementById("analyze-impact");
const impactSummaryEl = document.getElementById("impact-summary");
const changeBriefEl = document.getElementById("change-brief");
const briefSummaryEl = document.getElementById("brief-summary");
const analyzeOverviewEl = document.getElementById("analyze-overview");
const overviewSummaryEl = document.getElementById("overview-summary");
const setBaselineEl = document.getElementById("set-baseline");
const compareGraphEl = document.getElementById("compare-graph");
const compareSummaryEl = document.getElementById("compare-summary");

const STORAGE_KEY = "tensor_logic_workbench_state";
const HISTORY_LIMIT = 12;

const EXAMPLES = [
  {
    id: "kinship",
    name: "Kinship Closure",
    relation: "ancestor",
    arg1: "alice",
    arg2: "cara",
    recursive: true,
    source: `domain Person { alice bob cara dave }

relation parent(Person, Person)
relation ancestor(Person, Person)

fact parent(alice, bob)
fact parent(bob, cara)
fact parent(cara, dave)

rule ancestor(x,z) := (parent(x,z) + ancestor(x,y) * parent(y,z)).step()

query ancestor(alice, dave) recursive
prove ancestor(alice, dave) recursive`,
  },
  {
    id: "permissions",
    name: "Access Control",
    relation: "can_access",
    arg1: "alice",
    arg2: "customer_record",
    recursive: false,
    source: `domain User { alice bob }
domain Role { admin viewer }
domain Resource { invoice dashboard customer_record }

relation has_role(User, Role)
relation role_can_access(Role, Resource)
relation can_access(User, Resource)

fact has_role(alice, admin)
fact has_role(bob, viewer)

fact role_can_access(admin, invoice)
fact role_can_access(admin, customer_record)
fact role_can_access(viewer, dashboard)

rule can_access(u,r) := (has_role(u,role) * role_can_access(role,r)).step()

query can_access(alice, customer_record)
prove can_access(alice, customer_record)`,
  },
  {
    id: "planning",
    name: "Launch Plan",
    relation: "must_finish_before",
    arg1: "design",
    arg2: "launch",
    recursive: true,
    source: `domain Task {
  design
  backend
  frontend
  tests
  launch
}

relation blocks(Task, Task)
relation must_finish_before(Task, Task)

fact blocks(design, backend)
fact blocks(backend, frontend)
fact blocks(frontend, tests)
fact blocks(tests, launch)

rule must_finish_before(x,z) := (blocks(x,z) + must_finish_before(x,y) * blocks(y,z)).step()

query must_finish_before(design, launch) recursive
prove must_finish_before(design, launch) recursive`,
  },
  {
    id: "repo",
    name: "Repo Dependencies",
    relation: "depends_on",
    arg1: "worker",
    arg2: "models",
    recursive: true,
    source: `domain Module {
  api
  db
  models
  worker
  auth
}

relation imports(Module, Module)
relation depends_on(Module, Module)

fact imports(worker, api)
fact imports(api, db)
fact imports(db, models)
fact imports(api, auth)

rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()

query depends_on(worker, models) recursive
prove depends_on(worker, models) recursive`,
  },
  {
    id: "memory",
    name: "Follow Ups",
    relation: "should_follow_up",
    arg1: "ryan",
    arg2: "tensor_demo",
    recursive: false,
    source: `domain Person { jwalin ryan steven }
domain Topic { tensor_logic robotics local_ai openhuman }
domain Project { openhuman tensor_demo jarvis }

relation mentioned(Person, Topic)
relation project_about(Project, Topic)
relation should_follow_up(Person, Project)

fact mentioned(ryan, tensor_logic)
fact mentioned(ryan, robotics)
fact mentioned(steven, local_ai)

fact project_about(tensor_demo, tensor_logic)
fact project_about(openhuman, local_ai)
fact project_about(jarvis, robotics)

rule should_follow_up(p,proj) := (mentioned(p,t) * project_about(proj,t)).step()

query should_follow_up(ryan, tensor_demo)
prove should_follow_up(ryan, tensor_demo)`,
  },
];

let history = [];
let saveTimer = 0;
let repoMetadata = null;
let repoBaseline = null;

init();

function init() {
  renderPresets();
  const saved = loadState();
  if (saved?.source) {
    presetEl.value = saved.presetId || EXAMPLES[0].id;
    sourceEl.value = saved.source;
    relationEl.value = saved.relation || EXAMPLES[0].relation;
    arg1El.value = saved.arg1 || EXAMPLES[0].arg1;
    arg2El.value = saved.arg2 || EXAMPLES[0].arg2;
    recursiveEl.checked = saved.recursive ?? EXAMPLES[0].recursive;
    repoMetadata = saved.repoMetadata || null;
    repoBaseline = saved.repoBaseline || null;
  } else {
    applyPreset(EXAMPLES[0].id, false);
  }

  updateActivePreset();
  analyzeSource();
  renderHistory();
  renderIdle();
  updateBaselineSummary();
  bindEvents();
}

function bindEvents() {
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => invoke(button.dataset.action));
  });

  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });

  presetEl.addEventListener("change", () => applyPreset(presetEl.value));
  resetSourceEl.addEventListener("click", () => applyPreset(presetEl.value));
  clearHistoryEl.addEventListener("click", () => {
    history = [];
    renderHistory();
  });
  loadRepoEl.addEventListener("click", loadRepoGraph);
  analyzeImpactEl.addEventListener("click", analyzeImpact);
  changeBriefEl.addEventListener("click", buildChangeBrief);
  analyzeOverviewEl.addEventListener("click", analyzeOverview);
  setBaselineEl.addEventListener("click", setRepoBaseline);
  compareGraphEl.addEventListener("click", compareRepoGraph);

  sourceEl.addEventListener("input", () => {
    repoMetadata = null;
    analyzeSource();
    queueSave();
  });

  [relationEl, arg1El, arg2El, recursiveEl].forEach((el) => {
    el.addEventListener("input", queueSave);
    el.addEventListener("change", queueSave);
  });

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      invoke(event.shiftKey ? "prove" : "run");
    }
  });
}

function renderPresets() {
  presetEl.replaceChildren();
  EXAMPLES.forEach((example) => {
    const option = document.createElement("option");
    option.value = example.id;
    option.textContent = example.name;
    presetEl.append(option);
  });
}

function applyPreset(id, shouldSave = true) {
  const example = EXAMPLES.find((item) => item.id === id) || EXAMPLES[0];
  repoMetadata = null;
  presetEl.value = example.id;
  sourceEl.value = example.source;
  relationEl.value = example.relation;
  arg1El.value = example.arg1;
  arg2El.value = example.arg2;
  recursiveEl.checked = example.recursive;
  updateActivePreset();
  analyzeSource();
  if (shouldSave) {
    saveState();
  }
}

async function loadRepoGraph() {
  loadRepoEl.disabled = true;
  loadRepoEl.textContent = "Loading...";
  ingestSummaryEl.textContent = "Parsing Python imports...";
  const started = performance.now();

  try {
    const res = await fetch("/api/ingest-python", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: ingestPathEl.value }),
    });
    const data = await res.json();
    data.client_duration_ms = Math.round(performance.now() - started);

    if (res.ok && data.payload?.source) {
      const query = data.payload.suggested_query;
      sourceEl.value = data.payload.source;
      relationEl.value = query?.relation || "depends_on";
      arg1El.value = query?.arg1 || data.payload.imports?.[0]?.[0] || "";
      arg2El.value = query?.arg2 || data.payload.imports?.[0]?.[1] || "";
      impactModuleEl.value = query?.arg2 || arg2El.value;
      repoMetadata = repoMetadataFromPayload(data.payload);
      recursiveEl.checked = query?.recursive ?? true;
      activePresetEl.textContent = "Repo Import Graph";
      ingestSummaryEl.textContent = `${data.payload.modules.length} modules / ${data.payload.imports.length} imports`;
      analyzeSource();
      saveState();
    } else {
      ingestSummaryEl.textContent = data.error || "Import graph failed";
    }

    renderResult(data, res.ok);
    pushHistory(data, res.ok);
    if (res.ok && data.payload?.suggested_query) {
      await invoke("prove");
      await analyzeImpact();
      await analyzeOverview();
    }
  } catch (error) {
    const data = {
      action: "ingest-python",
      error: String(error),
      exit_code: 1,
      stderr: String(error),
      client_duration_ms: Math.round(performance.now() - started),
    };
    ingestSummaryEl.textContent = "Import graph failed";
    renderResult(data, false);
    pushHistory(data, false);
  } finally {
    loadRepoEl.disabled = false;
    loadRepoEl.textContent = "Load Imports";
  }
}

async function analyzeOverview() {
  analyzeOverviewEl.disabled = true;
  analyzeOverviewEl.textContent = "Ranking...";
  overviewSummaryEl.textContent = "Computing repo overview...";
  const started = performance.now();

  try {
    const res = await fetch("/api/repo-overview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: sourceEl.value, metadata: repoMetadata }),
    });
    const data = await res.json();
    data.client_duration_ms = Math.round(performance.now() - started);
    if (res.ok && data.payload) {
      overviewSummaryEl.textContent = `${data.payload.modules} modules / ${data.payload.cycles.length} cycles`;
    } else {
      overviewSummaryEl.textContent = data.error || "Overview failed";
    }
    renderResult(data, res.ok);
    pushHistory(data, res.ok);
  } catch (error) {
    const data = {
      action: "repo-overview",
      error: String(error),
      exit_code: 1,
      stderr: String(error),
      client_duration_ms: Math.round(performance.now() - started),
    };
    overviewSummaryEl.textContent = "Overview failed";
    renderResult(data, false);
    pushHistory(data, false);
  } finally {
    analyzeOverviewEl.disabled = false;
    analyzeOverviewEl.textContent = "Repo Overview";
  }
}

async function analyzeImpact() {
  analyzeImpactEl.disabled = true;
  analyzeImpactEl.textContent = "Analyzing...";
  impactSummaryEl.textContent = "Computing blast radius...";
  const started = performance.now();

  try {
    const res = await fetch("/api/repo-impact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: sourceEl.value, module: impactModuleEl.value, metadata: repoMetadata }),
    });
    const data = await res.json();
    data.client_duration_ms = Math.round(performance.now() - started);
    if (res.ok && data.payload) {
      impactSummaryEl.textContent = `${data.payload.transitive_dependents.length} dependents / ${data.payload.transitive_dependencies.length} dependencies`;
    } else {
      impactSummaryEl.textContent = data.error || "Impact analysis failed";
    }
    renderResult(data, res.ok);
    pushHistory(data, res.ok);
  } catch (error) {
    const data = {
      action: "repo-impact",
      error: String(error),
      exit_code: 1,
      stderr: String(error),
      client_duration_ms: Math.round(performance.now() - started),
    };
    impactSummaryEl.textContent = "Impact analysis failed";
    renderResult(data, false);
    pushHistory(data, false);
  } finally {
    analyzeImpactEl.disabled = false;
    analyzeImpactEl.textContent = "Analyze Impact";
  }
}

async function buildChangeBrief() {
  changeBriefEl.disabled = true;
  changeBriefEl.textContent = "Building...";
  briefSummaryEl.textContent = "Building change brief...";
  const started = performance.now();

  try {
    const res = await fetch("/api/repo-brief", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: sourceEl.value, module: impactModuleEl.value, metadata: repoMetadata }),
    });
    const data = await res.json();
    data.client_duration_ms = Math.round(performance.now() - started);
    if (res.ok && data.payload) {
      briefSummaryEl.textContent = `${data.payload.risk_level} risk / ${data.payload.blast_radius} dependents`;
    } else {
      briefSummaryEl.textContent = data.error || "Brief failed";
    }
    renderResult(data, res.ok);
    pushHistory(data, res.ok);
  } catch (error) {
    const data = {
      action: "repo-brief",
      error: String(error),
      exit_code: 1,
      stderr: String(error),
      client_duration_ms: Math.round(performance.now() - started),
    };
    briefSummaryEl.textContent = "Brief failed";
    renderResult(data, false);
    pushHistory(data, false);
  } finally {
    changeBriefEl.disabled = false;
    changeBriefEl.textContent = "Change Brief";
  }
}

function setRepoBaseline() {
  repoBaseline = {
    source: sourceEl.value,
    metadata: repoMetadata,
    summary: programSummaryEl.textContent,
    at: new Date().toISOString(),
  };
  updateBaselineSummary();
  saveState();
}

async function compareRepoGraph() {
  if (!repoBaseline?.source) {
    setRepoBaseline();
    compareSummaryEl.textContent = "Baseline captured. Edit or reload, then compare.";
    return;
  }

  compareGraphEl.disabled = true;
  compareGraphEl.textContent = "Comparing...";
  compareSummaryEl.textContent = "Comparing baseline to current graph...";
  const started = performance.now();

  try {
    const res = await fetch("/api/repo-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        before_source: repoBaseline.source,
        after_source: sourceEl.value,
        before_metadata: repoBaseline.metadata,
        after_metadata: repoMetadata,
      }),
    });
    const data = await res.json();
    data.client_duration_ms = Math.round(performance.now() - started);
    if (res.ok && data.payload) {
      compareSummaryEl.textContent = `${data.payload.risk_level} risk / ${data.payload.summary}`;
    } else {
      compareSummaryEl.textContent = data.error || "Compare failed";
    }
    renderResult(data, res.ok);
    pushHistory(data, res.ok);
  } catch (error) {
    const data = {
      action: "repo-compare",
      error: String(error),
      exit_code: 1,
      stderr: String(error),
      client_duration_ms: Math.round(performance.now() - started),
    };
    compareSummaryEl.textContent = "Compare failed";
    renderResult(data, false);
    pushHistory(data, false);
  } finally {
    compareGraphEl.disabled = false;
    compareGraphEl.textContent = "Compare Graph";
  }
}

function repoMetadataFromPayload(payloadData) {
  return {
    symbol_to_module: payloadData.symbol_to_module || {},
    symbol_to_file: payloadData.symbol_to_file || {},
  };
}

function payload() {
  return {
    source: sourceEl.value,
    relation: relationEl.value,
    arg1: arg1El.value,
    arg2: arg2El.value,
    recursive: recursiveEl.checked,
  };
}

async function invoke(action) {
  const button = document.querySelector(`[data-action="${action}"]`);
  setRunning(action, button);
  const started = performance.now();

  try {
    const res = await fetch(`/api/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    const data = await res.json();
    data.client_duration_ms = Math.round(performance.now() - started);
    renderResult(data, res.ok);
    pushHistory(data, res.ok);
  } catch (error) {
    const data = {
      action,
      error: String(error),
      exit_code: 1,
      stderr: String(error),
      client_duration_ms: Math.round(performance.now() - started),
    };
    renderResult(data, false);
    pushHistory(data, false);
  } finally {
    setRunning(null, button);
  }
}

function setRunning(action, button) {
  document.querySelectorAll("[data-action]").forEach((el) => {
    el.disabled = Boolean(action);
    el.classList.toggle("loading", Boolean(action && el === button));
  });
  if (!action) {
    return;
  }
  answerStripEl.className = "answer-strip running";
  answerTitleEl.textContent = `${labelForAction(action)} running`;
  durationEl.textContent = "...";
  exitCodeEl.textContent = "running";
  resultEl.textContent = `$ ${action} ${targetText()}`;
  proofEl.replaceChildren(emptyState("Waiting for result"));
  jsonEl.textContent = "";
}

function renderResult(data, ok) {
  const payloadData = data.payload;
  const answer = payloadData?.answer;
  const hasAnswer = typeof answer === "boolean";
  const action = data.action || "action";
  const duration = data.duration_ms ?? data.client_duration_ms ?? 0;
  const state = ok ? (hasAnswer && !answer ? "false" : "true") : "error";

  answerStripEl.className = `answer-strip ${state}`;
  answerTitleEl.textContent = resultTitle(action, answer, ok);
  durationEl.textContent = `${duration} ms`;
  exitCodeEl.textContent = ok ? "exit 0" : `exit ${data.exit_code ?? 1}`;

  resultEl.textContent = data.stdout || data.stderr || data.error || "(no output)";
  jsonEl.textContent = JSON.stringify(payloadData || data, null, 2);
  renderProof(payloadData, data.stdout);
  setTab(
    action === "prove" ||
      action === "why-not" ||
      action === "repo-impact" ||
      action === "repo-overview" ||
      action === "repo-brief" ||
      action === "repo-compare"
      ? "proof"
      : "output",
  );
}

function renderIdle() {
  answerStripEl.className = "answer-strip";
  answerTitleEl.textContent = "Ready";
  durationEl.textContent = "0 ms";
  exitCodeEl.textContent = "idle";
  resultEl.textContent = "Run the current program or inspect a query.";
  jsonEl.textContent = "{}";
  proofEl.replaceChildren(emptyState("No proof yet"));
}

function renderProof(payloadData, fallbackText = "") {
  proofEl.replaceChildren();

  if (!payloadData) {
    proofEl.append(emptyState("No structured payload"));
    return;
  }

  if (payloadData.proof && typeof payloadData.proof === "object") {
    proofEl.append(renderProofNode(payloadData.proof, false));
    return;
  }

  if (payloadData.explanation) {
    proofEl.append(renderProofNode(payloadData.explanation, true));
    return;
  }

  if (Array.isArray(payloadData.outputs)) {
    const list = document.createElement("div");
    list.className = "output-stack";
    payloadData.outputs.forEach((text, index) => {
      const item = document.createElement("pre");
      item.textContent = text || `(command ${index + 1} produced no output)`;
      list.append(item);
    });
    proofEl.append(list);
    return;
  }

  if (typeof payloadData.proof === "string") {
    const pre = document.createElement("pre");
    pre.textContent = payloadData.proof || fallbackText;
    proofEl.append(pre);
    return;
  }

  if (payloadData.source && Array.isArray(payloadData.modules)) {
    const summary = document.createElement("div");
    summary.className = "query-summary";
    summary.textContent = `${payloadData.modules.length} modules / ${payloadData.imports.length} imports`;
    proofEl.append(summary);
    return;
  }

  if (Array.isArray(payloadData.transitive_dependents)) {
    proofEl.append(renderImpactView(payloadData));
    return;
  }

  if (Array.isArray(payloadData.top_dependents)) {
    proofEl.append(renderOverviewView(payloadData));
    return;
  }

  if (Array.isArray(payloadData.read_first) && Array.isArray(payloadData.proof_checks)) {
    proofEl.append(renderBriefView(payloadData));
    return;
  }

  if (Array.isArray(payloadData.added_imports) && Array.isArray(payloadData.removed_imports)) {
    proofEl.append(renderCompareView(payloadData));
    return;
  }

  if ("answer" in payloadData) {
    const summary = document.createElement("div");
    summary.className = "query-summary";
    summary.textContent = `${targetText()} = ${payloadData.answer}`;
    proofEl.append(summary);
    return;
  }

  proofEl.append(emptyState("No proof tree for this action"));
}

function renderProofNode(node, negative) {
  node = normalizeProofNode(node);
  const wrap = document.createElement("div");
  wrap.className = `proof-node ${negative ? "negative" : "positive"}`;

  const header = document.createElement("div");
  header.className = "proof-head";
  const title = document.createElement("strong");
  title.textContent = formatHead(node.head);
  header.append(title);

  const meta = document.createElement("span");
  meta.textContent = negative ? node.reason || "not derivable" : confidenceText(node.confidence);
  header.append(meta);
  wrap.append(header);

  if (node.source) {
    const source = document.createElement("div");
    source.className = "proof-source";
    source.textContent = `${node.source.file}:${node.source.lineno}`;
    wrap.append(source);
  }

  if (Array.isArray(node.body) && node.body.length) {
    const children = document.createElement("div");
    children.className = "proof-children";
    node.body.forEach((child) => children.append(renderProofNode(child, negative)));
    wrap.append(children);
  }

  return wrap;
}

function renderImpactView(payloadData) {
  const wrap = document.createElement("div");
  wrap.className = "impact-view";

  const headline = document.createElement("div");
  headline.className = "query-summary";
  headline.textContent = `${formatModule(payloadData.module, payloadData)}: ${payloadData.transitive_dependents.length} transitive dependents, ${payloadData.transitive_dependencies.length} transitive dependencies`;
  wrap.append(headline);

  [
    ["Direct imports", payloadData.direct_imports],
    ["Direct imported by", payloadData.direct_imported_by],
    ["Transitive dependents", payloadData.transitive_dependents],
    ["Transitive dependencies", payloadData.transitive_dependencies],
  ].forEach(([label, items]) => {
    wrap.append(renderListBlock(label, items, payloadData));
  });

  wrap.append(renderPathBlock("Dependent paths", payloadData.dependent_paths, payloadData));
  wrap.append(renderPathBlock("Dependency paths", payloadData.dependency_paths, payloadData));
  return wrap;
}

function renderOverviewView(payloadData) {
  const wrap = document.createElement("div");
  wrap.className = "impact-view";

  const headline = document.createElement("div");
  headline.className = "query-summary";
  headline.textContent = `${payloadData.modules} modules, ${payloadData.imports} imports, ${payloadData.cycles.length} cycles`;
  wrap.append(headline);

  wrap.append(renderRankBlock("Highest blast radius", payloadData.top_dependents, payloadData, "dependents"));
  wrap.append(renderRankBlock("Highest dependency fanout", payloadData.top_dependencies, payloadData, "dependencies"));
  wrap.append(renderRankBlock("Most direct imports", payloadData.direct_import_hubs, payloadData, "imports"));
  wrap.append(renderRankBlock("Most directly imported", payloadData.direct_imported_hubs, payloadData, "callers"));
  wrap.append(renderListBlock("Entrypoints", payloadData.entrypoints.slice(0, 12), payloadData));
  wrap.append(renderListBlock("Leaves", payloadData.leaves.slice(0, 12), payloadData));
  wrap.append(renderCyclesBlock(payloadData.cycles, payloadData));

  return wrap;
}

function renderBriefView(payloadData) {
  const wrap = document.createElement("div");
  wrap.className = "impact-view";

  const headline = document.createElement("div");
  headline.className = "query-summary";
  headline.textContent = `${formatModule(payloadData.module, payloadData)}: ${payloadData.risk_level} risk, ${payloadData.blast_radius} dependents, ${payloadData.coupling} dependencies`;
  wrap.append(headline);

  wrap.append(renderDetailRowsBlock("Read first", payloadData.read_first));
  wrap.append(renderDetailRowsBlock("Regression watch", payloadData.regression_targets));
  wrap.append(renderBriefPathBlock("Proof paths", payloadData.path_examples, payloadData));
  wrap.append(renderProofChecksBlock(payloadData.proof_checks, payloadData));

  const pre = document.createElement("pre");
  pre.textContent = payloadData.markdown;
  wrap.append(pre);
  return wrap;
}

function renderCompareView(payloadData) {
  const wrap = document.createElement("div");
  wrap.className = "impact-view";

  const headline = document.createElement("div");
  headline.className = "query-summary";
  headline.textContent = `${payloadData.risk_level} risk: ${payloadData.summary}`;
  wrap.append(headline);

  wrap.append(renderCompareStats(payloadData));
  wrap.append(renderEdgeBlock("Added imports", payloadData.added_imports, payloadData, "added"));
  wrap.append(renderEdgeBlock("Removed imports", payloadData.removed_imports, payloadData, "removed"));
  wrap.append(renderListBlock("Added modules", payloadData.added_modules, payloadData));
  wrap.append(renderListBlock("Removed modules", payloadData.removed_modules, payloadData));
  wrap.append(renderDeltaBlock(payloadData.blast_radius_deltas, payloadData));
  wrap.append(renderCompareCyclesBlock("Introduced cycles", payloadData.introduced_cycles, payloadData));
  wrap.append(renderCompareCyclesBlock("Resolved cycles", payloadData.resolved_cycles, payloadData));
  wrap.append(renderProofChecksBlock(payloadData.suggested_checks, payloadData));

  const pre = document.createElement("pre");
  pre.textContent = payloadData.markdown;
  wrap.append(pre);
  return wrap;
}

function renderCompareStats(payloadData) {
  const block = document.createElement("div");
  block.className = "compare-stats";
  [
    ["Before", payloadData.before],
    ["After", payloadData.after],
  ].forEach(([label, stats]) => {
    const item = document.createElement("div");
    item.className = "stat-card";
    const title = document.createElement("strong");
    title.textContent = label;
    const body = document.createElement("span");
    body.textContent = `${stats.modules} modules / ${stats.imports} imports / ${stats.cycles} cycles`;
    item.append(title, body);
    block.append(item);
  });
  return block;
}

function renderEdgeBlock(label, edges, payloadData, className) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `${label} (${edges.length})`;
  block.append(title);
  if (!edges.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "edge-list";
  edges.forEach(([src, dst]) => {
    const button = document.createElement("button");
    button.className = `edge-item ${className}`;
    button.type = "button";
    button.textContent = `${formatModule(src, payloadData)} -> ${formatModule(dst, payloadData)}`;
    button.addEventListener("click", () => {
      relationEl.value = "depends_on";
      arg1El.value = src;
      arg2El.value = dst;
      recursiveEl.checked = true;
      queueSave();
      invoke(className === "removed" ? "why-not" : "prove");
    });
    list.append(button);
  });
  block.append(list);
  return block;
}

function renderDeltaBlock(items, payloadData) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `Blast-radius changes (${items.length})`;
  block.append(title);
  if (!items.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "rank-list";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.className = "rank-item";
    button.type = "button";
    button.textContent = `${formatModule(item.module, payloadData)}: ${item.before} -> ${item.after} (${formatSigned(item.delta)})`;
    button.addEventListener("click", () => {
      impactModuleEl.value = item.module;
      analyzeImpact();
    });
    list.append(button);
  });
  block.append(list);
  return block;
}

function renderCompareCyclesBlock(label, cycles, payloadData) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `${label} (${cycles.length})`;
  block.append(title);
  if (!cycles.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "path-list";
  cycles.forEach((cycle) => {
    const item = document.createElement("button");
    item.className = "path-item";
    item.type = "button";
    item.textContent = cycle.map((symbol) => formatModule(symbol, payloadData)).join(" -> ");
    item.addEventListener("click", () => {
      impactModuleEl.value = cycle[0];
      analyzeImpact();
    });
    list.append(item);
  });
  block.append(list);
  return block;
}

function renderRankBlock(label, items, payloadData, unit) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = label;
  block.append(title);
  if (!items.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "rank-list";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.className = "rank-item";
    button.type = "button";
    button.textContent = `${formatModule(item.module, payloadData)}: ${item.count} ${unit}`;
    const detail = moduleDetail(item.module, payloadData);
    if (detail?.file) {
      button.title = detail.file;
    }
    button.addEventListener("click", () => {
      impactModuleEl.value = item.module;
      analyzeImpact();
    });
    list.append(button);
  });
  block.append(list);
  return block;
}

function renderDetailRowsBlock(label, rows) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `${label} (${rows.length})`;
  block.append(title);
  if (!rows.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "detail-list";
  rows.forEach((row) => {
    const button = document.createElement("button");
    button.className = "detail-item";
    button.type = "button";
    button.textContent = formatModule(row.symbol, { module_details: { [row.symbol]: row } });
    if (row.file) {
      button.title = row.file;
      const file = document.createElement("span");
      file.textContent = row.file;
      button.append(file);
    }
    button.addEventListener("click", () => {
      impactModuleEl.value = row.symbol;
      analyzeImpact();
    });
    list.append(button);
  });
  block.append(list);
  return block;
}

function renderBriefPathBlock(label, paths, payloadData) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `${label} (${paths.length})`;
  block.append(title);
  if (!paths.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "path-list";
  paths.forEach((item) => {
    const button = document.createElement("button");
    button.className = "path-item";
    button.type = "button";
    button.textContent = `${item.kind}: ${item.path.map((symbol) => formatModule(symbol, payloadData)).join(" -> ")}`;
    button.addEventListener("click", () => {
      relationEl.value = "depends_on";
      arg1El.value = item.path[0];
      arg2El.value = item.path[item.path.length - 1];
      recursiveEl.checked = true;
      queueSave();
      invoke("prove");
    });
    list.append(button);
  });
  block.append(list);
  return block;
}

function renderProofChecksBlock(checks, payloadData) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `Suggested checks (${checks.length})`;
  block.append(title);
  if (!checks.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "check-list";
  checks.forEach((check) => {
    const button = document.createElement("button");
    button.className = "check-item";
    button.type = "button";
    const action = check.action || "prove";
    const prefix = check.kind ? `${check.kind}: ` : "";
    button.textContent = `${prefix}${action} ${check.relation}(${formatModule(check.arg1, payloadData)}, ${formatModule(check.arg2, payloadData)})`;
    button.addEventListener("click", () => {
      relationEl.value = check.relation;
      arg1El.value = check.arg1;
      arg2El.value = check.arg2;
      recursiveEl.checked = Boolean(check.recursive);
      queueSave();
      invoke(action);
    });
    list.append(button);
  });
  block.append(list);
  return block;
}

function renderCyclesBlock(cycles, payloadData) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `Cycles (${cycles.length})`;
  block.append(title);
  if (!cycles.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "path-list";
  cycles.slice(0, 8).forEach((cycle) => {
    const item = document.createElement("button");
    item.className = "path-item";
    item.type = "button";
    item.textContent = cycle.map((symbol) => formatModule(symbol, payloadData)).join(" -> ");
    item.addEventListener("click", () => {
      impactModuleEl.value = cycle[0];
      analyzeImpact();
    });
    list.append(item);
  });
  block.append(list);
  return block;
}

function renderListBlock(label, items, payloadData) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = `${label} (${items.length})`;
  block.append(title);
  if (!items.length) {
    block.append(emptyState("none"));
    return block;
  }
  const chips = document.createElement("div");
  chips.className = "impact-chips";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.className = "chip";
    button.type = "button";
    button.textContent = formatModule(item, payloadData);
    const detail = moduleDetail(item, payloadData);
    if (detail?.file) {
      button.title = detail.file;
    }
    button.addEventListener("click", () => {
      impactModuleEl.value = item;
      analyzeImpact();
    });
    chips.append(button);
  });
  block.append(chips);
  return block;
}

function renderPathBlock(label, paths, payloadData) {
  const block = document.createElement("div");
  block.className = "impact-block";
  const title = document.createElement("strong");
  title.textContent = label;
  block.append(title);
  const entries = Object.entries(paths || {});
  if (!entries.length) {
    block.append(emptyState("none"));
    return block;
  }
  const list = document.createElement("div");
  list.className = "path-list";
  entries.slice(0, 12).forEach(([module, path]) => {
    const item = document.createElement("button");
    item.className = "path-item";
    item.type = "button";
    item.textContent = `${formatModule(module, payloadData)}: ${path.map((symbol) => formatModule(symbol, payloadData)).join(" -> ")}`;
    item.addEventListener("click", () => {
      relationEl.value = "depends_on";
      arg1El.value = path[0];
      arg2El.value = path[path.length - 1];
      recursiveEl.checked = true;
      queueSave();
      invoke("prove");
    });
    list.append(item);
  });
  if (entries.length > 12) {
    const more = document.createElement("div");
    more.className = "muted";
    more.textContent = `+${entries.length - 12} more paths`;
    list.append(more);
  }
  block.append(list);
  return block;
}

function formatModule(symbol, payloadData) {
  const detail = moduleDetail(symbol, payloadData);
  if (!detail || !detail.module || detail.module === symbol) {
    return symbol;
  }
  return `${symbol} (${detail.module})`;
}

function moduleDetail(symbol, payloadData) {
  return payloadData?.module_details?.[symbol] || null;
}

function normalizeProofNode(node) {
  if (node?.proof && typeof node.proof === "object") {
    return node.proof;
  }
  if (node?.explanation && typeof node.explanation === "object") {
    return node.explanation;
  }
  return node || {};
}

function analyzeSource() {
  const lines = sourceEl.value.split("\n");
  const logicalLines = lines.map((line) => line.trim()).filter((line) => line && !line.startsWith("#"));
  const groups = {
    domains: [],
    relations: [],
    facts: [],
    rules: [],
    commands: [],
  };

  logicalLines.forEach((line) => {
    collectMatch(groups.domains, line, /^domain\s+(\w+)/);
    collectMatch(groups.relations, line, /^relation\s+(\w+)/);
    collectMatch(groups.facts, line, /^fact\s+([\w(,\s)]+)/);
    collectMatch(groups.rules, line, /^rule\s+(.+)/);
    if (/^(query|prove)\s+/.test(line)) {
      groups.commands.push(line);
    }
  });

  const summary = `${groups.domains.length} domains / ${groups.relations.length} relations / ${groups.facts.length} facts / ${groups.rules.length} rules`;
  programSummaryEl.textContent = summary;
  lineCountEl.textContent = `${logicalLines.length} lines`;
  renderProgramIndex(groups);
  renderRelationOptions(groups.relations);
  renderArgumentOptions(extractDomainSymbols(sourceEl.value));
}

function collectMatch(target, line, pattern) {
  const match = line.match(pattern);
  if (match) {
    target.push(match[1].replace(/\s+/g, " ").trim());
  }
}

function renderProgramIndex(groups) {
  programIndexEl.replaceChildren();
  const sections = [
    ["domains", groups.domains],
    ["relations", groups.relations],
    ["facts", groups.facts],
    ["rules", groups.rules],
    ["commands", groups.commands],
  ];

  sections.forEach(([label, items]) => {
    const block = document.createElement("div");
    block.className = "index-group";
    const head = document.createElement("div");
    head.className = "index-head";
    head.textContent = `${label} ${items.length}`;
    block.append(head);

    if (items.length === 0) {
      const empty = document.createElement("span");
      empty.className = "muted";
      empty.textContent = "none";
      block.append(empty);
    } else {
      items.slice(0, 8).forEach((item) => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.type = "button";
        chip.textContent = item;
        chip.addEventListener("click", () => {
          if (label === "relations") {
            relationEl.value = item;
            queueSave();
          }
        });
        block.append(chip);
      });
      if (items.length > 8) {
        const more = document.createElement("span");
        more.className = "muted";
        more.textContent = `+${items.length - 8} more`;
        block.append(more);
      }
    }
    programIndexEl.append(block);
  });
}

function renderRelationOptions(relations) {
  relationOptionsEl.replaceChildren();
  relations.forEach((relation) => {
    const option = document.createElement("option");
    option.value = relation;
    relationOptionsEl.append(option);
  });
}

function renderArgumentOptions(symbols) {
  argumentOptionsEl.replaceChildren();
  symbols.forEach((symbol) => {
    const option = document.createElement("option");
    option.value = symbol;
    argumentOptionsEl.append(option);
  });
}

function extractDomainSymbols(source) {
  const symbols = [];
  const pattern = /domain\s+\w+\s*\{([\s\S]*?)\}/g;
  let match;
  while ((match = pattern.exec(source)) !== null) {
    match[1]
      .split(/[\s,]+/)
      .map((symbol) => symbol.trim())
      .filter(Boolean)
      .forEach((symbol) => symbols.push(symbol));
  }
  return [...new Set(symbols)].sort();
}

function pushHistory(data, ok) {
  history.unshift({
    id: globalThis.crypto?.randomUUID?.() || String(Date.now() + Math.random()),
    ok,
    data,
    at: new Date(),
  });
  history = history.slice(0, HISTORY_LIMIT);
  renderHistory();
}

function renderHistory() {
  historyEl.replaceChildren();
  if (!history.length) {
    historyEl.append(emptyState("No runs yet"));
    return;
  }
  history.forEach((entry) => {
    const button = document.createElement("button");
    button.className = `history-item ${entry.ok ? "ok" : "bad"}`;
    button.type = "button";
    button.addEventListener("click", () => renderResult(entry.data, entry.ok));

    const top = document.createElement("span");
    top.textContent = `${labelForAction(entry.data.action)} ${historyAnswer(entry.data)}`;
    const bottom = document.createElement("small");
    bottom.textContent = `${entry.data.command || targetText()} / ${entry.at.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })}`;

    button.append(top, bottom);
    historyEl.append(button);
  });
}

function setTab(name) {
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === name);
  });
}

function saveState() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        presetId: presetEl.value,
        source: sourceEl.value,
        relation: relationEl.value,
        arg1: arg1El.value,
        arg2: arg2El.value,
        recursive: recursiveEl.checked,
        repoMetadata,
        repoBaseline,
      }),
    );
  } catch {
    // Local storage is optional for the workbench.
  }
}

function queueSave() {
  clearTimeout(saveTimer);
  saveTimer = window.setTimeout(saveState, 150);
}

function loadState() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  } catch {
    return null;
  }
}

function updateActivePreset() {
  const example = EXAMPLES.find((item) => item.id === presetEl.value) || EXAMPLES[0];
  activePresetEl.textContent = example.name;
}

function updateBaselineSummary() {
  if (!repoBaseline?.source) {
    compareSummaryEl.textContent = "Set a baseline before editing or reloading imports.";
    return;
  }
  const time = new Date(repoBaseline.at).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  compareSummaryEl.textContent = `Baseline: ${repoBaseline.summary || "saved graph"} at ${time}`;
}

function targetText() {
  return `${relationEl.value || "relation"}(${arg1El.value || "arg1"}, ${arg2El.value || "arg2"})`;
}

function resultTitle(action, answer, ok) {
  if (!ok) {
    return `${labelForAction(action)} failed`;
  }
  if (typeof answer === "boolean") {
    return `${targetText()} = ${answer}`;
  }
  return `${labelForAction(action)} complete`;
}

function historyAnswer(data) {
  const answer = data.payload?.answer;
  if (typeof answer === "boolean") {
    return answer ? "true" : "false";
  }
  return data.exit_code === 0 ? "ok" : "failed";
}

function labelForAction(action = "") {
  return action
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function confidenceText(confidence) {
  if (typeof confidence !== "number") {
    return "derived";
  }
  return `${Math.round(confidence * 1000) / 1000}`;
}

function formatSigned(value) {
  return value > 0 ? `+${value}` : String(value);
}

function formatHead(head) {
  if (!Array.isArray(head) || head.length < 3) {
    return "unknown";
  }
  return `${head[0]}(${head[1]}, ${head[2]})`;
}

function emptyState(text) {
  const node = document.createElement("div");
  node.className = "empty";
  node.textContent = text;
  return node;
}
