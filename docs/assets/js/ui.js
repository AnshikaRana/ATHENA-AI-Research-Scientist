/* Athena - the interface (the browser counterpart of the Streamlit UI and CLI). */
(function (A) {
  "use strict";
  const { Settings, DB, Log, Util, Uploads, LongTermMemory } = A;
  const esc = Util.escape;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const STAGE_LABELS = {
    ingest: ["Ingest sources", "document ingestion"],
    planning: ["Plan", "Principal Investigator"],
    literature: ["Literature", "Literature Review Agent"],
    synthesis: ["Synthesis", "Knowledge Synthesist"],
    gap_analysis: ["Gap analysis", "Research Gap Analyst"],
    hypothesis: ["Hypotheses", "Hypothesis Generator"],
    experiment: ["Experiment design", "Experiment Planner"],
    citations: ["Citations", "Citation Manager"],
    report: ["Report", "Scientific Writer"],
  };
  const WEB_STAGES = ["planning", "literature", "synthesis", "gap_analysis", "hypothesis", "experiment", "citations", "report"];
  const EXAMPLES = [
    "Do vision transformers outperform CNNs on small (<10k image) datasets without pretraining?",
    "How does retrieval-augmented generation reduce hallucination in large language models?",
    "Does spaced repetition improve long-term retention of vocabulary in adult second-language learners?",
  ];

  const state = {
    pipeline: null,     // the live pipeline, while running or just finished
    view: null,         // { ctx, manifest, trace, stages, status, mode } of a run opened from History
    running: false,
    sourceMode: "web",
    tab: "report",
    stageStatus: {},
    renderQueued: false,
  };

  // ---------------------------------------------------------------------
  // Rendering helpers
  // ---------------------------------------------------------------------

  const CITE_RE = /\b(?:arxiv|doi|openalex|upload):[^\s,;\])"'<&]+/g;
  function citeLink(id, ctx) {
    const p = ctx && ctx.corpus[id];
    if (!p) return null;
    const href = p.url && !p.url.startsWith("local:") ? p.url : "#";
    return `<a class="cite" href="${esc(href)}" target="_blank" rel="noopener" title="${esc(p.title)}">${esc(id)}</a>`;
  }
  function linkify(escaped, ctx) {
    if (!ctx) return escaped;
    return escaped.replace(CITE_RE, (m) => {
      const id = m.replace(/[.,;)\]]+$/, "");
      const link = citeLink(id, ctx);
      return link ? link + m.slice(id.length) : m;
    });
  }
  const plain = (text, ctx) => linkify(esc(text == null ? "" : text), ctx);
  function md(text, ctx) {
    let t = String(text || "");
    if (ctx) t = t.replace(CITE_RE, (m) => { const id = m.replace(/[.,;)\]]+$/, ""); const link = citeLink(id, ctx); return link ? link + m.slice(id.length) : m; });
    if (window.marked && window.DOMPurify) {
      return window.DOMPurify.sanitize(window.marked.parse(t), { ADD_ATTR: ["target", "rel"] });
    }
    return t.split(/\n{2,}/).map((p) => `<p>${p.replace(/<(?!\/?a[\s>])/g, "&lt;").replace(/\n/g, "<br>")}</p>`).join("");
  }
  const list = (items, ctx) => (items && items.length ? `<ul class="list-tight">${items.map((i) => `<li>${plain(i, ctx)}</li>`).join("")}</ul>` : `<p class="muted small">None recorded.</p>`);
  const badge = (text, kind) => `<span class="badge ${kind || "neutral"}">${esc(text)}</span>`;
  const confBadge = (c) => badge(c, c === "high" ? "ok" : c === "medium" ? "info" : "warn");
  function toast(message, ms = 3200) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), ms);
  }
  const fmtDate = (iso) => { try { return new Date(iso).toLocaleString(); } catch (e) { return iso; } };

  /** The run currently on screen: the live pipeline, or one opened from History. */
  function current() {
    if (state.view) return state.view;
    const p = state.pipeline;
    if (!p || !p.ctx) return null;
    return { ctx: p.ctx, manifest: p.runManifest(), trace: Log.events, stages: p.stages, status: p.status, mode: p.live ? "live" : "offline", live: true };
  }

  function queueRender() {
    if (state.renderQueued) return;
    state.renderQueued = true;
    requestAnimationFrame(() => { state.renderQueued = false; renderWorkspace(); });
  }

  // ---------------------------------------------------------------------
  // Mode pill, theme, routing
  // ---------------------------------------------------------------------

  function updateModePill() {
    const pill = $("#mode-pill");
    const live = Settings.live;
    pill.classList.toggle("live", live);
    $("#mode-label").textContent = live ? `Live · ${Settings.get().reasoning_model}` : Settings.apiKey() ? "Offline (forced)" : "Offline demo";
    renderComposerNotice();
  }

  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("athena.theme"); } catch (e) { /* blocked */ }
    if (saved) document.documentElement.dataset.theme = saved;
    $("#theme-btn").addEventListener("click", () => {
      const isLight = document.documentElement.dataset.theme
        ? document.documentElement.dataset.theme === "light"
        : window.matchMedia("(prefers-color-scheme: light)").matches;
      const next = isLight ? "dark" : "light";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("athena.theme", next); } catch (e) { /* blocked */ }
    });
  }

  function route() {
    const view = (location.hash || "#study").slice(1).split("?")[0] || "study";
    const known = ["study", "history", "memory", "agents", "about"];
    const v = known.includes(view) ? view : "study";
    for (const k of known) $(`#view-${k}`).classList.toggle("hidden", k !== v);
    $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === v));
    if (v === "history") renderHistory();
    if (v === "memory") renderMemory();
    if (v === "agents") renderAgents();
    window.scrollTo({ top: 0 });
  }

  // ---------------------------------------------------------------------
  // Composer
  // ---------------------------------------------------------------------

  function renderComposerNotice() {
    const el = $("#composer-notice");
    if (!el) return;
    const notes = [];
    if (!Settings.live) {
      notes.push(`<div class="notice warn small"><b>Offline demo mode.</b> Without an OpenAI API key, deterministic stand-in agents run the whole pipeline: literature search, retrieval, citation checks, power analysis and approvals are real, but the writing is templated. <a href="#" data-open-settings>Add your key</a> for full model-written studies.</div>`);
    }
    if (state.sourceMode === "documents" && !Uploads.list().some((f) => /\.pdf$/i.test(f.name))) {
      notes.push(`<div class="notice small">Document-grounded mode reads only your PDFs — attach at least one.</div>`);
    }
    el.innerHTML = notes.join("");
  }

  function renderFileChips() {
    $("#file-chips").innerHTML = Uploads.list().map((f) =>
      `<span class="chip" title="${esc(f.type || "")}">${esc(f.name)} <span class="muted">${(f.size / 1024).toFixed(0)} KB</span><button type="button" data-remove-file="${esc(f.name)}" aria-label="Remove ${esc(f.name)}">✕</button></span>`).join("");
    renderComposerNotice();
  }

  function addFiles(files) {
    for (const f of files) {
      if (f.size > 40 * 1024 * 1024) { toast(`${f.name} is larger than 40 MB and was skipped.`); continue; }
      Uploads.add(f);
    }
    renderFileChips();
  }

  function initComposer() {
    $("#examples").innerHTML = EXAMPLES.map((q) => `<button type="button">${esc(q)}</button>`).join("");
    $("#examples").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) { $("#question").value = b.textContent; $("#question").focus(); }
    });
    $("#file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
    const dz = $("#dropzone");
    ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
    dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
    $("#file-chips").addEventListener("click", (e) => {
      const b = e.target.closest("[data-remove-file]");
      if (b) { Uploads.remove(b.dataset.removeFile); renderFileChips(); }
    });
    const setMode = (mode) => {
      state.sourceMode = mode;
      $("#src-web").classList.toggle("on", mode === "web");
      $("#src-docs").classList.toggle("on", mode === "documents");
      renderComposerNotice();
      if (!state.running && !current()) renderStages();
    };
    $("#src-web").addEventListener("click", () => setMode("web"));
    $("#src-docs").addEventListener("click", () => setMode("documents"));
    $("#approval-toggle").checked = Settings.get().require_human_approval;
    $("#approval-toggle").addEventListener("change", (e) => Settings.save({ require_human_approval: e.target.checked }));
    $("#run-btn").addEventListener("click", runStudy);
    $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runStudy(); });
    $("#stop-btn").addEventListener("click", () => {
      if (state.pipeline) { state.pipeline.stopped = true; toast("Stopping after the current stage…"); $("#stop-btn").disabled = true; }
    });
    document.addEventListener("click", (e) => {
      if (e.target.closest("[data-open-settings]")) { e.preventDefault(); openSettings(); }
    });
  }

  // ---------------------------------------------------------------------
  // Running a study
  // ---------------------------------------------------------------------

  function onPipelineEvent(type, data) {
    if (data && data.pipeline) state.pipeline = data.pipeline;
    if (type === "stage") {
      state.stageStatus[data.stage] = data.status;
      if (data.status === "ok" && ["planning", "literature", "synthesis", "gap_analysis", "hypothesis", "experiment", "citations", "report"].includes(data.stage)) {
        const autoTab = { planning: "plan", literature: "corpus", synthesis: "synthesis", gap_analysis: "gaps", hypothesis: "hypotheses", experiment: "experiment", citations: "citations", report: "report" }[data.stage];
        if (state.followTabs) state.tab = autoTab;
      }
    }
    if (type === "status") {
      if (data.status === "awaiting_approval") toast("Run checkpointed. Resume it any time from History.", 4500);
    }
    queueRender();
  }

  async function runStudy() {
    if (state.running) return;
    const question = $("#question").value.trim();
    if (!question) { toast("Enter a research question first."); $("#question").focus(); return; }
    if (state.sourceMode === "documents" && !Uploads.list().some((f) => /\.pdf$/i.test(f.name))) {
      toast("Attach at least one PDF for document-grounded mode."); return;
    }
    Settings.save({ require_human_approval: $("#approval-toggle").checked });
    state.view = null;
    state.stageStatus = {};
    state.tab = "plan";
    state.followTabs = true;
    state.running = true;
    $("#feed").innerHTML = "";
    setRunning(true);
    const pipeline = new A.ResearchPipeline({ approvalHandler, onEvent: onPipelineEvent });
    state.pipeline = pipeline;
    renderWorkspace();
    try {
      const ctx = await pipeline.runStudy(question, { attachments: Uploads.list().map((f) => f.name), sourceMode: state.sourceMode });
      if (ctx.report) { state.tab = "report"; toast("Dossier ready."); }
      else if (ctx.input_guardrail) toast("The input guardrail stopped this run.");
    } catch (err) {
      console.error(err);
      toast(`Run failed: ${err.message}`, 6000);
      if (pipeline.ctx) { pipeline.ctx.recordError(`run aborted: ${err.message}`); pipeline.status = "failed"; await pipeline.saveRun("failed").catch(() => {}); }
    } finally {
      state.running = false;
      state.followTabs = false;
      setRunning(false);
      renderWorkspace();
    }
  }

  function setRunning(on) {
    $("#run-btn").disabled = on;
    $("#stop-btn").disabled = !on;
    $("#run-btn").innerHTML = on
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.2-8.6"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.9s" repeatCount="indefinite"/></path></svg>Running…`
      : `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>Run study`;
  }

  // ---------------------------------------------------------------------
  // Human approval (static gate on saving, dynamic gate on deep sweeps)
  // ---------------------------------------------------------------------

  function approvalHandler(item) {
    return new Promise((resolve, reject) => {
      const dlg = $("#approval-dialog");
      let args = {};
      try { args = JSON.parse(item.arguments || "{}"); } catch (e) { args = {}; }
      $("#appr-tool").textContent = item.name;
      $("#appr-agent").textContent = item.agent || "—";
      $("#appr-note").value = "";
      const ctx = state.pipeline && state.pipeline.ctx;
      if (item.name === "save_research_artifact") {
        $("#appr-sub").textContent = "The Scientific Writer wants to save the research dossier.";
        $("#appr-why").textContent = "Saving a dossier under your name is the one outward-facing act in the pipeline, so a person signs off every time.";
        $("#appr-preview").innerHTML = `<div class="small muted" style="margin-bottom:8px">${esc(args.filename || "")} · ${esc(args.artifact_type || "")} · ${(args.content || "").length.toLocaleString()} chars</div>${md(args.content || "", ctx)}`;
      } else if (item.name === "commission_deep_sweep") {
        $("#appr-sub").textContent = "The Literature Review Agent wants to run a wide literature sweep.";
        $("#appr-why").textContent = "Sweeps above 30 total results pause for authorisation because of their API cost.";
        $("#appr-preview").innerHTML = `<p><b>${esc(args.total_results)}</b> results across:</p>${list(args.queries || [])}`;
      } else {
        $("#appr-sub").textContent = "A gated tool call needs your decision.";
        $("#appr-why").textContent = "This tool is configured to require approval.";
        $("#appr-preview").innerHTML = `<pre class="code">${esc(JSON.stringify(args, null, 2))}</pre>`;
      }
      const finish = (fn) => { cleanup(); dlg.close(); fn(); };
      const onApprove = () => finish(() => resolve({ approved: true, note: $("#appr-note").value.trim() || "approved interactively" }));
      const onReject = () => finish(() => resolve({ approved: false, note: $("#appr-note").value.trim() || "rejected interactively" }));
      const onDefer = () => finish(() => { const e = new A.errors.ApprovalDeferred(item.name, "run checkpointed"); e.item = item; reject(e); });
      const onCancel = (e) => e.preventDefault();
      function cleanup() {
        $("#appr-approve").removeEventListener("click", onApprove);
        $("#appr-reject").removeEventListener("click", onReject);
        $("#appr-defer").removeEventListener("click", onDefer);
        dlg.removeEventListener("cancel", onCancel);
      }
      $("#appr-approve").addEventListener("click", onApprove);
      $("#appr-reject").addEventListener("click", onReject);
      $("#appr-defer").addEventListener("click", onDefer);
      dlg.addEventListener("cancel", onCancel);
      dlg.showModal();
      queueRender();
    });
  }

  // ---------------------------------------------------------------------
  // Workspace: stages, metrics, feed, results
  // ---------------------------------------------------------------------

  function stageList(cur) {
    const mode = cur ? cur.ctx.source_mode : state.sourceMode;
    return mode === "documents" ? ["ingest"].concat(WEB_STAGES) : WEB_STAGES;
  }

  function renderStages() {
    const cur = current();
    const stages = stageList(cur);
    const results = {};
    for (const s of (cur && cur.stages) || []) results[s.stage] = s;
    const awaiting = cur && (cur.status === "awaiting_approval");
    $("#stages").innerHTML = stages.map((key, i) => {
      const r = results[key];
      let cls = "", t = "";
      if (r) { cls = r.ok ? "ok" : "failed"; t = `${(r.duration_s || 0).toFixed(1)}s`; }
      else if (state.stageStatus[key] === "running" && state.running) cls = "running";
      else if (awaiting && cur.ctx.stage === key) { cls = "waiting"; t = "approval"; }
      else if (key === "ingest" && state.stageStatus.ingest === "ok") cls = "ok";
      const [label, agent] = STAGE_LABELS[key];
      const icon = cls === "ok" ? "✓" : cls === "failed" ? "!" : i + 1;
      return `<li class="${cls}" title="${r && !r.ok ? esc(r.detail) : ""}"><span class="num">${icon}</span><span><span class="lbl">${label}</span><span class="agent">${esc((r && r.agent) || agent)}</span></span><span class="t">${t}</span></li>`;
    }).join("");
  }

  function renderMetrics() {
    const cur = current();
    const s = cur ? cur.ctx.summary() : null;
    const m = [
      ["Papers", s ? s.corpus_size : "–"],
      ["Chunks", s ? s.indexed_chunks : "–"],
      ["Tokens", s ? s.tokens.toLocaleString() : "–"],
      ["Cost", s ? Util.fmtUsd(s.estimated_usd) : "–"],
      ["Approvals", s ? s.approvals : "–"],
      ["Errors", s ? s.errors : "–"],
    ];
    $("#metrics").innerHTML = m.map(([k, v]) => `<div class="metric"><div class="v">${esc(v)}</div><div class="k">${k}</div></div>`).join("");
    $("#run-meta").innerHTML = cur
      ? `${badge(cur.mode === "live" ? "live" : "offline", cur.mode === "live" ? "ok" : "warn")} ${statusBadge(cur.status)} <span class="mono">${esc(cur.ctx.run_id)}</span>`
      : "No run yet.";
  }

  function statusBadge(status) {
    const map = { running: "info", complete: "ok", awaiting_approval: "warn", incomplete: "warn", failed: "fail" };
    return badge((status || "idle").replace("_", " "), map[status] || "neutral");
  }

  function feedLine(entry) {
    const time = (entry.ts || "").slice(11, 19);
    let html = null, err = entry.level === "ERROR";
    const m = entry.message;
    if (m === "tool start") html = `<b>${esc(entry.tool)}</b> <span class="muted">← ${esc(entry.agent)}</span><span class="pv">${esc(entry.args || "")}</span>`;
    else if (m === "tool end") html = `<span class="muted">↳ ${esc(entry.tool)}:</span> <span class="pv">${esc((entry.preview || "").split("\n")[0])}</span>`;
    else if (m.startsWith("stage start")) html = `<b>▸ ${esc(STAGE_LABELS[entry.stage] ? STAGE_LABELS[entry.stage][0] : entry.stage)}</b> <span class="muted">${esc(entry.agent)}</span>`;
    else if (m === "output guardrail") html = entry.fabricated && entry.fabricated.length ? `<b>citation guardrail</b> blocked ${esc(entry.fabricated.join(", "))}` : `<span class="muted">citation guardrail: ${entry.citations_seen || 0} ids verified${entry.overclaim_flags && entry.overclaim_flags.length ? `, ${entry.overclaim_flags.length} overclaim flag(s)` : ""}</span>`;
    else if (m === "input guardrail") html = `<span class="muted">input guardrail: ${entry.passed ? "passed" : "<b>blocked</b>"}</span>`;
    else if (/approval|checkpoint|re-asking|aborting|budget/.test(m) || err) { html = `<b>${esc(m)}</b> <span class="pv">${esc(entry.tool || entry.note || "")}</span>`; }
    if (!html) return null;
    return `<div class="ev${err ? " err" : ""}"><time>${time}</time><div>${html}</div></div>`;
  }

  function renderFeed() {
    const cur = current();
    const feed = $("#feed");
    if (!cur) return;
    const lines = (cur.trace || []).map(feedLine).filter(Boolean).slice(-160);
    feed.innerHTML = lines.join("") || `<div class="muted small">Waiting for activity…</div>`;
    feed.scrollTop = feed.scrollHeight;
  }

  const TABS = [
    ["report", "Report"], ["plan", "Plan"], ["corpus", "Corpus"], ["synthesis", "Synthesis"], ["gaps", "Gaps"],
    ["hypotheses", "Hypotheses"], ["experiment", "Experiment"], ["citations", "Citations"], ["trace", "Trace"],
  ];

  function renderTabs(cur) {
    const ctx = cur && cur.ctx;
    const counts = ctx ? {
      corpus: Object.keys(ctx.corpus).length,
      synthesis: ctx.synthesis ? ctx.synthesis.key_claims.length : 0,
      gaps: ctx.gap_analysis ? ctx.gap_analysis.gaps.length : 0,
      hypotheses: ctx.hypotheses ? ctx.hypotheses.hypotheses.length : 0,
      citations: ctx.citations ? ctx.citations.citations.length : 0,
    } : {};
    $("#tabs").innerHTML = TABS.map(([k, label]) =>
      `<button role="tab" data-tab="${k}" class="${state.tab === k ? "on" : ""}">${label}${counts[k] ? `<span class="n">${counts[k]}</span>` : ""}</button>`).join("");
  }

  function renderWorkspace() {
    renderStages();
    renderMetrics();
    renderFeed();
    const cur = current();
    renderTabs(cur);
    $("#toolbar").classList.toggle("hidden", !cur);
    const panel = $("#tabpanel");
    if (!cur) return;
    const ctx = cur.ctx;
    const renderers = { report: viewReport, plan: viewPlan, corpus: viewCorpus, synthesis: viewSynthesis, gaps: viewGaps, hypotheses: viewHypotheses, experiment: viewExperiment, citations: viewCitations, trace: viewTrace };
    panel.innerHTML = renderers[state.tab](ctx, cur);
  }

  const waiting = (what, stage) => {
    const running = state.running && current() && current().ctx.stage === stage;
    return `<div class="empty"><div>${running ? `${esc(what)} is being produced…` : `${esc(what)} is not available for this run yet.`}</div></div>`;
  };

  function viewReport(ctx, cur) {
    const r = ctx.report;
    const notices = [];
    if (ctx.input_guardrail) {
      notices.push(`<div class="notice err"><b>The input guardrail stopped this run.</b> ${esc(ctx.input_guardrail.reasoning)}${ctx.input_guardrail.suggested_rewrite ? `<br>Suggested rewrite: <i>${esc(ctx.input_guardrail.suggested_rewrite)}</i>` : ""}</div>`);
    }
    if (cur.status === "awaiting_approval") notices.push(`<div class="notice warn">This run is paused for a human decision. <a href="#history">Resume it from History</a>.</div>`);
    if (!r) return notices.join("") + waiting("The dossier", "report");
    return `${notices.join("")}<article class="prose">
      <div class="row" style="margin-bottom:8px">${confBadge(r.confidence)} <span class="muted small">confidence · ${r.word_count} words · ${(r.references || []).length} references</span></div>
      <h1>${esc(r.title)}</h1>
      <p class="muted"><b>Research question:</b> ${esc(r.research_question)}</p>
      <h2>Executive summary</h2>
      <div class="summary-box">${md(r.executive_summary, ctx)}</div>
      ${r.key_findings && r.key_findings.length ? `<h2>Key findings</h2>${list(r.key_findings, ctx)}` : ""}
      ${(r.sections || []).map((s) => `<h2>${esc(s.heading)}</h2>${md(s.body_markdown, ctx)}`).join("")}
      ${r.limitations && r.limitations.length ? `<h2>Stated limitations</h2>${list(r.limitations, ctx)}` : ""}
      ${r.references && r.references.length ? `<h2>References</h2>${md(r.references.map((x) => `- ${x}`).join("\n"))}` : ""}
    </article>`;
  }

  function viewPlan(ctx) {
    const p = ctx.plan;
    if (!p) return waiting("The research plan", "planning");
    return `<div class="prose">
      <h2 style="margin-top:0">${esc(p.research_question)}</h2>
      <div class="summary-box">${md(p.interpretation)}</div>
      <h3>Sub-questions</h3>${list(p.sub_questions)}
      <h3>Search keywords</h3><div class="row">${p.search_keywords.map((k) => `<span class="chip">${esc(k)}</span>`).join("")}</div>
      <h3>Steps</h3>
      <div class="table-wrap"><table class="data"><thead><tr><th>#</th><th>Stage</th><th>Objective</th><th>Rationale</th></tr></thead><tbody>
      ${p.steps.map((s) => `<tr><td>${s.order}</td><td>${esc(s.stage)}</td><td>${esc(s.objective)}</td><td class="muted">${esc(s.rationale)}</td></tr>`).join("")}
      </tbody></table></div>
      <div class="grid-2"><div><h3>Success criteria</h3>${list(p.success_criteria)}</div><div><h3>Out of scope</h3>${list(p.out_of_scope)}</div></div>
    </div>`;
  }

  function viewCorpus(ctx) {
    const papers = Object.values(ctx.corpus);
    const lit = ctx.literature;
    if (!papers.length) return waiting("The corpus", "literature");
    const srcKind = { arxiv: "warn", openalex: "info", crossref: "ok", user_upload: "neutral" };
    return `${lit ? `<div class="summary-box" style="margin-bottom:12px"><b>Coverage.</b> ${plain(lit.coverage_assessment)}${lit.notable_gaps.length ? `<div style="margin-top:6px"><b>Searched for but not found:</b>${list(lit.notable_gaps)}</div>` : ""}${lit.query_terms_used.length ? `<div class="row" style="margin-top:6px">${lit.query_terms_used.map((q) => `<span class="chip">${esc(q)}</span>`).join("")}</div>` : ""}</div>` : ""}
      <div class="table-wrap"><table class="data"><thead><tr><th>Paper</th><th>Year</th><th>Venue</th><th>Cites</th></tr></thead><tbody>
      ${papers.map((p) => `<tr><td><div>${badge(p.source, srcKind[p.source])} <span class="mono small">${esc(p.paper_id)}</span></div>
        <div style="margin-top:4px">${p.url && !p.url.startsWith("local:") ? `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a>` : esc(p.title)}</div>
        <div class="muted small">${esc(p.authors.slice(0, 4).join(", "))}${p.authors.length > 4 ? " et al." : ""}</div>
        ${p.abstract ? `<details class="abstract"><summary>Abstract</summary><p>${esc(p.abstract.slice(0, 2400))}${p.abstract.length > 2400 ? "…" : ""}</p></details>` : ""}</td>
        <td>${p.year || "n.d."}</td><td class="small">${esc(p.venue || "—")}</td><td>${p.citation_count}</td></tr>`).join("")}
      </tbody></table></div>`;
  }

  function viewSynthesis(ctx) {
    const s = ctx.synthesis;
    if (!s) return waiting("The synthesis", "synthesis");
    return `<div class="prose" style="max-width:none">
      <div class="summary-box">${md(s.summary, ctx)}</div>
      <h3>Key claims</h3>
      ${s.key_claims.map((c) => `<div class="item"><div class="meta">${confBadge(c.confidence)} ${c.contested ? badge("contested", "fail") : ""}</div><div>${plain(c.statement, ctx)}</div><div class="row small" style="margin-top:6px">${c.supporting_paper_ids.map((id) => citeLink(id, ctx) || `<span class="mono">${esc(id)}</span>`).join(" ")}</div></div>`).join("") || `<p class="muted">No claims.</p>`}
      <div class="grid-2"><div><h3>Consensus</h3>${list(s.consensus_points, ctx)}</div><div><h3>Disagreements</h3>${list(s.disagreements, ctx)}</div></div>
      <h3>Gaps flagged during synthesis</h3>
      ${s.gaps.map((g) => `<div class="item"><h4>${plain(g.gap, ctx)}</h4><div class="small muted">${plain(g.why_it_matters, ctx)}</div><div class="row small" style="margin-top:6px">${g.evidence_paper_ids.map((id) => citeLink(id, ctx) || esc(id)).join(" ")}</div></div>`).join("")}
    </div>`;
  }

  const scoreBar = (label, v) => `<div class="score"><div class="lab"><span>${label}</span><b>${v}/10</b></div><div class="bar"><span style="width:${Math.max(0, Math.min(10, v)) * 10}%"></span></div></div>`;

  function viewGaps(ctx) {
    const g = ctx.gap_analysis;
    if (!g) return waiting("The gap analysis", "gap_analysis");
    const sev = { foundational: "fail", significant: "warn", incremental: "info" };
    return `<div class="prose" style="max-width:none">
      <div class="summary-box"><b>Priority: ${esc(g.priority_gap_id)}.</b> ${plain(g.prioritisation_rationale, ctx)}</div>
      ${g.gaps.map((x) => `<div class="item ${x.gap_id === g.priority_gap_id ? "recommended" : ""}">
        <div class="meta"><b class="mono">${esc(x.gap_id)}</b> ${badge(x.severity, sev[x.severity])} ${badge(x.gap_type, "neutral")}</div>
        <h4>${plain(x.title, ctx)}</h4><div>${plain(x.description, ctx)}</div>
        <dl class="kv small" style="margin-top:10px"><dt>Why unaddressed</dt><dd>${plain(x.why_unaddressed, ctx)}</dd><dt>Potential impact</dt><dd>${plain(x.potential_impact, ctx)}</dd><dt>Evidence</dt><dd>${x.evidence_paper_ids.map((id) => citeLink(id, ctx) || esc(id)).join(" ") || "—"}</dd></dl>
        <div style="max-width:260px;margin-top:8px">${scoreBar("Tractability", x.tractability)}</div></div>`).join("")}
      <div class="grid-2"><div><h3>Saturated areas</h3>${list(g.saturated_areas, ctx)}</div><div><h3>Corpus limitations</h3><p>${plain(g.corpus_limitations, ctx)}</p></div></div>
    </div>`;
  }

  function viewHypotheses(ctx) {
    const h = ctx.hypotheses;
    if (!h) return waiting("The hypotheses", "hypothesis");
    return `<div class="prose" style="max-width:none">
      <div class="summary-box"><b>Recommended: ${esc(h.recommended_hypothesis_id)}.</b> ${plain(h.selection_rationale, ctx)}</div>
      ${h.hypotheses.map((x) => `<div class="item ${x.hypothesis_id === h.recommended_hypothesis_id ? "recommended" : ""}">
        <div class="meta"><b class="mono">${esc(x.hypothesis_id)}</b>${x.addresses_gap ? ` <span>addresses ${esc(x.addresses_gap)}</span>` : ""}${x.hypothesis_id === h.recommended_hypothesis_id ? ` ${badge("recommended", "ok")}` : ""}</div>
        <h4>${plain(x.statement, ctx)}</h4>
        <div class="small">${plain(x.rationale, ctx)}</div>
        <dl class="kv small" style="margin-top:10px"><dt>Null hypothesis</dt><dd>${plain(x.null_hypothesis, ctx)}</dd><dt>Independent vars</dt><dd>${esc(x.independent_variables.join(", "))}</dd><dt>Dependent vars</dt><dd>${esc(x.dependent_variables.join(", "))}</dd><dt>Grounded in</dt><dd>${x.grounded_in_paper_ids.map((id) => citeLink(id, ctx) || esc(id)).join(" ")}</dd></dl>
        <div class="scores">${scoreBar("Novelty", x.novelty_score)}${scoreBar("Feasibility", x.feasibility_score)}${scoreBar("Impact", x.impact_score)}</div></div>`).join("")}
    </div>`;
  }

  function viewExperiment(ctx) {
    const e = ctx.experiment;
    if (!e) return waiting("The experiment design", "experiment");
    const pa = e.power_analysis;
    return `<div class="prose" style="max-width:none">
      <div class="meta muted small">Tests <b class="mono">${esc(e.hypothesis_id)}</b></div>
      <h2 style="margin-top:4px">${plain(e.title, ctx)}</h2>
      <div class="grid-3" style="margin:12px 0">
        <div class="metric"><div class="v">${pa.required_sample_size}</div><div class="k">n per group</div></div>
        <div class="metric"><div class="v">${pa.required_sample_size * 2}</div><div class="k">total n</div></div>
        <div class="metric"><div class="v">d=${pa.effect_size} · α=${pa.alpha} · ${Math.round(pa.power * 100)}%</div><div class="k">effect · alpha · power</div></div>
      </div>
      <p class="small muted"><b>Effect-size justification:</b> ${plain(pa.justification, ctx)}</p>
      <h3>Methodology</h3>${md(e.methodology, ctx)}
      <div class="grid-2"><div><h3>Independent variables</h3>${list(e.independent_variables)}</div><div><h3>Dependent variables</h3>${list(e.dependent_variables)}</div></div>
      <h3>Controls</h3>${list(e.controls, ctx)}
      <h3>Procedure</h3><ol class="list-tight">${e.procedure_steps.map((s) => `<li>${plain(s, ctx)}</li>`).join("")}</ol>
      <h3>Datasets and materials</h3>${list(e.datasets_or_materials, ctx)}
      <h3>Statistical tests (pre-specified)</h3>${list(e.statistical_tests, ctx)}
      <div class="grid-2"><div><h3>Threats to validity</h3>${list(e.threats_to_validity, ctx)}</div><div><h3>Ethical considerations</h3>${list(e.ethical_considerations, ctx)}</div></div>
      <dl class="kv" style="margin-top:12px"><dt>Estimated duration</dt><dd>${esc(e.estimated_duration)}</dd><dt>Expected outcomes</dt><dd>${plain(e.expected_outcomes, ctx)}</dd></dl>
    </div>`;
  }

  function viewCitations(ctx) {
    const c = ctx.citations;
    if (!c) return waiting("The bibliography", "citations");
    return `<div class="prose" style="max-width:none">
      <div class="row" style="margin-bottom:10px">${badge(c.style, "info")} ${c.unverified_ids.length ? badge(`${c.unverified_ids.length} unverified`, "fail") : badge("all verified", "ok")}</div>
      <div class="summary-box">${plain(c.integrity_note, ctx)}</div>
      ${c.unverified_ids.length ? `<div class="notice err" style="margin-top:10px">Excluded (not in corpus): <span class="mono">${esc(c.unverified_ids.join(", "))}</span></div>` : ""}
      <h3>Reference list</h3>${md(c.reference_list_markdown)}
      <h3>In-text forms</h3>
      <div class="table-wrap"><table class="data"><thead><tr><th>Corpus id</th><th>In-text</th><th>BibTeX key</th></tr></thead><tbody>
      ${c.citations.map((x) => `<tr><td>${citeLink(x.paper_id, ctx) || esc(x.paper_id)}</td><td>${esc(x.in_text)}</td><td class="mono small">${esc(x.bibtex_key)}</td></tr>`).join("")}</tbody></table></div>
      <h3>BibTeX</h3><pre class="code">${esc(c.bibtex_file)}</pre>
    </div>`;
  }

  function viewTrace(ctx, cur) {
    const m = cur.manifest;
    return `<div class="prose" style="max-width:none">
      <dl class="kv"><dt>Run</dt><dd class="mono">${esc(m.run_id)}</dd><dt>Mode</dt><dd>${esc(m.mode)} · ${esc(ctx.source_mode === "documents" ? "document-grounded" : "web literature")}</dd><dt>Status</dt><dd>${statusBadge(cur.status)}</dd><dt>Started</dt><dd>${esc(fmtDate(ctx.started_at))}</dd><dt>Spend</dt><dd>${Util.fmtUsd(ctx.costs.usd)} · ${ctx.costs.requests} model requests · ${(ctx.costs.input_tokens + ctx.costs.output_tokens).toLocaleString()} tokens · ${ctx.costs.tool_calls} tool calls · ${ctx.costs.api_calls} external API calls</dd></dl>
      <h3>Handoff chain</h3><div class="chain">${(m.handoffs || []).map((a) => `<span class="chip">${esc(a)}</span>`).join('<span class="arrow">→</span>') || "<span class='muted'>—</span>"}</div>
      <h3>Stages</h3>
      <div class="table-wrap"><table class="data"><thead><tr><th>Stage</th><th>Agent</th><th>Status</th><th>Time</th><th>Detail</th></tr></thead><tbody>
      ${(m.stages || []).map((s) => `<tr><td>${esc(s.stage)}</td><td>${esc(s.agent)}</td><td>${s.ok ? badge("ok", "ok") : badge("failed", "fail")}</td><td>${(s.duration_s || 0).toFixed(1)}s</td><td class="small">${esc(s.detail)}</td></tr>`).join("")}</tbody></table></div>
      <h3>Tool usage</h3><div class="row">${Object.entries(m.tool_usage || {}).map(([k, v]) => `<span class="chip"><span class="mono">${esc(k)}</span> × ${v}</span>`).join("") || "<span class='muted'>—</span>"}</div>
      <h3>Human approvals</h3>
      ${(ctx.approvals || []).length ? `<div class="table-wrap"><table class="data"><thead><tr><th>Tool</th><th>Decision</th><th>When</th><th>Note</th></tr></thead><tbody>${ctx.approvals.map((a) => `<tr><td class="mono">${esc(a.tool_name)}</td><td>${badge(a.decision, a.decision === "approved" ? "ok" : "fail")}</td><td class="small">${esc(fmtDate(a.decided_at))}</td><td class="small">${esc(a.note)}</td></tr>`).join("")}</tbody></table></div>` : "<p class='muted small'>No gated tool calls yet.</p>"}
      <h3>Overclaim flags (reported, not fatal)</h3>${ctx.overclaim_flags && ctx.overclaim_flags.length ? list(ctx.overclaim_flags.map((f) => `${f.agent}: ${f.statement}`), ctx) : "<p class='muted small'>None.</p>"}
      <h3>Recorded issues</h3>${list(ctx.errors.map((e) => e.replace(/^\S+\s/, "")))}
      <h3>Event log</h3><pre class="code">${esc((cur.trace || []).slice(-300).map((e) => JSON.stringify(e)).join("\n"))}</pre>
    </div>`;
  }

  function initWorkspace() {
    $("#tabs").addEventListener("click", (e) => {
      const b = e.target.closest("[data-tab]");
      if (b) { state.tab = b.dataset.tab; state.followTabs = false; renderWorkspace(); }
    });
    $("#toolbar").addEventListener("click", (e) => {
      const b = e.target.closest("[data-export]");
      if (!b) return;
      const cur = current();
      if (!cur) return;
      const ctx = cur.ctx;
      const base = ctx.run_id;
      switch (b.dataset.export) {
        case "md":
          if (!ctx.report) return toast("No report yet.");
          return Util.download(`${A.Docs.slug(ctx.report.title)}.md`, A.reportMarkdown(ctx.report, ctx), "text/markdown;charset=utf-8");
        case "print":
          if (!ctx.report) return toast("No report yet.");
          state.tab = "report"; renderWorkspace(); return setTimeout(() => window.print(), 100);
        case "bib":
          if (!ctx.citations) return toast("No bibliography yet.");
          return Util.download(`${base}.bib`, ctx.citations.bibtex_file, "application/x-bibtex");
        case "manifest":
          return Util.download(`${base}_manifest.json`, JSON.stringify(cur.manifest, null, 2), "application/json");
        case "trace":
          return Util.download(`${base}.jsonl`, (cur.trace || []).map((x) => JSON.stringify(x)).join("\n"), "application/x-ndjson");
        case "context":
          return Util.download(`${base}_context.json`, JSON.stringify(ctx.toDict(), null, 2), "application/json");
        default:
      }
    });
    Log.listeners.add(() => { if (!state.view) queueRender(); });
  }

  // ---------------------------------------------------------------------
  // History
  // ---------------------------------------------------------------------

  async function renderHistory() {
    const runs = (await DB.all("runs")).sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
    const cps = new Set((await DB.all("checkpoints")).map((c) => c.run_id));
    const el = $("#history-list");
    el.innerHTML = runs.length ? runs.map((r) => {
      const s = (r.manifest && r.manifest.summary) || {};
      return `<div class="run-row"><div><div class="q">${esc(r.question)}</div>
        <div class="meta">${statusBadge(r.status)} ${badge(r.mode, r.mode === "live" ? "ok" : "warn")} <span>${esc(fmtDate(r.updated_at))}</span><span>${s.corpus_size || 0} papers</span><span>${Util.fmtUsd(s.estimated_usd || 0)}</span><span class="mono">${esc(r.run_id)}</span></div></div>
        <div class="row">${cps.has(r.run_id) ? `<button class="btn sm primary" data-resume="${esc(r.run_id)}">Resume</button>` : ""}<button class="btn sm" data-open="${esc(r.run_id)}">Open</button><button class="btn sm" data-export-run="${esc(r.run_id)}">Export</button><button class="btn sm danger" data-delete-run="${esc(r.run_id)}">Delete</button></div></div>`;
    }).join("") : `<div class="empty">No runs yet. Your studies will be listed here.</div>`;

    const arts = (await DB.all("artifacts")).sort((a, b) => (b.meta && b.meta.created_at || "").localeCompare(a.meta && a.meta.created_at || ""));
    $("#artifact-list").innerHTML = arts.length ? arts.map((a) => `<div class="run-row"><div><div class="q mono">${esc(a.filename)}</div>
      <div class="meta">${badge(a.artifact_type, "info")}<span>${esc(fmtDate(a.meta && a.meta.created_at))}</span><span>${esc((a.meta && a.meta.research_question) || "")}</span></div></div>
      <div class="row"><button class="btn sm" data-art-dl="${a.id}">Download</button><button class="btn sm" data-art-meta="${a.id}">Provenance</button><button class="btn sm danger" data-art-del="${a.id}">Delete</button></div></div>`).join("")
      : `<div class="empty">No saved artefacts yet. Approve the Scientific Writer's save to keep a dossier here.</div>`;
  }

  async function openRun(runId) {
    const r = await DB.get("runs", runId);
    if (!r) return toast("Run not found.");
    state.view = { ctx: A.ResearchContext.fromDict(r.ctx), manifest: r.manifest, trace: r.trace || [], stages: (r.manifest && r.manifest.stages) || [], status: r.status, mode: r.mode };
    state.tab = r.ctx.report ? "report" : "trace";
    location.hash = "#study";
    renderWorkspace();
  }

  async function resumeRun(runId) {
    if (state.running) return toast("Another run is in progress.");
    const cp = await DB.get("checkpoints", runId);
    if (!cp) return toast("No checkpoint found for this run.");
    let decision;
    try {
      state.pipeline = null;
      state.view = { ctx: A.ResearchContext.fromDict(cp.ctx), manifest: { run_id: cp.run_id, stages: cp.stages, handoffs: [], tool_usage: {} }, trace: cp.trace, stages: cp.stages, status: "awaiting_approval", mode: cp.live ? "live" : "offline" };
      location.hash = "#study";
      renderWorkspace();
      const item = cp.item || { name: cp.tool, arguments: "{}", agent: "" };
      if (!cp.item && cp.state && cp.state.pending && cp.state.pending[0]) item.arguments = cp.state.pending[0].function.arguments;
      state.pipeline = { ctx: state.view.ctx }; // so the approval preview can link citations
      decision = await approvalHandler(item);
    } catch (err) {
      state.pipeline = null;
      if (err instanceof A.errors.ApprovalDeferred) return toast("Still deferred. The checkpoint is kept.");
      throw err;
    }
    state.view = null;
    state.pipeline = null;
    state.running = true;
    state.followTabs = false;
    state.tab = "report";
    $("#feed").innerHTML = "";
    setRunning(true);
    try {
      await A.ResearchPipeline.resume(runId, decision, { approvalHandler, onEvent: onPipelineEvent });
      toast("Run resumed and finished.");
    } catch (err) {
      console.error(err);
      toast(err.message, 6000);
    } finally {
      state.running = false;
      setRunning(false);
      renderWorkspace();
    }
  }

  function initHistory() {
    $("#view-history").addEventListener("click", async (e) => {
      const t = e.target.closest("button");
      if (!t) return;
      if (t.dataset.open) return openRun(t.dataset.open);
      if (t.dataset.resume) return resumeRun(t.dataset.resume);
      if (t.dataset.exportRun) {
        const r = await DB.get("runs", t.dataset.exportRun);
        return Util.download(`${r.run_id}.json`, JSON.stringify(r, null, 2), "application/json");
      }
      if (t.dataset.deleteRun) {
        if (!confirm("Delete this run and any pending checkpoint? This cannot be undone.")) return;
        await DB.delete("runs", t.dataset.deleteRun);
        await DB.delete("checkpoints", t.dataset.deleteRun);
        if (state.view && state.view.ctx.run_id === t.dataset.deleteRun) state.view = null;
        return renderHistory();
      }
      const art = async (id) => DB.get("artifacts", Number(id));
      if (t.dataset.artDl) { const a = await art(t.dataset.artDl); return Util.download(a.filename, a.content, "text/markdown;charset=utf-8"); }
      if (t.dataset.artMeta) { const a = await art(t.dataset.artMeta); return Util.download(`${a.filename}.meta.json`, JSON.stringify(a.meta, null, 2), "application/json"); }
      if (t.dataset.artDel) {
        if (!confirm("Delete this saved artefact? This cannot be undone.")) return;
        await DB.delete("artifacts", Number(t.dataset.artDel));
        return renderHistory();
      }
      if (t.id === "import-run-btn") $("#import-run-input").click();
    });
    $("#import-run-input").addEventListener("change", async (e) => {
      const f = e.target.files[0];
      e.target.value = "";
      if (!f) return;
      try {
        const r = JSON.parse(await f.text());
        if (!r.run_id || !r.ctx) throw new Error("not an Athena run export");
        await DB.put("runs", r);
        toast("Run imported.");
        renderHistory();
      } catch (err) { toast(`Import failed: ${err.message}`); }
    });
  }

  // ---------------------------------------------------------------------
  // Memory
  // ---------------------------------------------------------------------

  async function renderMemory(records, label) {
    const stats = await LongTermMemory.stats();
    $("#mem-stats").innerHTML = Object.entries(stats).map(([k, v]) => `<div class="metric"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`).join("");
    const rows = records || (await LongTermMemory.recent(50));
    const kindBadge = { finding: "ok", question: "info", preference: "warn", critique: "fail" };
    $("#mem-list").innerHTML = (label ? `<div class="card-pad small muted" style="border-bottom:1px solid var(--line)">${esc(label)}</div>` : "") + (rows.length ? rows.map((r) => `<div class="mem">
      <div>${badge(r.kind, kindBadge[r.kind])}</div>
      <div><div><b>${esc(r.topic)}</b>${r.score !== undefined ? ` <span class="muted small">relevance ${r.score.toFixed(2)}</span>` : ""}</div><div>${esc(r.content)}</div><div class="muted small">${esc(fmtDate(r.created_at))} · ${esc(r.run_id)}</div></div>
      <div><button class="btn sm danger" data-mem-del="${r.id}" aria-label="Delete memory">✕</button></div></div>`).join("")
      : `<div class="empty">Nothing remembered yet. The Scientific Writer stores the durable insight of every completed study here.</div>`);
  }

  function initMemory() {
    $("#mem-search").addEventListener("click", async () => {
      const q = $("#mem-query").value.trim();
      if (!q) return renderMemory();
      renderMemory(await LongTermMemory.recall(q, { k: 10 }), `Recall: ${q}`);
    });
    $("#mem-query").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#mem-search").click(); });
    $("#mem-recent").addEventListener("click", () => renderMemory());
    $("#mem-add").addEventListener("click", async () => {
      const content = $("#mem-content").value.trim();
      if (!content) return toast("Write the memory first.");
      await LongTermMemory.remember({ session_id: "manual", run_id: "manual", kind: $("#mem-kind").value, topic: $("#mem-topic").value.trim() || "general", content });
      $("#mem-content").value = "";
      renderMemory();
    });
    $("#mem-list").addEventListener("click", async (e) => {
      const b = e.target.closest("[data-mem-del]");
      if (!b) return;
      await LongTermMemory.remove(Number(b.dataset.memDel));
      renderMemory();
    });
    $("#mem-export").addEventListener("click", async () => Util.download("athena_long_term_memory.json", await LongTermMemory.exportJson(), "application/json"));
    $("#mem-import-btn").addEventListener("click", () => $("#mem-import").click());
    $("#mem-import").addEventListener("change", async (e) => {
      const f = e.target.files[0];
      e.target.value = "";
      if (!f) return;
      try { const n = await LongTermMemory.importJson(await f.text()); toast(`Imported ${n} memories.`); renderMemory(); }
      catch (err) { toast(`Import failed: ${err.message}`); }
    });
    $("#mem-clear").addEventListener("click", async () => {
      if (!confirm("Permanently delete every long-term memory in this browser? This cannot be undone.")) return;
      await LongTermMemory.clear();
      renderMemory();
    });
  }

  // ---------------------------------------------------------------------
  // Agents
  // ---------------------------------------------------------------------

  function renderAgents() {
    const team = A.buildTeam();
    $("#agents-table").innerHTML = `<thead><tr><th>Agent</th><th>Output</th><th>Model</th><th>Temp.</th><th>Tools</th><th>Guardrails</th></tr></thead><tbody>${team.describe().map((a) => `<tr>
      <td><b>${esc(a.name)}</b><div class="muted small">${esc(a.handoff_description)}</div></td>
      <td class="mono small">${esc(a.output_type)}</td><td class="mono small">${esc(a.model)}</td><td>${a.temperature}</td>
      <td><div class="row">${a.tools.map((t) => `<span class="chip mono">${esc(t)}</span>`).join("")}</div></td>
      <td class="small">${a.input_guardrails.map((g) => `input: ${esc(g)}`).concat(a.output_guardrails.map((g) => `output: ${esc(g)}`)).join("<br>") || "—"}</td></tr>`).join("")}</tbody>`;
    const groupOf = {};
    for (const [g, names] of Object.entries(A.ToolGroups)) for (const n of names) (groupOf[n] = groupOf[n] || []).push(g.replace("_TOOLS", "").toLowerCase());
    $("#tools-table").innerHTML = `<thead><tr><th>Tool</th><th>Group</th><th>Approval</th><th>Description</th></tr></thead><tbody>${Object.values(A.Tools).map((t) => `<tr>
      <td class="mono small">${esc(t.name)}</td><td class="small">${esc((groupOf[t.name] || []).join(", "))}</td>
      <td>${t.name === "save_research_artifact" ? badge("always", "warn") : t.name === "commission_deep_sweep" ? badge("> 30 results", "warn") : badge("none", "neutral")}</td>
      <td class="small">${esc(t.description)}${A.DOCUMENT_MODE_BLOCKED_TOOLS.has(t.name) ? ` <span class="muted">(disabled in document mode)</span>` : ""}</td></tr>`).join("")}</tbody>`;
  }

  // ---------------------------------------------------------------------
  // Settings
  // ---------------------------------------------------------------------

  function openSettings() {
    const s = Settings.get();
    $("#set-key").value = "";
    $("#set-key").placeholder = Settings.apiKey() ? `Saved key …${Settings.apiKey().slice(-4)} (leave blank to keep)` : "sk-...";
    $("#set-remember").checked = !!s.remember_key;
    $("#set-mock").checked = !!s.mock_mode;
    $("#set-fast").value = s.fast_model;
    $("#set-reasoning").value = s.reasoning_model;
    $("#set-vision").value = s.vision_model;
    $("#set-transcription").value = s.transcription_model;
    $("#set-usd").value = s.max_run_usd;
    $("#set-turns").value = s.max_turns;
    $("#set-per-source").value = s.max_papers_per_source;
    $("#set-corpus").value = s.max_corpus_papers;
    $("#set-timeout").value = s.request_timeout_seconds;
    $("#set-retries").value = s.max_tool_retries;
    $("#set-style").value = s.citation_style;
    $("#set-email").value = s.contact_email;
    $("#set-approval").checked = !!s.require_human_approval;
    $("#settings-dialog").showModal();
  }

  function initSettings() {
    $("#settings-btn").addEventListener("click", openSettings);
    $("#mode-pill").addEventListener("click", openSettings);
    const num = (id, fallback, min, max) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : fallback; };
    $("#settings-form").addEventListener("submit", (e) => {
      if (e.submitter && e.submitter.value !== "save") return;
      const d = Settings.defaults;
      Settings.save({
        mock_mode: $("#set-mock").checked,
        fast_model: $("#set-fast").value.trim() || d.fast_model,
        reasoning_model: $("#set-reasoning").value.trim() || d.reasoning_model,
        vision_model: $("#set-vision").value.trim() || d.vision_model,
        transcription_model: $("#set-transcription").value.trim() || d.transcription_model,
        max_run_usd: num("#set-usd", d.max_run_usd, 0, 1000),
        max_turns: Math.round(num("#set-turns", d.max_turns, 2, 40)),
        max_papers_per_source: Math.round(num("#set-per-source", d.max_papers_per_source, 1, 15)),
        max_corpus_papers: Math.round(num("#set-corpus", d.max_corpus_papers, 3, 60)),
        request_timeout_seconds: num("#set-timeout", d.request_timeout_seconds, 5, 120),
        max_tool_retries: Math.round(num("#set-retries", d.max_tool_retries, 1, 6)),
        citation_style: $("#set-style").value,
        contact_email: $("#set-email").value.trim(),
        require_human_approval: $("#set-approval").checked,
      });
      const key = $("#set-key").value.trim();
      const remember = $("#set-remember").checked;
      if (key) Settings.setApiKey(key, remember);
      else if (Settings.apiKey()) Settings.setApiKey(Settings.apiKey(), remember);
      $("#approval-toggle").checked = Settings.get().require_human_approval;
      updateModePill();
      toast("Settings saved.");
    });
    $("#set-reset").addEventListener("click", () => {
      const keep = Settings.apiKey();
      Settings.reset();
      if (keep) Settings.setApiKey(keep, false);
      $("#settings-dialog").close();
      openSettings();
      updateModePill();
    });
    $("#set-forget").addEventListener("click", () => {
      Settings.setApiKey(null, false);
      $("#set-key").placeholder = "sk-...";
      updateModePill();
      toast("API key removed from this browser.");
    });
  }

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------

  async function boot() {
    Settings.load();
    await DB.open();
    initTheme();
    initComposer();
    initWorkspace();
    initHistory();
    initMemory();
    initSettings();
    updateModePill();
    renderFileChips();
    renderWorkspace();
    window.addEventListener("hashchange", route);
    route();
    if (!DB.persistent) toast("This browser blocks IndexedDB; history and memory will last only for this tab.", 6000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})((window.Athena = window.Athena || {}));
