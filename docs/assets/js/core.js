/* Athena - configuration, storage, logging, errors, HTTP and the run context.
 *
 * Browser port of config.py, errors.py, logging_setup.py, _http.py and
 * context.py. Settings live in localStorage; durable state (long-term memory,
 * run history, checkpoints, saved artefacts) lives in IndexedDB, so it survives
 * a page reload exactly the way the Python version's SQLite files survive a
 * process restart.
 */
(function (A) {
  "use strict";

  // ---------------------------------------------------------------------
  // Settings (config.py)
  // ---------------------------------------------------------------------

  const DEFAULTS = {
    mock_mode: false,
    fast_model: "gpt-4.1-mini",
    reasoning_model: "gpt-4.1-mini",
    vision_model: "gpt-4.1-mini",
    transcription_model: "whisper-1",
    max_turns: 12,
    max_papers_per_source: 5,
    max_corpus_papers: 12,
    max_run_usd: 1.0,
    request_timeout_seconds: 30,
    max_tool_retries: 3,
    require_human_approval: true,
    contact_email: "",
    citation_style: "APA",
    remember_key: false,
  };
  const SETTINGS_KEY = "athena.settings.v1";
  const API_KEY_KEY = "athena.openai_key";

  function safeGet(storage, key) {
    try { return storage.getItem(key); } catch (e) { return null; }
  }
  function safeSet(storage, key, value) {
    try {
      if (value === null || value === undefined || value === "") storage.removeItem(key);
      else storage.setItem(key, value);
    } catch (e) { /* storage blocked: settings stay in memory */ }
  }

  const Settings = {
    _values: null,
    _key: null,
    load() {
      let stored = {};
      try { stored = JSON.parse(safeGet(localStorage, SETTINGS_KEY) || "{}"); } catch (e) { stored = {}; }
      this._values = Object.assign({}, DEFAULTS, stored);
      this._key = safeGet(localStorage, API_KEY_KEY) || safeGet(sessionStorage, API_KEY_KEY) || null;
      return this._values;
    },
    get() { return this._values || this.load(); },
    save(patch) {
      Object.assign(this.get(), patch || {});
      safeSet(localStorage, SETTINGS_KEY, JSON.stringify(this._values));
      return this._values;
    },
    reset() {
      this._values = Object.assign({}, DEFAULTS);
      safeSet(localStorage, SETTINGS_KEY, JSON.stringify(this._values));
      return this._values;
    },
    apiKey() { if (!this._values) this.load(); return this._key; },
    setApiKey(key, remember) {
      this._key = (key || "").trim() || null;
      safeSet(localStorage, API_KEY_KEY, null);
      safeSet(sessionStorage, API_KEY_KEY, null);
      if (this._key) safeSet(remember ? localStorage : sessionStorage, API_KEY_KEY, this._key);
      this.save({ remember_key: !!remember });
    },
    /** True when we should actually hit the OpenAI API. */
    get live() { return !this.get().mock_mode && !!this.apiKey(); },
    defaults: DEFAULTS,
  };

  // ---------------------------------------------------------------------
  // Errors (errors.py)
  // ---------------------------------------------------------------------

  class AthenaError extends Error {
    constructor(message) { super(message); this.name = this.constructor.name; }
  }
  class ExternalAPIError extends AthenaError {
    constructor(source, detail) { super(`${source}: ${detail}`); this.source = source; }
  }
  class ToolExecutionError extends AthenaError {
    constructor(tool, detail) { super(`${tool}: ${detail}`); this.tool = tool; }
  }
  class DocumentParseError extends ToolExecutionError {}
  class UnsafeCodeError extends ToolExecutionError {}
  class ApprovalDeferred extends AthenaError {
    constructor(tool, detail, state) { super(`${tool}: ${detail}`); this.tool = tool; this.state = state || null; }
  }
  class InputGuardrailTripwireTriggered extends AthenaError {
    constructor(info) { super(`input guardrail tripped: ${info.reasoning || ""}`); this.info = info; }
  }
  class OutputGuardrailTripwireTriggered extends AthenaError {
    constructor(info) { super(`fabricated citations: ${JSON.stringify(info.fabricated || [])}`); this.info = info; }
  }
  class MaxTurnsExceeded extends AthenaError {}
  class ModelError extends AthenaError {
    constructor(message, status) { super(message); this.status = status; }
  }

  // ---------------------------------------------------------------------
  // IndexedDB wrapper (sessions.db / long_term_memory.db / artifacts dir)
  // ---------------------------------------------------------------------

  const DB_NAME = "athena";
  const DB_VERSION = 1;
  const STORES = {
    memories: { keyPath: "id", autoIncrement: true },
    runs: { keyPath: "run_id" },
    checkpoints: { keyPath: "run_id" },
    artifacts: { keyPath: "id", autoIncrement: true },
  };

  const DB = {
    _db: null,
    _fallback: null, // in-memory store when IndexedDB is unavailable (private mode)
    async open() {
      if (this._db || this._fallback) return;
      if (typeof indexedDB === "undefined") { this._initFallback(); return; }
      try {
        this._db = await new Promise((resolve, reject) => {
          const request = indexedDB.open(DB_NAME, DB_VERSION);
          request.onupgradeneeded = () => {
            const db = request.result;
            for (const [name, opts] of Object.entries(STORES)) {
              if (!db.objectStoreNames.contains(name)) db.createObjectStore(name, opts);
            }
          };
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
        });
      } catch (e) {
        this._initFallback();
      }
    },
    _initFallback() {
      this._fallback = {};
      for (const name of Object.keys(STORES)) this._fallback[name] = { rows: new Map(), seq: 0 };
    },
    async _tx(store, mode, fn) {
      await this.open();
      if (this._fallback) return fn(null);
      return new Promise((resolve, reject) => {
        const tx = this._db.transaction(store, mode);
        const os = tx.objectStore(store);
        let result;
        Promise.resolve(fn(os)).then((r) => { result = r; });
        tx.oncomplete = () => resolve(result);
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      });
    },
    _req(request) {
      return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    },
    async put(store, value) {
      await this.open();
      if (this._fallback) {
        const fb = this._fallback[store];
        const keyPath = STORES[store].keyPath;
        if (STORES[store].autoIncrement && value[keyPath] == null) value[keyPath] = ++fb.seq;
        fb.rows.set(value[keyPath], JSON.parse(JSON.stringify(value)));
        return value[keyPath];
      }
      let key;
      await this._tx(store, "readwrite", async (os) => { key = await this._req(os.put(value)); });
      return key;
    },
    async get(store, key) {
      await this.open();
      if (this._fallback) {
        const row = this._fallback[store].rows.get(key);
        return row ? JSON.parse(JSON.stringify(row)) : undefined;
      }
      let value;
      await this._tx(store, "readonly", async (os) => { value = await this._req(os.get(key)); });
      return value;
    },
    async all(store) {
      await this.open();
      if (this._fallback) return [...this._fallback[store].rows.values()].map((r) => JSON.parse(JSON.stringify(r)));
      let values;
      await this._tx(store, "readonly", async (os) => { values = await this._req(os.getAll()); });
      return values || [];
    },
    async delete(store, key) {
      await this.open();
      if (this._fallback) { this._fallback[store].rows.delete(key); return; }
      await this._tx(store, "readwrite", async (os) => { await this._req(os.delete(key)); });
    },
    async clear(store) {
      await this.open();
      if (this._fallback) { this._fallback[store].rows.clear(); return; }
      await this._tx(store, "readwrite", async (os) => { await this._req(os.clear()); });
    },
    get persistent() { return !this._fallback; },
  };

  // ---------------------------------------------------------------------
  // Logging (logging_setup.py) - structured JSON events per run
  // ---------------------------------------------------------------------

  const Log = {
    events: [],
    listeners: new Set(),
    runId: null,
    configure(runId) { this.runId = runId; this.events = []; },
    event(logger, message, fields) {
      const entry = Object.assign(
        { ts: new Date().toISOString(), level: "INFO", logger: `athena.${logger}`, run_id: this.runId, message },
        fields || {}
      );
      this.events.push(entry);
      for (const fn of this.listeners) { try { fn(entry); } catch (e) { /* UI listener errors never break a run */ } }
      return entry;
    },
    error(logger, message, fields) {
      return this.event(logger, message, Object.assign({ level: "ERROR" }, fields || {}));
    },
    jsonl() { return this.events.map((e) => JSON.stringify(e)).join("\n"); },
  };

  function newRunId() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const stamp = `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
    const rand = Math.random().toString(16).slice(2, 8);
    return `run_${stamp}_${rand}`;
  }

  // ---------------------------------------------------------------------
  // HTTP with timeout and retries (_http.py)
  // ---------------------------------------------------------------------

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function withParams(url, params) {
    if (!params) return url;
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    return `${url}?${qs.toString()}`;
  }

  async function fetchWithRetry(url, { params, as = "json", init } = {}) {
    const settings = Settings.get();
    const target = withParams(url, params);
    const attempts = Math.max(1, settings.max_tool_retries || 1);
    let lastError;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), (settings.request_timeout_seconds || 30) * 1000);
      try {
        const response = await fetch(target, Object.assign({ signal: controller.signal }, init || {}));
        clearTimeout(timer);
        if (response.status === 429 || response.status >= 500) {
          lastError = new ExternalAPIError(new URL(url).host, `HTTP ${response.status}`);
          await sleep(Math.min(8000, 600 * 2 ** (attempt - 1)));
          continue;
        }
        if (!response.ok) throw new ExternalAPIError(new URL(url).host, `HTTP ${response.status}`);
        return as === "text" ? await response.text() : await response.json();
      } catch (err) {
        clearTimeout(timer);
        lastError = err.name === "AbortError" ? new ExternalAPIError(new URL(url).host, "request timed out") : err;
        if (err instanceof ExternalAPIError && !/HTTP 5|HTTP 429|timed out/.test(err.message)) throw err;
        if (attempt < attempts) await sleep(Math.min(8000, 600 * 2 ** (attempt - 1)));
      }
    }
    if (lastError instanceof TypeError) {
      throw new ExternalAPIError(new URL(url).host, "network or CORS failure");
    }
    throw lastError || new ExternalAPIError(url, "request failed");
  }

  // ---------------------------------------------------------------------
  // Cost ledger and run context (context.py)
  // ---------------------------------------------------------------------

  const MODEL_RATES = {
    "gpt-4o-mini": [0.15, 0.6],
    "gpt-4o": [2.5, 10.0],
    "gpt-4.1-mini": [0.4, 1.6],
    "gpt-4.1-nano": [0.1, 0.4],
    "gpt-4.1": [2.0, 8.0],
  };

  /** Per-million-token (input, output) USD rates; unknown models price high. */
  function modelRates(model) {
    let best = "";
    for (const prefix of Object.keys(MODEL_RATES)) {
      if ((model || "").startsWith(prefix) && prefix.length > best.length) best = prefix;
    }
    return MODEL_RATES[best] || [2.5, 10.0];
  }

  class CostLedger {
    constructor(data) {
      Object.assign(this, { requests: 0, input_tokens: 0, output_tokens: 0, tool_calls: 0, api_calls: 0, usd: 0 }, data || {});
    }
    get total_tokens() { return this.input_tokens + this.output_tokens; }
    record(model, inputTokens, outputTokens) {
      const [inRate, outRate] = modelRates(model);
      this.requests += 1;
      this.input_tokens += inputTokens || 0;
      this.output_tokens += outputTokens || 0;
      this.usd += ((inputTokens || 0) / 1e6) * inRate + ((outputTokens || 0) / 1e6) * outRate;
    }
    estimatedUsd() { return this.usd; }
    toJSON() {
      return {
        requests: this.requests, input_tokens: this.input_tokens, output_tokens: this.output_tokens,
        tool_calls: this.tool_calls, api_calls: this.api_calls, usd: this.usd,
      };
    }
  }

  const normTitle = (title) => (title || "").toLowerCase().replace(/[^a-z0-9]/g, "");

  class ResearchContext {
    constructor(runId, sessionId, question) {
      this.run_id = runId;
      this.session_id = sessionId;
      this.research_question = question;
      this.plan = null;
      this.literature = null;
      this.synthesis = null;
      this.gap_analysis = null;
      this.hypotheses = null;
      this.experiment = null;
      this.citations = null;
      this.report = null;
      this.corpus = {};
      this.attachments = [];
      this.source_texts = {};
      this.source_mode = "web";
      this.saved_artifacts = {};
      this.vector_store = null;
      this.stage = A.Stage.PLANNING;
      this.reflection_round = 0;
      this.costs = new CostLedger();
      this.approvals = [];
      this.errors = [];
      this.overclaim_flags = [];
      this.started_at = new Date().toISOString();
    }

    /** Merge papers into the corpus, deduplicating by id and title. */
    addPapers(papers, { force = false } = {}) {
      const ceiling = force ? Infinity : Settings.get().max_corpus_papers;
      const seen = new Set(Object.values(this.corpus).map((p) => normTitle(p.title)));
      let added = 0;
      for (const paper of papers) {
        if (Object.keys(this.corpus).length >= ceiling) break;
        if (this.corpus[paper.paper_id]) continue;
        const key = normTitle(paper.title);
        if (key && seen.has(key)) continue;
        this.corpus[paper.paper_id] = paper;
        seen.add(key);
        added += 1;
      }
      return added;
    }
    resolveCitations(ids) { return (ids || []).filter((id) => this.corpus[id]).map((id) => this.corpus[id]); }
    unknownCitations(ids) { return (ids || []).filter((id) => !this.corpus[id]); }

    recordApproval(toolName, args, decision, note) {
      this.approvals.push({ tool_name: toolName, arguments: args, decision, decided_at: new Date().toISOString(), note: note || "" });
    }
    recordError(message) {
      this.errors.push(`${new Date().toISOString()} ${message}`);
    }

    toDict() {
      return {
        run_id: this.run_id, session_id: this.session_id, research_question: this.research_question,
        stage: this.stage, reflection_round: this.reflection_round, started_at: this.started_at,
        plan: this.plan, literature: this.literature, synthesis: this.synthesis,
        gap_analysis: this.gap_analysis, hypotheses: this.hypotheses, experiment: this.experiment,
        citations: this.citations, report: this.report, corpus: this.corpus,
        attachments: this.attachments, source_texts: this.source_texts, source_mode: this.source_mode,
        saved_artifacts: this.saved_artifacts, approvals: this.approvals, errors: this.errors,
        overclaim_flags: this.overclaim_flags, costs: this.costs.toJSON(),
        vector_chunks: this.vector_store ? this.vector_store.chunks : [],
      };
    }

    static fromDict(data) {
      const ctx = new ResearchContext(data.run_id, data.session_id, data.research_question);
      for (const key of [
        "stage", "reflection_round", "started_at", "plan", "literature", "synthesis", "gap_analysis",
        "hypotheses", "experiment", "citations", "report", "corpus", "attachments", "source_texts",
        "source_mode", "saved_artifacts", "approvals", "errors", "overclaim_flags",
      ]) {
        if (data[key] !== undefined) ctx[key] = data[key];
      }
      ctx.costs = new CostLedger(data.costs);
      if (data.vector_chunks && data.vector_chunks.length && A.VectorStore) {
        ctx.vector_store = A.VectorStore.fromChunks(ctx.run_id, data.vector_chunks);
      }
      return ctx;
    }

    summary() {
      return {
        run_id: this.run_id,
        session_id: this.session_id,
        stage: this.stage,
        corpus_size: Object.keys(this.corpus).length,
        indexed_chunks: this.vector_store ? this.vector_store.chunks.length : 0,
        reflection_round: this.reflection_round,
        has_plan: !!this.plan,
        has_synthesis: !!this.synthesis,
        has_gap_analysis: !!this.gap_analysis,
        has_experiment: !!this.experiment,
        has_citations: !!this.citations,
        has_report: !!this.report,
        approvals: this.approvals.length,
        errors: this.errors.length,
        tokens: this.costs.total_tokens,
        estimated_usd: Math.round(this.costs.estimatedUsd() * 10000) / 10000,
      };
    }
  }

  // ---------------------------------------------------------------------
  // Small shared utilities
  // ---------------------------------------------------------------------

  const Util = {
    sleep,
    clean(text) {
      const el = typeof document !== "undefined" ? document.createElement("textarea") : null;
      let value = text || "";
      if (el) { el.innerHTML = value; value = el.value; }
      return value.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    },
    escape(text) {
      return String(text == null ? "" : text)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    },
    download(filename, content, mime) {
      const blob = content instanceof Blob ? content : new Blob([content], { type: mime || "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    },
    loadScript(src) {
      return new Promise((resolve, reject) => {
        if (document.querySelector(`script[data-src="${src}"]`)) return resolve();
        const s = document.createElement("script");
        s.src = src; s.async = true; s.dataset.src = src;
        s.onload = () => resolve();
        s.onerror = () => reject(new Error(`could not load ${src}`));
        document.head.appendChild(s);
      });
    },
    fmtUsd(v) { return `$${(v || 0).toFixed(4)}`; },
  };

  Object.assign(A, {
    Settings, DB, Log, newRunId, fetchWithRetry, CostLedger, modelRates, ResearchContext, Util,
    errors: {
      AthenaError, ExternalAPIError, ToolExecutionError, DocumentParseError, UnsafeCodeError,
      ApprovalDeferred, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered,
      MaxTurnsExceeded, ModelError,
    },
  });
})((window.Athena = window.Athena || {}));
