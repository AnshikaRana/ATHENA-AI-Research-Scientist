/* Athena - the tool layer.
 *
 * Browser ports of tools/literature.py, rag.py, memory/vector_store.py,
 * citations.py, analysis.py, datasets.py, documents.py, memory_tools.py,
 * memory/long_term.py and artifacts.py. Tools mutate the shared
 * ResearchContext locally and hand the model a compact digest, never the full
 * corpus - the same context-management discipline as the Python version.
 */
(function (A) {
  "use strict";
  const { Settings, Log, fetchWithRetry, DB, Util } = A;
  const { ExternalAPIError, ToolExecutionError, DocumentParseError } = A.errors;

  // =====================================================================
  // OpenAI client (used by agents, vision and transcription)
  // =====================================================================

  const OpenAI = {
    base: "https://api.openai.com/v1",
    async request(path, body, { form = false, signal } = {}) {
      const key = Settings.apiKey();
      if (!key) throw new A.errors.ModelError("No OpenAI API key configured.", 401);
      let lastErr;
      for (let attempt = 1; attempt <= 4; attempt++) {
        let response;
        try {
          response = await fetch(this.base + path, {
            method: "POST",
            headers: form ? { Authorization: `Bearer ${key}` } : { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
            body: form ? body : JSON.stringify(body),
            signal,
          });
        } catch (err) {
          lastErr = new A.errors.ModelError(`network error contacting OpenAI: ${err.message}`, 0);
          await Util.sleep(800 * attempt);
          continue;
        }
        if (response.ok) return response.json();
        let detail = "";
        try { const j = await response.json(); detail = (j.error && j.error.message) || JSON.stringify(j); } catch (e) { detail = response.statusText; }
        lastErr = new A.errors.ModelError(`OpenAI ${response.status}: ${detail}`, response.status);
        if (response.status === 429 || response.status >= 500) {
          const retryAfter = parseFloat(response.headers.get("retry-after") || "0");
          await Util.sleep(retryAfter ? retryAfter * 1000 : 1500 * attempt);
          continue;
        }
        throw lastErr;
      }
      throw lastErr;
    },
  };

  // =====================================================================
  // Uploaded files (data/uploads in the Python version)
  // =====================================================================

  const Uploads = {
    files: new Map(),
    add(file) { this.files.set(file.name, file); },
    remove(name) { this.files.delete(name); },
    clear() { this.files.clear(); },
    list() { return [...this.files.values()]; },
    resolve(path) {
      const name = String(path || "").split(/[\\/]/).pop();
      if (this.files.has(name)) return this.files.get(name);
      const lower = name.toLowerCase();
      for (const [k, v] of this.files) if (k.toLowerCase() === lower) return v;
      return null;
    },
  };

  // =====================================================================
  // Literature: arXiv, OpenAlex, Crossref (tools/literature.py)
  // =====================================================================

  const ARXIV_API = "https://export.arxiv.org/api/query";
  const OPENALEX_API = "https://api.openalex.org/works";
  const CROSSREF_API = "https://api.crossref.org/works";

  function reconstructAbstract(inverted) {
    if (!inverted) return "";
    const positions = [];
    for (const [word, idx] of Object.entries(inverted)) for (const i of idx) positions.push([i, word]);
    positions.sort((a, b) => a[0] - b[0]);
    return Util.clean(positions.map((p) => p[1]).join(" "));
  }

  function parseArxiv(xml, limit) {
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const entries = [...doc.getElementsByTagName("entry")].slice(0, limit);
    const papers = [];
    for (const entry of entries) {
      const text = (tag) => { const el = entry.getElementsByTagName(tag)[0]; return el ? el.textContent : ""; };
      const rawId = text("id");
      const arxivId = rawId ? rawId.split("/").pop() : "";
      if (!arxivId) continue;
      const published = text("published");
      const year = /^\d{4}/.test(published) ? parseInt(published.slice(0, 4), 10) : 0;
      const authors = [...entry.getElementsByTagName("author")]
        .map((a) => { const n = a.getElementsByTagName("name")[0]; return n ? n.textContent.trim() : ""; })
        .filter(Boolean).slice(0, 10);
      papers.push({
        paper_id: `arxiv:${arxivId}`, title: Util.clean(text("title")), authors, year,
        venue: "arXiv preprint", abstract: Util.clean(text("summary")),
        url: `https://arxiv.org/abs/${arxivId}`, source: "arxiv", citation_count: 0,
      });
    }
    return papers;
  }

  function parseOpenalex(payload, limit) {
    const papers = [];
    for (const work of (payload.results || []).slice(0, limit)) {
      const workId = (work.id || "").split("/").pop();
      if (!workId) continue;
      const source = ((work.primary_location || {}).source) || {};
      const doi = (work.doi || "").replace("https://doi.org/", "");
      papers.push({
        paper_id: doi ? `doi:${doi}` : `openalex:${workId}`,
        title: Util.clean(work.title || work.display_name || ""),
        authors: (work.authorships || []).slice(0, 10).map((a) => Util.clean(((a.author || {}).display_name) || "")).filter(Boolean),
        year: parseInt(work.publication_year || 0, 10) || 0,
        venue: Util.clean(source.display_name || ""),
        abstract: reconstructAbstract(work.abstract_inverted_index),
        url: work.doi || work.id || "",
        source: "openalex",
        citation_count: parseInt(work.cited_by_count || 0, 10) || 0,
      });
    }
    return papers;
  }

  function safeYear(parts) {
    try { const v = parts[0][0]; return v == null ? 0 : parseInt(v, 10) || 0; } catch (e) { return 0; }
  }

  function parseCrossref(payload, limit) {
    const papers = [];
    for (const item of ((payload.message || {}).items || []).slice(0, limit)) {
      const doi = item.DOI || "";
      if (!doi) continue;
      const titles = item.title || [];
      const containers = item["container-title"] || [];
      papers.push({
        paper_id: `doi:${doi}`,
        title: Util.clean(titles[0] || ""),
        authors: (item.author || []).slice(0, 10).map((a) => Util.clean(`${a.given || ""} ${a.family || ""}`)).filter(Boolean),
        year: safeYear((item.issued || {})["date-parts"]),
        venue: Util.clean(containers[0] || ""),
        abstract: Util.clean(item.abstract || ""),
        url: item.URL || `https://doi.org/${doi}`,
        source: "crossref",
        citation_count: parseInt(item["is-referenced-by-count"] || 0, 10) || 0,
      });
    }
    return papers;
  }

  const SOURCES = {
    async arxiv(query, limit) {
      const text = await fetchWithRetry(ARXIV_API, {
        as: "text",
        params: { search_query: `all:${query}`, start: 0, max_results: limit, sortBy: "relevance" },
      });
      return parseArxiv(text, limit);
    },
    async openalex(query, limit) {
      const payload = await fetchWithRetry(OPENALEX_API, {
        params: { search: query, per_page: limit, sort: "relevance_score:desc", mailto: Settings.get().contact_email || undefined },
      });
      return parseOpenalex(payload, limit);
    },
    async crossref(query, limit) {
      const payload = await fetchWithRetry(CROSSREF_API, {
        params: { query, rows: limit, sort: "relevance", mailto: Settings.get().contact_email || undefined },
      });
      return parseCrossref(payload, limit);
    },
  };

  function digest(ctx, papers, label, added) {
    if (!papers.length) return `${label}: no results.`;
    const lines = [`${label}: ${papers.length} results, ${added} new (corpus now ${Object.keys(ctx.corpus).length}).`];
    for (const p of papers.slice(0, 12)) {
      const who = p.authors.length ? `${p.authors[0]} et al.` : "unknown";
      lines.push(`- [${p.paper_id}] ${p.title} (${who}, ${p.year}${p.citation_count ? `, ${p.citation_count} cites` : ""})`);
    }
    return lines.join("\n");
  }

  async function searchLiterature(ctx, { query, sources, max_results_per_source }) {
    const ceiling = Settings.get().max_papers_per_source;
    const limit = Math.max(1, Math.min(parseInt(max_results_per_source || ceiling, 10) || ceiling, ceiling));
    let chosen = (sources || []).filter((s) => SOURCES[s]);
    if (!chosen.length) chosen = Object.keys(SOURCES);
    Log.event("tools.literature", "literature search", { query, sources: chosen, limit });

    const results = await Promise.allSettled(chosen.map((name) => SOURCES[name](query, limit)));
    const all = [];
    const report = [];
    results.forEach((result, i) => {
      const name = chosen[i];
      if (result.status === "rejected") {
        const message = `${name} unavailable (${(result.reason && result.reason.name) || "Error"})`;
        ctx.recordError(message);
        Log.event("tools.literature", message, { source: name, error: String(result.reason && result.reason.message) });
        report.push(`${name}: FAILED - ${message}`);
        return;
      }
      all.push(...result.value);
      report.push(`${name}: ${result.value.length} records`);
    });
    if (!all.length) throw new ExternalAPIError("search_literature", "all sources returned no results");
    const added = ctx.addPapers(all);
    ctx.costs.api_calls += chosen.length;
    return digest(ctx, all, `Query '${query}' [${report.join("; ")}]`, added);
  }

  // =====================================================================
  // Vector store: lexical hashed embeddings (memory/vector_store.py)
  // =====================================================================

  const DIM = 512;
  const STOPWORDS = new Set(
    `a an the of and or to in on for with by is are was were be been being
    this that these those it its as at from we our they their he she you i
    do does did how what which who whom when where why can could should would
    will shall may might must have has had not no but if then than so such
    about into over under more most other some any each`.split(/\s+/)
  );
  const tokenize = (t) => (String(t || "").toLowerCase().match(/[a-z0-9]+/g) || []);
  function features(text) {
    const tokens = tokenize(text).filter((t) => !STOPWORDS.has(t) && t.length > 1);
    const bigrams = [];
    for (let i = 0; i + 1 < tokens.length; i++) bigrams.push(`${tokens[i]}_${tokens[i + 1]}`);
    return tokens.concat(bigrams);
  }
  function fnv1a(str, seed) {
    let h = seed >>> 0;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    return h >>> 0;
  }
  /** Deterministic bag-of-features embedding via signed feature hashing. */
  function hashedEmbedding(text) {
    const counts = new Map();
    for (const f of features(text)) counts.set(f, (counts.get(f) || 0) + 1);
    const v = new Float32Array(DIM);
    for (const [f, c] of counts) {
      const index = fnv1a(f, 2166136261) % DIM;
      const sign = fnv1a(f, 374761393) % 2 === 0 ? 1 : -1;
      v[index] += sign * (1 + Math.log(c)); // sublinear tf
    }
    let norm = 0;
    for (let i = 0; i < DIM; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < DIM; i++) v[i] /= norm;
    return v;
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  /** Split on sentence boundaries into overlapping windows. */
  function chunkText(text, size = 900, overlap = 150) {
    text = String(text || "").replace(/\s+/g, " ").trim();
    if (!text) return [];
    if (text.length <= size) return [text];
    const sentences = text.split(/(?<=[.!?])\s+/);
    const chunks = [];
    let current = "";
    for (const sentence of sentences) {
      if (current.length + sentence.length + 1 <= size) current = `${current} ${sentence}`.trim();
      else {
        if (current) chunks.push(current);
        const tail = overlap && current ? current.slice(-overlap) : "";
        current = `${tail} ${sentence}`.trim();
      }
    }
    if (current) chunks.push(current);
    return chunks;
  }

  class VectorStore {
    constructor(name) { this.name = name; this.chunks = []; this.vectors = []; this.indexed = new Set(); }
    static fromChunks(name, chunks) {
      const store = new VectorStore(name);
      for (const c of chunks) {
        store.chunks.push(c);
        store.vectors.push(hashedEmbedding(`${c.title}. ${c.text}`));
        store.indexed.add(c.paper_id);
      }
      return store;
    }
    addDocument(paperId, title, text) {
      if (this.indexed.has(paperId)) return 0;
      const pieces = chunkText(text);
      if (!pieces.length) return 0;
      pieces.forEach((piece, i) => {
        const chunk = { chunk_id: `${paperId}#${i}`, paper_id: paperId, title, text: piece, position: i };
        this.chunks.push(chunk);
        this.vectors.push(hashedEmbedding(`${title}. ${piece}`));
      });
      this.indexed.add(paperId);
      return pieces.length;
    }
    search(query, k = 6) {
      if (!this.chunks.length) return [];
      const q = hashedEmbedding(query);
      const scored = this.vectors.map((v, i) => [i, dot(v, q)]);
      scored.sort((a, b) => b[1] - a[1]);
      const results = scored.slice(0, Math.min(k, scored.length)).map(([i, s]) => [this.chunks[i], s]);
      Log.event("rag", "rag search", { query: query.slice(0, 120), returned: results.length, top_score: results.length ? +results[0][1].toFixed(4) : null });
      return results;
    }
    stats() { return { name: this.name, documents: this.indexed.size, chunks: this.chunks.length, dimensions: DIM }; }
  }
  const ensureStore = (ctx) => (ctx.vector_store || (ctx.vector_store = new VectorStore(ctx.run_id)));

  // =====================================================================
  // Citations (tools/citations.py) - deterministic, from corpus metadata
  // =====================================================================

  function splitName(name) {
    name = String(name || "").replace(/\s+/g, " ").trim();
    if (!name) return ["Unknown", ""];
    if (name.includes(",")) { const i = name.indexOf(","); return [name.slice(0, i).trim(), name.slice(i + 1).trim()]; }
    const parts = name.split(" ");
    return parts.length > 1 ? [parts[parts.length - 1], parts.slice(0, -1).join(" ")] : [parts[0], ""];
  }
  const initials = (given) => given.split(/\s+/).filter(Boolean).map((p) => `${p[0]}.`).join(" ");
  function apaAuthors(authors) {
    if (!authors.length) return "Unknown";
    const f = authors.slice(0, 20).map((a) => { const [fam, giv] = splitName(a); return `${fam}, ${initials(giv)}`.trim().replace(/,$/, ""); });
    if (f.length === 1) return f[0];
    return `${f.slice(0, -1).join(", ")}, & ${f[f.length - 1]}`;
  }
  function ieeeAuthors(authors) {
    if (!authors.length) return "Unknown";
    const f = authors.slice(0, 6).map((a) => { const [fam, giv] = splitName(a); return `${initials(giv)} ${fam}`.trim(); });
    if (authors.length > 6) f.push("et al.");
    return f.join(", ");
  }
  function mlaAuthors(authors) {
    if (!authors.length) return "Unknown";
    const [fam, giv] = splitName(authors[0]);
    const first = `${fam}, ${giv}`.trim().replace(/,$/, "");
    return authors.length === 1 ? first : `${first}, et al.`;
  }
  function inText(authors, year) {
    const y = year ? String(year) : "n.d.";
    if (!authors.length) return `(Unknown, ${y})`;
    const [fam] = splitName(authors[0]);
    if (authors.length === 1) return `(${fam}, ${y})`;
    if (authors.length === 2) return `(${fam} & ${splitName(authors[1])[0]}, ${y})`;
    return `(${fam} et al., ${y})`;
  }
  function bibtexKey(p) {
    const fam = p.authors.length ? splitName(p.authors[0])[0] : "unknown";
    const word = ((p.title.match(/[A-Za-z]+/g) || []).find((w) => w.length > 3) || "untitled").toLowerCase();
    const slug = fam.toLowerCase().replace(/[^a-z0-9]/g, "") || "unknown";
    return `${slug}${p.year || "nd"}${word}`;
  }
  function bibtexEntry(p, key) {
    const type = p.venue && !p.venue.toLowerCase().includes("arxiv") ? "article" : "misc";
    const fields = [`  title        = {${p.title}}`, `  author       = {${p.authors.join(" and ") || "Unknown"}}`];
    if (p.year) fields.push(`  year         = {${p.year}}`);
    if (p.venue) fields.push(`  journal      = {${p.venue}}`);
    if (p.paper_id.startsWith("doi:")) fields.push(`  doi          = {${p.paper_id.slice(4)}}`);
    if (p.paper_id.startsWith("arxiv:")) { fields.push(`  eprint       = {${p.paper_id.slice(6)}}`); fields.push("  archivePrefix = {arXiv}"); }
    if (p.url) fields.push(`  url          = {${p.url}}`);
    return `@${type}{${key},\n${fields.join(",\n")}\n}`;
  }
  /** Render one reference-list entry. Missing fields degrade, never fabricate. */
  function renderReference(p, style) {
    const year = p.year ? String(p.year) : "n.d.";
    const venue = p.venue || "Preprint";
    const title = p.title.replace(/\.+$/, "");
    let base;
    if (style === "APA") base = `${apaAuthors(p.authors)} (${year}). ${title}. *${venue}*.`;
    else if (style === "IEEE") base = `${ieeeAuthors(p.authors)}, "${title}," ${venue}, ${year}.`;
    else if (style === "MLA") base = `${mlaAuthors(p.authors).replace(/\.$/, "")}. "${title}." ${venue}, ${year}.`;
    else base = `${ieeeAuthors(p.authors)}. ${title}. *${venue}* (${year}).`;
    if (p.paper_id.startsWith("doi:")) base += ` https://doi.org/${p.paper_id.slice(4)}`;
    else if (p.url && !p.url.startsWith("local:")) base += ` ${p.url}`;
    return base;
  }
  const VALID_STYLES = ["APA", "IEEE", "MLA", "Nature"];
  const sortByAuthor = (papers) => [...papers].sort((a, b) => {
    const fa = a.authors.length ? splitName(a.authors[0])[0].toLowerCase() : "";
    const fb = b.authors.length ? splitName(b.authors[0])[0].toLowerCase() : "";
    return fa < fb ? -1 : fa > fb ? 1 : (b.year || 0) - (a.year || 0);
  });

  /** Unique BibTeX keys with letter suffixes on collision (smith2020a, smith2020b). */
  function uniqueKeys(papers) {
    const base = papers.map(bibtexKey);
    const counts = {};
    base.forEach((k) => { counts[k] = (counts[k] || 0) + 1; });
    const used = {};
    return base.map((k) => {
      if (counts[k] === 1) return k;
      used[k] = (used[k] || 0) + 1;
      return k + String.fromCharCode(96 + used[k]);
    });
  }

  /** Deterministic bundle construction, shared by the offline agent and the UI. */
  function buildCitationBundle(ctx, ids, style) {
    style = VALID_STYLES.includes(style) ? style : "APA";
    const all = ids && ids.length ? ids : Object.keys(ctx.corpus);
    const papers = sortByAuthor(ctx.resolveCitations(all));
    const missing = ctx.unknownCitations(all);
    const keys = uniqueKeys(papers);
    const citations = papers.map((p, i) => ({
      paper_id: p.paper_id, in_text: inText(p.authors, p.year), full_reference: renderReference(p, style),
      bibtex_key: keys[i], bibtex_entry: bibtexEntry(p, keys[i]),
    }));
    const incomplete = papers.filter((p) => !p.year || !p.authors.length || !p.venue).map((p) => p.paper_id);
    return {
      style,
      citations,
      reference_list_markdown: citations.map((c) => `- ${c.full_reference}`).join("\n"),
      bibtex_file: citations.map((c) => c.bibtex_entry).join("\n\n"),
      unverified_ids: missing,
      integrity_note:
        `Verified ${papers.length} of ${all.length} ids against the retrieved corpus; ` +
        `${missing.length} unverified id(s) were excluded. Every entry was generated from stored metadata. ` +
        (incomplete.length ? `Records with a missing year, author list or venue: ${incomplete.join(", ")}.` : "All records had a year, authors and venue."),
    };
  }

  // =====================================================================
  // Statistics (tools/analysis.py, tools/datasets.py)
  // =====================================================================

  /** Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.2e-9 after refinement). */
  function invNorm(p) {
    if (p <= 0 || p >= 1) throw new Error("p must be in (0,1)");
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425;
    let q, r, x;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    else if (p <= 1 - pl) { q = p - 0.5; r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    else { q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    // One Halley refinement step using erfc.
    const e = 0.5 * erfc(-x / Math.SQRT2) - p;
    const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
    return x - u / (1 + (x * u) / 2);
  }
  function erfc(x) {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }

  /** Per-group n for a two-sample comparison of means (normal approximation + 2). */
  function requiredSampleSize(effect, alpha = 0.05, power = 0.8, twoSided = true) {
    if (!(effect > 0)) throw new ToolExecutionError("statistical_power_analysis", "effect_size must be greater than 0");
    if (!(alpha > 0 && alpha < 1) || !(power > 0 && power < 1)) throw new ToolExecutionError("statistical_power_analysis", "alpha and power must both be in (0, 1)");
    const za = twoSided ? invNorm(1 - alpha / 2) : invNorm(1 - alpha);
    const zp = invNorm(power);
    return Math.ceil((2 * (za + zp) ** 2) / effect ** 2) + 2;
  }

  function logGamma(x) {
    const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (const c of cof) { y += 1; ser += c / y; }
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }
  function betacf(a, b, x) {
    const MAXIT = 200, EPS = 3e-7, FPMIN = 1e-30;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function betai(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return (front * betacf(a, b, x)) / a;
    return 1 - (front * betacf(b, a, 1 - x)) / b;
  }
  const twoSidedP = (t, df) => (df <= 0 ? NaN : betai(0.5 * df, 0.5, df / (df + t * t)));

  // --- Dataset parsing ----------------------------------------------------

  function parseDelimited(text, sep) {
    const rows = [];
    let row = [], field = "", inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false; }
        else field += ch;
      } else if (ch === '"') inQuotes = true;
      else if (ch === sep) { row.push(field); field = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field); field = "";
        if (row.length > 1 || row[0] !== "") rows.push(row);
        row = [];
      } else field += ch;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    if (!rows.length) return { columns: [], rows: [] };
    const columns = rows[0].map((c, i) => c.trim() || `col${i}`);
    return { columns, rows: rows.slice(1).map((r) => Object.fromEntries(columns.map((c, i) => [c, r[i] === undefined ? "" : r[i]]))) };
  }

  async function loadDataframe(file) {
    const name = file.name.toLowerCase();
    const ext = name.slice(name.lastIndexOf("."));
    let frame;
    try {
      if (ext === ".csv" || ext === ".txt") frame = parseDelimited(await file.text(), ",");
      else if (ext === ".tsv" || ext === ".tab") frame = parseDelimited(await file.text(), "\t");
      else if (ext === ".json") {
        const data = JSON.parse(await file.text());
        const rows = Array.isArray(data) ? data : Object.keys(data).length && Array.isArray(Object.values(data)[0])
          ? Object.values(data)[0].map((_, i) => Object.fromEntries(Object.keys(data).map((k) => [k, data[k][i]])))
          : [data];
        const columns = [...new Set(rows.flatMap((r) => Object.keys(r || {})))];
        frame = { columns, rows };
      } else if (ext === ".xlsx" || ext === ".xls") {
        await Util.loadScript("https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js");
        const wb = window.XLSX.read(await file.arrayBuffer(), { type: "array" });
        const rows = window.XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { defval: "" });
        frame = { columns: rows.length ? Object.keys(rows[0]) : [], rows };
      } else throw new DocumentParseError("analyze_dataset", `unsupported file type '${ext}'`);
    } catch (err) {
      if (err instanceof DocumentParseError) throw err;
      throw new DocumentParseError("analyze_dataset", `could not parse ${file.name}: ${err.message}`);
    }
    if (frame.rows.length * frame.columns.length > 2000000) {
      throw new DocumentParseError("analyze_dataset", `${file.name} is above the 2,000,000 cell limit`);
    }
    return frame;
  }
  const isMissing = (v) => v === null || v === undefined || (typeof v === "string" && (v.trim() === "" || /^(na|nan|null|none)$/i.test(v.trim())));
  const toNum = (v) => (typeof v === "number" ? v : isMissing(v) ? NaN : Number(String(v).trim()));
  function numericColumns(frame) {
    return frame.columns.filter((c) => {
      let seen = 0;
      for (const r of frame.rows) { if (isMissing(r[c])) continue; seen++; if (!Number.isFinite(toNum(r[c]))) return false; }
      return seen > 0;
    });
  }
  const colValues = (frame, c) => frame.rows.map((r) => toNum(r[c])).filter(Number.isFinite);
  const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
  const variance = (xs) => { const m = mean(xs); return xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1); };
  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function pearson(frame, a, b) {
    const xs = [], ys = [];
    for (const r of frame.rows) { const x = toNum(r[a]), y = toNum(r[b]); if (Number.isFinite(x) && Number.isFinite(y)) { xs.push(x); ys.push(y); } }
    if (xs.length < 3) return NaN;
    const mx = mean(xs), my = mean(ys);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
    return sxx && syy ? sxy / Math.sqrt(sxx * syy) : NaN;
  }
  function table(header, rows) {
    const widths = header.map((h, i) => Math.max(String(h).length, ...rows.map((r) => String(r[i]).length)));
    const line = (cells) => cells.map((c, i) => String(c).padStart(widths[i])).join("  ");
    return [line(header), ...rows.map(line)].join("\n");
  }
  const r4 = (x) => (Number.isFinite(x) ? Math.round(x * 10000) / 10000 : "NaN");

  // =====================================================================
  // Documents: PDF, figures, audio (tools/documents.py)
  // =====================================================================

  const MAX_PDF_PAGES = 40;
  const slug = (t) => (String(t).replace(/[^0-9a-zA-Z]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase().slice(0, 48).replace(/_+$/, "")) || "document";
  const dehyphenate = (t) => t.replace(/(\w)-\n(?=[a-z])/g, "$1");

  async function loadPdfJs() {
    if (window.pdfjsLib) return window.pdfjsLib;
    await Util.loadScript("https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js");
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
    return window.pdfjsLib;
  }

  async function extractPdfText(file) {
    const pdfjs = await loadPdfJs();
    let pdf;
    try { pdf = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise; }
    catch (err) {
      if (err && err.name === "PasswordException") throw new DocumentParseError("ingest_pdf", `${file.name} is password protected`);
      throw new DocumentParseError("ingest_pdf", `cannot open ${file.name}: ${err.message}`);
    }
    const pages = Math.min(pdf.numPages, MAX_PDF_PAGES);
    const parts = [];
    for (let n = 1; n <= pages; n++) {
      try {
        const page = await pdf.getPage(n);
        const content = await page.getTextContent();
        let text = "", lastY = null;
        for (const item of content.items) {
          const y = item.transform ? item.transform[5] : null;
          if (lastY !== null && y !== null && Math.abs(y - lastY) > 2 && !text.endsWith("\n")) text += "\n";
          text += item.str;
          if (item.hasEOL) text += "\n";
          lastY = y;
        }
        parts.push(text);
      } catch (e) { parts.push(""); }
    }
    const body = dehyphenate(parts.join("\n").trim());
    if (!body) throw new DocumentParseError("ingest_pdf", `${file.name} yielded no extractable text - it is probably a scan. Run OCR on it first.`);
    let metaTitle = "";
    try { const meta = await pdf.getMetadata(); metaTitle = String((meta.info && meta.info.Title) || "").trim(); } catch (e) { /* no metadata */ }
    let title = "";
    const junk = ["microsoft word", "untitled", ".doc", ".tex", ".dvi"];
    if (metaTitle && !junk.some((m) => metaTitle.toLowerCase().includes(m))) title = metaTitle;
    if (!title) {
      for (const line of body.split("\n")) {
        const c = line.trim();
        if (c.length >= 15 && c.length <= 200 && !c.toLowerCase().startsWith("arxiv:")) { title = c; break; }
      }
    }
    if (!title) title = file.name.replace(/\.pdf$/i, "").replace(/_/g, " ");
    return { title, body, pages };
  }

  const ARXIV_STAMP = /arXiv:(\d{4}\.\d{4,5})(?:v\d+)?\s*\[[^\]]+\]\s*(\d{1,2}\s+\w+\s+(\d{4}))?/;
  const NAME = /^[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){1,3}$/;
  function deriveMetadata(body, title) {
    let authors = [];
    const lines = body.split("\n").map((l) => l.trim()).filter(Boolean);
    let start = lines.indexOf(title);
    start = start >= 0 ? start + 1 : 1;
    for (const line of lines.slice(start, start + 3)) {
      const names = line.split(/,|\band\b/).map((n) => n.trim()).filter(Boolean);
      if (names.length >= 2 && names.every((n) => NAME.test(n))) { authors = names; break; }
    }
    let year = 0, venue = "", url = "";
    const stamp = ARXIV_STAMP.exec(body);
    if (stamp) {
      venue = `arXiv preprint arXiv:${stamp[1]}`;
      url = `https://arxiv.org/abs/${stamp[1]}`;
      year = stamp[3] ? parseInt(stamp[3], 10) : 2000 + parseInt(stamp[1].slice(0, 2), 10);
    }
    return { authors, year, venue, url };
  }

  /** Parse a PDF into the corpus and the RAG index (shared by the tool and document mode). */
  async function ingestPdfFile(ctx, filePath) {
    const file = Uploads.resolve(filePath);
    if (!file) throw new DocumentParseError("ingest_pdf", `no such file: ${filePath}`);
    const { title, body, pages } = await extractPdfText(file);
    const paperId = `upload:${slug(file.name.replace(/\.pdf$/i, ""))}`;
    const meta = deriveMetadata(body, title);
    const paper = {
      paper_id: paperId, title, authors: meta.authors, year: meta.year, venue: meta.venue || "user upload",
      abstract: body.slice(0, 4000), url: meta.url || `local:${file.name}`, source: "user_upload", citation_count: 0,
    };
    ctx.addPapers([paper], { force: true });
    ctx.attachments.push({ kind: "pdf", path: file.name, title, pages });
    ctx.source_texts[paperId] = body;
    const chunks = ensureStore(ctx).addDocument(paperId, title, body);
    Log.event("tools.documents", "pdf ingested", { path: file.name, pages, chunks });
    return { paper, chunks, body, pages };
  }

  const IMAGE_EXT = [".png", ".jpg", ".jpeg", ".gif", ".webp"];
  const AUDIO_EXT = [".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm"];
  const extOf = (n) => n.slice(n.lastIndexOf(".")).toLowerCase();
  const fileToDataUrl = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

  // =====================================================================
  // Long-term memory (memory/long_term.py) - IndexedDB instead of SQLite
  // =====================================================================

  const LongTermMemory = {
    async remember({ session_id, run_id, kind, topic, content }) {
      const embedding = Array.from(hashedEmbedding(`${topic}. ${content}`));
      const id = await DB.put("memories", { session_id, run_id, kind, topic, content, embedding, created_at: new Date().toISOString() });
      Log.event("memory.long_term", "memory stored", { memory_id: id, kind, topic });
      return id;
    },
    async recall(query, { k = 5, kinds = null } = {}) {
      const rows = (await DB.all("memories")).filter((r) => !kinds || kinds.includes(r.kind));
      if (!rows.length) return [];
      const q = hashedEmbedding(query);
      return rows
        .filter((r) => r.embedding && r.embedding.length === DIM)
        .map((r) => Object.assign({}, r, { score: dot(r.embedding, q) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
    },
    async recent(limit = 10) {
      return (await DB.all("memories")).sort((a, b) => b.id - a.id).slice(0, limit);
    },
    async stats() {
      const rows = await DB.all("memories");
      const byKind = {};
      rows.forEach((r) => { byKind[r.kind] = (byKind[r.kind] || 0) + 1; });
      return Object.assign({ total: rows.length, sessions: new Set(rows.map((r) => r.session_id)).size }, byKind);
    },
    async exportJson() {
      return JSON.stringify((await this.recent(1000)).map(({ embedding, ...rest }) => rest), null, 2);
    },
    async importJson(text) {
      const records = JSON.parse(text);
      let n = 0;
      for (const r of Array.isArray(records) ? records : []) {
        if (!r || !r.content) continue;
        await this.remember({ session_id: r.session_id || "imported", run_id: r.run_id || "imported", kind: r.kind || "finding", topic: r.topic || "imported", content: r.content });
        n++;
      }
      return n;
    },
    async remove(id) { await DB.delete("memories", id); },
    async clear() { await DB.clear("memories"); },
  };

  // =====================================================================
  // Python sandbox (run_python_analysis) - Pyodide in a Web Worker
  // =====================================================================

  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  const Sandbox = {
    worker: null,
    run(code) {
      return new Promise((resolve, reject) => {
        if (!this.worker) this.worker = new Worker(new URL("assets/js/pyworker.js", document.baseURI));
        const worker = this.worker;
        let execTimer = null;
        const loadTimer = setTimeout(() => fail(new ToolExecutionError("run_python_analysis", "the Python runtime did not load within 120s")), 120000);
        const done = () => { clearTimeout(loadTimer); clearTimeout(execTimer); worker.onmessage = null; worker.onerror = null; };
        const fail = (err) => { done(); worker.terminate(); this.worker = null; reject(err); };
        worker.onmessage = (e) => {
          const msg = e.data;
          if (msg.type === "started") {
            clearTimeout(loadTimer);
            execTimer = setTimeout(() => fail(new ToolExecutionError("run_python_analysis", "execution exceeded 15s and was terminated")), 15000);
          } else if (msg.type === "result") { done(); resolve(msg.payload); }
          else if (msg.type === "error") { done(); reject(new ToolExecutionError("run_python_analysis", msg.message)); }
        };
        worker.onerror = (e) => fail(new ToolExecutionError("run_python_analysis", e.message || "worker error"));
        worker.postMessage({ code, indexURL: PYODIDE_URL });
      });
    },
  };

  // =====================================================================
  // Agent-facing tool registry
  // =====================================================================

  const S = A.S;
  const DEEP_SWEEP_APPROVAL_THRESHOLD = 30;
  const Tools = {};
  function tool(def) { Tools[def.name] = def; }

  tool({
    name: "search_literature",
    description: "Search academic databases for papers and add them to the shared corpus. Queries the requested sources concurrently and merges the results, removing duplicates by identifier and by normalised title.",
    parameters: {
      query: S.str('A focused search phrase, e.g. "graph neural network expressivity".'),
      sources: S.arr(S.enm(["arxiv", "openalex", "crossref"]), 'Any of "arxiv", "openalex", "crossref". Use all three for breadth.'),
      max_results_per_source: S.int("How many records to pull from each source (1-15)."),
    },
    failure: (err) => `Literature search failed: ${err.message}`,
    run: (ctx, args) => searchLiterature(ctx, args),
  });

  tool({
    name: "list_corpus",
    description: "List every paper currently in the shared corpus with its citation id. Use this before writing any claim, so that citations refer to real papers.",
    parameters: {},
    run: async (ctx) => {
      const papers = Object.values(ctx.corpus);
      if (!papers.length) return "Corpus is empty. Run search_literature first.";
      const ranked = [...papers].sort((a, b) => b.citation_count - a.citation_count || b.year - a.year);
      return [`Corpus contains ${papers.length} papers:`, ...ranked.map((p) =>
        `- [${p.paper_id}] ${p.title} (${p.year}, ${p.venue || p.source}, ${p.citation_count} cites)`)].join("\n");
    },
  });

  tool({
    name: "get_paper_details",
    description: "Fetch full abstracts for specific papers by their corpus ids.",
    parameters: { paper_ids: S.arr(S.str(), 'Citation ids exactly as shown by list_corpus, e.g. "arxiv:2301.00001".') },
    run: async (ctx, { paper_ids }) => {
      const papers = ctx.resolveCitations(paper_ids);
      const missing = ctx.unknownCitations(paper_ids);
      if (!papers.length) return `None of ${JSON.stringify(paper_ids)} are in the corpus. Unknown ids: ${JSON.stringify(missing)}`;
      const blocks = papers.map((p) =>
        `### [${p.paper_id}] ${p.title}\nAuthors: ${p.authors.join(", ") || "unknown"}\nYear: ${p.year} | Venue: ${p.venue || "n/a"} | Citations: ${p.citation_count}\nURL: ${p.url}\nAbstract: ${p.abstract || "(not available from source)"}`);
      if (missing.length) blocks.push(`WARNING - not in corpus, do not cite: ${JSON.stringify(missing)}`);
      return blocks.join("\n\n");
    },
  });

  tool({
    name: "commission_deep_sweep",
    description: "Run a wide multi-query literature sweep. Large sweeps need approval. Sweeps requesting more than 30 total results pause for human authorisation because of their API cost; smaller ones proceed immediately.",
    parameters: {
      queries: S.arr(S.str(), "Distinct search phrases to run, typically 2-5."),
      total_results: S.int("Total records to retrieve across all queries."),
    },
    // Dynamic gate: a function of the arguments, not of the tool.
    needsApproval: (ctx, args) => {
      const requested = parseInt(args.total_results || 0, 10);
      const needs = requested > DEEP_SWEEP_APPROVAL_THRESHOLD;
      Log.event("tools.artifacts", "approval policy evaluated", { tool: "commission_deep_sweep", requested, threshold: DEEP_SWEEP_APPROVAL_THRESHOLD, approval_required: needs });
      return needs;
    },
    run: async (ctx, { queries, total_results }) => {
      queries = (queries || []).filter((q) => q.trim()).slice(0, 5);
      if (!queries.length) return "No queries supplied.";
      const names = Object.keys(SOURCES);
      const ceiling = Settings.get().max_papers_per_source;
      const perSource = Math.max(1, Math.min(Math.floor(total_results / (queries.length * names.length)) || 1, ceiling));
      const legs = queries.flatMap((q) => names.map((n) => SOURCES[n](q, perSource)));
      const results = await Promise.allSettled(legs);
      const papers = [];
      let failures = 0;
      for (const r of results) {
        if (r.status === "rejected") { failures++; ctx.recordError(`deep sweep leg failed: ${r.reason && r.reason.message}`); continue; }
        papers.push(...r.value);
      }
      const added = ctx.addPapers(papers);
      ctx.costs.api_calls += legs.length;
      Log.event("tools.artifacts", "deep sweep complete", { queries, legs: legs.length, failures, retrieved: papers.length, added });
      return `Deep sweep over ${queries.length} queries x ${names.length} sources (${legs.length} parallel calls, ${failures} failed): retrieved ${papers.length} records, ${added} new. Corpus now holds ${Object.keys(ctx.corpus).length} papers.`;
    },
  });

  tool({
    name: "index_corpus",
    description: "Embed every paper in the corpus into the vector index for retrieval. Call this once after literature search and before asking any grounded question. Papers already indexed are skipped, so calling it again is cheap.",
    parameters: {},
    failure: (err) => `Indexing failed: ${err.message}`,
    run: async (ctx) => {
      const store = ensureStore(ctx);
      const papers = Object.values(ctx.corpus);
      if (!papers.length) return "Corpus is empty - nothing to index. Run search_literature first.";
      let added = 0, docs = 0;
      const skipped = [];
      for (const p of papers) {
        if (store.indexed.has(p.paper_id)) continue;
        const body = p.abstract || p.title;
        if (!body.trim()) { skipped.push(p.paper_id); continue; }
        const n = store.addDocument(p.paper_id, p.title, body);
        added += n; docs += n ? 1 : 0;
      }
      const stats = store.stats();
      Log.event("tools.rag", "corpus indexed", Object.assign({}, stats, { newly_indexed: docs }));
      let msg = `Indexed ${docs} new documents (${added} chunks). Index now holds ${stats.documents} documents / ${stats.chunks} chunks.`;
      if (skipped.length) msg += ` Skipped ${skipped.length} records with no text.`;
      return msg;
    },
  });

  tool({
    name: "rag_search",
    description: "Retrieve the passages most relevant to a question, with citation ids. Returns verbatim excerpts. Ground every claim you write in these excerpts and cite the paper_id shown in brackets.",
    parameters: {
      question: S.str('A specific question, e.g. "what limits message passing expressivity?".'),
      top_k: S.int("How many passages to retrieve (3-10 is usually right)."),
    },
    failure: (err) => `Retrieval failed: ${err.message}`,
    run: async (ctx, { question, top_k }) => {
      const store = ensureStore(ctx);
      if (!store.chunks.length) return "The index is empty. Call index_corpus first.";
      const k = Math.max(1, Math.min(parseInt(top_k || 6, 10) || 6, 12));
      const hits = store.search(question, k);
      if (!hits.length) return `No passages retrieved for: ${question}`;
      const blocks = [`Retrieved ${hits.length} passages for: ${question}\n`];
      hits.forEach(([chunk, score], i) => blocks.push(`[${i + 1}] paper_id=${chunk.paper_id} | relevance=${score.toFixed(3)}\n    Title: ${chunk.title}\n    Excerpt: ${chunk.text}`));
      blocks.push("\nCite only the paper_id values listed above. If none of these passages support the claim you want to make, say the evidence is missing instead.");
      return blocks.join("\n");
    },
  });

  tool({
    name: "ingest_pdf",
    description: "Read an uploaded PDF, add it to the corpus and index it for retrieval.",
    parameters: { file_path: S.str("File name of the uploaded PDF.") },
    failure: (err) => `PDF ingestion failed: ${err.message}`,
    run: async (ctx, { file_path }) => {
      const { paper, chunks, body, pages } = await ingestPdfFile(ctx, file_path);
      return `Ingested '${paper.title}' (${pages} pages, ${body.length.toLocaleString()} chars) as [${paper.paper_id}] and indexed ${chunks} chunks. It is now searchable via rag_search.`;
    },
  });

  tool({
    name: "describe_figure",
    description: "Interpret a scientific figure, chart or diagram image.",
    parameters: {
      file_path: S.str("File name of an uploaded .png/.jpg/.webp image."),
      question: S.str('What to determine, e.g. "what trend does this plot show?".'),
    },
    failure: (err) => `Figure analysis failed: ${err.message}`,
    run: async (ctx, { file_path, question }) => {
      const file = Uploads.resolve(file_path);
      if (!file) throw new DocumentParseError("describe_figure", `no such file: ${file_path}`);
      if (!IMAGE_EXT.includes(extOf(file.name))) throw new DocumentParseError("describe_figure", `${extOf(file.name)} is not a supported image type`);
      ctx.attachments.push({ kind: "image", path: file.name, question });
      if (!Settings.live) {
        return `[OFFLINE MODE] Figure '${file.name}' was registered as an attachment but not visually analysed, because no OpenAI API key is configured. Question was: ${question}. Do not invent findings from this figure.`;
      }
      const model = Settings.get().vision_model;
      const response = await OpenAI.request("/chat/completions", {
        model,
        max_tokens: 700,
        messages: [
          { role: "system", content: "You are a scientific figure analyst. Describe only what is visibly present: axes, units, series, trends and magnitudes. If something is illegible, say so rather than guessing." },
          { role: "user", content: [{ type: "text", text: question }, { type: "image_url", image_url: { url: await fileToDataUrl(file) } }] },
        ],
      });
      ctx.costs.api_calls += 1;
      if (response.usage) ctx.costs.record(model, response.usage.prompt_tokens, response.usage.completion_tokens);
      const description = (response.choices[0].message.content || "").trim();
      Log.event("tools.documents", "figure analysed", { path: file.name, chars: description.length });
      return `Figure analysis of ${file.name}:\n${description}`;
    },
  });

  tool({
    name: "transcribe_audio",
    description: "Transcribe a recorded talk, interview or voice note into text.",
    parameters: { file_path: S.str("File name of an uploaded audio file (.mp3, .wav, .m4a, .webm).") },
    failure: (err) => `Transcription failed: ${err.message}`,
    run: async (ctx, { file_path }) => {
      const file = Uploads.resolve(file_path);
      if (!file) throw new DocumentParseError("transcribe_audio", `no such file: ${file_path}`);
      if (!AUDIO_EXT.includes(extOf(file.name))) throw new DocumentParseError("transcribe_audio", `${extOf(file.name)} is not a supported audio type`);
      ctx.attachments.push({ kind: "audio", path: file.name });
      if (!Settings.live) return `[OFFLINE MODE] Audio '${file.name}' registered but not transcribed - no OpenAI API key configured. Do not invent its contents.`;
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("model", Settings.get().transcription_model);
      const result = await OpenAI.request("/audio/transcriptions", form, { form: true });
      ctx.costs.api_calls += 1;
      const text = result.text || "";
      Log.event("tools.documents", "audio transcribed", { path: file.name, chars: text.length });
      return `Transcript of ${file.name} (${text.length.toLocaleString()} chars):\n\n${text}`;
    },
  });

  tool({
    name: "statistical_power_analysis",
    description: "Compute the sample size an experiment needs to detect a given effect. Always call this before stating a sample size. Never estimate one by eye.",
    parameters: {
      effect_size: S.num("Expected Cohen's d. Conventionally 0.2 small, 0.5 medium, 0.8 large."),
      alpha: S.num("Significance level, normally 0.05."),
      power: S.num("Desired power, normally 0.8 (or 0.9 for a confirmatory study)."),
    },
    run: async (ctx, { effect_size, alpha, power }) => {
      const n = requiredSampleSize(effect_size, alpha, power);
      Log.event("tools.analysis", "power analysis", { effect_size, alpha, power, n_per_group: n });
      const label = effect_size < 0.35 ? "small" : effect_size < 0.65 ? "medium" : "large";
      return `Power analysis (two-sample, two-sided):\n  effect size d = ${effect_size} (${label})\n  alpha = ${alpha}, target power = ${power}\n  REQUIRED n = ${n} per group (${n * 2} total)\nState this number and these assumptions explicitly in the design.`;
    },
  });

  tool({
    name: "run_python_analysis",
    description: "Execute a short Python snippet to compute something exactly. Use for simulations, effect-size conversions, sanity checks on estimates - anything where a wrong number would invalidate the design. Print results; stdout is what comes back. Restrictions: no file, network or OS access. Imports limited to math, statistics, random, json, itertools, collections, decimal, fractions, datetime, re and numpy. 15-second limit.",
    parameters: {
      code: S.str("Python source. Use print() to emit results."),
      purpose: S.str("One line on what this computes, recorded in the audit log."),
    },
    failure: (err) => `Analysis code failed: ${err.message}`,
    run: async (ctx, { code, purpose }) => {
      Log.event("tools.analysis", "sandbox exec", { purpose, code_chars: code.length });
      const result = await Sandbox.run(code);
      if (result.unsafe) throw new A.errors.UnsafeCodeError("run_python_analysis", result.unsafe);
      const out = (result.stdout || "").trim(), err = (result.stderr || "").trim();
      if (result.status) return `Code exited with status ${result.status}.\nstderr:\n${err.slice(0, 2000)}`;
      if (!out) return "Code ran successfully but printed nothing. Use print() to emit results.";
      return `Output (${purpose}):\n${out.slice(0, 4000)}`;
    },
  });

  tool({
    name: "analyze_dataset",
    description: "Load an uploaded dataset and compute descriptive statistics and correlations.",
    parameters: {
      file_path: S.str("File name of an uploaded .csv, .tsv, .json or .xlsx file."),
      target_column: S.str("Column of primary interest for correlations. Pass an empty string to skip the targeted analysis."),
    },
    failure: (err) => `Dataset analysis failed: ${err.message}`,
    run: async (ctx, { file_path, target_column }) => {
      const file = Uploads.resolve(file_path);
      if (!file) throw new DocumentParseError("analyze_dataset", `no such file: ${file_path}`);
      const frame = await loadDataframe(file);
      ctx.attachments.push({ kind: "dataset", path: file.name, rows: frame.rows.length, cols: frame.columns.length });
      const numeric = numericColumns(frame);
      const lines = [`Dataset: ${file.name}`, `Shape: ${frame.rows.length.toLocaleString()} rows x ${frame.columns.length} columns`, `Columns: ${frame.columns.slice(0, 40).join(", ")}`, "", "## Missing values"];
      const missing = frame.columns.map((c) => [c, frame.rows.filter((r) => isMissing(r[c])).length]).filter(([, n]) => n > 0);
      if (!missing.length) lines.push("None.");
      else missing.forEach(([c, n]) => lines.push(`  ${c}: ${n.toLocaleString()} (${((n / frame.rows.length) * 100).toFixed(1)}%)`));
      if (numeric.length) {
        lines.push("\n## Descriptive statistics (numeric columns)");
        const stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
        const cols = numeric.map((c) => {
          const xs = colValues(frame, c), s = [...xs].sort((a, b) => a - b);
          return [xs.length, mean(xs), xs.length > 1 ? Math.sqrt(variance(xs)) : NaN, s[0], quantile(s, 0.25), quantile(s, 0.5), quantile(s, 0.75), s[s.length - 1]].map(r4);
        });
        lines.push(table(["", ...numeric], stats.map((st, i) => [st, ...cols.map((c) => c[i])])));
        if (numeric.length >= 2) {
          lines.push("\n## Correlation matrix (Pearson)");
          const corr = numeric.map((a) => numeric.map((b) => (a === b ? 1 : pearson(frame, a, b))));
          lines.push(table(["", ...numeric], numeric.map((a, i) => [a, ...corr[i].map((v) => (Number.isFinite(v) ? v.toFixed(3) : "NaN"))])));
          const pairs = [];
          numeric.forEach((a, i) => numeric.slice(i + 1).forEach((b, j) => { const v = corr[i][i + 1 + j]; if (Number.isFinite(v)) pairs.push([Math.abs(v), a, b, v]); }));
          pairs.sort((x, y) => y[0] - x[0]);
          if (pairs.length) {
            lines.push("\n## Strongest relationships");
            pairs.slice(0, 6).forEach(([, a, b, v]) => {
              const strength = Math.abs(v) >= 0.7 ? "strong" : Math.abs(v) >= 0.4 ? "moderate" : "weak";
              lines.push(`  ${a} ~ ${b}: r = ${v >= 0 ? "+" : ""}${v.toFixed(3)} (${strength}). Correlation is not causation; do not report it as an effect.`);
            });
          }
        }
      } else lines.push("\nNo numeric columns - only categorical analysis is possible.");
      if (target_column && numeric.includes(target_column)) {
        lines.push(`\n## Correlations with '${target_column}'`);
        numeric.filter((c) => c !== target_column).map((c) => [c, pearson(frame, target_column, c)])
          .filter(([, v]) => Number.isFinite(v)).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10)
          .forEach(([c, v]) => lines.push(`  ${c}: r = ${v >= 0 ? "+" : ""}${v.toFixed(3)}`));
      } else if (target_column) lines.push(`\nRequested target '${target_column}' is not a numeric column; skipped.`);
      lines.push("\n## Preview (first 8 rows)");
      lines.push(table(frame.columns, frame.rows.slice(0, 8).map((r) => frame.columns.map((c) => String(r[c] == null ? "" : r[c]).slice(0, 24)))));
      Log.event("tools.datasets", "dataset analysed", { path: file.name, rows: frame.rows.length, cols: frame.columns.length });
      return lines.join("\n");
    },
  });

  tool({
    name: "compare_groups",
    description: "Compare a numeric measure across groups with a Welch t-test and Cohen's d. Welch's test is used rather than Student's because equal variances are an assumption that real data rarely honours.",
    parameters: {
      file_path: S.str("File name of the uploaded dataset."),
      group_column: S.str("Categorical column defining the groups."),
      value_column: S.str("Numeric column to compare."),
    },
    failure: (err) => `Group comparison failed: ${err.message}`,
    run: async (ctx, { file_path, group_column, value_column }) => {
      const file = Uploads.resolve(file_path);
      if (!file) throw new DocumentParseError("compare_groups", `no such file: ${file_path}`);
      const frame = await loadDataframe(file);
      for (const c of [group_column, value_column]) if (!frame.columns.includes(c)) return `Column '${c}' not found. Available: ${JSON.stringify(frame.columns.slice(0, 30))}`;
      const groupsMap = new Map();
      for (const r of frame.rows) {
        const v = toNum(r[value_column]);
        if (!Number.isFinite(v) || isMissing(r[group_column])) continue;
        const g = String(r[group_column]);
        if (!groupsMap.has(g)) groupsMap.set(g, []);
        groupsMap.get(g).push(v);
      }
      const groups = [...groupsMap.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).filter(([, v]) => v.length >= 2);
      if (groups.length < 2) return `Need at least two groups with 2+ observations; found ${groups.length}.`;
      const lines = [`Comparing '${value_column}' across '${group_column}' (${groups.length} groups)\n`];
      groups.forEach(([n, v]) => lines.push(`  ${n}: n=${v.length}, mean=${mean(v).toFixed(4)}, sd=${Math.sqrt(variance(v)).toFixed(4)}`));
      if (groups.length === 2) {
        const [[na, a], [nb, b]] = groups;
        const diff = mean(a) - mean(b);
        const va = variance(a), vb = variance(b);
        const se = Math.sqrt(va / a.length + vb / b.length);
        const t = se > 0 ? diff / se : 0;
        const dfDen = (va / a.length) ** 2 / (a.length - 1) + (vb / b.length) ** 2 / (b.length - 1);
        const df = dfDen > 0 ? (va / a.length + vb / b.length) ** 2 / dfDen : 1;
        const pooled = Math.sqrt(((a.length - 1) * va + (b.length - 1) * vb) / (a.length + b.length - 2));
        const d = pooled > 0 ? diff / pooled : 0;
        const p = twoSidedP(t, df);
        const mag = Math.abs(d) < 0.2 ? "negligible" : Math.abs(d) < 0.5 ? "small" : Math.abs(d) < 0.8 ? "medium" : "large";
        lines.push(`\nWelch's t-test: t(${df.toFixed(1)}) = ${t.toFixed(3)}, p = ${p.toPrecision(5)} (two-sided)\nMean difference (${na} - ${nb}): ${diff >= 0 ? "+" : ""}${diff.toFixed(4)}\nCohen's d = ${d >= 0 ? "+" : ""}${d.toFixed(3)} (${mag} effect)\n\nReport the effect size alongside p: with a large n, a trivial difference will still reach significance.`);
      } else {
        lines.push("\nMore than two groups: pairwise t-tests would need a multiple-comparison correction. Use run_python_analysis for ANOVA.");
      }
      Log.event("tools.datasets", "groups compared", { group_column, value_column });
      return lines.join("\n");
    },
  });

  tool({
    name: "format_citations",
    description: "Generate verified citations for papers, in a named style. Citations are built from stored corpus metadata, so any id not in the corpus is reported as unverified rather than invented.",
    parameters: {
      paper_ids: S.arr(S.str(), "Corpus ids to cite. Pass an empty list to cite the whole corpus."),
      style: S.enm(VALID_STYLES, 'One of "APA", "IEEE", "MLA", "Nature".'),
    },
    run: async (ctx, { paper_ids, style }) => {
      const bundle = buildCitationBundle(ctx, paper_ids, style);
      if (!bundle.citations.length) return `None of the requested ids are in the corpus. Unverified: ${JSON.stringify(bundle.unverified_ids)}. Do not cite them.`;
      const lines = [`${bundle.citations.length} verified citations in ${bundle.style} style:\n`];
      bundle.citations.forEach((c) => lines.push(`[${c.paper_id}]\n  in-text:   ${c.in_text}\n  reference: ${c.full_reference}\n  bibtex key: ${c.bibtex_key}`));
      if (bundle.unverified_ids.length) lines.push(`\nUNVERIFIED - not in corpus, do not cite: ${JSON.stringify(bundle.unverified_ids)}`);
      Log.event("tools.citations", "citations formatted", { style: bundle.style, verified: bundle.citations.length, unverified: bundle.unverified_ids.length });
      lines.push(`\nBibTeX:\n${bundle.bibtex_file}`);
      return lines.join("\n");
    },
  });

  tool({
    name: "verify_citations",
    description: "Check whether cited ids exist in the retrieved corpus.",
    parameters: { paper_ids: S.arr(S.str(), "The ids appearing in a draft, to be checked before publication.") },
    run: async (ctx, { paper_ids }) => {
      const known = ctx.resolveCitations(paper_ids), missing = ctx.unknownCitations(paper_ids);
      const lines = [`Checked ${paper_ids.length} ids: ${known.length} verified, ${missing.length} unverified.`];
      known.forEach((p) => lines.push(`  OK   [${p.paper_id}] ${p.title.slice(0, 70)} (${p.year})`));
      missing.forEach((id) => lines.push(`  FAIL [${id}] not in corpus - remove it or retrieve the source`));
      return lines.join("\n");
    },
  });

  tool({
    name: "remember_finding",
    description: "Save a durable fact that future research sessions should know. Use sparingly, for conclusions that generalise beyond this run - a confirmed finding, a recurring methodological pitfall, or a stated user preference.",
    parameters: {
      kind: S.enm(["finding", "question", "preference", "critique"], 'One of "finding", "question", "preference", "critique".'),
      topic: S.str('A short subject label used for grouping, e.g. "GNN expressivity".'),
      content: S.str("The fact itself, self-contained and understandable months later."),
    },
    run: async (ctx, { kind, topic, content }) => {
      const k = ["finding", "question", "preference", "critique"].includes(kind) ? kind : "finding";
      const id = await LongTermMemory.remember({ session_id: ctx.session_id, run_id: ctx.run_id, kind: k, topic, content });
      return `Stored long-term memory #${id} (${k}: ${topic}).`;
    },
  });

  tool({
    name: "recall_memory",
    description: "Search everything Athena has learned in previous sessions. Call this at the start of a run: prior work on the same topic saves the whole pipeline from rediscovering what is already known.",
    parameters: {
      query: S.str("What you want to remember, in natural language."),
      limit: S.int("Maximum memories to return (1-10)."),
    },
    run: async (ctx, { query, limit }) => {
      const k = Math.max(1, Math.min(parseInt(limit || 5, 10) || 5, 10));
      const records = (await LongTermMemory.recall(query, { k })).filter((r) => r.score > 0);
      if (!records.length) return "No prior memories match. This appears to be a new topic.";
      return [`Recalled ${records.length} prior memories:`, ...records.map((r) =>
        `- [${r.kind}] ${r.topic} (relevance ${r.score.toFixed(2)}, from ${r.created_at.slice(0, 10)}): ${r.content}`)].join("\n");
    },
  });

  tool({
    name: "save_research_artifact",
    description: "Write a research artefact to the artefact store. Requires human approval.",
    parameters: {
      filename: S.str('Base name without a path, e.g. "gnn_expressivity_report".'),
      content: S.str("The full Markdown or JSON body to persist."),
      artifact_type: S.enm(["report", "design", "synthesis", "notes"], 'One of "report", "design", "synthesis", "notes".'),
    },
    // Static gate, once per run: an approved save is not a new question.
    needsApproval: (ctx) => {
      if (!Settings.get().require_human_approval) return false;
      return !ctx.approvals.some((a) => a.tool_name === "save_research_artifact" && a.decision === "approved");
    },
    run: async (ctx, { filename, content, artifact_type }) => {
      const stem = String(filename || "").split("").filter((ch) => /[A-Za-z0-9._-]/.test(ch)).join("").replace(/^\.+|\.+$/g, "") || "artifact";
      const ext = /^\s*[{[]/.test(content) ? ".json" : ".md";
      const previous = ctx.saved_artifacts[artifact_type];
      const stamp = new Date().toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
      const record = {
        run_id: ctx.run_id, artifact_type, content,
        filename: previous && previous.filename.endsWith(ext) ? previous.filename : `${stamp}_${stem}${ext}`,
        meta: {
          run_id: ctx.run_id, session_id: ctx.session_id, research_question: ctx.research_question,
          artifact_type, created_at: new Date().toISOString(), corpus_size: Object.keys(ctx.corpus).length,
          reflection_rounds: ctx.reflection_round, approvals: ctx.approvals,
        },
      };
      if (previous && previous.filename.endsWith(ext)) record.id = previous.id;
      const id = await DB.put("artifacts", record);
      ctx.saved_artifacts[artifact_type] = { id, filename: record.filename };
      Log.event("tools.artifacts", "artifact saved", { path: record.filename, artifact_type, bytes: new Blob([content]).size });
      return `Saved ${artifact_type} to ${record.filename} (${content.length.toLocaleString()} chars), with provenance sidecar.`;
    },
  });

  const GROUPS = {
    LITERATURE_TOOLS: ["search_literature", "list_corpus", "get_paper_details", "commission_deep_sweep"],
    DOCUMENT_TOOLS: ["ingest_pdf", "describe_figure", "transcribe_audio"],
    RAG_TOOLS: ["index_corpus", "rag_search", "list_corpus"],
    ANALYSIS_TOOLS: ["statistical_power_analysis", "run_python_analysis", "analyze_dataset", "compare_groups"],
    CITATION_TOOLS: ["verify_citations", "format_citations"],
    MEMORY_TOOLS: ["remember_finding", "recall_memory"],
    ARTIFACT_TOOLS: ["save_research_artifact"],
  };

  Object.assign(A, {
    OpenAI, Uploads, Tools, ToolGroups: GROUPS, VectorStore, LongTermMemory, Sandbox,
    Lit: { SOURCES, parseArxiv, parseOpenalex, parseCrossref, searchLiterature },
    Cite: { buildCitationBundle, renderReference, inText, bibtexKey, VALID_STYLES },
    Stats: { invNorm, requiredSampleSize, twoSidedP, betai },
    Docs: { ingestPdfFile, extractPdfText, deriveMetadata, slug, chunkText },
    Embed: { hashedEmbedding, features },
  });
})((window.Athena = window.Athena || {}));
