/* Athena - orchestration: the eight-stage research pipeline.
 *
 * Browser port of pipeline.py. The *sequence* is code; the *judgement* inside
 * each stage is the model's. Fabricated citations are blocked
 * deterministically, a correctable guardrail trip is re-asked with the bad ids
 * named, the run stops starting new stages at the spending cap, and a pending
 * human approval is checkpointed to IndexedDB so the decision can outlive the
 * page that asked for it.
 */
(function (A) {
  "use strict";
  const { Settings, Log, DB, ResearchContext, Stage } = A;
  const E = A.errors;

  const MAX_GUARDRAIL_RETRIES = 2;
  const SESSION_CHAR_BUDGET = 24000;
  const NL = "\n";

  const DOCUMENT_MODE_BLOCKED_TOOLS = new Set([
    "search_literature", "get_paper_details", "commission_deep_sweep", "ingest_pdf", "describe_figure",
    "transcribe_audio", "index_corpus", "rag_search", "list_corpus", "recall_memory",
  ]);

  const DOCUMENT_MODE_RULES = `## Evidence rules for this run (they override any conflicting instruction)
This study is confined to the source document(s) the user supplied. Their full
text is reproduced below and is the ONLY evidence base.
- Every factual statement - methods, datasets, numbers, comparisons, limitations -
  must come from the source text. Quote figures exactly as printed (e.g. an AUC of
  "0.996 +/- 0.002") and say where they come from (Table 2, Section 6.1.2, Fig. 4).
- Cite the source by its paper_id in square brackets, e.g. [{paper_id}]. The
  source's own bibliography entries (e.g. "[4]") are not corpus entries: mention
  them as "ref. [4] in the source", never as a paper_id.
- Do not bring in results from outside knowledge. When the source does not report
  something, say it is not reported rather than filling the gap.
- Web search and retrieval tools are disabled for this run on purpose: the complete
  text is already in front of you, so read it directly.`;

  const DOCUMENT_REPORT_STRUCTURE = `## Report structure for this run (replaces the default section list)
Write these sections, in this order, as a study of the question against the source:
  1. Introduction - the question, the source paper (title, authors, venue) and why
     the problem matters.
  2. Method Overview - what the proposed approach is and how it works: inputs, the
     architecture, each loss or component and what it is for, the training data.
  3. Evaluation Setup - training and test datasets, baselines, metrics, how the
     experimental conditions are defined and binned, how uncertainty is estimated.
  4. Results: Primary Evidence - the quantitative evidence that answers the
     question, organised by experimental condition. Include Markdown tables that
     reproduce the source's key numbers exactly, naming the table they came from.
  5. Results: Secondary Evidence and Trade-offs - ablations, quality/realism
     evidence, comparisons with alternative approaches, supplementary statistics.
  6. Answer to the Research Question - a direct, evidence-weighted verdict: where the
     approach helps, where it is neutral, where it is worse, and why (per the source).
  7. Research Gaps - from the gap analysis.
  8. Proposed Hypotheses - with scores and the recommendation.
  9. Proposed Experiment - including the computed sample size.
 10. Limitations - of the source's evidence and of this single-source analysis.
 11. Future Work.

Length: each section body up to 350 words (tables do not count); whole report
under 3,500 words. Every key_findings entry must carry a concrete number from the
source. The references field is the verified bibliography, verbatim.`;

  // ---------------------------------------------------------------------
  // Hooks (hooks.py / AthenaRunHooks): trace, handoff chain, tool usage
  // ---------------------------------------------------------------------

  class AthenaRunHooks {
    constructor(emit) { this.emit = emit; this.chain = []; this.usage = {}; }
    agentStart(agent) {
      if (this.chain[this.chain.length - 1] !== agent.name) this.chain.push(agent.name);
      Log.event("hooks", "agent start", { agent: agent.name });
      this.emit("agent", { agent: agent.name });
    }
    agentEnd(agent) { Log.event("hooks", "agent end", { agent: agent.name }); }
    toolStart(agent, tool, args) {
      this.usage[tool] = (this.usage[tool] || 0) + 1;
      Log.event("hooks", "tool start", { agent: agent.name, tool, args: JSON.stringify(args || {}).slice(0, 300) });
      this.emit("tool", { agent: agent.name, tool, phase: "start", args });
    }
    toolEnd(agent, tool, result) {
      Log.event("hooks", "tool end", { agent: agent.name, tool, result_chars: String(result || "").length, preview: String(result || "").slice(0, 240) });
      this.emit("tool", { agent: agent.name, tool, phase: "end", result });
    }
    llmEnd(agent, model, usage) {
      Log.event("hooks", "llm end", { agent: agent.name, model, input_tokens: usage.prompt_tokens || 0, output_tokens: usage.completion_tokens || 0 });
      this.emit("cost", {});
    }
    handoffChain() { return this.chain.slice(); }
    toolUsage() { return Object.assign({}, this.usage); }
  }

  // ---------------------------------------------------------------------
  // Offline report composition (used by the offline Scientific Writer)
  // ---------------------------------------------------------------------

  function composeOfflineReport(ctx) {
    const q = (ctx.plan && ctx.plan.research_question) || ctx.research_question;
    const corpusN = Object.keys(ctx.corpus).length;
    const syn = ctx.synthesis, gaps = ctx.gap_analysis, hyp = ctx.hypotheses, exp = ctx.experiment;
    const rec = hyp && (hyp.hypotheses.find((h) => h.hypothesis_id === hyp.recommended_hypothesis_id) || hyp.hypotheses[0]);
    const cite = (ids) => (ids && ids.length ? ` [${ids.join("; ")}]` : "");
    const sections = [
      { heading: "Introduction", body_markdown: `This dossier examines: **${q}**\n\n${ctx.plan ? ctx.plan.interpretation : ""}` },
      { heading: "Related Work", body_markdown: syn ? syn.key_claims.map((c) => `- ${c.statement}${cite(c.supporting_paper_ids)} *(confidence: ${c.confidence})*`).join("\n") || "No claims could be grounded." : "Synthesis did not complete." },
      { heading: "Research Gaps", body_markdown: gaps ? gaps.gaps.map((g) => `- **${g.gap_id}** (${g.severity}, ${g.gap_type}, tractability ${g.tractability}/10): ${g.description}${cite(g.evidence_paper_ids)}`).join("\n") + `\n\nPriority: **${gaps.priority_gap_id}** — ${gaps.prioritisation_rationale}` : "Gap analysis did not complete." },
      { heading: "Proposed Hypotheses", body_markdown: hyp ? hyp.hypotheses.map((h) => `- **${h.hypothesis_id}**: ${h.statement} (novelty ${h.novelty_score}, feasibility ${h.feasibility_score}, impact ${h.impact_score})${cite(h.grounded_in_paper_ids)}`).join("\n") + `\n\nRecommended: **${hyp.recommended_hypothesis_id}** — ${hyp.selection_rationale}` : "Hypothesis generation did not complete." },
      { heading: "Proposed Experiment", body_markdown: exp ? `**${exp.title}**\n\n${exp.methodology}\n\nComputed sample size: **${exp.power_analysis.required_sample_size} per group** (d = ${exp.power_analysis.effect_size}, α = ${exp.power_analysis.alpha}, power = ${exp.power_analysis.power}). ${exp.power_analysis.justification}` : "Experiment design did not complete." },
      { heading: "Limitations", body_markdown: [
        `The corpus holds ${corpusN} papers and was built from abstracts, not full texts.`,
        "Retrieval is lexical, so a relevant paper with no shared vocabulary may have been missed.",
        "This dossier was produced in offline mode: stage outputs are deterministic stand-ins, not model judgements.",
        "Hypotheses are proposed, not tested; the experiment is designed, not run.",
      ].concat(ctx.errors.slice(0, 5).map((e) => `Run error: ${e.replace(/^\S+\s/, "")}`)).map((l) => `- ${l}`).join("\n") },
      { heading: "Future Work", body_markdown: "Re-run in live mode with an OpenAI API key for model-judged synthesis, broaden search terms, and ingest full texts of the most relevant papers." },
    ];
    const keyFindings = syn ? syn.key_claims.slice(0, 3).map((c) => `${c.statement}${cite(c.supporting_paper_ids)}`) : [];
    const exec = `Athena reviewed ${corpusN} papers on "${q}". ` +
      (syn ? `${syn.key_claims.length} evidence-anchored claims were extracted, ` : "") +
      (gaps ? `${gaps.gaps.length} research gap(s) were identified (priority ${gaps.priority_gap_id}), ` : "") +
      (rec ? `and hypothesis ${rec.hypothesis_id} is recommended for testing. ` : "") +
      (exp ? `A two-arm experiment needs ${exp.power_analysis.required_sample_size} units per group to detect d = ${exp.power_analysis.effect_size} at 80% power. ` : "") +
      "This is an offline-mode dossier: retrieval, citation verification and the power calculation are real, but the analysis text is deterministic rather than model-written, so treat its conclusions as a structured starting point.";
    const report = {
      title: `Research Dossier: ${q.replace(/\?$/, "")}`,
      research_question: q,
      executive_summary: exec,
      sections,
      key_findings: keyFindings,
      limitations: [`Corpus of ${corpusN} papers built from abstracts`, "Offline mode: no model judgement", "Lexical retrieval"],
      references: ctx.citations ? ctx.citations.citations.map((c) => c.full_reference) : [],
      confidence: "low",
      word_count: 0,
    };
    report.word_count = reportMarkdown(report).split(/\s+/).filter(Boolean).length;
    return report;
  }

  function reportMarkdown(report, ctx) {
    const lines = [`# ${report.title}`, "", `**Research question:** ${report.research_question}`, "", "## Executive summary", "", report.executive_summary, ""];
    if (report.key_findings && report.key_findings.length) lines.push("## Key findings", "", ...report.key_findings.map((k) => `- ${k}`), "");
    for (const s of report.sections || []) lines.push(`## ${s.heading}`, "", s.body_markdown, "");
    if (report.references && report.references.length) lines.push("## References", "", ...report.references.map((r) => `- ${r}`), "");
    lines.push(`*Overall confidence: ${report.confidence}*`);
    if (ctx) lines.push("", `*Generated by Athena (browser edition) · run ${ctx.run_id} · ${new Date().toISOString().slice(0, 10)}*`);
    return lines.join("\n");
  }

  // ---------------------------------------------------------------------
  // The pipeline
  // ---------------------------------------------------------------------

  class ResearchPipeline {
    constructor({ sessionId = "default", runId = null, approvalHandler = null, onEvent = null } = {}) {
      this.settings = Settings.get();
      this.run_id = runId || A.newRunId();
      this.session_id = sessionId;
      Log.configure(this.run_id);
      this.team = A.buildTeam();
      this.approvalHandler = approvalHandler || (async () => ({ approved: true, note: "auto-approved (no interactive handler configured)" }));
      this.listener = onEvent || (() => {});
      this.hooks = new AthenaRunHooks((type, data) => this.emit(type, data));
      this.stages = [];
      this.session = [];
      this.ctx = null;
      this.status = "idle";
      this.stopped = false;
      this.live = Settings.live;
    }

    emit(type, data) { try { this.listener(type, Object.assign({ pipeline: this }, data || {})); } catch (e) { /* UI errors never break a run */ } }
    toolAllowed(name) { return !(this.ctx && this.ctx.source_mode === "documents" && DOCUMENT_MODE_BLOCKED_TOOLS.has(name)); }
    approve(item) { return this.approvalHandler(item); }
    sessionMessages() { return this.session.slice(); }
    _remember(prompt, output) {
      this.session.push({ role: "user", content: prompt.slice(0, 6000) }, { role: "assistant", content: JSON.stringify(output).slice(0, 6000) });
      while (this.session.reduce((n, m) => n + m.content.length, 0) > SESSION_CHAR_BUDGET && this.session.length > 2) this.session.splice(0, 2);
    }

    // -- one stage --------------------------------------------------------

    async _stage(stage, agentKey, prompt, ctx, { useSession = true, resume = null } = {}) {
      const agent = this.team[agentKey];
      ctx.stage = stage;
      const started = performance.now();
      Log.event("pipeline", `stage start: ${stage}`, { stage, agent: agent.name });
      this.emit("stage", { stage, agent: agent.name, status: "running" });

      const spent = ctx.costs.estimatedUsd();
      const cap = Settings.get().max_run_usd;
      if (this.live && cap > 0 && spent >= cap) {
        return this._fail(stage, agent, ctx, started, `budget reached: $${spent.toFixed(2)} spent of the $${cap.toFixed(2)} run cap; ${stage} was not started`);
      }
      if (ctx.source_mode === "documents") useSession = false;

      let attemptPrompt = prompt;
      for (let attempt = 0; attempt <= MAX_GUARDRAIL_RETRIES; attempt++) {
        try {
          const output = await A.runAgent(this, agent, attemptPrompt, ctx, { useSession, resume: attempt === 0 ? resume : null });
          const elapsed = (performance.now() - started) / 1000;
          this.stages.push({ stage, agent: agent.name, ok: true, detail: `produced ${agent.output}`, duration_s: elapsed });
          Log.event("pipeline", `stage ok: ${stage}`, { stage, agent: agent.name, duration_s: +elapsed.toFixed(2) });
          if (useSession) this._remember(prompt, output);
          this.emit("stage", { stage, agent: agent.name, status: "ok" });
          return output;
        } catch (err) {
          if (err instanceof E.ApprovalDeferred) throw err;
          if (err instanceof E.InputGuardrailTripwireTriggered) {
            ctx.input_guardrail = err.info;
            this.stopped = true;
            return this._fail(stage, agent, ctx, started, `Input guardrail blocked the run: ${err.info.reasoning}`);
          }
          if (err instanceof E.OutputGuardrailTripwireTriggered) {
            const fabricated = err.info.fabricated || [];
            if (attempt < MAX_GUARDRAIL_RETRIES) {
              Log.event("pipeline", "citation guardrail tripped; re-asking", { stage, fabricated });
              this.emit("guardrail", { stage, fabricated });
              attemptPrompt = `${prompt}${NL}${NL}## Correction required${NL}Your previous attempt cited ids that are not in the corpus and was rejected: ${JSON.stringify(fabricated)}. Do not use them again. Cite only from this list, copying each id character for character:${NL}${JSON.stringify(Object.keys(ctx.corpus).sort())}`;
              continue;
            }
            return this._fail(stage, agent, ctx, started, `Output guardrail rejected ${agent.name}'s output: ${err.message}`);
          }
          if (err instanceof E.MaxTurnsExceeded) return this._fail(stage, agent, ctx, started, `${agent.name} exceeded the turn budget: ${err.message}`);
          if (err instanceof E.AthenaError) return this._fail(stage, agent, ctx, started, `${err.name}: ${err.message}`);
          return this._fail(stage, agent, ctx, started, `unexpected ${err.name || "Error"}: ${err.message}`);
        }
      }
      return null;
    }

    _fail(stage, agent, ctx, started, message) {
      const elapsed = (performance.now() - started) / 1000;
      ctx.recordError(message);
      this.stages.push({ stage, agent: agent.name, ok: false, detail: message, duration_s: elapsed });
      Log.error("pipeline", message, { stage, agent: agent.name });
      this.emit("stage", { stage, agent: agent.name, status: "failed", detail: message });
      return null;
    }

    // -- prompt helpers ----------------------------------------------------

    _citableIds(ctx) {
      const ids = Object.keys(ctx.corpus).sort();
      if (!ids.length) return "";
      return `\n\n## Citable paper_ids (the complete list)\nCite only these, copied character for character. Any other id - including one you recall from elsewhere - is treated as fabricated and the output is rejected:\n${JSON.stringify(ids)}`;
    }

    _sourceBlock(ctx) {
      const ids = Object.keys(ctx.source_texts).sort();
      const parts = [DOCUMENT_MODE_RULES.replace("{paper_id}", ids[0] || "upload:...")];
      for (const pid of ids) {
        const p = ctx.corpus[pid];
        const byline = p ? `${p.title} - ${p.authors.join(", ") || "authors not parsed"} (${p.year || "n.d."}), ${p.venue}` : "";
        parts.push(`## Source document [${pid}]\n${byline}\n<<<SOURCE TEXT START>>>\n${ctx.source_texts[pid]}\n<<<SOURCE TEXT END>>>`);
      }
      return `\n\n${parts.join("\n\n")}`;
    }

    _datasetNote() {
      const data = A.Uploads.list().filter((f) => /\.(csv|tsv|tab|json|xlsx|xls)$/i.test(f.name));
      return data.length ? `\n\nThe user attached these datasets; analyse them with \`analyze_dataset\` / \`compare_groups\` where they bear on the effect size:\n${data.map((f) => `- ${f.name}`).join("\n")}` : "";
    }

    _buildReportBrief(ctx, question) {
      const documents = ctx.source_mode === "documents" && Object.keys(ctx.source_texts).length > 0;
      const cap = documents ? 12000 : 3000;
      const dump = (o) => JSON.stringify(o, null, 2).slice(0, cap);
      const parts = [`Write the final research dossier for: ${question}`, ""];
      if (documents) parts.push(DOCUMENT_REPORT_STRUCTURE);
      if (ctx.literature) parts.push(`## Corpus\n${Object.keys(ctx.corpus).length} papers. Coverage: ${ctx.literature.coverage_assessment}`);
      if (ctx.synthesis) parts.push(`## Synthesis\n${dump(ctx.synthesis)}`);
      if (ctx.gap_analysis) parts.push(`## Gap analysis\n${dump(ctx.gap_analysis)}`);
      if (ctx.hypotheses) parts.push(`## Hypotheses\n${dump(ctx.hypotheses)}`);
      if (ctx.experiment) parts.push(`## Experiment design\n${dump(ctx.experiment)}`);
      if (ctx.citations) parts.push(`## Verified bibliography (use this verbatim for the references field; do not add to it)\n${ctx.citations.reference_list_markdown.slice(0, 3000)}`);
      if (Object.keys(ctx.corpus).length) parts.push(this._citableIds(ctx).trim());
      if (ctx.errors.length) parts.push(`## Run errors (disclose these in Limitations)\n${ctx.errors.slice(0, 10).map((e) => `- ${e}`).join("\n")}`);
      parts.push("\nBuild the references list from the corpus. Store the durable insight with `remember_finding`, then save the dossier with `save_research_artifact` - once. Then return the ResearchReport.");
      let brief = parts.join("\n\n");
      if (documents) brief += this._sourceBlock(ctx);
      return brief;
    }

    _chosenHypothesis(ctx) {
      if (!ctx.hypotheses || !ctx.hypotheses.hypotheses.length) return null;
      return ctx.hypotheses.hypotheses.find((h) => h.hypothesis_id === ctx.hypotheses.recommended_hypothesis_id) || ctx.hypotheses.hypotheses[0];
    }

    // -- the study, as resumable steps ---------------------------------------

    _webSteps(question) {
      const effective = () => (this.ctx.plan ? this.ctx.plan.research_question : question);
      return [
        { stage: Stage.PLANNING, run: async (ctx, resume) => {
          ctx.plan = await this._stage(Stage.PLANNING, "principal_investigator", `Plan a research programme for this question:\n\n${question}`, ctx, { resume });
        } },
        { stage: Stage.LITERATURE, run: async (ctx, resume) => {
          const pending = ctx.attachments.filter((a) => a.kind === "pending");
          const note = pending.length
            ? `\n\nThe user attached these files. Ingest the PDFs with \`ingest_pdf\`, analyse images with \`describe_figure\`, and transcribe audio with \`transcribe_audio\` before searching:\n${pending.map((a) => `- ${a.path}`).join("\n")}`
            : "";
          const keywords = ctx.plan ? ctx.plan.search_keywords.join(", ") : effective();
          ctx.literature = await this._stage(Stage.LITERATURE, "literature_review",
            `Build the evidence corpus for: ${effective()}\n\nSuggested search terms: ${keywords}\nRun several distinct searches across all three sources.${note}`, ctx, { resume });
          if (!Object.keys(ctx.corpus).length) {
            Log.event("pipeline", "aborting: corpus is empty after literature stage");
            ctx.recordError("No papers retrieved; downstream stages cannot be grounded.");
            this.stopped = true;
          }
        } },
        { stage: Stage.SYNTHESIS, run: async (ctx, resume) => {
          const subs = ctx.plan ? ctx.plan.sub_questions.map((q) => `- ${q}`).join("\n") : "";
          ctx.synthesis = await this._stage(Stage.SYNTHESIS, "synthesist",
            `Synthesise the evidence on: ${effective()}\n\nAddress each sub-question with its own retrieval:\n${subs}` + this._citableIds(ctx), ctx, { resume });
        } },
        { stage: Stage.GAP_ANALYSIS, run: async (ctx, resume) => {
          const digest = ctx.synthesis ? JSON.stringify(ctx.synthesis, null, 2).slice(0, 3000) : "(none)";
          ctx.gap_analysis = await this._stage(Stage.GAP_ANALYSIS, "gap_analyst",
            `Identify and prioritise the research gaps for: ${effective()}\n\nEvidence synthesis:\n${digest}\n\nProbe the corpus with the field's own terminology before calling anything a gap.` + this._citableIds(ctx), ctx, { resume });
        } },
        { stage: Stage.HYPOTHESIS, run: async (ctx, resume) => {
          let gaps = "";
          if (ctx.gap_analysis) {
            gaps = ctx.gap_analysis.gaps.map((g) => `- [${g.gap_id}] (${g.severity}/${g.gap_type}, tractability ${g.tractability}/10) ${g.title}: ${g.description}`).join("\n") + `\n\nPriority gap: ${ctx.gap_analysis.priority_gap_id}`;
          } else if (ctx.synthesis) gaps = ctx.synthesis.gaps.map((g) => `- ${g.gap}: ${g.why_it_matters}`).join("\n");
          ctx.hypotheses = await this._stage(Stage.HYPOTHESIS, "hypothesis_generator",
            `Generate hypotheses for: ${effective()}\n\nIdentified gaps:\n${gaps || "(none recorded - derive your own from the corpus)"}` + this._citableIds(ctx), ctx, { resume });
        } },
        { stage: Stage.EXPERIMENT, run: async (ctx, resume) => {
          const h = this._chosenHypothesis(ctx);
          if (!h) return;
          ctx.experiment = await this._stage(Stage.EXPERIMENT, "experiment_planner",
            `Design an experiment to test hypothesis ${h.hypothesis_id}:\n\nStatement: ${h.statement}\nNull: ${h.null_hypothesis}\nIVs: ${h.independent_variables.join(", ")}\nDVs: ${h.dependent_variables.join(", ")}\n\nCompute the sample size with the power analysis tool.` + this._datasetNote(), ctx, { resume });
        } },
        { stage: Stage.CITATIONS, run: async (ctx, resume) => {
          const ids = Object.keys(ctx.corpus).sort();
          ctx.citations = await this._stage(Stage.CITATIONS, "citation_manager",
            `Produce a verified ${Settings.get().citation_style} bibliography for this study.\n\nCorpus ids to cite (${ids.length}): ${JSON.stringify(ids)}\n\nVerify every id before including it.` + this._citableIds(ctx), ctx, { resume });
        } },
        { stage: Stage.REPORT, run: async (ctx, resume) => {
          ctx.report = await this._stage(Stage.REPORT, "scientific_writer", this._buildReportBrief(ctx, effective()), ctx, { useSession: false, resume });
        } },
      ];
    }

    _documentSteps(question) {
      const S = () => this._sourceBlock(this.ctx);
      const ids = () => Object.keys(this.ctx.source_texts).sort();
      return [
        { stage: "ingest", run: async (ctx) => {
          for (const att of ctx.attachments.filter((a) => a.kind === "pending")) {
            if (!att.path.toLowerCase().endsWith(".pdf")) { ctx.recordError(`${att.path}: only PDFs are used in document-grounded mode`); continue; }
            try {
              const { paper, chunks, body } = await A.Docs.ingestPdfFile(ctx, att.path);
              Log.event("pipeline", "source ingested", { paper_id: paper.paper_id, chars: body.length, chunks });
            } catch (err) { ctx.recordError(`could not read ${att.path}: ${err.message}`); }
          }
          ctx.attachments = ctx.attachments.filter((a) => a.kind !== "pending");
          if (!Object.keys(ctx.source_texts).length) {
            ctx.recordError("Document-grounded mode needs at least one readable PDF; none was ingested.");
            this.stopped = true;
          }
          this.emit("stage", { stage: "ingest", status: "ok" });
        } },
        { stage: Stage.PLANNING, run: async (ctx, resume) => {
          ctx.plan = await this._stage(Stage.PLANNING, "principal_investigator",
            `Plan a study of this research question:\n\n${question}\n\nScope the plan to what the source document(s) below can answer. Make the sub-questions specific to the source's methods, datasets, metrics and reported results. Search keywords should be the source's own terminology.` + S(), ctx, { resume });
        } },
        { stage: Stage.LITERATURE, run: async (ctx) => {
          const papers = ids().map((pid) => ctx.corpus[pid]).filter(Boolean);
          const pages = ctx.attachments.filter((a) => a.kind === "pdf").reduce((n, a) => n + (a.pages || 0), 0);
          const chars = Object.values(ctx.source_texts).reduce((n, t) => n + t.length, 0);
          ctx.literature = {
            query_terms_used: [], papers, total_found: papers.length,
            coverage_assessment: `Document-grounded study of ${papers.length} user-supplied source(s) (${pages} pages, ${chars.toLocaleString()} characters of full text, all of it read by every stage). External literature search was disabled by design, so the analysis reflects the source's own evidence and framing, including how it characterises prior work.`,
            notable_gaps: ["No independent replication or third-party evaluation of the source's results was consulted (single-source scope)."],
          };
          this.stages.push({ stage: Stage.LITERATURE, agent: "document ingestion", ok: true, detail: `ingested ${papers.length} source(s)`, duration_s: 0 });
          this.emit("stage", { stage: Stage.LITERATURE, agent: "document ingestion", status: "ok" });
        } },
        { stage: Stage.SYNTHESIS, run: async (ctx, resume) => {
          const plan = ctx.plan;
          const subs = plan ? plan.sub_questions.map((q) => `- ${q}`).join("\n") : "";
          const note = plan ? `\nPlanner's interpretation: ${plan.interpretation}\n` : "";
          ctx.synthesis = await this._stage(Stage.SYNTHESIS, "synthesist",
            `Synthesise what the source establishes about:\n${question}\n${note}\nAddress each sub-question with specific evidence, including the exact numbers the source reports:\n${subs}\n\nKey claims should be precise and quantitative. Mark a claim contested when the source's own results cut both ways (e.g. one metric or condition improves while another degrades).` + S(), ctx, { resume });
        } },
        { stage: Stage.GAP_ANALYSIS, run: async (ctx, resume) => {
          const digest = ctx.synthesis ? JSON.stringify(ctx.synthesis, null, 2) : "(none)";
          ctx.gap_analysis = await this._stage(Stage.GAP_ANALYSIS, "gap_analyst",
            `Identify and prioritise the research gaps for:\n${question}\n\nA gap here is something the source leaves unestablished: limitations it states, conditions it did not test, results that are mixed or not statistically separable, and threats to generalisation.\n\nEvidence synthesis:\n${digest}` + S(), ctx, { resume });
        } },
        { stage: Stage.HYPOTHESIS, run: async (ctx, resume) => {
          const gaps = ctx.gap_analysis ? ctx.gap_analysis.gaps.map((g) => `- [${g.gap_id}] (${g.severity}/${g.gap_type}, tractability ${g.tractability}/10) ${g.title}: ${g.description}`).join("\n") + `\n\nPriority gap: ${ctx.gap_analysis.priority_gap_id}` : "";
          ctx.hypotheses = await this._stage(Stage.HYPOTHESIS, "hypothesis_generator",
            `Generate follow-up hypotheses for:\n${question}\n\nIdentified gaps:\n${gaps || "(none recorded - derive them from the source)"}\n\nEach hypothesis must be testable with the datasets, baselines and metrics the source describes, and its rationale must point to specific results.` + S(), ctx, { resume });
        } },
        { stage: Stage.EXPERIMENT, run: async (ctx, resume) => {
          const h = this._chosenHypothesis(ctx);
          if (!h) return;
          ctx.experiment = await this._stage(Stage.EXPERIMENT, "experiment_planner",
            `Design an experiment to test hypothesis ${h.hypothesis_id}:\n\nStatement: ${h.statement}\nNull: ${h.null_hypothesis}\nIVs: ${h.independent_variables.join(", ")}\nDVs: ${h.dependent_variables.join(", ")}\n\nReuse the source's datasets, baselines and evaluation protocol where appropriate. Base the effect-size assumption on differences the source actually reports, and compute the sample size with the power analysis tool.` + this._datasetNote() + S(), ctx, { resume });
        } },
        { stage: Stage.CITATIONS, run: async (ctx, resume) => {
          ctx.citations = await this._stage(Stage.CITATIONS, "citation_manager",
            `Produce a verified ${Settings.get().citation_style} bibliography for this study.\n\nCorpus ids to cite (${ids().length}): ${JSON.stringify(ids())}\n\nVerify every id once, then format them.` + this._citableIds(ctx), ctx, { resume });
        } },
        { stage: Stage.REPORT, run: async (ctx, resume) => {
          ctx.report = await this._stage(Stage.REPORT, "scientific_writer", this._buildReportBrief(ctx, question), ctx, { useSession: false, resume });
        } },
      ];
    }

    _steps() {
      return this.ctx.source_mode === "documents" ? this._documentSteps(this.ctx.research_question) : this._webSteps(this.ctx.research_question);
    }

    /** Execute all eight stages and return the populated context. */
    async runStudy(question, { attachments = [], sourceMode = "web" } = {}) {
      this.ctx = new ResearchContext(this.run_id, this.session_id, question);
      this.ctx.source_mode = sourceMode;
      for (const name of attachments) this.ctx.attachments.push({ kind: "pending", path: name });
      Log.event("pipeline", "study start", { question, session: this.session_id, source_mode: sourceMode, mode: this.live ? "live" : "offline" });
      if (!this.live) Log.event("pipeline", "running in OFFLINE mode - model calls are deterministic stand-ins, tools are real");
      return this._execute(0, null);
    }

    async _execute(from, resume) {
      const ctx = this.ctx;
      const steps = this._steps();
      this.status = "running";
      this.emit("status", { status: "running" });
      for (let i = from; i < steps.length; i++) {
        if (this.stopped) break;
        try {
          await steps[i].run(ctx, i === from ? resume : null);
        } catch (err) {
          if (err instanceof E.ApprovalDeferred) {
            await this._checkpoint(i, err);
            this.status = "awaiting_approval";
            this.emit("status", { status: "awaiting_approval", tool: err.tool });
            return ctx;
          }
          throw err;
        }
        this.emit("progress", {});
      }
      this.status = ctx.report ? "complete" : "incomplete";
      Log.event("pipeline", "study complete", ctx.summary());
      await this.saveRun();
      await DB.delete("checkpoints", this.run_id);
      this.emit("status", { status: this.status });
      return ctx;
    }

    async _checkpoint(step, err) {
      const record = {
        run_id: this.run_id, session_id: this.session_id, created_at: new Date().toISOString(),
        question: this.ctx.research_question, step, tool: err.tool, state: err.state, item: err.item || null,
        ctx: this.ctx.toDict(), session: this.session, stages: this.stages, trace: Log.events,
        hooks: { chain: this.hooks.chain, usage: this.hooks.usage }, live: this.live,
      };
      await DB.put("checkpoints", record);
      Log.event("pipeline", "approval deferred; run checkpointed", { tool: err.tool, step });
      await this.saveRun("awaiting_approval");
    }

    /** Resume a run that was checkpointed while awaiting human approval. */
    static async resume(runId, decision, options = {}) {
      const cp = await DB.get("checkpoints", runId);
      if (!cp) throw new E.AthenaError(`No checkpoint found for run ${runId}`);
      const pipeline = new ResearchPipeline(Object.assign({}, options, { runId, sessionId: cp.session_id }));
      Log.events = cp.trace || [];
      pipeline.ctx = ResearchContext.fromDict(cp.ctx);
      pipeline.session = cp.session || [];
      pipeline.stages = cp.stages || [];
      pipeline.hooks.chain = (cp.hooks && cp.hooks.chain) || [];
      pipeline.hooks.usage = (cp.hooks && cp.hooks.usage) || {};
      if (cp.live && !Settings.live) throw new E.AthenaError("This run was started in live mode; add your OpenAI API key in Settings to resume it.");
      pipeline.live = Settings.live;
      Log.event("pipeline", "resuming from checkpoint", { tool: cp.tool, decision: decision.approved ? "approved" : "rejected" });
      const resume = Object.assign({}, cp.state || {}, { tool: cp.tool, decision });
      return { pipeline, ctx: await pipeline._execute(cp.step, resume) };
    }

    // -- reporting ---------------------------------------------------------

    runManifest() {
      const ctx = this.ctx;
      return {
        run_id: this.run_id,
        session_id: this.session_id,
        mode: this.live ? "live" : "offline",
        source_mode: ctx.source_mode,
        summary: ctx.summary(),
        stages: this.stages.map((s) => ({ stage: s.stage, agent: s.agent, ok: s.ok, detail: String(s.detail).slice(0, 300), duration_s: Math.round(s.duration_s * 100) / 100 })),
        handoffs: this.hooks.handoffChain(),
        tool_usage: this.hooks.toolUsage(),
        approvals: ctx.approvals,
        errors: ctx.errors,
        overclaim_flags: ctx.overclaim_flags,
      };
    }

    async saveRun(status) {
      const ctx = this.ctx;
      await DB.put("runs", {
        run_id: this.run_id,
        question: ctx.research_question,
        created_at: ctx.started_at,
        updated_at: new Date().toISOString(),
        status: status || this.status,
        mode: this.live ? "live" : "offline",
        manifest: this.runManifest(),
        ctx: ctx.toDict(),
        trace: Log.events,
      });
    }
  }

  Object.assign(A, { ResearchPipeline, composeOfflineReport, reportMarkdown, DOCUMENT_MODE_BLOCKED_TOOLS });
})((window.Athena = window.Athena || {}));
