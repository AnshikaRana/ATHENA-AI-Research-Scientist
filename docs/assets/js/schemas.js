/* Athena - structured hand-off schemas.
 *
 * Browser port of athena/schemas.py. Every agent returns a typed object that
 * is validated by OpenAI structured outputs in strict mode: every object is
 * closed (additionalProperties: false) and every field is required, exactly as
 * the Python runtime's AgentOutputSchema enforces.
 */
(function (A) {
  "use strict";

  const S = {
    str: (d) => ({ type: "string", description: d }),
    int: (d) => ({ type: "integer", description: d }),
    num: (d) => ({ type: "number", description: d }),
    bool: (d) => ({ type: "boolean", description: d }),
    arr: (items, d) => ({ type: "array", items, description: d }),
    enm: (values, d) => ({ type: "string", enum: values, description: d }),
    obj: (props) => ({
      type: "object",
      properties: props,
      required: Object.keys(props),
      additionalProperties: false,
    }),
  };

  const Stage = {
    PLANNING: "planning",
    LITERATURE: "literature",
    SYNTHESIS: "synthesis",
    GAP_ANALYSIS: "gap_analysis",
    HYPOTHESIS: "hypothesis",
    EXPERIMENT: "experiment",
    INTERPRETATION: "interpretation",
    REVIEW: "review",
    CITATIONS: "citations",
    REPORT: "report",
  };
  const CONFIDENCE = ["low", "medium", "high"];

  const Paper = S.obj({
    paper_id: S.str("Stable id, e.g. arXiv id or DOI."),
    title: S.str(),
    authors: S.arr(S.str(), "Author display names, may be truncated."),
    year: S.int("Publication year; 0 when unknown."),
    venue: S.str("Journal or conference; empty string when unknown."),
    abstract: S.str("Abstract text; empty string when unavailable."),
    url: S.str("Canonical link to the work."),
    source: S.enm(["arxiv", "openalex", "crossref", "user_upload"], "Which connector produced this record."),
    citation_count: S.int("Citations known to the source; 0 if absent."),
  });

  const ResearchPlan = S.obj({
    research_question: S.str("The refined, sharpened question."),
    interpretation: S.str("How the PI understands the question, including scope boundaries."),
    sub_questions: S.arr(S.str(), "2-5 decomposed sub-questions that together answer the main one."),
    search_keywords: S.arr(S.str(), "4-10 precise search terms for the literature databases."),
    steps: S.arr(
      S.obj({
        order: S.int("1-based execution order."),
        stage: S.enm(Object.values(Stage), "Which pipeline stage this step belongs to."),
        objective: S.str("What this step must achieve, one sentence."),
        rationale: S.str("Why this step is necessary for the question."),
      }),
      "The ordered plan of work."
    ),
    success_criteria: S.arr(S.str(), "Observable conditions that mean the research succeeded."),
    out_of_scope: S.arr(S.str(), "Explicitly excluded topics, to stop the run drifting."),
  });

  const LiteratureReview = S.obj({
    query_terms_used: S.arr(S.str(), "Terms actually searched."),
    papers: S.arr(Paper, "Deduplicated, relevance-ranked papers."),
    total_found: S.int("Count before deduplication and ranking."),
    coverage_assessment: S.str("Honest note on how well the corpus covers the question."),
    notable_gaps: S.arr(S.str(), "Topics the search could not find evidence for."),
  });

  const KnowledgeSynthesis = S.obj({
    summary: S.str("A tight narrative of the state of the field."),
    key_claims: S.arr(
      S.obj({
        statement: S.str("The claim, stated precisely."),
        supporting_paper_ids: S.arr(S.str(), "paper_id values from the corpus that support this claim."),
        confidence: S.enm(CONFIDENCE),
        contested: S.bool("True when sources disagree; the disagreement goes in the statement."),
      })
    ),
    consensus_points: S.arr(S.str()),
    disagreements: S.arr(S.str()),
    gaps: S.arr(
      S.obj({
        gap: S.str("What the field has not established."),
        why_it_matters: S.str(),
        evidence_paper_ids: S.arr(S.str(), "Papers whose limitations sections point at this gap."),
      })
    ),
    papers_cited: S.arr(S.str(), "All paper_ids actually used."),
  });

  const GapAnalysis = S.obj({
    gaps: S.arr(
      S.obj({
        gap_id: S.str("Short slug, e.g. 'G1'."),
        title: S.str("The gap in one line."),
        description: S.str("What specifically the field has not established."),
        severity: S.enm(["foundational", "significant", "incremental"]),
        gap_type: S.enm(["empirical", "theoretical", "methodological", "population", "replication"], "What kind of missing knowledge this is."),
        evidence_paper_ids: S.arr(S.str(), "Papers whose limitations or future-work sections show this gap."),
        why_unaddressed: S.str("Why the field has not closed it: hard, unfashionable, or newly possible."),
        tractability: S.int("1-10; how addressable with current methods."),
        potential_impact: S.str("What closing it would enable."),
      })
    ),
    priority_gap_id: S.str("The gap most worth attacking now."),
    prioritisation_rationale: S.str("Why that gap beats the others on impact x tractability."),
    saturated_areas: S.arr(S.str(), "Sub-topics the corpus shows are already well covered."),
    corpus_limitations: S.str("How the corpus itself constrains this analysis - a gap in the search is not a gap in the field."),
  });

  const HypothesisSet = S.obj({
    hypotheses: S.arr(
      S.obj({
        hypothesis_id: S.str("Short slug, e.g. 'H1'."),
        statement: S.str("Falsifiable claim in 'If X then Y' form."),
        rationale: S.str("Why the literature makes this plausible."),
        null_hypothesis: S.str("The H0 that an experiment would reject."),
        independent_variables: S.arr(S.str()),
        dependent_variables: S.arr(S.str()),
        novelty_score: S.int("1-10; how new relative to the corpus."),
        feasibility_score: S.int("1-10; testable with normal resources."),
        impact_score: S.int("1-10; how much the field would care."),
        grounded_in_paper_ids: S.arr(S.str()),
        addresses_gap: S.str("Which identified gap this attacks."),
      })
    ),
    recommended_hypothesis_id: S.str("The single best candidate to design an experiment for."),
    selection_rationale: S.str(),
  });

  const ExperimentDesign = S.obj({
    hypothesis_id: S.str(),
    title: S.str(),
    methodology: S.str("Design type and full procedure."),
    independent_variables: S.arr(S.str()),
    dependent_variables: S.arr(S.str()),
    controls: S.arr(S.str(), "Control conditions and confounds handled."),
    datasets_or_materials: S.arr(S.str()),
    procedure_steps: S.arr(S.str(), "Ordered, executable steps."),
    power_analysis: S.obj({
      effect_size: S.num("Assumed Cohen's d or equivalent."),
      alpha: S.num("Significance threshold, typically 0.05."),
      power: S.num("Target statistical power, typically 0.8."),
      required_sample_size: S.int("Per-group n from the calculation."),
      justification: S.str("Where the effect-size assumption came from."),
    }),
    statistical_tests: S.arr(S.str(), "Planned analyses, chosen in advance."),
    threats_to_validity: S.arr(S.str()),
    ethical_considerations: S.arr(S.str()),
    estimated_duration: S.str(),
    expected_outcomes: S.str("What supports vs. refutes the hypothesis."),
  });

  const CitationBundle = S.obj({
    style: S.enm(["APA", "IEEE", "MLA", "Nature"], "Style applied."),
    citations: S.arr(
      S.obj({
        paper_id: S.str("Corpus id this citation was built from."),
        in_text: S.str("In-text form, e.g. '(Scarselli et al., 2008)'."),
        full_reference: S.str("Full reference-list entry in the target style."),
        bibtex_key: S.str("Citation key, e.g. 'scarselli2008graph'."),
        bibtex_entry: S.str("Complete BibTeX record."),
      })
    ),
    reference_list_markdown: S.str("The rendered reference list."),
    bibtex_file: S.str("All BibTeX entries concatenated, ready to save."),
    unverified_ids: S.arr(S.str(), "Cited ids absent from the corpus. Must be empty in a clean run."),
    integrity_note: S.str("Statement of what was verified and what could not be."),
  });

  const ResearchReport = S.obj({
    title: S.str(),
    research_question: S.str(),
    executive_summary: S.str("150-250 words for a non-specialist."),
    sections: S.arr(
      S.obj({ heading: S.str(), body_markdown: S.str() }),
      "Introduction, Related Work, Gaps, Hypotheses, Proposed Experiment, Limitations, Future Work."
    ),
    key_findings: S.arr(S.str()),
    limitations: S.arr(S.str()),
    references: S.arr(S.str(), "Formatted citation strings."),
    confidence: S.enm(CONFIDENCE),
    word_count: S.int(),
  });

  const RelevanceCheck = S.obj({
    is_research_question: S.bool(),
    reasoning: S.str(),
    suggested_rewrite: S.str("A better-formed question, or empty string if already fine."),
  });

  const IntegrityCheck = S.obj({
    has_unsupported_claims: S.bool(),
    flagged_statements: S.arr(S.str()),
    reasoning: S.str(),
  });

  A.S = S;
  A.Stage = Stage;
  A.Schemas = {
    Paper, ResearchPlan, LiteratureReview, KnowledgeSynthesis, GapAnalysis,
    HypothesisSet, ExperimentDesign, CitationBundle, ResearchReport,
    RelevanceCheck, IntegrityCheck,
  };
})((window.Athena = window.Athena || {}));
