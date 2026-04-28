# Roadmap

Forward-looking work for this repo, split into two scales:

- **Research extensions** — bigger, paper-shaped follow-ups. The kind of
  thing that warrants its own experiment sweep, ablation, or follow-up paper.
- **Near-term codebase TODOs** — smaller code-hygiene / migration items.
  No new science; just finishing or polishing what's here.

When an item is shipped, remove it from this file and let the PR / commit /
paper revision carry the history.

---

## Research extensions

### AstuteRAG as a fifth architecture
Originally scoped out of the initial study. Worth revisiting as an extension:
it's the natural "consistency-checking" comparison point against MADAM-RAG
and would round out the architecture grid.

- [ ] Re-read Wang et al. and decide on a faithful implementation
- [ ] Add `src/architectures/astute_rag.py` mirroring the existing arch interface
- [ ] Re-run the 3 attack conditions on the 1,150-question slice
- [ ] Update analysis notebook + paper to include the 5th column

### Updated MADAM-RAG
Current MADAM-RAG implementation is a faithful read of the COLM 2025 paper,
but several knobs are under-explored:

- [ ] Vary number of debate rounds (currently fixed)
- [ ] Vary number of agents
- [ ] Try heterogeneous-backbone agents (different LLMs per agent)
- [ ] Probe: does the "highest CD rate, lowest INCORRECT, but resolution
      doesn't follow" finding hold under these variations?

### Multiple K values
K=10 was held fixed across all RAG architectures. A K-sweep is the cheapest
way to test how brittle the headline numbers are.

- [ ] Pick K grid (e.g. {3, 5, 10, 20, 50})
- [ ] Re-run Vanilla, Agentic, MADAM at each K under all 3 attack conditions
- [ ] Decide whether RLM gets a comparable "context budget" sweep
- [ ] Add a K-vs-ASR figure to the paper / blog

### Additional attack families
The current grid uses Naive injection and CorruptRAG-AK (N=1). The next
natural attacks to add:

- [ ] **PoisonedRAG** (Zou et al.) — black-box attack, established baseline
- [ ] **CorruptRAG-AS** — the adversarial-suffix variant (paired with the
      AK variant we already run)
- [ ] **AuthChain** — authority-chain style attacks
- [ ] Decide whether to scale up `N` (multiple injected docs) or keep N=1
      to isolate the per-attack signal

### Defensive prompting
Once the attack grid is broader, the natural counterpart is testing
prompt-level defenses on each architecture.

- [ ] Catalog defense strategies to test (skepticism prompts, source-
      reliability instructions, "verify before answering" scaffolds, etc.)
- [ ] Add a defense axis to the experiment matrix (architecture × attack ×
      defense)
- [ ] Be careful not to over-tune defenses to the specific attacks in the
      grid — hold out at least one attack family

### Behavioral analysis from execution logs
Right now we capture final answers and judge labels. Each architecture also
generates rich intermediate artifacts (Agentic tool-call traces, MADAM
debate transcripts, RLM recursion paths) that we currently discard or only
spot-check.

- [ ] Decide on a logging schema that's uniform enough across architectures
      to support cross-arch comparison
- [ ] Persist these traces alongside the existing per-question result JSONs
- [ ] Failure-mode taxonomy: when an architecture gets a question wrong
      under attack, what *kind* of wrong? (e.g. did MADAM's agents converge
      on the wrong answer, or did they disagree and the aggregator picked
      the wrong one?)
- [ ] Quantify the "MADAM detects but doesn't resolve" finding with
      transcript-level evidence rather than only outcome-level metrics

### Other candidates (capture-as-you-go)

<!-- Stub section. Drop ideas here as they come up; promote to a real
     subsection above when one becomes concrete. -->

- [ ] Additional datasets beyond NQ (HotpotQA, TriviaQA?)
- [ ] Backbone-LLM sweep (gpt-5-mini vs. larger / open-weight)
- [ ] Re-examine the 229 noise-filtered questions — are any actually answerable?

---

## Near-term codebase TODOs

### Query / question rename — Level 2 (on-disk schema)
Level 1 (internal identifiers) shipped in
[#27](https://github.com/samkorn/rag-poisoning-architecture-bench/pull/27).
Level 2 threads the same `query` / `question` standard through the on-disk
artifacts:

- [ ] Migrate persisted result JSONs: `question_id` → `query_id`,
      `question_text` → `question` (or whatever the final field name is)
- [ ] Rename `QuestionResult` dataclass + its field names
- [ ] Update `analysis.ipynb` to read the new schema
- [ ] Update paper tables / figure-generation code that currently keys on
      `question_id`
- [ ] Decide what to do with the ~14k existing result files (in-place
      migration script vs. rewrite-on-read shim vs. clean-slate re-run)

### Other small items (capture-as-you-go)

<!-- One-liners are fine here. Promote to a real section if it grows. -->

- [ ]
