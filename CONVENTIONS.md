# Conventions

## Naming: `query` vs `question`

The codebase distinguishes three related terms with specific meanings.
Read this before adding code that touches query records or result files.

| Term | Meaning |
|---|---|
| **query** | The whole record — id + question text + answers + metadata. Use this name for full record dicts and the iterables that hold them. |
| **query_id** | The unique identifier of a query (e.g. `"test0"`, `"test3451"`). |
| **question** | The natural-language question text *only* — a field of a query record. |

In practice:

- Variables, parameters, and loop names: `query`, `query_id`, `queries`,
  `query_ids` — never `qid`, `q_id`, `question_id`, or `question` for a
  full record.
- The `'question'` *field* in a query record holds just the natural-language
  text. That string-literal field name is correct per this standard.

### Exceptions (frozen for compatibility)

A small number of `question`-flavored names are kept deliberately because
changing them would break on-disk JSON schemas, function-name stability, or
file-naming history:

- The `QuestionResult` dataclass and its `question_id` / `question_text`
  fields. The class is serialized via `dataclasses.asdict()` to ~14k
  per-result JSONs — the field names are the on-disk schema.
- The `'question_id'` string literal used as a dict key in any JSON written
  to disk (judge results, noise filter results, etc.).
- The function names `run_single_question`, `run_question_batch`, and
  `_run_single_question`. Renaming them would cascade through Modal log
  history and downstream scripts; their parameters use `query` correctly.
- The data files `nq-questions.jsonl` and `nq-questions-gold-filtered.jsonl`.
  The filenames are referenced from analysis notebooks, paper tables, and
  upstream BEIR conventions.

Each of these sites carries an inline comment explaining the exception.

### Why

The `question_id` / `qid` / `question: dict` naming was scattered across
the judge, noise filter, and result-write layers, while the upstream
producer path already used `query_id` / `query`. The mismatch made it
hard to tell whether a `question`-typed parameter held a full record or
just the question text. The convention above reflects how the data
*actually* flows.
