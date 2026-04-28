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

## Module docstrings

Every Python file in this repo has a module-level docstring. The format
below is the canonical shape. Sections are optional — skip any that
don't apply (a small library helper often only needs the one-line
summary).

### Shape

```
"""<One-line descriptive summary — what the file does.>

<Optional longer paragraph elaborating on the summary.>

Prerequisites:
    <Indented prose or bullet list of what must already exist on disk
    or in the environment before this file can run / import cleanly.>

Usage:
    python src/foo/bar.py
    python src/foo/bar.py --flag value

Output:
    <Path or list of paths the script writes, indented.>

Notes:
    <Anything a reader might naturally have questions about — design
    quirks, on-disk schema constraints, why this script doesn't reuse
    a related abstraction, etc.>
"""
```

### Rules

* **Always at least a one-line docstring**, even for `__init__.py`
  files. For an empty package init, a single line naming the package
  and what it contains is enough.
* **One-line summary is descriptive mood**: "Builds 3 FAISS indexes…",
  not "Build 3 FAISS indexes…". (PEP 257 allows either; we standardize
  on descriptive because most existing docstrings are already in that
  style.)
* **No filename header** at the top. Don't write `experiments/foo.py`
  on its own line — the module path is already obvious from the
  import.
* **Single backticks** for object names, file names, paths, and CLI
  flags: `` `VectorStore` ``, `` `nq-questions.jsonl` ``,
  `` `src/data/utils.py` ``, `` `--dry-run` ``. Never RST-style double
  backticks.
* **Bullets use `*`** with 2-space indent under sections (matches the
  Sphinx-flavored style we'll standardize on for function docstrings).
* **`Usage:` block**: 4-space indented, bare command (no leading `$` /
  `>` / `::`). One command per line. `Usage:` (single colon), not
  `Usage::`.
* **`Output:` block**: 4-space indented. List paths the script writes;
  brief annotation per path is fine.
* **`Prerequisites:` block**: 4-space indented prose or bulleted list.
  Use this for things like "run `src/data/create_questions.py` first"
  or "needs an OpenAI key in the environment".
* **`Notes:` block**: bottom of the docstring. Pre-emptively explains
  anything a reader might find confusing — design quirks, on-disk
  schema constraints, why a related abstraction wasn't reused, known
  simplifications.
* **Library modules** (those without a `__main__` guard) typically
  skip `Usage:` and `Output:`. A `Defines:` or `Core functions:`
  bulleted block listing the public surface is allowed in place of a
  long prose paragraph.
* **Exception for test files**: tests use Sphinx cross-refs like
  `:class:` already and the bullet character there is `*` — same rule.
