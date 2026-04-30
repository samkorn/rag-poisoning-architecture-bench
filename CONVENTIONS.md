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
* **Bullets use `*`** with 2-space indent under sections.
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

## Function and class docstrings

Function, method, and class docstrings use **Google style** (renderable
via Sphinx Napoleon if we ever publish HTML). Type information lives
exclusively in type hints — never in the docstring.

### Mood

* **Function and method docstrings use imperative mood** ("Return the
  result", "Build the index") per PEP 257 and Google style.
* **Module docstrings use descriptive mood** (see above).

The split is intentional: imperative reads better at function level
(verb-led command), while modules are more naturally described as a
thing.

### Function / method shape

```
def foo(name: str, retries: int = 3) -> Result:
    """Imperative one-line summary, ending in a period.

    Optional longer paragraph elaborating on behavior, edge cases,
    or anything a caller would want to know that isn't obvious from
    the signature.

    Args:
        name: Description of the param.
        retries: Description of the param. Default behavior or
            constraints can wrap onto subsequent lines indented an
            additional 4 spaces.

    Returns:
        Description of what comes back. For dicts/tuples with named
        fields, a sub-block listing them is encouraged.

    Raises:
        ValueError: When and why.
        TimeoutError: When and why.

    Notes:
        Anything a reader might naturally have questions about —
        design quirks, why this function doesn't reuse a related
        helper, on-disk schema constraints, etc.
    """
```

### Class shape

```
class Foo:
    """Imperative-flavored summary of what the class does.

    Optional longer paragraph.

    Attributes:
        bar: Class-level constant or computed attribute.
        baz: Another class-level attribute.
    """

    def __init__(self, name: str, retries: int = 3):
        """Initialize Foo.

        Args:
            name: Description.
            retries: Description.
        """
```

For **dataclasses, NamedTuples, and frozen objects** (no custom
`__init__`): document fields in the class docstring's `Attributes:`
block.

### Rules

* **No types in docstrings** — type hints are the source of truth.
  Sphinx Napoleon picks them up automatically when rendering, and IDE
  hovers already show the signature alongside the docstring.
* **One docstring site per attribute.**
  * Dataclasses / NamedTuples → `Attributes:` block in the class
    docstring.
  * Regular classes with custom `__init__` → params documented in
    `__init__`; the class docstring covers what the class *is*, plus
    class-level constants or computed properties not passed to
    `__init__`. Avoid duplicating `__init__` params in the class
    docstring.
* **Coverage:**
  * All public functions, methods, and classes get a docstring.
  * All `__init__` with non-trivial params get a docstring.
  * Private helpers (`_foo`) — only when the *why* isn't obvious from
    name + signature.
  * Trivial getters / `@property` returning a stored attribute with
    no side effect — skip.
* **One-line docstrings allowed** when the summary captures everything
  and there are no params/returns worth documenting separately. The
  closing `"""` goes on the same line.
* **Test methods** get a single-line summary describing what's being
  tested. No `Args:` block (only `self`). pytest renders them with
  `--verbose`.
* **`@property`**: docstring on the getter, not as a class-attribute
  listing entry.
* **Code references**: single backticks `` `VectorStore` `` (matches
  module-docstring rule).
* **`Notes:` / `Examples:` blocks** — same shape as in module
  docstrings, indented as a free-form block at the end. `Examples:`
  (plural) is the Google-style convention.
* **No Sphinx cross-references** (`:func:`, `:class:`, `:mod:`) —
  inline backticks only. Cross-refs add visual noise on GitHub and
  we're not publishing Sphinx HTML.
* **Skip log**: every site where we deliberately omit a docstring
  (per the coverage rules above) is recorded in
  `.claude/plans/PHASE_7C_DOCSTRING_SKIPS.md` so the omissions are
  reviewable rather than implicit.

## LaTeX style (paper/)

The paper follows the design patterns from Adam Gleave's *Writing
Beautifully in LaTeX* (https://www.gleave.me/post/latex-design-patterns/).
The conventions below are the ones from that guide that apply to
`paper/paper.tex`; each entry notes the rule and how we follow it (or
deliberately diverge).

### Macros: Don't Repeat Yourself

* **Define a macro for any string or piece of notation that appears
  more than a couple of times.** This is the LaTeX expression of DRY:
  a single `\renewcommand` re-skins every site at once and rules out
  inconsistent notation across sections.
* **Math notation in particular gets macros**, not raw symbols. If the
  same operator, set, or distribution appears repeatedly, wrap it
  (e.g., `\newcommand{\E}{\mathbb{E}}`,
  `\newcommand{\set}[1]{\{#1\}}`).
* **Boilerplate inside figure captions / table notes that recurs is
  also macro'd** so wording stays in lock-step.
* We currently have `\metric{#1}` for `\texttt`-formatted metric names;
  add new macros when a literal phrase (architecture name, attack
  name, dataset shorthand) starts repeating across sections.

### Citations and bibliography

* **Prefer BibLaTeX over BibTeX** for new projects — it's the modern
  replacement, supports more entry types, and handles localization.
  Our paper currently uses `natbib`; if we migrate, the equivalent
  command mapping is `\citep` → `\parencite` and `\citet` →
  `\textcite`.
* **Use the right citation command for the grammatical role:**
  * Parenthetical (citation is an aside): `\citep{...}` (natbib) /
    `\parencite{...}` (BibLaTeX). E.g., "...inject a single crafted
    document~\citep{zou2025poisonedrag}."
  * In-text (the author is the subject of the sentence): `\citet{...}`
    / `\textcite{...}`. E.g., "\citet{zou2025poisonedrag} introduced
    PoisonedRAG..."
  * Don't write "[1] showed that..." — use `\citet`/`\textcite` so the
    author name is rendered.
* **Always tie a citation to the preceding word with `~`**, never a
  regular space: `LaTeX~\citep{lewis2020rag}`. This prevents the
  citation from being orphaned at the start of a new line.

### Cross-references

* **Use `cleveref`'s `\cref{...}` / `\Cref{...}`** rather than writing
  the cross-reference type by hand. `\cref{fig:teaser}` renders as
  "Fig. 1", `\Cref{...}` as "Figure 1" at the start of a sentence.
* **Anti-pattern**: hand-rolled `Section~\ref{sec:foo}` /
  `Eq.~(\ref{eq:bar})`. Reasons: cleveref keeps the type label
  consistent everywhere, switches automatically when a label moves
  between environments, and handles ranges (`\cref{a,b,c}` →
  "Sections 1 to 3"). Our current paper still uses
  `Section~\ref{...}`; new cross-references should prefer `\cref`,
  and a future cleanup should `\usepackage{cleveref}` and migrate the
  existing sites.
* **Always label the thing you might cite** (`\label{sec:...}`,
  `\label{fig:...}`, `\label{tab:...}`, `\label{eq:...}`) using the
  prefix convention already in the paper.

### Math

* **Punctuate displayed math.** Equations are part of the surrounding
  sentence. If an equation closes a sentence, the equation ends with
  a period inside the math environment; a comma if the sentence
  continues afterward. Skipping math punctuation is one of Gleave's
  core anti-patterns.
* **Macro the recurring symbols** (see DRY above) so a future
  notational change is one edit, not dozens.

### Figures

* **Use vector graphics (PDF or SVG), not raster (PNG/JPEG).** Vectors
  scale to any resolution without artifacts and are usually smaller
  on disk than the equivalent raster. All architecture diagrams in
  `paper/assets/` are PDFs; matplotlib outputs in
  `analysis/figures/` should be saved as PDF for inclusion.
* **Make every figure self-contained.** A reader scanning only
  figures + captions should understand the result. That means:
  legible axis labels, a legend that names the series, and a caption
  that states what is being shown, what the takeaway is, and any
  context the axes alone don't convey (sample sizes, error bars,
  filtering).
* **Caption boilerplate that recurs** (e.g., "Error bars are 95%
  bootstrap CIs over N=921 questions") should live in a macro per
  the DRY rule.

### Tables

* **Use `booktabs`** (`\toprule`, `\midrule`, `\bottomrule`) instead
  of `\hline` and vertical rules. Already enabled in `paper.tex`.
* **No vertical rules**, and no double horizontal rules — booktabs'
  spacing carries the visual structure.
* **For numeric columns that should align on the decimal point**,
  prefer the `siunitx` `S` column type over manual padding.

### Quotation marks

* **Use LaTeX's directional quotes**, never the straight ASCII `"`:
  * Double quotes: `` ``like this'' ``.
  * Single quotes: `` `like this' ``.
* The straight `"` produces wrong-direction glyphs; this is one of
  the most visible LaTeX-newbie tells.

### Emphasis

* **Use `\emph{...}` for emphasis, not `\textit{...}`.** `\emph`
  flips italic ↔ upright when nested inside an italic block, so it
  always reads as "emphasized" regardless of context. `\textit` is
  idempotent and breaks nested emphasis.
* Reserve `\textbf{...}` for structural elements (table headers,
  defined terms on first introduction); don't use bold for emphasis
  in body prose.

### Abbreviations and inter-sentence spacing

* LaTeX inserts extra horizontal space after a period it thinks ends a
  sentence. Periods inside abbreviations (`e.g.`, `i.e.`, `cf.`,
  `et al.`, `Fig.`, `Eq.`) trigger this incorrectly.
* **Fix abbreviations followed by a word** with a backslash-space:
  `e.g.\ this`, `i.e.\ that`, `et al.\ showed`.
* **Fix abbreviations followed by a non-breaking tie** (e.g., before a
  number/reference): `Fig.~\ref{...}` is fine because `~` is already
  a non-sentence-ending space.
* When a real sentence ends with a capital letter (`...the LLM. Then
  ...`), insert `\@` before the period — `...the LLM\@. Then...` —
  to restore the sentence-ending space LaTeX would otherwise drop.

### Whitespace and line breaking

* **Non-breaking space (`~`) holds together** tokens that read badly
  if split across lines: `Section~3`, `Figure~\ref{fig:teaser}`,
  `N~=~1`, `\citet{...}`'s author–year pair (handled by the citation
  command itself).
* **Don't manually insert `\\` or `\newline`** to fix layout; let
  LaTeX paginate. If a line is overfull, fix the source (rephrase,
  hyphenate, or raise `\emergencystretch` — already set to `2em` in
  `paper.tex`).

### Hyperlinks

* **Load `hyperref` last** among packages that interact with it
  (after `caption`, before `cleveref` if/when added). Already the
  case in `paper.tex`.
* **Color links subtly**, not bright primary colors: our setup uses
  `blue!60!black` for `linkcolor`, `citecolor`, and `urlcolor`.
* **Use `\url{...}` for raw URLs** and `\href{URL}{anchor text}` when
  the anchor reads as prose. Already used for the code/data footnote.

### Build hygiene: zero warnings

* **Treat the LaTeX log's "Warning" lines as errors.** A clean baseline
  ("0 warnings") makes any *new* warning visible immediately on the
  next compile. The common offenders:
  * `LaTeX Warning: Reference '...' on page ... undefined.` — fix the
    `\label` / `\ref` mismatch; don't ship undefined references.
  * `LaTeX Warning: Citation '...' undefined.` — fix the bib key.
  * `Overfull \hbox` / `Underfull \hbox` — fix the source line, or
    suppress invisible-bad ones globally (`hbadness` is set to 2500
    in `paper.tex` for that reason).
* **Don't silence warnings you haven't read.** Suppression is for
  warnings that are demonstrably cosmetic; everything else gets fixed.

### Version control

* **The `.tex`, `.bib`, and figure-source files live in git**, not
  Overleaf's built-in version history. Commit small, captioned
  changes; treat the paper like code.
* **Never commit `.aux`, `.log`, `.out`, `.bbl`, `.blg`, `.toc`, or
  the built `.pdf` to a code review branch** — `.gitignore` should
  exclude them. The exception is a final `paper.pdf` snapshot that
  we do track for reviewer convenience.
* **Latexmk** (`latexmk -pdf paper.tex`) is the canonical build; it
  re-runs pdflatex/bibtex the right number of times to resolve all
  cross-references and citations.

### Source notes

* This section summarizes the conventions from Gleave's guide that
  apply to our paper. The guide covers additional patterns (e.g.,
  thesis-scale file decomposition, multi-language typesetting) that
  aren't relevant here and are intentionally omitted.
* The original guide should be the tiebreaker for any case not
  covered above; if a new pattern from the guide gets adopted in the
  paper, add it here with a short note on how it's applied.
