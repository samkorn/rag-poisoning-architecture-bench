# Analysis

This directory holds the results-analysis notebook and its committed outputs.

## Files

- `analysis.ipynb` — the analysis notebook. Committed with cell outputs so figures and tables render inline on GitHub.
- `figures/` — PDF + PNG figures referenced by the paper. Committed.
- `intermediate/merged_results.parquet` — cached join of all experiment results + judge labels + human labels. Gitignored. Regenerated on every notebook run.
- `human_labels.csv` — hand-labeled ground-truth subset. Gitignored; symlinked from workspace.

## Running the notebook

> **WARNING**: Running this notebook regenerates committed deliverables.
> Every file in `analysis/figures/` and `paper/tables/` is overwritten in
> place. After running, review `git diff` and commit deliberately.

### Font dependency

Figures use **Latin Modern Roman** to match the paper's Computer Modern body font. Install the OTF font family from <https://www.gust.org.pl/projects/e-foundry/latin-modern/download> if you want regenerated figures to be visually identical to the committed ones. Without it, matplotlib falls back to STIX Two Text (close to CM, ships with macOS) or DejaVu Serif. Data and layout are unaffected by the fallback — only the typeface differs.

```bash
venv/bin/jupyter nbconvert --to notebook --execute analysis/analysis.ipynb --output /tmp/analysis_out.ipynb
```

The `--output` target is a throwaway file — we don't commit executed notebooks. Figures, tables, and the parquet are written by the notebook to their real paths as side effects.

## What the notebook writes

| Path | Tracked? | Role |
|---|---|---|
| `analysis/figures/*.{pdf,png}` | yes | Paper-accompanying deliverables |
| `paper/tables/*.tex` | yes | LaTeX table fragments `\input{}`'d by the paper |
| `analysis/intermediate/merged_results.parquet` | no | Internal join cache |
| `/tmp/analysis_out.ipynb` (via `--output`) | no | Throwaway executed copy |
