# Analysis

This directory holds the results-analysis notebook and its committed outputs.

## Files

- `analysis.ipynb` — the analysis notebook. Committed without cell outputs.
- `figures/` — PDF + PNG figures referenced by the paper. Committed.
- `intermediate/merged_results.parquet` — cached join of all experiment results + judge labels + human labels. Gitignored. Regenerated on every notebook run.
- `human_labels.csv` — hand-labeled ground-truth subset. Gitignored; symlinked from workspace.

## Running the notebook

> **WARNING**: Running this notebook regenerates committed deliverables.
> Every file in `analysis/figures/` and `paper/tables/` is overwritten in
> place. After running, review `git diff` and commit deliberately.

### Executing

```bash
venv/bin/jupyter nbconvert --to notebook --execute analysis/analysis.ipynb --output /tmp/analysis_out.ipynb
```

The `--output` target is a throwaway file — we don't commit executed notebooks. Figures, tables, and the parquet are written by the notebook to their real paths as side effects.

### Editing

The `.ipynb` is the source of truth. To edit in a real Python file rather than JSON:

```bash
# 1. Convert to .py for editing:
venv/bin/jupytext --to py analysis/analysis.ipynb -o /tmp/analysis_edit.py

# 2. Edit /tmp/analysis_edit.py however you like.

# 3. Sync back to the .ipynb (--update preserves cell IDs and outputs):
venv/bin/jupytext --update --to notebook /tmp/analysis_edit.py -o analysis/analysis.ipynb
```

After editing, re-execute the notebook as above to regenerate figures and tables, then review `git diff` and commit.

## What the notebook writes

| Path | Tracked? | Role |
|---|---|---|
| `analysis/figures/*.{pdf,png}` | yes | Paper-accompanying deliverables |
| `paper/tables/*.tex` | yes | LaTeX table fragments `\input{}`'d by the paper |
| `analysis/intermediate/merged_results.parquet` | no | Internal join cache |
| `/tmp/analysis_out.ipynb` (via `--output`) | no | Throwaway executed copy |
