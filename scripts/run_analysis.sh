#!/usr/bin/env bash
#
# Execute analysis/analysis.ipynb to regenerate figures, intermediate
# parquet files, and paper tables from the experiment + judge + noise
# results.
#
# Outputs (written by the notebook):
#   analysis/figures/*.png, *.pdf         (15 figures)
#   analysis/intermediate/*.parquet       (analysis data)
#   paper/tables/*.tex                    (7 tables)
#   analysis/analysis.ipynb               (in-place, with fresh cell outputs)
#
# Prerequisites:
#   - src/experiments/results/{experiments,judge,noise}/ are populated
#     (either by running experiments locally, or via scripts/download_data.sh)
#   - analysis/human_labels.csv exists
#
# Usage:
#   scripts/run_analysis.sh               # execute notebook in place
#   scripts/run_analysis.sh --help

set -euo pipefail

show_help() {
    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "${arg}" in
        --help|-h) show_help; exit 0 ;;
        *)         echo "ERROR: unknown arg '${arg}'" >&2; show_help >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

JUPYTER="${REPO_ROOT}/venv/bin/jupyter"
if [[ ! -x "${JUPYTER}" ]]; then
    echo "ERROR: jupyter not found at ${JUPYTER}" >&2
    echo "       Run: scripts/setup_environment.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
REQUIRED=(
    "src/experiments/results/experiments"
    "src/experiments/results/judge"
    "src/experiments/results/noise"
    "analysis/human_labels.csv"
)
for path in "${REQUIRED[@]}"; do
    if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
        echo "ERROR: prerequisite missing: ${path}" >&2
        echo "       Run: scripts/download_data.sh  (or scripts/run_experiments.sh)" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Ensure output dirs exist
# ---------------------------------------------------------------------------
mkdir -p analysis/figures analysis/intermediate paper/tables

# ---------------------------------------------------------------------------
# Execute notebook in place
# ---------------------------------------------------------------------------
echo "==> Executing analysis/analysis.ipynb"
echo "    This takes several minutes."
"${JUPYTER}" nbconvert \
    --to notebook \
    --execute analysis/analysis.ipynb \
    --inplace \
    --ExecutePreprocessor.timeout=1800

echo
echo "Done. Outputs:"
echo "  Figures:       $(find analysis/figures -type f \( -name '*.png' -o -name '*.pdf' \) | wc -l | tr -d ' ')"
echo "  Intermediate:  $(find analysis/intermediate -type f -name '*.parquet' | wc -l | tr -d ' ')"
echo "  Paper tables:  $(find paper/tables -type f -name '*.tex' | wc -l | tr -d ' ')"
