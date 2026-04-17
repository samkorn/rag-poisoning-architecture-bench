#!/usr/bin/env bash
#
# Compile paper/paper.tex into a PDF.
#
# Canonical LaTeX + BibTeX build sequence:
#   pdflatex -> bibtex -> pdflatex -> pdflatex
#
# Prerequisites:
#   - /Library/TeX/texbin/pdflatex and bibtex installed (MacTeX / BasicTeX)
#   - paper/tables/*.tex regenerated if analysis changed
#       (run scripts/run_analysis.sh first)
#   - analysis/figures/*.pdf exist if the paper references them
#
# Usage:
#   scripts/generate_paper.sh             # full 4-pass build
#   scripts/generate_paper.sh --quick     # single pdflatex pass (cached .bbl/.aux)
#   scripts/generate_paper.sh --help

set -euo pipefail

show_help() {
    awk 'NR==1 {next} /^[^#]/ {exit} {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
}

quick=0
for arg in "$@"; do
    case "${arg}" in
        --help|-h) show_help; exit 0 ;;
        --quick)   quick=1 ;;
        *)         echo "ERROR: unknown arg '${arg}'" >&2; show_help >&2; exit 2 ;;
    esac
done

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ ! -d "${REPO_ROOT}/paper" ]]; then
    echo "ERROR: paper directory not found at ${REPO_ROOT}/paper" >&2
    exit 1
fi
cd "${REPO_ROOT}/paper"

PDFLATEX=/Library/TeX/texbin/pdflatex
BIBTEX=/Library/TeX/texbin/bibtex
DOC=paper

if [[ ! -x "${PDFLATEX}" ]]; then
    echo "ERROR: pdflatex not found at ${PDFLATEX}" >&2
    echo "       Install MacTeX or BasicTeX." >&2
    exit 1
fi
if [[ ! -x "${BIBTEX}" ]]; then
    echo "ERROR: bibtex not found at ${BIBTEX}" >&2
    exit 1
fi
if [[ ! -f "${DOC}.tex" ]]; then
    echo "ERROR: ${DOC}.tex not found in $(pwd)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
# -halt-on-error so failures surface immediately.
# -interaction=nonstopmode so pdflatex doesn't prompt on missing refs.
PDFLATEX_ARGS=(-halt-on-error -interaction=nonstopmode "${DOC}.tex")

if [[ "${quick}" == "1" ]]; then
    echo "==> Quick build (1 pdflatex pass)"
    "${PDFLATEX}" "${PDFLATEX_ARGS[@]}"
else
    echo "==> [1/4] pdflatex";  "${PDFLATEX}" "${PDFLATEX_ARGS[@]}"
    echo "==> [2/4] bibtex";    "${BIBTEX}"   "${DOC}"
    echo "==> [3/4] pdflatex";  "${PDFLATEX}" "${PDFLATEX_ARGS[@]}"
    echo "==> [4/4] pdflatex";  "${PDFLATEX}" "${PDFLATEX_ARGS[@]}"
fi

echo
echo "Done. Output: paper/${DOC}.pdf"
