#!/usr/bin/env bash
#
# Embed the three corpus variants with Contriever (Modal GPU), build FAISS
# indexes locally, then filter questions to those with a gold doc in top-10.
#
# Pipeline:
#   1. Embed corpora on Modal GPU         (~30 GB output, ~$5-10 GPU cost)
#   2. Build 3 FAISS indexes locally      (~5-10 min, memory-heavy ~16 GB)
#   3. Filter to gold-retrieval questions (~1 min, produces 1,150 questions)
#
# Prerequisites:
#   - scripts/prepare_data.sh has completed — the three corpora must exist:
#       src/data/original-datasets/nq/
#       src/data/experiment-datasets/nq-naive-poisoning/
#       src/data/experiment-datasets/nq-corruptrag-ak-poisoning/
#
# Outputs:
#   src/data/vector-store/*.pkl   (embeddings + doc-id maps)
#   src/data/vector-store/*.faiss (indexes)
#   src/data/experiment-datasets/nq-questions-gold-filtered.jsonl
#
# Usage:
#   scripts/prepare_embeddings.sh              # run full pipeline
#   scripts/prepare_embeddings.sh --from <n>   # resume from step 1-3
#   scripts/prepare_embeddings.sh --help
#
# Run in tmux: the embedding step is long-running and expensive.

set -euo pipefail

show_help() {
    sed -n '3,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

from_step=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) show_help; exit 0 ;;
        --from)    from_step="$2"; shift 2 ;;
        *)         echo "ERROR: unknown arg '$1'" >&2; show_help >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/venv/bin/python"
MODAL="${REPO_ROOT}/venv/bin/modal"

if [[ ! -x "${PYTHON}" || ! -x "${MODAL}" ]]; then
    echo "ERROR: venv Python or modal CLI missing. Run scripts/setup_environment.sh first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Prerequisite check — corpora must exist
# ---------------------------------------------------------------------------
REQUIRED_CORPORA=(
    "src/data/original-datasets/nq/corpus.jsonl"
    "src/data/experiment-datasets/nq-naive-poisoning/corpus.jsonl"
    "src/data/experiment-datasets/nq-corruptrag-ak-poisoning/corpus.jsonl"
)
for f in "${REQUIRED_CORPORA[@]}"; do
    if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
        echo "ERROR: prerequisite corpus missing: ${f}" >&2
        echo "       Run: scripts/prepare_data.sh" >&2
        exit 1
    fi
done

run_step() {
    local n="$1"; local label="$2"; shift 2
    if [[ "${n}" -lt "${from_step}" ]]; then
        echo "==> [${n}/3] SKIP ${label}  (--from ${from_step})"
        return
    fi
    echo
    echo "==> [${n}/3] ${label}"
    echo "    \$ $*"
    "$@"
}

# ---------------------------------------------------------------------------
# 1. Embed corpora on Modal GPU
# ---------------------------------------------------------------------------
run_step 1 "Embed corpora (Contriever on Modal GPU)" \
    "${MODAL}" run src/embeddings/embed_datasets.py

# ---------------------------------------------------------------------------
# 2. Build FAISS indexes locally
# ---------------------------------------------------------------------------
run_step 2 "Build FAISS indexes from embeddings" \
    "${PYTHON}" src/embeddings/build_vector_indexes.py

# ---------------------------------------------------------------------------
# 3. Filter to gold-retrieval questions
# ---------------------------------------------------------------------------
run_step 3 "Filter to gold-retrieval questions (top-10)" \
    "${PYTHON}" src/data/filter_gold_questions.py

echo
echo "Done. FAISS indexes + gold-filtered questions ready for experiments."
