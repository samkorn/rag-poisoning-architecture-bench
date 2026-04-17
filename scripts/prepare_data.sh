#!/usr/bin/env bash
#
# Build the raw dataset + LLM-generated artifacts needed for experiments.
#
# Pipeline:
#   1. Download NQ corpus from BEIR           (~1.4 GB, free)
#   2. Generate correct answers     (Modal)   (~$30-50 in OpenAI calls)
#   3. Generate naive poisoned docs (Modal)   (~$30-50 in OpenAI calls)
#   4. Generate CorruptRAG-AK docs  (Modal)   (~$20-30 in OpenAI calls)
#   5. Merge into nq-questions.jsonl          (local, free)
#   6. Assemble poisoned corpora              (local, free; ~2.8 GB disk)
#
# Each Modal step blocks until completion (NOT --detach). Total wall time
# with Modal parallelism is roughly 30-60 minutes. Run in a tmux session.
#
# Usage:
#   scripts/prepare_data.sh                      # run full pipeline
#   scripts/prepare_data.sh --skip-download      # skip step 1 (NQ already present)
#   scripts/prepare_data.sh --skip-llm           # skip steps 2-4 (already generated)
#   scripts/prepare_data.sh --from <step>        # resume from step N (1-6)
#   scripts/prepare_data.sh --help

set -euo pipefail

show_help() {
    sed -n '3,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

skip_download=0
skip_llm=0
from_step=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)       show_help; exit 0 ;;
        --skip-download) skip_download=1; shift ;;
        --skip-llm)      skip_llm=1; shift ;;
        --from)          from_step="$2"; shift 2 ;;
        *)               echo "ERROR: unknown arg '$1'" >&2; show_help >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/venv/bin/python"
MODAL="${REPO_ROOT}/venv/bin/modal"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: venv Python not found at ${PYTHON}" >&2
    echo "       Run: scripts/setup_environment.sh" >&2
    exit 1
fi
if [[ ! -x "${MODAL}" ]]; then
    echo "ERROR: modal CLI not found at ${MODAL}" >&2
    exit 1
fi

run_step() {
    local n="$1"; local label="$2"; shift 2
    if [[ "${n}" -lt "${from_step}" ]]; then
        echo "==> [${n}/6] SKIP ${label}  (--from ${from_step})"
        return
    fi
    echo
    echo "==> [${n}/6] ${label}"
    echo "    \$ $*"
    "$@"
}

# ---------------------------------------------------------------------------
# 1. Download NQ corpus
# ---------------------------------------------------------------------------
if [[ "${skip_download}" == "0" ]]; then
    run_step 1 "Download NQ from BEIR" \
        "${PYTHON}" src/data/download_datasets.py
else
    echo "==> [1/6] SKIP Download (--skip-download)"
fi

# ---------------------------------------------------------------------------
# 2-4. LLM-generated artifacts (Modal)
# ---------------------------------------------------------------------------
# NOTE: create_corruptrag_ak_poisoned_docs.py depends on
# experiment-datasets/nq-correct-answers.jsonl, so correct answers must run
# first. The original MIGRATION_PLAN script order was not dependency-correct.
if [[ "${skip_llm}" == "0" ]]; then
    run_step 2 "Generate correct answers (Modal)" \
        "${MODAL}" run src/data/create_correct_answers.py

    run_step 3 "Generate naive poisoned docs (Modal)" \
        "${MODAL}" run src/data/create_incorrect_answers_poisoned_docs.py

    run_step 4 "Generate CorruptRAG-AK poisoned docs (Modal)" \
        "${MODAL}" run src/data/create_corruptrag_ak_poisoned_docs.py
else
    echo "==> [2-4/6] SKIP LLM generation (--skip-llm)"
fi

# ---------------------------------------------------------------------------
# 5-6. Merge + assemble (local)
# ---------------------------------------------------------------------------
run_step 5 "Build nq-questions.jsonl" \
    "${PYTHON}" src/data/create_questions.py

run_step 6 "Assemble poisoned corpora (naive + CorruptRAG-AK)" \
    "${PYTHON}" src/data/create_poisoned_datasets.py

echo
echo "Done. Outputs in src/data/experiment-datasets/ and src/data/original-datasets/."
