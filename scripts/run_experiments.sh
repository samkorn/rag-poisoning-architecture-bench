#!/usr/bin/env bash
#
# Run the full experiment + judge + noise-filter pipeline on Modal.
#
# Pipeline:
#   1. Upload local data to Modal Volume              (~30-60 min for ~30 GB)
#   2. Launch 12 experiments (orchestrator, --detach) (~$280-440 OpenAI cost)
#   3. Launch LLM judge pass (--detach)               (~$50-100 OpenAI cost)
#   4. Launch noise filter pass (--detach)            (~$20-30 OpenAI cost)
#
# Steps 2-4 are `modal run --detach`: they return quickly and run in the
# background on Modal. Use `modal app logs <name>` to monitor.
#
# Apps:
#   - rag-poisoning-bench  (orchestrator)
#   - rag-poisoning-judge  (judge)
#   - rag-poisoning-noise  (noise filter)
#
# Prerequisites:
#   scripts/prepare_embeddings.sh has completed (vector store + gold questions)
#
# Usage:
#   scripts/run_experiments.sh                 # full pipeline
#   scripts/run_experiments.sh --skip-upload   # skip step 1 (volume already populated)
#   scripts/run_experiments.sh --experiments   # only step 2
#   scripts/run_experiments.sh --judge         # only step 3
#   scripts/run_experiments.sh --noise         # only step 4
#   scripts/run_experiments.sh --dry-run       # print commands without executing
#   scripts/run_experiments.sh --help
#
# Judge and noise should only be launched AFTER orchestrator has finished.
# This script launches them in sequence with --detach; if you want to wait
# for experiments to finish before judging, use the phase flags separately.

set -euo pipefail

show_help() {
    awk 'NR==1 {next} /^[^#]/ {exit} {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
}

mode="all"
skip_upload=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)     show_help; exit 0 ;;
        --skip-upload) skip_upload=1; shift ;;
        --experiments) mode="experiments"; shift ;;
        --judge)       mode="judge"; shift ;;
        --noise)       mode="noise"; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        *)             echo "ERROR: unknown arg '$1'" >&2; show_help >&2; exit 2 ;;
    esac
done

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/venv/bin/python"
MODAL="${REPO_ROOT}/venv/bin/modal"

if [[ ! -x "${PYTHON}" || ! -x "${MODAL}" ]]; then
    echo "ERROR: venv Python or modal CLI missing. Run scripts/setup_environment.sh first." >&2
    exit 1
fi

if [[ ! -f "${HOME}/.modal.toml" ]]; then
    echo "ERROR: Modal not configured. Run: ${MODAL} token new" >&2
    exit 1
fi

maybe_exec() {
    if [[ "${DRY_RUN}" == "0" ]]; then
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: upload data to Modal Volume
# ---------------------------------------------------------------------------
step_upload() {
    if [[ "${skip_upload}" == "1" ]]; then
        echo "==> SKIP upload (--skip-upload)"
        return
    fi
    echo
    echo "==> [1/4] Upload data to Modal Volume 'rag-poisoning-data'"
    echo "    \$ ${PYTHON} src/experiments/upload_data.py"
    maybe_exec "${PYTHON}" src/experiments/upload_data.py
}

# ---------------------------------------------------------------------------
# Step 2: launch experiments (detached)
# ---------------------------------------------------------------------------
step_experiments() {
    echo
    echo "==> [2/4] Launch 12 experiments (detached)"
    echo "    \$ ${MODAL} run --detach src/experiments/orchestrator.py"
    maybe_exec "${MODAL}" run --detach src/experiments/orchestrator.py
    echo "    Monitor: ${MODAL} app logs rag-poisoning-bench"
}

# ---------------------------------------------------------------------------
# Step 3: launch judge (detached)
# ---------------------------------------------------------------------------
step_judge() {
    echo
    echo "==> [3/4] Launch LLM judge (detached)"
    echo "    \$ ${MODAL} run --detach src/experiments/run_judge_modal.py"
    maybe_exec "${MODAL}" run --detach src/experiments/run_judge_modal.py
    echo "    Monitor: ${MODAL} app logs rag-poisoning-judge"
}

# ---------------------------------------------------------------------------
# Step 4: launch noise filter (detached)
# ---------------------------------------------------------------------------
step_noise() {
    echo
    echo "==> [4/4] Launch noise filter (detached)"
    echo "    \$ ${MODAL} run --detach src/experiments/run_noise_modal.py"
    maybe_exec "${MODAL}" run --detach src/experiments/run_noise_modal.py
    echo "    Monitor: ${MODAL} app logs rag-poisoning-noise"
}

case "${mode}" in
    all)
        step_upload
        step_experiments
        echo
        echo "NOTE: Judge + noise launched immediately (detached). They query the"
        echo "      Modal Volume for results, so if experiments aren't done yet,"
        echo "      they will skip un-judged questions and need to be re-run."
        echo "      Prefer launching judge + noise separately after experiments"
        echo "      complete:  scripts/run_experiments.sh --judge  (then --noise)"
        step_judge
        step_noise
        ;;
    experiments) step_upload; step_experiments ;;
    judge)       step_judge ;;
    noise)       step_noise ;;
esac

echo
echo "Done. Use 'modal app list' to see running apps."
