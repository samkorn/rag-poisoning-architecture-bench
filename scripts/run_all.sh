#!/usr/bin/env bash
#
# Full reproduction: env setup -> data -> embeddings -> experiments ->
# analysis -> paper. Everything from scratch, no Zenodo download.
#
# Expected wall time: multi-day end-to-end, dominated by experiments on
# Modal ($280-440 OpenAI spend) + embeddings ($5-10 GPU).
#
# Default mode runs the synchronous, local stages (env + data + embeddings)
# then launches the experiment orchestrator on Modal detached and exits.
# Judge, noise, analysis, and paper aren't launched here — they depend on
# the orchestrator completing, which can take ~24h. The closing banner
# prints the exact follow-up commands; re-invoke with --resume once
# experiments + judge + noise have all finished to run analysis + paper.
#
# --analysis-only skips regenerating experiment results and instead
# pulls them from Zenodo (~40 MB), then runs analysis + paper. Good when
# you want a fresh figure/table pass without rerunning Modal.
#
# Usage:
#   scripts/run_all.sh                   # env -> data -> embeddings -> launch experiments
#   scripts/run_all.sh --resume          # analysis + paper (after Modal runs are done)
#   scripts/run_all.sh --analysis-only   # env -> download Zenodo data -> analysis + paper
#   scripts/run_all.sh --dry-run         # print what every downstream script would do
#   scripts/run_all.sh --help

set -euo pipefail

show_help() {
    awk 'NR==1 {next} /^[^#]/ {exit} {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
}

resume=0
analysis_only=0
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --help|-h)       show_help; exit 0 ;;
        --resume)        resume=1 ;;
        --analysis-only) analysis_only=1 ;;
        --dry-run)       DRY_RUN=1 ;;
        *)               echo "ERROR: unknown arg '${arg}'" >&2; show_help >&2; exit 2 ;;
    esac
done

if [[ "${resume}" == "1" && "${analysis_only}" == "1" ]]; then
    echo "ERROR: --resume and --analysis-only are mutually exclusive" >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

SCRIPTS="${REPO_ROOT}/scripts"

# Scripts that support --dry-run get it propagated; setup_environment.sh
# does not, so it's always skipped entirely when DRY_RUN is on.
dry_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
    dry_args=(--dry-run)
fi

if [[ "${resume}" == "1" ]]; then
    echo "==> Resume: analysis + paper"
    if [[ "${DRY_RUN}" == "0" ]]; then
        "${SCRIPTS}/run_analysis.sh"
        "${SCRIPTS}/generate_paper.sh"
    else
        echo "    \$ ${SCRIPTS}/run_analysis.sh    (no --dry-run support; would execute)"
        echo "    \$ ${SCRIPTS}/generate_paper.sh  (no --dry-run support; would execute)"
    fi
    echo
    echo "Done."
    exit 0
fi

if [[ "${analysis_only}" == "1" ]]; then
    # None of the four downstream scripts here implement --dry-run, so
    # DRY_RUN collapses to "show the invocation and skip".
    run_or_echo() {
        local script_path="$1"
        if [[ "${DRY_RUN}" == "0" ]]; then
            "${script_path}"
        else
            echo "    \$ ${script_path}  (no --dry-run support; would execute)"
        fi
    }

    echo "==> [1/4] setup_environment.sh"
    run_or_echo "${SCRIPTS}/setup_environment.sh"
    echo
    echo "==> [2/4] download_data.sh"
    run_or_echo "${SCRIPTS}/download_data.sh"
    echo
    echo "==> [3/4] run_analysis.sh"
    run_or_echo "${SCRIPTS}/run_analysis.sh"
    echo
    echo "==> [4/4] generate_paper.sh"
    run_or_echo "${SCRIPTS}/generate_paper.sh"
    echo
    echo "Done."
    exit 0
fi

echo "==> [1/4] setup_environment.sh"
if [[ "${DRY_RUN}" == "0" ]]; then
    "${SCRIPTS}/setup_environment.sh"
else
    echo "    \$ ${SCRIPTS}/setup_environment.sh  (no --dry-run support; would execute)"
fi

echo
echo "==> [2/4] prepare_data.sh"
"${SCRIPTS}/prepare_data.sh" "${dry_args[@]+"${dry_args[@]}"}"

echo
echo "==> [3/4] prepare_embeddings.sh"
"${SCRIPTS}/prepare_embeddings.sh" "${dry_args[@]+"${dry_args[@]}"}"

echo
echo "==> [4/4] run_experiments.sh --experiments (orchestrator, detached)"
"${SCRIPTS}/run_experiments.sh" --experiments "${dry_args[@]+"${dry_args[@]}"}"

cat <<'EOF'

================================================================
Experiments are running detached on Modal. When they finish:

  scripts/run_experiments.sh --judge
  # wait for judge to finish
  scripts/run_experiments.sh --noise
  # wait for noise to finish
  scripts/run_all.sh --resume     # analysis + paper

Monitor:
  modal app logs rag-poisoning-bench
  modal app logs rag-poisoning-judge
  modal app logs rag-poisoning-noise
================================================================
EOF
