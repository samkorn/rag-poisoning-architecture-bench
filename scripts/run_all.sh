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
#   scripts/run_all.sh                   # env -> data -> embeddings -> launch experiments (prompts)
#   scripts/run_all.sh --resume          # analysis + paper (after Modal runs are done)
#   scripts/run_all.sh --analysis-only   # env -> download Zenodo data -> analysis + paper
#   scripts/run_all.sh --force           # skip the y/N confirmation prompt
#   scripts/run_all.sh --dry-run         # print what every downstream script would do
#   scripts/run_all.sh --help

set -euo pipefail

show_help() {
    awk 'NR==1 {next} /^[^#]/ {exit} {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
}

resume=0
analysis_only=0
force=0
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --help|-h)       show_help; exit 0 ;;
        --resume)        resume=1 ;;
        --analysis-only) analysis_only=1 ;;
        --force|-f)      force=1 ;;
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

# Chained downstream scripts that have their own confirmation prompts get
# --force here so the outer prompt covers them and the user isn't prompted
# multiple times.
force_args=(--force)

# ---------------------------------------------------------------------------
# Confirmation prompt — covers the entire downstream chain
# ---------------------------------------------------------------------------
if [[ "${force}" != "1" && "${DRY_RUN}" != "1" ]]; then
    if [[ "${resume}" == "1" ]]; then
        cat <<'EOF'

This will run the analysis pipeline (resume mode):
    - Execute analysis/analysis.ipynb (overwrites figures + tables)
    - Compile paper/paper.tex

Cost:    free (local computation only)
Runtime: ~10 minutes
EOF
    elif [[ "${analysis_only}" == "1" ]]; then
        cat <<'EOF'

This will run the analysis-only pipeline:
    - Set up environment
    - Download experiment results from Zenodo
    - Execute analysis/analysis.ipynb (overwrites figures + tables)
    - Compile paper/paper.tex

Cost:    free (no API calls)
Runtime: ~12 minutes
EOF
    else
        cat <<'EOF'

WARNING: Full pipeline regen — spends real money and takes a long time.
    - Set up environment + prep data + prep embeddings (local, ~hours)
    - Launch experiment orchestrator on Modal (--detach)

Cost:    ~$300-450 OpenAI + ~$5-10 Modal GPU
Runtime: over 24 hours wall time end-to-end
         (this script returns after launching Modal jobs;
          re-invoke with --resume after they complete)
EOF
    fi
    echo
    echo "Use --dry-run to preview commands. Use --force to skip this prompt."
    read -r -p "Continue? [y/N]: " reply
    case "${reply}" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

if [[ "${resume}" == "1" ]]; then
    echo "==> Resume: analysis + paper"
    if [[ "${DRY_RUN}" == "0" ]]; then
        "${SCRIPTS}/run_analysis.sh" "${force_args[@]}"
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
    if [[ "${DRY_RUN}" == "0" ]]; then
        "${SCRIPTS}/run_analysis.sh" "${force_args[@]}"
    else
        echo "    \$ ${SCRIPTS}/run_analysis.sh --force  (no --dry-run support; would execute)"
    fi
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
"${SCRIPTS}/run_experiments.sh" --experiments "${force_args[@]}" "${dry_args[@]+"${dry_args[@]}"}"

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
