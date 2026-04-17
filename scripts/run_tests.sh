#!/usr/bin/env bash
#
# Single entry point for the test suite.
#
# Usage:
#   scripts/run_tests.sh                # default: unit + integration (no modal)
#   scripts/run_tests.sh --unit         # unit suite only (no data, no API, no Modal)
#   scripts/run_tests.sh --integration  # integration suite (requires data + OpenAI key)
#   scripts/run_tests.sh --modal        # Modal suite (requires Modal credentials)
#   scripts/run_tests.sh --all          # everything
#
# Any unrecognized args are passed through to pytest, so you can do e.g.:
#   scripts/run_tests.sh --unit -v -k poison
#   scripts/run_tests.sh --integration -x
#
# Run from anywhere — the script cd's to the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTEST="${REPO_ROOT}/venv/bin/pytest"

if [[ ! -x "${PYTEST}" ]]; then
    echo "ERROR: pytest not found at ${PYTEST}" >&2
    echo "       Run: python3 -m venv venv && venv/bin/pip install -e . -r requirements.txt" >&2
    exit 1
fi

cd "${REPO_ROOT}"

mode="default"
extra_args=()

for arg in "$@"; do
    case "${arg}" in
        --unit|--integration|--modal|--all)
            if [[ "${mode}" != "default" ]]; then
                echo "ERROR: only one of --unit / --integration / --modal / --all may be passed" >&2
                exit 1
            fi
            mode="${arg#--}"
            ;;
        *)
            extra_args+=("${arg}")
            ;;
    esac
done

case "${mode}" in
    unit)
        marker_expr='not integration and not modal'
        ;;
    integration)
        marker_expr='integration and not modal'
        ;;
    modal)
        marker_expr='modal'
        ;;
    all)
        marker_expr=''
        ;;
    default)
        marker_expr='not modal'
        ;;
esac

echo "==> mode=${mode}  marker='${marker_expr}'"
# `"${extra_args[@]+"${extra_args[@]}"}"` vanishes when the array is empty —
# required under `set -u`, which otherwise treats an empty array expansion as
# an unset variable and aborts.
if [[ -n "${marker_expr}" ]]; then
    exec "${PYTEST}" tests/ -m "${marker_expr}" "${extra_args[@]+"${extra_args[@]}"}"
else
    exec "${PYTEST}" tests/ "${extra_args[@]+"${extra_args[@]}"}"
fi
