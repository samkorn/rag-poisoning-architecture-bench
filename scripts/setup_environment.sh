#!/usr/bin/env bash
#
# Create the project's Python 3.12 venv, install dependencies, and check
# that Modal + OpenAI credentials are available.
#
# Usage:
#   scripts/setup_environment.sh            # default
#   scripts/setup_environment.sh --force    # recreate venv from scratch
#   scripts/setup_environment.sh --help
#
# Idempotent: skips venv creation if venv/ already exists (unless --force).
# Always re-runs pip install so new requirements pick up on subsequent calls.

set -euo pipefail

show_help() {
    sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

force=0
for arg in "$@"; do
    case "${arg}" in
        --help|-h) show_help; exit 0 ;;
        --force)   force=1 ;;
        *)         echo "ERROR: unknown arg '${arg}'" >&2; show_help >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

# ---------------------------------------------------------------------------
# Locate a system Python 3.12
# ---------------------------------------------------------------------------
PYTHON312=""
for candidate in python3.12 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        ver="$("${candidate}" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
        if [[ "${ver}" == "3.12" ]]; then
            PYTHON312="$(command -v "${candidate}")"
            break
        fi
    fi
done
if [[ -z "${PYTHON312}" ]]; then
    echo "ERROR: Python 3.12 not found on PATH." >&2
    echo "       Install it (e.g. 'brew install python@3.12') and retry." >&2
    exit 1
fi
echo "==> Using system Python: ${PYTHON312}"

# ---------------------------------------------------------------------------
# Create venv
# ---------------------------------------------------------------------------
if [[ -d "${VENV_DIR}" && "${force}" == "1" ]]; then
    echo "==> Removing existing venv (--force)"
    rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Creating venv at ${VENV_DIR}"
    "${PYTHON312}" -m venv "${VENV_DIR}"
else
    echo "==> Reusing existing venv at ${VENV_DIR}"
fi

# ---------------------------------------------------------------------------
# Install dependencies (editable project + requirements)
# ---------------------------------------------------------------------------
echo "==> Upgrading pip"
"${PIP_BIN}" install --quiet --upgrade pip

echo "==> Installing project (editable) + requirements.txt"
"${PIP_BIN}" install --quiet -e . -r requirements.txt

# ---------------------------------------------------------------------------
# Credential / config sanity checks (warn-only; don't fail the setup)
# ---------------------------------------------------------------------------
echo
echo "==> Credential checks"

if [[ -f "${HOME}/.modal.toml" ]]; then
    echo "  OK    Modal config found (~/.modal.toml)"
else
    echo "  WARN  Modal config missing — run: ${VENV_DIR}/bin/modal token new"
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "  OK    OPENAI_API_KEY is set in environment"
elif [[ -f "${REPO_ROOT}/.env" ]] && grep -q '^OPENAI_API_KEY=' "${REPO_ROOT}/.env"; then
    echo "  OK    OPENAI_API_KEY found in .env"
else
    echo "  WARN  OPENAI_API_KEY not set — export it or add to .env before running experiments"
fi

if command -v /Library/TeX/texbin/pdflatex >/dev/null 2>&1; then
    echo "  OK    pdflatex found at /Library/TeX/texbin/pdflatex"
else
    echo "  WARN  pdflatex not found at /Library/TeX/texbin/pdflatex — paper compilation will fail"
fi

echo
echo "Done. Activate with:  source venv/bin/activate"
