#!/usr/bin/env bash
#
# Download the research-data bundle from Zenodo and unpack it into the
# repo's expected directory layout.
#
# Zenodo DOI: 10.5281/zenodo.19582217
# File:       rag-poisoning-architecture-bench-data.zip (~40 MB)
#
# The zip contains:
#   human_labels.csv
#   results/experiments/   (12 experiment dirs, ~173 MB)
#   results/judge/         (12 experiment dirs, ~43 MB)
#   results/noise/         (~4.5 MB)
#   data/nq-incorrect-answers-poisoned-docs.jsonl
#   data/nq-corruptrag-ak-poisoned-docs.jsonl
#   data/nq-correct-answers.jsonl
#
# These unpack to:
#   analysis/human_labels.csv
#   src/experiments/results/{experiments,judge,noise}/
#   src/data/experiment-datasets/*.jsonl
#
# Usage:
#   scripts/download_data.sh              # default
#   scripts/download_data.sh --force      # overwrite existing targets
#   scripts/download_data.sh --keep-zip   # keep the downloaded .zip after unpack
#   scripts/download_data.sh --help

set -euo pipefail

show_help() {
    awk 'NR==1 {next} /^[^#]/ {exit} {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
}

force=0
keep_zip=0
for arg in "$@"; do
    case "${arg}" in
        --help|-h)  show_help; exit 0 ;;
        --force)    force=1 ;;
        --keep-zip) keep_zip=1 ;;
        *)          echo "ERROR: unknown arg '${arg}'" >&2; show_help >&2; exit 2 ;;
    esac
done

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

# Zenodo has two IDs: the concept DOI (citation; resolves to the latest
# version) and a version-specific record ID where files actually live.
# DOI 10.5281/zenodo.19582217 redirects to record 19582218.
ZENODO_VERSION_ID="19582218"
ZIP_NAME="rag-poisoning-architecture-bench-data.zip"
ZENODO_URL="https://zenodo.org/records/${ZENODO_VERSION_ID}/files/${ZIP_NAME}"

STAGING_DIR="${REPO_ROOT}/.zenodo-staging"
ZIP_PATH="${STAGING_DIR}/${ZIP_NAME}"

# Destination mapping: <zip-relative-source> -> <repo-relative-destination>.
# All destinations are directories (or a single file, handled specially).
EXPERIMENTS_DEST="${REPO_ROOT}/src/experiments/results/experiments"
JUDGE_DEST="${REPO_ROOT}/src/experiments/results/judge"
NOISE_DEST="${REPO_ROOT}/src/experiments/results/noise"
DATA_DEST="${REPO_ROOT}/src/data/experiment-datasets"
HUMAN_LABELS_DEST="${REPO_ROOT}/analysis/human_labels.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Return 0 iff $1 is a directory and contains any entry other than .gitkeep.
is_populated_dir() {
    local dir="$1"
    [[ -d "${dir}" ]] || return 1
    local n
    n=$(find "${dir}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' | wc -l | tr -d ' ')
    [[ "${n}" -gt 0 ]]
}

abort_if_populated() {
    local dir="$1"
    if is_populated_dir "${dir}"; then
        echo "ERROR: ${dir} is already populated." >&2
        echo "       Re-run with --force to overwrite, or clear it manually." >&2
        exit 1
    fi
}

# Move the contents of $1 into $2, preserving $2/.gitkeep if present.
# Removes $1 afterward. Overwrites any colliding children under $2.
merge_into() {
    local src_dir="$1"
    local dst_dir="$2"

    mkdir -p "${dst_dir}"
    # Preserve the placeholder so the dir stays tracked if results are ever removed.
    local preserve=""
    if [[ -f "${dst_dir}/.gitkeep" ]]; then preserve="yes"; fi

    # Move every top-level child; -T removes dst_dir on rename when using mv
    # of a whole dir, so we iterate instead.
    find "${src_dir}" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' child; do
        local base; base="$(basename "${child}")"
        # :? ensures dst_dir is non-empty, guarding against `rm -rf /${base}`.
        rm -rf "${dst_dir:?}/${base}"
        mv "${child}" "${dst_dir}/${base}"
    done
    rmdir "${src_dir}"

    if [[ "${preserve}" == "yes" && ! -f "${dst_dir}/.gitkeep" ]]; then
        touch "${dst_dir}/.gitkeep"
    fi
}

cleanup() {
    if [[ "${keep_zip}" == "0" && -d "${STAGING_DIR}" ]]; then
        rm -rf "${STAGING_DIR}"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-flight: refuse to clobber existing data unless --force
# ---------------------------------------------------------------------------
if [[ "${force}" == "0" ]]; then
    abort_if_populated "${EXPERIMENTS_DEST}"
    abort_if_populated "${JUDGE_DEST}"
    abort_if_populated "${NOISE_DEST}"
    if [[ -f "${HUMAN_LABELS_DEST}" ]]; then
        echo "ERROR: ${HUMAN_LABELS_DEST} already exists." >&2
        echo "       Re-run with --force to overwrite." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
mkdir -p "${STAGING_DIR}"
echo "==> Downloading ${ZIP_NAME} from Zenodo"
echo "    ${ZENODO_URL}"
curl --fail --location --show-error --output "${ZIP_PATH}" "${ZENODO_URL}"

# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------
echo "==> Unpacking"
unzip -q -o "${ZIP_PATH}" -d "${STAGING_DIR}"

# Sanity-check the unpack layout
for required in results/experiments results/judge results/noise data human_labels.csv; do
    if [[ ! -e "${STAGING_DIR}/${required}" ]]; then
        echo "ERROR: Zenodo bundle is missing expected entry: ${required}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Distribute into repo layout
# ---------------------------------------------------------------------------
echo "==> Installing results/experiments -> ${EXPERIMENTS_DEST#"${REPO_ROOT}"/}"
merge_into "${STAGING_DIR}/results/experiments" "${EXPERIMENTS_DEST}"

echo "==> Installing results/judge       -> ${JUDGE_DEST#"${REPO_ROOT}"/}"
merge_into "${STAGING_DIR}/results/judge" "${JUDGE_DEST}"

echo "==> Installing results/noise       -> ${NOISE_DEST#"${REPO_ROOT}"/}"
merge_into "${STAGING_DIR}/results/noise" "${NOISE_DEST}"

# results/ is now empty — drop it.
rmdir "${STAGING_DIR}/results"

echo "==> Installing data/*.jsonl         -> ${DATA_DEST#"${REPO_ROOT}"/}"
mkdir -p "${DATA_DEST}"
find "${STAGING_DIR}/data" -mindepth 1 -maxdepth 1 -type f -print0 \
    | while IFS= read -r -d '' f; do
        mv "${f}" "${DATA_DEST}/$(basename "${f}")"
    done
rmdir "${STAGING_DIR}/data"

echo "==> Installing human_labels.csv     -> ${HUMAN_LABELS_DEST#"${REPO_ROOT}"/}"
mv "${STAGING_DIR}/human_labels.csv" "${HUMAN_LABELS_DEST}"

echo
echo "Done."
echo "  Experiment results: $(find "${EXPERIMENTS_DEST}" -type f | wc -l | tr -d ' ') files"
echo "  Judge results:      $(find "${JUDGE_DEST}" -type f | wc -l | tr -d ' ') files"
echo "  Noise results:      $(find "${NOISE_DEST}" -type f | wc -l | tr -d ' ') files"
echo "  Data JSONLs:        $(find "${DATA_DEST}" -maxdepth 1 -type f -name '*.jsonl' | wc -l | tr -d ' ') files"
