"""
experiments/upload_data.py

Upload pre-computed local data to a Modal Volume so experiment workers
can access it without baking multi-GB data into the container image.

Uploads:
    data/vector-store/           → /vol/vector-store/
    data/original-datasets/nq/   → /vol/original-datasets/nq/
    data/experiment-datasets/    → /vol/experiment-datasets/

Prerequisites:
    python src/data/create_questions.py          # builds experiment-datasets/nq-questions.jsonl
    python src/data/filter_gold_questions.py     # builds experiment-datasets/nq-questions-gold-filtered.jsonl

Usage:
    python experiments/upload_data.py
    python experiments/upload_data.py --force   # re-upload everything

Uses `modal volume put` per file via subprocess. Previous version used Modal's
Python batch_upload API but it choked on 30GB+ uploads with connection errors.
Archived as upload_data_ARCHIVED_batch_upload.py.
"""

import argparse
import os
import subprocess
import time


VOLUME_NAME = 'rag-poisoning-data'

# ---------------------------------------------------------------------------
# Local paths (relative to src/)
# ---------------------------------------------------------------------------

_SRC_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

_UPLOAD_DIRS: list[tuple[str, str]] = [
    # (local_dir, remote_dir_on_volume)
    (
        os.path.join(_SRC_ROOT, 'data', 'vector-store'),
        'vector-store',
    ),
    (
        os.path.join(_SRC_ROOT, 'data', 'original-datasets', 'nq'),
        'original-datasets/nq',
    ),
    (
        os.path.join(_SRC_ROOT, 'data', 'experiment-datasets'),
        'experiment-datasets',
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Files to skip during upload — build artifacts not needed at runtime.
_SKIP_SUFFIXES = {
    # Raw embedding pickles (~10 GB each). The .faiss indexes are the
    # compiled form; the pickles are only consumed by build_vector_indexes.py.
    '-documents-embeddings.pkl',
}

# Substrings that mark archived/backup paths — skip entire directories and files.
_ARCHIVE_MARKERS = {'ARCHIVE', 'BACKUP'}


def _is_archived(path: str) -> bool:
    """Check if any component of a path contains an archive/backup marker."""
    return any(marker in path.upper() for marker in _ARCHIVE_MARKERS)


def collect_local_files(local_dir: str) -> list[tuple[str, str]]:
    """Walk a local directory and return (local_path, relative_path) pairs."""
    pairs: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(local_dir):
        if _is_archived(root):
            continue
        for fname in files:
            if fname == '.DS_Store':
                continue
            if _is_archived(fname):
                continue
            if any(fname.endswith(suffix) for suffix in _SKIP_SUFFIXES):
                continue
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir)
            pairs.append((local_path, rel))
    return pairs


def fmt_size(nbytes: int | float) -> str:
    """Human-readable file size."""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def get_remote_files() -> set[str]:
    """List all files currently on the Modal volume."""
    result = subprocess.run(
        ['modal', 'volume', 'ls', VOLUME_NAME, '--json'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Volume might be empty or --json not supported; fall back to empty set
        return set()
    import json
    try:
        entries = json.loads(result.stdout)
        return {entry['filename'] for entry in entries if entry['type'] == 'file'}
    except (json.JSONDecodeError, KeyError):
        return set()


def upload_file(local_path: str, remote_path: str) -> bool:
    """Upload a single file via `modal volume put`. Returns True on success."""
    result = subprocess.run(
        ['modal', 'volume', 'put', VOLUME_NAME, local_path, remote_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload experiment data to Modal Volume")
    parser.add_argument('--force', action='store_true', help="Re-upload all files, even if they already exist")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Poisoning Bench — Data Upload")
    print("=" * 60)
    print()

    # --- Preflight: check that required question files have been built --------
    qjsonl_path = os.path.join(
        _SRC_ROOT, 'data', 'experiment-datasets', 'nq-questions.jsonl'
    )
    if not os.path.exists(qjsonl_path):
        print("ERROR: nq-questions.jsonl not found. Run the data prep step first:")
        print("  python src/data/create_questions.py")
        raise SystemExit(1)

    gold_filtered_path = os.path.join(
        _SRC_ROOT, 'data', 'experiment-datasets', 'nq-questions-gold-filtered.jsonl'
    )
    if not os.path.exists(gold_filtered_path):
        print("ERROR: nq-questions-gold-filtered.jsonl not found. Run:")
        print("  python src/data/filter_gold_questions.py")
        raise SystemExit(1)

    print("Preflight OK: nq-questions.jsonl and nq-questions-gold-filtered.jsonl found")
    print()

    # --- Upload data directories via `modal volume put` per file -------------
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0

    for local_dir, remote_prefix in _UPLOAD_DIRS:
        if not os.path.isdir(local_dir):
            print(f"WARNING: {local_dir} not found, skipping")
            continue

        files = collect_local_files(local_dir)
        total_bytes = sum(os.path.getsize(p) for p, _ in files)
        print(f"Processing {remote_prefix}/ ({len(files)} files, {fmt_size(total_bytes)})")

        t0 = time.time()
        dir_uploaded = 0
        dir_skipped = 0

        for i, (local_path, rel) in enumerate(files, 1):
            remote_path = f'{remote_prefix}/{rel}'
            file_size = os.path.getsize(local_path)

            if not args.force:
                # Check if file exists on volume by attempting ls
                # (Not perfectly efficient, but simple and reliable)
                check = subprocess.run(
                    ['modal', 'volume', 'ls', VOLUME_NAME, remote_path],
                    capture_output=True, text=True,
                )
                if check.returncode == 0 and remote_path.split('/')[-1] in check.stdout:
                    dir_skipped += 1
                    total_skipped += 1
                    continue

            print(f"  [{i}/{len(files)}] {remote_path} ({fmt_size(file_size)})")
            if upload_file(local_path, remote_path):
                dir_uploaded += 1
                total_uploaded += 1
            else:
                total_failed += 1

        elapsed = time.time() - t0
        print(f"  {dir_uploaded} uploaded, {dir_skipped} skipped in {elapsed:.1f}s")
        print()

    print()
    print("=" * 60)
    print(f"Upload complete: {total_uploaded} uploaded, {total_skipped} skipped, {total_failed} failed")
    print()
    print("Verify with:")
    print(f"  modal volume ls {VOLUME_NAME}")
    print(f"  modal volume ls {VOLUME_NAME} vector-store/")
    print(f"  modal volume ls {VOLUME_NAME} experiment-datasets/")
    print(f"  modal volume ls {VOLUME_NAME} results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
