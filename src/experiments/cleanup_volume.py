"""
experiments/cleanup_volume.py

One-shot Modal script to clean up experiment results before a re-run.

Actions:
  1. DELETE all RLM result directories entirely (rlm_clean, rlm_naive,
     rlm_corruptrag_ak) — RLM needs a full re-run because of the silent
     failure path in the rlm library's _llm_query().
  2. DELETE only error result files from MADAM directories (madam_clean,
     madam_naive, madam_corruptrag_ak) — keep existing successes.
  3. DELETE stale summary.json files for all affected experiments so the
     orchestrator writes fresh ones.

Usage:
    modal run experiments/cleanup_volume.py
"""

import json
import os
import shutil

import modal


# Re-use the same app name / volume as orchestrator.
app = modal.App('rag-poisoning-bench-cleanup')

volume = modal.Volume.from_name('rag-poisoning-data', create_if_missing=False)
VOLUME_MOUNT = '/vol'
RESULTS_DIR = f'{VOLUME_MOUNT}/results'


# Experiments to fully delete (RLM — re-run from scratch).
RLM_EXPERIMENTS = [
    'rlm_clean',
    'rlm_naive',
    'rlm_corruptrag_ak',
]

# Experiments where we delete only error results (MADAM — keep successes).
MADAM_EXPERIMENTS = [
    'madam_clean',
    'madam_naive',
    'madam_corruptrag_ak',
]


@app.function(
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 10,
)
def cleanup():
    """Clean up experiment results on the Modal volume."""
    print(f"{'=' * 60}")
    print("Volume cleanup — preparing for re-run")
    print(f"{'=' * 60}")
    print()

    # --- RLM: delete entire directories ---
    for exp_id in RLM_EXPERIMENTS:
        exp_dir = os.path.join(RESULTS_DIR, exp_id)
        if not os.path.isdir(exp_dir):
            print(f"[RLM] {exp_id}: directory not found, nothing to delete")
            continue

        n_files = len([f for f in os.listdir(exp_dir) if f.endswith('.json')])
        shutil.rmtree(exp_dir)
        print(f"[RLM] {exp_id}: deleted directory ({n_files} files)")

    print()

    # --- MADAM: delete error results only, keep successes ---
    for exp_id in MADAM_EXPERIMENTS:
        exp_dir = os.path.join(RESULTS_DIR, exp_id)
        if not os.path.isdir(exp_dir):
            print(f"[MADAM] {exp_id}: directory not found, nothing to delete")
            continue

        n_success = 0
        n_error_deleted = 0
        n_corrupt_deleted = 0

        for filename in os.listdir(exp_dir):
            if not filename.endswith('.json') or filename == 'summary.json':
                continue

            filepath = os.path.join(exp_dir, filename)
            try:
                with open(filepath) as f:
                    result = json.loads(f.read())

                if result.get('error') is None:
                    n_success += 1
                else:
                    os.remove(filepath)
                    n_error_deleted += 1
            except (json.JSONDecodeError, OSError):
                os.remove(filepath)
                n_corrupt_deleted += 1

        # Delete stale summary.json
        summary_path = os.path.join(exp_dir, 'summary.json')
        if os.path.exists(summary_path):
            os.remove(summary_path)

        print(
            f"[MADAM] {exp_id}: kept {n_success} successes, "
            f"deleted {n_error_deleted} errors"
            + (f", {n_corrupt_deleted} corrupt" if n_corrupt_deleted else "")
            + ", deleted summary.json"
        )

    # Commit changes so they're visible everywhere.
    volume.commit()

    print()
    print(f"{'=' * 60}")
    print("Cleanup complete. Ready for re-run.")
    print(f"{'=' * 60}")


@app.local_entrypoint()
def main():
    """Run cleanup on a Modal container."""
    cleanup.remote()
