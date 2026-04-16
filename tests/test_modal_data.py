"""
Phase 4 migration test: verify Modal data scripts and experiment pipeline.

Requires Modal credentials, OpenAI API key, and branch pushed to GitHub.
Tests progress from dry-run (no API cost) to small live batches.

Run from repo root:
    modal run tests/test_modal_data.py::app.run_all
    modal run tests/test_modal_data.py::app.dry_runs_only
"""

import json
import os
import sys
import time

import modal

from src.experiments.orchestrator import (
    app,
    run_worker,
    image,
    volume,
    VOLUME_MOUNT,
    RESULTS_DIR,
)
from src.experiments.experiment import ExperimentConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMOKE_PREFIX = '_phase4_test_'
TEST_QUERY_IDS = ['test0', 'test1']


# ---------------------------------------------------------------------------
# Helper functions (on Modal)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60)
def cleanup_test_results() -> list[str]:
    """Delete all _phase4_test_* result directories from the volume."""
    import shutil

    deleted: list[str] = []
    if os.path.isdir(RESULTS_DIR):
        for name in os.listdir(RESULTS_DIR):
            if name.startswith(SMOKE_PREFIX):
                shutil.rmtree(os.path.join(RESULTS_DIR, name))
                deleted.append(name)
    volume.commit()
    return deleted


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60)
def verify_results_on_volume(experiment_id: str, expected_ids: list[str]) -> dict:
    """Read result JSONs from the volume and return a verification summary."""
    volume.reload()
    exp_dir = os.path.join(RESULTS_DIR, experiment_id)

    if not os.path.isdir(exp_dir):
        return {'found_dir': False, 'results': {}}

    results: dict[str, dict] = {}
    for qid in expected_ids:
        path = os.path.join(exp_dir, f'{qid}.json')
        if not os.path.exists(path):
            results[qid] = {'found': False}
            continue

        with open(path) as f:
            data = json.load(f)
        results[qid] = {
            'found': True,
            'answer': data.get('system_answer', '')[:120],
            'error': data.get('error'),
            'latency': data.get('latency_seconds'),
        }

    return {'found_dir': True, 'results': results}


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60)
def check_volume_data() -> dict:
    """Check if expected data exists on the Modal volume."""
    volume.reload()

    checks = {}
    for path_label, path in [
        ('vector-store', f'{VOLUME_MOUNT}/vector-store'),
        ('original-nq', f'{VOLUME_MOUNT}/original-datasets/nq'),
        ('experiment-datasets', f'{VOLUME_MOUNT}/experiment-datasets'),
    ]:
        exists = os.path.isdir(path)
        file_count = len(os.listdir(path)) if exists else 0
        checks[path_label] = {'exists': exists, 'file_count': file_count}

    return checks


# ---------------------------------------------------------------------------
# Test: dry runs (no API cost)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def dry_runs_only():
    """Run only the dry-run tests (no API cost)."""
    print(f"{'=' * 60}")
    print("Phase 4 Modal Tests — DRY RUN ONLY")
    print(f"{'=' * 60}")

    test_volume_data()
    test_import_on_container()

    print(f"\n{'=' * 60}")
    print("DRY RUN TESTS PASSED")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Test: full suite
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_all():
    """Run all Modal tests including live API calls."""
    print(f"{'=' * 60}")
    print("Phase 4 Modal Tests — FULL SUITE")
    print(f"  Test questions: {TEST_QUERY_IDS}")
    print(f"{'=' * 60}")

    test_volume_data()
    test_import_on_container()
    test_vanilla_clean()
    test_agentic_clean()

    print(f"\n{'=' * 60}")
    print("ALL MODAL TESTS PASSED")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_volume_data():
    """Verify expected data exists on Modal volume."""
    print(f"\n{'─' * 60}")
    print("[test_volume_data]")

    checks = check_volume_data.remote()
    for label, info in checks.items():
        status = "OK" if info['exists'] else "MISSING"
        print(f"  {status}  {label} ({info['file_count']} items)")
        assert info['exists'], f"Missing volume data: {label}"

    print("  PASSED")


def test_import_on_container():
    """Verify all src modules import on the Modal container."""
    print(f"\n{'─' * 60}")
    print("[test_import_on_container]")

    # The run_worker function imports the full stack on the container.
    # If it can be called (even with an error), imports work.
    config = ExperimentConfig(
        experiment_id=f'{SMOKE_PREFIX}import_test',
        architecture='vanilla',
        attack_type='clean',
        k=10,
    )

    # Run with 0 questions — just tests container setup + imports
    try:
        summary = run_worker.remote(config.to_dict(), [])
        print(f"  Worker returned: {summary}")
        assert summary['completed'] == 0
        assert summary['errors'] == 0
        print("  PASSED")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def test_vanilla_clean():
    """Run vanilla RAG on 2 questions via Modal worker."""
    print(f"\n{'─' * 60}")
    print(f"[test_vanilla_clean] questions={TEST_QUERY_IDS}")

    config = ExperimentConfig(
        experiment_id=f'{SMOKE_PREFIX}vanilla_clean',
        architecture='vanilla',
        attack_type='clean',
        k=10,
    )

    # Cleanup any previous test results
    cleanup_test_results.remote()

    start = time.time()
    summary = run_worker.remote(config.to_dict(), TEST_QUERY_IDS)
    elapsed = time.time() - start

    print(f"  Worker returned in {elapsed:.1f}s")
    print(f"  completed={summary['completed']}  errors={summary['errors']}")

    assert summary['completed'] == len(TEST_QUERY_IDS), f"Expected {len(TEST_QUERY_IDS)} completed"
    assert summary['errors'] == 0, f"Unexpected errors: {summary['errors']}"

    # Verify results on volume
    verification = verify_results_on_volume.remote(config.experiment_id, TEST_QUERY_IDS)
    assert verification['found_dir'], "Result directory not found on volume"

    for qid, info in verification['results'].items():
        assert info['found'], f"Missing result for {qid}"
        assert not info.get('error'), f"Error for {qid}: {info.get('error')}"
        print(f"  {qid}: OK ({info.get('latency', 0):.1f}s) — {info.get('answer', '')[:60]}")

    # Cleanup
    cleanup_test_results.remote()
    print("  PASSED")


def test_agentic_clean():
    """Run agentic RAG on 2 questions via Modal worker."""
    print(f"\n{'─' * 60}")
    print(f"[test_agentic_clean] questions={TEST_QUERY_IDS}")

    config = ExperimentConfig(
        experiment_id=f'{SMOKE_PREFIX}agentic_clean',
        architecture='agentic',
        attack_type='clean',
        k=10,
    )

    cleanup_test_results.remote()

    start = time.time()
    summary = run_worker.remote(config.to_dict(), TEST_QUERY_IDS)
    elapsed = time.time() - start

    print(f"  Worker returned in {elapsed:.1f}s")
    print(f"  completed={summary['completed']}  errors={summary['errors']}")

    assert summary['completed'] == len(TEST_QUERY_IDS), f"Expected {len(TEST_QUERY_IDS)} completed"
    assert summary['errors'] == 0, f"Unexpected errors: {summary['errors']}"

    # Cleanup
    cleanup_test_results.remote()
    print("  PASSED")
