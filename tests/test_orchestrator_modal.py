"""
Modal integration smoke test for the orchestrator pipeline.

Runs a single 3-question batch through each architecture (clean, k=10)
to verify the end-to-end Modal worker pipeline: container setup, data
loading from Volume, architecture execution, and result checkpointing.

RLM is expected to fail until the Modal image installs it correctly.

Prerequisite:
    modal run experiments/upload_data.py

Run from repo root:
    modal run tests/test_orchestrator_modal.py::app.smoke_test
"""

import json
import os
import sys
import time

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

TEST_QUERY_IDS = ['test0', 'test1', 'test2']
SMOKE_PREFIX = '_smoketest_'

# Architectures where we expect failure (e.g. missing rlm install).
EXPECTED_FAILURES = {}


# ---------------------------------------------------------------------------
# Smoke-test configs
# ---------------------------------------------------------------------------

def build_smoke_configs() -> list[ExperimentConfig]:
    """One clean experiment per architecture, k=10 (fixed)."""
    configs: list[ExperimentConfig] = []
    for arch, k in [('vanilla', 10), ('agentic', 10), ('madam', 10), ('rlm', None)]:
        configs.append(
            ExperimentConfig(
                experiment_id=f'{SMOKE_PREFIX}{arch}_clean',
                architecture=arch,
                attack_type='clean',
                k=k,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Helper Modal functions (volume access)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60)
def cleanup_smoke_results() -> list[str]:
    """Delete all _smoketest_* result directories from the volume."""
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
def verify_results_on_volume(
    experiment_id: str,
    expected_ids: list[str],
) -> dict:
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke_test():
    configs = build_smoke_configs()

    print(f"{'=' * 60}")
    print("Modal Integration Smoke Test")
    print(f"  Architectures: {[c.architecture for c in configs]}")
    print(f"  Questions:     {TEST_QUERY_IDS}")
    print(f"  Attack:        clean")
    print(f"{'=' * 60}")

    # ── Clean up stale smoketest results ──────────────────────────
    print("\nCleaning up previous smoketest results...")
    deleted = cleanup_smoke_results.remote()
    if deleted:
        print(f"  Deleted: {deleted}")
    else:
        print("  (none found)")

    # ── Run each architecture ─────────────────────────────────────
    passed: list[str] = []
    failed: list[str] = []
    xfailed: list[str] = []

    for config in configs:
        arch = config.architecture
        print(f"\n{'─' * 60}")
        print(f"[{arch}] experiment_id={config.experiment_id}  k={config.k or 'all'}")

        config_dict = config.to_dict()
        start = time.time()

        try:
            summary = run_worker.remote(config_dict, TEST_QUERY_IDS)
            elapsed = time.time() - start

            print(f"  Worker returned in {elapsed:.1f}s")
            print(
                f"  completed={summary['completed']}  "
                f"skipped={summary['skipped']}  "
                f"errors={summary['errors']}  "
                f"total={summary['total']}"
            )

            # Verify actual result files on the volume.
            verification = verify_results_on_volume.remote(
                config.experiment_id, TEST_QUERY_IDS,
            )
            if not verification['found_dir']:
                print("  Volume verification: result directory not found!")
            else:
                for qid, info in verification['results'].items():
                    if not info.get('found'):
                        print(f"    {qid}: MISSING")
                    elif info.get('error'):
                        err_preview = info['error'][:100].replace('\n', ' ')
                        print(f"    {qid}: ERROR — {err_preview}")
                    else:
                        lat = info.get('latency', 0)
                        ans = info.get('answer', '')[:80]
                        print(f"    {qid}: OK ({lat:.1f}s) — {ans}")

            # Decide pass/fail.
            if summary['errors'] > 0 or summary['completed'] != len(TEST_QUERY_IDS):
                if arch in EXPECTED_FAILURES:
                    xfailed.append(arch)
                    print(f"  -> XFAIL (expected for {arch})")
                else:
                    failed.append(arch)
                    print(f"  -> FAILED")
            else:
                passed.append(arch)
                if arch in EXPECTED_FAILURES:
                    # Unexpectedly passed — good news, promote it.
                    print(f"  -> PASSED (was expected to fail — nice!)")
                else:
                    print(f"  -> PASSED")

        except Exception as e:
            elapsed = time.time() - start
            err_msg = str(e)[:200].replace('\n', ' ')
            print(f"  Exception after {elapsed:.1f}s: {err_msg}")

            if arch in EXPECTED_FAILURES:
                xfailed.append(arch)
                print(f"  -> XFAIL (expected for {arch})")
            else:
                failed.append(arch)
                print(f"  -> FAILED")

    # ── Cleanup ───────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Cleaning up smoketest results...")
    deleted = cleanup_smoke_results.remote()
    print(f"  Deleted: {deleted or '(none)'}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS")
    if passed:
        print(f"  Passed:        {', '.join(passed)}")
    if xfailed:
        print(f"  Expected fail: {', '.join(xfailed)}")
    if failed:
        print(f"  FAILED:        {', '.join(failed)}")
    print(f"{'=' * 60}")

    if failed:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED (expected failures excluded)")
