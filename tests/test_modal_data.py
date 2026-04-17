"""Modal data-availability + small live experiment runs.

These tests open an ``app.run()`` context from inside the test process and
call deployed Modal functions via ``.remote()``. They require Modal
credentials (~/.modal.toml) and an OpenAI key on the deployed Modal
secret. Skipped cleanly otherwise.

* :class:`ModalConfigShapeUnitTests` — pure-Python shape of the
  ExperimentConfig values used by the smoke runner. No Modal call.
* :class:`ModalVolumeIntegrationTests` — verify the expected data dirs
  are present on the rag-poisoning-data volume.
* :class:`ModalContainerImportIntegrationTests` — call ``run_worker`` with
  zero questions to exercise container setup + imports.
* :class:`ModalVanillaCleanIntegrationTests` — small live run on Modal.
* :class:`ModalAgenticCleanIntegrationTests` — small live run on Modal.
"""

import json
import os
import time
import unittest

import pytest


SMOKE_PREFIX = '_phase4_test_'
TEST_QUERY_IDS = ['test0']


def _modal_credentials_or_skip() -> None:
    if not os.path.exists(os.path.expanduser('~/.modal.toml')):
        raise unittest.SkipTest(
            "Modal credentials not found at ~/.modal.toml. "
            "Run `modal token new` to authenticate."
        )


# ===========================================================================
# Unit suite
# ===========================================================================

class ModalConfigShapeUnitTests(unittest.TestCase):
    """ExperimentConfig values used by the Modal smoke runner — pure shape."""

    def test_smoke_configs_are_well_formed(self):
        from src.experiments.experiment import ExperimentConfig

        config = ExperimentConfig(
            experiment_id=f'{SMOKE_PREFIX}vanilla_clean',
            architecture='vanilla',
            attack_type='clean',
            k=10,
        )
        d = config.to_dict()
        # Worker reconstructs from this dict (drops corpus_type).
        cfg_kwargs = {k: v for k, v in d.items() if k != 'corpus_type'}
        reconstructed = ExperimentConfig(**cfg_kwargs)
        self.assertEqual(reconstructed.experiment_id, config.experiment_id)
        self.assertTrue(d['experiment_id'].startswith(SMOKE_PREFIX),
                        "Smoke configs MUST be prefixed so cleanup is safe")


# ===========================================================================
# Modal helper functions — defined as @app.function so they run in containers.
#
# We attach them to the orchestrator's existing app at import time so
# `app.run()` below brings them online together with run_worker.
# ===========================================================================

from src.experiments.orchestrator import (  # noqa: E402
    app, run_worker, image, volume, VOLUME_MOUNT, EXPERIMENTS_DIR,
)


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60)
def cleanup_test_results() -> list[str]:
    """Delete all _phase4_test_* result directories from the volume."""
    import shutil

    deleted: list[str] = []
    if os.path.isdir(EXPERIMENTS_DIR):
        for name in os.listdir(EXPERIMENTS_DIR):
            if name.startswith(SMOKE_PREFIX):
                shutil.rmtree(os.path.join(EXPERIMENTS_DIR, name))
                deleted.append(name)
    volume.commit()
    return deleted


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60)
def verify_results_on_volume(experiment_id: str, expected_ids: list[str]) -> dict:
    """Read result JSONs from the volume and return a verification summary."""
    volume.reload()
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)

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
    for label, path in [
        ('vector-store', f'{VOLUME_MOUNT}/vector-store'),
        ('original-nq', f'{VOLUME_MOUNT}/original-datasets/nq'),
        ('experiment-datasets', f'{VOLUME_MOUNT}/experiment-datasets'),
    ]:
        exists = os.path.isdir(path)
        file_count = len(os.listdir(path)) if exists else 0
        checks[label] = {'exists': exists, 'file_count': file_count}
    return checks


# ===========================================================================
# Integration suite — every Modal-touching class shares the app.run() context
# via class-level setUp/tearDown. Each class isolates one scenario.
# ===========================================================================

class _ModalRunContextMixin:
    """Class-level ``with app.run()`` lifecycle.

    Subclasses get a live Modal app context for the duration of every
    method in the class, so ``func.remote(...)`` calls work directly.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _modal_credentials_or_skip()
        cls._app_ctx = app.run()
        cls._app_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        ctx = getattr(cls, '_app_ctx', None)
        if ctx is not None:
            ctx.__exit__(None, None, None)
        super().tearDownClass()


@pytest.mark.modal
class ModalVolumeIntegrationTests(_ModalRunContextMixin, unittest.TestCase):
    """Verify expected data dirs exist on the rag-poisoning-data volume."""

    def test_volume_has_required_directories(self):
        checks = check_volume_data.remote()
        for label, info in checks.items():
            self.assertTrue(info['exists'], f"Missing volume data: {label}")


@pytest.mark.modal
class ModalContainerImportIntegrationTests(_ModalRunContextMixin, unittest.TestCase):
    """Container setup + src/ imports run cleanly on the Modal worker image."""

    def test_run_worker_with_empty_question_list(self):
        from src.experiments.experiment import ExperimentConfig
        config = ExperimentConfig(
            experiment_id=f'{SMOKE_PREFIX}import_test',
            architecture='vanilla',
            attack_type='clean',
            k=10,
        )
        # Zero questions = pure container/import smoke.
        summary = run_worker.remote(config.to_dict(), [])
        self.assertEqual(summary['completed'], 0)
        self.assertEqual(summary['errors'], 0)


@pytest.mark.modal
class ModalVanillaCleanIntegrationTests(_ModalRunContextMixin, unittest.TestCase):
    """Vanilla RAG, clean corpus, one question on Modal."""

    def setUp(self):
        super().setUp()
        cleanup_test_results.remote()

    def tearDown(self):
        cleanup_test_results.remote()
        super().tearDown()

    def test_single_question_completes_with_result_on_volume(self):
        from src.experiments.experiment import ExperimentConfig
        config = ExperimentConfig(
            experiment_id=f'{SMOKE_PREFIX}vanilla_clean',
            architecture='vanilla',
            attack_type='clean',
            k=10,
        )
        start = time.time()
        summary = run_worker.remote(config.to_dict(), TEST_QUERY_IDS)
        elapsed = time.time() - start

        self.assertEqual(summary['completed'], len(TEST_QUERY_IDS),
                         f"Expected {len(TEST_QUERY_IDS)} completed in {elapsed:.1f}s")
        self.assertEqual(summary['errors'], 0)

        verification = verify_results_on_volume.remote(config.experiment_id, TEST_QUERY_IDS)
        self.assertTrue(verification['found_dir'])
        for qid, info in verification['results'].items():
            self.assertTrue(info['found'], f"Missing result for {qid}")
            self.assertFalse(info.get('error'), f"Error for {qid}: {info.get('error')}")


@pytest.mark.modal
class ModalAgenticCleanIntegrationTests(_ModalRunContextMixin, unittest.TestCase):
    """Agentic RAG, clean corpus, one question on Modal."""

    def setUp(self):
        super().setUp()
        cleanup_test_results.remote()

    def tearDown(self):
        cleanup_test_results.remote()
        super().tearDown()

    def test_single_question_completes(self):
        from src.experiments.experiment import ExperimentConfig
        config = ExperimentConfig(
            experiment_id=f'{SMOKE_PREFIX}agentic_clean',
            architecture='agentic',
            attack_type='clean',
            k=10,
        )
        summary = run_worker.remote(config.to_dict(), TEST_QUERY_IDS)
        self.assertEqual(summary['completed'], len(TEST_QUERY_IDS))
        self.assertEqual(summary['errors'], 0)
