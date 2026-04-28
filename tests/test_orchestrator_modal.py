"""End-to-end Modal smoke for every architecture.

One single-question run per architecture (clean, k=10) verifying the
full container + worker + checkpoint pipeline. All four architectures
should pass in roughly the same shape; per-architecture isolation makes
failures easy to pinpoint.

* :class:`OrchestratorSmokeConfigUnitTests` — pure shape of the smoke
  configs (no Modal call).
* :class:`OrchestratorSmoke<Arch>IntegrationTests` — one class per
  architecture, all marked `modal`. Skipped when credentials are
  missing.
"""

import json
import os
import time
import unittest

import pytest


TEST_QUERY_IDS = ['test0']
SMOKE_PREFIX = '_smoketest_'


def _modal_credentials_or_skip() -> None:
    """Skip the calling test class when Modal credentials aren't configured."""
    if not os.path.exists(os.path.expanduser('~/.modal.toml')):
        raise unittest.SkipTest(
            "Modal credentials not found at ~/.modal.toml. "
            "Run `modal token new` to authenticate."
        )


# ===========================================================================
# Unit suite
# ===========================================================================

class OrchestratorSmokeConfigUnitTests(unittest.TestCase):
    """The per-architecture smoke configs match the production matrix shape."""

    def test_one_smoke_config_per_architecture(self):
        """All four smoke configs round-trip and are correctly prefixed."""
        from src.experiments.experiment import ExperimentConfig

        configs = []
        for arch, k in (('vanilla', 10), ('agentic', 10), ('madam', 10), ('rlm', None)):
            configs.append(ExperimentConfig(
                experiment_id=f'{SMOKE_PREFIX}{arch}_clean',
                architecture=arch,
                attack_type='clean',
                k=k,
            ))

        archs = {c.architecture for c in configs}
        self.assertEqual(archs, {'vanilla', 'agentic', 'madam', 'rlm'})

        # RLM uses k=None; everyone else uses k=10.
        for c in configs:
            if c.architecture == 'rlm':
                self.assertIsNone(c.k)
            else:
                self.assertEqual(c.k, 10)

        # All smoke configs MUST be prefixed for safe cleanup.
        for c in configs:
            self.assertTrue(c.experiment_id.startswith(SMOKE_PREFIX))


# ===========================================================================
# Modal helpers — register on the orchestrator's existing app.
# ===========================================================================

from src.experiments.orchestrator import (  # noqa: E402
    app, run_worker, image, volume, VOLUME_MOUNT, EXPERIMENTS_DIR,
)


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60)
def cleanup_smoke_results() -> list[str]:
    """Delete all _smoketest_* result directories from the volume."""
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
def verify_smoke_results(experiment_id: str, expected_ids: list[str]) -> dict:
    """Read smoke result JSONs and return a verification dict."""
    volume.reload()
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    if not os.path.isdir(exp_dir):
        return {'found_dir': False, 'results': {}}

    results: dict[str, dict] = {}
    for query_id in expected_ids:
        path = os.path.join(exp_dir, f'{query_id}.json')
        if not os.path.exists(path):
            results[query_id] = {'found': False}
            continue
        with open(path) as f:
            data = json.load(f)
        results[query_id] = {
            'found': True,
            'answer': data.get('system_answer', '')[:120],
            'error': data.get('error'),
            'latency': data.get('latency_seconds'),
        }
    return {'found_dir': True, 'results': results}


# ===========================================================================
# Integration suite — one class per architecture
# ===========================================================================

class _OrchestratorSmokeMixin:
    """Shared lifecycle for the per-architecture orchestrator smoke tests.

    Opens an `app.run()` context at the class level and cleans up
    `_smoketest_*` directories before and after each test.

    Attributes:
        architecture: Architecture name set by each subclass
            (`vanilla`, `agentic`, `madam`, `rlm`).
        k: Top-K value for the smoke run. `None` for RLM, `10`
            otherwise.
    """

    architecture: str
    k: int | None

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

    def setUp(self):
        super().setUp()
        cleanup_smoke_results.remote()

    def tearDown(self):
        cleanup_smoke_results.remote()
        super().tearDown()

    def _run_smoke(self) -> None:
        """Run the architecture's single-question smoke + verify the result."""
        from src.experiments.experiment import ExperimentConfig

        config = ExperimentConfig(
            experiment_id=f'{SMOKE_PREFIX}{self.architecture}_clean',
            architecture=self.architecture,
            attack_type='clean',
            k=self.k,
        )

        start = time.time()
        summary = run_worker.remote(config.to_dict(), TEST_QUERY_IDS)
        elapsed = time.time() - start

        self.assertEqual(
            summary['completed'], len(TEST_QUERY_IDS),
            f"{self.architecture}: expected {len(TEST_QUERY_IDS)} completed "
            f"in {elapsed:.1f}s, got summary={summary}",
        )
        self.assertEqual(summary['errors'], 0,
                         f"{self.architecture}: errors={summary['errors']}")

        verification = verify_smoke_results.remote(config.experiment_id, TEST_QUERY_IDS)
        self.assertTrue(verification['found_dir'],
                        f"{self.architecture}: no results dir on volume")
        for query_id, info in verification['results'].items():
            self.assertTrue(info['found'], f"{self.architecture}/{query_id}: missing")
            self.assertFalse(info.get('error'),
                             f"{self.architecture}/{query_id}: {info.get('error')}")


@pytest.mark.modal
class OrchestratorSmokeVanillaIntegrationTests(_OrchestratorSmokeMixin, unittest.TestCase):
    """One-question Vanilla RAG smoke run on Modal."""
    architecture = 'vanilla'
    k = 10

    def test_single_question_completes(self):
        """Vanilla one-question smoke completes with no errors on Modal."""
        self._run_smoke()


@pytest.mark.modal
class OrchestratorSmokeAgenticIntegrationTests(_OrchestratorSmokeMixin, unittest.TestCase):
    """One-question Agentic RAG smoke run on Modal."""
    architecture = 'agentic'
    k = 10

    def test_single_question_completes(self):
        """Agentic one-question smoke completes with no errors on Modal."""
        self._run_smoke()


@pytest.mark.modal
class OrchestratorSmokeMADAMIntegrationTests(_OrchestratorSmokeMixin, unittest.TestCase):
    """One-question MADAM-RAG smoke run on Modal."""
    architecture = 'madam'
    k = 10

    def test_single_question_completes(self):
        """MADAM-RAG one-question smoke completes with no errors on Modal."""
        self._run_smoke()


@pytest.mark.modal
class OrchestratorSmokeRLMIntegrationTests(_OrchestratorSmokeMixin, unittest.TestCase):
    """One-question RLM smoke run on Modal."""
    architecture = 'rlm'
    k = None

    def test_single_question_completes(self):
        """RLM one-question smoke completes with no errors on Modal."""
        self._run_smoke()
