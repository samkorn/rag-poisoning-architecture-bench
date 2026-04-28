"""Unit tests for `src.experiments.orchestrator` (pure-Python helpers).

Covers `build_experiment_matrix()` shape/contents and
`is_experiment_complete()` file-count logic. Also round-trips
`ExperimentConfig` through `to_dict()` / reconstruction (the
serialization contract the Modal worker relies on).

No data, no Modal, no API calls — runs in the unit suite.
"""

import json
import os
import re
import shutil
import tempfile
import unittest

from src.experiments.experiment import ExperimentConfig
from src.experiments.orchestrator import build_experiment_matrix, is_experiment_complete


class ExperimentMatrixUnitTests(unittest.TestCase):
    """Properties of the 12-experiment matrix produced by build_experiment_matrix()."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.experiments = build_experiment_matrix()

    def test_count_is_twelve(self):
        self.assertEqual(len(self.experiments), 12)

    def test_ordering_is_vanilla_agentic_rlm_madam(self):
        archs = [e.architecture for e in self.experiments]

        first_agentic = archs.index('agentic')
        first_rlm = archs.index('rlm')
        first_madam = archs.index('madam')
        last_vanilla = len(archs) - 1 - archs[::-1].index('vanilla')
        last_agentic = len(archs) - 1 - archs[::-1].index('agentic')
        last_rlm = len(archs) - 1 - archs[::-1].index('rlm')

        self.assertLess(last_vanilla, first_agentic, "Vanilla should come before agentic")
        self.assertLess(last_agentic, first_rlm, "Agentic should come before RLM")
        self.assertLess(last_rlm, first_madam, "RLM should come before MADAM")

    def test_three_experiments_per_architecture(self):
        counts: dict[str, int] = {}
        for e in self.experiments:
            counts[e.architecture] = counts.get(e.architecture, 0) + 1

        self.assertEqual(counts.get('vanilla'), 3)
        self.assertEqual(counts.get('agentic'), 3)
        self.assertEqual(counts.get('rlm'), 3)
        self.assertEqual(counts.get('madam'), 3)

    def test_rlm_has_no_k(self):
        rlm_exps = [e for e in self.experiments if e.architecture == 'rlm']
        self.assertGreater(len(rlm_exps), 0)
        for e in rlm_exps:
            self.assertIsNone(e.k, f"RLM {e.experiment_id} should have k=None, got {e.k}")

    def test_non_rlm_uses_k_10(self):
        for arch in ('vanilla', 'agentic', 'madam'):
            k_vals = {e.k for e in self.experiments if e.architecture == arch}
            self.assertEqual(k_vals, {10}, f"{arch}: expected K={{10}}, got {k_vals}")

    def test_every_arch_covers_all_attacks(self):
        expected_attacks = {'clean', 'naive', 'corruptrag_ak'}
        for arch in ('vanilla', 'agentic', 'rlm', 'madam'):
            attacks = {e.attack_type for e in self.experiments if e.architecture == arch}
            self.assertEqual(attacks, expected_attacks, f"{arch}: missing attacks")

    def test_experiment_ids_are_unique(self):
        ids = [e.experiment_id for e in self.experiments]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate experiment IDs")

    def test_experiment_ids_match_arch_attack_pattern(self):
        valid_archs = {'vanilla', 'agentic', 'rlm', 'madam'}
        valid_attacks = {'clean', 'naive', 'corruptrag_ak'}

        for e in self.experiments:
            parts = e.experiment_id.split('_', 1)
            self.assertEqual(len(parts), 2, f"ID {e.experiment_id!r} doesn't match arch_attack")
            arch, attack = parts
            self.assertIn(arch, valid_archs)
            self.assertIn(attack, valid_attacks)
            self.assertIsNone(re.search(r'_k\d+$', e.experiment_id),
                              f"ID {e.experiment_id!r} has unexpected _k suffix")


class ExperimentConfigSerializationUnitTests(unittest.TestCase):
    """ExperimentConfig.to_dict() round-trips through dict reconstruction."""

    def test_round_trip_preserves_all_fields(self):
        for config in build_experiment_matrix():
            d = config.to_dict()

            # Worker reconstructs ExperimentConfig by dropping the derived
            # 'corpus_type' key. Mirror that here.
            cfg_kwargs = {k: v for k, v in d.items() if k != 'corpus_type'}
            reconstructed = ExperimentConfig(**cfg_kwargs)

            self.assertEqual(reconstructed.experiment_id, config.experiment_id)
            self.assertEqual(reconstructed.architecture, config.architecture)
            self.assertEqual(reconstructed.attack_type, config.attack_type)
            self.assertEqual(reconstructed.k, config.k)
            self.assertEqual(reconstructed.corpus_type, config.corpus_type)

            # The dict crosses container boundaries via Modal — must be JSON-safe.
            json.dumps(d)


class IsExperimentCompleteUnitTests(unittest.TestCase):
    """File-count-based completion detection in orchestrator.is_experiment_complete()."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='rag_orch_test_')

        # is_experiment_complete reads EXPERIMENTS_DIR at the module level —
        # monkey-patch it to point at the tempdir for the duration of the test.
        import src.experiments.orchestrator as orch_mod
        self._orch_mod = orch_mod
        self._orig_experiments_dir = orch_mod.EXPERIMENTS_DIR
        orch_mod.EXPERIMENTS_DIR = self.tmp_dir

        self.exp_id = 'test_vanilla_clean'
        self.n_queries = 10
        self.exp_dir = os.path.join(self.tmp_dir, self.exp_id)

    def tearDown(self):
        self._orch_mod.EXPERIMENTS_DIR = self._orig_experiments_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def _write_result(self, filename: str, payload: dict) -> None:
        os.makedirs(self.exp_dir, exist_ok=True)
        with open(os.path.join(self.exp_dir, filename), 'w') as f:
            json.dump(payload, f)

    def test_missing_dir_is_incomplete(self):
        self.assertFalse(is_experiment_complete(self.exp_id, self.n_queries))

    def test_empty_dir_is_incomplete(self):
        os.makedirs(self.exp_dir)
        self.assertFalse(is_experiment_complete(self.exp_id, self.n_queries))

    def test_partial_results_is_incomplete(self):
        for i in range(5):
            self._write_result(f'test{i}.json', {'question_id': f'test{i}'})
        self.assertFalse(is_experiment_complete(self.exp_id, self.n_queries))

    def test_summary_json_does_not_count_toward_completion(self):
        for i in range(5):
            self._write_result(f'test{i}.json', {'question_id': f'test{i}'})
        self._write_result('summary.json', {'completed': 5})
        self.assertFalse(is_experiment_complete(self.exp_id, self.n_queries))

    def test_full_results_is_complete(self):
        for i in range(self.n_queries):
            self._write_result(f'test{i}.json', {'question_id': f'test{i}'})
        self.assertTrue(is_experiment_complete(self.exp_id, self.n_queries))

    def test_excess_results_still_complete(self):
        for i in range(self.n_queries):
            self._write_result(f'test{i}.json', {'question_id': f'test{i}'})
        self._write_result('test_extra.json', {'question_id': 'extra'})
        self.assertTrue(is_experiment_complete(self.exp_id, self.n_queries))
