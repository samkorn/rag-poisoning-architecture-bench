"""Imports + path-constant resolution across the src/ package.

Two suites:

* :class:`ImportUnitTests` — imports a curated set of modules that do
  *not* perform import-time IO. Runs in the unit suite (no data).
* :class:`ModulePathsIntegrationTests` — imports every module under
  ``src/`` and verifies the path constants they expose actually resolve
  to files/dirs on disk. Requires the dataset to be present locally.

Note on import-time IO: ``src.data.utils`` calls
``_load_noise_question_ids()`` at import time, so any module that
transitively imports it (most of ``src/`` except the orchestrator/
experiment helpers) only loads when noise filter results are available.
That's a code smell to revisit in Phase 7 — for now the unit suite
just sticks to modules that don't trigger the cascade.
"""

import importlib
import os
import unittest

import pytest


# Modules whose import does not require any on-disk data.
# Verified by hand: experiment.py is pure-Python, orchestrator.py only
# pulls in modal + experiment.
_PURE_IMPORT_MODULES = [
    'src.experiments.experiment',
    'src.experiments.orchestrator',
]

# Every module under src/ — used by the integration suite for full coverage.
_ALL_LOCAL_MODULES = [
    'src.data.utils',
    'src.data.create_questions',
    'src.data.filter_gold_questions',
    'src.data.create_poisoned_datasets',
    'src.data.download_datasets',
    'src.embeddings.embeddings',
    'src.embeddings.vector_store',
    'src.embeddings.build_vector_indexes',
    'src.architectures.qa_system',
    'src.architectures.vanilla_rag',
    'src.architectures.agentic_rag',
    'src.architectures.madam_rag',
    'src.architectures.recursive_lm',
    'src.architectures.utils',
    'src.experiments.experiment',
    'src.experiments.llm_judge',
    'src.experiments.noise_filter',
]

_MODAL_MODULES = [
    'src.data.create_correct_answers',
    'src.data.create_incorrect_answers_poisoned_docs',
    'src.data.create_corruptrag_ak_poisoned_docs',
    'src.embeddings.embed_datasets',
    'src.experiments.orchestrator',
    'src.experiments.run_judge_modal',
    'src.experiments.run_noise_modal',
    'src.experiments.upload_data',
    'src.experiments.run_judge_local',
]


class ImportUnitTests(unittest.TestCase):
    """Smoke-test imports for modules that don't require data on disk."""

    def test_pure_python_modules_import(self):
        """ExperimentConfig + orchestrator helpers must import without data."""
        for mod_name in _PURE_IMPORT_MODULES:
            with self.subTest(module=mod_name):
                importlib.import_module(mod_name)


@pytest.mark.integration
class ModulePathsIntegrationTests(unittest.TestCase):
    """Every module under src/ imports, every path constant resolves."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Suppress logfire interactive prompt when importing agentic_rag.
        os.environ.setdefault('LOGFIRE_SEND_TO_LOGFIRE', 'false')

        # Probe for the noise results dir — its absence breaks src.data.utils
        # at import time, so skip the whole class with a clear message.
        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        )
        noise_dir = os.path.join(repo_root, 'src', 'experiments', 'results', 'noise')
        if not os.path.isdir(noise_dir):
            raise unittest.SkipTest(
                f"Integration tests require noise results at {noise_dir}. "
                f"Either run scripts/download_data.sh to fetch the published dataset, "
                f"or regenerate the data by running the experiment pipeline."
            )
        cls.repo_root = repo_root

    def test_all_modules_import(self):
        failed: list[tuple[str, str]] = []
        for mod_name in _ALL_LOCAL_MODULES + _MODAL_MODULES:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                failed.append((mod_name, f"{type(e).__name__}: {e}"))
        self.assertFalse(failed, f"Failed imports: {failed}")

    def test_data_path_constants_resolve(self):
        from src.data.utils import _DATA_BASE, _QUERIES_PATH, _QRELS_PATH, _CORPUS_PATHS
        from src.data.create_questions import _DATA_DIR as cq_data_dir
        from src.data.filter_gold_questions import _DATA_DIR as fg_data_dir
        from src.data.create_correct_answers import _DATA_DIR as ca_data_dir
        from src.data.create_incorrect_answers_poisoned_docs import _DATA_DIR as ia_data_dir
        from src.data.create_corruptrag_ak_poisoned_docs import _DATA_DIR as crak_data_dir
        from src.data.create_poisoned_datasets import DATA_DIR as cp_data_dir

        for label, d in [
            ('data.utils._DATA_BASE', _DATA_BASE),
            ('create_questions._DATA_DIR', cq_data_dir),
            ('filter_gold._DATA_DIR', fg_data_dir),
            ('create_correct._DATA_DIR', ca_data_dir),
            ('create_incorrect._DATA_DIR', ia_data_dir),
            ('create_corruptrag._DATA_DIR', crak_data_dir),
            ('create_poisoned.DATA_DIR', cp_data_dir),
        ]:
            self.assertTrue(os.path.isdir(d), f"{label}: not a directory: {d}")

        key_files = [
            ('queries', _QUERIES_PATH),
            ('qrels', _QRELS_PATH),
            ('corpus_original', _CORPUS_PATHS['original']),
            ('corpus_naive', _CORPUS_PATHS['naive_poisoned']),
            ('corpus_crak', _CORPUS_PATHS['corruptrag_ak_poisoned']),
            ('questions', os.path.join(cq_data_dir, 'experiment-datasets', 'nq-questions.jsonl')),
            ('gold_filtered', os.path.join(fg_data_dir, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl')),
            ('correct_answers', os.path.join(ca_data_dir, 'experiment-datasets', 'nq-correct-answers.jsonl')),
            ('incorrect_answers', os.path.join(ia_data_dir, 'experiment-datasets', 'nq-incorrect-answers-poisoned-docs.jsonl')),
            ('corruptrag_docs', os.path.join(crak_data_dir, 'experiment-datasets', 'nq-corruptrag-ak-poisoned-docs.jsonl')),
        ]
        for label, path in key_files:
            self.assertTrue(os.path.exists(path), f"{label}: path does not exist: {path}")

    def test_vector_store_path_constants_resolve(self):
        from src.embeddings.vector_store import VECTOR_STORE_DIR, _DATA_BASE
        from src.embeddings.build_vector_indexes import VECTOR_STORE_DIR as bvi_vs_dir

        self.assertTrue(os.path.isdir(VECTOR_STORE_DIR))
        self.assertTrue(os.path.isdir(_DATA_BASE))
        self.assertTrue(os.path.isdir(bvi_vs_dir))

        for fname in ('nq-original.faiss', 'nq-naive-poisoned.faiss', 'nq-corruptrag-ak-poisoned.faiss'):
            path = os.path.join(VECTOR_STORE_DIR, fname)
            self.assertTrue(os.path.exists(path), f"missing FAISS: {path}")

        for prefix in ('nq-original', 'nq-naive-poisoned', 'nq-corruptrag-ak-poisoned'):
            pkl = os.path.join(VECTOR_STORE_DIR, f'{prefix}-doc-ids.pkl')
            self.assertTrue(os.path.exists(pkl), f"missing doc-ids pickle: {pkl}")

        qe = os.path.join(VECTOR_STORE_DIR, 'nq-queries-embeddings.pkl')
        self.assertTrue(os.path.exists(qe))

    def test_experiment_path_constants_resolve(self):
        from src.experiments.llm_judge import RESULTS_DIR, JUDGE_RESULTS_DIR, load_judge_prompt
        from src.experiments.noise_filter import QUESTIONS_PATH, NOISE_OUTPUT_DIR
        from src.experiments.run_judge_local import REVIEW_CSV, _RESULTS_DIR

        self.assertTrue(os.path.isdir(RESULTS_DIR))
        self.assertTrue(os.path.isdir(JUDGE_RESULTS_DIR))
        self.assertTrue(os.path.isdir(NOISE_OUTPUT_DIR))
        self.assertTrue(os.path.isdir(_RESULTS_DIR))
        self.assertTrue(os.path.exists(QUESTIONS_PATH))
        self.assertTrue(os.path.exists(REVIEW_CSV))

        # Judge prompt loads + has expected placeholders.
        system_prompt, user_template = load_judge_prompt()
        self.assertGreater(len(system_prompt), 100)
        self.assertIn('{', user_template)

        for exp_id in ('vanilla_clean', 'madam_corruptrag_ak'):
            judge_exp_dir = os.path.join(JUDGE_RESULTS_DIR, exp_id)
            self.assertTrue(os.path.isdir(judge_exp_dir), f"missing judge dir: {exp_id}")
            n = len([f for f in os.listdir(judge_exp_dir) if f.endswith('.json')])
            self.assertGreater(n, 0)

        noise_files = [f for f in os.listdir(NOISE_OUTPUT_DIR) if f.endswith('.json')]
        self.assertGreater(len(noise_files), 100)

    def test_cross_directory_paths(self):
        """Paths that cross src/ subdirectory boundaries resolve correctly."""
        from src.embeddings.vector_store import VECTOR_STORE_DIR, _DATA_BASE
        from src.experiments.noise_filter import QUESTIONS_PATH
        from src.experiments.run_judge_local import REVIEW_CSV

        db_real = os.path.realpath(_DATA_BASE)
        vs_real = os.path.realpath(VECTOR_STORE_DIR)
        self.assertIn('data', db_real)
        self.assertIn('vector-store', vs_real)

        qp_real = os.path.realpath(QUESTIONS_PATH)
        self.assertIn('experiment-datasets', qp_real)

        rc_real = os.path.realpath(REVIEW_CSV)
        self.assertTrue('analysis' in rc_real or 'human_labels' in rc_real)
