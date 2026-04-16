"""Local data-pipeline scripts: parse helpers (unit) and end-to-end runs (integration).

Integration tests are split across multiple ``TestCase`` classes — one per
script under exercise — for two reasons:

1. Each script is logically independent.
2. FAISS objects don't garbage-collect cleanly between tests in a single
   process; loading the index in one test and then loading the embeddings
   in another (without a process boundary) can segfault. Putting each
   FAISS-touching step in its own class reduces the chance of cross-test
   state collisions, and pytest collects classes in file declaration order
   so the working ordering is preserved.

Skip cleanly when the data symlinks aren't in place. See
``scripts/setup_test_symlinks.sh`` (in the parent project).
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import pytest


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

# Reference counts pinned to the workspace data the paper was written against.
EXPECTED_TOTAL_QUESTIONS = 3452
EXPECTED_FILTERED_QUESTIONS = 1150


def _data_present_or_skip() -> None:
    """Skip the calling class if the workspace data isn't on disk."""
    sentinel = os.path.join(
        _REPO_ROOT, 'src', 'data', 'original-datasets', 'nq', 'queries.jsonl'
    )
    if not os.path.exists(sentinel):
        raise unittest.SkipTest(
            f"Integration test requires {sentinel}. "
            f"Run scripts/setup_test_symlinks.sh (local dev) or "
            f"scripts/download_data.sh (once Phase 5 lands)."
        )


# ===========================================================================
# Unit suite
# ===========================================================================

class DataPipelineInvariantUnitTests(unittest.TestCase):
    """Cross-module invariants the rest of the pipeline depends on."""

    def test_poisoned_doc_id_format_matches_detector(self):
        """The IDs constructed by create_poisoned_datasets must trip is_poison_doc_id.

        If anyone renames the prefix in one place but not the other, every
        downstream metric (ASR, poison rank) silently corrupts.
        """
        from src.experiments.experiment import is_poison_doc_id

        # Mirror the construction in create_poisoned_datasets.main():
        #   f'poisoned-naive-q:{query_id}' / f'poisoned-corruptrag-ak-q:{query_id}'
        for qid in ('test0', 'test1234'):
            for prefix in ('poisoned-naive-q', 'poisoned-corruptrag-ak-q'):
                doc_id = f'{prefix}:{qid}'
                self.assertTrue(
                    is_poison_doc_id(doc_id),
                    f"is_poison_doc_id({doc_id!r}) should return True",
                )

        for doc_id in ('doc0', 'doc12345', 'doc999999'):
            self.assertFalse(
                is_poison_doc_id(doc_id),
                f"is_poison_doc_id({doc_id!r}) should return False",
            )

    def test_attack_to_corpus_keys_cover_all_attacks(self):
        """ATTACK_TO_CORPUS must cover every attack type the matrix generates."""
        from src.experiments.experiment import ATTACK_TO_CORPUS
        from src.experiments.orchestrator import build_experiment_matrix

        matrix_attacks = {e.attack_type for e in build_experiment_matrix()}
        self.assertEqual(matrix_attacks, set(ATTACK_TO_CORPUS.keys()))


# ===========================================================================
# Integration suite — one class per script, declared in working order
# ===========================================================================

@pytest.mark.integration
class CreateQuestionsIntegrationTests(unittest.TestCase):
    """Run create_questions.main() with output redirected to a temp file."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _data_present_or_skip()

    def test_produces_expected_count_and_keys(self):
        import src.data.create_questions as cq

        tmp_path = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False).name
        try:
            original_open = open

            def patched_open(path, *args, **kwargs):
                mode = args[0] if args else kwargs.get('mode', 'r')
                if isinstance(path, str) and path.endswith('nq-questions.jsonl') and 'w' in mode:
                    return original_open(tmp_path, *args, **kwargs)
                return original_open(path, *args, **kwargs)

            with mock.patch('builtins.open', side_effect=patched_open):
                cq.main()

            with open(tmp_path) as f:
                lines = f.readlines()

            self.assertEqual(
                len(lines), EXPECTED_TOTAL_QUESTIONS,
                f"Expected {EXPECTED_TOTAL_QUESTIONS}, got {len(lines)}",
            )

            first = json.loads(lines[0])
            required_keys = {
                'query_id', 'question', 'correct_answer', 'target_answer', 'gold_doc_ids',
            }
            self.assertTrue(required_keys.issubset(first.keys()))
            self.assertTrue(first['query_id'].startswith('test'))
            self.assertIsInstance(first['gold_doc_ids'], list)

            for i, line in enumerate(lines):
                rec = json.loads(line)
                missing = required_keys - rec.keys()
                self.assertFalse(missing, f"Line {i}: missing {missing}")
        finally:
            os.unlink(tmp_path)


@pytest.mark.integration
class FilterGoldQuestionsIntegrationTests(unittest.TestCase):
    """Run filter_gold_questions.main() against the real FAISS index."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _data_present_or_skip()

    def test_produces_expected_count_and_keys(self):
        import src.data.filter_gold_questions as fg

        tmp_path = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False).name
        original_output_path = fg.OUTPUT_PATH
        fg.OUTPUT_PATH = tmp_path
        try:
            fg.main()

            with open(tmp_path) as f:
                lines = f.readlines()

            self.assertEqual(
                len(lines), EXPECTED_FILTERED_QUESTIONS,
                f"Expected {EXPECTED_FILTERED_QUESTIONS}, got {len(lines)}",
            )

            first = json.loads(lines[0])
            self.assertIn('query_id', first)
            self.assertIn('gold_doc_ids', first)
            self.assertGreater(len(first['gold_doc_ids']), 0)
        finally:
            fg.OUTPUT_PATH = original_output_path
            os.unlink(tmp_path)


@pytest.mark.integration
class CreatePoisonedDatasetsIntegrationTests(unittest.TestCase):
    """Verify poisoned dataset construction inputs parse correctly."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _data_present_or_skip()

    def test_parses_inputs_and_constructs_valid_poisoned_doc(self):
        from src.data.create_poisoned_datasets import (
            ORIGINAL_NQ_DIR,
            EXPERIMENT_DIR,
        )
        from src.experiments.experiment import is_poison_doc_id

        self.assertTrue(os.path.isdir(ORIGINAL_NQ_DIR))
        self.assertTrue(os.path.isdir(EXPERIMENT_DIR))

        queries: dict[str, str] = {}
        with open(os.path.join(ORIGINAL_NQ_DIR, 'queries.jsonl')) as f:
            for line in f:
                line_dict = json.loads(line)
                queries[line_dict['_id']] = line_dict['text']
        self.assertGreater(len(queries), 1000)

        ak_path = os.path.join(EXPERIMENT_DIR, 'nq-corruptrag-ak-poisoned-docs.jsonl')
        ak_texts: dict[str, str] = {}
        with open(ak_path) as f:
            for line in f:
                line_dict = json.loads(line)
                ak_texts[line_dict['query_id']] = line_dict['corruptrag_ak_text']
        self.assertGreater(len(ak_texts), 1000)

        sample_qid = next(iter(ak_texts))
        poisoned_doc = {
            '_id': f'poisoned-corruptrag-ak-q:{sample_qid}',
            'title': 'test_title',
            'text': f'{queries[sample_qid]} {ak_texts[sample_qid]}',
            'metadata': {},
        }
        self.assertTrue(is_poison_doc_id(poisoned_doc['_id']))
        self.assertIn(queries[sample_qid], poisoned_doc['text'])


@pytest.mark.integration
class BuildVectorIndexesIntegrationTests(unittest.TestCase):
    """Verify the inputs/outputs of the FAISS index build are present."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _data_present_or_skip()

    def test_inputs_and_outputs_present(self):
        from src.embeddings.build_vector_indexes import VECTOR_STORE_DIR

        self.assertTrue(os.path.isdir(VECTOR_STORE_DIR))

        for prefix in ('nq-original', 'nq-naive-poisoned', 'nq-corruptrag-ak-poisoned'):
            emb_pkl = os.path.join(VECTOR_STORE_DIR, f'{prefix}-documents-embeddings.pkl')
            self.assertTrue(os.path.exists(emb_pkl), f"missing pkl: {emb_pkl}")
            faiss_path = os.path.join(VECTOR_STORE_DIR, f'{prefix}.faiss')
            self.assertTrue(os.path.exists(faiss_path), f"missing FAISS: {faiss_path}")


@pytest.mark.integration
class VectorStoreRetrievalIntegrationTests(unittest.TestCase):
    """Load the VectorStore and run a single retrieval query."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _data_present_or_skip()

    def test_loads_and_retrieves_top_k(self):
        from src.embeddings.vector_store import VectorStore

        vs = VectorStore(corpus_type='original')
        results = vs.retrieve("who is the current president of the united states", top_k=5)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIn('doc_id', r)
            self.assertIn('score', r)
            self.assertIn('text', r)
