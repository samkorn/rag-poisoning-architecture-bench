"""Tests for `src.experiments.experiment`.

* :class:`ExperimentHelpersUnitTests` — pure-Python helpers
  (`is_poison_doc_id`, `detect_*`, `make_log_tag`, `split_query_ids`).
  No data.
* :class:`RetrievalCaptureUnitTests` — `RetrievalCapture` intercepts
  the vector store calls. Requires the FAISS index, so it's marked
  integration even though the test itself is structural.
* :class:`<Architecture>RunIntegrationTests` — one class per
  architecture (vanilla clean, vanilla poisoned, agentic, RLM,
  MADAM) and one for the batch runner. Each makes live OpenAI calls
  and requires the vector store + question fixtures on disk.
"""

import json
import os
import shutil
import tempfile
import unittest

import pytest


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

# One question is enough to verify end-to-end behavior — the batch
# checkpoint/resume test is the only scenario where we need more than one.
TEST_QUERY_ID = 'test0'
BATCH_TEST_QUERY_IDS = ['test0', 'test1', 'test2']


def _data_present_or_skip() -> str:
    """Return the questions.jsonl path; skip if data isn't on disk."""
    questions_path = os.path.join(
        _REPO_ROOT, 'src', 'data', 'experiment-datasets', 'nq-questions.jsonl'
    )
    if not os.path.exists(questions_path):
        raise unittest.SkipTest(
            f"Integration test requires {questions_path}. "
            f"Either run scripts/download_data.sh to fetch the published dataset, "
            f"or regenerate the data by running the experiment pipeline."
        )
    return questions_path


def _load_test_queries(query_ids: list[str]) -> dict[str, dict]:
    """Load query records from data/experiment-datasets/nq-questions.jsonl."""
    questions_path = _data_present_or_skip()
    query_id_set = set(query_ids)
    queries: dict[str, dict] = {}
    with open(questions_path) as f:
        for line in f:
            line_dict = json.loads(line)
            if line_dict['query_id'] in query_id_set:
                queries[line_dict['query_id']] = line_dict
    return queries


# ===========================================================================
# Unit suite
# ===========================================================================

class ExperimentHelpersUnitTests(unittest.TestCase):
    """Pure-Python helpers from src.experiments.experiment — no IO."""

    def test_is_poison_doc_id_positive(self):
        """All `poisoned`-prefixed IDs are detected as poison."""
        from src.experiments.experiment import is_poison_doc_id
        for doc_id in (
            'poisoned-naive-q:test0',
            'poisoned-corruptrag-ak-q:test1234',
            'poisoned',  # bare prefix still counts
        ):
            self.assertTrue(is_poison_doc_id(doc_id))

    def test_is_poison_doc_id_negative(self):
        """Original IDs and non-matching prefixes are not flagged as poison."""
        from src.experiments.experiment import is_poison_doc_id
        for doc_id in ('doc0', 'doc12345', '', 'poison-not-quite', 'POISONED-uppercase'):
            self.assertFalse(is_poison_doc_id(doc_id))

    def test_detect_poison_in_results_no_poison(self):
        """No poison in results -> `(False, None)`."""
        from src.experiments.experiment import detect_poison_in_results
        docs = [{'doc_id': f'doc{i}'} for i in range(5)]
        found, rank = detect_poison_in_results(docs)
        self.assertFalse(found)
        self.assertIsNone(rank)

    def test_detect_poison_in_results_at_rank_one(self):
        """Poison at the top of the result list -> `(True, 1)`."""
        from src.experiments.experiment import detect_poison_in_results
        docs = [
            {'doc_id': 'poisoned-naive-q:test0'},
            {'doc_id': 'doc1'},
            {'doc_id': 'doc2'},
        ]
        found, rank = detect_poison_in_results(docs)
        self.assertTrue(found)
        self.assertEqual(rank, 1)

    def test_detect_poison_in_results_at_rank_five(self):
        """Poison mid-list returns the correct 1-indexed rank."""
        from src.experiments.experiment import detect_poison_in_results
        docs = [{'doc_id': f'doc{i}'} for i in range(4)]
        docs.append({'doc_id': 'poisoned-corruptrag-ak-q:test0'})
        docs.extend({'doc_id': f'doc{i}'} for i in range(5, 10))
        found, rank = detect_poison_in_results(docs)
        self.assertTrue(found)
        self.assertEqual(rank, 5)

    def test_detect_gold_in_results_no_gold(self):
        """No gold IDs in retrieval -> empty rank list."""
        from src.experiments.experiment import detect_gold_in_results
        docs = [{'doc_id': f'doc{i}'} for i in range(5)]
        ranks = detect_gold_in_results(docs, gold_doc_ids=['doc99', 'doc100'])
        self.assertEqual(ranks, [])

    def test_detect_gold_in_results_single(self):
        """One gold hit returns its 1-indexed rank."""
        from src.experiments.experiment import detect_gold_in_results
        docs = [{'doc_id': f'doc{i}'} for i in range(5)]
        ranks = detect_gold_in_results(docs, gold_doc_ids=['doc2'])
        self.assertEqual(ranks, [3])  # 1-indexed

    def test_detect_gold_in_results_multiple(self):
        """Multiple gold hits return all ranks in retrieval order."""
        from src.experiments.experiment import detect_gold_in_results
        docs = [{'doc_id': f'doc{i}'} for i in range(10)]
        ranks = detect_gold_in_results(docs, gold_doc_ids=['doc0', 'doc5', 'doc9'])
        self.assertEqual(ranks, [1, 6, 10])

    def test_make_log_tag_with_k(self):
        """Tag includes architecture, k, attack, and query."""
        from src.experiments.experiment import ExperimentConfig, make_log_tag
        config = ExperimentConfig(
            experiment_id='vanilla_clean',
            architecture='vanilla',
            attack_type='clean',
            k=10,
        )
        self.assertEqual(make_log_tag(config, 'test0'), '[vanilla k=10 clean test0]')

    def test_make_log_tag_rlm_no_k(self):
        """RLM tag omits the `k=` segment when `config.k` is `None`."""
        from src.experiments.experiment import ExperimentConfig, make_log_tag
        config = ExperimentConfig(
            experiment_id='rlm_clean',
            architecture='rlm',
            attack_type='clean',
            k=None,
        )
        self.assertEqual(make_log_tag(config, 'test0'), '[rlm clean test0]')

    def test_make_log_tag_with_batch_info(self):
        """Tag appends `q=N/M` when batch position is provided."""
        from src.experiments.experiment import ExperimentConfig, make_log_tag
        config = ExperimentConfig(
            experiment_id='madam_naive',
            architecture='madam',
            attack_type='naive',
            k=10,
        )
        tag = make_log_tag(config, 'test5', question_num=3, batch_size=12)
        self.assertEqual(tag, '[madam k=10 naive test5 q=3/12]')

    def test_split_query_ids_round_robin(self):
        """Round-robin splits 10 IDs across 3 workers in 4/3/3 batches."""
        from src.experiments.experiment import split_query_ids
        ids = [f'test{i}' for i in range(10)]
        batches = split_query_ids(ids, n_workers=3)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], ['test0', 'test3', 'test6', 'test9'])
        self.assertEqual(batches[1], ['test1', 'test4', 'test7'])
        self.assertEqual(batches[2], ['test2', 'test5', 'test8'])
        self.assertEqual(sorted(query_id for b in batches for query_id in b), sorted(ids))


# ===========================================================================
# Integration suite — one class per scenario, declared in working order
# ===========================================================================

@pytest.mark.integration
class VanillaCleanIntegrationTests(unittest.TestCase):
    """Vanilla RAG against the clean corpus, single question."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.queries = _load_test_queries([TEST_QUERY_ID])

    def test_run_single_question(self):
        """Vanilla clean run produces a result with gold-doc ranks."""
        from src.experiments.experiment import (
            ExperimentConfig, create_qa_system, run_single_question,
        )
        config = ExperimentConfig(
            experiment_id='test_vanilla_clean',
            architecture='vanilla',
            attack_type='clean',
            k=10,
        )
        qa_system = create_qa_system(config)
        result = run_single_question(config, self.queries[TEST_QUERY_ID], qa_system)

        self.assertIsNone(result.error)
        self.assertTrue(result.system_answer)
        self.assertEqual(len(result.retrieved_doc_ids), 10)
        self.assertIsNone(result.poison_retrieved, "Clean run shouldn't have poison_retrieved set")
        self.assertIsNone(result.poison_rank)
        self.assertIsInstance(result.gold_doc_ranks, list)
        self.assertGreater(
            len(result.gold_doc_ranks), 0,
            "test0 gold docs should appear in top-10 clean retrieval",
        )
        self.assertTrue(all(1 <= r <= 10 for r in result.gold_doc_ranks))
        self.assertGreater(result.metadata.get('passages_text_length', 0), 0)


@pytest.mark.integration
class VanillaNaiveIntegrationTests(unittest.TestCase):
    """Vanilla RAG against the naive-poisoned corpus."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.queries = _load_test_queries([TEST_QUERY_ID])

    def test_run_single_question(self):
        """Vanilla naive-poisoned run sets poison + target fields."""
        from src.experiments.experiment import (
            ExperimentConfig, create_qa_system, run_single_question,
        )
        config = ExperimentConfig(
            experiment_id='test_vanilla_naive',
            architecture='vanilla',
            attack_type='naive',
            k=10,
        )
        qa_system = create_qa_system(config)
        result = run_single_question(config, self.queries[TEST_QUERY_ID], qa_system)

        self.assertIsNone(result.error)
        self.assertTrue(result.system_answer)
        self.assertIsNotNone(result.poison_retrieved)
        self.assertIsNotNone(result.target_answer)
        self.assertIsInstance(result.gold_doc_ranks, list)


@pytest.mark.integration
class RetrievalCaptureIntegrationTests(unittest.TestCase):
    """RetrievalCapture intercepts the architecture's vector-store calls."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _data_present_or_skip()

    def test_vanilla_records_one_retrieve_no_doc_fetches(self):
        """Vanilla RAG triggers exactly one `retrieve` and no doc fetches."""
        from src.experiments.experiment import (
            ExperimentConfig, RetrievalCapture, create_qa_system,
        )
        config = ExperimentConfig(
            experiment_id='test_capture',
            architecture='vanilla',
            attack_type='clean',
            k=10,
        )
        qa_system = create_qa_system(config)

        with RetrievalCapture(qa_system.vector_store) as capture:
            qa_system._run("what is non controlling interest on balance sheet", 'test0')

        self.assertEqual(len(capture.retrieve_calls), 1)
        self.assertEqual(len(capture.retrieve_calls[0]['results']), 10)
        # Vanilla doesn't fetch by ID.
        self.assertEqual(len(capture.doc_fetches), 0)


@pytest.mark.integration
class AgenticCleanIntegrationTests(unittest.TestCase):
    """Agentic RAG, clean corpus, single question."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.queries = _load_test_queries([TEST_QUERY_ID])

    def test_run_single_question(self):
        """Agentic RAG clean run completes and emits an answer."""
        from src.experiments.experiment import (
            ExperimentConfig, create_qa_system, run_single_question,
        )
        config = ExperimentConfig(
            experiment_id='test_agentic_clean',
            architecture='agentic',
            attack_type='clean',
            k=10,
        )
        qa_system = create_qa_system(config)
        result = run_single_question(config, self.queries[TEST_QUERY_ID], qa_system)

        self.assertIsNone(result.error)
        self.assertTrue(result.system_answer)


@pytest.mark.integration
class RLMCleanIntegrationTests(unittest.TestCase):
    """RLM, clean corpus, single question.

    Note: RLM uses a large topic-scoped context and may take 60–120s.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.queries = _load_test_queries([TEST_QUERY_ID])

    def test_run_single_question(self):
        """RLM clean run completes without error and emits an answer."""
        from src.experiments.experiment import (
            ExperimentConfig, create_qa_system, run_single_question,
        )
        config = ExperimentConfig(
            experiment_id='test_rlm_clean',
            architecture='rlm',
            attack_type='clean',
            k=None,
        )
        qa_system = create_qa_system(config)
        result = run_single_question(config, self.queries[TEST_QUERY_ID], qa_system)

        self.assertIsNone(result.error)
        self.assertTrue(result.system_answer)


@pytest.mark.integration
class MADAMCleanIntegrationTests(unittest.TestCase):
    """MADAM-RAG, clean corpus, single question."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.queries = _load_test_queries([TEST_QUERY_ID])

    def test_run_single_question(self):
        """MADAM-RAG clean run completes without error and emits an answer."""
        from src.experiments.experiment import (
            ExperimentConfig, create_qa_system, run_single_question,
        )
        config = ExperimentConfig(
            experiment_id='test_madam_clean',
            architecture='madam',
            attack_type='clean',
            k=10,
        )
        qa_system = create_qa_system(config)
        result = run_single_question(config, self.queries[TEST_QUERY_ID], qa_system)

        self.assertIsNone(result.error)
        self.assertTrue(result.system_answer)


@pytest.mark.integration
class QuestionBatchIntegrationTests(unittest.TestCase):
    """run_question_batch with checkpoint/resume across 3 questions.

    The batch test is the one place we need >1 question — that's the
    whole point of testing batching. Other integration tests use a
    single question."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.queries = _load_test_queries(BATCH_TEST_QUERY_IDS)

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='rag_test_')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_batch_runs_then_skips_on_rerun(self):
        """First batch completes 3 queries; second run finds them cached."""
        from src.experiments.experiment import ExperimentConfig, run_question_batch

        config = ExperimentConfig(
            experiment_id='test_batch_vanilla_clean',
            architecture='vanilla',
            attack_type='clean',
            k=10,
        )

        summary = run_question_batch(
            config=config,
            query_ids=BATCH_TEST_QUERY_IDS,
            queries=self.queries,
            results_dir=self.tmp_dir,
        )
        self.assertEqual(summary['completed'], 3)
        self.assertEqual(summary['errors'], 0)

        exp_dir = os.path.join(self.tmp_dir, config.experiment_id)
        for query_id in BATCH_TEST_QUERY_IDS:
            path = os.path.join(exp_dir, f'{query_id}.json')
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data['question_id'], query_id)
            self.assertTrue(data['system_answer'])
            self.assertIn('gold_doc_ranks', data)
            self.assertIsInstance(data['gold_doc_ranks'], list)

        # Second run should skip everything (checkpoint recovery).
        summary2 = run_question_batch(
            config=config,
            query_ids=BATCH_TEST_QUERY_IDS,
            queries=self.queries,
            results_dir=self.tmp_dir,
        )
        self.assertEqual(summary2['skipped'], 3)
        self.assertEqual(summary2['completed'], 0)
