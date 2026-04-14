"""
Quick smoke test for experiment.py.

Tests run_single_question() and run_question_batch() with 2-3 real NQ
questions against vanilla RAG (fastest architecture).

Run from workspace/:
    python experiments/test_experiment.py
"""

import json
import os
import shutil
import tempfile

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

from src.experiments.experiment import (
    ExperimentConfig,
    QuestionResult,
    RetrievalCapture,
    create_qa_system,
    detect_gold_in_results,
    detect_poison_in_results,
    is_poison_doc_id,
    run_question_batch,
    run_single_question,
    split_questions,
)


# ---------------------------------------------------------------------------
# Load test questions from pre-built questions.jsonl
# ---------------------------------------------------------------------------

def load_test_questions(query_ids: list[str]) -> dict[str, dict]:
    """Load question dicts from data/experiment-datasets/nq-questions.jsonl.

    Prerequisite: ``cd workspace/data && python create_questions.py``
    """
    questions_path = os.path.join(
        _REPO_ROOT, 'src', 'data', 'experiment-datasets', 'nq-questions.jsonl'
    )
    query_id_set = set(query_ids)
    questions: dict[str, dict] = {}
    with open(questions_path) as f:
        for line in f:
            line_dict = json.loads(line)
            if line_dict['query_id'] in query_id_set:
                questions[line_dict['query_id']] = line_dict
    return questions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

TEST_QUERY_IDS = ['test0', 'test1', 'test2']


def test_run_single_question_clean():
    """Vanilla RAG, clean corpus, single question."""
    print("\n=== test_run_single_question_clean ===")
    questions = load_test_questions(TEST_QUERY_IDS)
    config = ExperimentConfig(
        experiment_id='test_vanilla_clean',
        architecture='vanilla',
        attack_type='clean',
        k=10,
    )
    qa_system = create_qa_system(config)
    result = run_single_question(config, questions['test0'], qa_system)

    print(f"  Question:  {result.question_text}")
    print(f"  Answer:    {result.system_answer}")
    print(f"  Correct:   {result.correct_answer}")
    print(f"  Latency:   {result.latency_seconds:.2f}s")
    print(f"  Error:     {result.error}")
    print(f"  Retrieved: {result.retrieved_doc_ids}")
    print(f"  Poison:    retrieved={result.poison_retrieved}  rank={result.poison_rank}")
    print(f"  Gold:      ranks={result.gold_doc_ranks}  (expected gold_doc_ids={questions['test0'].get('gold_doc_ids')})")
    print(f"  Metadata:  {result.metadata}")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.system_answer, "Empty answer"
    assert len(result.retrieved_doc_ids) == 10, f"Expected 10 docs, got {len(result.retrieved_doc_ids)}"
    assert result.poison_retrieved is None, "Clean run should have poison_retrieved=None"
    assert result.poison_rank is None, "Clean run should have poison_rank=None"
    assert isinstance(result.gold_doc_ranks, list), "gold_doc_ranks should be a list"
    assert len(result.gold_doc_ranks) > 0, "test0 gold docs (doc0, doc1) should appear in top-10 clean retrieval"
    assert all(1 <= r <= 10 for r in result.gold_doc_ranks), f"Gold ranks should be 1-10, got {result.gold_doc_ranks}"
    assert result.metadata.get('passages_text_length', 0) > 0
    print("  PASSED")


def test_run_single_question_poisoned():
    """Vanilla RAG, naive-poisoned corpus, single question."""
    print("\n=== test_run_single_question_poisoned ===")
    questions = load_test_questions(TEST_QUERY_IDS)
    config = ExperimentConfig(
        experiment_id='test_vanilla_naive',
        architecture='vanilla',
        attack_type='naive',
        k=10,
    )
    qa_system = create_qa_system(config)
    result = run_single_question(config, questions['test0'], qa_system)

    print(f"  Question:  {result.question_text}")
    print(f"  Answer:    {result.system_answer}")
    print(f"  Correct:   {result.correct_answer}")
    print(f"  Target:    {result.target_answer}")
    print(f"  Latency:   {result.latency_seconds:.2f}s")
    print(f"  Error:     {result.error}")
    print(f"  Retrieved: {result.retrieved_doc_ids}")
    print(f"  Poison:    retrieved={result.poison_retrieved}  rank={result.poison_rank}")
    print(f"  Gold:      ranks={result.gold_doc_ranks}  (expected gold_doc_ids={questions['test0'].get('gold_doc_ids')})")
    print(f"  Metadata:  {result.metadata}")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.system_answer, "Empty answer"
    assert result.poison_retrieved is not None, "Poisoned run should report poison_retrieved"
    assert result.target_answer is not None, "Should have target_answer for poisoned run"
    assert isinstance(result.gold_doc_ranks, list), "gold_doc_ranks should be a list"
    # Gold docs may or may not appear in poisoned retrieval (poison displaces),
    # but the field should always be populated as a list.
    print("  PASSED")


def test_retrieval_capture_records():
    """Verify RetrievalCapture actually intercepts the architecture's calls."""
    print("\n=== test_retrieval_capture_records ===")
    config = ExperimentConfig(
        experiment_id='test_capture',
        architecture='vanilla',
        attack_type='clean',
        k=10,
    )
    qa_system = create_qa_system(config)

    with RetrievalCapture(qa_system.vector_store) as capture:
        qa_system._run("what is non controlling interest on balance sheet", 'test0')

    print(f"  retrieve() calls: {len(capture.retrieve_calls)}")
    print(f"  get_doc() calls:  {len(capture.doc_fetches)}")
    assert len(capture.retrieve_calls) == 1, f"Vanilla should have exactly 1 retrieve call, got {len(capture.retrieve_calls)}"
    results = capture.retrieve_calls[0]['results']
    assert len(results) == 10, f"Expected 10 results for k=10, got {len(results)}"
    print(f"  Top-10 doc_ids: {[d['doc_id'] for d in results]}")
    # Vanilla doesn't call get_document_from_doc_id
    assert len(capture.doc_fetches) == 0, f"Vanilla shouldn't fetch by ID, got {len(capture.doc_fetches)}"
    print("  PASSED")


def test_run_question_batch():
    """run_question_batch with 3 questions, checkpointing."""
    print("\n=== test_run_question_batch ===")
    questions = load_test_questions(TEST_QUERY_IDS)
    config = ExperimentConfig(
        experiment_id='test_batch_vanilla_clean',
        architecture='vanilla',
        attack_type='clean',
        k=10,
    )

    # Use a temp dir for results
    tmp_dir = tempfile.mkdtemp(prefix='rag_test_')
    try:
        summary = run_question_batch(
            config=config,
            question_ids=TEST_QUERY_IDS,
            questions=questions,
            results_dir=tmp_dir,
        )
        print(f"  Summary: {summary}")
        assert summary['completed'] == 3, f"Expected 3 completed, got {summary}"
        assert summary['errors'] == 0, f"Unexpected errors: {summary}"

        # Verify per-question files exist and are valid JSON
        exp_dir = os.path.join(tmp_dir, config.experiment_id)
        for qid in TEST_QUERY_IDS:
            path = os.path.join(exp_dir, f'{qid}.json')
            assert os.path.exists(path), f"Missing result file: {path}"
            with open(path) as f:
                data = json.load(f)
            assert data['question_id'] == qid
            assert data['system_answer'], f"Empty answer for {qid}"
            assert 'gold_doc_ranks' in data, f"Missing gold_doc_ranks in {qid} result JSON"
            assert isinstance(data['gold_doc_ranks'], list), f"gold_doc_ranks should be list for {qid}"
            print(f"  {qid}: answer={data['system_answer'][:60]}...  gold_ranks={data['gold_doc_ranks']}")

        # Re-run: all should be skipped (checkpoint recovery)
        summary2 = run_question_batch(
            config=config,
            question_ids=TEST_QUERY_IDS,
            questions=questions,
            results_dir=tmp_dir,
        )
        print(f"  Re-run summary: {summary2}")
        assert summary2['skipped'] == 3, f"Expected 3 skipped on re-run, got {summary2}"
        assert summary2['completed'] == 0
        print("  PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



def test_split_questions():
    """Verify round-robin splitting."""
    print("\n=== test_split_questions ===")
    ids = [f'test{i}' for i in range(10)]
    batches = split_questions(ids, n_workers=3)
    assert len(batches) == 3
    assert batches[0] == ['test0', 'test3', 'test6', 'test9']
    assert batches[1] == ['test1', 'test4', 'test7']
    assert batches[2] == ['test2', 'test5', 'test8']
    # All IDs accounted for
    flat = [qid for b in batches for qid in b]
    assert sorted(flat) == sorted(ids)
    print(f"  Batches: {[len(b) for b in batches]}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_split_questions()
    test_retrieval_capture_records()
    test_run_single_question_clean()
    test_run_single_question_poisoned()
    test_run_question_batch()
    print("\n=== ALL TESTS PASSED ===")
    os._exit(0)  # Skip slow GC of FAISS indexes
