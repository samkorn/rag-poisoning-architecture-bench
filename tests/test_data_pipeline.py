"""
Phase 4 migration test: verify local data scripts execute correctly.

No API calls, no Modal. Requires data symlinks to workspace.

Tests run the actual pipeline logic but redirect output to temp files
to avoid clobbering workspace data.

Run from repo root:
    python tests/test_data_pipeline.py
"""

import json
import os
import pickle
import tempfile

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('LOGFIRE_SEND_TO_LOGFIRE', 'false')

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

# Reference counts from workspace data
EXPECTED_TOTAL_QUESTIONS = 3452
EXPECTED_FILTERED_QUESTIONS = 1150


# ---------------------------------------------------------------------------
# Test: create_questions.py produces correct output
# ---------------------------------------------------------------------------

def test_create_questions():
    """Run create_questions.main() with output redirected to temp file."""
    print("\n=== test_create_questions ===")

    import src.data.create_questions as cq

    # Monkeypatch the output path to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        tmp_path = tmp.name

    original_main = cq.main

    # Patch by replacing the output path inline
    original_data_dir = cq._DATA_DIR
    try:
        # Override the output path: main() writes to _DATA_DIR/experiment-datasets/nq-questions.jsonl
        # We monkeypatch by replacing the open() target inside main
        import unittest.mock as mock

        original_open = open

        def patched_open(path, *args, **kwargs):
            if isinstance(path, str) and path.endswith('nq-questions.jsonl') and 'w' in (args[0] if args else kwargs.get('mode', 'r')):
                return original_open(tmp_path, *args, **kwargs)
            return original_open(path, *args, **kwargs)

        with mock.patch('builtins.open', side_effect=patched_open):
            cq.main()

        # Verify output
        with open(tmp_path) as f:
            lines = f.readlines()

        print(f"  Wrote {len(lines)} questions")
        assert len(lines) == EXPECTED_TOTAL_QUESTIONS, (
            f"Expected {EXPECTED_TOTAL_QUESTIONS}, got {len(lines)}"
        )

        # Check structure of first record
        first = json.loads(lines[0])
        required_keys = {'query_id', 'question', 'correct_answer', 'target_answer', 'gold_doc_ids'}
        assert required_keys.issubset(first.keys()), f"Missing keys: {required_keys - first.keys()}"
        assert first['query_id'].startswith('test'), f"Unexpected query_id format: {first['query_id']}"
        assert isinstance(first['gold_doc_ids'], list), "gold_doc_ids should be a list"
        print(f"  First record: query_id={first['query_id']}, keys={sorted(first.keys())}")

        # Verify all records are valid JSON with required keys
        for i, line in enumerate(lines):
            rec = json.loads(line)
            missing = required_keys - rec.keys()
            assert not missing, f"Line {i}: missing keys {missing}"

        print("  PASSED")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: filter_gold_questions.py produces correct output
# ---------------------------------------------------------------------------

def test_filter_gold_questions():
    """Run filter_gold_questions.main() with output redirected to temp file."""
    print("\n=== test_filter_gold_questions ===")

    import src.data.filter_gold_questions as fg

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        tmp_path = tmp.name

    # Monkeypatch OUTPUT_PATH
    original_output_path = fg.OUTPUT_PATH
    fg.OUTPUT_PATH = tmp_path
    try:
        fg.main()

        with open(tmp_path) as f:
            lines = f.readlines()

        print(f"  Wrote {len(lines)} filtered questions")
        assert len(lines) == EXPECTED_FILTERED_QUESTIONS, (
            f"Expected {EXPECTED_FILTERED_QUESTIONS}, got {len(lines)}"
        )

        # Check structure
        first = json.loads(lines[0])
        assert 'query_id' in first
        assert 'gold_doc_ids' in first
        assert len(first['gold_doc_ids']) > 0, "Filtered questions should have gold docs"
        print(f"  First: query_id={first['query_id']}, gold_docs={first['gold_doc_ids']}")

        print("  PASSED")
    finally:
        fg.OUTPUT_PATH = original_output_path
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: create_poisoned_datasets.py reads and parses correctly
# ---------------------------------------------------------------------------

def test_create_poisoned_datasets_parsing():
    """Verify poisoned dataset construction logic without writing corpus files."""
    print("\n=== test_create_poisoned_datasets_parsing ===")

    from src.data.create_poisoned_datasets import (
        DATA_DIR,
        ORIGINAL_NQ_DIR,
        EXPERIMENT_DIR,
    )

    # Verify input paths resolve
    assert os.path.isdir(ORIGINAL_NQ_DIR), f"Missing: {ORIGINAL_NQ_DIR}"
    assert os.path.isdir(EXPERIMENT_DIR), f"Missing: {EXPERIMENT_DIR}"

    # Parse queries (same as main() does)
    queries: dict[str, str] = {}
    with open(os.path.join(ORIGINAL_NQ_DIR, 'queries.jsonl')) as f:
        for line in f:
            line_dict = json.loads(line)
            queries[line_dict['_id']] = line_dict['text']
    print(f"  Loaded {len(queries):,} queries")
    assert len(queries) > 1000

    # Parse CorruptRAG-AK docs
    ak_path = os.path.join(EXPERIMENT_DIR, 'nq-corruptrag-ak-poisoned-docs.jsonl')
    ak_texts: dict[str, str] = {}
    with open(ak_path) as f:
        for line in f:
            line_dict = json.loads(line)
            ak_texts[line_dict['query_id']] = line_dict['corruptrag_ak_text']
    print(f"  Loaded {len(ak_texts):,} CorruptRAG-AK texts")
    assert len(ak_texts) > 1000

    # Verify poisoned doc construction for a sample
    sample_qid = list(ak_texts.keys())[0]
    query_text = queries[sample_qid]
    poisoned_text = f'{query_text} {ak_texts[sample_qid]}'
    poisoned_doc = {
        '_id': f'poisoned-corruptrag-ak-q:{sample_qid}',
        'title': 'test_title',
        'text': poisoned_text,
        'metadata': {},
    }
    assert poisoned_doc['_id'].startswith('poisoned-corruptrag-ak-q:')
    assert query_text in poisoned_doc['text']
    print(f"  Sample poisoned doc ID: {poisoned_doc['_id']}")
    print(f"  Poisoned text starts with query: {poisoned_doc['text'][:80]}...")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: build_vector_indexes imports and paths resolve
# ---------------------------------------------------------------------------

def test_build_vector_indexes_paths():
    """Verify build_vector_indexes path constants without rebuilding indexes."""
    print("\n=== test_build_vector_indexes_paths ===")

    from src.embeddings.build_vector_indexes import VECTOR_STORE_DIR

    assert os.path.isdir(VECTOR_STORE_DIR), f"Missing: {VECTOR_STORE_DIR}"

    # Verify embedding pickle files exist (inputs to index building)
    for prefix in ('nq-original', 'nq-naive-poisoned', 'nq-corruptrag-ak-poisoned'):
        emb_pkl = os.path.join(VECTOR_STORE_DIR, f'{prefix}-documents-embeddings.pkl')
        assert os.path.exists(emb_pkl), f"Missing embedding pickle: {emb_pkl}"
        size_mb = os.path.getsize(emb_pkl) / (1024**2)
        print(f"  OK  {prefix}-documents-embeddings.pkl ({size_mb:.0f} MB)")

    # Verify FAISS indexes exist (outputs of index building)
    for prefix in ('nq-original', 'nq-naive-poisoned', 'nq-corruptrag-ak-poisoned'):
        faiss_path = os.path.join(VECTOR_STORE_DIR, f'{prefix}.faiss')
        assert os.path.exists(faiss_path), f"Missing FAISS index: {faiss_path}"

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: VectorStore loads and retrieves
# ---------------------------------------------------------------------------

def test_vector_store_retrieval():
    """Load a VectorStore and run a single retrieval query."""
    print("\n=== test_vector_store_retrieval ===")

    from src.embeddings.vector_store import VectorStore

    vs = VectorStore(corpus_type='original')
    print(f"  Loaded VectorStore: {vs._index.ntotal:,} vectors")

    results = vs.retrieve("who is the current president of the united states", top_k=5)
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    for i, r in enumerate(results):
        assert 'doc_id' in r
        assert 'score' in r
        assert 'text' in r
        print(f"  [{i+1}] doc_id={r['doc_id']}, score={r['score']:.4f}, text={r['text'][:60]}...")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_create_questions()
    test_filter_gold_questions()
    test_create_poisoned_datasets_parsing()
    test_build_vector_indexes_paths()
    test_vector_store_retrieval()
    print("\n=== ALL TESTS PASSED ===")
