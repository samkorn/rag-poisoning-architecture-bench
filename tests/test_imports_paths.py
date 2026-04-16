"""
Phase 4 migration test: verify all modules import and path constants resolve.

No API calls, no Modal — just imports and filesystem checks.
Requires data symlinks to workspace (see MIGRATION_PROGRESS.md Phase 4).

Run from repo root:
    python tests/test_imports_paths.py
"""

import importlib
import os

# Suppress logfire interactive prompt when importing agentic_rag
os.environ.setdefault('LOGFIRE_SEND_TO_LOGFIRE', 'false')

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_file_exists(path: str, label: str):
    assert os.path.exists(path), f"{label}: path does not exist: {path}"


def _assert_dir_exists(path: str, label: str):
    assert os.path.isdir(path), f"{label}: not a directory: {path}"


# ---------------------------------------------------------------------------
# Test: all modules import
# ---------------------------------------------------------------------------

ALL_MODULES = [
    # data
    'src.data.utils',
    'src.data.create_questions',
    'src.data.filter_gold_questions',
    'src.data.create_poisoned_datasets',
    'src.data.download_datasets',
    # embeddings
    'src.embeddings.embeddings',
    'src.embeddings.vector_store',
    'src.embeddings.build_vector_indexes',
    # architectures
    'src.architectures.qa_system',
    'src.architectures.vanilla_rag',
    'src.architectures.agentic_rag',
    'src.architectures.madam_rag',
    'src.architectures.recursive_lm',
    'src.architectures.utils',
    # experiments
    'src.experiments.experiment',
    'src.experiments.llm_judge',
    'src.experiments.noise_filter',
]

# Modal-dependent modules — import separately since they register Modal apps
MODAL_MODULES = [
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


def test_import_all_modules():
    """Every module under src/ imports without error."""
    print("\n=== test_import_all_modules ===")
    failed = []

    for mod_name in ALL_MODULES + MODAL_MODULES:
        try:
            importlib.import_module(mod_name)
            print(f"  OK  {mod_name}")
        except Exception as e:
            print(f"  FAIL {mod_name}: {e}")
            failed.append((mod_name, str(e)))

    assert not failed, f"Failed imports: {failed}"
    print(f"  Imported {len(ALL_MODULES) + len(MODAL_MODULES)} modules")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: data directory path constants resolve
# ---------------------------------------------------------------------------

def test_data_dir_paths_resolve():
    """Path constants in data scripts point to real files via symlinks."""
    print("\n=== test_data_dir_paths_resolve ===")

    from src.data.utils import _DATA_BASE, _QUERIES_PATH, _QRELS_PATH, _CORPUS_PATHS
    from src.data.create_questions import _DATA_DIR as cq_data_dir
    from src.data.filter_gold_questions import _DATA_DIR as fg_data_dir
    from src.data.create_correct_answers import _DATA_DIR as ca_data_dir
    from src.data.create_incorrect_answers_poisoned_docs import _DATA_DIR as ia_data_dir
    from src.data.create_corruptrag_ak_poisoned_docs import _DATA_DIR as crak_data_dir
    from src.data.create_poisoned_datasets import DATA_DIR as cp_data_dir

    # All _DATA_DIR constants should resolve to src/data/
    for label, d in [
        ('data.utils._DATA_BASE', _DATA_BASE),
        ('create_questions._DATA_DIR', cq_data_dir),
        ('filter_gold._DATA_DIR', fg_data_dir),
        ('create_correct._DATA_DIR', ca_data_dir),
        ('create_incorrect._DATA_DIR', ia_data_dir),
        ('create_corruptrag._DATA_DIR', crak_data_dir),
        ('create_poisoned.DATA_DIR', cp_data_dir),
    ]:
        _assert_dir_exists(d, label)
        print(f"  OK  {label} -> {d}")

    # Key data files should exist via symlinks
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
        _assert_file_exists(path, label)
        print(f"  OK  {label}")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: vector store path constants resolve
# ---------------------------------------------------------------------------

def test_vector_store_paths_resolve():
    """Embeddings path constants cross-reference data/ correctly."""
    print("\n=== test_vector_store_paths_resolve ===")

    from src.embeddings.vector_store import VECTOR_STORE_DIR, _DATA_BASE
    from src.embeddings.build_vector_indexes import VECTOR_STORE_DIR as bvi_vs_dir

    _assert_dir_exists(VECTOR_STORE_DIR, 'vector_store.VECTOR_STORE_DIR')
    _assert_dir_exists(_DATA_BASE, 'vector_store._DATA_BASE')
    _assert_dir_exists(bvi_vs_dir, 'build_vector_indexes.VECTOR_STORE_DIR')
    print(f"  OK  vector_store.VECTOR_STORE_DIR -> {VECTOR_STORE_DIR}")
    print(f"  OK  vector_store._DATA_BASE -> {_DATA_BASE}")
    print(f"  OK  build_vector_indexes.VECTOR_STORE_DIR -> {bvi_vs_dir}")

    # FAISS index files should exist
    faiss_files = [
        'nq-original.faiss',
        'nq-naive-poisoned.faiss',
        'nq-corruptrag-ak-poisoned.faiss',
    ]
    for fname in faiss_files:
        path = os.path.join(VECTOR_STORE_DIR, fname)
        _assert_file_exists(path, f'faiss:{fname}')
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  OK  {fname} ({size_gb:.1f} GB)")

    # Doc-ids pickles
    for prefix in ('nq-original', 'nq-naive-poisoned', 'nq-corruptrag-ak-poisoned'):
        pkl = os.path.join(VECTOR_STORE_DIR, f'{prefix}-doc-ids.pkl')
        _assert_file_exists(pkl, f'doc-ids:{prefix}')

    # Query embeddings
    qe = os.path.join(VECTOR_STORE_DIR, 'nq-queries-embeddings.pkl')
    _assert_file_exists(qe, 'query-embeddings')

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: experiments path constants resolve
# ---------------------------------------------------------------------------

def test_experiments_paths_resolve():
    """Experiment script path constants point to real files/dirs."""
    print("\n=== test_experiments_paths_resolve ===")

    from src.experiments.llm_judge import RESULTS_DIR, JUDGE_RESULTS_DIR
    from src.experiments.noise_filter import QUESTIONS_PATH, NOISE_OUTPUT_DIR
    from src.experiments.run_judge_local import REVIEW_CSV, DEFAULT_OUTPUT_DIR, _RESULTS_DIR

    # Results directories
    _assert_dir_exists(RESULTS_DIR, 'llm_judge.RESULTS_DIR')
    _assert_dir_exists(JUDGE_RESULTS_DIR, 'llm_judge.JUDGE_RESULTS_DIR')
    _assert_dir_exists(NOISE_OUTPUT_DIR, 'noise_filter.NOISE_OUTPUT_DIR')
    _assert_dir_exists(_RESULTS_DIR, 'run_judge_local._RESULTS_DIR')
    print(f"  OK  RESULTS_DIR -> {RESULTS_DIR}")
    print(f"  OK  JUDGE_RESULTS_DIR -> {JUDGE_RESULTS_DIR}")
    print(f"  OK  NOISE_OUTPUT_DIR -> {NOISE_OUTPUT_DIR}")

    # Questions file
    _assert_file_exists(QUESTIONS_PATH, 'noise_filter.QUESTIONS_PATH')
    print(f"  OK  QUESTIONS_PATH -> {QUESTIONS_PATH}")

    # Review CSV (human_labels.csv via symlink)
    _assert_file_exists(REVIEW_CSV, 'run_judge_local.REVIEW_CSV')
    print(f"  OK  REVIEW_CSV -> {REVIEW_CSV}")

    # Judge prompt file
    from src.experiments.llm_judge import load_judge_prompt
    system_prompt, user_template = load_judge_prompt()
    assert len(system_prompt) > 100, f"Judge system prompt too short: {len(system_prompt)} chars"
    assert '{answer}' in user_template or '{' in user_template, "Judge user template missing placeholders"
    print(f"  OK  judge prompt loaded ({len(system_prompt)} chars)")

    # Judge result subdirs (spot-check a few)
    for exp_id in ('vanilla_clean', 'madam_corruptrag_ak'):
        judge_exp_dir = os.path.join(JUDGE_RESULTS_DIR, exp_id)
        _assert_dir_exists(judge_exp_dir, f'judge/{exp_id}')
        n_files = len([f for f in os.listdir(judge_exp_dir) if f.endswith('.json')])
        print(f"  OK  judge/{exp_id}/ ({n_files} JSON files)")

    # Noise result files
    noise_files = [f for f in os.listdir(NOISE_OUTPUT_DIR) if f.endswith('.json')]
    assert len(noise_files) > 100, f"Expected 100+ noise results, got {len(noise_files)}"
    print(f"  OK  noise/ ({len(noise_files)} JSON files)")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: cross-directory path resolution
# ---------------------------------------------------------------------------

def test_cross_directory_paths():
    """Paths that cross src/ subdirectory boundaries resolve correctly."""
    print("\n=== test_cross_directory_paths ===")

    # embeddings/ -> data/vector-store
    from src.embeddings.vector_store import VECTOR_STORE_DIR, _DATA_BASE
    vs_real = os.path.realpath(VECTOR_STORE_DIR)
    db_real = os.path.realpath(_DATA_BASE)
    assert 'data' in db_real, f"_DATA_BASE should contain 'data': {db_real}"
    assert 'vector-store' in vs_real, f"VECTOR_STORE_DIR should contain 'vector-store': {vs_real}"
    print(f"  OK  embeddings -> data cross-ref")

    # experiments/ -> data/ (noise_filter.QUESTIONS_PATH)
    from src.experiments.noise_filter import QUESTIONS_PATH
    qp_real = os.path.realpath(QUESTIONS_PATH)
    assert 'experiment-datasets' in qp_real, f"QUESTIONS_PATH should cross into data/: {qp_real}"
    print(f"  OK  experiments -> data cross-ref")

    # experiments/ -> analysis/ (run_judge_local.REVIEW_CSV)
    from src.experiments.run_judge_local import REVIEW_CSV
    rc_real = os.path.realpath(REVIEW_CSV)
    assert 'analysis' in rc_real or 'human_labels' in rc_real, f"REVIEW_CSV should cross into analysis/: {rc_real}"
    print(f"  OK  experiments -> analysis cross-ref")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_import_all_modules()
    test_data_dir_paths_resolve()
    test_vector_store_paths_resolve()
    test_experiments_paths_resolve()
    test_cross_directory_paths()
    print("\n=== ALL TESTS PASSED ===")
