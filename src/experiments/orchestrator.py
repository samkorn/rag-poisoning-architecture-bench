"""
experiments/orchestrator.py

Modal application for the RAG Poisoning Architecture Bench experiments.

Defines:
  - Modal App with image, volume, and secrets
  - run_worker: processes a batch of questions for one experiment (up to 99 workers)
  - run_orchestrator: iterates through all 12 experiments sequentially
  - main: local entrypoint for ``modal run --detach``

Architecture:
    Your laptop
        | (modal run --detach)
    Orchestrator container (#100)   timeout=24h
        | (run_worker.starmap)
        +-- Worker #1   (questions 1-11)
        +-- Worker #2   (questions 12-22)
        +-- ...
        +-- Worker #99  (questions ~1140-1150)
    All writing to shared Modal Volume

Usage:
    # 1. Upload pre-computed data to Modal Volume (separate script)
    modal run experiments/upload_data.py

    # 2. Launch the full experiment run (detaches from terminal)
    modal run --detach experiments/orchestrator.py

    # 3. Monitor progress
    modal app logs rag-poisoning-bench

    # 4. Check results
    modal volume ls rag-poisoning-data results/

Volume layout expected (uploaded before experiments):
    /vol/
    +-- vector-store/            FAISS indexes, doc-ID pickles, query embeddings
    +-- original-datasets/nq/    BEIR NQ: queries.jsonl, corpus.jsonl, qrels/
    +-- experiment-datasets/     Poisoned corpus variants
    +-- results/                 (output — one subdir per experiment)
"""

import json
import os
import time

import modal


# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App('rag-poisoning-bench')

# --- Image ----------------------------------------------------------------
# Shared by orchestrator + all workers.  Contains project code (lightweight)
# but NOT the heavy data files (FAISS indexes, corpora) — those live on the
# Volume.

image = (
    modal.Image.debian_slim(python_version='3.12')
    .run_commands("pip install --upgrade pip")
    .pip_install(
        'openai',
        'python-dotenv',
        'pydantic-ai',
        'faiss-cpu',
        'transformers',
        'torch',
        'numpy',
        'scikit-learn',
        'tqdm',
        'rlms',
        'tenacity',
        'timeout-decorator',
    )
    # Pre-download Contriever model weights so workers don't fetch at runtime.
    .run_commands(
        "python -c \""
        "from transformers import AutoTokenizer, AutoModel; "
        "AutoTokenizer.from_pretrained('facebook/contriever'); "
        "AutoModel.from_pretrained('facebook/contriever')"
        "\""
    )
    # Ensure /app is on PYTHONPATH so `from src.X.Y import Z` works at import time.
    .env({'PYTHONPATH': '/app'})
    # Mount project Python code into /app/src/ so `from src.X.Y import Z` works.
    .add_local_file('src/__init__.py', remote_path='/app/src/__init__.py')
    .add_local_dir('src/architectures/', remote_path='/app/src/architectures/')
    .add_local_dir('src/embeddings/', remote_path='/app/src/embeddings/')
    .add_local_dir('src/experiments/', remote_path='/app/src/experiments/')
    # data/ — Python modules only, NOT the multi-GB data files.
    # The heavy data subdirectories (vector-store/, original-datasets/, etc.)
    # are symlinked from the Volume at container startup (see setup_container).
    .add_local_dir('src/data/', remote_path='/app/src/data/')
)

# --- Volume ---------------------------------------------------------------

# Persistent storage for pre-computed data + experiment results.
volume = modal.Volume.from_name('rag-poisoning-data', create_if_missing=True)
VOLUME_MOUNT = '/vol'

# Results directory on the volume.
RESULTS_DIR = f'{VOLUME_MOUNT}/results'

# --- Secrets --------------------------------------------------------------

secrets = [
    modal.Secret.from_name('openai-rag-poisoning'),
    modal.Secret.from_name('huggingface-rag-poisoning'),
]


# ---------------------------------------------------------------------------
# Container setup
# ---------------------------------------------------------------------------

def setup_container():
    """Prepare sys.path and data symlinks inside a Modal container.

    Called at the top of every worker / orchestrator function.

    VectorStore (in /app/src/embeddings/vector_store.py) resolves data paths
    relative to its own __file__:
        VECTOR_STORE_DIR = <__file__>/../../data/vector-store  ->  /app/src/data/vector-store
        _DATA_BASE       = <__file__>/../../data               ->  /app/src/data

    data/utils.py resolves paths from its own __file__:
        _DATA_BASE = os.path.dirname(__file__)                 ->  /app/src/data

    So we symlink the Volume's data subdirectories into /app/src/data/ where
    the code expects them.  The Python modules (baked into the image) coexist
    with these symlinks in the same directory.
    """
    import sys

    if '/app' not in sys.path:
        sys.path.insert(0, '/app')

    # Symlink heavy data directories from Volume into wherever src/data/ lives.
    # Modal may mount code at /app/src/ (via image.add_local_dir) or /root/src/
    # (via auto-mount from test files), so create symlinks in both locations.
    data_dirs = ['/app/src/data', '/root/src/data']
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        for subdir in ('vector-store', 'original-datasets', 'experiment-datasets'):
            link = os.path.join(data_dir, subdir)
            target = f'{VOLUME_MOUNT}/{subdir}'
            if os.path.exists(target) and not os.path.exists(link):
                os.symlink(target, link)


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

def build_experiment_matrix() -> list:
    """Build the ordered list of 12 experiments (4 architectures × 3 attacks).

    Fixed k=10 for non-RLM architectures; RLM uses full topic context (k=None).
    Order: fast experiments first for early sanity checking.
      Phase 1 — Vanilla RAG   (3 exp)
      Phase 2 — Agentic RAG   (3 exp)
      Phase 3 — RLM           (3 exp)
      Phase 4 — MADAM-RAG     (3 exp)
    """
    from src.experiments.experiment import ExperimentConfig

    experiments: list[ExperimentConfig] = []
    attack_types = ['clean', 'naive', 'corruptrag_ak']

    # Phase 1: Vanilla RAG
    for attack in attack_types:
        experiments.append(
            ExperimentConfig(
                experiment_id=f'vanilla_{attack}',
                architecture='vanilla',
                attack_type=attack,
                k=10,
            )
        )

    # Phase 2: Agentic RAG
    for attack in attack_types:
        experiments.append(
            ExperimentConfig(
                experiment_id=f'agentic_{attack}',
                architecture='agentic',
                attack_type=attack,
                k=10,
            )
        )

    # Phase 3: RLM
    for attack in attack_types:
        experiments.append(
            ExperimentConfig(
                experiment_id=f'rlm_{attack}',
                architecture='rlm',
                attack_type=attack,
                k=None,
            )
        )

    # Phase 4: MADAM-RAG
    for attack in attack_types:
        experiments.append(
            ExperimentConfig(
                experiment_id=f'madam_{attack}',
                architecture='madam',
                attack_type=attack,
                k=10,
            )
        )

    assert len(experiments) == 12, f"Expected 12 experiments, got {len(experiments)}"
    return experiments


def is_experiment_complete(experiment_id: str, n_questions: int) -> bool:
    """Check if all questions have *successful* result JSONs for this experiment.

    Only counts results where ``error`` is None.  Error results from previous
    rate-limit failures etc. are ignored so the orchestrator will re-dispatch
    workers for that experiment.
    """
    exp_dir = os.path.join(RESULTS_DIR, experiment_id)
    if not os.path.isdir(exp_dir):
        return False
    n_success = 0
    for f in os.listdir(exp_dir):
        if not f.endswith('.json') or f == 'summary.json':
            continue
        try:
            with open(os.path.join(exp_dir, f)) as fh:
                result = json.loads(fh.read())
            if result.get('error') is None:
                n_success += 1
        except (json.JSONDecodeError, OSError):
            pass
    return n_success >= n_questions


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60*60,  # 1 hour
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
    memory=14336,  # 14 GiB — FAISS index + corpus + Contriever model
    max_containers=99, # 100 default Modal limit minus 1 for orchestrator
)
def run_worker(config_dict: dict, question_ids: list[str]) -> dict:
    """Worker: process a batch of questions for one experiment.

    Each invocation runs in its own Modal container.  Loads the FAISS index
    and QA system once, then iterates through its assigned questions with
    per-question checkpointing.
    """
    setup_container()

    from src.experiments.experiment import ExperimentConfig, run_question_batch

    # Reconstruct ExperimentConfig from dict (drop derived 'corpus_type' key).
    cfg_kwargs = {k: v for k, v in config_dict.items() if k != 'corpus_type'}
    config = ExperimentConfig(**cfg_kwargs)

    # Load questions from Volume.
    questions_path = os.path.join(VOLUME_MOUNT, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl')
    questions: dict[str, dict] = {}
    with open(questions_path) as f:
        for line in f:
            line_dict = json.loads(line)
            questions[line_dict['query_id']] = line_dict

    summary = run_question_batch(
        config=config,
        question_ids=question_ids,
        questions=questions,
        results_dir=RESULTS_DIR,
        modal_volume=volume,
    )

    return summary


# ---------------------------------------------------------------------------
# Orchestrator function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60*60*24,  # 24 hours - estimated time for all experiments
)
def run_orchestrator():
    """Main orchestrator: run all 12 experiments sequentially.

    Each experiment dispatches up to 99 parallel workers via starmap().
    Fully checkpoint-recoverable: re-running skips completed experiments
    and completed questions.
    """
    setup_container()

    from src.experiments.experiment import split_questions

    # Load question IDs from nq-questions.jsonl on the Volume.
    questions_path = os.path.join(VOLUME_MOUNT, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl')
    all_question_ids: list[str] = []
    with open(questions_path) as f:
        for line in f:
            line_dict = json.loads(line)
            all_question_ids.append(line_dict['query_id'])

    n_questions = len(all_question_ids)
    experiments = build_experiment_matrix()
    batches = split_questions(all_question_ids, n_workers=99)

    print(f"Starting {len(experiments)} experiments, {n_questions} questions each")
    print(f"Workers per experiment: {len(batches)}, ~{n_questions // len(batches)} questions/worker")
    print()

    for i, config in enumerate(experiments):
        exp_start = time.time()
        print(f"{'=' * 60}")
        print(f"[{i + 1}/{len(experiments)}] {config.experiment_id}")
        print(f"  arch={config.architecture}  attack={config.attack_type}  k={config.k or 'all'}")
        print(f"{'=' * 60}")

        # Refresh volume to see results written by previous experiments.
        volume.reload()

        if is_experiment_complete(config.experiment_id, n_questions):
            print("  -> Already complete, skipping")
            continue

        # Dispatch 99 workers in parallel.
        config_dict = config.to_dict()
        worker_args = [(config_dict, batch) for batch in batches]
        results = list(run_worker.starmap(worker_args))

        # Aggregate worker summaries.
        total_completed = sum(r['completed'] for r in results)
        total_skipped = sum(r['skipped'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        elapsed = time.time() - exp_start

        print(f"  Completed: {total_completed}")
        print(f"  Skipped (checkpoint): {total_skipped}")
        print(f"  Errors: {total_errors}")
        print(f"  Wall time: {elapsed / 60:.1f} min")

        # Write experiment-level summary.
        summary_dir = os.path.join(RESULTS_DIR, config.experiment_id)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(
                {
                    'config': config_dict,
                    'completed': total_completed,
                    'skipped': total_skipped,
                    'errors': total_errors,
                    'wall_time_seconds': elapsed,
                    'avg_seconds_per_question': (
                        elapsed / total_completed if total_completed > 0 else None
                    ),
                },
                f,
                indent=2,
            )
        volume.commit()

    print()
    print(f"{'=' * 60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Dispatch orchestrator to a Modal container.

    Usage:  modal run --detach experiments/orchestrator.py
    """
    run_orchestrator.remote()
