"""Modal application for the RAG Poisoning Architecture Bench experiments.

Defines:

  * Modal App with image, volume, and secrets.
  * `run_worker` — processes a batch of questions for one experiment
    (up to 99 workers run in parallel).
  * `run_orchestrator` — iterates through all 12 experiments
    sequentially, fanning out workers per experiment.
  * `main` — local entrypoint for `modal run --detach`.

Topology:

    Your laptop
        | (modal run --detach)
    Orchestrator container (#100)   timeout=24h
        | (run_worker.starmap)
        +-- Worker #1   (questions 1-11)
        +-- Worker #2   (questions 12-22)
        +-- ...
        +-- Worker #99  (questions ~1140-1150)
    All writing to shared Modal Volume

Prerequisites:
    Modal Volume `rag-poisoning-data` populated by
    `src/experiments/upload_data.py`. Modal credentials and the
    `openai-rag-poisoning` Modal Secret.

Usage:
    modal run src/experiments/upload_data.py
    modal run --detach src/experiments/orchestrator.py
    modal app logs rag-poisoning-bench
    modal volume ls rag-poisoning-data results/

Output:
    Per-question result JSONs under `/vol/results/<experiment_dir>/`
    on the Modal Volume. Each `<experiment_dir>` corresponds to one
    of the 12 (architecture, attack) cells.

Notes:
    Volume layout expected before experiments run:

      * `/vol/vector-store/` — FAISS indexes, doc-ID pickles, query
        embeddings.
      * `/vol/original-datasets/nq/` — BEIR NQ (`queries.jsonl`,
        `corpus.jsonl`, `qrels/`).
      * `/vol/experiment-datasets/` — poisoned corpus variants.
      * `/vol/results/` — output, one subdirectory per experiment.
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
    # Mount the src package into /root/src/ (Modal's default PYTHONPATH).
    # This makes `from src.X.Y import Z` work at import time without
    # custom PYTHONPATH or sys.path hacks.
    # Note: data/ Python modules are included, but the heavy data subdirs
    # (vector-store/, original-datasets/, etc.) are symlinked from the
    # Volume at container startup (see setup_container).
    .add_local_python_source('src')
)

# --- Volume ---------------------------------------------------------------

# Persistent storage for pre-computed data + experiment results.
volume = modal.Volume.from_name('rag-poisoning-data', create_if_missing=True)
VOLUME_MOUNT = '/vol'

# Results tree on the volume:
#   /vol/results/experiments/<experiment_id>/<query_id>.json
#   /vol/results/judge/<experiment_id>/<query_id>.json
#   /vol/results/noise/<query_id>.json
#   /vol/results/archive/
RESULTS_DIR = f'{VOLUME_MOUNT}/results'
EXPERIMENTS_DIR = f'{RESULTS_DIR}/experiments'

# --- Secrets --------------------------------------------------------------

secrets = [
    modal.Secret.from_name('openai-rag-poisoning'),
]


# ---------------------------------------------------------------------------
# Container setup
# ---------------------------------------------------------------------------

def setup_container():
    """Symlink the Modal Volume's data subdirs into `/root/src/data/`.

    Called at the top of every worker and orchestrator function.
    The `src` package is mounted at `/root/src/` via
    `add_local_python_source`; `VectorStore` and `data/utils.py`
    resolve paths relative to their own `__file__`, so they look
    for data under `/root/src/data/vector-store`, etc. The Volume
    is mounted at `VOLUME_MOUNT` and we symlink each subdirectory
    into the expected location.
    """
    # Symlink heavy data directories from Volume into /root/src/data/
    # (where add_local_python_source mounts the src package).
    symlinks = {
        '/root/src/data/vector-store': f'{VOLUME_MOUNT}/vector-store',
        '/root/src/data/original-datasets': f'{VOLUME_MOUNT}/original-datasets',
        '/root/src/data/experiment-datasets': f'{VOLUME_MOUNT}/experiment-datasets',
    }
    for link, target in symlinks.items():
        if os.path.exists(target) and not os.path.exists(link):
            os.symlink(target, link)


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

def build_experiment_matrix() -> list:
    """Build the ordered list of 12 experiments (4 architectures x 3 attacks).

    Fixed `k=10` for non-RLM architectures; RLM uses full topic
    context (`k=None`). Order is fast-first for early sanity
    checking:

      1. Vanilla RAG (3 cells)
      2. Agentic RAG (3 cells)
      3. RLM (3 cells)
      4. MADAM-RAG (3 cells)

    Returns:
        List of 12 `ExperimentConfig` instances in the order above.
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


def is_experiment_complete(experiment_id: str, n_queries: int) -> bool:
    """Return whether every query has a successful result JSON on disk.

    Only counts results where `error` is `None`. Error results
    from previous rate-limit failures etc. are ignored, so the
    orchestrator will re-dispatch workers for that experiment.

    Args:
        experiment_id: Identifier of the experiment cell.
        n_queries: Expected total number of successful results.

    Returns:
        `True` when at least `n_queries` successful JSONs exist
        in the experiment's result directory.
    """
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
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
    return n_success >= n_queries


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
def run_worker(config_dict: dict, query_ids: list[str]) -> dict:
    """Process one worker's batch of queries for one experiment.

    Runs in its own Modal container. Sets up data symlinks, loads
    the FAISS index and QA system once, then iterates through the
    assigned queries with per-query checkpointing.

    Args:
        config_dict: Serialized `ExperimentConfig` (round-tripped
            via `to_dict()`).
        query_ids: Subset of query IDs this worker should process.

    Returns:
        Counts dict from `run_question_batch` (`completed`,
        `skipped`, `errors`, `total`).
    """
    setup_container()

    from src.experiments.experiment import ExperimentConfig, run_question_batch

    # Reconstruct ExperimentConfig from dict (drop derived 'corpus_type' key).
    cfg_kwargs = {k: v for k, v in config_dict.items() if k != 'corpus_type'}
    config = ExperimentConfig(**cfg_kwargs)

    # Load queries from Volume.
    queries_path = os.path.join(VOLUME_MOUNT, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl')
    queries: dict[str, dict] = {}
    with open(queries_path) as f:
        for line in f:
            line_dict = json.loads(line)
            queries[line_dict['query_id']] = line_dict

    summary = run_question_batch(
        config=config,
        query_ids=query_ids,
        queries=queries,
        results_dir=EXPERIMENTS_DIR,
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
    """Run all 12 experiments sequentially, fanning workers out per experiment.

    Each experiment dispatches up to 99 parallel `run_worker`
    containers via `.starmap`. Fully checkpoint-recoverable:
    re-running skips experiments that are already complete and
    queries within an experiment that already have a successful
    result JSON. Per-experiment summaries are written to
    `summary.json` under each experiment's result directory.
    """
    setup_container()

    from src.experiments.experiment import split_query_ids

    # Load query IDs from nq-questions.jsonl on the Volume.
    queries_path = os.path.join(VOLUME_MOUNT, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl')
    all_query_ids: list[str] = []
    with open(queries_path) as f:
        for line in f:
            line_dict = json.loads(line)
            all_query_ids.append(line_dict['query_id'])

    n_queries = len(all_query_ids)
    experiments = build_experiment_matrix()
    batches = split_query_ids(all_query_ids, n_workers=99)

    print(f"Starting {len(experiments)} experiments, {n_queries} queries each")
    print(f"Workers per experiment: {len(batches)}, ~{n_queries // len(batches)} queries/worker")
    print()

    for i, config in enumerate(experiments):
        exp_start = time.time()
        print(f"{'=' * 60}")
        print(f"[{i + 1}/{len(experiments)}] {config.experiment_id}")
        print(f"  arch={config.architecture}  attack={config.attack_type}  k={config.k or 'all'}")
        print(f"{'=' * 60}")

        # Refresh volume to see results written by previous experiments.
        volume.reload()

        if is_experiment_complete(config.experiment_id, n_queries):
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
        summary_dir = os.path.join(EXPERIMENTS_DIR, config.experiment_id)
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
    """Local entrypoint — dispatch the orchestrator to a Modal container.

    Invoke via `modal run --detach src/experiments/orchestrator.py`.
    """
    run_orchestrator.remote()
