"""Modal runner for the NOISE filter — parallelizes noise classification across up to 99 containers.

Prerequisites:
    Modal credentials, the `openai-rag-poisoning` Modal Secret, and
    `nq-questions-gold-filtered.jsonl` produced by
    `src/data/filter_gold_questions.py`.

Usage:
    modal run --detach src/experiments/run_noise_modal.py
    modal run src/experiments/run_noise_modal.py --dry-run
    python src/experiments/run_noise_modal.py --report-only
    modal run --detach src/experiments/run_noise_modal.py --model gpt-5-nano
    modal run --detach src/experiments/run_noise_modal.py --no-web-search

Output:
    Per-question NOISE JSONs under `/vol/results/noise/` on the Modal
    Volume; downloaded locally to `src/experiments/results/noise/`
    after each run. Each JSON contains the model's NOISE/non-NOISE
    classification plus reasoning.

Notes:
    Run all commands from the repo root — the Modal image mounts
    `src/__init__.py`, `src/experiments/__init__.py`, and
    `src/experiments/noise_filter.py` using paths relative to CWD.
"""

import json
import os
import sys
import time

import modal


# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App('rag-poisoning-noise')

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(
        'openai',
        'pydantic',
        'python-dotenv',
    )
    # Mount only the noise filter module + dependencies.
    # Paths are relative to CWD (repo root), not this file.
    .add_local_file('src/__init__.py', remote_path='/app/src/__init__.py')
    .add_local_file('src/experiments/__init__.py', remote_path='/app/src/experiments/__init__.py')
    .add_local_file(
        'src/experiments/noise_filter.py',
        remote_path='/app/src/experiments/noise_filter.py',
    )
)

volume = modal.Volume.from_name('rag-poisoning-data', create_if_missing=True)
VOLUME_MOUNT = '/vol'
NOISE_RESULTS_DIR = f'{VOLUME_MOUNT}/results/noise'

secrets = [modal.Secret.from_name('openai-rag-poisoning')]

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60 * 2,  # 2 min per question (generous)
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    max_containers=99,
)
def classify_noise(
    query: dict,
    model: str,
    reasoning_effort: str,
    web_search: bool = True,
) -> dict:
    """Classify a single query as NOISE or not. One unit of starmap work."""
    import sys
    if '/app' not in sys.path:
        sys.path.insert(0, '/app')

    from src.experiments.noise_filter import check_noise

    query_id = query['query_id']
    # 'question_id' key matches the on-disk schema for noise JSONs (kept for
    # backwards compatibility with persisted results); value is a query_id.
    result = {
        'question_id': query_id,
        'question': query['question'],
        'correct_answer': query['correct_answer'],
        'target_answer': query['target_answer'],
        'model': model,
        'reasoning_effort': reasoning_effort,
        'web_search': web_search,
        'is_noise': None,
        'noise_type': None,
        'confidence': None,
        'reasoning': None,
        'input_tokens': None,
        'output_tokens': None,
        'total_tokens': None,
        'latency_seconds': None,
        'error': None,
    }

    try:
        t0 = time.monotonic()
        noise_result, usage = check_noise(
            question=query['question'],
            correct_answer=query['correct_answer'],
            target_answer=query['target_answer'],
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
        )
        elapsed = time.monotonic() - t0

        result['is_noise'] = noise_result.is_noise
        result['noise_type'] = noise_result.noise_type
        result['confidence'] = noise_result.confidence
        result['reasoning'] = noise_result.reasoning
        result['input_tokens'] = usage['input_tokens']
        result['output_tokens'] = usage['output_tokens']
        result['total_tokens'] = usage['total_tokens']
        result['latency_seconds'] = round(elapsed, 3)
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {e}"
        result['latency_seconds'] = round(time.monotonic() - t0, 3)

    # Write to volume.
    os.makedirs(NOISE_RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(NOISE_RESULTS_DIR, f'{query_id}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    volume.commit()

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60 * 60,  # 1 hour
)
def run_noise_orchestrator(model: str, reasoning_effort: str, web_search: bool = True, dry_run: bool = False):
    """Load queries from volume, skip already-classified, dispatch workers."""
    volume.reload()

    # Load gold-filtered queries from volume.
    queries_path = os.path.join(
        VOLUME_MOUNT, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl'
    )
    queries = []
    with open(queries_path) as f:
        for line in f:
            queries.append(json.loads(line))

    print(f"Total queries: {len(queries)}")

    # Check which are already done.
    already_done = set()
    if os.path.isdir(NOISE_RESULTS_DIR):
        for fname in os.listdir(NOISE_RESULTS_DIR):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(NOISE_RESULTS_DIR, fname)
            try:
                with open(fpath) as f:
                    r = json.load(f)
                if r.get('noise_type') is not None and r.get('error') is None:
                    already_done.add(r['question_id'])
            except (json.JSONDecodeError, OSError, KeyError):
                pass

    to_classify = [query for query in queries if query['query_id'] not in already_done]
    print(f"Already done: {len(already_done)}")
    print(f"To classify:  {len(to_classify)}")

    if dry_run:
        print("\n--- DRY RUN ---")
        print(f"Would classify {len(to_classify)} queries with {model}")
        return

    if not to_classify:
        print("All queries already classified.")
        return

    # Dispatch via starmap.
    print(f"\nDispatching {len(to_classify)} queries to up to 99 containers...")

    worker_args = [(query, model, reasoning_effort, web_search) for query in to_classify]

    completed = 0
    errors = 0
    noise_count = 0
    start_time = time.time()

    for result in classify_noise.starmap(worker_args, order_outputs=False):
        if result.get('error'):
            errors += 1
        else:
            completed += 1
            if result.get('is_noise'):
                noise_count += 1

        done = completed + errors
        elapsed = time.time() - start_time
        rate = done / elapsed * 60 if elapsed > 0 else 0

        if done % 50 == 0 or done == len(to_classify):
            print(f"  [{done}/{len(to_classify)}] "
                  f"{completed} ok, {errors} err, {noise_count} noise "
                  f"({rate:.0f}/min)", flush=True)

    print(f"\n{'=' * 60}")
    print("NOISE FILTER COMPLETE")
    print(f"{'=' * 60}")
    print(f"Classified:  {completed}")
    print(f"Errors:      {errors}")
    print(f"NOISE found: {noise_count}")

    # Write summary.
    summary_path = os.path.join(NOISE_RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'completed': completed,
            'errors': errors,
            'noise_found': noise_count,
            'previously_done': len(already_done),
            'total_questions': len(queries),
            'model': model,
            'reasoning_effort': reasoning_effort,
            'web_search': web_search,
        }, f, indent=2)
    volume.commit()


# ---------------------------------------------------------------------------
# Download results from volume to local disk
# ---------------------------------------------------------------------------

def download_results() -> str:
    """Download noise results from Modal Volume to local results/noise/ dir.

    Returns the local output directory path.
    """
    vol = modal.Volume.from_name('rag-poisoning-data')
    local_dir = os.path.join(_EXPERIMENTS_DIR, 'results', 'noise')
    os.makedirs(local_dir, exist_ok=True)

    try:
        entries = [
            e for e in vol.listdir('results/noise/')
            if e.path.endswith('.json')
        ]
    except Exception:
        print("No noise results found on volume.")
        return local_dir

    # Count local files.
    local_count = sum(1 for f in os.listdir(local_dir) if f.endswith('.json'))

    if local_count >= len(entries) and local_count > 0:
        print(f"Local results already up to date ({local_count} files).")
        return local_dir

    print(f"Downloading {len(entries)} files (local has {local_count})...")
    downloaded = 0
    for entry in entries:
        fname = os.path.basename(entry.path)
        local_path = os.path.join(local_dir, fname)

        content = b''
        for chunk in vol.read_file(entry.path):
            content += chunk
        with open(local_path, 'wb') as f:
            f.write(content)
        downloaded += 1

    print(f"Downloaded {downloaded} files to {local_dir}")
    return local_dir


# ---------------------------------------------------------------------------
# Local entrypoint (Modal CLI)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = 'gpt-5-mini',
    reasoning_effort: str = 'high',
    web_search: bool = True,
    dry_run: bool = False,
):
    run_noise_orchestrator.remote(
        model=model,
        reasoning_effort=reasoning_effort,
        web_search=web_search,
        dry_run=dry_run,
    )

    if not dry_run:
        print("\nDownloading results from volume...")
        local_dir = download_results()

        # Print report.
        _repo_root = os.path.normpath(os.path.join(_EXPERIMENTS_DIR, '..', '..'))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

        from src.experiments.noise_filter import print_report
        print_report(local_dir)


# ---------------------------------------------------------------------------
# Direct invocation for report-only (no Modal image upload)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    _repo_root = os.path.normpath(os.path.join(_EXPERIMENTS_DIR, '..', '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true',
                        help="Download results from volume and print report")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model', default='gpt-5-mini')
    parser.add_argument('--reasoning-effort', default='high',
                        choices=['low', 'medium', 'high'])
    parser.add_argument('--no-web-search', action='store_true',
                        help="Disable web search (enabled by default)")
    args = parser.parse_args()

    if args.report_only:
        print("Downloading noise results from Modal Volume...")
        local_dir = download_results()
        from src.experiments.noise_filter import print_report
        print_report(local_dir)
    else:
        print("Direct invocation only supports --report-only.")
        print("For full runs, use: modal run src/experiments/run_noise_modal.py")
        raise SystemExit(1)
