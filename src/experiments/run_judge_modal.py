"""
experiments/run_judge_modal.py

Parallel Modal runner for the LLM judge pipeline.

Supports two modes:
  1. Full run — judges all ~12,840 experiment results
  2. Validation — judges the 492-question human-labeled sample

Usage:
    # --- Full judge run ---
    modal run --detach experiments/run_judge_modal.py

    # --- Validation (41-question sample x 12 experiments) ---
    modal run --detach experiments/run_judge_modal.py --validation

    # --- Dry run / report (local-only, no Modal upload) ---
    python experiments/run_judge_modal.py --validation --dry-run
    python experiments/run_judge_modal.py --validation --report-only
    python experiments/run_judge_modal.py --validation --report-only --timestamp 20260313-0015

    # Override model / reasoning (works for both modes)
    modal run --detach experiments/run_judge_modal.py --validation --model gpt-5-nano --reasoning-effort medium
"""

import csv
import json
import os
import time
from datetime import datetime

import modal


# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App('rag-poisoning-judge')

# Lighter image than the experiment runner — no FAISS, Contriever, torch.
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(
        'openai',
        'pydantic',
        'nltk',
        'numpy',
        'tenacity',
        'timeout-decorator',
        'python-dotenv',
    )
    # Mount only the files workers/orchestrator actually import.
    # Experiment results are read from the Modal Volume, not the image.
    .add_local_file('src/__init__.py', remote_path='/app/src/__init__.py')
    .add_local_file('src/experiments/__init__.py', remote_path='/app/src/experiments/__init__.py')
    .add_local_file('src/experiments/llm_judge.py', remote_path='/app/src/experiments/llm_judge.py')
    .add_local_file('src/experiments/llm-judge-prompt.md', remote_path='/app/src/experiments/llm-judge-prompt.md')
    .add_local_file('src/architectures/__init__.py', remote_path='/app/src/architectures/__init__.py')
    .add_local_file('src/architectures/utils.py', remote_path='/app/src/architectures/utils.py')
    .add_local_file('src/architectures/qa_system.py', remote_path='/app/src/architectures/qa_system.py')
    .add_local_file('src/data/__init__.py', remote_path='/app/src/data/__init__.py')
    .add_local_file('src/data/utils.py', remote_path='/app/src/data/utils.py')
)

volume = modal.Volume.from_name('rag-poisoning-data', create_if_missing=True)
VOLUME_MOUNT = '/vol'
# Results tree on the volume:
#   /vol/results/experiments/<experiment_id>/<question_id>.json
#   /vol/results/judge/<experiment_id>/<question_id>.json
#   /vol/results/judge_validation/judge_validation_<alias>_<effort>_<ts>/...
RESULTS_DIR = f'{VOLUME_MOUNT}/results'
EXPERIMENTS_DIR = f'{RESULTS_DIR}/experiments'
JUDGE_RESULTS_DIR = f'{RESULTS_DIR}/judge'
JUDGE_VALIDATION_DIR = f'{RESULTS_DIR}/judge_validation'
JUDGE_VALIDATION_BASE = f'{JUDGE_VALIDATION_DIR}/judge_validation'

MODEL_ALIASES = {'gpt-5-nano': 'nano', 'gpt-5-mini': 'mini', 'gpt-5': 'full'}

secrets = [modal.Secret.from_name('openai-rag-poisoning')]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = [
    f'{arch}_{attack}'
    for arch in ('vanilla', 'agentic', 'rlm', 'madam')
    for attack in ('clean', 'naive', 'corruptrag_ak')
]

# Pricing estimates (per 1M tokens).
MODEL_PRICING = {
    # (input_per_1M, output_per_1M)
    'gpt-5-nano': (0.10, 0.40),
    'gpt-5-mini': (0.40, 1.60),
    'gpt-5': (2.00, 8.00),
}
EMBEDDING_PER_1M = 0.02
EST_JUDGE_INPUT_TOKENS = 2500
EST_JUDGE_OUTPUT_TOKENS = 500
EST_EMBEDDING_TOKENS = 200

# Rough reasoning token multipliers relative to base output.
# High reasoning generates substantially more internal tokens.
REASONING_MULTIPLIER = {
    'low': 1.0,
    'medium': 2.0,
    'high': 4.0,
}

# Local paths (for validation CSV and local output).
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_CSV = os.path.join(
    _EXPERIMENTS_DIR,
    '..', '..', 'analysis', 'human_labels.csv',
)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60 * 5,  # 5 minutes per result
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    max_containers=99,
)
def judge_single_result(
    result_dict: dict,
    system_message: str,
    user_message_template: str,
    model: str,
    reasoning_effort: str,
    embedding_model: str,
    output_base_dir: str = '',
) -> dict:
    """Judge a single experiment result. One unit of Modal starmap work."""
    import sys
    if '/app' not in sys.path:
        sys.path.insert(0, '/app')

    from src.experiments.llm_judge import evaluate_response
    from openai import OpenAI

    openai_client = OpenAI()
    experiment_id = result_dict['experiment_id']
    question_id = result_dict['question_id']

    judge_result = evaluate_response(
        question=result_dict['question_text'],
        correct_answer=result_dict['correct_answer'],
        target_answer=result_dict.get('target_answer'),
        system_answer=result_dict['system_answer'],
        experiment_id=experiment_id,
        question_id=question_id,
        system_message=system_message,
        user_message_template=user_message_template,
        model=model,
        reasoning_effort=reasoning_effort,
        embedding_model=embedding_model,
        openai_client=openai_client,
    )

    # Write to volume.
    base_dir = output_base_dir or JUDGE_RESULTS_DIR
    exp_dir = os.path.join(base_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    judge_path = os.path.join(exp_dir, f'{question_id}.json')
    with open(judge_path, 'w') as f:
        json.dump(judge_result, f, indent=2)
    volume.commit()

    return judge_result


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60 * 5,  # 5 minutes per result
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    max_containers=99,
)
def judge_single_result_from_volume(
    experiment_id: str,
    question_id: str,
    system_message: str,
    user_message_template: str,
    model: str,
    reasoning_effort: str,
    embedding_model: str,
) -> dict:
    """Judge a single result by loading its JSON from the volume. Used for full runs."""
    import sys
    if '/app' not in sys.path:
        sys.path.insert(0, '/app')

    from src.experiments.llm_judge import evaluate_response
    from openai import OpenAI

    openai_client = OpenAI()

    fpath = os.path.join(EXPERIMENTS_DIR, experiment_id, f'{question_id}.json')
    with open(fpath) as f:
        result_dict = json.load(f)

    judge_result = evaluate_response(
        question=result_dict['question_text'],
        correct_answer=result_dict['correct_answer'],
        target_answer=result_dict.get('target_answer'),
        system_answer=result_dict['system_answer'],
        experiment_id=experiment_id,
        question_id=question_id,
        system_message=system_message,
        user_message_template=user_message_template,
        model=model,
        reasoning_effort=reasoning_effort,
        embedding_model=embedding_model,
        openai_client=openai_client,
    )

    exp_dir = os.path.join(JUDGE_RESULTS_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    judge_path = os.path.join(exp_dir, f'{question_id}.json')
    with open(judge_path, 'w') as f:
        json.dump(judge_result, f, indent=2)
    volume.commit()

    return judge_result


# ---------------------------------------------------------------------------
# Helpers (run inside orchestrator container)
# ---------------------------------------------------------------------------

def count_result_files() -> dict[str, int]:
    """Count result JSON files per experiment without parsing any JSON."""
    counts = {}
    for experiment_id in ALL_EXPERIMENTS:
        exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
        if not os.path.isdir(exp_dir):
            print(f"  WARNING: {experiment_id} -- no results directory")
            counts[experiment_id] = 0
            continue
        counts[experiment_id] = sum(
            1 for f in os.listdir(exp_dir)
            if f.endswith('.json') and f != 'summary.json'
        )
    return counts


def list_result_ids() -> list[tuple[str, str]]:
    """Return (experiment_id, question_id) pairs for all result files without parsing JSON."""
    ids = []
    for experiment_id in ALL_EXPERIMENTS:
        exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
        if not os.path.isdir(exp_dir):
            print(f"  WARNING: {experiment_id} -- no results directory")
            continue
        for fname in sorted(os.listdir(exp_dir)):
            if fname.endswith('.json') and fname != 'summary.json':
                ids.append((experiment_id, fname[:-5]))
    return ids


def get_already_judged(base_dir: str = '') -> set[tuple[str, str]]:
    """Return (experiment_id, question_id) pairs already successfully judged."""
    judge_dir = base_dir or JUDGE_RESULTS_DIR
    judged = set()
    if not os.path.isdir(judge_dir):
        return judged
    for experiment_id in os.listdir(judge_dir):
        exp_dir = os.path.join(judge_dir, experiment_id)
        if not os.path.isdir(exp_dir):
            continue
        for fname in os.listdir(exp_dir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(exp_dir, fname)
            try:
                with open(fpath) as f:
                    prev = json.loads(f.read())
                if prev.get('classification') is not None:
                    question_id = fname.replace('.json', '')
                    judged.add((experiment_id, question_id))
            except (json.JSONDecodeError, OSError):
                pass
    return judged


def print_cost_estimate(
    counts: dict[str, int],
    model: str = 'gpt-5-mini',
    reasoning_effort: str = 'high',
):
    """Print estimated API cost for the run.

    counts: per-experiment result counts (e.g. from count_result_files() or
            collections.Counter over a list of dicts).
    """
    n = sum(counts.values())

    input_per_1m, output_per_1m = MODEL_PRICING.get(
        model, MODEL_PRICING['gpt-5-mini']
    )
    reason_mult = REASONING_MULTIPLIER.get(reasoning_effort, 2.0)

    # Reasoning tokens are billed at output rate.
    est_reasoning_tokens = int(EST_JUDGE_OUTPUT_TOKENS * reason_mult)
    est_total_output = EST_JUDGE_OUTPUT_TOKENS + est_reasoning_tokens

    judge_input_cost = n * EST_JUDGE_INPUT_TOKENS / 1_000_000 * input_per_1m
    judge_output_cost = n * est_total_output / 1_000_000 * output_per_1m
    embed_cost = n * EST_EMBEDDING_TOKENS / 1_000_000 * EMBEDDING_PER_1M
    total_cost = judge_input_cost + judge_output_cost + embed_cost

    print(f"\n--- COST ESTIMATE ---")
    print(f"Model:            {model} (reasoning={reasoning_effort})")
    print(f"Results to judge: {n} (estimate — excludes NOISE/already-judged)")
    print(f"API calls:        {n} judge + {n} embedding = {2 * n} total")
    print(f"Est tokens/call:  ~{EST_JUDGE_INPUT_TOKENS} in + ~{est_total_output} out "
          f"({EST_JUDGE_OUTPUT_TOKENS} answer + ~{est_reasoning_tokens} reasoning)")
    print(f"Judge cost:       ${judge_input_cost + judge_output_cost:.2f}")
    print(f"Embedding cost:   ${embed_cost:.2f}")
    print(f"Estimated total:  ${total_cost:.2f}")

    print(f"\nPer experiment:")
    for exp_id in ALL_EXPERIMENTS:
        count = counts.get(exp_id, 0)
        if count > 0:
            print(f"  {exp_id}: {count}")


def _stream_progress(results_iter, total: int, exp_total) -> tuple[int, int]:
    """Stream a judge starmap iterator, printing per-result progress.

    Returns (total_completed, total_errors).
    """
    from collections import Counter

    total_completed = 0
    total_errors = 0
    exp_completed: Counter[str] = Counter()
    exp_errors: Counter[str] = Counter()
    start_time = time.time()

    for result in results_iter:
        exp_id = result.get('experiment_id', '?')
        if result.get('error'):
            total_errors += 1
            exp_errors[exp_id] += 1
        else:
            total_completed += 1
            exp_completed[exp_id] += 1

        done = total_completed + total_errors
        elapsed = time.time() - start_time
        rate = done / elapsed * 60 if elapsed > 0 else 0
        remaining = total - done
        eta_min = remaining / rate if rate > 0 else 0

        exp_done = exp_completed[exp_id] + exp_errors[exp_id]
        exp_tot = exp_total[exp_id]
        q_id = result.get('question_id', '?')

        print(f"  [{done}/{total}] {exp_id}-{q_id} ({exp_done}/{exp_tot})"
              f"  ({rate:.0f}/min, ~{eta_min:.0f}min left)", flush=True)

    # Final per-experiment summary.
    print(f"\n{'=' * 60}")
    print(f"{'Experiment':<30} {'OK':>5} {'Err':>5} {'Total':>6}")
    print('-' * 60)
    for exp_id in ALL_EXPERIMENTS:
        ok = exp_completed.get(exp_id, 0)
        err = exp_errors.get(exp_id, 0)
        tot = exp_total.get(exp_id, 0)
        if tot > 0:
            print(f"  {exp_id:<28} {ok:>5} {err:>5} {tot:>6}")

    return total_completed, total_errors


def _run_judge(
    to_judge: list[dict],
    system_message: str,
    user_message_template: str,
    model: str,
    reasoning_effort: str,
    embedding_model: str,
    output_base_dir: str = '',
) -> tuple[int, int]:
    """Dispatch validation judge work (inline dicts) via starmap.

    Used by run_validation_orchestrator, where result data is already in memory.
    Modal's max_containers=99 handles concurrency limiting.
    """
    from collections import Counter

    total = len(to_judge)
    exp_total: Counter[str] = Counter(r['experiment_id'] for r in to_judge)

    print(f"\nDispatching {total} results to up to 99 parallel containers...\n")

    worker_args = [
        (r, system_message, user_message_template,
         model, reasoning_effort, embedding_model, output_base_dir)
        for r in to_judge
    ]

    return _stream_progress(
        judge_single_result.starmap(worker_args, order_outputs=False),
        total, exp_total,
    )


def _run_judge_from_volume(
    to_judge: list[tuple[str, str]],
    system_message: str,
    user_message_template: str,
    model: str,
    reasoning_effort: str,
    embedding_model: str,
) -> tuple[int, int]:
    """Dispatch full judge run via starmap, passing only IDs to workers.

    Workers load their own result JSON from the volume, avoiding the need to
    deserialize 13,800 result files in the orchestrator.
    """
    from collections import Counter

    total = len(to_judge)
    exp_total: Counter[str] = Counter(exp_id for exp_id, _ in to_judge)

    print(f"\nDispatching {total} results to up to 99 parallel containers...\n")

    worker_args = [
        (exp_id, q_id, system_message, user_message_template,
         model, reasoning_effort, embedding_model)
        for exp_id, q_id in to_judge
    ]

    return _stream_progress(
        judge_single_result_from_volume.starmap(worker_args, order_outputs=False),
        total, exp_total,
    )


# ---------------------------------------------------------------------------
# Full judge orchestrator
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60 * 60 * 6,  # 6 hours
)
def run_judge_orchestrator(
    model: str,
    reasoning_effort: str,
    embedding_model: str,
    dry_run: bool = False,
):
    """Orchestrate the full judge run over all experiment results."""
    import sys
    if '/app' not in sys.path:
        sys.path.insert(0, '/app')

    volume.reload()

    if dry_run:
        from src.data.utils import get_noise_question_ids
        noise_ids = get_noise_question_ids()
        counts = count_result_files()
        already_judged = get_already_judged()
        # Subtract already-judged and NOISE from per-experiment counts for estimate.
        from collections import Counter
        judged_per_exp: Counter[str] = Counter(exp_id for exp_id, _ in already_judged)
        noise_per_exp: Counter[str] = Counter()
        for exp_id in ALL_EXPERIMENTS:
            exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
            if os.path.isdir(exp_dir):
                noise_per_exp[exp_id] = sum(
                    1 for f in os.listdir(exp_dir)
                    if f[:-5] in noise_ids and f.endswith('.json')
                )
        adjusted = {
            exp_id: max(0, counts.get(exp_id, 0) - judged_per_exp[exp_id] - noise_per_exp[exp_id])
            for exp_id in ALL_EXPERIMENTS
        }
        print_cost_estimate(adjusted, model, reasoning_effort)
        return

    from src.data.utils import get_noise_question_ids
    from src.experiments.llm_judge import load_judge_prompt

    noise_ids = get_noise_question_ids()

    print("Listing experiment result IDs from volume...")
    all_ids = list_result_ids()
    print(f"Total results: {len(all_ids)}")

    already_judged = get_already_judged()
    print(f"Already judged: {len(already_judged)}")

    to_judge = [
        (exp_id, q_id) for exp_id, q_id in all_ids
        if (exp_id, q_id) not in already_judged
        and q_id not in noise_ids
    ]
    print(f"To judge: {len(to_judge)} (excluding NOISE and already-judged)")

    if not to_judge:
        print("Nothing to judge -- all results already processed.")
        return

    system_message, user_message_template = load_judge_prompt()

    total_completed, total_errors = _run_judge_from_volume(
        to_judge, system_message, user_message_template,
        model, reasoning_effort, embedding_model,
    )

    print(f"\n{'=' * 60}")
    print("JUDGE RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"Completed:  {total_completed}")
    print(f"Errors:     {total_errors}")

    # Write summary.
    os.makedirs(JUDGE_RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(JUDGE_RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'completed': total_completed,
            'errors': total_errors,
            'previously_judged': len(already_judged),
            'total_results': len(all_ids),
            'model': model,
            'reasoning_effort': reasoning_effort,
        }, f, indent=2)
    volume.commit()


# ---------------------------------------------------------------------------
# Validation orchestrator
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=60 * 60,  # 1 hour (plenty for 492 results)
)
def run_validation_orchestrator(
    review_rows: list[dict],
    model: str,
    reasoning_effort: str,
    embedding_model: str,
    timestamp: str,
):
    """Orchestrate the validation judge run over the human-labeled sample."""
    import sys
    if '/app' not in sys.path:
        sys.path.insert(0, '/app')

    from src.data.utils import get_noise_question_ids
    from src.experiments.llm_judge import load_judge_prompt

    noise_ids = get_noise_question_ids()

    volume.reload()

    # Filter NOISE questions before dispatching.
    pre_filter = len(review_rows)
    review_rows = [r for r in review_rows if r['question_id'] not in noise_ids]
    n_filtered = pre_filter - len(review_rows)
    if n_filtered:
        print(f"Filtered {n_filtered} NOISE questions ({', '.join(sorted(noise_ids))})")

    alias = MODEL_ALIASES.get(model, model)
    output_dir = f'{JUDGE_VALIDATION_BASE}_{alias}_{reasoning_effort}_{timestamp}'

    print(f"Validation sample: {len(review_rows)} rows")
    print(f"Output dir: {output_dir}")

    system_message, user_message_template = load_judge_prompt()

    total_completed, total_errors = _run_judge(
        review_rows, system_message, user_message_template,
        model, reasoning_effort, embedding_model,
        output_base_dir=output_dir,
    )

    print(f"\n{'=' * 60}")
    print("VALIDATION JUDGE RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"Completed:  {total_completed}")
    print(f"Errors:     {total_errors}")
    print(f"\nResults at: {output_dir}")
    print("Run with --report-only to download results and generate the report.")


# ---------------------------------------------------------------------------
# Local helpers for validation report
# ---------------------------------------------------------------------------

def _load_review_csv() -> list[dict]:
    """Load the human-reviewed sample data from the local CSV."""
    rows = []
    with open(REVIEW_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for line_dict in reader:
            rows.append(line_dict)
    return rows


def _find_validation_dir(vol: modal.Volume, timestamp: str | None = None) -> str | None:
    """Find a judge_validation_* directory on the volume.

    Looks in `results/judge_validation/` (current runs) first, then falls
    back to `results/archive/` (superseded runs). If `timestamp` is given
    (e.g. '20260313-1430'), returns the dir ending with that timestamp;
    otherwise returns the most recent across both locations.
    """
    validation_dirs: list[str] = []
    for parent in ('results/judge_validation/', 'results/archive/'):
        try:
            entries = list(vol.listdir(parent))
        except Exception:
            continue
        # Match both new (judge_validation_) and old (judge-validation-) naming.
        validation_dirs.extend(
            entry.path for entry in entries
            if (os.path.basename(entry.path).startswith('judge_validation_')
                or os.path.basename(entry.path).startswith('judge-validation-'))
        )

    validation_dirs = sorted(set(validation_dirs))

    if timestamp:
        matches = [d for d in validation_dirs if d.endswith(timestamp)]
        return matches[0] if matches else None

    return validation_dirs[-1] if validation_dirs else None


def _load_local_judge_results(
    local_output_dir: str,
    review_data: list[dict],
) -> tuple[list[dict], str]:
    """Load judge results from local files, excluding NOISE questions."""
    from src.data.utils import get_noise_question_ids

    noise_ids = get_noise_question_ids()

    human_lookup = {}
    for row in review_data:
        key = (row['experiment_id'], row['question_id'])
        human_lookup[key] = {
            'human_label': row.get('human_label', ''),
            'human_target_present': row.get('target_present', ''),
        }

    judge_results = []
    noise_skipped = 0
    for experiment_id in ALL_EXPERIMENTS:
        local_exp_dir = os.path.join(local_output_dir, experiment_id)
        if not os.path.isdir(local_exp_dir):
            continue
        for fname in sorted(os.listdir(local_exp_dir)):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(local_exp_dir, fname)) as f:
                    result = json.load(f)
                if result.get('classification') is not None:
                    if result['question_id'] in noise_ids:
                        noise_skipped += 1
                        continue
                    key = (result['experiment_id'], result['question_id'])
                    if key in human_lookup:
                        result['human_label'] = human_lookup[key]['human_label']
                        result['human_target_present'] = human_lookup[key]['human_target_present']
                    judge_results.append(result)
            except (json.JSONDecodeError, KeyError, OSError):
                continue

    if noise_skipped:
        print(f"Excluded {noise_skipped} NOISE question results from report")

    return judge_results, local_output_dir


def _find_local_validation_dir(timestamp: str) -> str | None:
    """Find a local judge_validation_* dir matching the given timestamp.

    Looks in `results/judge_validation/` (current runs) first, then falls
    back to `results/archive/` (superseded runs).
    """
    local_results = os.path.join(_EXPERIMENTS_DIR, 'results')
    for bucket in ('judge_validation', 'archive'):
        parent = os.path.join(local_results, bucket)
        if not os.path.isdir(parent):
            continue
        matches = [
            d for d in sorted(os.listdir(parent))
            if (d.startswith('judge_validation_') or d.startswith('judge-validation-'))
            and d.endswith(timestamp)
        ]
        if matches:
            return os.path.join(parent, matches[0])
    return None


def _download_validation_results(
    review_data: list[dict],
    timestamp: str | None = None,
) -> tuple[list[dict], str]:
    """Download validation judge results from Modal Volume to local disk.

    Checks for a matching local dir first (by timestamp). If found, skips
    the Modal download entirely. Otherwise finds the dir on the volume and
    downloads it.

    Returns (judge_results, local_output_dir).
    """
    # Check locally first — avoids duplicate downloads when local dirs
    # have been renamed to the new naming convention.
    if timestamp:
        local_match = _find_local_validation_dir(timestamp)
        if local_match:
            local_count = 0
            for experiment_id in ALL_EXPERIMENTS:
                local_exp_dir = os.path.join(local_match, experiment_id)
                if os.path.isdir(local_exp_dir):
                    local_count += sum(
                        1 for f in os.listdir(local_exp_dir) if f.endswith('.json')
                    )
            if local_count > 0:
                print(f"Found local results: {local_match} ({local_count} files)")
                local_output_dir = local_match
                # Skip straight to loading results below.
                return _load_local_judge_results(
                    local_output_dir, review_data,
                )

    vol = modal.Volume.from_name('rag-poisoning-data')

    remote_dir = _find_validation_dir(vol, timestamp)
    if not remote_dir:
        print("No judge_validation_* directories found on volume.")
        return [], ''

    # Mirror the remote path locally: remote_dir is like
    # "results/judge_validation/judge_validation_mini_high_20260313-1934" or
    # "results/archive/judge-validation-20260313-0046" — preserve the bucket
    # (judge_validation/ or archive/) when mirroring.
    bucket = os.path.basename(os.path.dirname(remote_dir))
    dir_name = os.path.basename(remote_dir)
    local_output_dir = os.path.join(_EXPERIMENTS_DIR, 'results', bucket, dir_name)
    os.makedirs(local_output_dir, exist_ok=True)

    print(f"Remote: {remote_dir}")
    print(f"Local:  {local_output_dir}")

    # Scan remote files once — cache entries to avoid repeated listdir calls
    # (Modal Volume API can return inconsistent results across calls).
    remote_entries: dict[str, list] = {}
    remote_count = 0
    for experiment_id in ALL_EXPERIMENTS:
        try:
            entries = [
                e for e in vol.listdir(f'{remote_dir}/{experiment_id}/')
                if e.path.endswith('.json')
            ]
            remote_entries[experiment_id] = entries
            remote_count += len(entries)
        except Exception:
            pass

    # Count existing local files.
    local_count = 0
    if os.path.isdir(local_output_dir):
        for experiment_id in ALL_EXPERIMENTS:
            local_exp_dir = os.path.join(local_output_dir, experiment_id)
            if os.path.isdir(local_exp_dir):
                local_count += sum(
                    1 for f in os.listdir(local_exp_dir) if f.endswith('.json')
                )

    if local_count >= remote_count and local_count > 0:
        print(f"Local results already up to date ({local_count} files). Skipping download.")
    else:
        if local_count > 0:
            print(f"Local has {local_count} files, remote has {remote_count}. Re-downloading...")
        else:
            print(f"Downloading {remote_count} files...")

        downloaded = 0
        for experiment_id, entries in remote_entries.items():
            local_exp_dir = os.path.join(local_output_dir, experiment_id)
            os.makedirs(local_exp_dir, exist_ok=True)

            for entry in entries:
                fname = os.path.basename(entry.path)
                local_path = os.path.join(local_exp_dir, fname)

                content = b''
                for chunk in vol.read_file(entry.path):
                    content += chunk
                with open(local_path, 'wb') as f:
                    f.write(content)
                downloaded += 1

        print(f"Downloaded {downloaded} files to {local_output_dir}")

    return _load_local_judge_results(local_output_dir, review_data)


def _generate_validation_report(
    judge_results: list[dict],
    review_data: list[dict],
    output_dir: str,
):
    """Generate and save the validation agreement report locally."""
    from src.experiments.run_judge_local import build_agreement_report

    report = build_agreement_report(judge_results, review_data)
    print(report)

    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
        f.write('\n')
    print(f"\nReport saved to {report_path}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = 'gpt-5-mini',
    reasoning_effort: str = 'high',
    embedding_model: str = 'text-embedding-3-small',
    dry_run: bool = False,
    validation: bool = False,
    report_only: bool = False,
):
    if validation:
        review_data = _load_review_csv()
        n_questions = len(set(r['question_id'] for r in review_data))
        n_experiments = len(set(r['experiment_id'] for r in review_data))
        print(f"Loaded {len(review_data)} rows ({n_questions} questions x {n_experiments} experiments)")

        if report_only:
            print(f"\nDownloading validation results from Modal Volume...")
            judge_results, local_dir = _download_validation_results(review_data, timestamp=None)
            if not judge_results:
                print("No results found. Run validation first.")
                return
            print(f"Loaded {len(judge_results)} judge results")
            _generate_validation_report(judge_results, review_data, local_dir)
            return

        if dry_run:
            from collections import Counter
            counts = Counter(r['experiment_id'] for r in review_data)
            print_cost_estimate(dict(counts), model, reasoning_effort)
            return

        # Filter NOISE questions before sending to Modal.
        from src.data.utils import get_noise_question_ids
        noise_ids = get_noise_question_ids()
        pre_filter = len(review_data)
        review_data = [r for r in review_data if r['question_id'] not in noise_ids]
        n_filtered = pre_filter - len(review_data)
        if n_filtered:
            print(f"Filtered {n_filtered} NOISE questions ({', '.join(sorted(noise_ids))})")
            print(f"Sending {len(review_data)} rows to Modal")

        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        run_validation_orchestrator.remote(
            review_rows=review_data,
            model=model,
            reasoning_effort=reasoning_effort,
            embedding_model=embedding_model,
            timestamp=timestamp,
        )
    else:
        if report_only:
            print("--report-only is only supported with --validation")
            return

        run_judge_orchestrator.remote(
            model=model,
            reasoning_effort=reasoning_effort,
            embedding_model=embedding_model,
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# Direct invocation for local-only operations (skips Modal image upload).
#   python experiments/run_judge_modal.py --validation --dry-run
#   python experiments/run_judge_modal.py --validation --report-only
# For actual judge runs, use: modal run --detach experiments/run_judge_modal.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys as _sys

    # Ensure repo root is on sys.path so `from src.X.Y import Z` works.
    _repo_root = os.path.normpath(os.path.join(_EXPERIMENTS_DIR, '..', '..'))
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-5-mini')
    parser.add_argument('--reasoning-effort', default='high',
                        choices=['low', 'medium', 'high'])
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--timestamp',
                        help="Download a specific validation run (e.g. '20260313-0015'). "
                             "Defaults to latest.")
    args = parser.parse_args()

    if not args.validation:
        print("Direct invocation only supports --validation mode.")
        print("For full judge runs, use: modal run experiments/run_judge_modal.py")
        raise SystemExit(1)

    review_data = _load_review_csv()
    n_questions = len(set(r['question_id'] for r in review_data))
    n_experiments = len(set(r['experiment_id'] for r in review_data))
    print(f"Loaded {len(review_data)} rows ({n_questions} questions x {n_experiments} experiments)")

    if args.dry_run:
        from collections import Counter
        counts = Counter(r['experiment_id'] for r in review_data)
        print_cost_estimate(dict(counts), args.model, args.reasoning_effort)
    elif args.report_only:
        if args.timestamp:
            print(f"\nDownloading validation run {args.timestamp} from Modal Volume...")
        else:
            print(f"\nDownloading latest validation results from Modal Volume...")
        judge_results, local_dir = _download_validation_results(review_data, args.timestamp)
        if not judge_results:
            print("No results found. Run validation first.")
            raise SystemExit(1)
        print(f"Loaded {len(judge_results)} judge results")
        _generate_validation_report(judge_results, review_data, local_dir)
    else:
        print("Direct invocation only supports --dry-run or --report-only.")
        print("To run validation: modal run experiments/run_judge_modal.py --validation")
        raise SystemExit(1)
