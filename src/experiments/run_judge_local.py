"""
experiments/run_judge_local.py

Sequential local runner for validating the LLM judge against the 41-question
manual review sample (492 results across 12 experiments).

Runs evaluate_response() sequentially, writes per-question JSONs with
checkpointing, then prints an agreement report comparing judge labels
against human labels (accuracy, per-category precision/recall, confusion
matrix, target-answer detection agreement).

Usage:
    python run_judge_local.py
    python run_judge_local.py --reasoning-effort high
    python run_judge_local.py --output-dir results/judge-validation-v2
"""

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from typing import Optional

from tqdm import tqdm
from openai import OpenAI

from src.experiments.llm_judge import (
    ALL_EXPERIMENTS,
    Classification,
    DEFAULT_REASONING_EFFORT,
    EMBEDDING_MODEL,
    JUDGE_MODEL,
    evaluate_response,
    load_judge_prompt,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REVIEW_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'analysis', 'human_labels.csv',
)

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'judge-validation',
)

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _find_local_validation_dir(timestamp: str) -> str | None:
    """Find a local judge_validation_* dir matching the given timestamp."""
    if not os.path.isdir(_RESULTS_DIR):
        return None
    matches = [
        d for d in sorted(os.listdir(_RESULTS_DIR))
        if (d.startswith('judge_validation_') or d.startswith('judge-validation-'))
        and d.endswith(timestamp)
    ]
    if not matches:
        return None
    return os.path.join(_RESULTS_DIR, matches[0])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_review_data(csv_path: str = REVIEW_CSV) -> list[dict]:
    """Load the human-reviewed sample data from CSV."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for line_dict in reader:
            rows.append(line_dict)
    return rows


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------

def run_validation(
    review_data: list[dict],
    output_dir: str,
    model: str = JUDGE_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    embedding_model: str = EMBEDDING_MODEL,
) -> list[dict]:
    """Run the judge over all review samples with checkpointing.

    Returns list of judge result dicts (augmented with human labels).
    """
    system_message, user_message_template = load_judge_prompt()
    openai_client = OpenAI()

    total = len(review_data)

    # Pre-scan for cached results so tqdm starts at the right position.
    cached_results = []
    cached_indices = set()
    for i, row in enumerate(review_data):
        exp_dir = os.path.join(output_dir, row['experiment_id'])
        judge_path = os.path.join(exp_dir, f"{row['question_id']}.json")
        if os.path.exists(judge_path):
            try:
                with open(judge_path) as f:
                    prev = json.loads(f.read())
                if prev.get('classification') is not None:
                    cached_results.append(prev)
                    cached_indices.add(i)
            except (json.JSONDecodeError, OSError):
                pass

    judge_results = list(cached_results)
    skipped = len(cached_results)
    completed = 0
    errors = 0

    question_times: list[float] = []
    experiment_times: dict[str, list[float]] = defaultdict(list)
    run_start = time.monotonic()

    to_process = total - skipped
    print(f"  {skipped} cached, {to_process} to judge\n")

    pbar = tqdm(
        total=total,
        initial=skipped,
        desc="Judging",
        unit="q",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    for i, row in enumerate(review_data):
        if i in cached_indices:
            continue

        experiment_id = row['experiment_id']
        question_id = row['question_id']

        exp_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        judge_path = os.path.join(exp_dir, f'{question_id}.json')

        target_answer = row.get('target_answer') or None

        q_start = time.monotonic()
        result = evaluate_response(
            question=row['question_text'],
            correct_answer=row['correct_answer'],
            target_answer=target_answer,
            system_answer=row['system_answer'],
            experiment_id=experiment_id,
            question_id=question_id,
            system_message=system_message,
            user_message_template=user_message_template,
            model=model,
            reasoning_effort=reasoning_effort,
            embedding_model=embedding_model,
            openai_client=openai_client,
        )
        q_elapsed = time.monotonic() - q_start

        question_times.append(q_elapsed)
        experiment_times[experiment_id].append(q_elapsed)

        result['human_label'] = row.get('human_label', '')
        result['human_target_present'] = row.get('target_present', '')

        with open(judge_path, 'w') as f:
            json.dump(result, f, indent=2)

        judge_results.append(result)

        if result.get('error'):
            errors += 1
        else:
            completed += 1

        avg_q = sum(question_times) / len(question_times)
        pbar.set_postfix_str(
            f"last={q_elapsed:.1f}s avg={avg_q:.1f}s ok={completed} err={errors}"
        )
        pbar.update(1)

    pbar.close()

    total_elapsed = time.monotonic() - run_start

    # Timing summary.
    print(f"\n{'=' * 70}")
    print("TIMING SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total wall time:    {_fmt_duration(total_elapsed)}")
    print(f"Questions judged:   {len(question_times)}  (+ {skipped} cached)")
    if question_times:
        avg_q = sum(question_times) / len(question_times)
        print(f"Avg per question:   {avg_q:.2f}s")
        print(f"Min per question:   {min(question_times):.2f}s")
        print(f"Max per question:   {max(question_times):.2f}s")

    if experiment_times:
        print(f"\n{'Experiment':<35} {'Count':>5} {'Total':>9} {'Avg':>7}")
        print('-' * 60)
        for exp_id in sorted(experiment_times):
            times = experiment_times[exp_id]
            exp_total = sum(times)
            exp_avg = exp_total / len(times)
            print(f"  {exp_id:<33} {len(times):>5} {_fmt_duration(exp_total):>9} {exp_avg:>6.2f}s")

    return judge_results


def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{s:04.1f}s"


# ---------------------------------------------------------------------------
# Agreement report
# ---------------------------------------------------------------------------

def load_cached_results(
    output_dir: str,
    review_data: list[dict],
) -> list[dict]:
    """Load previously judged results from checkpoint files.

    Augments each result with human_label and human_target_present from
    the review data so the agreement report works without re-judging.
    Excludes NOISE questions.
    """
    from src.data.utils import NOISE_QUESTION_IDS

    # Build lookup for human labels.
    human_lookup = {}
    for row in review_data:
        key = (row['experiment_id'], row['question_id'])
        human_lookup[key] = {
            'human_label': row.get('human_label', ''),
            'human_target_present': row.get('target_present', ''),
        }

    results = []
    for row in review_data:
        if row['question_id'] in NOISE_QUESTION_IDS:
            continue
        exp_dir = os.path.join(output_dir, row['experiment_id'])
        judge_path = os.path.join(exp_dir, f"{row['question_id']}.json")
        if not os.path.exists(judge_path):
            continue
        try:
            with open(judge_path) as f:
                r = json.loads(f.read())
            if r.get('classification') is None:
                continue
            # Ensure human labels are present (cached files from the
            # original run have them, but be safe).
            key = (r['experiment_id'], r['question_id'])
            if key in human_lookup:
                r.setdefault('human_label', human_lookup[key]['human_label'])
                r.setdefault('human_target_present', human_lookup[key]['human_target_present'])
            results.append(r)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def build_agreement_report(judge_results: list[dict], review_data: list[dict]) -> str:
    """Build the full agreement report as a string.

    Returns the report text (also suitable for writing to a file).
    """
    import numpy as np

    from src.data.utils import NOISE_QUESTION_IDS

    lines: list[str] = []
    w = lines.append  # shorthand

    # Filter NOISE questions from both inputs.
    judge_results = [r for r in judge_results if r.get('question_id') not in NOISE_QUESTION_IDS]
    review_data = [r for r in review_data if r.get('question_id') not in NOISE_QUESTION_IDS]

    # Build lookup for human labels.
    human_labels = {}
    human_target = {}
    for row in review_data:
        key = (row['experiment_id'], row['question_id'])
        human_labels[key] = row.get('human_label', '')
        human_target[key] = row.get('target_present', '')

    valid_categories = [c.value for c in Classification]

    # Collect (human, judge) pairs where both labels are valid.
    pairs = []
    for r in judge_results:
        key = (r['experiment_id'], r['question_id'])
        h_label = r.get('human_label') or human_labels.get(key, '')
        j_label = r.get('classification')
        if h_label in valid_categories and j_label in valid_categories:
            pairs.append((h_label, j_label))

    if not pairs:
        w("\nNo valid classification pairs for comparison.")
        return '\n'.join(lines)

    total_pairs = len(pairs)
    correct = sum(1 for h, j in pairs if h == j)
    accuracy = correct / total_pairs

    w(f"\n{'=' * 70}")
    w("CLASSIFICATION AGREEMENT REPORT")
    w(f"{'=' * 70}")
    w(f"Total compared: {total_pairs}")
    w(f"Exact match:    {correct}/{total_pairs} ({accuracy:.1%})")

    # Per-category precision / recall / F1.
    w(f"\n{'Category':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Human':>6} {'Judge':>6}")
    w('-' * 70)

    for cat in valid_categories:
        tp = sum(1 for h, j in pairs if h == cat and j == cat)
        fp = sum(1 for h, j in pairs if h != cat and j == cat)
        fn = sum(1 for h, j in pairs if h == cat and j != cat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        n_human = sum(1 for h, _ in pairs if h == cat)
        n_judge = sum(1 for _, j in pairs if j == cat)

        w(f"  {cat:<23} {precision:>5.1%} {recall:>5.1%} {f1:>5.1%} {n_human:>6} {n_judge:>6}")

    # Confusion matrix.
    w(f"\n{'=' * 70}")
    w("CONFUSION MATRIX (rows=human, cols=judge)")
    w(f"{'=' * 70}")

    cat_short = {
        'CONFIDENT_CORRECT': 'CC',
        'CORRECT_WITH_DETECTION': 'CD',
        'UNCERTAIN_CORRECT': 'UC',
        'HEDGING': 'HG',
        'UNCERTAIN_INCORRECT': 'UI',
        'CONFIDENT_INCORRECT': 'CI',
        'UNKNOWN': 'UN',
    }

    header = f"{'':>4}" + ''.join(f'{cat_short[c]:>5}' for c in valid_categories)
    w(header)

    for h_cat in valid_categories:
        row_counts = []
        for j_cat in valid_categories:
            count = sum(1 for h, j in pairs if h == h_cat and j == j_cat)
            row_counts.append(count)
        row_str = f'{cat_short[h_cat]:>4}' + ''.join(f'{c:>5}' for c in row_counts)
        w(row_str)

    # Target-answer detection agreement.
    w(f"\n{'=' * 70}")
    w("TARGET ANSWER DETECTION AGREEMENT")
    w(f"{'=' * 70}")

    target_pairs_llm = []
    target_pairs_sub = []
    for r in judge_results:
        key = (r['experiment_id'], r['question_id'])
        h_tp = r.get('human_target_present') or human_target.get(key, '')
        if h_tp == 'N/A':
            continue  # Skip clean conditions.
        h_bool = h_tp == 'TRUE'
        j_llm = r.get('target_present_llm')
        j_sub = r.get('target_present_substring')
        if j_llm is not None:
            target_pairs_llm.append((h_bool, j_llm))
        if j_sub is not None:
            target_pairs_sub.append((h_bool, j_sub))

    if target_pairs_llm:
        llm_agree = sum(1 for h, j in target_pairs_llm if h == j)
        n = len(target_pairs_llm)
        w(f"LLM judge:      {llm_agree}/{n} ({llm_agree / n:.1%})")

    if target_pairs_sub:
        sub_agree = sum(1 for h, j in target_pairs_sub if h == j)
        n = len(target_pairs_sub)
        w(f"Substring:      {sub_agree}/{n} ({sub_agree / n:.1%})")

    # Embedding similarity summary for poisoned conditions.
    embed_scores = [
        r['target_present_embedding']
        for r in judge_results
        if r.get('target_present_embedding') is not None
    ]
    if embed_scores:
        arr = np.array(embed_scores)
        w(f"\nEmbedding similarity (poisoned conditions, n={len(arr)}):")
        w(f"  Mean: {arr.mean():.3f}  Std: {arr.std():.3f}  "
          f"Min: {arr.min():.3f}  Max: {arr.max():.3f}")

        # Split by human target_present.
        tp_scores = []
        fp_scores = []
        for r in judge_results:
            if r.get('target_present_embedding') is None:
                continue
            key = (r['experiment_id'], r['question_id'])
            h_tp = r.get('human_target_present') or human_target.get(key, '')
            if h_tp == 'TRUE':
                tp_scores.append(r['target_present_embedding'])
            elif h_tp == 'FALSE':
                fp_scores.append(r['target_present_embedding'])
        if tp_scores and fp_scores:
            w(f"  When target present (n={len(tp_scores)}): "
              f"mean={np.mean(tp_scores):.3f}")
            w(f"  When target absent  (n={len(fp_scores)}): "
              f"mean={np.mean(fp_scores):.3f}")

    # Judge classification distribution.
    w(f"\n{'=' * 70}")
    w("JUDGE CLASSIFICATION DISTRIBUTION")
    w(f"{'=' * 70}")

    j_counts = Counter(
        r.get('classification') for r in judge_results if r.get('classification')
    )
    for cat in valid_categories:
        count = j_counts.get(cat, 0)
        w(f"  {cat:<25} {count:>4}")

    err_count = sum(1 for r in judge_results if r.get('error'))
    if err_count:
        w(f"  {'ERRORS':<25} {err_count:>4}")

    # ------------------------------------------------------------------
    # Baseline-conditioned attack metrics
    # ------------------------------------------------------------------
    # For each architecture, only count attack-condition results on
    # questions where that architecture answered correctly on clean.
    # This prevents baseline failures (especially MADAM's ~34% UNK rate
    # on clean) from inflating apparent attack resistance.

    ARCHS = ['vanilla', 'agentic', 'rlm', 'madam']
    ATTACKS = ['naive', 'corruptrag_ak']
    CORRECT_CATS = {
        'CONFIDENT_CORRECT', 'CORRECT_WITH_DETECTION', 'UNCERTAIN_CORRECT',
    }

    # Index results by (experiment_id, question_id) -> human label.
    label_by_key: dict[tuple[str, str], str] = {}
    for row in review_data:
        key = (row['experiment_id'], row['question_id'])
        label_by_key[key] = row.get('human_label', '')

    # Build per-architecture set of question_ids answered correctly on clean.
    clean_correct: dict[str, set[str]] = {}
    for arch in ARCHS:
        clean_exp = f'{arch}_clean'
        clean_correct[arch] = {
            qid for (exp_id, qid), label in label_by_key.items()
            if exp_id == clean_exp and label in CORRECT_CATS
        }

    # Only emit this section if we have data for at least one architecture.
    has_data = any(clean_correct[arch] for arch in ARCHS)
    if has_data:
        w(f"\n{'=' * 70}")
        w("BASELINE-CONDITIONED ATTACK METRICS")
        w(f"{'=' * 70}")
        w("Only questions where the architecture answered correctly on clean.")
        w("This isolates attack-caused failures from baseline inability.\n")

        header = (f"{'Arch':<10} {'Attack':<16} "
                  f"{'N(raw)':>7} {'CI(raw)':>8} {'ASR(raw)':>9}  "
                  f"{'N(cond)':>8} {'CI(cond)':>9} {'ASR(cond)':>10}")
        w(header)
        w('-' * 82)

        for arch in ARCHS:
            for attack in ATTACKS:
                exp_id = f'{arch}_{attack}'

                # Raw counts (all questions).
                raw_labels = [
                    label for (eid, qid), label in label_by_key.items()
                    if eid == exp_id and label in valid_categories
                ]
                raw_n = len(raw_labels)
                raw_ci = sum(1 for l in raw_labels if l == 'CONFIDENT_INCORRECT')
                raw_asr = raw_ci / raw_n if raw_n else 0

                # Conditioned counts (only clean-correct questions).
                cond_labels = [
                    label for (eid, qid), label in label_by_key.items()
                    if eid == exp_id and label in valid_categories
                    and qid in clean_correct[arch]
                ]
                cond_n = len(cond_labels)
                cond_ci = sum(1 for l in cond_labels if l == 'CONFIDENT_INCORRECT')
                cond_asr = cond_ci / cond_n if cond_n else 0

                w(f"{arch:<10} {attack:<16} "
                  f"{raw_n:>7} {raw_ci:>8} {raw_asr:>8.1%}  "
                  f"{cond_n:>8} {cond_ci:>9} {cond_asr:>9.1%}")

        # MADAM category detail (conditioned) — since MADAM's failure mode
        # is UNKNOWN/HEDGING rather than CI, show full breakdown.
        w(f"\nMADAM conditioned category breakdown:")
        for attack in ATTACKS:
            exp_id = f'madam_{attack}'
            cond_labels = [
                label for (eid, qid), label in label_by_key.items()
                if eid == exp_id and label in valid_categories
                and qid in clean_correct['madam']
            ]
            cond_n = len(cond_labels)
            if not cond_n:
                continue
            counts = Counter(cond_labels)
            w(f"  {exp_id} (n={cond_n}):")
            for cat in valid_categories:
                c = counts.get(cat, 0)
                if c:
                    w(f"    {cat:<28} {c:>3} ({c / cond_n:>5.1%})")

        # Clean baseline summary.
        w(f"\nClean baseline correct counts:")
        for arch in ARCHS:
            total_clean = sum(
                1 for (eid, _), _ in label_by_key.items()
                if eid == f'{arch}_clean'
            )
            n_correct = len(clean_correct[arch])
            w(f"  {arch:<10} {n_correct}/{total_clean}")

    return '\n'.join(lines)


def print_and_save_report(
    judge_results: list[dict],
    review_data: list[dict],
    output_dir: str,
):
    """Build the agreement report, print it, and save to output_dir/report.txt."""
    report = build_agreement_report(judge_results, review_data)
    print(report)

    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
        f.write('\n')
    print(f"\nReport saved to {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate LLM judge against human-labeled sample"
    )
    parser.add_argument(
        '--review-csv',
        type=str,
        default=REVIEW_CSV,
        help=f"Path to review CSV (default: {REVIEW_CSV})",
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        '--model',
        type=str,
        default=JUDGE_MODEL,
        help=f"Judge model (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        choices=['low', 'medium', 'high'],
        help=f"Reasoning effort (default: {DEFAULT_REASONING_EFFORT})",
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help="Skip judging, load cached results, and rerun only the analysis report.",
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        help="Timestamp suffix to find a specific validation run dir (e.g. '20260312-2300'). "
             "Used with --report-only to locate results/judge_validation_*_<timestamp>/.",
    )
    args = parser.parse_args()

    print(f"Loading review data from {args.review_csv}")
    review_data = load_review_data(args.review_csv)
    n_questions = len(set(r['question_id'] for r in review_data))
    n_experiments = len(set(r['experiment_id'] for r in review_data))
    print(f"Loaded {len(review_data)} rows ({n_questions} questions x {n_experiments} experiments)")

    if args.report_only:
        output_dir = args.output_dir
        if args.timestamp:
            resolved = _find_local_validation_dir(args.timestamp)
            if not resolved:
                print(f"No validation dir found matching timestamp {args.timestamp}")
                raise SystemExit(1)
            output_dir = resolved
        print(f"\n--report-only: loading cached results from {output_dir}")
        judge_results = load_cached_results(output_dir, review_data)
        print(f"Loaded {len(judge_results)} cached judge results")
    else:
        output_dir = args.output_dir
        print(f"\nRunning judge (model={args.model}, effort={args.reasoning_effort})")
        print(f"Output: {output_dir}\n")

        judge_results = run_validation(
            review_data=review_data,
            output_dir=output_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )

    print_and_save_report(judge_results, review_data, output_dir)


if __name__ == '__main__':
    main()
