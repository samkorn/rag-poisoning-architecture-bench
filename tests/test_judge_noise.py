"""
Phase 4 migration test: verify judge and noise filter scripts work.

Tests progress from no-API-cost verification to small live batches.

Run from repo root:
    python tests/test_judge_noise.py
    python tests/test_judge_noise.py --skip-api   # path verification only
"""

import argparse
import json
import os
import sys
import tempfile

os.environ.setdefault('LOGFIRE_SEND_TO_LOGFIRE', 'false')


# ---------------------------------------------------------------------------
# Test: judge prompt loads correctly
# ---------------------------------------------------------------------------

def test_judge_prompt_loads():
    """Verify the judge prompt file loads and has expected structure."""
    print("\n=== test_judge_prompt_loads ===")

    from src.experiments.llm_judge import load_judge_prompt

    system_prompt, user_template = load_judge_prompt()
    assert len(system_prompt) > 100, f"System prompt too short: {len(system_prompt)}"
    assert '{question}' in user_template, "Missing {question} placeholder"
    assert '{system_answer}' in user_template, "Missing {system_answer} placeholder"
    assert '{correct_answer}' in user_template, "Missing {correct_answer} placeholder"
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  User template placeholders: question, correct_answer, target_answer, system_answer")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: judge local report-only mode (no API calls)
# ---------------------------------------------------------------------------

def test_judge_local_report_only():
    """Run run_judge_local.py in --report-only mode (reads cached results, no API)."""
    print("\n=== test_judge_local_report_only ===")

    from src.experiments.run_judge_local import (
        REVIEW_CSV,
        load_review_data,
        _RESULTS_DIR,
    )

    # Verify review CSV loads
    review_data = load_review_data(REVIEW_CSV)
    assert len(review_data) > 0, "No review data loaded"
    print(f"  Loaded {len(review_data)} review samples from human_labels.csv")

    # Check expected fields in review data
    required_fields = {'experiment_id', 'question_id', 'question_text', 'system_answer', 'correct_answer'}
    first = review_data[0]
    missing = required_fields - set(first.keys())
    assert not missing, f"Missing fields in review CSV: {missing}"
    print(f"  Fields: {sorted(first.keys())}")

    # Verify we can find existing judge-validation results
    # (they may or may not exist depending on workspace state)
    validation_dirs = [
        d for d in os.listdir(_RESULTS_DIR)
        if d.startswith('judge_validation') or d.startswith('judge-validation')
    ] if os.path.isdir(_RESULTS_DIR) else []
    print(f"  Found {len(validation_dirs)} existing validation dirs")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: noise filter loads questions and existing results
# ---------------------------------------------------------------------------

def test_noise_filter_loads():
    """Verify noise filter can load its input data."""
    print("\n=== test_noise_filter_loads ===")

    from src.experiments.noise_filter import (
        QUESTIONS_PATH,
        NOISE_OUTPUT_DIR,
        load_noise_exclusions,
    )

    # Verify questions file loads
    assert os.path.exists(QUESTIONS_PATH), f"Missing: {QUESTIONS_PATH}"
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    print(f"  Loaded {len(questions)} questions")
    assert len(questions) == 1150, f"Expected 1150, got {len(questions)}"

    # Verify existing noise results load
    exclusions = load_noise_exclusions()
    print(f"  Noise exclusions: {len(exclusions)} questions")
    assert len(exclusions) > 0, "Expected some noise exclusions"

    # Spot-check a noise result file
    noise_files = [f for f in os.listdir(NOISE_OUTPUT_DIR) if f.endswith('.json')]
    sample_path = os.path.join(NOISE_OUTPUT_DIR, noise_files[0])
    with open(sample_path) as f:
        sample = json.load(f)
    required = {'question_id', 'is_noise'}
    missing = required - set(sample.keys())
    assert not missing, f"Missing fields in noise result: {missing}"
    print(f"  Sample noise result: {noise_files[0]} -> is_noise={sample.get('is_noise')}")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: judge local small batch (API calls)
# ---------------------------------------------------------------------------

def test_judge_local_small_batch():
    """Run judge on 3 review samples, output to temp dir."""
    print("\n=== test_judge_local_small_batch ===")

    from src.experiments.run_judge_local import load_review_data, run_validation, REVIEW_CSV

    review_data = load_review_data(REVIEW_CSV)

    # Take first 3 samples
    small_batch = review_data[:3]
    print(f"  Judging {len(small_batch)} samples...")

    with tempfile.TemporaryDirectory(prefix='rag_judge_test_') as tmp_dir:
        results = run_validation(
            review_data=small_batch,
            output_dir=tmp_dir,
        )

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        for r in results:
            assert 'classification' in r, f"Missing classification in result"
            assert r['classification'] is not None, "Classification is None"
            print(f"  {r.get('experiment_id')}/{r.get('question_id')}: {r['classification']}")

        # Verify checkpoint files were written
        for r in results:
            exp_dir = os.path.join(tmp_dir, r['experiment_id'])
            json_path = os.path.join(exp_dir, f"{r['question_id']}.json")
            assert os.path.exists(json_path), f"Missing checkpoint: {json_path}"

    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-api', action='store_true', help='Skip tests that make API calls')
    args = parser.parse_args()

    # No-API tests
    test_judge_prompt_loads()
    test_judge_local_report_only()
    test_noise_filter_loads()

    if args.skip_api:
        print("\n=== SKIPPED API TESTS (--skip-api) ===")
    else:
        # API tests
        test_judge_local_small_batch()

    print("\n=== ALL TESTS PASSED ===")
