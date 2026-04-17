"""Tests for src.experiments.llm_judge and src.experiments.noise_filter.

* :class:`JudgeHelpersUnitTests` — pure-Python text-matching helpers
  (``_normalize_text``, ``check_target_substring``). No data, no API.
* :class:`JudgePromptUnitTests` — verify the markdown prompt file loads
  and has the expected placeholders. The prompt is committed to the
  repo, so no downloaded data is required.
* :class:`JudgeLocalReportIntegrationTests` — load the human-labels
  review CSV and confirm shape (downloaded data, no API).
* :class:`NoiseFilterIntegrationTests` — load the noise-filter results
  and confirm the exclusion set populates (downloaded data, no API).
* :class:`JudgeLocalSmallBatchIntegrationTests` — run the judge against
  3 samples (live OpenAI calls).
"""

import json
import os
import tempfile
import unittest

import pytest


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)


def _data_present_or_skip(sentinel: str) -> None:
    if not os.path.exists(sentinel):
        raise unittest.SkipTest(
            f"Integration test requires {sentinel}. "
            f"Either run scripts/download_data.sh to fetch the published dataset, "
            f"or regenerate the data by running the experiment pipeline."
        )


# ===========================================================================
# Unit suite
# ===========================================================================

class JudgeHelpersUnitTests(unittest.TestCase):
    """Pure-Python normalization + substring matching from llm_judge."""

    def test_normalize_text_lowercases(self):
        from src.experiments.llm_judge import _normalize_text
        # 'apple' is a stop-word-free content word; stemming maps it to 'appl'.
        self.assertEqual(_normalize_text('APPLE'), _normalize_text('apple'))

    def test_normalize_text_strips_punctuation(self):
        from src.experiments.llm_judge import _normalize_text
        self.assertEqual(_normalize_text('hello, world!'), _normalize_text('hello world'))

    def test_normalize_text_drops_stop_words(self):
        from src.experiments.llm_judge import _normalize_text
        # 'the' is in STOP_WORDS; both should normalize to the same content.
        self.assertEqual(_normalize_text('the cat'), _normalize_text('cat'))

    def test_normalize_text_stems(self):
        from src.experiments.llm_judge import _normalize_text
        # PorterStemmer maps 'running' / 'runs' / 'run' to a common stem.
        a = _normalize_text('running')
        b = _normalize_text('runs')
        c = _normalize_text('run')
        self.assertEqual(a, b)
        self.assertEqual(b, c)

    def test_check_target_substring_exact_match(self):
        from src.experiments.llm_judge import check_target_substring
        self.assertTrue(check_target_substring('John Smith', 'John Smith'))

    def test_check_target_substring_near_match_with_stopwords(self):
        from src.experiments.llm_judge import check_target_substring
        # 'the' is dropped, 'capital' stems consistently.
        self.assertTrue(
            check_target_substring(
                target_answer='Paris',
                system_answer='The capital of France is Paris.',
            )
        )

    def test_check_target_substring_word_reorder(self):
        from src.experiments.llm_judge import check_target_substring
        # Both words present, just out of order — second check should catch it.
        self.assertTrue(
            check_target_substring(
                target_answer='John Smith',
                system_answer='Smith, John was the founder.',
            )
        )

    def test_check_target_substring_clean_non_match(self):
        from src.experiments.llm_judge import check_target_substring
        self.assertFalse(
            check_target_substring(
                target_answer='Paris',
                system_answer='The answer is London.',
            )
        )

    def test_check_target_substring_empty_target_returns_false(self):
        from src.experiments.llm_judge import check_target_substring
        self.assertFalse(check_target_substring('', 'anything'))
        self.assertFalse(check_target_substring('none', 'anything'))
        self.assertFalse(check_target_substring('NONE', 'anything'))


class JudgePromptUnitTests(unittest.TestCase):
    """The judge prompt markdown loads and parses correctly.

    `llm-judge-prompt.md` lives in the repo at src/experiments/, so this
    runs in the unit suite — no downloaded data, no API calls.
    """

    def test_load_judge_prompt(self):
        from src.experiments.llm_judge import load_judge_prompt

        system_prompt, user_template = load_judge_prompt()
        self.assertGreater(len(system_prompt), 100)
        self.assertIn('{question}', user_template)
        self.assertIn('{system_answer}', user_template)
        self.assertIn('{correct_answer}', user_template)


# ===========================================================================
# Integration suite — one class per scenario
# ===========================================================================

@pytest.mark.integration
class JudgeLocalReportIntegrationTests(unittest.TestCase):
    """Load the human-labels review CSV (no API)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from src.experiments.run_judge_local import REVIEW_CSV
        _data_present_or_skip(REVIEW_CSV)
        cls.review_csv = REVIEW_CSV

    def test_review_data_loads_with_expected_fields(self):
        from src.experiments.run_judge_local import load_review_data, _RESULTS_DIR

        review_data = load_review_data(self.review_csv)
        self.assertGreater(len(review_data), 0)

        required_fields = {
            'experiment_id', 'question_id', 'question_text',
            'system_answer', 'correct_answer',
        }
        first = review_data[0]
        missing = required_fields - set(first.keys())
        self.assertFalse(missing, f"Missing fields in review CSV: {missing}")

        # Sanity-check that the results dir is at least addressable.
        self.assertTrue(os.path.isdir(_RESULTS_DIR))


@pytest.mark.integration
class NoiseFilterIntegrationTests(unittest.TestCase):
    """Load the noise-filter results and confirm exclusion set populates."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from src.experiments.noise_filter import QUESTIONS_PATH, NOISE_OUTPUT_DIR
        _data_present_or_skip(QUESTIONS_PATH)
        if not os.path.isdir(NOISE_OUTPUT_DIR):
            raise unittest.SkipTest(f"Missing noise output dir: {NOISE_OUTPUT_DIR}")
        cls.questions_path = QUESTIONS_PATH
        cls.noise_output_dir = NOISE_OUTPUT_DIR

    def test_questions_file_loads_at_expected_count(self):
        with open(self.questions_path) as f:
            questions = [json.loads(line) for line in f]
        self.assertEqual(len(questions), 1150)

    def test_noise_exclusions_populate(self):
        from src.experiments.noise_filter import load_noise_exclusions
        exclusions = load_noise_exclusions()
        self.assertGreater(len(exclusions), 0)

    def test_noise_result_files_have_expected_shape(self):
        noise_files = [f for f in os.listdir(self.noise_output_dir) if f.endswith('.json')]
        self.assertGreater(len(noise_files), 100)

        sample_path = os.path.join(self.noise_output_dir, noise_files[0])
        with open(sample_path) as f:
            sample = json.load(f)
        self.assertIn('question_id', sample)
        self.assertIn('is_noise', sample)


@pytest.mark.integration
class JudgeLocalSmallBatchIntegrationTests(unittest.TestCase):
    """Run the judge against 3 samples (live OpenAI calls)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from src.experiments.run_judge_local import REVIEW_CSV
        _data_present_or_skip(REVIEW_CSV)
        cls.review_csv = REVIEW_CSV

    def test_three_sample_batch_classifies_each(self):
        from src.experiments.run_judge_local import load_review_data, run_validation

        review_data = load_review_data(self.review_csv)
        small_batch = review_data[:3]

        with tempfile.TemporaryDirectory(prefix='rag_judge_test_') as tmp_dir:
            results = run_validation(
                review_data=small_batch,
                output_dir=tmp_dir,
            )

            self.assertEqual(len(results), 3)
            for r in results:
                self.assertIn('classification', r)
                self.assertIsNotNone(r['classification'])

            for r in results:
                exp_dir = os.path.join(tmp_dir, r['experiment_id'])
                json_path = os.path.join(exp_dir, f"{r['question_id']}.json")
                self.assertTrue(os.path.exists(json_path))
