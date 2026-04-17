"""Unit tests for the Phase 5 bash orchestration scripts in ``scripts/``.

These scripts are thin glue — their job is to sequence the Python
entrypoints with the right args. We can't economically exercise the real
pipeline (Modal spend + wall time), so the tests cover the three failure
modes that a glue script can plausibly have:

1. Usage / argument parsing broken        — ``--help`` exits 0,
                                              unknown flags exit 2.
2. Step sequence reordered or mis-quoted   — ``--dry-run`` output is
                                              pinned to a golden file.
3. Prerequisite checks silently skipped    — running against a fake
                                              ``REPO_ROOT`` that's missing
                                              the required inputs must exit
                                              1 with a helpful message.

Every script also gets ``shellcheck``'d and ``bash -n``'d separately at
the CI / pre-commit layer; this file is the behavioral check.
"""

import os
import shutil
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / 'scripts'
GOLDEN_DIR = Path(__file__).resolve().parent / 'golden' / 'scripts'

ALL_SCRIPTS = [
    'setup_environment.sh',
    'download_data.sh',
    'prepare_data.sh',
    'prepare_embeddings.sh',
    'run_experiments.sh',
    'run_analysis.sh',
    'generate_paper.sh',
    'run_all.sh',
]

DRY_RUN_SCRIPTS = [
    'prepare_data.sh',
    'prepare_embeddings.sh',
    'run_experiments.sh',
    'run_all.sh',
]


def _run(script: str, *args: str, env_overrides: dict | None = None,
         cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke a script with a clean env + optional REPO_ROOT override."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [str(SCRIPTS_DIR / script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


class ScriptHelpUnitTests(unittest.TestCase):
    """Every script's ``--help`` exits 0, prints a Usage block, and does
    not bleed shell code into the output (guards against the
    hardcoded-line-number bug that used to live in ``show_help``)."""

    def test_help_exits_zero_and_shows_usage(self):
        for script in ALL_SCRIPTS:
            with self.subTest(script=script):
                result = _run(script, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Usage:', result.stdout)
                # Catches header-range overshoot regressions.
                self.assertNotIn('set -euo pipefail', result.stdout)
                self.assertNotIn('show_help()', result.stdout)

    def test_help_short_flag(self):
        for script in ALL_SCRIPTS:
            with self.subTest(script=script):
                result = _run(script, '-h')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Usage:', result.stdout)


class ScriptUnknownArgUnitTests(unittest.TestCase):
    """Unknown flags exit 2 with a clear message on stderr."""

    def test_unknown_flag_rejected(self):
        for script in ALL_SCRIPTS:
            with self.subTest(script=script):
                result = _run(script, '--not-a-real-flag')
                self.assertEqual(
                    result.returncode, 2,
                    f"{script}: expected exit 2, got {result.returncode}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}",
                )
                self.assertIn('unknown arg', result.stderr)


class ScriptDryRunGoldenUnitTests(unittest.TestCase):
    """``--dry-run`` output is byte-identical to the pinned golden file.

    Absolute REPO_ROOT paths are normalized to ``${REPO_ROOT}`` before
    comparison so the goldens are portable.
    """

    def test_dry_run_matches_golden(self):
        for script in DRY_RUN_SCRIPTS:
            with self.subTest(script=script):
                result = _run(script, '--dry-run')
                self.assertEqual(
                    result.returncode, 0,
                    f"{script}: exit {result.returncode}\n"
                    f"stderr: {result.stderr}",
                )
                actual = result.stdout.replace(str(REPO_ROOT), '${REPO_ROOT}')
                golden_path = GOLDEN_DIR / f'{script}.dry-run.txt'
                golden = golden_path.read_text()
                self.assertEqual(
                    actual, golden,
                    f"{script}: dry-run output diverges from {golden_path.name}.\n"
                    f"To regenerate (after reviewing):\n"
                    f"  scripts/{script} --dry-run > tests/golden/scripts/{script}.dry-run.txt"
                )


class ScriptPrereqFailureUnitTests(unittest.TestCase):
    """Scripts exit 1 with a ``Run: scripts/<upstream>`` pointer when
    their inputs are missing.

    We fake a repo by creating a minimal tempdir with just ``venv/bin/``
    populated (via symlinks to the real venv so Modal/Python binaries
    resolve) and no data, then point ``REPO_ROOT`` at it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._fake_root = REPO_ROOT / '.test-scripts-fake-repo'
        # Clean up any leftover from an earlier failed run.
        if cls._fake_root.exists():
            shutil.rmtree(cls._fake_root)
        (cls._fake_root / 'venv' / 'bin').mkdir(parents=True)
        for binname in ['python', 'modal', 'jupyter', 'pip']:
            real = REPO_ROOT / 'venv' / 'bin' / binname
            if real.exists():
                (cls._fake_root / 'venv' / 'bin' / binname).symlink_to(real)
        # scripts/ contains empty marker dirs some scripts reference; the
        # scripts themselves still live at SCRIPTS_DIR since we invoke
        # them by absolute path — only their REPO_ROOT-derived lookups
        # are faked.

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fake_root.exists():
            shutil.rmtree(cls._fake_root)

    def _run_against_fake_root(self, script: str, *args: str):
        return _run(
            script, *args,
            env_overrides={'REPO_ROOT': str(self._fake_root)},
        )

    def test_prepare_embeddings_errors_on_missing_corpus(self):
        result = self._run_against_fake_root('prepare_embeddings.sh')
        self.assertEqual(result.returncode, 1)
        self.assertIn('prerequisite corpus missing', result.stderr)
        self.assertIn('scripts/prepare_data.sh', result.stderr)

    def test_run_analysis_errors_on_missing_results(self):
        result = self._run_against_fake_root('run_analysis.sh')
        self.assertEqual(result.returncode, 1)
        self.assertIn('prerequisite missing', result.stderr)
        self.assertIn('scripts/download_data.sh', result.stderr)

    def test_generate_paper_errors_on_missing_paper_dir(self):
        result = self._run_against_fake_root('generate_paper.sh')
        self.assertEqual(result.returncode, 1)
        self.assertIn('paper directory not found', result.stderr)


class ScriptShellcheckUnitTests(unittest.TestCase):
    """Optional static analysis. Skipped when shellcheck isn't installed."""

    def test_shellcheck_clean(self):
        shellcheck = shutil.which('shellcheck')
        if shellcheck is None:
            self.skipTest('shellcheck not installed (brew install shellcheck)')
        result = subprocess.run(
            [shellcheck, *[str(SCRIPTS_DIR / s) for s in ALL_SCRIPTS]],
            capture_output=True, text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"shellcheck found issues:\n{result.stdout}"
        )


class ScriptDownloadDataUnitTests(unittest.TestCase):
    """``download_data.sh`` refuses to clobber populated data."""

    def test_refuses_clobber_when_data_populated(self):
        # The live repo has the results symlinks pointing at workspace
        # data, so this test only makes sense when those are populated.
        experiments = REPO_ROOT / 'src' / 'experiments' / 'results' / 'experiments'
        if not experiments.exists() or not any(
            p.name != '.gitkeep' for p in experiments.iterdir()
        ):
            self.skipTest(
                "Live repo has no populated results; nothing to clobber. "
                "Run scripts/setup_test_symlinks.sh to populate."
            )
        result = _run('download_data.sh')
        self.assertEqual(result.returncode, 1)
        self.assertIn('already populated', result.stderr)
        self.assertIn('--force', result.stderr)


if __name__ == '__main__':
    unittest.main()
