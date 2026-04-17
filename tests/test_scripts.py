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

The file is fully self-contained: tests that need a fake repo build it
in a ``tempfile.TemporaryDirectory`` and never touch the live tree, so
they don't depend on whether integration-test data is staged.

Every script also gets ``shellcheck``'d and ``bash -n``'d separately at
the CI / pre-commit layer; this file is the behavioral check.
"""

import os
import shutil
import subprocess
import tempfile
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

# Scripts check `${REPO_ROOT}/venv/bin/<tool>` with `[[ -x ... ]]`. In the
# fake-repo tests we don't run the tools, we just need the existence
# check to pass so the script proceeds to the next check we want to
# exercise. A trivial stub shell script satisfies that.
_VENV_TOOLS = ['python', 'modal', 'jupyter', 'pip']
_STUB_BODY = '#!/bin/sh\nexit 0\n'


def _build_fake_venv(root: Path) -> None:
    """Populate ``root/venv/bin/`` with stub executables."""
    bin_dir = root / 'venv' / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    for tool in _VENV_TOOLS:
        stub = bin_dir / tool
        stub.write_text(_STUB_BODY)
        stub.chmod(0o755)


# Paths that prepare_embeddings.sh checks before running — needed for
# dry-run to pass the prereq gate.
_PREPARE_EMBEDDINGS_PREREQS = [
    'src/data/original-datasets/nq/corpus.jsonl',
    'src/data/experiment-datasets/nq-naive-poisoning/corpus.jsonl',
    'src/data/experiment-datasets/nq-corruptrag-ak-poisoning/corpus.jsonl',
]


def _build_fake_prereqs(root: Path) -> None:
    """Create empty files at every path checked by a dry-run prereq gate.

    Dry-run runs prereq checks (by design — so ``--dry-run`` still
    surfaces "you forgot X"), which means the test fake repo must satisfy
    them. Files are zero-byte; the scripts only `[[ -f ]]` them.
    """
    for rel in _PREPARE_EMBEDDINGS_PREREQS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


# ``run_all.sh`` chains downstream scripts via ``${REPO_ROOT}/scripts/<x>``.
# In the golden test we only care that the right scripts are invoked with
# the right args — not that their dry-run contents reappear — so we stub
# them in the fake repo's scripts/ dir. Each real script already has its
# own dedicated golden covering its content.
_RUN_ALL_STUB_SCRIPTS = [
    'prepare_data.sh',
    'prepare_embeddings.sh',
    'run_experiments.sh',
]
_STUB_ECHO_BODY = (
    '#!/bin/sh\n'
    '# Test stub — see tests/test_scripts.py\n'
    'echo "[stub $(basename "$0")] args: $*"\n'
)


def _build_fake_downstream_scripts(root: Path) -> None:
    """Populate ``root/scripts/`` with stub versions of the scripts that
    ``run_all.sh`` invokes. Each stub echoes its args and exits 0."""
    scripts_dir = root / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for name in _RUN_ALL_STUB_SCRIPTS:
        stub = scripts_dir / name
        stub.write_text(_STUB_ECHO_BODY)
        stub.chmod(0o755)


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

    Runs against a fake tempdir REPO_ROOT with stub venv binaries and
    empty files at every path the dry-run prereq gates check, so the
    test doesn't depend on integration-test data being staged. Absolute
    REPO_ROOT paths in the output are normalized to ``${REPO_ROOT}``
    before comparison.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._fake_root = Path(self._tmpdir.name)
        _build_fake_venv(self._fake_root)
        _build_fake_prereqs(self._fake_root)
        _build_fake_downstream_scripts(self._fake_root)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_dry_run_matches_golden(self):
        for script in DRY_RUN_SCRIPTS:
            with self.subTest(script=script):
                result = _run(
                    script, '--dry-run',
                    env_overrides={'REPO_ROOT': str(self._fake_root)},
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"{script}: exit {result.returncode}\n"
                    f"stderr: {result.stderr}",
                )
                actual = result.stdout.replace(
                    str(self._fake_root), '${REPO_ROOT}',
                )
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

    A fake repo is built in a tempdir with just stub ``venv/bin/``
    executables and no data, then ``REPO_ROOT`` env var redirects the
    scripts to it.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._fake_root = Path(self._tmpdir.name)
        _build_fake_venv(self._fake_root)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

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
    """``download_data.sh`` refuses to clobber populated destinations.

    Builds a fake repo whose ``src/experiments/results/experiments/``
    already contains a stray file, then runs the script. The populated-dir
    check runs before any network call, so the script must exit 1 with a
    pointer to ``--force`` — no Zenodo round-trip happens.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._fake_root = Path(self._tmpdir.name)
        _build_fake_venv(self._fake_root)

        # Pre-populate EXPERIMENTS_DEST with a non-.gitkeep file so
        # `is_populated_dir` returns true. The other destinations stay
        # empty — the script aborts on the first populated one.
        populated = self._fake_root / 'src' / 'experiments' / 'results' / 'experiments'
        populated.mkdir(parents=True)
        (populated / 'dummy.json').write_text('{}')

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_refuses_clobber_when_data_populated(self):
        result = _run(
            'download_data.sh',
            env_overrides={'REPO_ROOT': str(self._fake_root)},
        )
        self.assertEqual(
            result.returncode, 1,
            f"expected exit 1, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertIn('already populated', result.stderr)
        self.assertIn('--force', result.stderr)
        # Regression guard: the pre-flight must reject BEFORE any Zenodo
        # call — so stdout must not contain the download banner.
        self.assertNotIn('Downloading', result.stdout)


if __name__ == '__main__':
    unittest.main()
