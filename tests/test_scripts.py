"""Unit tests for the Phase 5 bash orchestration scripts in `scripts/`.

These scripts are thin glue — their job is to sequence the Python
entrypoints with the right args. We can't economically exercise the
real pipeline (Modal spend + wall time), so the tests cover the
three failure modes that a glue script can plausibly have:

1. Usage / argument parsing broken        — `--help` exits 0,
                                              unknown flags exit 2.
2. Step sequence reordered or mis-quoted   — `--dry-run` output is
                                              pinned to a golden file.
3. Prerequisite checks silently skipped    — running against a fake
                                              `REPO_ROOT` that's missing
                                              the required inputs must exit
                                              1 with a helpful message.

The file is fully self-contained: tests that need a fake repo build
it in a `tempfile.TemporaryDirectory` and never touch the live tree,
so they don't depend on whether integration-test data is staged.

Every script also gets `shellcheck`'d and `bash -n`'d separately at
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
    """Every script's `--help` exits 0, prints a Usage block, and stays clean.

    Guards against the hardcoded-line-number bug that used to live
    in `show_help` and would bleed shell code into the help output.
    """

    def test_help_exits_zero_and_shows_usage(self):
        """`--help` exits 0, prints `Usage:`, no shell-source leakage."""
        for script in ALL_SCRIPTS:
            with self.subTest(script=script):
                result = _run(script, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Usage:', result.stdout)
                # Catches header-range overshoot regressions.
                self.assertNotIn('set -euo pipefail', result.stdout)
                self.assertNotIn('show_help()', result.stdout)

    def test_help_short_flag(self):
        """`-h` exits 0 and prints a `Usage:` block (mirrors `--help`)."""
        for script in ALL_SCRIPTS:
            with self.subTest(script=script):
                result = _run(script, '-h')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Usage:', result.stdout)


class ScriptUnknownArgUnitTests(unittest.TestCase):
    """Unknown flags exit 2 with a clear message on stderr."""

    def test_unknown_flag_rejected(self):
        """Unknown flags exit 2 and emit `unknown arg` on stderr."""
        for script in ALL_SCRIPTS:
            with self.subTest(script=script):
                result = _run(script, '--not-a-real-flag')
                self.assertEqual(
                    result.returncode, 2,
                    f"{script}: expected exit 2, got {result.returncode}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}",
                )
                self.assertIn('unknown arg', result.stderr)


class ScriptRunAllMutexUnitTests(unittest.TestCase):
    """``run_all.sh`` rejects combined flags that don't make sense together."""

    def test_resume_and_analysis_only_mutex(self):
        """`--resume` + `--analysis-only` exit 2 with a mutex error."""
        result = _run('run_all.sh', '--resume', '--analysis-only')
        self.assertEqual(result.returncode, 2)
        self.assertIn('mutually exclusive', result.stderr)


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

    def _assert_golden(self, script: str, *args: str, golden_name: str) -> None:
        """Run a script and assert its stdout matches a pinned golden file."""
        result = _run(
            script, *args,
            env_overrides={'REPO_ROOT': str(self._fake_root)},
        )
        self.assertEqual(
            result.returncode, 0,
            f"{script} {' '.join(args)}: exit {result.returncode}\n"
            f"stderr: {result.stderr}",
        )
        actual = result.stdout.replace(str(self._fake_root), '${REPO_ROOT}')
        golden_path = GOLDEN_DIR / golden_name
        self.assertEqual(
            actual, golden_path.read_text(),
            f"{script} {' '.join(args)}: output diverges from {golden_path.name}.\n"
            f"To regenerate (after reviewing):\n"
            f"  scripts/{script} {' '.join(args)} > tests/golden/scripts/{golden_name}"
        )

    def test_dry_run_matches_golden(self):
        """Each `--dry-run` script matches its golden file byte-for-byte."""
        for script in DRY_RUN_SCRIPTS:
            with self.subTest(script=script):
                self._assert_golden(
                    script, '--dry-run',
                    golden_name=f'{script}.dry-run.txt',
                )

    def test_analysis_only_dry_run_matches_golden(self):
        """``run_all.sh --analysis-only --dry-run`` cascades through the
        4 analysis-path scripts (env → download → analysis → paper)
        without touching Modal."""
        self._assert_golden(
            'run_all.sh', '--analysis-only', '--dry-run',
            golden_name='run_all.sh.analysis-only.dry-run.txt',
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
        """Run a script with `REPO_ROOT` pointed at the fake tempdir."""
        return _run(
            script, *args,
            env_overrides={'REPO_ROOT': str(self._fake_root)},
        )

    def test_prepare_embeddings_errors_on_missing_corpus(self):
        """Missing corpus prereq exits 1 and points at `prepare_data.sh`."""
        result = self._run_against_fake_root('prepare_embeddings.sh')
        self.assertEqual(result.returncode, 1)
        self.assertIn('prerequisite corpus missing', result.stderr)
        self.assertIn('scripts/prepare_data.sh', result.stderr)

    def test_run_analysis_errors_on_missing_results(self):
        """Missing experiment results exit 1, point at `download_data.sh`."""
        result = self._run_against_fake_root('run_analysis.sh')
        self.assertEqual(result.returncode, 1)
        self.assertIn('prerequisite missing', result.stderr)
        self.assertIn('scripts/download_data.sh', result.stderr)

    def test_generate_paper_errors_on_missing_paper_dir(self):
        """Missing `paper/` directory exits 1 with a clear stderr message."""
        result = self._run_against_fake_root('generate_paper.sh')
        self.assertEqual(result.returncode, 1)
        self.assertIn('paper directory not found', result.stderr)


class ScriptConfirmationPromptUnitTests(unittest.TestCase):
    """Scripts that overwrite committed deliverables or spend money on
    Modal/OpenAI expose a y/N confirmation prompt + a ``--force`` flag to
    skip it. Without ``--force``, they prompt; with ``--force``, they
    proceed unattended.

    Tests build a fake repo with all prereqs satisfied + stub venv
    binaries + (for Modal scripts) a fake Modal config. The downstream
    operations are stubbed to exit 0, so the script can reach the prompt
    and proceed past it without spending real money or wall time.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._fake_root = Path(self._tmpdir.name)
        _build_fake_venv(self._fake_root)
        # run_all.sh chains to these — stub them so its dry-run works here.
        _build_fake_downstream_scripts(self._fake_root)

        # Prereqs for run_analysis.sh
        for d in ('src/experiments/results/experiments',
                  'src/experiments/results/judge',
                  'src/experiments/results/noise'):
            (self._fake_root / d).mkdir(parents=True, exist_ok=True)
        analysis_dir = self._fake_root / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / 'human_labels.csv').touch()
        (analysis_dir / 'analysis.ipynb').touch()

        # Fake HOME so run_experiments.sh's ~/.modal.toml check passes.
        self._fake_home = Path(self._tmpdir.name) / 'home'
        self._fake_home.mkdir()
        (self._fake_home / '.modal.toml').touch()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _run_with_stdin(self, script: str, *args: str, stdin: str = '',
                        ) -> subprocess.CompletedProcess:
        """Run a script with `stdin`, fake `REPO_ROOT`, and fake `HOME`."""
        env = os.environ.copy()
        env.update({
            'REPO_ROOT': str(self._fake_root),
            'HOME': str(self._fake_home),
        })
        return subprocess.run(
            [str(SCRIPTS_DIR / script), *args],
            input=stdin, capture_output=True, text=True, env=env,
        )

    # ---- run_analysis.sh -------------------------------------------------

    # NOTE on prompt detection: `bash read -p` only prints the prompt text
    # when stdin is a TTY (per the bash manual). Tests pipe stdin via
    # subprocess, which is non-TTY, so the literal "Continue?" string
    # never appears anywhere even when the prompt fires. We instead use:
    #   - WARNING text (printed via cat/echo before the read) as proof
    #     the prompt step was reached
    #   - "Aborted." (printed via echo after rejection) as proof the prompt
    #     fired and got rejected
    #   - "==> Executing" or similar downstream banner as proof the prompt
    #     fired and got accepted (or was bypassed by --force)

    def test_run_analysis_force_skips_prompt(self):
        """`--force` proceeds past the prompt without reading stdin."""
        result = self._run_with_stdin('run_analysis.sh', '--force')
        self.assertEqual(result.returncode, 0, result.stderr)
        # No WARNING text means the prompt block was skipped entirely.
        self.assertNotIn('WARNING', result.stdout)
        self.assertIn('Executing analysis/analysis.ipynb', result.stdout)

    def test_run_analysis_n_aborts(self):
        """Piping 'n' rejects the prompt cleanly without executing."""
        result = self._run_with_stdin('run_analysis.sh', stdin='n\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('WARNING', result.stdout)  # prompt block ran
        self.assertIn('Aborted', result.stdout)  # rejected
        self.assertNotIn('Executing analysis/analysis.ipynb', result.stdout)

    def test_run_analysis_warning_text_present(self):
        """Warning lists the directories that get overwritten."""
        result = self._run_with_stdin('run_analysis.sh', stdin='n\n')
        self.assertIn('analysis/figures/', result.stdout)
        self.assertIn('paper/tables/', result.stdout)
        self.assertIn('5-10 minutes', result.stdout)

    # ---- run_experiments.sh ---------------------------------------------

    def test_run_experiments_force_skips_prompt(self):
        """`--force` skips the prompt for `run_experiments.sh --noise`."""
        result = self._run_with_stdin('run_experiments.sh', '--force', '--noise')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('WARNING', result.stdout)
        self.assertNotIn('Aborted', result.stdout)

    def test_run_experiments_n_aborts(self):
        """Piping `n` aborts `run_experiments.sh --noise` cleanly."""
        result = self._run_with_stdin('run_experiments.sh', '--noise', stdin='n\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('WARNING', result.stdout)
        self.assertIn('Aborted', result.stdout)

    def test_run_experiments_warning_is_phase_aware(self):
        """The cost / runtime estimates differ per phase."""
        all_out = self._run_with_stdin('run_experiments.sh', stdin='n\n').stdout
        noise_out = self._run_with_stdin('run_experiments.sh', '--noise', stdin='n\n').stdout
        self.assertIn('$280-440', all_out)  # full-pipeline experiments cost
        self.assertIn('$20-30', noise_out)  # noise-only cost
        self.assertNotIn('$280-440', noise_out)

    def test_run_experiments_dry_run_skips_prompt(self):
        """``--dry-run`` bypasses the prompt (no stdin needed)."""
        result = self._run_with_stdin('run_experiments.sh', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('WARNING', result.stdout)

    # ---- run_all.sh ------------------------------------------------------
    #
    # No standalone --force test for run_all.sh: the --force pattern is
    # identical to the other two scripts (already covered) and the only
    # way to exercise --force without --dry-run is to also stub every
    # downstream script the chain invokes, which is more setup than the
    # test is worth.

    def test_run_all_n_aborts(self):
        """Piping `n` aborts `run_all.sh --analysis-only` after the warning."""
        result = self._run_with_stdin('run_all.sh', '--analysis-only', stdin='n\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('analysis-only pipeline', result.stdout)  # warning fired
        self.assertIn('Aborted', result.stdout)

    def test_run_all_warning_is_mode_aware(self):
        """`run_all.sh` warning text differs across modes."""
        default_out = self._run_with_stdin('run_all.sh', stdin='n\n').stdout
        resume_out = self._run_with_stdin('run_all.sh', '--resume', stdin='n\n').stdout
        analysis_only_out = self._run_with_stdin(
            'run_all.sh', '--analysis-only', stdin='n\n',
        ).stdout
        # Default mode warns about real cost
        self.assertIn('$300-450', default_out)
        self.assertIn('over 24 hours', default_out)
        # Resume + analysis-only are free / fast
        self.assertIn('free', resume_out)
        self.assertIn('free', analysis_only_out)
        self.assertNotIn('$300-450', resume_out)
        self.assertNotIn('$300-450', analysis_only_out)

    def test_run_all_dry_run_skips_prompt(self):
        """`run_all.sh --dry-run` bypasses the confirmation prompt."""
        result = self._run_with_stdin('run_all.sh', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('WARNING', result.stdout)


class ScriptShellcheckUnitTests(unittest.TestCase):
    """Optional static analysis. Skipped when shellcheck isn't installed."""

    def test_shellcheck_clean(self):
        """`shellcheck` exits 0 across every script in `ALL_SCRIPTS`."""
        shellcheck = shutil.which('shellcheck')
        if shellcheck is None:
            # shellcheck-py is in requirements.txt and provides venv/bin/shellcheck;
            # scripts/run_tests.sh prepends venv/bin to PATH so shutil.which finds it.
            # Only skips if a user invokes pytest in some other way without venv/bin
            # on PATH.
            self.skipTest('shellcheck not on PATH (use scripts/run_tests.sh)')
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
        """Populated dest dirs exit 1 with a `--force` pointer."""
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
