"""Tests for the analysis notebook and paper compilation.

* :class:`NotebookValidityUnitTests` — load analysis.ipynb via nbformat
  and verify it parses (catches accidental JSON corruption). No data,
  no execution.
* :class:`NotebookExecutionIntegrationTests` — execute the notebook end
  to end via nbconvert; verify every figure the paper references is
  produced and the merged-results parquet is written.
* :class:`PaperCompilationIntegrationTests` — two-pass pdflatex.
"""

import os
import re
import subprocess
import tempfile
import unittest

import pytest


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
_ANALYSIS_DIR = os.path.join(_REPO_ROOT, 'analysis')
_PAPER_DIR = os.path.join(_REPO_ROOT, 'paper')
_VENV_BIN = os.path.join(_REPO_ROOT, 'venv', 'bin')
_PDFLATEX = '/Library/TeX/texbin/pdflatex'


def _missing_data_skip_message(path: str) -> str:
    return (
        f"Integration test requires {path}. "
        f"Either run scripts/download_data.sh to fetch the published dataset, "
        f"or regenerate the data by running the experiment pipeline."
    )


def _paper_figure_filenames() -> list[str]:
    """Parse paper_draft_working.tex for every \\includegraphics target."""
    tex_path = os.path.join(_PAPER_DIR, 'paper_draft_working.tex')
    with open(tex_path) as f:
        tex = f.read()
    # \includegraphics[...]{filename.pdf}
    return sorted(set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)))


# ===========================================================================
# Unit suite
# ===========================================================================

class NotebookValidityUnitTests(unittest.TestCase):
    """analysis.ipynb is well-formed JSON conforming to nbformat."""

    def test_notebook_parses_as_nbformat(self):
        import nbformat

        notebook_path = os.path.join(_ANALYSIS_DIR, 'analysis.ipynb')
        self.assertTrue(os.path.exists(notebook_path), notebook_path)

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Raises nbformat.ValidationError if the notebook is malformed.
        nbformat.validate(nb)
        self.assertGreater(len(nb.cells), 5)


# ===========================================================================
# Integration suite
# ===========================================================================

@pytest.mark.integration
class NotebookExecutionIntegrationTests(unittest.TestCase):
    """Run analysis.ipynb via nbconvert and verify outputs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        notebook_path = os.path.join(_ANALYSIS_DIR, 'analysis.ipynb')
        if not os.path.exists(notebook_path):
            raise unittest.SkipTest(f"Missing notebook: {notebook_path}")

        # Notebook reads from these paths (see analysis.ipynb).
        required = [
            os.path.join(_ANALYSIS_DIR, 'human_labels.csv'),
            os.path.join(_REPO_ROOT, 'src', 'experiments', 'results'),
            os.path.join(_REPO_ROOT, 'src', 'experiments', 'results', 'experiments'),
            os.path.join(_REPO_ROOT, 'src', 'experiments', 'results', 'noise'),
            os.path.join(_REPO_ROOT, 'src', 'data', 'original-datasets', 'nq', 'corpus.jsonl'),
            os.path.join(_REPO_ROOT, 'src', 'data', 'original-datasets', 'nq', 'qrels', 'test.tsv'),
            os.path.join(_REPO_ROOT, 'src', 'data', 'experiment-datasets', 'nq-corruptrag-ak-poisoned-docs.jsonl'),
            os.path.join(_REPO_ROOT, 'src', 'data', 'experiment-datasets', 'nq-incorrect-answers-poisoned-docs.jsonl'),
        ]
        for path in required:
            if not os.path.exists(path):
                raise unittest.SkipTest(_missing_data_skip_message(path))
        cls.notebook_path = notebook_path

    def test_notebook_executes_and_produces_outputs(self):
        # Run nbconvert into a temp output filename so we don't clobber the
        # committed notebook.
        with tempfile.NamedTemporaryFile(
            suffix='.ipynb', delete=False, dir=_ANALYSIS_DIR,
        ) as tmp:
            output_path = tmp.name

        try:
            jupyter = os.path.join(_VENV_BIN, 'jupyter')
            result = subprocess.run(
                [
                    jupyter, 'nbconvert',
                    '--to', 'notebook',
                    '--execute',
                    '--ExecutePreprocessor.timeout=600',
                    self.notebook_path,
                    '--output', os.path.basename(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=_ANALYSIS_DIR,
                env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE'},
            )
            self.assertEqual(
                result.returncode, 0,
                f"Notebook execution failed: {result.stderr[-500:]}",
            )

            # Every \includegraphics target in the paper must exist on disk.
            figures_dir = os.path.join(_ANALYSIS_DIR, 'figures')
            self.assertTrue(os.path.isdir(figures_dir), figures_dir)
            generated = set(os.listdir(figures_dir))

            expected = _paper_figure_filenames()
            self.assertGreater(len(expected), 5, "no figures parsed from paper.tex")

            missing = [f for f in expected if f not in generated]
            self.assertFalse(
                missing,
                f"Notebook didn't produce {len(missing)} paper-referenced figure(s): {missing}",
            )

            parquet_path = os.path.join(_ANALYSIS_DIR, 'intermediate', 'merged_results.parquet')
            self.assertTrue(os.path.exists(parquet_path), parquet_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@pytest.mark.integration
class PaperCompilationIntegrationTests(unittest.TestCase):
    """Two-pass pdflatex compilation of paper_draft_working.tex."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tex_file = os.path.join(_PAPER_DIR, 'paper_draft_working.tex')
        if not os.path.exists(tex_file):
            raise unittest.SkipTest(f"Missing TeX source: {tex_file}")
        if not os.path.exists(_PDFLATEX):
            raise unittest.SkipTest(f"pdflatex not installed at {_PDFLATEX}")

        # Paper uses \graphicspath{{../analysis/figures/}}; every figure it
        # references must be on disk before the compile can succeed.
        figures_dir = os.path.join(_ANALYSIS_DIR, 'figures')
        if not os.path.isdir(figures_dir):
            raise unittest.SkipTest(_missing_data_skip_message(figures_dir))

        generated = set(os.listdir(figures_dir))
        missing = [f for f in _paper_figure_filenames() if f not in generated]
        if missing:
            raise unittest.SkipTest(
                _missing_data_skip_message(
                    f"{figures_dir} (missing: {missing[:3]}{'...' if len(missing) > 3 else ''})"
                )
            )

        cls.tex_file = tex_file

    def test_paper_compiles_and_produces_pdf(self):
        for pass_num in (1, 2):
            result = subprocess.run(
                [_PDFLATEX, '-interaction=nonstopmode', os.path.basename(self.tex_file)],
                capture_output=True,
                text=True,
                cwd=_PAPER_DIR,
            )
            if result.returncode != 0 and pass_num == 2:
                log_path = os.path.join(_PAPER_DIR, 'paper_draft_working.log')
                fatal = []
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        fatal = [l for l in f if l.startswith('! ')]
                self.assertFalse(fatal, f"Fatal LaTeX errors: {fatal[:5]}")

        pdf_path = os.path.join(_PAPER_DIR, 'paper_draft_working.pdf')
        self.assertTrue(os.path.exists(pdf_path), pdf_path)
