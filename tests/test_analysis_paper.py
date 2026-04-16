"""Tests for the analysis notebook and paper compilation.

* :class:`NotebookValidityUnitTests` — load analysis.ipynb via nbformat
  and verify it parses (catches accidental JSON corruption). No data,
  no execution.
* :class:`NotebookExecutionIntegrationTests` — execute the notebook end
  to end via nbconvert; verify figures and parquet are produced.
* :class:`PaperCompilationIntegrationTests` — two-pass pdflatex.
"""

import os
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

        # nbformat will raise ValidationError if the notebook is malformed.
        nbformat.validate(nb)

        # Sanity: should have at least a few cells.
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
        human_labels = os.path.join(_ANALYSIS_DIR, 'human_labels.csv')
        if not os.path.exists(human_labels):
            raise unittest.SkipTest(
                f"Missing {human_labels}. "
                f"Run scripts/setup_test_symlinks.sh (local dev) or "
                f"scripts/download_data.sh (once Phase 5 lands)."
            )
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

            figures_dir = os.path.join(_ANALYSIS_DIR, 'figures')
            generated = os.listdir(figures_dir) if os.path.isdir(figures_dir) else []
            for fig in (
                'accuracy_fullcond.pdf',
                'asr_fullcond.pdf',
                'detection_heatmap.pdf',
                'safety_profile.pdf',
            ):
                self.assertIn(fig, generated, f"Missing expected figure: {fig}")

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
        cls.tex_file = tex_file

    def test_paper_compiles_and_produces_pdf(self):
        # The paper uses \graphicspath{{../analysis/figures/}}; if the
        # figures dir is missing or empty, skip with a hint.
        figures_dir = os.path.join(_ANALYSIS_DIR, 'figures')
        if not os.path.isdir(figures_dir) or len(
            [f for f in os.listdir(figures_dir) if f.endswith('.pdf')]
        ) == 0:
            self.skipTest(
                f"No figures in {figures_dir}. Run the notebook test first "
                f"or symlink workspace figures via "
                f"scripts/setup_test_symlinks.sh."
            )

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
