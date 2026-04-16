"""
Phase 4 migration test: verify analysis notebook and paper compilation.

No API calls. Requires data symlinks + human_labels.csv symlink.

Run from repo root:
    python tests/test_analysis_paper.py
    python tests/test_analysis_paper.py --skip-notebook   # paper compilation only
    python tests/test_analysis_paper.py --skip-paper      # notebook only
"""

import argparse
import os
import subprocess
import sys
import tempfile


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
_ANALYSIS_DIR = os.path.join(_REPO_ROOT, 'analysis')
_PAPER_DIR = os.path.join(_REPO_ROOT, 'paper')
_VENV_BIN = os.path.join(_REPO_ROOT, 'venv', 'bin')


# ---------------------------------------------------------------------------
# Test: notebook executes without errors
# ---------------------------------------------------------------------------

def test_notebook_executes():
    """Run analysis.ipynb via nbconvert and verify outputs."""
    print("\n=== test_notebook_executes ===")

    notebook_path = os.path.join(_ANALYSIS_DIR, 'analysis.ipynb')
    assert os.path.exists(notebook_path), f"Missing notebook: {notebook_path}"

    # Verify data dependencies exist
    human_labels = os.path.join(_ANALYSIS_DIR, 'human_labels.csv')
    assert os.path.exists(human_labels), (
        f"Missing human_labels.csv symlink at {human_labels}\n"
        "Run: ln -s ../../workspace/analysis/human_labels.csv analysis/human_labels.csv"
    )

    # Run notebook with output to temp file (don't clobber committed version)
    with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False, dir=_ANALYSIS_DIR) as tmp:
        output_path = tmp.name

    print(f"  Running notebook (this takes ~5-10 minutes)...")
    jupyter = os.path.join(_VENV_BIN, 'jupyter')
    result = subprocess.run(
        [
            jupyter, 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--ExecutePreprocessor.timeout=600',
            notebook_path,
            '--output', os.path.basename(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=_ANALYSIS_DIR,
        env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE'},
    )

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        os.unlink(output_path)
        assert False, f"Notebook execution failed (exit code {result.returncode})"

    print(f"  Notebook executed successfully")

    # Verify figures were generated
    figures_dir = os.path.join(_ANALYSIS_DIR, 'figures')
    expected_figures = [
        'accuracy_fullcond.pdf',
        'asr_fullcond.pdf',
        'detection_heatmap.pdf',
        'safety_profile.pdf',
    ]
    generated = os.listdir(figures_dir) if os.path.isdir(figures_dir) else []
    pdf_files = [f for f in generated if f.endswith('.pdf')]
    print(f"  Generated {len(pdf_files)} PDF figures")
    for fig in expected_figures:
        assert fig in generated, f"Missing expected figure: {fig}"
        print(f"    OK  {fig}")

    # Verify parquet was saved
    parquet_path = os.path.join(_ANALYSIS_DIR, 'intermediate', 'merged_results.parquet')
    assert os.path.exists(parquet_path), f"Missing parquet: {parquet_path}"
    size_kb = os.path.getsize(parquet_path) / 1024
    print(f"  Parquet: {parquet_path} ({size_kb:.0f} KB)")

    # Cleanup temp notebook
    os.unlink(output_path)

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test: paper compiles with pdflatex
# ---------------------------------------------------------------------------

def test_paper_compiles():
    """Run pdflatex on paper_draft_working.tex (two passes)."""
    print("\n=== test_paper_compiles ===")

    tex_file = os.path.join(_PAPER_DIR, 'paper_draft_working.tex')
    assert os.path.exists(tex_file), f"Missing: {tex_file}"

    pdflatex = '/Library/TeX/texbin/pdflatex'
    assert os.path.exists(pdflatex), f"pdflatex not found at {pdflatex}"

    # Figures must exist for compilation (paper uses \graphicspath{{../analysis/figures/}})
    figures_dir = os.path.join(_ANALYSIS_DIR, 'figures')
    figures_symlinked = False
    if not os.path.isdir(figures_dir) or len(os.listdir(figures_dir)) <= 1:
        # If notebook hasn't been run, symlink workspace figures
        workspace_figures = os.path.join(_REPO_ROOT, '..', 'workspace', 'analysis', 'figures')
        if os.path.isdir(workspace_figures):
            import shutil
            if os.path.isdir(figures_dir):
                shutil.rmtree(figures_dir)
            os.symlink(os.path.abspath(workspace_figures), figures_dir)
            figures_symlinked = True
            print(f"  Symlinked analysis/figures/ -> workspace figures")
        else:
            print(f"  WARNING: No workspace figures found at {workspace_figures}")

    # Two-pass compilation (run from paper/ dir so relative paths resolve)
    for pass_num in (1, 2):
        result = subprocess.run(
            [pdflatex, '-interaction=nonstopmode', os.path.basename(tex_file)],
            capture_output=True,
            text=True,
            cwd=_PAPER_DIR,
        )
        if result.returncode != 0 and pass_num == 2:
            # pdflatex often returns non-zero on warnings; check for actual errors
            log_path = os.path.join(_PAPER_DIR, 'paper_draft_working.log')
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log_content = f.read()
                # Fatal errors contain "! " at start of line
                fatal_errors = [l for l in log_content.split('\n') if l.startswith('! ')]
                if fatal_errors:
                    print(f"  Fatal LaTeX errors:")
                    for e in fatal_errors[:5]:
                        print(f"    {e}")
                    assert False, "Paper compilation has fatal errors"
        print(f"  Pass {pass_num}: {'OK' if result.returncode == 0 else 'warnings'}")

    # Verify PDF was generated
    pdf_path = os.path.join(_PAPER_DIR, 'paper_draft_working.pdf')
    assert os.path.exists(pdf_path), f"Missing output PDF: {pdf_path}"
    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"  Output: {pdf_path} ({size_kb:.0f} KB)")

    # Check for undefined references in log
    log_path = os.path.join(_PAPER_DIR, 'paper_draft_working.log')
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_content = f.read()
        undef_refs = log_content.count('undefined')
        if undef_refs > 0:
            print(f"  WARNING: {undef_refs} 'undefined' occurrences in log")
        else:
            print(f"  No undefined references")

    # Restore figures dir if we symlinked it
    if figures_symlinked:
        os.unlink(figures_dir)
        os.makedirs(figures_dir, exist_ok=True)
        print(f"  Restored analysis/figures/ directory")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-notebook', action='store_true', help='Skip notebook execution')
    parser.add_argument('--skip-paper', action='store_true', help='Skip paper compilation')
    args = parser.parse_args()

    if not args.skip_notebook:
        test_notebook_executes()
    else:
        print("\n=== SKIPPED notebook test (--skip-notebook) ===")

    if not args.skip_paper:
        test_paper_compiles()
    else:
        print("\n=== SKIPPED paper test (--skip-paper) ===")

    print("\n=== ALL TESTS PASSED ===")
