#!/usr/bin/env bash
# Build an arXiv-ready tar.gz bundle of the paper.
#
# Pipeline:
#   1. Audit paper.tex for dependencies (figures, tables, bib, packages).
#   2. Stage clean source into ../arxiv-staging/ (flat layout).
#   3. Compile in a scratch dir (../arxiv-staging-build/) to verify and
#      generate the .bbl that arXiv needs.
#   4. Copy the .bbl into staging.
#   5. Diff scratch-built PDF against repo paper.pdf (warn-only).
#   6. tar from staging → ../paper_arxiv.tar.gz
#
# arXiv runs latex+bibtex on the uploaded source, so we ship .tex + .bbl
# + figures + .bib + any custom .sty/.cls. We do NOT ship .aux/.log/.pdf.

set -euo pipefail

# ---------- Locate repo root ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
PAPER_TEX="$PAPER_DIR/paper.tex"
REPO_PDF="$PAPER_DIR/paper.pdf"

STAGING_DIR="$REPO_ROOT/../arxiv-staging"
BUILD_DIR="$REPO_ROOT/../arxiv-staging-build"
BUNDLE_PATH="$REPO_ROOT/../paper_arxiv.tar.gz"

# Resolve to absolute paths
STAGING_DIR="$(cd "$REPO_ROOT/.." && pwd)/arxiv-staging"
BUILD_DIR="$(cd "$REPO_ROOT/.." && pwd)/arxiv-staging-build"
BUNDLE_PATH="$(cd "$REPO_ROOT/.." && pwd)/paper_arxiv.tar.gz"

# Tool paths (TeX Live not on PATH on this machine)
PDFLATEX="${PDFLATEX:-/Library/TeX/texbin/pdflatex}"
BIBTEX="${BIBTEX:-/Library/TeX/texbin/bibtex}"

if [[ ! -x "$PDFLATEX" ]]; then
    if command -v pdflatex >/dev/null 2>&1; then
        PDFLATEX="$(command -v pdflatex)"
    else
        echo "ERROR: pdflatex not found (looked at $PDFLATEX and PATH)." >&2
        exit 1
    fi
fi
if [[ ! -x "$BIBTEX" ]]; then
    if command -v bibtex >/dev/null 2>&1; then
        BIBTEX="$(command -v bibtex)"
    else
        echo "ERROR: bibtex not found (looked at $BIBTEX and PATH)." >&2
        exit 1
    fi
fi

# ---------- Banner / version check ----------
echo "============================================================"
echo "arXiv submission bundle builder"
echo "============================================================"
echo "Repo root:    $REPO_ROOT"
echo "Paper:        $PAPER_TEX"
echo "Staging:      $STAGING_DIR"
echo "Build (tmp):  $BUILD_DIR"
echo "Bundle:       $BUNDLE_PATH"
echo

TEX_VERSION_LINE="$("$PDFLATEX" --version 2>&1 | head -1)"
echo "pdflatex:     $TEX_VERSION_LINE"
case "$TEX_VERSION_LINE" in
    *"TeX Live 2023"*|*"TeX Live 2025"*)
        ;;
    *)
        echo "WARNING: TeX Live 2023 or 2025 recommended; got: $TEX_VERSION_LINE"
        echo "         arXiv currently runs TeX Live 2023; older/newer may compile differently."
        ;;
esac
echo

# ---------- Step 1: audit ----------
echo "[1/12] Auditing $PAPER_TEX for dependencies..."

# Figures referenced via \includegraphics{...} (basename, no extension assumed
# present). Capture the brace argument; ignore the optional [..] arguments.
FIGURES=()
while IFS= read -r ref; do
    [[ -n "$ref" ]] && FIGURES+=("$ref")
done < <(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$PAPER_TEX" \
    | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u)

# Inputs (\input{...}, \include{...}). Anything not a tables/* path is unusual.
INPUTS=()
while IFS= read -r ref; do
    [[ -n "$ref" ]] && INPUTS+=("$ref")
done < <(grep -oE '\\(input|include)\{[^}]+\}' "$PAPER_TEX" \
    | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u)

# Bibliography
BIB_REFS=()
while IFS= read -r ref; do
    [[ -n "$ref" ]] && BIB_REFS+=("$ref")
done < <(grep -oE '\\bibliography\{[^}]+\}' "$PAPER_TEX" \
    | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u)

# Custom packages — flag any \usepackage referencing a local .sty (rare).
# Standard CTAN packages (geometry, hyperref, natbib, etc.) need no shipping.
CUSTOM_STY=()
for sty in "$PAPER_DIR"/*.sty "$PAPER_DIR"/*.cls; do
    [[ -f "$sty" ]] || continue
    CUSTOM_STY+=("$sty")
done

echo "  Figures referenced (${#FIGURES[@]}):"
for f in "${FIGURES[@]}"; do echo "    - $f"; done
echo "  \\input/\\include (${#INPUTS[@]}):"
for f in "${INPUTS[@]}"; do echo "    - $f"; done
echo "  \\bibliography (${#BIB_REFS[@]}):"
for f in "${BIB_REFS[@]}"; do echo "    - $f"; done
echo "  Custom .sty/.cls in paper dir (${#CUSTOM_STY[@]}):"
if (( ${#CUSTOM_STY[@]} > 0 )); then
    for f in "${CUSTOM_STY[@]}"; do echo "    - $f"; done
fi
echo

# ---------- Step 2: clean staging dir ----------
echo "[2/12] Resetting staging dir at $STAGING_DIR ..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# ---------- Step 3: stage paper.tex with path rewrites ----------
echo "[3/12] Copying paper.tex → staging with path flattening..."
cp "$PAPER_TEX" "$STAGING_DIR/paper.tex"

# Rewrite \graphicspath to local-only (./), since figures will be flat.
# Note: \graphicspath uses nested braces like \graphicspath{{a/}{b/}}, so we
# need a balanced-brace match — sed can't do this, perl can.
perl -pi -e 's|\\graphicspath\{(?:\{[^{}]*\})+\}|\\graphicspath{{./}}|g' "$STAGING_DIR/paper.tex"

# Flatten table inputs: \input{tables/foo} -> \input{foo}
perl -pi -e 's|\\input\{tables/([^}]+)\}|\\input{$1}|g' "$STAGING_DIR/paper.tex"

# Flatten figure paths if any subdir prefix is present (defensive — current
# paper uses \graphicspath so figures are already by basename, but be safe).
perl -pi -e 's|\\includegraphics(\[[^\]]*\])?\{[^{}]*/([^}/]+)\}|\\includegraphics$1\{$2\}|g' "$STAGING_DIR/paper.tex"

# ---------- Step 4: copy figures flat ----------
echo "[4/12] Copying figures flat into staging..."
FIG_SEARCH_DIRS=(
    "$REPO_ROOT/analysis/figures"
    "$PAPER_DIR/assets"
)

MISSING_FIGS=()
for fig in "${FIGURES[@]}"; do
    # If no extension on the reference, prefer .pdf
    base="$fig"
    if [[ "$base" != *.* ]]; then
        base="$base.pdf"
    fi
    found=""
    for d in "${FIG_SEARCH_DIRS[@]}"; do
        if [[ -f "$d/$base" ]]; then
            found="$d/$base"
            break
        fi
    done
    if [[ -z "$found" ]]; then
        MISSING_FIGS+=("$fig")
        continue
    fi
    cp "$found" "$STAGING_DIR/$(basename "$base")"
    echo "    + $(basename "$base")  (from $found)"
done

if (( ${#MISSING_FIGS[@]} > 0 )); then
    echo "ERROR: missing figures (no source found in any of: ${FIG_SEARCH_DIRS[*]}):" >&2
    for f in "${MISSING_FIGS[@]}"; do echo "    - $f" >&2; done
    exit 1
fi

# ---------- Step 5: copy tables flat ----------
echo "[5/12] Copying tables flat into staging..."
for inp in "${INPUTS[@]}"; do
    # Strip any directory prefix
    leaf="${inp##*/}"
    src=""
    if [[ -f "$REPO_ROOT/paper/$inp.tex" ]]; then
        src="$REPO_ROOT/paper/$inp.tex"
    elif [[ -f "$REPO_ROOT/paper/$inp" ]]; then
        src="$REPO_ROOT/paper/$inp"
    elif [[ -f "$REPO_ROOT/$inp.tex" ]]; then
        src="$REPO_ROOT/$inp.tex"
    elif [[ -f "$REPO_ROOT/$inp" ]]; then
        src="$REPO_ROOT/$inp"
    else
        echo "ERROR: \\input target not found: $inp" >&2
        exit 1
    fi
    cp "$src" "$STAGING_DIR/$leaf.tex"
    echo "    + $leaf.tex  (from $src)"
done

# ---------- Step 6: copy bib + custom styles ----------
echo "[6/12] Copying .bib and any custom .sty/.cls..."
for ref in "${BIB_REFS[@]}"; do
    src="$REPO_ROOT/paper/$ref.bib"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: \\bibliography target not found: $src" >&2
        exit 1
    fi
    cp "$src" "$STAGING_DIR/$ref.bib"
    echo "    + $ref.bib"
done
if (( ${#CUSTOM_STY[@]} > 0 )); then
    for sty in "${CUSTOM_STY[@]}"; do
        cp "$sty" "$STAGING_DIR/$(basename "$sty")"
        echo "    + $(basename "$sty")"
    done
fi

# ---------- Step 7: replace \today in the staged paper.tex ----------
echo "[7/12] Replacing \\today with hardcoded date in staged paper.tex..."
TODAY_STR="$(date "+%B %-d, %Y")"
if grep -q '\\today' "$STAGING_DIR/paper.tex"; then
    # Replace \today only inside \date{...} (typical usage); to be safe,
    # replace any occurrence of \today in the staged file.
    # macOS sed needs literal escapes; use a perl one-liner for clarity.
    perl -pi -e "s/\\\\today/$TODAY_STR/g" "$STAGING_DIR/paper.tex"
    echo "    \\today → $TODAY_STR"
else
    echo "    (no \\today found — nothing to replace)"
fi

# ---------- Step 8: compile in scratch build dir ----------
echo "[8/12] Compiling staged paper in scratch dir $BUILD_DIR ..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp -R "$STAGING_DIR"/* "$BUILD_DIR/"

cd "$BUILD_DIR"

run_pdflatex() {
    "$PDFLATEX" -interaction=nonstopmode -halt-on-error paper.tex >/tmp/arxiv_pdflatex.log 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "ERROR: pdflatex failed (rc=$rc). Last 50 lines of log:" >&2
        tail -50 /tmp/arxiv_pdflatex.log >&2
        echo >&2
        echo "Full log: $BUILD_DIR/paper.log" >&2
        exit 1
    fi
}

run_pdflatex
if (( ${#BIB_REFS[@]} > 0 )); then
    "$BIBTEX" paper >/tmp/arxiv_bibtex.log 2>&1 || {
        echo "ERROR: bibtex failed. Log:" >&2
        cat /tmp/arxiv_bibtex.log >&2
        exit 1
    }
fi
run_pdflatex
run_pdflatex
echo "    Compilation OK ($(ls -l paper.pdf | awk '{print $5}') bytes)."

# ---------- Step 9: copy .bbl back to staging ----------
if (( ${#BIB_REFS[@]} > 0 )); then
    echo "[9/12] Copying paper.bbl back to staging..."
    cp "$BUILD_DIR/paper.bbl" "$STAGING_DIR/paper.bbl"
else
    echo "[9/12] No external bib — skipping .bbl copy."
fi

# ---------- Step 10: PDF diff (warn-only) ----------
echo "[10/12] Comparing scratch PDF to repo paper.pdf (warn-only)..."
NEW_PDF="$BUILD_DIR/paper.pdf"
# Disable -e for this whole block — diff exits 1 on differences and we don't
# want that to fail the script.
set +e
if [[ -f "$REPO_PDF" ]]; then
    if command -v pdftotext >/dev/null 2>&1; then
        pdftotext -layout "$REPO_PDF" /tmp/arxiv_repo.txt 2>/dev/null
        pdftotext -layout "$NEW_PDF"  /tmp/arxiv_new.txt  2>/dev/null
        # Strip the date string we just rewrote so it doesn't dominate the diff
        sed -i.bak -E "s/[A-Z][a-z]+ +[0-9]+,? +[0-9]{4}//" /tmp/arxiv_repo.txt /tmp/arxiv_new.txt 2>/dev/null
        rm -f /tmp/arxiv_repo.txt.bak /tmp/arxiv_new.txt.bak
        diff -q /tmp/arxiv_repo.txt /tmp/arxiv_new.txt >/dev/null 2>&1
        diff_rc=$?
        if [[ $diff_rc -eq 0 ]]; then
            echo "    Text content identical (modulo date)."
        else
            DIFF_LINES="$(diff /tmp/arxiv_repo.txt /tmp/arxiv_new.txt | wc -l | tr -d ' ')"
            echo "    WARN: text-level diff vs repo paper.pdf ($DIFF_LINES diff lines)."
            echo "          first 20 differing lines:"
            diff /tmp/arxiv_repo.txt /tmp/arxiv_new.txt | head -20 | sed 's/^/          /'
        fi
    else
        REPO_BYTES="$(stat -f %z "$REPO_PDF" 2>/dev/null || stat -c %s "$REPO_PDF")"
        NEW_BYTES="$(stat -f %z "$NEW_PDF" 2>/dev/null || stat -c %s "$NEW_PDF")"
        echo "    pdftotext not available; byte sizes: repo=$REPO_BYTES new=$NEW_BYTES"
    fi
else
    echo "    (no repo paper.pdf to compare against — skipping)"
fi
set -e

# ---------- Step 11: bundle ----------
echo "[11/12] Building tar.gz bundle from staging..."
cd "$STAGING_DIR"

# Sanity-check: nothing in staging should be a build artifact.
BAD=()
shopt -s nullglob
for f in *.aux *.log *.out *.toc *.blg *.fls *.fdb_latexmk *.synctex.gz paper.pdf; do
    [[ -f "$f" ]] && BAD+=("$f")
done
shopt -u nullglob
if (( ${#BAD[@]} > 0 )); then
    echo "ERROR: build artifacts leaked into staging dir: ${BAD[*]}" >&2
    exit 1
fi

# Create the bundle. Exclude any dotfiles defensively.
rm -f "$BUNDLE_PATH"
# shellcheck disable=SC2035
tar --exclude='.*' -czvf "$BUNDLE_PATH" *

# ---------- Step 12: summary + cleanup ----------
echo
echo "[12/12] Summary"
echo "------------------------------------------------------------"
BUNDLE_BYTES="$(stat -f %z "$BUNDLE_PATH" 2>/dev/null || stat -c %s "$BUNDLE_PATH")"
BUNDLE_HUMAN="$(du -h "$BUNDLE_PATH" | awk '{print $1}')"
FILE_COUNT="$(tar tzf "$BUNDLE_PATH" | wc -l | tr -d ' ')"
echo "Bundle:     $BUNDLE_PATH"
echo "Size:       $BUNDLE_HUMAN ($BUNDLE_BYTES bytes)"
echo "Files:      $FILE_COUNT"
if (( BUNDLE_BYTES > 10 * 1024 * 1024 )); then
    echo "WARN: bundle exceeds 10 MB."
fi
echo
echo "Contents:"
tar tzvf "$BUNDLE_PATH" | sed 's/^/  /'
echo

# Clean up scratch build dir; keep staging dir for inspection.
echo "Cleaning up scratch build dir $BUILD_DIR ..."
rm -rf "$BUILD_DIR"

echo
echo "Done. Bundle ready at: $BUNDLE_PATH"
echo "Staging dir kept at:   $STAGING_DIR  (safe to delete)"
