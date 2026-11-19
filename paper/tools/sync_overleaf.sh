#!/usr/bin/env bash
# sync_overleaf.sh — bring the Overleaf version of the paper into paper/ as the RELEASE build (see tools/README.md).
#
#   paper/tools/sync_overleaf.sh [--repo PATH|URL] [--ref REF] [--arxiv OUTDIR] [--skip-experiment-checks]
#
#   --repo   the Overleaf git repository: a local clone (its remote is fetched) or a URL; default $OVERLEAF_REPO (required)
#   --ref    the commit to sync; default $OVERLEAF_REF, else `blens/main` in a local clone that has the remote `blens`, else `main`
#   --arxiv  also write OUTDIR/arxiv-source.tar.gz (TeX source + main.bbl, full-line comments stripped, no \paperID) and test-build it
#   --skip-experiment-checks   do not run the experiments' paper checks (the command is printed either way)
#
# Every step fails loudly (exit ≠ 0) when what it expects is missing; the tree is left as far as it got (git checkout -- paper to undo).
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="$(dirname "$TOOLS")"
ROOT="$(git -C "$PAPER" rev-parse --show-toplevel)"
PY="python3 -I"

REPO="${OVERLEAF_REPO:-}"; REF="${OVERLEAF_REF:-}"; ARXIV=""; RUN_JL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --arxiv) ARXIV="$2"; shift 2 ;;
    --skip-experiment-checks) RUN_JL=0; shift ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$REPO" ] || die "no Overleaf repository: pass --repo PATH|URL or set OVERLEAF_REPO"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# ---- 1. fetch + export the ref's src/ tree --------------------------------------------------------------------------------
step "1. fetch and export"
if [ -d "$REPO" ] && git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  GIT=(git -C "$REPO")
  if [ -z "$REF" ]; then
    if git -C "$REPO" remote | grep -qx blens; then REF=blens/main; else REF=main; fi
  fi
  remote="${REF%%/*}"
  if [ "$remote" != "$REF" ] && git -C "$REPO" remote | grep -qx "$remote"; then
    git -C "$REPO" fetch --quiet "$remote" || die "git fetch $remote failed in $REPO"
  fi
else
  REF="${REF:-main}"
  git clone --quiet --bare "$REPO" "$WORK/repo.git" || die "git clone $REPO failed"
  GIT=(git -C "$WORK/repo.git")
fi
COMMIT="$("${GIT[@]}" rev-parse --verify --quiet "$REF^{commit}")" || die "ref $REF not found in $REPO"
CDATE="$("${GIT[@]}" log -1 --format=%cd --date=iso-strict "$COMMIT")"
CSUBJ="$("${GIT[@]}" log -1 --format=%s "$COMMIT")"
echo "upstream: $REF = ${COMMIT:0:9} ($CDATE) $CSUBJ"
"${GIT[@]}" cat-file -e "$COMMIT:src/main.tex" 2>/dev/null || die "$REF has no src/main.tex (not the Overleaf layout)"
mkdir -p "$WORK/export"
"${GIT[@]}" archive "$COMMIT" src | tar -x -C "$WORK/export"
if [ -f "$TOOLS/SYNCED" ]; then
  prev="$(awk '{print $1}' "$TOOLS/SYNCED")"
  if "${GIT[@]}" cat-file -e "$prev^{commit}" 2>/dev/null; then
    echo "upstream commits since the last sync (${prev:0:9}):"
    "${GIT[@]}" log --format='  %h %cd %an: %s' --date=short "$prev..$COMMIT" -- src || true
  fi
fi

# ---- 2. flatten into paper/ (src/* -> paper/*); the release drops review and author-kit material ---------------------------
step "2. flatten into paper/"
for x in rebuttal.tex sec_full README.md .github; do
  if [ -e "$WORK/export/src/$x" ]; then rm -rf "$WORK/export/src/$x"; echo "  dropped (release build): src/$x"; fi
done
# paper/'s own files (not from Overleaf): Makefile, .gitignore, tools/, the local build/
find "$PAPER" -mindepth 1 -maxdepth 1 ! -name Makefile ! -name .gitignore ! -name tools ! -name build -exec rm -rf {} +
cp -a "$WORK/export/src/." "$PAPER/"

# ---- 3./4. release transforms (main.tex, paths, provenance scrubs, tools/release_edits.toml) and their checks -------------
step "3. release transforms"
$PY "$TOOLS/release.py" transform "$PAPER" || die "release transforms failed (see above)"

# ---- 5. build, then prune fig/ and latex/ to what the build reads -----------------------------------------------------------
step "5. build (latexmk) and prune"
rm -rf "$PAPER/build"
( cd "$PAPER" && TEXINPUTS=.: BIBINPUTS=.: BSTINPUTS=.: latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir=build main.tex >"$WORK/latexmk.out" 2>&1 ) || { tail -40 "$WORK/latexmk.out"; die "the LaTeX build failed (paper/build/main.log)"; }
$PY "$TOOLS/release.py" prune "$PAPER" "$PAPER/build/main.fls" || die "prune failed"

# ---- 6. checks -------------------------------------------------------------------------------------------------------------
step "6. checks"
LOG="$PAPER/build/main.log"; PDF="$PAPER/build/main.pdf"
if grep -E "LaTeX Warning: (Reference|Citation).*undefined|There were undefined references|Label\(s\) may have changed" "$LOG"; then
  die "undefined references or unsettled labels in the build"
fi
PAGES="$(pdfinfo "$PDF" | awk '/^Pages:/{print $2}')"
pdftotext "$PDF" "$WORK/main.txt"
grep -n -E "Submission #|CONFIDENTIAL ""REVIEW COPY|DO NOT DISTRIBUTE" "$WORK/main.txt" && die "review-build text in the PDF"
[ "$(grep -c -x -E '0[0-9]{2}' "$WORK/main.txt" || true)" -lt 5 ] || die "the PDF has review line numbers"
echo "  build: $PAGES pages, no undefined references, no review marks or line numbers"
CVPR_RX="cvpr ?20""27|cvpr20""27|141""60|under sub""mission|confidential rev""iew"   # split so that this file passes it
hits="$(git -C "$ROOT" grep --untracked -n -i -E "$CVPR_RX" -- paper ':!paper/cvpr.sty' ':!paper/build' || true)"
[ -z "$hits" ] || { echo "$hits"; die "CVPR wording under paper/ (outside cvpr.sty)"; }
echo "  CVPR grep under paper/: clean (cvpr.sty excepted); the whole repository:"
git -C "$ROOT" grep --untracked -c -i -E "$CVPR_RX" -- . ':!paper' | sed 's/^/    /' || true
JL_CMD="julia --project=experiments experiments/test/runtests.jl 'paper-project|reproduce index|the manuscript|data policy'"
echo "  experiments' paper checks: (cd $ROOT && $JL_CMD)"
if [ "$RUN_JL" = 1 ]; then
  ( cd "$ROOT" && RETESTITEMS_NWORKERS="${RETESTITEMS_NWORKERS:-2}" eval "$JL_CMD" ) || die "the experiments' paper checks failed"
else
  echo "  (skipped: --skip-experiment-checks; run the command above)"
fi

# ---- 7. optional: the arXiv tarball ----------------------------------------------------------------------------------------
if [ -n "$ARXIV" ]; then
  step "7. arXiv source"
  mkdir -p "$ARXIV"; ARXIV="$(cd "$ARXIV" && pwd)"
  rm -rf "$WORK/arxiv"; mkdir -p "$WORK/arxiv"
  $PY "$TOOLS/release.py" arxiv "$PAPER" "$WORK/arxiv" || die "arxiv tree failed"
  cp "$PAPER/build/main.bbl" "$WORK/arxiv/main.bbl"
  tar -czf "$ARXIV/arxiv-source.tar.gz" -C "$WORK/arxiv" .
  mkdir -p "$WORK/arxiv-test" && tar -xzf "$ARXIV/arxiv-source.tar.gz" -C "$WORK/arxiv-test"
  ( cd "$WORK/arxiv-test" && for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || exit 1; done ) \
    || die "the arXiv tarball does not build with pdflatex (no bibtex)"
  APAGES="$(pdfinfo "$WORK/arxiv-test/main.pdf" | awk '/^Pages:/{print $2}')"
  grep -q -E "LaTeX Warning: (Reference|Citation).*undefined" "$WORK/arxiv-test/main.log" && die "arXiv build: undefined references"
  [ "$APAGES" = "$PAGES" ] || die "arXiv build: $APAGES pages, the release build $PAGES"
  cp "$WORK/arxiv-test/main.pdf" "$ARXIV/preview.pdf"
  echo "  $ARXIV/arxiv-source.tar.gz ($(du -h "$ARXIV/arxiv-source.tar.gz" | cut -f1)), builds with pdflatex to $APAGES pages; preview.pdf beside it"
fi

# ---- 8. summary ------------------------------------------------------------------------------------------------------------
step "8. summary"
echo "$COMMIT $CDATE $REF" > "$TOOLS/SYNCED"
echo "synced: $REF = ${COMMIT:0:9} ($CDATE) $CSUBJ — $PAGES pages"
echo "paper/ against HEAD:"
git -C "$ROOT" diff --stat -- paper | sed 's/^/  /'
git -C "$ROOT" ls-files --others --exclude-standard -- paper | sed 's/^/  new: /'
