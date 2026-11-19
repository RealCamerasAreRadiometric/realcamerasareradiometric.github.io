# paper/tools — syncing the paper from Overleaf

The manuscript is written on Overleaf; Overleaf's git repository holds it under `src/` (the review build of the author kit).
`paper/` here is the **release build** of the same text, flattened (`src/*` → `paper/*`). `sync_overleaf.sh` makes the one from
the other, and is meant to be run again whenever the Overleaf text changes.

## Run

```sh
# from a local clone that has the Overleaf repository as a remote (fetched by the tool):
OVERLEAF_REPO=/path/to/clone OVERLEAF_REF=<remote>/main paper/tools/sync_overleaf.sh
# or from the repository URL (cloned into a temporary directory):
paper/tools/sync_overleaf.sh --repo <git-url> --ref main
# and the arXiv upload beside it:
paper/tools/sync_overleaf.sh --repo <git-url> --arxiv /some/dir      # /some/dir/arxiv-source.tar.gz + preview.pdf
```

| option | default | |
|---|---|---|
| `--repo PATH\|URL` | `$OVERLEAF_REPO` (required) | the Overleaf git repository; the URL is private and is not written in this repository |
| `--ref REF` | `$OVERLEAF_REF`, else `blens/main` in a clone with a remote `blens`, else `main` | the commit to sync |
| `--arxiv OUTDIR` | — | also write the arXiv source tarball and test-build it |
| `--skip-experiment-checks` | — | do not run the experiments' paper checks (the command is printed) |

Requirements: git, python3 (≥ 3.11), latexmk/pdflatex, poppler-utils (pdfinfo, pdftotext), julia (the experiments' checks).

## What it does

1. **fetch and export** the ref's `src/` tree (fails if there is no `src/main.tex`); lists the upstream commits since the last sync
   (`SYNCED`).
2. **flatten** into `paper/`: everything in `paper/` except `Makefile`, `.gitignore`, `tools/` and `build/` is replaced by the
   export; `rebuttal.tex`, `sec_full/`, the author kit's `README.md` and `.github/` are dropped.
3. **release transforms** (`release.py transform`): `src/{fig,latex,sec}` paths → `{fig,latex,sec}`; in `main.tex`
   `\usepackage[review]{cvpr}` → `\usepackage[pagenumbers]{cvpr}`, the `\paperID`/`\confName`/`\confYear` lines removed, the
   `\author{…}` block replaced by `authors.tex`; comment scrubs (the experiments' directory name, no private commit hashes); then
   the anchored text edits of `release_edits.toml`. A required anchor that is missing is an error, and so is any `\todo`/`\TODO`
   left in the text: an open note of the authors never reaches the release, and it is resolved on Overleaf, never by a release edit.
4. Release-only structure: none is needed today (the runtime section's placement is the same on Overleaf); a future one goes into
   `release_edits.toml` as an anchored edit.
5. **build** with latexmk (`paper/build/`), then **prune** `fig/` and `latex/` to exactly the files the build read (`build/main.fls`).
6. **checks**: no undefined references, no review marks or line numbers in the PDF, the venue grep over `paper/` (only
   `paper/cvpr.sty` may match) and the whole repository (printed), and the experiments' paper checks
   (`julia --project=experiments experiments/test/runtests.jl 'paper-project|reproduce index|the manuscript|data policy'`, from the
   repository root). A new figure must have its rows in `experiments/REPRODUCE.md` (and, when it reads panels through `\rcfigdir`,
   an entry in `experiments/test/fixtures/paper_allow.toml`) for these to pass.
7. `--arxiv OUTDIR`: `OUTDIR/arxiv-source.tar.gz` — the build's inputs with full-line comments removed (a comment-only file becomes
   `\relax`), `main.bbl` instead of `main.bib`/`.bst`, no `\paperID`; it is extracted and built with pdflatex (no bibtex) and must
   give the same page count; `OUTDIR/preview.pdf` is that build.
8. **summary**: `git diff --stat` of `paper/` and the new files; `SYNCED` records the upstream commit, its date and the ref.

The tool is idempotent: a second run on the same upstream commit leaves the tree unchanged. To undo a run: `git checkout -- paper`
and `git clean -fd paper` (careful: also removes untracked files you added under `paper/`).

## Files

| file | |
|---|---|
| `sync_overleaf.sh` | the tool |
| `release.py` | the transforms, the prune and the arXiv tree (called by the tool) |
| `authors.tex` | the release's author block |
| `release_edits.toml` | the anchored release edits of the Overleaf text, each with its reason |
| `SYNCED` | the upstream commit last synced (written by the tool) |
