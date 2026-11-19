#!/usr/bin/env bash
# Assembles the whole project site into one tree, the one .github/workflows/site.yml deploys:
#   <out>/            the landing page (site/make.jl)
#   <out>/docs/       the package documentation (RadiometricCamera/docs/make.jl)
#   <out>/paper.pdf   the paper and its supplement (paper/build/main.pdf: `make -C paper pdf`, or CI's paper job)
#   <out>/assets/landing/teaser.png   the paper's teaser on the landing page (paper/build/teaser.png: `make -C paper teaser`,
#                     or CI's paper job), rendered from paper/fig/teaser.tex
# Usage: site/build.sh [out]   (default site/build). Instantiate site/ and RadiometricCamera/docs/ first (README.md).
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=$(realpath -m "${1:-$root/site/build}")
julia=${JULIA:-julia}
pdf=$root/paper/build/main.pdf
teaser=$root/paper/build/teaser.png
[ -f "$pdf" ] || { echo "site/build.sh: $pdf missing: build it with 'make -C paper pdf'" >&2; exit 1; }
[ -f "$teaser" ] || { echo "site/build.sh: $teaser missing: build it with 'make -C paper teaser'" >&2; exit 1; }

# The landing page first: its makedocs empties <out>.
"$julia" --color=yes --startup-file=no --project="$root/site" "$root/site/make.jl" "$out"
"$julia" --color=yes --startup-file=no --project="$root/RadiometricCamera/docs" "$root/RadiometricCamera/docs/make.jl" "$out/docs"
cp "$pdf" "$out/paper.pdf"
cp "$teaser" "$out/assets/landing/teaser.png"
touch "$out/.nojekyll"   # GitHub Pages serves the files as they are (no Jekyll pass)

# The landing page and the docs link to each other and to the paper by relative paths, and the targets exist.
for target in docs/ paper.pdf; do
    grep -q "href=\"$target\"" "$out/index.html" || { echo "site/build.sh: index.html has no link to $target" >&2; exit 1; }
done
for target in ../ ../paper.pdf; do
    grep -q "href=\"$target\"" "$out/docs/index.html" || { echo "site/build.sh: docs/index.html has no link to $target" >&2; exit 1; }
done
grep -q 'src="assets/landing/teaser.png"' "$out/index.html" || { echo "site/build.sh: index.html does not show the teaser" >&2; exit 1; }
[ -f "$out/docs/index.html" ] && [ -s "$out/paper.pdf" ] && [ -s "$out/assets/landing/teaser.png" ] ||
    { echo "site/build.sh: docs/index.html, paper.pdf or assets/landing/teaser.png missing in $out" >&2; exit 1; }
echo "site/build.sh: the site is in $out"
