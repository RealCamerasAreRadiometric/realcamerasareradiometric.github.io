#!/bin/bash
# texture_learned_photograph_20261121.sh: the band "training photograph" of fig:texture-learned (the paper's Fig. 7, right), cut on
# 2026-11-21 from the texture run of 2026-11-16 (RadiometricCamera 0.11.5) by texture_learned_photograph.py — the photograph part of
# T. Harder's texture_learned.py, the generator of the submitted figure:
#   the design and the page's size      texture_recovery evaluate/16f9d84088c7714b (design.png, report.json)
#   the photographs' poses              texture_recovery init/a2902f6141e57d0e (poses.json)
#   the photographs' roles              texture_recovery refine/270c783b87521b12 (page.json)
#   the RAW photographs                 <data root>/paper-data/defocus/14mm/texture-learning (106 ARW; the tool picks DSC03316.ARW)
# Python 3.10.12 with rawpy 0.27.1 (LibRaw 0.22.1), OpenCV 5.0.0, NumPy 2.2.6 and Pillow 12.3.0 (the machines of the run of record
# have neither rawpy nor cv2). It writes photograph.png (573 x 76 px; sha256 ae5d0392…) and meta.toml (sha256 6f0c2833…): the files
# <data root>/geometric_accuracy/night_stars/data/fx3-14mm-textex-20261109-small-g7/texture_learned_photograph/ of tier
# rc_baselines_2026-11-21, the inputs texture_learned_photograph and texture_learned_photograph_record of
# configs/experiments/texture_recovery_14mm_small.toml. The step's record: configs/inputs/texture_learned_photograph_20261121.toml.
#   bash texture_learned_photograph_20261121.sh <results root> <data root> <out dir>
set -euo pipefail
RESULTS=$1; DATA=$2; OUT=$3; T=$(cd "$(dirname "$0")" && pwd)
python3 -B "$T/texture_learned_photograph.py" "$RESULTS/texture_recovery/evaluate/16f9d84088c7714b" \
    "$RESULTS/texture_recovery/init/a2902f6141e57d0e/poses.json" "$DATA/paper-data/defocus/14mm/texture-learning" "$OUT" \
    --roles="$RESULTS/texture_recovery/refine/270c783b87521b12/page.json"
( cd "$OUT" && sha256sum photograph.png meta.toml )
