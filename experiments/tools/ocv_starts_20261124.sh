#!/bin/bash
# ocv_starts_20261124.sh: the six OpenCV starts of the Sigma 14 mm sets — Brown–Conrady, OpenCV-14 and Kannala–Brandt on the Siemens
# stars at ≈ 30 cm and on the AprilGrid — made as the Tamron sets' OpenCV records of record were (frontends_history/drivers of tag
# v0.10.0): fe_si7.sh — opencv_calibrate_schoeps.py on Schöps's dataset.bin, --texfeat-size 4000 4000 --page-mm 1000 1000, MINF = 20 —
# and fe_ag.sh — opencv_calibrate_aprilgrid.py on the green copies of the photographs, the corners its own ArucoDetector's; the
# models pinhole and ocv14 from OpenCV's own start, the fisheye model with --f-init, this lens's 1755 px (paper_fe_chain.sh:39);
# then ocv_record.jl, for the fisheye model cvfisheye_record.jl:
#   the Siemens stars on frontends schoeps/57dd94a82ac869ea ([schoeps.central30cm_14mm_siemens_w20])
#   the AprilGrid on the 408 green copies of the paper night's front end (fe_ag.sh's board_srgb_green, the images its Schöps start was made on)
# OpenCV 5.0.0 (numpy 2.2.6, Python 3.10.12: the versions of the records of record of 2026-11-07 … 09): the two tools reproduce the
# Tamron records' calibrations on their own inputs (configs/inputs/opencv_starts_20261124.toml, evidence).
# The records: <data root>/geometric_accuracy/records/{central30cm_all,aprilgrid400_all}/ (rc_baselines_2026-11-24).
#   bash ocv_starts_20261124.sh <dir with si30_dataset.bin, texture_features.bin, ag30_green/> <out dir>
set -euo pipefail
IN=$1; OUT=$2; T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"/central30cm "$OUT"/aprilgrid400/{pinhole,ocv14,fisheye}
python3 "$T/opencv_calibrate_schoeps.py" --dataset "$IN/si30_dataset.bin" --texfeat "$IN/texture_features.bin" --texfeat-size 4000 4000 --page-mm 1000 1000 --model pinhole --min-features 20 --out "$OUT/central30cm/pinhole" 2>&1 | tee "$OUT/central30cm/pinhole.log"
python3 "$T/opencv_calibrate_schoeps.py" --dataset "$IN/si30_dataset.bin" --texfeat "$IN/texture_features.bin" --texfeat-size 4000 4000 --page-mm 1000 1000 --model ocv14 --min-features 20 --out "$OUT/central30cm/ocv14" 2>&1 | tee "$OUT/central30cm/ocv14.log"
python3 "$T/opencv_calibrate_schoeps.py" --dataset "$IN/si30_dataset.bin" --texfeat "$IN/texture_features.bin" --texfeat-size 4000 4000 --page-mm 1000 1000 --model fisheye --f-init 1755 --min-features 20 --out "$OUT/central30cm/fisheye" 2>&1 | tee "$OUT/central30cm/fisheye.log"
python3 "$T/opencv_calibrate_aprilgrid.py" --images "$IN/ag30_green" --out "$OUT/aprilgrid400/pinhole" --model pinhole --jobs 8 2>&1 | tee "$OUT/aprilgrid400/pinhole.log"
cp "$OUT/aprilgrid400/pinhole/detections.json" "$OUT/aprilgrid400/ocv14/"; cp "$OUT/aprilgrid400/pinhole/detections.json" "$OUT/aprilgrid400/fisheye/"   # one detection for the three models, as fe_ag.sh
python3 "$T/opencv_calibrate_aprilgrid.py" --images "$IN/ag30_green" --out "$OUT/aprilgrid400/fisheye" --model fisheye --f-init 1755 --jobs 8 2>&1 | tee "$OUT/aprilgrid400/fisheye.log"
python3 "$T/opencv_calibrate_aprilgrid.py" --images "$IN/ag30_green" --out "$OUT/aprilgrid400/ocv14" --model ocv14 --jobs 8 2>&1 | tee "$OUT/aprilgrid400/ocv14.log"   # last: about an hour on the 404 views
# then, in the tree (julia --project=experiments), <records> = <data root>/geometric_accuracy/records:
#   experiments/tools/ocv_record.jl <central30cm/pinhole/calibration.json> <records>/central30cm_all/14mm_central30cm_si_all_green_bc5_cv2.json 14mm_central30cm_si_all_green_bc5_cv2 '<note>'
#   experiments/tools/ocv_record.jl <central30cm/ocv14/calibration.json> <records>/central30cm_all/14mm_central30cm_si_all_green_ocv14_cv2.json 14mm_central30cm_si_all_green_ocv14_cv2 '<note>'
#   experiments/tools/cvfisheye_record.jl <central30cm/fisheye/calibration.json> <records>/central30cm_all/14mm_central30cm_si_all_green_kb_cv2.json 14mm_central30cm_si_all_green_kb_cv2 '<note>'
#   experiments/tools/ocv_record.jl <aprilgrid400/pinhole/calibration.json> <records>/aprilgrid400_all/ag400_all_green_bc5_cv2.json ag400_all_green_bc5_cv2 '<note>'
#   experiments/tools/ocv_record.jl <aprilgrid400/ocv14/calibration.json> <records>/aprilgrid400_all/ag400_all_green_ocv14_cv2.json ag400_all_green_ocv14_cv2 '<note>'
#   experiments/tools/cvfisheye_record.jl <aprilgrid400/fisheye/calibration.json> <records>/aprilgrid400_all/ag400_all_green_kb_cv2.json ag400_all_green_kb_cv2 '<note>'
