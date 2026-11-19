#!/bin/bash
# ocv_starts_20261117.sh: the two OpenCV starts of 2026-11-17 (owner GO 05:00, "run them ourselves"), made as the Siemens-set OpenCV
# records of record were (frontends_history/drivers/fe_si7.sh of tag v0.10.0: opencv_calibrate_schoeps.py on Schöps's dataset.bin,
# --texfeat-size 4000 4000 --page-mm 1000 1000, MINF = 20 as every Siemens record's log states; then ocv_record.jl):
#   the fisheye 30 cm OpenCV-14 on frontends schoeps/43a409607a28914d ([schoeps.fisheye30cm_siemens_w30])
#   the zoom 48 mm f/10 Brown–Conrady (OpenCV's default model) on frontends schoeps/732112ae6c6130a6 ([schoeps.zoom48_f10_siemens_w20])
# OpenCV 4.12.0 (the desktop's python3; dgx01 has no cv2): the tool reproduces the record of record tamron17cm_si_all_green_ocv14_cv2
# (OpenCV 5.0.0) on its own dataset.bin at rms 0.942937 px (equal to 6 digits), its pixel map within 0.027 px (max over the sensor).
# The records: <data root>/geometric_accuracy/records/{fisheye30cm_si_all,zoom48_f10_si_all}/ (rc_baselines_2026-11-17 on dgx01).
#   bash ocv_starts_20261117.sh <dir with fe30_dataset.bin, zoom48_dataset.bin, texture_features.bin> <out dir>
set -euo pipefail
IN=$1; OUT=$2; T=$(cd "$(dirname "$0")" && pwd)
python3 "$T/opencv_calibrate_schoeps.py" --dataset "$IN/fe30_dataset.bin" --texfeat "$IN/texture_features.bin" --texfeat-size 4000 4000 --page-mm 1000 1000 --model ocv14 --min-features 20 --out "$OUT/fisheye30cm/ocv14"
python3 "$T/opencv_calibrate_schoeps.py" --dataset "$IN/zoom48_dataset.bin" --texfeat "$IN/texture_features.bin" --texfeat-size 4000 4000 --page-mm 1000 1000 --model pinhole --min-features 20 --out "$OUT/zoom48/pinhole"
# then, in the tree on dgx01 (julia --project=experiments):
#   experiments/tools/ocv_record.jl <ocv14/calibration.json> <records>/fisheye30cm_si_all/fisheye30cm_si_all_green_ocv14_cv2.json fisheye30cm_si_all_green_ocv14_cv2 '<note>'
#   experiments/tools/ocv_record.jl <pinhole/calibration.json> <records>/zoom48_f10_si_all/zoom48_f10_si_all_green_bc5_cv2.json zoom48_f10_si_all_green_bc5_cv2 '<note>'
