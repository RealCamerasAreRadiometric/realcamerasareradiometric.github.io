#!/usr/bin/env bash
# The raw photographs (248 ARW / CR2 files, 6.2 GB; the tier rc_raw_photos of experiments/configs/data_artifacts.toml) from the Hugging Face
# dataset they were kept in until 2026-10-29, radiometric-camera/radiometric-camera-data (private: ask to be added to the Hugging Face
# organization radiometric-camera), into the data root at the paths of raw_photos.sha256 beside this script (relative to the data root;
# the dataset stores them under experiments/, the retired tree's prefix). The checksums are the Git LFS ids the files had in
# radiometric-camera c3766d7a; every file is verified. Does nothing when all files are present and correct. The published route is the
# artifact tier (scripts/fetch_data.jl --only=rc_raw_photos_<date>); this script is the author's fallback while the record is unpublished.
#   VGTK_DATA_ROOT=<data root> bash experiments/scripts/py/fetch_raw_photos.sh   needs the `hf` CLI (pip install -U huggingface_hub) and
#                                                                               `hf auth login` with read access
#   HF_REVISION=<commit> / HF_DATASET=<id> / HF=<path to hf> override the defaults
set -euo pipefail
MANIFEST=$(cd "$(dirname "$0")" && pwd)/raw_photos.sha256
ROOT=${VGTK_DATA_ROOT:?set VGTK_DATA_ROOT to the data root the photographs go into}
DATASET=${HF_DATASET:-radiometric-camera/radiometric-camera-data}
REVISION=${HF_REVISION:-cb9c70691063505205c07a1684c841d40e788ebb}   # the dataset commit of 2026-10-26 (all 248 files verified against the manifest)
HF=${HF:-hf}
mkdir -p "$ROOT"; cd "$ROOT"
if sha256sum --quiet -c "$MANIFEST" > /dev/null 2>&1; then echo "all $(wc -l < "$MANIFEST") raw photos present and verified under $ROOT"; exit 0; fi
STAGE=$(mktemp -d "$ROOT/.hf_download.XXXXXX")
"$HF" download "$DATASET" --repo-type dataset --revision "$REVISION" --local-dir "$STAGE" --include "*.ARW" --include "*.CR2"
(cd "$STAGE/experiments" && find . -type f) | while read -r f; do mkdir -p "$(dirname "$f")"; mv -n "$STAGE/experiments/$f" "$f"; done
rm -rf "$STAGE"
cp "$MANIFEST" rc_raw_photos.sha256
sha256sum --quiet -c "$MANIFEST" && echo "$(wc -l < "$MANIFEST") raw photos downloaded and verified under $ROOT"
