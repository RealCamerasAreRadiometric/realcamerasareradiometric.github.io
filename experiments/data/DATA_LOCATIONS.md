# Where the data of the paper run is

The configs of record (calibration accuracy, vignetting, defocus) read two kinds of inputs. **The photographs** are published: the
author's folder `radiometric-camera-paper-data` on the university's MDS share, bound under `data_root()` as `paper-data/` (below).
**The derived inputs** of the baseline tier `rc_baselines` — the front ends' detections and calibrations, the decoded chrome
photographs, the flat fields and image-circle frames — are on the share beside the photographs since 2026-11-10, as
`derived/rc_baselines_2026-11-10/` in the data root's layout (below), and so are the targets the configs read (the paper's part of
`rc_calibration`: `geometric_accuracy/data/target/**`, as `derived/rc_calibration_2026-11-10/`) and the synthetic set of the
detector-bias experiment (`rc_synthetic`: the truth, the poses, the render state, the field stop, the renders, as
`derived/rc_synthetic_2026-11-10/`). Only the Thorlabs target's own photographs (`rc_thorlabs`, read by no config of the paper)
still exist only on the cluster node `rtx-01-mds`. This file says where everything is. `configs/data_artifacts.toml` `[[unpacked]]` lists every input path with its tier, the configs that
read it and its provenance.

## The photographs: the MDS share

| | |
|---|---|
| share | `//datarepo01.rz.uni-kiel.de/MDS_DATA`, folder `HPC_SHARE/tha/radiometric-camera-paper-data` (the author's; written 2026-11-10 with the author's account) |
| contents | 4362 Sony FX3 ARW in `geometric-accuracy/`, `vignetting/`, `defocus/` (one folder per lens and session), with `README.md` (what each folder is), `MANIFEST.tsv` (every file and its size) and `MD5SUMS` (`md5sum -c MD5SUMS` inside the folder checks a copy) |
| binding | the folder is `data_root()/paper-data`: on rtx-01-mds a link to the admin's mount of the share (`/cold/tha/radiometric-camera-paper-data`, read-only); elsewhere a mount of the share or a copy (`rclone copy` through rclone's SMB backend with your university account, then `md5sum -c`), linked or copied into the data root under that name |
| in the configs | every `rc_raw_photos` input is a path `paper-data/<experiment>/<lens>/<session>`; where a config uses a subset of a folder it says so beside the path — `exclude` (the Tamron 17 cm Siemens folder's three stray 14 mm photographs) or `frames` (every second frame of the Tamron panorama) — and that selection keys its runs (`src/drivers.jl` `INPUT_SELECTION_KEYS`, `selected_raw_dir`) |
| excluded | five photographs of the 14 mm Siemens-star session (DSC06045, DSC06052–DSC06055) are not in the published set and not in the view list of record |

```
VGTK_DATA_ROOT=<data root> …           # with <data root>/paper-data → the share's folder
```

## The derived inputs: `derived/rc_baselines_2026-11-10` on the share

| | |
|---|---|
| where | the same folder of the share, `radiometric-camera-paper-data/derived/rc_baselines_2026-11-10/` (written 2026-11-10 from the data root of the paper run; 1541 files, 16.5 GiB, of which 16.3 GiB are the five decoded chrome sets) |
| layout | the data root's: `chrome/*.photos` (the decoded chrome photographs, 16-bit linear PNG green copies), `geometric_accuracy/night_stars/data/<set>/` (Schöps's datasets, intrinsics and reports, the corner detections, OpenCV's and Kalibr's calibrations, the parametric fits, `views.txt`, the run logs; without the decoded calibration photographs), `geometric_accuracy/records/` (the camera records the refinements start from), `geometric_accuracy/detector_bias/data/v8/` (the synthetic set's front ends), `vignetting/{flat,image_circle,maps}` |
| binding | lay it over the data root: copy or link its top-level folders `chrome/`, `geometric_accuracy/`, `vignetting/` beside `paper-data/` |
| checks | `MANIFEST.tsv` (path, bytes, md5 of every file), `MD5SUMS` (`md5sum -c MD5SUMS` inside the folder), `README.md` |
| how it was made | `frontends/` beside it = the repository's `experiments/frontends_history/` as of tag `v0.10.0` (the folder left the working tree after that tag; a path `experiments/frontends_history/…` on this page is a path of that tree): the front ends' scripts as they ran, the tool versions, the view rules and the chains' logs (its README); each path's catalog entry names the configs that read it and its provenance |

## Its supplement: `rc_baselines_2026-11-12_supp` (tier `rc_baselines`)

| | |
|---|---|
| where | dgx01 `/raid/data/jpr/rc_baselines_2026-11-12_supp/` (`rc_baselines_supp_20261112`, manifest_sha256 `2b49c026adebe7e719389363cbe9d600d02fc9a01e4f24d6ed46c7ab957ef019`); its MDS mirror `radiometric-camera-paper-data/derived/rc_baselines_2026-11-12_supp/` is pending |
| holds | the panoramas' former camera records with a typed-in image circle as lens coverage, `geometric_accuracy/records/2710/{14mm,fisheye}_kb_fit_lens_coverage.json` (vgio/cameras/v5; retired 2026-11-15, read by no config: each sweep's camera is bounded by its measured image region, `src/image_region.jl`) |
| binding | lay it over the data root at the same relative paths, like `derived/rc_baselines_2026-11-10/`; its files are inputs of tier `rc_baselines` (the configs' tier vocabulary has no other name for them: each tier the configs name is a packed catalog version) |

## The targets: `derived/rc_calibration_2026-11-10` on the share

| | |
|---|---|
| where | `radiometric-camera-paper-data/derived/rc_calibration_2026-11-10/` (written 2026-11-10; 39 files, ≈ 505 MiB) |
| holds | `geometric_accuracy/data/target/**` of the data root — the 1 m Siemens-star and AprilGrid boards (designs, features, the signed-distance board), the A4 texture sheet `blobboard_a4_6eb` (its design inputs are in `derived/rc_baselines_2026-11-10/frontends/inputs/blobboard_a4/`), the Thorlabs R3L3S7N1 target record, the golden cases' blob boards; the paper's part of the tier `rc_calibration`, not the golden cases' other inputs |
| binding, checks | as above: lay it over the data root; `MANIFEST.tsv`, `MD5SUMS`, `README.md` |

## The synthetic set: `derived/rc_synthetic_2026-11-10` on the share

| | |
|---|---|
| where | `radiometric-camera-paper-data/derived/rc_synthetic_2026-11-10/` (written 2026-11-10) |
| holds | the tier `rc_synthetic` as the configs read it, version v8: `geometric_accuracy/detector_bias/data/{ds_truth,ds_poses_real,render_inputs}/` (the truth lens, the 100 poses, the render state) and `…/data/v8/{circle_field_stop, ds_aprilgrid/gt, ds_siemens/gt}/` (the field stop, both boards' 100 linear renders with their sRGB copies, truth reports, camera, image circle, state, trajectory); the front ends' outputs on the renders are in the baseline folder |
| how it was made | `scripts/make_detector_bias_inputs.jl` (the poses, the field stop), `scripts/detector_bias.jl` stage `prepare` with `[ground_truth] regenerate = "all"` (the renders; GPU) — both removed in rc-0116 with the synthetic detector bias (git history, tag v0.11.5); the golden calibration case still reads `fisheye_full` |
| binding, checks | as above: lay it over the data root; `MANIFEST.tsv`, `MD5SUMS`, `README.md` |

## On rtx-01-mds

| what | path |
|---|---|
| the data root of every config of record (`VGTK_DATA_ROOT`) | `/scratch/shared/tha/rc-data/vgtk-root-main` (read-only; `paper-data` → `/cold/tha/radiometric-camera-paper-data`, the derived inputs as below) |
| the runs the paper's assets were drawn from (`VGTK_RESULTS_ROOT`, keyed run directories) | `/scratch/shared/tha/rc-results/paper_2026-11-07/results` |
| the logs, the Slurm chains that ran them and the reports | `/scratch/shared/tha/rc-results/paper_2026-11-07/{logs,tools,reports}` |

Inside the data root, the folders the configs of record read (`[inputs]` of `configs/experiments/*.toml`):

| folder | contents | origin |
|---|---|---|
| `paper-data/geometric-accuracy/{14mm,tamron,fisheye}/…` | the Siemens-star, AprilGrid and chrome sessions of the calibration-accuracy experiment (the chrome sessions are read through their decodes below) | the MDS share (above) |
| `paper-data/vignetting/{14mm,tamron,fisheye}/…` | the large-aperture Siemens-star sessions, the diffuser frames and the indoor pure-rotation sweeps | the MDS share |
| `paper-data/defocus/{14mm,tamron,fisheye}/…` | the 14 mm texture experiment (calibration, texture-learning), the Tamron f/8 Siemens and chrome sessions, the fisheye f/4 set | the MDS share |
| `chrome/*.photos` | the decoded chrome-board photographs per set (linear green copies), the inputs of `thorlabs_distortion_fx3_*.toml` (tier `rc_baselines`: derived) | decoded from `paper-data/…/chrome*` (`scripts/py/convert_raw.py`, the green copies of the front-end chains) |
| `geometric_accuracy/data/target/` | the board records (`aprilgrid/board`, `aprilgrid/board_sdf_8ppm`, `siemens_1m/board`, `thorlabs_r3l3s7n1`) and the texture sheet (`blobboard_a4_6eb`) | `scripts/make_targets.jl`, the printed designs; the sheet by `experiments/frontends_history/scripts/sheet_build.py`; on the share under `derived/rc_calibration_2026-11-10/` |
| `geometric_accuracy/records/`, `geometric_accuracy/night_stars/data/` | the front ends' outputs of the real sets: Schöps's datasets and intrinsics, OpenCV and Kalibr detections and calibrations, the parametric fits (tier `rc_baselines`) | the front ends below (`experiments/frontends_history/`), on the cluster; on the share under `derived/rc_baselines_2026-11-10/` |
| `geometric_accuracy/detector_bias/data/` | the synthetic ground truth (`v8/`: renders, poses, the field stop; `ds_truth/`, `ds_poses_real/`, `render_inputs/`) and the detectors' and calibrations' outputs on it | `scripts/detector_bias.jl` stage `prepare` (`[ground_truth] regenerate`), `scripts/make_detector_bias_inputs.jl`, the front ends; on the share under `derived/rc_synthetic_2026-11-10/` (the truth and renders) and `derived/rc_baselines_2026-11-10/` (the front ends' outputs) |
| `vignetting/{flat,image_circle,maps}` | the diffuser flat fields, the image-circle frames per set and the exported vignetting maps | 28.10 and the sets' own photographs |

Folders in the data root no config of record reads (the raw sets the configs read before 2026-11-10 — `central_30cm*`, `14mmAprilGrid400_*`,
`fisheye10cmAprilGrid400_*`, `27.10.26/`, `28.10.26/`, `tamron*`, `texture_14mm_*`, `panorama_good/` — and `14mm175mmLargeApertureSiemens_*`,
`fisheye10cmSiemens_all`, `flatport_2026-11-06`, `texture_fisheye_*`, `texture_tamron_*`, `diag_2026-11-07`, `macro/`, `macro_defocus/`,
`vignetting/image_circle/_unused_2026-11-09`, older `records/` and `night_stars/data/` sets) are the published photographs' sources,
superseded attempts or experiments outside the paper; the catalog lists none of them.

## The front ends (frozen in `experiments/frontends_history/` of tag `v0.10.0`; their outputs are inputs of record)

Not drivers of this project: the chains that made the geometric starts and the chrome decodes ran by hand on the cluster
(2026-11-05 … 09); `experiments/frontends_history/` of tag `v0.10.0` holds them as they ran — drivers, scripts, container recipes, the chains' logs — with the
tool versions and the view rules in its README (`git show v0.10.0:experiments/frontends_history/README.md`; the folder left the working tree after that tag). The same folder is on the share beside the outputs (`frontends/`).

| front end | what ran |
|---|---|
| T. Schöps's `camera_calibration` (detection, central-generic, noncentral-generic) | puzzlepaint/camera_calibration commit 00ef4de (2023-06-30), the native build `/scratch/shared/tha/schoeps/camera_calibration/build/applications/camera_calibration/camera_calibration` on the cluster; `experiments/containers/schoeps.def` builds the same commit (CUDA 12.8), `experiments/src/frontends.jl` runs it (enroot) |
| Kalibr (ds-none, pinhole-radtan) | the docker image `mandulaj/kalibr:20.04` (digest `sha256:59a0451f…`) as the udocker container `kalibr` (`~/cluster/udocker`), driven by `experiments/frontends_history/scripts/kalibr/kalibr_calibrate_folder.py` (the tag ids 85–148 re-indexed onto the 8 × 8 target) |
| OpenCV AprilGrid detection and calibration, OpenCV's solvers on Schöps's star detections | cv2 5.0.0, numpy 2.2.6, Python 3.10.12: `experiments/frontends_history/scripts/{detect_aprilgrid,opencv_calibrate_aprilgrid,opencv_calibrate_schoeps}.py` |
| the parametric fits BC5 / OpenCV-14 / Kannala–Brandt / Double Sphere on Schöps's detections (`*_fit.json`) | `experiments/frontends_history/scripts/fit_lens_detections.jl` (the `experiments/` project of branch `paper-finalization`) |
| the Kannala–Brandt and Double Sphere starts on the Siemens sets | until 2026-11-13 the retired `scripts/make_parametric_starts.jl` (records `geometric_accuracy/records/<set>/*_start.json`, kept in the tier as history); since then the project's front ends, `experiments/scripts/frontends.jl` ([fits.<name>] of `configs/experiments/frontends.toml`: results, not data-root files) |
| BlobBoards (`find_boards`, the texture sheets' correspondences) | the package VisualGeometryToolkit/BlobBoards.jl, as the configs' provenance states |

## Regenerating on rtx-01-mds

    # a site profile of the machine first (VisualGeometryToolkitEval configs/sites/_template.toml; REPRODUCE.md, Adding your
    # machine), with the thread count the runs of record were keyed with: cpus_per_task = 16
    VGTK_DATA_ROOT=/scratch/shared/tha/rc-data/vgtk-root-main VGTK_RESULTS_ROOT=<a results root> bash experiments/scripts/reproduce/reproduce.sh --site=<its site profile>.toml --dry-run
    VGTK_DATA_ROOT=/scratch/shared/tha/rc-data/vgtk-root-main VGTK_RESULTS_ROOT=<its results root> julia -t 16 --project=experiments experiments/scripts/regenerate_paper.jl --gpu

(the runs of record were made with 16 threads per job, which keys them: a launch at another thread count recomputes them under new
keys; the node is shared, so every job goes through Slurm with bounded resources.)
