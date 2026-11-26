# Set up the data and the hardware

```@meta
CurrentModule = RadiometricCamera
```

The package installs as the [Installation](../index.md#Installation) says, from Julia's General registry.
This page is for what the package does not install: the hardware a calibration runs on, and the data of the experiments.

## Clone

The repository holds the package (`RadiometricCamera/`), the experiments that reproduce the paper (`experiments/`), the
manuscript and the project page, and no data. Its environment resolves from Julia's General registry:

```bash
git clone https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io.git
cd realcamerasareradiometric.github.io
julia +1.13 --project=RadiometricCamera -e 'using Pkg; Pkg.instantiate()'
```

## What runs where

| piece | needed for |
|---|---|
| Julia 1.13 | everything |
| a CUDA GPU with ≥ 20 GiB free (optional) | the `:gpu` test items ([Run the tests](testing.md)) and fast refinements; the CPU backend runs everything, more slowly (`JULIA_NUM_THREADS=auto`: it is multithreaded over pixels) |
| a calibration session (VisualGeometryToolkitIO records and the photographs) | [Calibrate a camera](calibrating.md) on one's own camera |
| the published data of the experiments | the paper's experiments, the example walkthrough `examples/radiometric_calibration.jl` and the end-to-end checks of [Check a change](testing.md#Check-a-change); not the package tests |

The package tests need no data: their fixtures are committed under `test/fixtures/`.

## Data

The experiments' inputs — the raw photographs, the synthetic renders, the baseline calibrations of the external front ends
(Schöps et al.'s pipeline, Kalibr, OpenCV, BlobBoards), the targets and the calibration cases — are the paper project's published
data, tiers of lazy Julia artifacts catalogued in `experiments/configs/data_artifacts.toml`. `experiments/REPRODUCE.md` says
how to fetch them (`experiments/scripts/fetch_data.jl`) and how to bind a copy of one's own (`VGTK_DATA_ROOT`); the
package's checks name the tier they read (`rc_calibration`).
