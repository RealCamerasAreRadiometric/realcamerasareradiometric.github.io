docs-pass: c2707ab27076f91cddd3ff24c17a5220fd550558
landing: site/ of this repository (site/src/index.md, deployed by the Site workflow with the package manual under /docs/): the install paragraph (General, no registry add), the example's `using CUDA`, `sz`, `CameraCalibrationMat(412.0, sz)` and the keyword camera changed in this pass

# RadiometricCamera 0.3.0 docs pass (julia-docs-pass), on the release tree; the first General registration

0.3.0 is breaking: the GPU is the default device (`default_backend()`: VisualGeometryToolkitCore's `default_device()`, a functional
GPU once `using CUDA` has loaded its package, else the CPU with a warning once per session). The package builds on
VisualGeometryToolkitCore 0.13.3, Estimation 0.12.2, Features 0.9.11, IO 0.11.2 and VisualRadiometryToolkitCore / IO 0.1.1.

[x] Public API since the 0.2.x docs pass (c7bd5f9): one name added, `default_backend` (public, documented, in api/pipeline.md
    through options/device.jl); none removed. `compute_mask`'s `backend` is optional; every docstring that named the CPU default
    (`CalibrationState`, `render_view`, `compute_mask`, `start_calibration`, `load_calibration_start`, `calibrate`,
    `measure_image_region`, `start_grid`, `exact_start`, `check_coverage`, `calibrated_folds`, `schoeps_*`, `edge_strength`) names
    `default_backend`; `rc_backend`'s docstring states that its GPU path ends in it (one CPU-fallback warning).
[x] Examples (README, docs/src/index.md, site/src/index.md — one example shown three times; guide/getting-started.md;
    guide/calibrating.md): `using CUDA`; no `rc_backend("cpu")`; one `sz = Size2(320, 240)`; `CameraCalibrationMat(412.0, sz)`;
    `Rect(extent(geometric))`; the keyword `RadiometricCameraModel(geometric; focus, aperture_radius, vignetting)`; image centres
    by `centroid(Rect(...))`; the checks hold to Float32 on a GPU (`cost(s)[2] < 1e-4`, `isapprox(…; rtol = 1e-4)`).
[x] Install text (README, docs/src/index.md, guide/setup.md, site/src/index.md, the repository README, experiments/README.md and
    REPRODUCE.md): from Julia's General registry, Julia 1.13 or later, `pkg> add RadiometricCamera …`, no `registry add`. The CI and
    site workflows add no CauRegistry; the run of record verifies the version in General. CauRegistry remains only in the CHANGELOG.
[x] guide/testing.md lists the environment variables the suite and its harnesses read (the package reads none).
[x] Gates: the strict docs build (checkdocs = :public, doctests, every @example, linkcheck, no warnonly) green on the CPU and on a
    B200 (tutorial check `(true, true)` on both); `docs` (`test_docs`: undocumented public names, detached docstrings, signature
    mismatches), `README` (fresh environment from the install line), Aqua and naming items pass.
[x] general_preflight.sh: every automatic check passes but VisualRadiometryToolkitIO's General registration (#177670, merging) and
    a CI run on main (the release commit is merged to main before registration). Manual items: the name has no confusable General
    neighbour (closest Damerau-Levenshtein distance 8); SVBSDF and REPRODUCE are an acronym and a file name; native libraries from
    JLLs (LibRaw_jll); the environment variables are documented.
