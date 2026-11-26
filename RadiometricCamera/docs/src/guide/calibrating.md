# Calibrate a camera

```@meta
CurrentModule = RadiometricCamera
```

## Plug in a geometric model

The radiometric camera needs one thing from a geometric model: its chief ray, VisualRadiometryToolkitCore's
`chief_ray(camera, 𝐩)` (one of its `CONTRACT_GENERICS`), the ray direction of the sensor point 𝐩 in the COLMAP pixels of the image
(𝓘 = [0, W] × [0, H], VisualGeometryToolkitCore's `Rect(sensor)`), plus [`in_domain`](@ref) and, to be refined,
VisualGeometryToolkitCore's parameter protocol. For any VisualGeometryToolkitCore camera the chief ray is its `backproject(camera, 𝐩)`;
a camera type of another package answers `chief_ray` with a method.

Any VisualGeometryToolkitCore lens plugs into the `RadiometricCameraModel` as Core's pixel camera — a `BoundedCamera` of
`CameraModel(LogicalIntrinsics(K), lens)`, K in the image's pixels (isbits: it travels into the GPU kernels as it is) — whose
back-projection is Core's `backproject` and whose parameters are Core's parameter protocol; the calibration refines it per view in its own
parameters (Core's `ParameterManifold` with the search box [`lm_box`](@ref) and the image-size coordinate scale
[`lm_scale`](@ref)). Any camera is sampled into the grid model, VisualGeometryToolkitCore's SplineLens camera (`CameraModel(K, SplineLens)`), by
[`GridStart`](@ref) (`solve(EstimationProblem(SplineLens, camera), GridStart(; lattice = ImageLattice(G_x, G_y))).u`) and refined per tile — the route of the paper's generic model (`lm_route`: the camera's type
picks it).

## Calibrate a session

A calibration session stored as VisualGeometryToolkitIO records (a camera, the board's target and the trajectory of its
frames) and its photographs give the start's geometry ([`start_geometry`](@ref): the camera, the poses, the observations and the
target), and the calibration is VisualGeometryToolkitCore's `solve` of it ([`RadiometricCalibration`](@ref): the options, the
optimizer; the stops its solve's, [`read_calibration`](@ref) reads all three from a run's TOML). The block below needs the reader's
own session, so it is not run here:

```julia
# Illustrative: needs a calibration session's records and its photographs (run.toml names them, [io] and [frontend]).
using RadiometricCamera, VisualGeometryToolkitCore
alg, stops = read_calibration("run.toml"; set = ["optimizer.max_iter=15"])   # the options, the optimizer, the solve's stops
g = start_geometry(alg.options)                                               # the front end's geometric calibration
fit = calibrate(g, start_photographs(alg.options, g.names), alg; stops)      # the start, then the joint LM over Θ̄
fit.u.camera                                                                  # the radiometric camera (lens, focus, r, V)
```

A SplineLens camera (the generic grid model) is refined per tile, any other camera per view in its own parameters. The device is
the problem's (`EstimationProblem(CalibrationEstimate, data, (; device))`; [`default_backend`](@ref) by default).
`JULIA_NUM_THREADS=auto`: the CPU backend is multithreaded over pixels.

## Run the refinement driver

The driver `bin/refine.jl` does this on real sessions (two front-ends: BlobBoards sessions and Schöps runs on a printed page)
with the options of [`PipelineOptions`](@ref), the optimizer and the solve's stops read from one TOML file ([`read_calibration`](@ref)) (`bin/refine.jl --config=<run.toml> [--set table.key=value …]`;
`configs/default.toml` lists every key with its default; a file without `config_version`, an RC ≤ 0.3 one with its `[lm]` table,
is read through the converter, [`RC_CONFIG_VERSIONS`](@ref): `[lm]` is the `[optimizer]` table). It writes `grid.json` / `lens.json` (a grid with its start's `initialization_region.json` beside it, so that every grid it writes, the checkpoints included, starts another run), `state.json`, `radiometric_camera.json`, `radiometric_target.json`, the poses,
`parameters.json` and the run's options (`pipeline_config.toml`), and, with `[io] checkpoint_each = "on"`, a lens / poses /
state file after every iteration (the convergence monitors of the experiments read them). It uses the GPU when CUDA is
functional; `[compute] backend = "cpu"` forces the CPU (the default device of a [`RadiometricCalibration`](@ref)'s start and of [`load_calibration_start`](@ref)
is [`default_backend`](@ref): the GPU when one is functional, else the CPU with a warning).

A run on RAW photographs names the camera's sensor characterisation: `[camera] sensor_characterization` is the path of the
body's `vgio/sensor_characterization` record (VisualGeometryToolkitCore's `SensorCharacterization`, read by
VisualGeometryToolkitIO); a named path that does not exist is refused. `[camera] sensor_model`
([`sensor_model_mode`](@ref RadiometricCamera.sensor_model_mode)) decides its use: `"auto"` (the default) uses a record present
and reads the file's black without one; `"on"` requires one; `"off"` does not use a record present, with one loud warning
([`warn_sensor_model_off`](@ref RadiometricCamera.warn_sensor_model_off)) — estimation and prediction may then disagree. The
retired `"none"` is refused, naming `sensor_model = "off"`. Each photograph's
`SensorModel` is resolved from it by its own EXIF capture settings ([`photo_set`](@ref RadiometricCamera.photo_set)): its black is the photographs'
black, its defects are masked out of the fit, and when its noise has a variance law (a gain) every residual and Jacobian row is
divided by the measured value's σ (whitened least squares; Core's `residuals` and `cost` are then the whitened ones,
[`photometric_residuals`](@ref) the unweighted Φ − ϕ). A run without a record keeps the file's black and the rows as measured. There is no
weighting option: the sensor model decides. `parameters.json` records the key and, per photograph, the resolver's reason and
the model's identity ([`sensor_record`](@ref RadiometricCamera.sensor_record)). Decoded (PNG) photographs have no sensor model: naming a record for them is
refused.

Its stages are the API: [`read_calibration`](@ref) reads the TOML, [`start_geometry`](@ref) reads the front end's geometric
calibration through VisualGeometryToolkitIO, the calibration's `init` lifts it into the refinement's problem (the masks, the
photographs' observations, the closed-form start: `init(EstimationProblem(CalibrationEstimate, (geometry, photographs)),
algorithm; stops)`, the refinement stage's `init` — an `EstimationCache` of a [`CalibrationCache`](@ref) of `RefineProblem(f, u0)`: the data [`CalibrationData`](@ref), the parameters
[`CalibrationParameters`](@ref)), [`save_calibration_start`](@ref) writes it as a start record
(`bin/refine.jl --stage=start`), [`load_calibration_start`](@ref) builds the refinement's problem from one (`--start=<dir>`),
`solve!` of the cache — or `solve(problem, algorithm.refinement; stops)` of a loaded problem ([`RadiometricRefinement`](@ref)) —
runs the sweeps and the LM (a `FitResult` of a [`CalibrationEstimate`](@ref), [`RadiometricCalibration`](@ref)), and
[`save_radiometric_calibration`](@ref) writes the records; [`calibrate`](@ref) is the start and the refinement in one call. A run
with neither flag computes its start, writes it (into `io.out/start/`, or `--start-out=<dir>`) and refines from it in the same
process, the photographs read once (the start's streamed observations handed to `load_calibration_start`).

Every spline component's smoothness is one table, dispatched by its `type`: `[grid.smoothness] type = "pspline", order = 3,
weight = 0.1` (the third differences of the grid's correction; `"bending"` or `"none"`; `weight = "gcv"` chooses β by
generalized cross-validation at the start), `[vignetting.smoothness] type = "pspline", order = 3, weight = 0.0` and
`[focus.smoothness] type = "bending", weight = 0.0`. `grid.prior`, `grid.prior_weight`, `vignetting.smoothness_weight` and
`focus.prior` are retired and refused by name.

## The walkthrough

`examples/radiometric_calibration.jl` is the commented walkthrough (and the source of the notebook beside it): a geometric
central model three ways in, the radiometric model around it, the refinement's problem and its masks, the calibration on the GPU, and the
night-stars validation. It reads the paper project's published data (`VGTK_DATA_ROOT`, the tier `rc_calibration`);
`examples/README.md` says how to run it.
