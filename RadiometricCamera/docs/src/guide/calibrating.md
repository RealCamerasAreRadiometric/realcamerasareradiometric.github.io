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
target), and the calibration is VisualGeometryToolkitCore's `solve` of it ([`RadiometricCalibration`](@ref): the optimizer and the
closed-form blocks' algorithm; the problem's structure, the calibration's [`CalibrationSettings`](@ref), and the run's start options
([`RunOptions`](@ref)) its data's, [`CalibrationPhotographs`](@ref) — no context; the stops its solve's — [`read_calibration`](@ref)
reads the run, the settings, the calibration and the stops from a run's TOML). The block below needs the reader's own session, so
it is not run here:

```julia
# Illustrative: needs a calibration session's records and its photographs (run.toml names them, [io] and [frontend]).
using RadiometricCamera, VisualGeometryToolkitCore
run, settings, alg, stops = read_calibration("run.toml"; set = ["levenberg_marquardt.stops.max_iter=15"])   # the run, the settings, the calibration, the solve's stops
g = start_geometry(run, settings)                                                 # the front end's geometric calibration
data = CalibrationPhotographs(g, start_photographs(run, g.names); run, settings)  # the geometry, its photographs' files, the start options
fit = solve(EstimationProblem(CalibrationEstimate, data), alg; stops)             # the start, then the joint LM over Θ̄
fit.u.camera                                                                      # the radiometric camera (lens, focus, r, V)
```

A SplineLens camera (the generic grid model) is refined per tile, any other camera per view in its own parameters. The device is
the data's (`CalibrationPhotographs(geometry, photographs; run, settings, device)`; [`default_backend`](@ref) by default).
A run's outputs are its driver's (ruling R1): the per-iteration checkpoints and the optimizer's log are the `callback` the driver
hands the refinement's solve from the started problem ([`compute_calibration_start`](@ref RadiometricCamera.compute_calibration_start):
`prob = RadiometricCamera.compute_calibration_start(alg, data)`; [`calibration_checkpoints`](@ref RadiometricCamera.calibration_checkpoints):
`solve(prob, alg.refinement; stops, callback = RadiometricCamera.calibration_checkpoints(prob, run))`, Core's
`callback(state::LmIterationState)` contract), and a resume is a problem built from a finished state ([`load_calibration_state!`](@ref RadiometricCamera.load_calibration_state!)).
`JULIA_NUM_THREADS=auto`: the CPU backend is multithreaded over pixels.

## Run the refinement driver

The driver `bin/refine.jl` does this on real sessions (two front-ends: BlobBoards sessions and Schöps runs on a printed page)
with the run's [`RunOptions`](@ref), the calibration's [`CalibrationSettings`](@ref), the optimizer and the solve's stops read from one TOML file ([`read_calibration`](@ref)) (`bin/refine.jl --config=<run.toml> [--set table.key=value …]`;
`configs/default.toml` lists every key with its default; a file without `config_version`, an RC ≤ 0.3 one with its `[lm]` table,
is read through the converter, [`RC_CONFIG_VERSIONS`](@ref): `[lm]` is the `[optimizer]` table, and a version-3 file's `[optimizer]`
stops are `[levenberg_marquardt.stops]`, its `log` `[io] optimizer_log`). It writes `grid.json` / `lens.json` (a grid with its start's `initialization_region.json` beside it, so that every grid it writes, the checkpoints included, starts another run), `state.json`, `radiometric_camera.json`, `radiometric_target.json`, the poses,
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
photographs' observations, the closed-form start: `init(EstimationProblem(CalibrationEstimate, CalibrationPhotographs(geometry, photographs; run, settings)),
algorithm; stops)` — its started problem alone [`compute_calibration_start`](@ref RadiometricCamera.compute_calibration_start) — the refinement stage's `init` — an `EstimationCache` of a [`CalibrationCache`](@ref) of `RefineProblem(f, u0)`: the data [`CalibrationData`](@ref), its
`calibration_settings` the problem's structure, and the parameters [`CalibrationParameters`](@ref)), [`save_calibration_start`](@ref) writes it as a start record
(`bin/refine.jl --stage=start`), [`load_calibration_start`](@ref) builds the refinement's problem from one (`--start=<dir>`),
[`load_calibration_state!`](@ref RadiometricCamera.load_calibration_state!) continues it from a finished state (`resume.state`; `sweep = resume.sweep` the driver's closed-form
sweep on it), `solve!` of the cache — or `solve(problem, algorithm.refinement; stops, callback)` of a loaded problem
([`RadiometricRefinement`](@ref), the callback the run's [`calibration_checkpoints`](@ref RadiometricCamera.calibration_checkpoints)) — runs the sweeps and the LM (a `FitResult` of a [`CalibrationEstimate`](@ref), [`RadiometricCalibration`](@ref)), and
[`save_radiometric_calibration`](@ref) writes the records; one `solve` of the calibration's problem is the start and the refinement in one call. A run
with neither flag computes its start, writes it (into `io.out/start/`, or `--start-out=<dir>`) and refines from it in the same
process, the photographs read once (the start's streamed observations handed to `load_calibration_start`).

Every spline component's smoothness is one table, dispatched by its `type`: `[grid.smoothness] type = "pspline", order = 3,
weight = 0.1` (the third differences of the grid's correction; `"bending"` or `"none"`; `weight = "gcv"` chooses β by
generalized cross-validation at the start: the weight is then VisualGeometryToolkitEstimation's `GeneralizedCrossValidation()`), `[vignetting.smoothness] type = "pspline", order = 3, weight = 0.0` and
`[focus.smoothness] type = "bending", weight = 0.0`. `grid.prior`, `grid.prior_weight`, `vignetting.smoothness_weight` and
`focus.prior` are retired and refused by name.

## The walkthrough

`examples/radiometric_calibration.jl` is the commented walkthrough (and the source of the notebook beside it): a geometric
central model three ways in, the radiometric model around it, the refinement's problem and its masks, the calibration on the GPU, and the
night-stars validation. It reads the paper project's published data (`VGTK_DATA_ROOT`, the tier `rc_calibration`);
`examples/README.md` says how to run it.
