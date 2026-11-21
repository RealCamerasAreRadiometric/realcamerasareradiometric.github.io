# Calibrate a camera

```@meta
CurrentModule = RadiometricCamera
```

## Plug in a geometric model

The radiometric camera needs one thing from a geometric model: Core's `backproject(camera, 𝐩)`, the ray direction of the sensor
point 𝐩 in the COLMAP pixels of the image (𝓘 = [0, W] × [0, H], Core's `Rect(sensor)`), plus [`in_domain`](@ref) and, to be
refined, VisualGeometryToolkitCore's parameter protocol.

Any VisualGeometryToolkitCore lens plugs into the `RadiometricCameraModel` as Core's pixel camera — a `BoundedCamera` of
`CameraModel(LogicalIntrinsics(K), lens)`, K in the image's pixels (isbits: it travels into the GPU kernels as it is) — whose
back-projection is Core's `backproject` and whose parameters are Core's parameter protocol; [`run_lm!`](@ref) refines it in its own
parameters (Core's `ParameterManifold` with the search box [`lm_box`](@ref) and the image-size coordinate scale
[`lm_scale`](@ref)). Any camera is sampled into the grid model, VisualGeometryToolkitCore's SplineLens camera (`CameraModel(K, SplineLens)`), by
[`grid_model`](@ref)`(camera, G_x, G_y)` and refined per tile by [`run_lm_grid!`](@ref) — the route of the paper's generic model.

## Calibrate a session

A calibration session stored as VisualGeometryToolkitIO records (a camera, the board's target and the trajectory of its
frames) and its photographs give the board, the camera's starting lens and focus surface and the photographs' poses
([`load_session`](@ref)). The block below needs the reader's
own session, so it is not run here:

```julia
# Illustrative: needs a calibration session's records and its photographs.
using RadiometricCamera, VisualGeometryToolkitCore
using VisualRadiometryToolkitCore: Material, geometric
using RadiometricCamera: RenderSettings, load_session, run_lm_grid!, save_grid, solve_linear_blocks!, sphere_radius, 𝔟
ds = load_session("session/records", "session/images"; lens = :grid, grid_size = (35, 25), focus_init = :board_center, downsample = 8)
b = 𝔟(ds.board)
r_SH = sphere_radius(b, [p.t for p in ds.poses])
W, H = Tuple(extent(ds.lens₀))                    # the image is the camera's
camera = RadiometricCameraModel(ds.lens₀, ds.focus₀, ds.r₀, RadialVignetting(16, Point2(W / 2, H / 2), hypot(W, H) / 2))
s = CalibrationState(ds.board, RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 8), camera, ds.poses, Material(),
                     EnvironmentLight(SphericalHarmonicMap(1.0, 2), b, r_SH), ds.targets; δ = 0.01)   # VisualRadiometryToolkitCore's light
solve_linear_blocks!(s)              # closed-form Θ_l, Θ_v, μ
run_lm_grid!(s; maxiters = 15)       # the joint LM over Θ̄ (poses, grid, focus, r, material, V, lighting)
save_grid("grid.json", geometric(s))
```

`lens = :grid` samples the session's camera into the generic grid model; `lens = :camera` keeps the pixel camera, which
[`run_lm!`](@ref) refines instead. The GPU backend is opt-in: `backend = CUDA.CUDABackend()` of [`CalibrationState`](@ref).
`JULIA_NUM_THREADS=auto`: the CPU backend is multithreaded over pixels.

## Run the refinement driver

The driver `bin/refine.jl` does this on real sessions (two front-ends: BlobBoards sessions and Schöps runs on a printed page)
with the options of [`PipelineOptions`](@ref) read from one TOML file (`bin/refine.jl --config=<run.toml> [--set table.key=value …]`;
`configs/default.toml` lists every key with its default). It writes `grid.json` / `lens.json` (a grid with its start's `initialization_region.json` beside it, so that every grid it writes, the checkpoints included, starts another run), `state.json`, `radiometric_camera.json`, `radiometric_target.json`, the poses,
`parameters.json` and the run's options (`pipeline_config.toml`), and, with `[io] checkpoint_each = "on"`, a lens / poses /
state file after every iteration (the convergence monitors of the experiments read them). It uses the GPU when CUDA is
functional; `[compute] backend = "cpu"` forces the CPU ([`start_calibration`](@ref)'s and [`load_calibration_start`](@ref)'s
default device is Core's `default_device()`, GPU first).

A run on RAW photographs names the camera's sensor characterisation: `[camera] sensor_characterisation` is the path of the
body's `vgio/sensor_characterisation` record (VisualGeometryToolkitCore's `SensorCharacterisation`, read by
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

Its stages are the API: [`pipeline_options`](@ref) reads the TOML, [`start_geometry`](@ref) reads the front end's geometric
calibration through VisualGeometryToolkitIO, [`start_calibration`](@ref) lifts it into a radiometric state (the masks, the
photographs' observations, the closed-form start), [`save_calibration_start`](@ref) writes it as a start record
(`bin/refine.jl --stage=start`), [`load_calibration_start`](@ref) builds the refinement's state from one (`--start=<dir>`),
[`refine_calibration!`](@ref) runs the sweeps and the LM (a `FitResult` of a [`CalibrationEstimate`](@ref)), and
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
central model three ways in, the radiometric model around it, the state and its masks, the calibration on the GPU, and the
night-stars validation. It reads the paper project's published data (`VGTK_DATA_ROOT`, the tier `rc_calibration`);
`examples/README.md` says how to run it.
