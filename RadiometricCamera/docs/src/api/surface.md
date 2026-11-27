# The API surface

RadiometricCamera exports its headline API only (`src/api.jl`, the one file that states the surface). Everything else a user may
reach is `public`: write it qualified (`RadiometricCamera.RenderSettings`) or import it by name (`using RadiometricCamera: RenderSettings`).

## Exported

<!-- the suite checks this list equals `names(RadiometricCamera)` (test/unit/api_surface_tests.jl) -->
- [`CalibrationPhotographs`](@ref) — the data of a calibration from scratch: its [`StartGeometry`](@ref), the photographs' files, the run's start options, the [`CalibrationSettings`](@ref) (the problem's structure) and the device
- [`RadiometricCalibration`](@ref) — the calibration, an algorithm of VisualGeometryToolkitCore's `solve` / `init` / `solve!` on `EstimationProblem(CalibrationEstimate, CalibrationPhotographs(geometry, photographs; run, settings))` (no context): `init` lifts the geometric calibration into the refinement's problem (the start; its cache the refinement stage's `init`, holding a [`CalibrationCache`](@ref)), `solve!` refines it (a `FitResult`)
- [`RadiometricRefinement`](@ref) — the refinement, a refine of VisualGeometryToolkitCore's `solve` on `RefineProblem(f, u0)` ([`calibration_problem`](@ref), [`load_calibration_start`](@ref)): the parameters `u0` refined in place on the data `f` (the problem's structure its `calibration_settings`); its solve keywords `stops` and `callback` (Core's `LmIterationState` contract)
- [`StartGeometry`](@ref), [`start_geometry`](@ref), [`start_photographs`](@ref) — the start's inputs: the geometric calibration and the photographs
- [`save_calibration_start`](@ref), [`load_calibration_start`](@ref) — the start record
- [`CalibrationEstimate`](@ref), [`CalibrationMetadata`](@ref) — its estimate and diagnostics (a Core `FitResult` read by field: `r.u`, `r.retcode`, `r.stop_rule`, `r.stats.nsteps`, `r.rms`, `r.metadata`; `successful_retcode(r)`)
- [`save_radiometric_calibration`](@ref) — the run's records
- [`CalibrationCache`](@ref), [`calibration_problem`](@ref) (its `RefineProblem` of the public `CalibrationData` and `CalibrationParameters`), `RadiometricCameraModel` (re-exported: VisualRadiometryToolkitCore's radiometric camera), [`RadiometricTarget`](@ref) — the refinement's problem and the models
- [`CalibratedCamera`](@ref) — the complete calibration: the camera model with its observation model (the sensor record its photographs were read under) and render settings; every photograph it predicts is read through `photo_set` / `observe`
- [`RunOptions`](@ref), [`CalibrationSettings`](@ref), [`pipeline_options`](@ref) — a run's options (the capture's facts and its
  input/output: the start reads the first from its data, the driver owns the second) and the calibration's typed settings (the
  problem's structure, its data's), and their TOML reader (defaults: `configs/default.toml`);
  [`read_calibration`](@ref) — a run's TOML as the run, the settings, the calibration and its solve's stops

## Public

A driver's pieces of a run (ruling R1: a run's I/O is its driver's): [`load_calibration_state`](@ref RadiometricCamera.load_calibration_state) (a resume, the refinement's
problem continued from a finished state) and [`calibration_checkpoints`](@ref RadiometricCamera.calibration_checkpoints) (a run's checkpoints and optimizer log, the
`callback` its driver hands the refinement's solve). The refinement problem's data and parameters (`CalibrationData`, `CalibrationParameters`), the image formation (`RenderSettings`, `Φ_pixel`, `render_view`: each of the camera at a pose), the camera's and the target's
parts (`TargetRadiometry`, … — `RadiometricTarget` is exported, above), the closed-form blocks, the LM's blocks for building a problem of one's own (`build_problem`,
`build_problem_grid`, …), the records (`save_state`, `load_state`, …), the observation model and THE entry point of photographs (`ObservationModel`, `calibrated_camera`, `photo_set`, `observe`), the diagnostics (`selfcheck`, `bearing_px`, …), the option tables and their configuration versions (`RC_CONFIG_VERSIONS`, `config_key`, `OPTIMIZER_TAGS`), and
the paper's symbols (`Φ_pixel`, `L_c`, …: public, not exported). The list is `src/api.jl`'s `public` statement.

The generic radiometric types and functions are VisualRadiometryToolkitCore's and are not public names of this package: the camera's
parts and maps (`geometric`, `focus`, `aperture_radius`, `with_…`, `chief_direction`, `focus_point`, `center_of_projection`, `backproject(camera, 𝐩, 𝐚)`, `world_ray`; the paper's symbols `𝔣`, `𝔭`, `π̂⁻¹`, `t̂` are its notation in prose, no alias since its 0.2.0), the focus surface
(`Focus`, `FocusSpline`, `inverse_focus_distance`, `focus_distance`), the vignetting fields, the sensor domain and quadrature (`sensor_position`, `SensorQuadrature`,
`ApertureQuadrature`, …), the light (`EnvironmentLight`), the material (`Material`, `cook_torrance`, …) and the board — the textured target
`TexturedTarget` with its own state `TargetState` (its chart, `albedo`, the printed edge, the warped page, the in-plane print error;
its centre VisualGeometryToolkitCore's `centroid`, its facing `with_facing`). There is no scene type: a render takes the camera, the pose and the board, and the
image size is the camera's. Import them from there:
`using VisualRadiometryToolkitCore: Focus, Material`.
