# The API surface

RadiometricCamera exports its headline API only (`src/api.jl`, the one file that states the surface). Everything else a user may
reach is `public`: write it qualified (`RadiometricCamera.RenderSettings`) or import it by name (`using RadiometricCamera: RenderSettings`).

## Exported

<!-- the suite checks this list equals `names(RadiometricCamera)` (test/unit/api_surface_tests.jl) -->
- [`calibrate`](@ref) — a radiometric calibration in one call
- [`RadiometricCalibration`](@ref) — the calibration, an algorithm of VisualGeometryToolkitCore's `solve` / `init` / `solve!` on `EstimationProblem(CalibrationEstimate, (geometry, photographs))`: `init` lifts the geometric calibration into a radiometric state (the start, a [`CalibrationCache`](@ref)), `solve!` refines it (a `FitResult`)
- [`StartGeometry`](@ref), [`start_geometry`](@ref), [`start_photographs`](@ref) — the start's inputs: the geometric calibration and the photographs
- [`save_calibration_start`](@ref), [`load_calibration_start`](@ref) — the start record
- [`CalibrationEstimate`](@ref), [`CalibrationMetadata`](@ref) — its estimate and diagnostics (a Core `FitResult` read by field: `r.u`, `r.retcode`, `r.stop_rule`, `r.stats.nsteps`, `r.rms`, `r.metadata`; `successful_retcode(r)`)
- [`save_radiometric_calibration`](@ref) — the run's records
- [`CalibrationCache`](@ref), [`CalibrationState`](@ref), `RadiometricCameraModel` (re-exported: VisualRadiometryToolkitCore's radiometric camera), [`RadiometricTarget`](@ref) — the state and the models
- [`CalibratedCamera`](@ref) — the complete calibration: the camera model with its observation model (the sensor record its photographs were read under) and render settings; every photograph it predicts is read through `photo_set` / `observe`
- [`PipelineOptions`](@ref), [`pipeline_options`](@ref) — the options and their TOML reader (defaults: `configs/default.toml`)

## Public

The image formation (`RenderSettings`, `Φ_pixel`, `render_view`: each of the camera at a pose), the camera's and the target's
parts (`TargetRadiometry`, … — `RadiometricTarget` is exported, above), the closed-form blocks, the LM's blocks for building a problem of one's own (`build_problem`,
`run_lm!`, …), the records (`save_state`, `load_state`, …), the observation model and THE entry point of photographs (`ObservationModel`, `calibrated_camera`, `photo_set`, `observe`), the diagnostics (`selfcheck`, `bearing_px`, …), the option tables and
the paper's symbols (`Φ_pixel`, `L_c`, …: public, not exported). The list is `src/api.jl`'s `public` statement.

The generic radiometric types and functions are VisualRadiometryToolkitCore's and are not public names of this package: the camera's
parts and maps (`geometric`, `focus`, `aperture_radius`, `with_…`, `chief_direction`, `focus_point`, `center_of_projection`, `backproject(camera, 𝐩, 𝐚)`, `world_ray`; the paper's symbols `𝔣`, `𝔭`, `π̂⁻¹`, `t̂`, … are its public aliases of the same functions), the focus surface
(`Focus`, `FocusSpline`, `t̂`, `𝔱`), the vignetting fields, the sensor domain and quadrature (`𝐩_of_pixel`, `SensorQuadrature`,
`ApertureQuadrature`, …), the light (`EnvironmentLight`), the material (`Material`, `cook_torrance`, …) and the board — the textured target
`TexturedTarget` with its own state `TargetState` (its chart, `albedo`, the printed edge, the warped page, the in-plane print error;
the paper's `𝔟` and `ρ_c` are aliases there). There is no scene type: a render takes the camera, the pose and the board, and the
image size is the camera's. Import them from there:
`using VisualRadiometryToolkitCore: Focus, Material`.
