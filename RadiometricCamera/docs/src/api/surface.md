# The API surface

RadiometricCamera exports its headline API only (`src/api.jl`, the one file that states the surface). Everything else a user may
reach is `public`: write it qualified (`RadiometricCamera.View`) or import it by name (`using RadiometricCamera: View`).

## Exported

<!-- the suite checks this list equals `names(RadiometricCamera)` (test/unit/api_surface_tests.jl) -->
- [`calibrate`](@ref) — a radiometric calibration in one call
- [`start_calibration`](@ref), [`StartGeometry`](@ref), [`start_geometry`](@ref), [`start_photographs`](@ref) — the start: the geometric calibration lifted into a radiometric state
- [`save_calibration_start`](@ref), [`load_calibration_start`](@ref) — the start record
- [`refine_calibration!`](@ref) — the refinement, a `FitResult`
- [`CalibrationEstimate`](@ref), [`CalibrationMetadata`](@ref) — its estimate and diagnostics (read with Core's accessors: `estimate`, `retcode`, `converged`, `iterations`, `rms`, `metadata`)
- [`save_radiometric_calibration`](@ref) — the run's records
- [`CalibrationState`](@ref), [`RadiometricCameraModel`](@ref), [`RadiometricTarget`](@ref) — the state and the models
- [`CalibratedCamera`](@ref) — the complete calibration: the camera model with its observation model (the sensor record its photographs were read under) and render settings; every photograph it predicts is read through `photo_set` / `observe`
- [`PipelineOptions`](@ref), [`pipeline_options`](@ref) — the options and their TOML reader (defaults: `configs/default.toml`)

## Public

The image formation (`Scene`, `RenderSettings`, `Φ_pixel`, `render_view`), the camera and target models (`View`, `Focus`,
`PrintedTarget`, `Material`, …), the closed-form blocks, the LM's blocks for building a problem of one's own (`build_problem`,
`run_lm!`, …), the records (`save_state`, `load_state`, …), the observation model and THE entry point of photographs (`ObservationModel`, `calibrated_camera`, `photo_set`, `observe`), the diagnostics (`selfcheck`, `bearing_px`, …), the option tables and
the paper's symbols (`𝔣`, `𝔭`, `ρ_c`, …: owner D14, public, not exported). The list is `src/api.jl`'s `public` statement.
