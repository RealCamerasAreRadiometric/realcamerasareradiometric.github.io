# Changelog

## 0.4.0 (unreleased; breaking)

RadiometricCamera on VisualGeometryToolkitCore 0.14's estimation interface: every estimator is `solve(problem, algorithm)`
(CommonSolve's verb, Core's front end), and a `FitResult` is read by field. No calibration number moves (below).

- **The grid start's context is read by Core's `read_context`** (`GridStart`: `coverage`, `device`; another key is an
  `ArgumentError`), and every termination of its regression is documented as accepted as a start (an exact resample that did not
  converge now says so in one line, as the parametric starts did).

- **Breaking: the calibration LMs on Core's block contract (R22: `LmProblem` and `lmsolve_chart!` are Core-internal).** The two LM
  problems are a `CalibrationBundle` (public: `variables`, `blocks`, the free mask `free`), a `VgtkCore.AbstractBlockProblem` of the
  block contract (`block_structure`, `fixed_variables`, plain least squares, no residual summary), solved by
  `init(RefineProblem(bundle, bundle.variables), LevenbergMarquardt(; damping); stops, callback)` and `solve!` (`_solve_bundle!`):
  - `build_problem(f, u0; fixed, height_degree, inplane)` and `build_problem_grid(f, u0; patch, fixed, height_degree, smoothness,
    inplane)` return `(bundle, idx)`; they take no `damping` (the LM's: `LevenbergMarquardt`'s field, the optimizer's), and its blocks
    are `bundle.blocks` (`get(problem.problem.costs, T)` → `filter(c -> c isa T, bundle.blocks)`; `tile_costs(bundle)`);
  - `_lm_views!`, `_lm_tiles!` and the refinement's chunks return the LM's `FitResult` (`retcode`, `stop_rule`, `stats.nsteps`,
    `metadata.cost` = ‖r‖²) in place of `(; code, rule, iterations, cost)`; `CalibrationMetadata.solver` holds it; `lm_termination`
    reads `(r.retcode, r.stop_rule)`; `read_back!(f, u0, vars, idx, extract)` takes the solution's variable vector (`r.u`);
  - the solve's blocks are evaluated in batches again (the concurrent window, the grid's view batches): the bundle's
    `prepare_evaluation!` (Core's block-contract generic) records the two vectors NLLSsolver evaluates them at in
    `bundle.solver`, where the requests find their variables, as they found them in the `LmProblem`'s fields before; the
    same values, bit for bit;
  - the per-iteration callback is Core's `callback = state -> Bool` on an `LmIterationState` (the iterate `state.u`, `cost`,
    `best_cost`, `damping`, `linear_solves`, `scaling`, `trial_ready`): the trace, the monitor, the `log` lines and the final
    iteration's mark read the same values from it; `verbose` prints RC's own line per iteration (`_lm_print`, was NLLSsolver's);
  - the closed-form blocks' joint LM (`JointRadiometry`) is a row problem, `_RadiometryLm{N} <: VgtkCore.AbstractRowProblem{N}`
    (`chart_size = (1, Val(N))`: its rows stay on the device), solved by `solve(RefineProblem(lm, zero(SVector{N})),
    LevenbergMarquardt(; damping); stops)`.
  No number moves: Core builds the same NLLSsolver problem from the same variables and blocks (the golden replay, every leaf `===`).
- **Breaking: American in identifiers, wave 2 (decision 11; VisualGeometryToolkitCore 0.14, VisualGeometryToolkitIO 0.12).** Core's
  `pixel_centers` (was `pixel_centres`) and `is_inside_texel_centers`, IO's `meters` (was `metres`), with no alias. The region kinds
  embed IO's region body, whose polygon now stores its star centre under `"center"`: **`rc/initialization_region/v2.0`** and
  **`rc/image_region/v2.0`**, each with a v1.0 → v2.0 version step calling IO's `migrate_region` (every other key as stored). A v1.0
  record reads through it, in place (never rewritten): checked on the 236 published `rc/initialization_region/v1.0` records and the
  12 `rc/image_region/v1.0` records of the paper's capture sets (each value `===` its v1.0 body's, each file's bytes unchanged), and
  tested on v1.0 records of a rect, a polygon intersection (its `"centre"`) and an ellipse.
- **The experiments on the breaking set.** The chrome evaluation's board refit is a row problem (`BoardRows{N} <:
  AbstractRowProblem{N}`, `solve(RefineProblem(rows, 0), LevenbergMarquardt())`, Core's `LmStops()` as before); the texture solves
  are `solve(PenalizedLeastSquaresProblem(op, b; w, u0, diagonal), CGLS(); stops = ToleranceStops(; rtol, max_iter))` (was Core's
  `cgls`), their records unchanged; a `CountedLO` call record is a `FitResult` (`stats.nsteps`); the project's residual statistics
  are `residual_statistics` (was `residual_summary`, the name of Core's block-contract hook; `src/residual_statistics.jl`); and a
  table row's legacy retcode spelling (`"converged"`, …: a record before Core 0.14) reads through IO's `termination_from_legacy`
  (it was refused). The chrome evaluation's tests are a contract test of the grid-pose problem's refit step (`refine!`).

- **Breaking: American in identifiers (owner decision 11).** Core's `SensorCharacterisation` is `SensorCharacterization` and its
  `ModelReason` `NoCharacterisation` is `NoCharacterization`; RC follows, with no alias:
  - the camera option `camera.sensor_characterisation` → `camera.sensor_characterization` (`CameraOptions.sensor_characterization`);
    a configuration of version 1 (no `config_version`) that spells the old key reads through the converter (`RC_CONFIG_VERSIONS`'
    v1 → v2 step), the value unchanged; a version-2 file is refused it, naming the new key;
  - `photo_set(…; characterisation)` → `photo_set(…; characterization)`, and `prediction_observation`'s third argument;
    `NO_SENSOR_CHARACTERISATION` → `NO_SENSOR_CHARACTERIZATION`; the retired stub `sensor_characterisation(…)` (an earlier
    release's, refused by name) is deleted: Core's retired-name list now names the spelling;
  - **`rc/run_parameters/v4.0`**: the run record's key `sensor_characterization` and its frames' reason `NoCharacterization`; a v3.0
    record (RC 0.3) reads through its version step (`_run_parameters_v4`), every other value as stored — tested on the
    experiments' recorded golden run (`siemens_coarse_full/parameters.json`, unchanged);
  - the experiments: the experiment and stage `sensor_characterization` (its driver, config, sbatch, source and test files
    renamed), its table `[characterize]` and `characterize_sensor`, the input `fx3_sensor_characterization`, the drivers'
    keyword `characterization`, `load_sensor_characterization_config`, `case_characterization`. The recorded data keep their
    names (the directory `sensor_characterisation/`, the files `fx3.sensor_characterisation.*`, the tier
    `rc_sensor_characterisation_20261113`, the former IO kind `vgio/sensor_characterisation/v2.0` the record of record holds,
    read through IO's steps). No number moves.
- **On VisualRadiometryToolkitCore 0.2.0** (`[compat]` `"0.2"`, VisualRadiometryToolkitIO `"0.1.2"`): its notation aliases are
  deleted, and RC calls the functions by their names, with no alias of its own. `𝔣` → `focus_point`, `t̂` → `inverse_focus_distance`, `π̂⁻¹` → Core's
  `backproject` (RC's `View` method of it), `𝔭` → `center_of_projection`, `𝐩_of_pixel` → `sensor_position`, `𝐩_sample` →
  `sensor_sample`, `𝐚_sample` → `aperture_sample`, `𝐚_weight` → `aperture_weight`, `𝐚_border` → `aperture_border_sample`,
  `Θ_m_nonlinear` → `nonlinear_parameters`, `with_μ` → `with_diffuse_weight`, `𝛂` → `roughness`, `𝛈` → `refractive_index`, `𝔟` → Core's
  `centroid`, `ρ_c` → `albedo` (the symbols stay the paper's notation in prose, fields and locals). A board's facing is
  `with_facing(TexturedTarget(…), viewpoint)` (the `facing` keyword is gone). `VignettingOptions.chart` is the radial chart's
  value (`LinearChart()`, `ProjectiveChart()`, `EquisolidChart()`), read and written through `RADIAL_CHART_TAGS`. The TOML keeps its
  strings (`chart = "projective"`), so there is no converter step. RC's refusals of the names that moved to VisualRadiometryToolkitCore
  stay, with the old "the paper's symbol stays its alias" hints dropped.
- **Breaking: the run's data are the problem's `p`; `PipelineOptions` is split (#16).** `PipelineOptions` is removed, with no alias and
  no property forwarding, into two values, each TOML table in one (the TOML does not change: no key moves, no converter step):
  - `RunOptions` (the run's data and input/output: `io`, `frontend`, `camera` — the capture's facts —, `resume`, `selfcheck`,
    `compute`) and `CalibrationSettings` (the calibration's typed settings: `fixed`, `grid`, `focus`, `masks`, `rendering`,
    `lighting`, `material`, `vignetting`, `warping`, `edge`); every option is read at its real path (`opts.io.out` → `run.io.out`,
    `opts.grid` → `settings.grid`);
  - `pipeline_options(d) -> (run, settings)`; `read_calibration(d) -> (run, algorithm, stops)`; `config_dict(run, settings)`,
    `config_dict(run, algorithm, stops)`, `write_pipeline_config(dir, run, settings)` / `(dir, run, algorithm, stops)`;
    `start_options(run, settings) -> (run, settings)`;
  - `RadiometricRefinement(settings; optimizer, radiometry, lm_chunk, log)` holds the `CalibrationSettings` alone
    (`alg.options` → `alg.refinement.settings`); the run is the problem's context: `EstimationProblem(CalibrationEstimate,
    (geometry, photographs), (; run[, device]))` and `RefineProblem(f, u0, run)` (`CalibrationCache.prob`,
    `load_calibration_start`'s result);
  - `calibrate(geometry, photographs, run, algorithm; stops, device)`; `start_geometry(run, settings; record)`;
    `start_photographs(run, names)`; `selfcheck(g; run)`; `save_calibration_start(dir, f, u0; run, settings)`;
    `load_calibration_start(dir; run, settings, device, …)`; `save_radiometric_calibration(f, u0, result; run, settings)`;
    a start record's `options` is its `(run, settings)`.
  No number moves: the same options reach the same code.
- **No upstream internal name (R22): the reductions' work items are Core's public `reduction_work_items(n)`; the row-job launcher is
  public, `launch_rows!`.** The closed-form radiometry's partial-sum count read Estimation's internal `_REDUCTION_ITEMS` through RC's
  own `_reduction_items` (removed): it is Core's `reduction_work_items` (VisualGeometryToolkitCore 0.14, the same `max(1, min(n, 2048))`).
  RC's row-kernel launch through Core's `pointwise!` (`_launch_row_job!`) is the public `launch_rows!(row!, out, n, args...; exec)`, and
  the experiments' sparse-operator kernels (experiments/src/texture_recovery.jl), which called Core's internal `_launch_rows!`, run
  through it. Bit for bit: a row's values do not depend on its launch.
- **The geometry image `"raw_green"` through Features' public `bilinear_green`** (VisualGeometryToolkitFeatures 0.10's export):
  `raw_green_image` called the internal `_bilinear_green` kernel (R22); it now reads the public function, the same kernel, bit for
  bit. The experiments' check that the decode is the chrome detect's image compares with `bilinear_green` and no longer reads
  Features' source text (experiments/test/drivers_tests.jl).
- **Breaking: a smoothness weight chosen by GCV is Estimation's `GeneralizedCrossValidation()` (spec §9 G-j).** Estimation retired
  `gcv_select` and the weight `:gcv`; RC follows, with no alias:
  - `PSplineSmoothness(order, :gcv)` / `BendingSmoothness(:gcv)` → `PSplineSmoothness(order, GeneralizedCrossValidation())` (a weight is a
    `Float64` or the algorithm value); `RadiometryBlockDescent(; vignetting_smoothness = GeneralizedCrossValidation())`, and
    `_update_vignetting!`'s `smoothness_weight`; a Symbol weight is refused (`check_penalty_weight`'s `ArgumentError`);
  - the selection is `solve(EstimationProblem(Hyperparameter, curve, (; scale)), GeneralizedCrossValidation())` in both GCV solves
    (the vignetting's closed form, the grid's linearization: `select_grid_smoothness_weight` takes the weight's algorithm from its
    smoothness); the records keep their shape, `(; β = r.u, r.metadata...)` = gcv_select's `(; β, λ, gcv, rss, dof, log10_β_bracket,
    at_bracket_end)`;
  - the TOML does not change: `weight = "gcv"` reads as `GeneralizedCrossValidation()` at its defaults, and writes back as `"gcv"`
    (`weight_record`; a non-default bracket or scan is no option spelling and is refused by the writer, never written as the default).
  No number moves: the same scan and Brent search on the same curves.
- **Core's public `matvec!` and `toml_leaves`** (VisualGeometryToolkitCore 0.14, Phase 1A): the device Jacobian's `Jᵀr` and the
  boolean-switch scan called the internal `_matvec!` and `_toml_leaves` (R22); they read the public functions, the same code, bit for bit.
- **Breaking: the refinement is a refine, `solve(RefineProblem(f, u0), RadiometricRefinement(…); stops)` (#18).**
  `CalibrationState` (data and estimate in one mutable struct) is removed, with no alias and no property forwarding:
  - `CalibrationData` (the problem's `f`: the board's design, the render settings, the masks and pixel lists, the photographs and
    their observations ϕ, the weights, the vignetting rule, the backend and its device cache, the channel, the focus smoothness,
    the μ interval, the code steps, the sensor models and observation model, the run record, the start) and
    `CalibrationParameters` (the problem's `u0`, refined in place: `camera`, `poses`, `mat`, `light` and the board's `surface`,
    the `TargetState` the board carried before; the kernels' board is the design with that surface, `_board(f, θ)`);
  - `CalibrationState(board, settings, camera, poses, mat, light, targets; …)` → `calibration_problem(…)` (the same arguments: a
    `RefineProblem` of `CalibrationData(board, settings, camera, poses, targets; …)` and `CalibrationParameters(f, camera, poses,
    mat, light)`), each field read at its real path (`s.ϕ` → `f.ϕ`, `s.camera` → `u0.camera`, `s.board.state` → `u0.surface`);
  - every function of a state takes the part(s) it reads: `cost(f, u0)`, `render_all(f, u0)`, `photometric_residuals(f, u0)`,
    `all_parts(f, u0)`, `build_problem(f, u0; …)`, `build_problem_grid(f, u0; …)`, `save_state(path, f, u0, names)`,
    `load_state_into!(f, u0, state, names)`, `set_radiometric_target!(f, u0, t)`; `n_views`, `geometric`, `focus`,
    `aperture_radius`, `board_surface`, `set_surface!`, `set_vignetting!`, `set_pose!` of `u0` alone; `data_norm2`, `lm_abstol`,
    `observation_model`, `residual_channels` of `f` alone; `RadiometryBlocks` on `EstimationProblem(RadiometryBlocks, (f, u0, parts))`;
  - `RadiometricRefinement(options; optimizer, radiometry, lm_chunk, log)` (a `RefineKind`: the former `RadiometricCalibration`'s
    fields) solves `RefineProblem(f, u0)`; `RadiometricCalibration(refinement)` (the `FitKind`: the start, then the refinement)
    holds it, `read_calibration` returns `RadiometricCalibration(RadiometricRefinement(…))` (`alg.options` → `alg.refinement.options`);
  - `CalibrationCache` holds `prob` (the started `RefineProblem`; was `state`); `load_calibration_start` returns the
    `RefineProblem` (`solve(prob, alg.refinement; stops)`, was `solve(EstimationProblem(CalibrationEstimate, state), alg; stops)`);
    `save_calibration_start(dir, f, u0; options)`; the retired View-form `CalibrationState` and `lens(::CalibrationState)` stubs are
    gone with the type. No arithmetic moves: the same kernels on the same values;
  - `RadiometricCalibration`'s `init` runs its refinement stage through the front end (spec §9 G-l): its cache is
    `init(RefineProblem(f, u0), refinement; stops)`, an `EstimationCache` of the `CalibrationCache` (`c.cache.prob` the started
    problem, as before), so `solve!` records the stage's problem and algorithm before relabelling the result with the calibration's.
- **Experiments: `experiments/chrome_evaluation/v2.0`, the FitResult's names in the eval record (decision 11).** Each fit of
  `eval_<label>.json`'s `fit` (`in_sample`, `held_out.even` / `.odd`) records `successful_retcode` (was `converged`) and
  `stats.nsteps` (was `iterations`), and the fit's iteration cap is `fit.max_iter` (was `fit.iterations`, the `LmStops`' `max_iter`
  it sets). A v1.0 record — the run of record's, written under the project's earlier name — reads through the version step of the
  one reader, `read_chrome_evaluation` (in memory; the files stay as published): scripts/chrome_index_grid.jl's rows read it. Tested
  on a fixture of the run of record's values and on its 253 eval records where results_root() binds them: every value `===`.
- **Breaking: the calibration is an algorithm.** `start_calibration` and `refine_calibration!` are removed (check_naming R15: a
  `refine_` name is a kind, never a function), replaced by `RadiometricCalibration(options)` (a fit, `FitKind`) on
  `EstimationProblem(CalibrationEstimate, (geometry, photographs)[, (; device)])`:
  - `start_calibration(geometry, photographs; options, device)` → `init(EstimationProblem(CalibrationEstimate, (geometry,
    photographs), (; device)), RadiometricCalibration(options))`, its cache a `CalibrationCache` (new: the started `state` and the
    run's `options`); the start reads `start_options(options)` with the run's `io.out`, as `calibrate` did. The four-argument form
    builds its `StartGeometry(camera, poses, observations, target; names)` first;
  - `refine_calibration!(state; options)` → `solve!(cache)`, or `solve(EstimationProblem(CalibrationEstimate, state),
    RadiometricCalibration(options))` of a started state (`load_calibration_start`'s);
  - `calibrate` (a task verb) is that `solve` in one call, unchanged.
- **Breaking: the optimizer is the algorithm's, the stops the solve's (the stack's `solve(problem, alg; stops)`).**
  `RadiometricCalibration(options; optimizer = LevenbergMarquardt(; damping = LevenbergDamping()), log = "")`: the refinement's
  optimizer (VisualGeometryToolkitCore's `LevenbergMarquardt`, its damping Levenberg's λ·I, the former `LM_DAMPING`, now the
  algorithm's default and no constant) and the LM's per-iteration log; the stop settings are Core's `LmStops`, the solve's
  keyword (`solve(problem, alg; stops)`, `init`'s, `calibrate`'s), configs/default.toml's by default. `calibrate(geometry,
  photographs, algorithm; stops, device)` takes the algorithm.
  - `LMOptions` and `PipelineOptions.lm` are removed: the TOML's `[lm]` table is **`[optimizer]`** (`type = "levenberg_marquardt"`,
    `OPTIMIZER_TAGS`; `max_iter`, `max_time`, `ftol`, `xtol_rel`, `log`), read by **`read_calibration(path_or_dict; set) ->
    (algorithm, stops)`**; `config_dict(algorithm, stops)` and `write_pipeline_config(dir, algorithm, stops)` write it.
    `pipeline_options` still reads (and checks) a whole run's file. `REFINE_ONLY_KEYS` and `PIPELINE_OUTPUT_KEYS` name
    `optimizer.*`.
  - **Configuration versions** (VisualGeometryToolkitCore's `ConfigVersions`): `RC_CONFIG_VERSIONS`, configs/default.toml
    `config_version = 2`. A file without `config_version` is version 1 (RC ≤ 0.3, every config of record) and is read through the
    converter: `[lm]` is the `[optimizer]` table, its keys kept. The configs of record stay version 1 on disk; their start and
    refinement settings are split as written (`split_refine_only` through **`config_key(table, key)`**, a current key as a
    table's version names it), so no product key moves (tested on every experiments config of record).
  - `run_lm!`, `run_lm_grid!` and `radiometry_lmsolve!` are internal (`_lm_views!`, `_lm_tiles!`, `_lm_radiometry!`, each
    taking `stops` and `damping`): the calibration is reached through `solve` / `init` / `solve!`. `build_problem`,
    `build_problem_grid` and `update_radiometry!(…; joint = true)` take the LM's `damping`.
- **Breaking: the closed-form blocks are a solve.** `update_radiometry!`, `update_lighting!`, `update_vignetting!` and
  `update_diffuse_weight!` are internal; the closed-form blocks are `solve(EstimationProblem(RadiometryBlocks, (state, parts),
  (; light, vignetting, mu)), algorithm)` with `RadiometryBlockDescent(; sweeps, reltol, vignetting_smoothness, verbose)` (the block
  sweeps) or `JointRadiometry(; damping, sweeps, reltol, verbose)` (their joint LM), a `FitResult{RadiometryBlocks}` (the blocks
  written into the state; `metadata.cost`, `metadata.gcv`). `RadiometricCalibration` takes `radiometry` (one of the two; `[sweep]
  joint`, `iterations`) and `lm_chunk` (`[sweep] lm_chunk`): `[sweep]` is the algorithm's table (`SweepOptions` and
  `PipelineOptions.sweep` removed), read by `read_calibration`, its keys unchanged. Same arithmetic, same numbers.
- **Breaking:** `calibrate(camera, poses, observations, photographs, algorithm; target, names)` is removed (keywords are configuration
  only): the geometric calibration is data, `calibrate(StartGeometry(camera, poses, observations, target; names), photographs,
  algorithm; stops, device)`.
- **Changed: VisualGeometryToolkitCore 0.13.4's device policy.** The backend choices are Core's: `KernelAbstractions.isgpu`,
  `kernel_float`, `dual_chunk`, `OnBackend(…; host_workgroupsize)`, `device_resident_threads`, `device_names` and
  `on_device(T, S, device, x)`; the test scenes' board warping is Core's `Synthetic.saddle_height`. The repository's
  DUPLICATION.toml is gone (no exception left). Same values.
- **Breaking: the grid starts are a solve.** `GridStart(; lattice, node_count)` (a fit) on `EstimationProblem(SplineLens, source[,
  (; coverage, device)])`, a `FitResult` of the bounded SplineLens camera (the regression's retcode, stop rule, stats and rms;
  `metadata.regression`, and `resample` — an exact start — or `spacing`, `origin`, `supported` — a region start): `DerivedLattice()`
  a SplineLens source resampled exactly (was `exact_start`), `NodeCountLattice()` a parametric source over the image region (was
  `region_start`), `ImageLattice(G_x, G_y)` a parametric source's full-image lattice (new; was `grid_model(src, G_x, G_y)`).
  `grid_model`, `exact_start`, `region_start`, `start_grid`, `schoeps_gridlens` and `schoeps_start` are removed (a Schöps run's
  model: `load_camera("schoeps:" * path)` and `GridStart(; lattice = DerivedLattice())`). Same cameras.
- **Public:** `PIPELINE_OUTPUT_KEYS`, `START_RECORD_FILE`, `START_MASKS_DIR`, `load_refinement_start`, `masked_observations` and
  `Unweighted` (read by the paper project).
- **Changed:** the start's per-view pose refit (`_refined_pose`) returns its `FitResult`; a start pose whose refine did not succeed
  is not taken (`successful_retcode`), and a session pose refit that did not succeed keeps the session's pose.
- **Breaking: the closed-form block updates are named `update_<block>!`** (check_naming R19: no identifier begins `solve_`):
  `solve_Θ_l!` → `update_lighting!`, `solve_Θ_v!` → `update_vignetting!`, `solve_μ!` → `update_diffuse_weight!`,
  `solve_linear_blocks!` → `update_radiometry!` (the composite block: its parts' updates per sweep, or the joint LM); the joint LM
  `solve_radiometry_gn!` → `radiometry_lmsolve!` (a step named by its mechanism, as Core's `lmsolve_chart!`; internal since: the optimizer entry above). Same arguments.
- **Breaking:** `fit_channels(s)` → `residual_channels(s)` (R15); the public constant `PRINT_PLANE` is removed (R20: a `const`
  naming one concrete stack type) — the print blur's plane is `VisualGeometryToolkitCore.TargetPlaneFrame{:board}`, spelled.
- **Changed: Estimation's estimators through `solve`.** The SplineLens starts (`grid_model`, `region_start`, `exact_start`) fit by
  `CameraRegression(; tolerance = Inf, grid_size | lattice_spacing, forward = false)` on `CameraRegressionProblem(SplineLens,
  source, region; support, device)` (one helper, `_spline_regression`; was `fit_camera`); a view's pose from its rays by Core's
  `LevenbergMarquardt(; optimization_strategy = nothing)` on `BearingPoseProblem(cs)` (one helper, `_refined_pose`; was
  `refine_absolute_pose`). Results are read by field (`r.u`, `r.retcode`, `r.stop_rule`, `r.stats.nsteps`, `r.rms`, `r.metadata`,
  `successful_retcode(r)`). RC's LM damping (`LM_DAMPING`) is the `LmProblem`'s and the chart LM's damping argument, no longer an
  `LmStops` field (Core 0.14). The generalized cross-validation is Estimation's `gcv_select` (RC keeps no copy).
- **Tests:** the README test develops the stack packages the `[sources]` take by path (TEMPORARY, with those entries); the
  experiments paper project's naming scan is skipped (TEMPORARY, `@test_skip`) until the owner rules whether experiments/ is ported to
  0.4.0 or stays on the released 0.3.0. The goldens hold unmodified (rtol 1e-10): no calibration number moves.
- **Tests:** the quality suite runs VisualGeometryToolkitCore's `check_duplication(RadiometricCamera)` (zero findings). Its
  exceptions file moves to the repository root, where the checker reads it (in RadiometricCamera/ no check read it); every
  remaining entry is TEMPORARY and names its removal (Core 0.13.4's device-policy functions and synthetic saddle, the ruling on
  experiments/), and the `_TOML_DEFAULTS` entry is gone (Core's per-package infrastructure). Removed duplicates: one Gauss–Newton
  block function for views and tiles (`_block_normal_equations` over `_block_jacobian` / `_cost_rows`); the moved names' refusals
  listed in `MOVED_NAMES` (Core's convention; helper `_moved_name_refusal`); the tests' central difference, cache-type check and
  two-view scene, one copy each.
- **Dependencies:** VisualGeometryToolkitCore 0.14, VisualGeometryToolkitEstimation 0.13, VisualGeometryToolkitFeatures 0.10,
  VisualGeometryToolkitIO 0.12 (none registered yet: the package's `[sources]` take them by path, TEMPORARY until they are).

### experiments

- **Changed:** on the Core 0.14 solve stack (TEMPORARY `[sources]` at the BlobBoards 0.8, LocalOptimizationDoneRight 0.8,
  PanoramaStitching 0.12, monorepo and Visual Radiometry Toolkit port worktrees; compat RadiometricCamera 0.4). Every estimator
  call is `solve(problem, algorithm)`: the index stage's RANSAC (`Ransac` on its `GridPoseProblem`, whose sample is
  `sample_models` through `MinimalProblem` + `P3P`), its pixel refit (`LevenbergMarquardt` on `AbsolutePoseProblem`), the MAD scale
  (`MedianNormScale`), the pooled scale (`PooledScale`), the rotation-centre start's least squares (`_linear_least_squares`:
  `ReweightedLeastSquares` on a `LinearModel`) and refine (`LevenbergMarquardt` on `CameraPosesProblem` of `RotationCenterViews`,
  its stops the problem's with the evaluation's cap), the ray models' best centre (`NearestPoint`), the parametric starts (`Ransac` on
  `AbsolutePoseProblem`, then `CameraPosesProblem`), the sensor characterisation (`MasterDark`, `PhotonTransfer`), the board's page
  map (`ReweightedLeastSquares` on `EuclideanMap{3}`) and Kabsch (`Kabsch`); results read by field.
- **Changed: BlobBoards 0.8's board finder.** The texture recovery's `sheet_poses` reads `find_boards`' `uid => FoundBoard` pairs (the
  pose `found.fit.u`, the scored set `found.correspondences`; was the `(uid, model, quality, scored)` tuple). The config of record's
  `[poses.find_boards]` (texture_recovery_14mm_small.toml) is BlobBoards' run configuration version 2: the arm's find under
  `[poses.find_boards.geometry.ransac]` (`scoring = { type = "student_t_uniform", outlier_halfwidth = 5.0 }`, its
  `local_optimization_strategy`), `pose_disambiguation = "none"` (was `resolve_pose_ambiguity = false`); the same configuration
  value for value (BlobBoards migrates the version-1 table to it). The boolean-switch lint's allowances for
  `poses.find_boards.geometry.resolve_pose_ambiguity` and PanoramaStitching's `example.decode.cache` and
  `example.registration.check_tracks` are gone: the first is a tag now, and no config of record holds a PanoramaStitching table
  (its version 2's `registration.bundle` is a tag, not a boolean).
- **Renamed** (check_naming R15: the kind words name no function): `fit_start` → `_fit_start`, `fit_rotation_center` →
  `_fit_rotation_center` (its records keep `"method" = "fit_rotation_center"`), `refine_product` → `refinement_product`,
  `refine_settings` → `refinement_settings`, `fit_operator` → `texture_operator`.
- **Changed:** `tools/texture_benchmark.jl` takes `--key=value` flags (VisualGeometryToolkitEval's `cli_option`, the paper project's
  one reader; was `key=value`). The reproduce test writes its number fragments through Eval's number macros.

## 0.3.0 (2026-11-25; breaking)

- **Breaking:** the GPU is the default device. `CalibrationState`, `render_view`, `compute_mask` (whose `backend` was required
  and is now optional), `start_calibration` and `load_calibration_start` default their device to the new `default_backend()`:
  VisualGeometryToolkitCore's `default_device()`, a functional GPU once its package is loaded (`using CUDA`), else the CPU, with
  a warning, once per session, that the run will be very slow. They defaulted to the CPU before (`start_calibration`,
  `load_calibration_start`, `calibrate`, `measure_image_region`, `start_grid`, `exact_start`, `check_coverage`, `calibrated_folds`,
  `schoeps_start`, `edge_strength` and the start's masks to `default_device()`, without a warning; they now default to
  `default_backend()` too). Supplying `backend = CPU()` chooses the CPU and does not
  warn. On a GPU the kernels compute in Float32, so a call that relied on the CPU default and compares to Float64 tolerances
  supplies `backend = CPU()`. The default does not load CUDA: a keyword default runs inside the caller's function, which cannot
  call the methods a package load defines then (world age), so the README, the landing page and Getting Started load it
  (`using CUDA`) and drop `rc_backend("cpu")`; their checks hold to Float32's precision on either device. Loading CUDA forgets a
  default device resolved before it (RadiometricCameraCUDAExt's `__init__`). The drivers are unchanged: they supply the device
  of their `[compute] backend` (`rc_backend`), and the runs of record with it. `rc_backend("gpu")` ends in `default_backend()`
  after loading the vendor packages, so its CPU fallback warns with the same one statement (`fallback = false` still refuses).

- **Tests:** the golden pipeline case (`experiments/test/pipeline/golden/siemens_coarse_full`) is re-recorded on a full B200 (GPU,
  Julia 1.13). Two causes moved it since its recording of 2026-11-16, neither a regression: RC 0.11.5's intended change to the
  sampling stream, whose golden re-record never reached the release line; and the Float32 rounding of the device Jacobian's chunks
  after the performance work of round 2 (1c4dc5f, 09551bf, 30ad4c2), which moves the device's result toward the CPU's (the CPU path
  is unchanged to 1e-13; CPU and GPU differ by 0.13° in the poses on this case). The final rel-RMS against the old golden: 0.24673 →
  0.24689; the new costs 5.074424987570717 (start), 4.931025570093097 (after the linear solve), 4.352524958361566 (final). The
  tolerance mode of `compare_outputs.jl` (1e-9) is meaningful only in the recorded environment (Julia, Manifest and device).

- **Changed:** on VisualGeometryToolkitCore 0.13.3, VisualGeometryToolkitEstimation 0.12.2, VisualGeometryToolkitFeatures 0.9.11
  and VisualGeometryToolkitIO 0.11.2 (the `[compat]` floors are the versions General holds; 0.13's removed spherical spline and
  dilogarithm were never called here), in the package and every workspace member.
- **Changed:** RadiometricCamera is registered in General (its 0.1.0 and 0.2.0 were in CauRegistry only); the README, the
  landing page, the project page and the setup guide install from General alone. The examples build K from the image,
  `CameraCalibrationMat(412.0, sz)` (0.13.2: the focal length and the principal point at the image's centre).

### experiments (the storage design: VisualGeometryToolkitEval 0.10)

- **Changed:** every input of record is a DataArtifacts tier of `configs/data_artifacts.toml`, one entry per tarball of the
  paper's Zenodo records (four input records under 50 GB each, and the container image's own record), packed 2026-11-21 from
  the data root of the run of record, tree hash and sha256 bound in `Artifacts.toml` (each record's placeholder until it is
  published) and the trees installed in the vcai site's artifact store. Each entry names the tier its inputs' configs use
  (`rc_raw_photos`, `rc_baselines`, `rc_calibration`, `rc_thorlabs`, `unpublished`), so no run is re-keyed. The tiers of
  2026-10-29 on the private Hugging Face dataset are retired (optional, kept bound).
- **Changed:** `scripts/fetch_data.jl` is DataArtifacts' `fetch_data!`: the trees install into the artifact store, never the
  depot. Nothing under `data/` is tracked but its README (`DATA_LOCATIONS.md` removed).
- **Changed:** no machine path or host name in a config, test, fixture or document: a machine's roots are its site profile's
  (`VGTK_SITE`), else the environment; the stage graph's per-machine `[sites]` roots of the B200 server are removed; the run of record's
  `[launch]` names `results_dir` and `tree_dir` under the machine's results base and takes julia and the depot from the
  site's environment; `run_of_record.jl` takes its site from `--site` or `VGTK_SITE`.
- **Added:** the privacy rule: the photographs DSC06045 and DSC06052–DSC06055 are in no tier, tree, tarball or record
  (`src/data.jl` `PRIVATE_PHOTOGRAPHS`; `make_artifacts.jl` refuses them and `--check` lists staged tarballs; the suite checks
  every entry and installed tree).

Performance and cleanup (round 3). No calibration number moves beyond the rounding stated below; the goldens hold unmodified.

- **Changed:** the Jacobian's row kernel launches through VisualGeometryToolkitCore's `_launch_rows_on!` (`OnBackend` with the
  per-pixel workgroup, `_pixel_rows_exec`) and `J r` through its `_matvec!` (on CUDA its one-column matrix product): RC's
  `_launch_pixel_rows!` and the CUDA extension's `_rows_times` are removed.
- **Added:** the run record (`parameters.json`) carries `focus_options` beside `vignetting_options`: `[focus.smoothness]` as applied
  and its row weight β Σϕ²/N (`focus_smoothness!`); `nothing` without a focus smoothness.
- **Changed:** one reader of a record ray's hit (`_recorded_hit`: the chunks' tangent plane and the parts).
- **Fixed:** `benchmark/run.jl` named its scene's board twice (a syntax error since the port onto the textured target).
- **Tests:** the focus columns of the sensor-grid thin plate spline against central differences of the tile residual (the check of
  the removed direction-chart focus had been the only one).

### experiments (round 3)

- **Changed:** `page_solves` preconditions each regularised texture solve (λ > 0) by the normal matrix's diagonal
  `diag(AᵀA) + λ diag(DᵀD)` (`column_sumsq`; VisualGeometryToolkitCore's `cgls` `diagonal`); the unregularised solve stays
  unpreconditioned (ill-posed: the scaling moves the texels the rows barely see by 10²–10³). **The regularised textures change**:
  the stopping rule is Core's, in the scaled norm, and the preconditioned iterations get much further. On the texture benchmark
  (B200, 10 photographs, 443,339 fit rows; `experiments/tools/texture_benchmark.jl`) the iterations of the weights 0.001 and 0.1
  are 195, 76 → 89, 41 (the unregularised 245, unchanged); in `run_page` the learned model's 229, 114 → 95, 52 and the pinhole's
  300 (not converged), 77 → 57, 21; the learned model's test SSE at weight 0.001 1.754e-4 → 2.571e-5, at 0.1 0.029939 → 0.029918
  (the unregularised texture's bit for bit). `run_page` 1.04 → 1.00 s warm, `page_solves` 0.157 → 0.141 s (the same job).
- **Tests:** the sampling-strategy study's `focus_metrics` (a dome, a flat focus, a node at infinity).

## 0.2.0 (2026-11-21)

The port onto the Visual Radiometry Toolkit, a breaking release. The generic radiometric layer is VisualRadiometryToolkitCore's
(2026-11-20). RadiometricCamera keeps the paper's calibration method and depends on VisualRadiometryToolkitCore and
VisualRadiometryToolkitIO (the Visual Radiometry Toolkit extends the Visual Geometry Toolkit). No number of a calibration changes:
the moved code is the same arithmetic, and the goldens hold unmodified.

The breaking changes in short (each is detailed below):

- the radiometric camera, the focus surface, the sensor quadrature, the material and the vignetting models and light are
  VisualRadiometryToolkitCore's: import them from there (`RadiometricCameraModel` stays exported here);
- the board is VisualRadiometryToolkitCore's `TexturedTarget` (was `PrintedTarget`) and holds its own state (`TargetState`, was
  `BoardSurface`); `Scene` and `with_surface` are gone, and the entry points take the camera, the pose and the board;
- a camera composed from its geometric camera has `NoVignetting` unless a vignetting is given; a calibration asks for its flat
  start, `flat_vignetting(geometric)`;
- a radiometric camera's geometric camera is `geometric` (was `lens`);
- the alternative focus parameterisations, their options (`focus.variant`, `focus.node_count`, `focus.perturb`) and the report
  `focus_ab.json` are removed: the focus surface is the sensor-grid thin plate spline;
- `green_photosite_mask` is `photosite_mask(m, :G)`;
- the Visual Geometry Toolkit releases it runs on: VisualGeometryToolkitCore 0.12, VisualGeometryToolkitIO 0.11.

The records are unchanged, except that a camera or state record of a removed focus surface type is refused by name.

- **Installation:** RadiometricCamera, VisualRadiometryToolkitCore and VisualRadiometryToolkitIO are registered in CauRegistry
  (<https://github.com/prittjam/CauRegistry.jl>): add that registry beside General, then `pkg> add RadiometricCamera` (the README's
  install lines). The package resolves from the registries alone: its project files hold no `[sources]` but the workspace members'
  path to the package.
- **Dependencies:** VisualRadiometryToolkitCore 0.1 and VisualRadiometryToolkitIO 0.1 are new; compat VisualGeometryToolkitCore 0.12
  (was 0.11), VisualGeometryToolkitIO 0.11 (was 0.10), VisualGeometryToolkitEstimation 0.12.1, VisualGeometryToolkitFeatures 0.9.9.
  QuadGK is a test dependency only (the quadrature rules moved with the quadrature).
- **Changed:** the vignetting and the light of a record (`rc/radiometric_camera`, `rc/calibration_state`) are written and read by
  VisualRadiometryToolkitIO (its bodies of VisualRadiometryToolkitCore's types; were VisualGeometryToolkitIO's). The records'
  schemas, keys and values are unchanged. A camera without vignetting has no vignetting body: saving it is refused by name.

- **Moved to VisualRadiometryToolkitCore (breaking: no longer public names of RadiometricCamera):** the radiometric camera
  `RadiometricCameraModel` with `geometric`, `focus`, `aperture_radius`, `with_geometric`, `with_focus`, `with_aperture_radius`,
  `with_vignetting`, `chief_dir`, `𝔣`, `t̂`, `π̂⁻¹`, `world_ray`, `𝔭`; the focus surface `Focus`, `FocusSpline`, `n_nodes`, `𝔱`;
  the sensor domain and quadrature (`𝐩_of_pixel`, `pixel_of_𝐩`, `ℐ_𝐮`, `luma`, `luma_image`, `SensorQuadrature`,
  `ApertureQuadrature`, `sensor_samples`, `aperture_samples`, `aperture_border`, `𝐩_sample`, `𝐚_sample`, `𝐚_weight`, `𝐚_border`,
  `n_disc`, `n_border`); the material (`Material`, `Θ_m_nonlinear`, `with_nonlinear`, `with_μ`, `texture_weight`, `𝛂`, `𝛈`, `D_ggx`,
  `Λ_ggx`, `G1_ggx`, `G2_ggx`, `F_dielectric`, `sample_vndf_caps`, `D_v`, `PDF_reflect`, `reflect`, `f_CT_parts`, `f_CT`). Import
  them from there (`using VisualRadiometryToolkitCore: Focus, Material`). `RadiometricCameraModel` stays exported here (a re-export: the
  headline's camera type). The moved names this package does not use itself (`with_geometric`, `with_focus`, `with_aperture_radius`,
  `pixel_of_𝐩`, `ℐ_𝐮`, `aperture_samples`, `aperture_border`, `Λ_ggx`, `G1_ggx`, `ae_chart`) are refused by name with the replacement.
- **Removed (breaking): the focus A/B experiment and its alternative focus parameterisations.** The experiment is decided on real
  data: the sensor-grid thin plate spline fits best, and it is the one focus surface (the former variant "A": VisualRadiometryToolkitCore's
  `Focus`, every node free). No number of a calibration changes: the kept path is the same arithmetic in the same order, and the
  goldens hold unmodified. Removed:
  - the variants "B" (the nodes outside the support frozen: `CalibrationState`'s `focus_fixed`, `focus_block`'s `fixed` keyword), "C"
    (the thin plate spline in the azimuthal-equidistant chart of the chief directions) and "D" (the spherical spline), with `Focus`'s
    `kind` and every branch on it (VisualRadiometryToolkitCore's `FOCUS_SENSOR`, `FOCUS_DIRECTION_CHART`, `FOCUS_SPHERE`,
    `focus_weights`, `polar_cap_nodes` and the direction method of `inverse_focus_distance` are removed there; `t̂_dir` is refused by
    name here);
  - `src/experiments/focus_ab.jl` (`focus_variant`, `focus_theta_max`, `focus_node_pixel_counts`, `write_focus_ab`); the bending
    smoothness of the focus surface stays (`[focus.smoothness]`; `focus_smoothness!`, now in `src/calibration/smoothness.jl`), with the
    focus start and the node-support rule (`focus.lens_coverage_scale`, `focus.support_min`);
  - the options `focus.variant`, `focus.node_count` and `focus.perturb`: each is refused by name (`RETIRED_KEYS`), whatever its value —
    delete the key (the retired `focus.nodes` names the same removal);
  - the report `focus_ab.json` and its record kind `rc/focus_ab`: a run no longer writes it, and a file of it (v1.0 or v2.0) is refused
    by name by `VisualGeometryToolkitIO.read_record`. A run's focus surface is in its state and camera records.
  Records: a camera or a state whose focus `type` is `sensor_thin_plate_spline` reads unchanged (`rc/radiometric_camera/v3.1` and
  `rc/calibration_state/v3.1`: no schema version changes — the body is the same, and the two removed values of `type` were refused
  values of the same key before they existed); one whose `type` is `direction_thin_plate_spline` or `direction_spherical_spline` is
  refused by name (the focus surface type was removed; refine again from the run's start).
- **Added:** a RAW photograph is fitted at the photosites of any one colour filter: `io.channel = "R"`, `"G"` or `"B"` (it was
  `"G"` alone), an `ObservationModel` of `Photosites` with the channel `:R`, `:G` or `:B` (`:RGB` stays refused: a photosite is ONE
  observation). `photosite_mask(m, channel)` gives the usable photosites of one filter on the raw grid (Core's `photosite_colors`: 0
  the red, 1 and 3 the greens, 2 the blue; censored at the ADC floor and the white level as before); `green_photosite_mask` is
  retired by name in its favour (`photosite_mask(m, :G)`, the same mask bit for bit). The edge-selective draw (`masks.edge_fraction`
  > 0) reads the diagonal neighbours of a photosite, which are of its own filter for the green photosites only: it is refused by
  name for the red and the blue (the weighted draw, `masks.gradient_bias`, reads any one filter's photosites).
- **Moved to VisualRadiometryToolkitCore (breaking): the board is its textured target.** `PrintedTarget` is `TexturedTarget` there
  (the same constructors), with its chart and geometry (`chart`, `world_point`, `corners`, `intersect_board`, `board_plane_hit`,
  `board_normal`, `inside`, `texture_size`, `with_sdf`, `has_sdf`, `posed_target`), its albedo (`ρ_c` is `albedo`, `𝔟` is
  `target_center`; the paper's symbols stay as aliases there), the printed-edge model (`edge_albedo`) and the warped page and in-plane
  print error (`HEIGHT_NC`, `HEIGHT_NODES`, `EDGE_NC`, `INPLANE_NC`, `warping_nodes`, `warping_aspect`, `warping_field`, `board_height`,
  `board_relief`, `BoardRays`, `inplane_calibration`, `design_chart`, `physical_chart`, `page_points`). They are no longer public
  names of this package; `PrintedTarget` is refused by name. The calibration's share stays here: `sphere_radius`, `HEIGHT_NFREE`,
  `height_nparams`, `height_manifold`, `height_basis`, `load_distance_map`, `TargetRadiometry`, `RadiometricTarget`.
- **Changed (breaking): the board holds its own state, and there is no scene.** `BoardSurface` (the warping h, the printed edge
  (δ, σ_print), the in-plane print error) is the textured target's own state, VisualRadiometryToolkitCore's `TargetState`, a field of
  the target replaced by value (`with_state(board, state)`, `with_h`, `with_edge`, `with_inplane` on the board). `Scene` is gone: its
  `board` and `surface` are the board, and its `W`, `H` were the camera's image size (`extent(geometric(camera))`), which is now
  stated once, by the camera. Every entry point takes the board where it took the scene — `CalibrationState(board, settings, camera,
  poses, …)`, `render_view(k, camera, pose, mat, light, board, settings, mask)`, `compute_mask(camera, pose, board, settings; …)`,
  `Φ_pixel(u, v, k, camera, pose, mat, light, board, settings)`, `pixel_parts(…)` — and `L_c(board, mat, light, sampler, key, 𝐛, 𝐱, ω)`,
  `radiance_parts(board, mat, light, …)` take the board with its state (no separate surface argument). `load_session` returns
  `board` (was `scene`). `BoardSurface`, `Scene` and `with_surface` are refused by name with the replacement. A mask or a photograph
  that is not of the camera's image, and a start camera whose image is not the run's photographs', are refused by name. The records
  are unchanged (`rc/radiometric_target`, `rc/calibration_state`, the start record: the same schemas and keys, read and written from
  the same values); no number of a calibration changes.
- **Changed (breaking): VisualRadiometryToolkitCore's public names are descriptive ASCII.** The paper's symbols are its public
  aliases of the same functions (`𝔭`, `𝔣`, `π̂⁻¹`, `t̂`, `𝔱`, `𝐩_of_pixel`, `pixel_of_𝐩`, `ℐ_𝐮`, `𝐩_sample`, `𝐚_sample`, `𝐚_weight`,
  `𝐚_border`, `Θ_m_nonlinear`, `with_μ`, `𝛂`, `𝛈`, `Λ_ggx`): this package's source keeps them. The plain spellings changed and have
  no alias: `chief_dir` → `chief_direction`, `D_ggx` → `ggx_distribution`, `G1_ggx` → `ggx_masking`, `G2_ggx` →
  `ggx_masking_shadowing`, `F_dielectric` → `fresnel_dielectric`, `sample_vndf_caps` → `sample_visible_normal`, `D_v` →
  `visible_normal_density`, `PDF_reflect` → `reflection_pdf`, `f_CT_parts` → `cook_torrance_terms`, `f_CT` → `cook_torrance`,
  `t̂_dir(focus, d)` → `t̂(focus, d)` (`inverse_focus_distance`); `ae_chart` and `ae_dir` are gone (VisualGeometryToolkitCore's
  equidistant lens, `project` / `backproject` of `Lens(Equidistant())`). Each old spelling is refused here by name with what to call.
  The radiometric back-projection `π̂⁻¹(camera, 𝐩, 𝐚)` is a method of VisualGeometryToolkitCore's `backproject` (the aperture point
  its third argument; `π̂⁻¹` is an alias of `backproject`); on a camera's parts it is VisualRadiometryToolkitCore's
  `aperture_direction`.
- **Changed (breaking):** `Material`'s fields and constructor keywords are ASCII: `roughness_ink`, `roughness_paper`,
  `refractive_index_ink`, `refractive_index_paper`, `diffuse_weight` (were `𝛂_ink`, `𝛂_noink`, `𝛈_ink`, `𝛈_noink`, `μ`). The records'
  JSON keys (`alpha_ink`, `alpha_noink`, `eta_ink`, `eta_noink`, `mu`) and the options' TOML keys are unchanged.
- **Changed (breaking):** the vignetting fields and the light (`TensorProductVignetting`, `RadialVignetting`, `EnvironmentLight`,
  `SphericalHarmonicMap` and their functions) are VisualRadiometryToolkitCore's, no longer VisualGeometryToolkitCore's: every use
  here names that package.
- **Changed (breaking):** a camera composed from its geometric camera has NO vignetting:
  `RadiometricCameraModel(geometric; focus, aperture_radius)` holds VisualRadiometryToolkitCore's `NoVignetting()` (V ≡ 1, no
  parameters; was the flat 4 × 4 surface). It renders as it is — the flux kernels hold no multiplication by V, on the host and on a
  device, and the image equals the flat surface's bit for bit. A calibration estimates V (Θ_v under the gauge V(𝐜) = 1), so
  `CalibrationState` refuses such a camera by name; the calibration's flat start is asked for:
  `vignetting = flat_vignetting(geometric)` (new, public; was the internal default).
- **Changed:** the kernels' `View` forwards the camera-frame maps to VisualRadiometryToolkitCore's one implementation on the parts
  they read (`π̂⁻¹(view, 𝐩, 𝐚) = π̂⁻¹(view.geometric, view.focus, 𝐩, 𝐚)`, …); the image formation's chief ray of a geometric camera is
  VisualRadiometryToolkitCore's `chief_ray` (was the internal `_chief_ray`).
- **Removed:** the refusal `lens(camera)` of a radiometric camera (the camera type is another package's: a method here would be type
  piracy); `lens(state)` is still refused by name. The record functions of a camera are RadiometricCamera's own (`_camera_body`), not
  methods of `JSON.lower` / `JSON.lift` on another package's type; the records are unchanged.

Performance and composition (2026-11-20; in this release too, and written before the port above: the names it adds —
`Focus`, `with_geometric`, `with_focus`, `with_aperture_radius`, `geometric`, `focus`, `aperture_radius` — are
VisualRadiometryToolkitCore's in 0.2.0, and the composed constructor's default vignetting is `NoVignetting`, as stated above). No number of a calibration changes beyond rounding:
the residuals are the same bits, the Jacobians' columns move by at most 2.2e-14 of their norm (the goldens hold at rtol 1e-10 on the
CPU, where this was measured; the device Jacobian's chunks round differently in Float32, see the golden re-record under Unreleased).

- **Added:** the radiometric camera composed from its geometric camera: `RadiometricCameraModel(geometric; focus, aperture_radius,
  vignetting)` — the focus surface (a `Focus`, or a distance: focused there everywhere), the aperture radius and the vignetting
  (V ≡ 1 by default: the flat 4 × 4 quadratic B-spline surface on the image); the image size is the geometric camera's.
  `Focus(geometric, distance; n = 3)`; `with_geometric`, `with_focus`, `with_aperture_radius` beside `with_vignetting`;
  `geometric`, `focus`, `aperture_radius` of a camera. The positional constructor is unchanged.
- **Changed (breaking):** a radiometric camera's geometric camera is its `geometric`: the field of `RadiometricCameraModel` (was
  `lens`), `geometric(camera)` and `geometric(state)` (were `lens(camera)`, `lens(state)`), `with_geometric` (the replacer). `lens`
  is VisualGeometryToolkitCore's lens model everywhere: `lens(geometric(camera))`. The old spellings are refused by name with the
  replacement (`with_lens`, `lens(camera)`, `lens(state)`; RadiometricCamera has no `lens` of its own). The records are unchanged:
  the geometric camera's key in `rc/radiometric_camera/v3.1` and `rc/calibration_state/v3.1` is `"camera"`, as before. The start
  records' and `load_session`'s `lens` entries (a start's camera, the `lens = :grid | :camera` choice) keep their names.
- **Changed:** a ray's hit with the board is found once per linearization. The radiance record keeps each ray's board point and the
  surface's normal there (`RadianceJac`: `[L | ∂L/∂ℓ | 𝐱 | 𝐧]`); the Jacobian chunks meet the recorded hit's tangent plane with their
  dual ray (the implicit-function theorem's partials, no Newton) and the parts kernel takes the recorded hit (`_ray_hit`,
  `_parts_hit`). On a warped board every chunk and the parts ran Core's height-field Newton at every ray before.
- **Changed:** a flat board whose heights are constants of a dual type takes the plane hit (`is_warped` reads the numbers, not the
  type): a Jacobian chunk of a flat board ran the height field's Newton at every ray.
- **Changed:** the nodal heights take no dual slot (`_jacobian_layout`, `_tile_layout`): the CPU backend's one chunk was 25 partials
  wider than its used slots, and split in two, with a free board warping.
- **Changed:** the per-pixel kernels and the Jacobian's row kernel run in workgroups of 16 pixels on the CPU backend
  (`_pixel_workgroup`, `_launch_pixel_rows!`): every thread works on a view of a few thousand pixels (KernelAbstractions' default
  group of 1024 items left a 1,728-pixel view to two threads).
- **Added:** `benchmark/` (a member of the package's workspace): `run.jl` times every stage of a calibration on one scene (CPU or
  GPU; a JSON record per run, `compare.jl` sets two side by side), `linearization.jl` one view's linearization kernel by kernel.

### experiments

- **Changed:** the texture page's operators are built on the device, where the rows are computed (`page_footprints`,
  `fit_operator`, `csr_rows`, `csr_transpose`): the rows compacted, selected, weighted and transposed by row-parallel kernels and one
  device sort, into the arrays Core's `PenalizedOperator` holds — array for array the host assembly's, so the textures are the
  same numbers. `page_rows` remains as the host form. `tools/texture_benchmark.jl` times the page solve on a synthetic sheet.

## 0.1.0

First public release, the first version registered in CauRegistry (not in Julia's General registry). Its code is that of 0.11.6, the last version of the
earlier private registry; the sections below are the history of those private versions (0.2.0–0.11.6).

## 0.11.6

Everything to reproduce the paper, and the exploratory studies behind it (owner 2026-11-18). The package code
(`src/`) is 0.11.5's; `bin/refine.jl --repeat` prints two more lines (below).

### experiments: the final paper and its experiment groups

- **Added:** the paper — the Blens manuscript (with T. Harder's Overleaf update of
  2026-11-18), merged from `ties-merge`. REPRODUCE.md and `configs/run_of_record.toml` follow its sections: the main paper (Sec. 1–5),
  the supplement (Sec. A.11–A.14) and the studies; every file the paper reads is indexed — ours, or external with its generator
  (T. Harder's figure and table generators; the detector-bias figure an external asset).
- **Added:** experiment groups beside `configs/experiments/`: `configs/supplement/` (experiments of record whose assets appear in the
  supplement alone) and `configs/studies/` (ideas tried that never came to fruition, each with its rationale and a required `decided`
  line: the sampling strategy and its three sample-count side studies, the lens-coverage margin, the warping ablation, the fisheye
  30 cm smoothness study). Their products lie under `results_root()/<group>/…` (`product_experiment`, `variant_location`): a location,
  never a key. The stochastic LM (branch `exp-stochastic-lm`) is referenced in REPRODUCE.md.
- **Added:** the timing experiment (`configs/supplement/timing.toml`, `scripts/timing.jl`; owner 2026-11-18): `si30_spline_warped`'s
  refinement and the calibrated camera's write, timed from its loaded start, 8 replicates one per B200 with a warm-up each; the record
  (n, mean, std, min, max, iterations, time per iteration, photographs, resolution, samples, observations, free parameters, GPU,
  threads), the supplement's runtime table in the paper's style (`runtime-table.tex`), a per-run table, a statement and macros.
  `bin/refine.jl --repeat=N` prints a `TIMING r/N key=value …` line per repeat (the refinement's and the write's seconds, iterations)
  and a `PARAMETERS …` line (the LM's free parameters per block).
- **Added:** Figure 5 (`fig:convergence-rms`) as the convergence driver's stage `panels`: the detections' true reprojection median and
  the held-out Thorlabs median per LM iteration, si30_spline_warped and ag30_spline_warped, in T. Harder's rms_figure.py style; the
  submitted files are reproduced from their medians (every line but the generator line).
- **Added:** the OpenCV starts of 2026-11-17 as a recorded front-end step (`configs/inputs/opencv_starts_20261117.toml`: OpenCV
  4.12.0, the tool of record, inputs, arguments, outputs) and the tier `rc_baselines_2026-11-17` in `configs/data_artifacts.toml`.
- **Changed:** `samples_per_view` in ONE place per model kind (`[kinds.lens]` 8,192, `[kinds.grid]` 16,384 in
  `radiometric_refinement.toml`, owner 2026-11-16 23:30), inherited by every case (`inherit_kinds!`; a restatement refused); the
  resolved cases, and so the product keys of the run of record's configs, are unchanged.
- **Removed (git history keeps them):** the panorama chains (panorama_stitching, panorama_vignetting, vignetting_agreement,
  vignetting_figures, the flat field), our synthetic detector bias (detector_bias, make_detector_bias_inputs, ground_truth), the
  colour-wheel deploy config, and the commented-out excluded tables and EXCLUDED blocks of the configs of record.

### Known issues (owner 2026-11-18; fixed in the next version, verified by the deferred run of record)

1. **The aperture start reads a nominal focal length from the config instead of the start camera.** The refinement's starting aperture
   radius is R₀ = f / (2N) with f = `camera.f_mm` (`src/start/prepare.jl`, `src/start/start_calibration.jl`), although the start's own,
   regressed focal length is at hand (`fpx_of(lens0, W, H)` × the pixel pitch). For the fisheye 30 cm Siemens cases this is 7.8057 mm
   while their front-end fits seed with 8.0 mm (the EXIF value): the same lens's nominal focal length stated twice, with two values
   (marked `@test_broken` in the suite). Only starting values differ; every reported camera is regressed. Next version: R₀ from the
   start's focal length, `camera.f_mm` removed from the refinement, one nominal focal length per lens (for the fits' initial guess only).
2. **The run of record of 0.11.6 has not run** (dgx01 offline after 2026-11-18): the release was gated on the test suites. The deferred
   run must reproduce the results of 2026-11-17 (RC 0.11.5) (Tab. 2/3, supplement A4/A5, numbers.tex); every product is new by code, the
   same by config.
3. **External assets:** most of the paper's figures and tables are drawn by T. Harder's generators (experiments/pgf, tools), not by a
   driver stage; the detector-bias figure (Mitsuba renders) is made outside this repository. Listed in REPRODUCE.md.

## 0.11.5

The final release for the paper and the base of its run of record (owner 2026-11-16: what is written, merged; nothing new).

### experiments: the boards' distance maps are a keyed step, never an input of record (owner 2026-11-16)

- **Changed:** the printed boards' edge distance maps (`edge.sdf`; the detector-bias ground truth's `distance_map`) are produced by
  `experiments/scripts/distance_maps.jl` (`configs/experiments/distance_maps.toml`, stage `distance_maps` of the stage graph): one
  keyed step per map, keyed by the board generator's `sdf.f32` + `sdf.json` (inputs of record, pinned by sha256) and the code of Core
  and IO, written through IO's `save_raster`. Every consumer (the refinements', the vignetting refinements', the detector bias's
  cases and ground truth, the golden calibration case) reads a map as `[runs.<id>]` (`edge.sdf = "run:<id>"`), produced on demand;
  the calibration configs take a `[runs]` table. `scripts/make_distance_maps.jl` (the one-off) and the maps' `[inputs]` entries are
  gone. The raster payloads equal the one-off's bit for bit; the record's provenance names the data-root-relative source.

### experiments: fig:vignetting's flat field on the RAW green photosites (owner decision 2026-11-16)

- **Changed (experiments):** the diffuser flat field of fig:vignetting is read off its RAW green photosites, the photosites the learned V
  is regressed on — no demosaic, no interpolation, no smoothing (the 9 × 9 box and the 60-px centre mean are gone): RC's entry point
  under the learned camera's own record (both greens, the record's black, defects missing, censored photosites excluded), RC's
  `masked_observations` and `𝐩_of_pixel` (each photosite at its sensor position), binned by the distance from 𝔠 (the learned V's
  centre) with the learned V over the same photosites, both normalised over the centre disc, the flat field's σ from the sensor
  model (Core `Uncertain`), and ln(V / flat) ± σ (src/flat_field.jl; the figure's new middle row). `[flat_field] observation =
  "green_photosites"` is required (another value or none is an error naming the key); `bin_px`, `rim_px` are its keys; the
  `[figures.vignetting]` keys `blur_px`, `centre_px`, `rim_px`, `diagonal_spacing_px`, `diagonal_ymax` and `[lenses] circle` are
  retired (`profile_ymax`, `ratio_min_range` new). The panorama's comparison (ln(V̂ / V), its flat stage) is left as it is for
  0.11.5: a documented deviation. The figures stage writes each lens's profile, `flat_field_<lens>.json`.

### Core 0.11.0: a tolerance stop of the LM is converged (owner ruling 2026-11-15)

- **Changed (with VisualGeometryToolkitCore 0.11.0):** an LM stop on `[lm] ftol` (`CostDecrease`) or `[lm] xtol_rel`
  (`IncrementSize`) is `ReturnCode.Success`, with no stationarity postcondition (until 0.11.4 one away from a stationary point was
  `Stalled`). The iteration and time caps stay `MaxIters` / `MaxTime`; the rejection cap is still judged by the postcondition. A
  run record of 0.11.4 or earlier keeps its recorded `Stalled` (rerun to read it under the new rule).
  `[lm] ftol = 5.0e-4`, `xtol_rel = 1.0e-8` (0.11.4) unchanged. RC had no stop-reason re-mapping of its own to drop; its fold
  check (`fold_termination`, the calibrated camera over its coverage) is RC's and stays.

### experiments: the paper reports no convergence (owner rule 2026-11-16)

- The generated tables (tab:refinement-chrome, tab:defocus, the chrome tables) carry no † and no "did not converge" footnote;
  the `\nonConverged<Table>` / `\nonConvergedTotal` macros and `[render] nonconverged_prefix` are gone. The paper prints the
  test-set error only. Every product keeps its termination record (retcode, stop rule, iterations; the tables' tsv/json
  `converged` and retcode columns, index.tsv) for debugging.

### The review's consolidations (RC 0.11.4 review item 6)

- **Renamed (breaking):** `SensorModelAuto` / `SensorModelOn` / `SensorModelOff` → `AutoSensorModel` / `OnSensorModel` /
  `OffSensorModel`; `SENSOR_MODEL_MODES` → `SENSOR_MODEL_TAGS` and `LATTICE_RULES` → `LATTICE_TAGS`, both Core `TagRegistry`s
  (the stack's tag convention). The TOML spellings are unchanged.
- One reader of a type-dispatched table (`typed_table_type`): the smoothness tables and `[masks] edge_detector`.

### Fixed: the start's slow mask test and edge copies, the chrome bootstrap's repeated replicates (the 0.11.4 review's item 7)

- **The mask test in the draw's key order** (`draw_samples` through `BoardValidity`, `board_valid`): every candidate of a view
  (the sampling region's pixels the observation model keeps) gets its uniform key, and where no weight reads the edge ranks
  (α = β = 0: the uniform draw of record, the rule of 2026-10-31's pool, the uniform add-on) the draw is the n smallest keys among
  the board-valid candidates — so the candidates are tested in key order, in batches sized by the view's own valid share, until n
  are valid: the same draw as testing every candidate, a few n pixels tested instead of the whole region (RC 0.11.4 tested every pixel of
  the region, ≈ 50× 0.11.3's Bernoulli candidates, and rebuilt the region's id list per view). α ≠ 0 or β ≠ 0 tests every
  candidate (the ECDF ranks are over every usable observation). **The stream changes:** a view's keys are over its candidates (not
  its usable observations), `rand(rng, #candidates)` per view in view order whatever the mode (a short view consumed none before):
  every case's sample changes, the goldens are re-recorded with 0.11.5's. The masks' record and `observe` are unchanged.
- **The edge detector's host copies:** `edge_plane` makes the one `Float32` plane (an uncertain image's values read in place; the
  `_plain_values` copy is gone), the support crosses as its `Bool`s, g comes back once (`Float32`, the same ranks): one or two
  full-frame host copies per view instead of 4–5, the same strengths bit for bit.
- experiments: **the chrome evaluation's paired photograph bootstrap resamples the photographs once per replicate** for the four
  statistics of a comparison, variant and region (`paired_difference`): their `bootstrap_ci` calls share one stream,
  `StableRNG(seed + g)` with `g` over comparisons × variants × regions, and a replicate's residual summaries are computed once. Each
  interval keeps its definition (Eval's BCa interval of `resamples` photograph draws); its Monte Carlo draws change (RC 0.11.4: each
  statistic its own stream `StableRNG(seed + j)`), so paired.tsv's bounds move within their Monte Carlo error.

### Compat floors (the Core 0.11 cascade)

- VisualGeometryToolkitCore 0.11.1 (an exact fit of the host LM is converged again: 0.11.0 read it Stalled, NonFinite), IO 0.10.6, Estimation 0.12, Features 0.9.8 (the chrome detect's FoerstnerTests table), Eval 0.9.2;
  experiments: PanoramaStitching 0.10.0, BlobBoards 0.7.2 (with LocalOptimizationDoneRight 0.7.1, RANSACScoringDoneRight 0.8.1).

### Known issues

- **On the 14 mm f/2.8 texture calibration, the focus surface's right-hand corner nodes converge to near infinity** (t̂ ≈ 0.007):
  the data there ask for more distance-independent blur than the focus model provides; an unboxed focus surface is under review.
  The case converges (Core 0.11.0's ftol stop: 12 iterations, Success, rel-RMS 0.2925 → 0.1868, no fold) — not a runaway.
- **The 14 mm learned vignetting is slightly negative on the last ≈ 2 px of its lens coverage** (V ≈ −0.007 at the rim of the
  1724.5 px coverage). 0.11.5 keeps the vignetting model of record: the log parametrization (rc-vignetting-rim) removes the
  negativity but agrees worse with the diffuser flat field at the rim (max |ln(V / flat)| 14 mm 0.061 → 0.144, fisheye 0.172 →
  0.192, Tamron 0.056 → 0.067); the trade is the owner's modelling decision, after 0.11.5.
- **The pipeline golden and the docs build are re-recorded after registration** (owner decision 2026-11-16: no gate before
  registration): the golden of 0.11.4 does not match 0.11.5's samples (the uniform draw, the key-order stream); a follow-up commit
  (0.11.6 if needed) records both.
- **The chrome products are keyed by the whole experiments src tree**: any src change re-detects every set (post-0.11.5: key each stage
  by its own code).

### Breaking: the samples are the uniform draw again, `[masks] gradient_bias = 0` (owner decision 2026-11-16)

- **The default and every config of record take α = 0** (the uniform draw, bit for bit; no edge detector run), replacing 0.11.4's
  α = 1. Derivation, the sampling-strategy experiment's record (paired 95 % intervals, α = 1 − uniform): on the 14 mm the texture
  recovered at 1 mm +0.0099 [+0.0092, +0.0106] and the Thorlabs held-out median +0.0046 px [+0.0000, +0.0097], both worse; on the
  fisheye f/4 indistinguishable on the median, +0.040 px at p95. The ECDF draw (`gradient_bias ≠ 0`, `[masks.edge_detector]`)
  stays an option: the sampling-strategy experiment's arms state it. Every case's sample changes: the paper's runs and the pipeline
  goldens are re-recorded; the config lock is hyperparameters_0115.toml.

### The samples' draw: the sampling-strategy experiment's modes (rc-sampling-strategy)

- **New `[masks]` keys, default off:** `edge_fraction = q` (the rule of 2026-10-31, restored verbatim from RC 0.11.4 as a mode
  of the one `draw_samples`, with `gradient_bias = 0` only; no longer refused) and `addon_samples = M`, `addon_gradient_bias = β`
  (M more samples per view ∝ u^β from the other usable observations, on the same stream: the n are the draw without the add-on,
  bit for bit). `check_draw` refuses an inapplicable combination.
- `masks.importance_weights` and the edge detector's `l0` type are refused by name.
- **Fixed:** the row weighting of a sparse design matrix is its row scaling `Diagonal(w) * A` (`_whiten`): the broadcast
  `w .* A` allocated the full shape and the texture page's noise-weighted solve ran out of host memory (7.5 M rows × 1 M texels,
  dgx01 2026-11-15); the same products bit for bit. SparseArrays is a dependency again.
- experiments: the sampling-strategy experiment (`scripts/sampling_strategy.jl`, `configs/experiments/sampling_strategy.toml`);
  `expand_overrides` names a table-valued override by its sorted keys.

### experiments: the texture recovery's cross-validation (rc-texture-0111, through rc-sampling-strategy)

- The page solve weights its rows by the sensor record (whitened RAW rows), the split's rotations (`[folds]`, `folds`,
  `fold_report`), the comparison of page runs by one scorer as IO records and rasters (`[compare]`, `compare`) and `summary.md`
  (`report`). The L0 regulariser of that branch is not in 0.11.5 (no config read it; RC 0.11.4's trim).

### experiments: the vignetting agreement experiment (owner 2026-11-15)

- `scripts/vignetting_agreement.jl` (`configs/experiments/vignetting_agreement.toml`, `src/vignetting_agreement.jl`; reproduce stages
  `vignetting_agreement`, `vignetting_agreement_summary`, run only when named): per lens ln(V / reference) per 5° ring with bootstrap
  intervals for the learned V of the vignetting case of record, its variants as overrides of it (H2: lighting, material, aperture held;
  H4: the 0.10.1-like sampling and chart; a second seed) and the lens's refinements at other board distances (H1′; the case of record's
  near and far views too), against the sweep's fit_vignetting V̂, PanoramaStitching's model-free pair-equation V̂
  (`ring_pair_vignetting`, branch pair-vignetting), both with the sweep's additive level removed (`fit_additive_level`), and the flat
  field cut at θ_max (H3). Needs the PanoramaStitching release that carries `pair_correspondences`.

### experiments: the board warping ablation (code and config; it runs after 0.11.5)

- `scripts/warping_ablation.jl` (`configs/experiments/warping_ablation.toml`, `src/warping_ablation.jl`; reproduce stage
  `warping_ablation`, run only when named): every global-model case of the paper's chrome sets and its capture set's central
  spline refined with a rigid board, the heights only and the config of record, scored by the chrome evaluation's held-out
  protocol. `override_variant` is the one rule of an experiment's variant (`expand_overrides` and the ablation's modes).

## 0.11.4

A patch release (owner, 2026-11-15: bugs and defects first; enhancements in 0.11.5). The compat floors: VisualGeometryToolkitCore
0.10.11, VisualGeometryToolkitIO 0.10.5, VisualGeometryToolkitEstimation 0.11.9; the paper project also PanoramaStitching 0.9.0.
Estimation 0.11.9 restores the exact start of a spline source (a Schöps camera's resample): under Estimation 0.11.8 its support came
out rounding off the lattice's area and the start's coverage assertion refused the sensor (`RegionBeyondSupport`).

### Bug fix: a parametric start of a lens that is not radially symmetric on Core 0.10.10 (its polygon valid domain)

- **Fixed:** Core 0.10.10 gives a lens without radial symmetry (Brown–Conrady, OpenCV, a tilt stage) a star `PolygonRegion` valid
  domain. The node-count lattice of a parametric start moves its image region into the lattice's pixel frame, and had no method for
  a polygon: a `MethodError` in `_session_start` (the release session's end-to-end job 19974). The region now moves through Core's
  `SimilarityMap` action, one method for every region Core has, which replaces RC's three hand-written translations. The initialization
  region (sensor ∩ polygon) saves and reloads through IO 0.10.4's `"polygon"` kind.
- **Compat floors:** VisualGeometryToolkitCore 0.10.11 and VisualGeometryToolkitIO 0.10.5 (`PolygonRegion` and IO's `"polygon"` kind
  do not exist below Core 0.10.10 and IO 0.10.4; the projective knots below need 0.10.11 and 0.10.5).

### Bug fix: a chrome dot the start pose misses is re-tested at the estimate (owner 2026-11-15)

- **Fixed:** `evaluate_grid_models` dropped every dot the rotation-centre START pose failed to project, so it was never scored, even
  where the refined ESTIMATE projects it. On the fisheye ag10fe set this lost 53 rim dots of the refined spline model: the detections
  are inside its lattice, only the rough start pose (median 14 px off) pointed outside its field. Such a dot is now left out of the
  fit (it has no start residual) and scored at its photograph's estimate; only a dot the estimate does not project either is counted
  in `unprojected_at_estimate` and not scored. A refused photograph (no dot projected at the start) is still not scored.
- The eval record's `start.dropped_unprojected` is renamed `start.unprojected_at_start` (same count; those dots are no longer dropped).
- **The paper tables' dot counts move** for every model whose start pose missed dots of a photograph it refines (the rim of the
  fisheye spline models): its `n`, coverage and per-photograph counts rise, `unprojected_at_estimate` may rise, and the head-to-head
  common-dot set can grow.

### The samples' edge bias: a weighted draw ∝ u^α of the edge strength's ECDF rank, α = 1 (owner decision 2026-11-15; rc-edge-sampling, trimmed)

- **Breaking: `masks.edge_fraction` is retired** (refused by name, the new keys named): its hard truncation (the strongest-|∇ϕ|²
  share q of a uniform pool) excluded flat regions entirely, and on the fisheye f/4 it scored +0.097 px [+0.070, +0.138] WORSE than a
  uniform draw on the Thorlabs chrome (paired held-out). Instead each view's `samples_per_view` usable observations are drawn without
  replacement with probability ∝ u^α: **`[masks] gradient_bias = α`**, default **1** (the same scoring: +0.006 px [−0.003, +0.019]
  against uniform, indistinguishable, and the focus dome holds); 0 = uniform, > 0 favours edges. u is the ECDF rank of the
  photograph's edge strength over the view's usable observations (StatsBase's `tiedrank` / n; invariant to contrast and exposure).
  The draw is `_subsample`'s weighted generalization on the same stable stream (exponential keys −log(1 − U)/w of the same
  `rand(rng, n)`): α = 0 is the uniform draw bit for bit.
- **New: `[masks.edge_detector]`** `{type = "gaussian_derivative", sigma = 1.5}`: Features' `gaussian_gradient`, normalized over the
  usable observations (a RAW mosaic's green photosites alone). The branch's L0 band detector and importance weights are NOT in 0.11.4
  (the sweep showed no gain; owner/controller ruling); L0 stays with the texture recovery's own work. New dependency: StatsBase.
- **Configs of record migrated:** `edge_fraction = 0.37` → `gradient_bias = 1.0`, each with its derivation; the golden
  siemens_coarse_full states `gradient_bias = 0.0` (its uniform draw, as recorded). `_edge_select` removed. Every case's sample changes.

### The focus surface on the imaged region (bug fix, Ties 2026-11-15: the fisheye f/4 focus surface no longer a dome since 0.9.0)

- **Fixed:** the focus surface's nodes, support and box are laid over the start lens's IMAGED region again (`focus_start_nodes`),
  never the initialization region ∩ the imaged region (0.9.0): on the fisheye f/4 the clipped box put four nodes on the
  region's border, seen by few pixels, which ran to infinite focus. The initialization region keeps its other consumers (the pose
  fits, the self-checks, the grid's record).
- **New: a rim margin for the focus nodes**, `[focus] lens_coverage_scale = 0.9` (named `node_rho_max` on its branch): the 5 × 5 grid's nodes are kept within that share of an
  image circle's normalised radius, so the box's side midpoints (ON the rim) are dropped by design, whatever the ellipse's
  orientation — before, membership alone kept 9 of a rotated circle's (0.006 px outside) and 13 of an axis-aligned one's. The
  sensor's rectangle keeps every node.
- **New:** a focus node of the run's support refined to t̂ ≤ 0 or non-finite flags the calibration non-converged
  (`focus_runaway`; one `validity_gate` with the fold check: `(Failure, ShapeGate)`; `CalibrationMetadata.focus_runaway`).
- **New: one support rule for a spline's nodes** (`node_support`, `data_weight`): the vignetting's `support_min` rule, shared;
  `[focus] support_min` (default 0, the rule before) makes a focus node below that share of the mean data weight dependent (the
  TPS is built on the supported nodes).
- **Breaking: the samples are a fixed count of the usable observations** (owner, 2026-11-15): `[masks] samples_per_view = 27500`
  (the fisheye f/4 dome run's 27 548 per view), drawn per view uniformly without replacement (`draw_samples`, StableRNG
  `masks.seed`) from the board-valid pixels of the sampling region that the photograph's observation model keeps (a RAW photograph's
  green photosites, no defect, uncensored) — so a small image circle and a lens filling the sensor get the same count; with
  `edge_fraction = q` the strongest-|∇ϕ|² share q of a uniform pool of N / q of them. A view with fewer uses them all; its shortfall is
  recorded (`sample_record`: the start record's and parameters.json's `samples`). `masks.lm_fraction` and
  `masks.max_pixels_per_view` are retired, refused by name. Cause: the Bernoulli share of the raw grid, filtered to the green
  photosites, gave the photosite route half the RGB route's samples (13 871 against 27 548 per view on the fisheye f/4), and with half
  the data a field-angle focus node ran to 3 m. Configs of record migrated: each case's binding cap is its `samples_per_view`; the
  fisheye f/4 takes 27 500 (its cap of 31 746 was not binding). Every case's sample changes: the pipeline golden is re-recorded.
- The CVT node layout of a scattered-node spline (weighted k-means, a Clustering dependency) is not in 0.11.4: no pipeline step
  called it (owner ruling 2026-11-16).

### Bug fix: the 14 mm slowdown — a Schöps start is his own lattice, never subdivided; `[grid] lattice` (owner rulings 2026-11-15)

- **Fixed:** since 0.11.0's exact start a Schöps start was resampled at `grid.lattice_spacing` (default 40 px) and its case's
  `grid.size = [35, 25]` was SILENTLY not read: T. Schöps's ≈ 156 × 165 px cell (texcal_14mm_spline_warped, schoeps_green_w25) was
  subdivided 4 × 4, ≈ 8–9.6 k grid nodes, and the single-threaded host LDL of the LM's normal equations took ≈ 200 s per LM step with
  the GPU idle. His own lattice (m = 1) runs 88 s/it at 97 % GPU, rel-RMS equal to 3 digits (0.2879 → 0.2360 → 0.2070 at 40 px,
  → 0.2362 → 0.2072 at his cell; dgx01 A/B jobs 19625/19626).
- **New key `[grid] lattice`**, the geometry lattice's mode, parsed ONCE into a type (`LatticeRule`: `DerivedLattice`,
  `NodeCountLattice`; `LATTICE_RULES`, through `read_switch`), every decision a dispatch (`start_grid`):
  `"derived"` (the default) — the start source's OWN lattice extended over the sensor ∩ lens coverage, never subdivided
  (`exact_start(source; coverage, device)`, m = 1); `"node_count"` with `grid.node_count` — a lattice laid over the image region and
  sized by the node count (`region_start`, the parametric start's route; `node_count` has no default). A SplineLens source under
  "node_count", a parametric source under "derived", "node_count" without its count, a count under "derived" and "node_count" for a
  Schöps front end are refused, naming the key and the fix. Derivation of the default: every grid start of record but the blob front
  end's parametric session cameras is a Schöps start (the A/B above).
- **Breaking:** `grid.lattice_spacing` and `grid.size` are retired, refused by name, naming `grid.lattice` (`grid.size` was ignored on
  the exact-start route since 0.11.0); `exact_start`, `schoeps_start` and `schoeps_gridlens` take no spacing (`schoeps_start(yaml,
  W, H; coverage, device)`). Subdivision stays Estimation's (`fit_camera(SplineLens, …; lattice_spacing)`), RC never asks for it.
- **A target for the exact start (owner 2026-11-15, 23:25):** under `"derived"`, `grid.node_count` is the target of PRESENT nodes over
  the image region: m is the integer whose present count comes closest to it without exceeding it (`exact_subdivision`, Core's exact
  knot insertion: the start stays his camera), and the target is the present-node bound (his own lattice above it is refused); the
  start logs `grid: N present of M, m = k (target T)`. No target: m = 1. Under `"node_count"` the count is likewise a bound
  (`region_spacing` takes the largest count at most it; `region_start` refuses more).
- **experiments:** every Schöps-started case of record states `lattice = "derived"` (its hand-computed `lattice_spacing = 131.0`, which
  already gave m = 1, removed); the golden siemens_coarse_full's `lattice_spacing = 160` (m = 1) is `lattice = "derived"`, bit for
  bit. The cases whose start is a parametric record (`io.init = "lens:…"`) state no grid key (refused there: below).

### A parametric start's lattice sized by the image region and a node count (owner, 2026-11-15)

- **`grid.lattice = "node_count"` with `[grid] node_count`** (e.g. 7875 = 105 · 75, the nodes of the lattice of record): a PARAMETRIC
  start's geometry lattice (Kannala–Brandt, Brown–Conrady, Double Sphere, OpenCV, division: the blob front end's session camera) is
  laid over the bounding box of the image region (sensor ∩ lens coverage, the image-region step's record) with square cells, one
  integer pixel spacing for both axes, chosen so the nodes the region supports (Core's present-node rule, `node_mask`) come nearest
  `node_count` (`region_start`, `region_spacing`, `region_layout`). The nodes the region does not support are absent (never a
  parameter), and the lattice is padded to the grid route's patch multiple by the one `pad_lattice` (`GRID_PATCH`). A fisheye whose
  image circle lies inside the sensor now spends its nodes inside the circle; on the FX3's whole 4256 × 2848 sensor 7875 is a 41 px
  cell, 107 × 73 = 7811 nodes, padded to 110 × 75.
- The controls are Estimation's public `fit_camera(SplineLens, source, region; grid_size, forward = false)` on the source in
  the pixels of the lattice's box (its principal point and region moved by the box's origin; the fitted camera moved back): no
  fitting code in RC. **STOPGAP:** the crop-and-shift exists only because Estimation's fit takes no lattice area; FOLLOW-UP
  Estimation: a lattice-area keyword (Core's `spline_lattice` already takes `calib_area`), after which RC's crop (`_origin_shifted`,
  `_in_frame`) is deleted. The region moves through Core's `SimilarityMap` action (the polygon fix above), one method for every
  region Core has.

### Vignetting: the projective chart with 19 nodes; the flat field cut at θ_max; the symmetric saturation mask's margin (T. Harder's analysis, 2026-11-15)

- **Default `[vignetting] chart = "projective"`, `node_count = 19`** (was linear, 16), with Core 0.10.11's projective knots (T. Harder's
  rule: m = n − 2p interior spans): 16 functions act on the fitted pixels, at the linear 16's knot spacing, and p − 1 knots continue
  past the rim at that spacing — the resolution of record, no pixel ever clamped. The 36 vignetting tables of the experiments configs of
  record move 16 → 19; the golden siemens_coarse_full pins `chart = "linear"`, `node_count = 16`, so its golden does not move. Core's
  knots need Core 0.10.11 and IO 0.10.5 (the floors): a projective product made under Core 0.10.10 holds the older knots under the
  same keys and is re-made.
- **Fixed (experiments, T. Harder):** `flat_ring_profile(flat, c, lo, hi, tmax)` averages the diffuser flat field's bins only below the
  domain's field angle (`min(h, tmax)`), as the rooftop's rings are; `compare_stage` hands it its record's `theta_max`.
- **experiments, PanoramaStitching 0.9:** `fit_vignetting(…; saturation_margin_px)` is required (the symmetric saturation mask): each
  sweep of record states `[example.domain] saturation_margin_px`, √χ²₂(99 %)·σ ≈ 3 × its bundle RMS (Tamron 13 px; 14 mm and
  fisheye 4.5 px), pinned by the config lock. The paper project's floor: PanoramaStitching 0.9.0 (the required
  `saturation_margin_px` of `fit_vignetting` and `score_seams`, the symmetric saturation masks, `ring_correspondences`, the seam
  numbers on the fit's observations). The panoramas stage's `score_seams` call takes the sweep's margin (it was missing).
- **experiments, a known difference:** `log_seams`' per-pair numbers (seam_pairs_<slug>.tsv) use per-frame masks; the symmetric rule is
  `score_seams`'. PanoramaStitching 0.9 has no public API for the symmetric samples, so the pairs can differ from the score where
  frames saturate (the docstring says so).

### Bug fix (experiments): a `run:` input is checked, produced on demand, and a case reads only its own (owner 2026-11-15)

- **Fixed:** an input path `run:<id>` (e.g. `frontend.schoeps = "run:fx3_14mm_texcal_20261109_schoeps"`) was turned into a path
  unchecked, and a missing front-end run died at its first read with a bare `SystemError` (dgx01, 2026-11-15: texcal and sitla on a
  fresh results root). Now `resolve_pipeline` reads it through `config_source`: a missing run is produced on demand, and a run its
  maker did not produce is an `ArgumentError` naming the run, its key, its directory and the command that produces it.
- **Consumer-side produce (with_upstream, lifted from rc-texture-0111, the drivers.jl helper alone):** a resolver of
  `RUN_RESOLVERS` returns `(; product, make, command)`; `config_source` makes a missing run under its own key (`make`: the upstream
  driver in a module of its own, Eval's `run_isolated` — the front ends' on the run's set through the stage that makes it, each
  stage producing or loading its own upstream). A Schöps run (camera_calibration needs CUDA) is made only where the process holds a
  GPU (`require_gpu`): refused in a CPU job, naming the command, never run on the CPU. Keying a product never produces anything
  (`resolve_pipeline(…; produce = false)`).
- **Every upstream product produced on demand, one mechanism (owner 2026-11-15):** `produce_upstream(p, what; make, command, gpu)`
  (`with_upstream`, the GPU gate `require_gpu`, the post-check) is the one path from a consumer to an upstream product. It replaces
  `require_produced`, deleted, whose 14 refusals ("is not produced: run its stage first") each now produce: a case's decode and its
  init start, the front ends' decode → Schöps run → fit → comparison, the chrome set's detect and index, the convergence
  experiment's refinement, start and chrome index, its figure's record, the texture report's page solve, the panorama chain's run,
  and every measured image region (`lens_coverage_record`). A maker is the upstream's own stage function, or its driver in a module
  of its own (`driver_run`, which the `RUN_RESOLVERS` makers use too). A GPU stage reached from a CPU job is refused, naming the
  command. Locked by test/upstream_tests.jl: the mechanism, a table of the consumer → upstream pairs, and no `require_produced` or
  `produced(…) || throw` outside the mechanism and the host's --gpu gate.
- **Fixed:** a case's start and refinement checked EVERY `[runs]` product of their config (`require_inputs_through`): ag10fe died on
  a missing 14 mm front end it never reads. `run_inputs_through(cfg, t, stage)`: the config's `[inputs]` of those stages and only
  the `[runs]` the case's own `[pipeline]` names.

### The LM stops when the data stop resolving a decrease: `[lm] ftol = 5.0e-4`, `[lm] xtol_rel = 1.0e-8` (owner rulings 2026-11-15, 2026-11-16)

- Why no run stopped before its cap: the calibration LM's tolerances were Julia constants (`LM_RELDCOST = 1e-8`, `xtol = 1e-12`) —
  the 14 mm plateau decreases the cost by ≈ 1e-4 per step, 10⁴ × above 1e-8, so only `max_iter` fired.
- **New keys** `[lm] ftol` and `[lm] xtol_rel`, TOML literals read into Core's `LmStops` (`_lm_stops`; `LM_RELDCOST` deleted, no
  tolerance hard-coded; both refinement-only, `REFINE_ONLY_KEYS`). `ftol = 5.0e-4`: √(2/N), the cost's relative noise, from the
  largest-N case of record (texcal_14mm, ≈ 7.7e6 residuals): no case stops before its data resolve. `xtol_rel = 1.0e-8`: Core
  0.10.11's `LmStops(; xtol_rel)`, Ceres' relative step test ‖Δx‖ ≤ xtol_rel·(‖x‖ + xtol_rel), under Core's name. The absolute step
  test (Core's `xtol`, NLLSsolver's dstep) is Core's of record, `CORE_LM_XTOL`, with no key; `[lm] xtol` is refused by name, naming
  `xtol_rel` (it existed only in this release's drafts). `[lm] max_iter = 50` is the safety cap (the owner's ruling of 2026-11-11; configs/default.toml stated 30 until 0.11.4).
- **The termination record is Core 0.10.11's:** an `ftol` stop (`CostDecrease`) is `Success` as before; a step-size stop
  (`IncrementSize`, `xtol_rel` or `xtol`) is `Success` or `Stalled` by the stationarity postcondition, and the cap is `MaxIters` (†).
  RC maps nothing. A tolerance stop read as converged arrives with Core 0.11.0, in RC 0.11.5.
  **Every refinement's iteration count and final numbers move** (the golden re-record).

### A TOML value the run cannot use is an ERROR, never ignored (owner ruling 2026-11-15)

- `INAPPLICABLE_KEYS` (public, with `inapplicable_reason(opts, table, key)`): a key a config STATES that the run's other options make
  inert is refused by `pipeline_options`, naming the key and the reason; the writer (`config_dict`) leaves such a key out, so a run's
  record reads back. The keys: every `[grid]` key for a parametric record start (`io.init = "lens:…"`: refined per view, no
  lattice); `focus.node_count` for the variants A / B (their 5 × 5 grid);
  `focus.support_min` and `focus.lens_coverage_scale` for the variants C / D; `focus.smoothness` and `focus.perturb` with "focus" in
  `fixed`.
- **`grid.node_count` on a Schöps start** (`frontend.type = "schoeps"`, `grid.lattice = "derived"`) is refused by the same rule
  (owner ruling 2026-11-16): his lattice is extended, never subdivided, so no node count acts on it. It was silently ignored there.

### Release review fixes (owner rulings 2026-11-16)

- **`focus_runaway` after a resume** compares each node with the start's value at the same node (matched by position). A resumed
  state or a start record on another node set threw a `BoundsError` after the whole LM, or compared another node's start value.
- **The focus support record** (`focus_info[:support]`) describes the rebuilt node set after `focus.support_min` drops nodes (one
  weight and support entry per node of the run's focus), with the dropped candidates' count and positions.

### Cleanup

- `[focus] node_rho_max` → **`[focus] lens_coverage_scale`** (default 0.9): the same `_scaled_coverage` of the lens coverage about its
  centre as `[masks] lens_coverage_scale`; the old key is refused by name (retired.jl). No number changes.
- The 0.11.3 workaround `_containment_box` (the coverage assertion tested a sensor-clipped circle by its box) is gone: Core 0.10.9
  tests `RegionIntersection ⊆ Rect` by the intersection, and the assertion hands Core the region (Core ≥ 0.10.9; the release's
  floors are Core 0.10.11 and Estimation 0.11.9: 0.11.8's `optical_axis` order continuation and 0.11.9's exact resample support).

### The stack's switch convention: a switch is "on" | "off" (owner, 2026-11-15)

- **A switch of a config is a string** — `"on"` | `"off"` — parsed ONCE at the config boundary by `read_switch(v, key) -> Bool`
  (`src/options/toml.jl`, public): every `Bool` field of `PipelineOptions` (`io.linearize`, `io.checkpoint_each`, `io.render_init`,
  `rendering.hammersley`, `sweep.joint`, `resume.sweep`, `selfcheck.only`) and the paper project's switches (`[evaluate] holdout`,
  `[prepare.<set>] srgb`, `start_from_init`, `[evaluate] gauge`, `[deploy] replace_submitted`, `reference_band`, `[tables.<name>]
  confidence`). The API keywords stay `Bool`. The writer (`config_dict`, the run's pipeline_config.toml) spells them "on" / "off".
- **TOML true/false is refused** for a switch, naming the new spelling (`holdout = true` → use `holdout = "on"`;
  `retired_boolean_switch`, `src/options/retired.jl`). No alias. `--set table.key=on` overrides one.
- **Every config of record migrated** (configs/default.toml, experiments/configs/**, the test fixtures), comments kept. Products keyed
  by a migrated table are re-keyed (no number changes).
- **A lint**, `boolean_switch_violations(dirs; data)` (public): a TOML boolean not declared data is a finding; the RC configs and the
  paper project's configs are linted (pipeline_options_tests.jl, experiments/test/calibration_tests.jl). The booleans kept: facts and
  keys of other packages' schemas (Eval's data catalog and stage graph, PanoramaStitching's [example], BlobBoards' [find_boards]).

### One word for a spline's free values: node; one key for how many: `node_count` (owner, 2026-11-15)

- `[vignetting] coefficients` → `[vignetting] node_count`, `[focus] nodes` → `[focus] node_count` (`VignettingOptions.node_count`,
  `FocusOptions.node_count`); the old keys are refused by name, `node_count` named. Every config of record migrated.
- Docstrings and comments: the vignetting's (and every spline's) "control points" are "nodes"; Core's identifiers are unchanged.
  A naming item (test/unit/quality_naming_tests.jl) keeps the old phrase and keys out.

### The chrome evaluation: the held-out headline and paired photograph bootstrap intervals (owner, 2026-11-15)

- `[evaluate] holdout = "on"` in the chrome configs of record (required): `motion_heldout`, the parity-interleaved hold-out, is the
  paper tables' headline (`variant = "motion_heldout"`), the refinement's in-sample score and the gap beside it (`in_sample_variant =
  "motion"`: per statistic start | refined | Δ % | 95 % CI | in-sample | gap | all).
- `[evaluate] bootstrap = {resamples, seed, alpha}` and `comparisons = [{model, reference}, …]` (required): the paired bootstrap over
  the PHOTOGRAPHS of every comparison's differences (median, mean, p95, RMS; all / centre / edge; both variants), on the common
  dots, no new fit — VisualGeometryToolkitEval's `bootstrap_ci`, each interval its own StableRNG; into eval_<label>.json (`paired`)
  and paired.tsv, and the tables' CI column (`confidence = "on"`), * where an interval excludes 0.

### The convergence experiment (experiments): the bias of Schöps's method per LM iteration

- `detection_bearings(lens, poses, features, fpx = 1)`: every view's detection bearing errors concatenated (radians, or px with `fpx`)
  — the refinement monitor's `bearing(detections, own poses)` and the run's final bearing error now call it (the same numbers).
  `state_bearings(geometry, state)`: those bearings [rad] of a start geometry's detections at a saved refinement state (a checkpoint
  `state_it<N>.json` or a start record's `state.json`), under the state's own camera, poses and surface — the monitor's quantity
  after the run, from the states the refinement saved.
- The per-iteration checkpoint state (`io.checkpoint_each`) records its iteration's LM cost and rel-RMS in its provenance, beside
  the iteration (the monitor's numbers).
- experiments: `scripts/convergence.jl` (`configs/experiments/convergence.toml`, stage `convergence` of `configs/reproduce.toml`): per
  case (`ag10fe_spline_warped`, `si30_spline_warped`, Schöps's start) the bearing divergence of his detections (deg, median and p95)
  and the Thorlabs reprojection error (px, the chrome scorer's held-out and in-sample head-to-head on the dots every iteration
  scores) at every LM iteration, one IO record and a dual-axis pgfplots figure; the chrome evaluate stage's scorer is one function
  (`score_chrome_models`) shared with it.


## 0.11.3 (unreleased)

A hotfix release (scope frozen to one fix).

### Bug fix: the exact start refused a coverage circle clipped by the sensor

- The 14 mm housing cases (f4_14mm_spline_warped and the other 14 mm housing vignetting and panorama cases) failed at the exact start
  with `RegionBeyondSupport` ("the start (exact resample): the region RegionIntersection(Rect, Ellipse) …"): the coverage assertion
  (`check_coverage`) tests `sensor ∩ image circle ⊆ valid_domain`, and Core 0.10.8's `RegionIntersection ⊆ Rect` holds only when one
  part fits alone — here the sensor reaches past the start's lattice in x and the housing's circle in y, while their intersection
  lies inside it. The assertion now tests the intersection's bounding box `Rect2(region)` against the valid-domain `Rect` (sound: the
  region lies in its box). A WORKAROUND of Core 0.10.8, which a Core 0.10.9 patch fixes; RC drops it once 0.10.9 is registered.
  No number moves for a case that started before.

## 0.11.2 (unreleased)

A patch release (scope frozen by the owner, 2026-11-15): the lattice padding fix, the sensor-model switch, the PanoramaStitching
compat raise, and the checkpoint crash fix (a coordinator ruling over the freeze: a crash).

### Bug fix: a checkpointed refinement crashed at its first LM iteration

- Every refinement with `io.checkpoint_each = true` (the paper's vignetting and detector-bias cases) failed at LM iteration 1 with
  `FieldError: NamedTuple has no field observation` in `save_state` since 0.11.1: the monitor's per-iteration `state_it<N>.json`
  omitted the run's observation model, which 0.11.1's `save_state` requires. It now carries `observation_model(s)`; a test runs a
  checkpointed refinement for 2 LM iterations and loads every checkpoint state back.

### Bug fix: the exact start's lattice padded to the grid route's patch multiple

- Every case with a Schöps start (`frontend.type = "schoeps"`) whose resampled lattice was not a multiple of 5 nodes per axis failed
  at its first LM step (`patch size must divide the grid (71 × 75)`, `build_problem_grid`): the exact resample's node counts are
  m·cells + 3, set by the data, while the grid route tiles with fixed 5 × 5 static patch blocks. `exact_start` now pads the lattice
  at its right and bottom edges to the next multiple (`pad_lattice`, Core's `extend_lattice`): the resample's controls and presence
  bit for bit, the camera unchanged on the region (backproject and project, bit for bit), the added nodes absent — never a
  parameter — except the ring a region whose edge is a knot touches (present, zero basis on the region). The start record's
  `resample` states `grid_size` (padded) and `padding` (`from`, `nodes`, the padded call's `folds`). **No number moves for a case
  already at multiples** (the golden siemens_coarse_full's 35 × 25 is untouched).
- `GRID_PATCH = (5, 5)`: the one patch constant, the default of `build_problem_grid`, `run_lm_grid!` and
  `select_grid_smoothness_weight`, and the multiple the start pads to.
- A parametric start whose `grid.size` is not a multiple of the patch is refused at the start (`start_grid`), by name, not deep in
  the LM.

### The sensor model's switch: `[camera] sensor_model` (owner, 2026-11-15)

- **New key** `camera.sensor_model`, parsed once into a type (`sensor_model_mode`: `SensorModelAuto`, `SensorModelOn`,
  `SensorModelOff`; every decision a dispatch): `"auto"` (the default) uses a sensor record present and runs without one (one info
  line, the records state "no sensor model"); `"on"` requires one (refused without); `"off"` does not use a record present, with
  ONE loud warning banner per run (`warn_sensor_model_off`: the record's location and sha256, that estimation and prediction may
  then disagree, the TOML line). A record is present when `camera.sensor_characterisation` names one on disk, or the camera the run
  continues (a start or resumed state) or predicts (`photo_set(camera, files; sensor_model)`, `prediction_observation`) states one.
- **Breaking:** `camera.sensor_characterisation = "none"` is retired, refused naming `sensor_model = "off"`; a run that names no
  record leaves the key empty (a RAW run then reads the file's black under "auto", formerly refused). A named path that does not
  exist is refused (a typo never turns the sensor model off). Every other mismatch stays refused: another record's sha256, another
  body or ISO, the other photograph kind; decoded images under a record or under "on" are refused.
- Records: the start record and the run's parameters state `sensor_model` (`mode`, `used`, the record `present`); an "off" run with
  a record present is a run that did not use it.
- experiments: chrome detection's `[detect] sensor_characterisation` for decoded (`luma`) photographs is empty, not `"none"` (the
  photographs' kind decides); `case_characterisation` reads an empty key as none.

### experiments

- PanoramaStitching compat raised to 0.8.3 (registered in General): its `[decode]`/`[registration] sensor_characterisation`, which
  0.11.1's panorama chain requires.

## 0.11.1 (unreleased)

### Sensor-model consistency, enforced by types and records (owner, 2026-11-15: "if a sensor was estimated with a model, then it must be used to predict the model")

- **New: the observation model of a calibration** (`src/camera/observation_model.jl`): `ObservationModel(sensor, kind, channel)` —
  `SensorRecord(sha256)` (the sensor characterisation the RAW photographs were linearised under, by its file's sha256) or
  `NoSensorRecord()`; `Photosites()` (RAW, the green photosites on the raw grid) or `Decoded(srgb)`; the channel. A sensor record
  of decoded images, or RAW photosites of another channel than G, is refused at construction. `CalibratedCamera(model, observation,
  settings)` is the COMPLETE calibration — the `RadiometricCameraModel` (the kernel type, unchanged), its observation model and its
  render settings; `CalibrationEstimate.camera` is one, `calibrated_camera(state)` makes one of a `CalibrationState` or a
  `load_state` record. The state's observation model is `observation_model(s)` (its new fields `sensor`, `photo_kind`, with
  `channel`).
- **New: THE entry point of photographs** — `photo_set(camera_or_observation_model, files; characterisation, lens)` and
  `observe(set, k, mask)`, by dispatch on the observation model's types (never a flag): a camera calibrated under a sensor record
  reads RAW photographs only, under exactly that record (`characterisation` is a location; another file is refused naming both
  sha256s), each frame's model Core's resolver by its EXIF, a frame the record does not describe (`UnmatchedISO`, `OtherBody`)
  refused by name; a camera calibrated without one refuses a characterisation; a camera of decoded images refuses RAW photographs.
  The calibration's start reads its photographs through it (`observation_model(files, camera.sensor_characterisation; linearize,
  channel)`), and so do the paper's consumers: chrome detection, the panorama comparison, the diffuser flat field and texture recovery.
- **Breaking: retired** `photo_observations`, `photo_sensor_models` and `sensor_characterisation` — refused by name with the entry
  point (no alias, no second decode path). `sensor_statement(spec)` states a characterisation (its sha256).
- **Breaking: records.** `rc/radiometric_camera/v3.0 → v3.1` (a `CalibratedCamera`: the camera body, `"observation_model"` and
  `"settings"`; `save_radiometric_camera(path, ::CalibratedCamera)`, `load_radiometric_camera` returns one) and
  `rc/calibration_state/v3.0 → v3.1` (`"observation_model"`; `save_state`'s NamedTuple states `observation`, `load_state` returns
  it). A v3.0 camera or state states no observation model: **refused, never migrated** (owner: "refuse old keys; we must get up to
  date"), naming the refinement step that regenerates it. The RC ≤ 0.8 states' version step (rc/calibration_state v1, v2 → v3.0,
  ruling A') is retired with it: those states are refused as every RC ≤ 0.8 record is. So is the in-focus-blur reader of records
  written before 2026-11-14 (a v3.1 record never holds one).
- **A state continues or predicts only photographs read as it was estimated** (`load_state_into!`: resume, a start record,
  `io.render_state`): another observation model is refused naming both.
- experiments: texture recovery holds the calibration's complete camera and reads the sheet's photographs through the entry point under
  its record (`case_characterisation`, the case's characterisation input, sha256 checked); chrome's `raw_photographs` resolves its
  frames' models through `photo_set`; panorama_stitching's new `flat` stage reads the diffuser's RAW photograph under the learned
  camera's own record (its green photosites, the record's dark subtracted, its defects masked; the median per 1° equisolid ring),
  replacing the convert_raw PNG and the off-repo flat_equisolid.py (LibRaw's black and demosaic, no defects), and the rooftop frames
  are refused unless PanoramaStitching's example states it decoded them under the same record (`require_frames_decoded_as`;
  PanoramaStitching ≤ 0.8.2 states none: its sensor-model branch adds `[decode] sensor_characterisation`); vignetting_figures reads
  the diffuser the same way. The image circle and the front ends' decode (`decode_raw`) are geometric and stay on the file's black,
  their exemption stated. A lint keeps one RAW path: no decode primitive outside the allow-list.

## 0.11.0 (unreleased)

### The exact start (Core 0.10.8, Estimation 0.11.6; owner, 2026-11-15) — a Schöps lattice extended and subdivided, never refitted

- **Breaking:** a SplineLens start camera (a Schöps run's central-generic model, a session's spline camera) is resampled EXACTLY by
  Estimation's `fit_camera(SplineLens, source, region; lattice_spacing, forward = false)` over `region` = the sensor ∩ the run's lens
  coverage (`coverage_region`; the coverage `run_lens_coverage`'s: the sensor, or the set's image-region record): his lattice extended on his own spacing until every node whose
  support meets the region exists (his present nodes bit for bit, the new ones by Core's bending-energy continuation, marked
  `extrapolated`), subdivided by the integers whose cell is nearest the target cell (Core's exact knot insertion), turned onto his
  optical axis (Core's exact `axis_gauged`). New `exact_start(source, lattice_spacing; coverage, device)` and
  `start_grid(source, grid; coverage, device)` — the start's lattice by the source's TYPE: a SplineLens source resampled at
  `grid.lattice_spacing`, any other fitted at `grid.size` (`grid_model`, unchanged).
- **New option `grid.lattice_spacing`** (px, default 40.0: the cell of the 105 × 75 lattice of record on the FX3's 4256 × 2848 frame);
  `grid.size` is now the fitted lattice of a parametric source only. `siemens_coarse_full` states 160 px (its 35 × 25 cell of
  record; Schöps's own ≈ 166 px, so his lattice is kept, 1 × 1, extended over the sensor).
- **Breaking:** `schoeps_gridlens(yaml, lattice_spacing, W, H; coverage, device)` and `schoeps_start(…)` (was `GX, GY`);
  `schoeps_start` also returns the resample's record.
- **The coverage assertion** (`check_coverage`, set-level): `region ⊆ valid_domain(model(camera))` (Core's `check_region_support`) and
  every pixel of the region imaged (a finite ray from Core's batched `backproject`). The start and the calibrated camera are held to it.
- **The start record** (`start.json`) holds the resample (`resample`): the lattice, the subdivision, the resulting cell, the extension
  offset, the extrapolated nodes' marks and the share of the region the source images no ray in, Core's fold report of the extension
  (`folded`, `samples`, `unimaged`), the fidelity to the source (`max_px`, `max_px_edge`, `rms`) and the axis turn. `parameters.json`'s
  `grid_size` is the start's actual lattice.
- **The fold policy** (owner, 2026-11-15): the start keeps the full region — its folds are reported, nothing trimmed or masked. After
  the refinement the CALIBRATED camera is checked (`calibrated_folds`: Core's `folded_pixels` over its coverage); a remaining fold makes
  the termination record `(Failure, ShapeGate)` (`fold_termination`): the case is non-converged in `parameters.json` and in the paper's
  tables (†) and is not accepted as a calibration. `CalibrationMetadata.folds` and `parameters.json` `calibrated_folds` record it.
- **Numbers:** the reference (`siemens_coarse_full`) is re-recorded ONCE for this switch together with the `π⁻¹`/`project_pixel`
  retirement below (owner-approved; 3.6e-4 relative from Core's normalized backproject in the start pose fits, amplified by the
  unconverged LM): see the re-record's commit for the measured comparison.
### The lens coverage of a capture set is ONE keyed step, read by every consumer (owner ruling, 2026-11-15)

- **Breaking: `camera.image_circle_frames` and `camera.image_circle_outlier_halfwidth_px` are retired**, refused by name: a run no
  longer measures its image circle. `camera.lens_coverage` states the coverage — `"sensor"` (configs/default.toml: the lens lights
  the whole sensor; nothing measured, the detector never runs) or the path of the capture set's measured image region, an
  `rc/image_region/v1.0` record (`save_image_region` / `load_image_region`, through IO: the ellipse, the photographs' size, the frames
  and the fit's diagnostics). `run_lens_coverage` is the run's one reader (a missing record, or one of other photographs, refused; a
  grid start's stored coverage is no longer a fallback). `measure_image_region` (the former `run_image_circle`'s measurement:
  Features' `detect_image_circle` on the device, RAW green cells or decoded renders) is the step's one implementation.
  Migration: delete the two keys; state `camera.lens_coverage`.
- experiments: every config of record states each set's coverage (`lens_coverage = "sensor"` or `"image_region:<name>"` with an
  `[image_regions.<name>]` table: the set's own frames pinned by sha256, the detector's settings); `scripts/image_regions.jl` measures
  them into keyed products (`image_region/detect/<key>`, keyed by the frames, the settings and the code alone), and the refinements'
  starts, the front ends' bounded cameras, the chrome evaluation's radial frame (its typed-in `centre`/`rmax` retired) and the panorama
  cameras read the records. Retired: the shared `vignetting/image_circle/*` inputs (refused by name), `scripts/make_panorama_cameras.jl`,
  `src/panorama_cameras.jl`, `configs/inputs/panorama_cameras.toml` and the `*_kb_fit_lens_coverage.json` camera records.
- Numbers: the measured sets move (the 14 mm f/4 sets: the RAW diffuser's region in place of the PNG's and the panorama's stale
  1680 px circle; the fisheye sets: their RAW frames' regions; the fisheye 10 cm chrome set's radial frame); the sensor sets do not
  (the chrome radial frame's rmax is now the exact half-diagonal, 2560.49995 px for 2560.5).

### Retired: `π⁻¹` and `project_pixel` (owner, 2026-11-15) — the back-projections and projections are Core's

- **Breaking:** `π⁻¹` (every method, the SplineLens one included) and `project_pixel` are refused by name with their replacement:
  VisualGeometryToolkitCore's `backproject(camera, 𝐩)` (a unit ray, NaN where the lens does not image 𝐩) and `project(camera, X)` —
  a SplineLens camera's batched `project(model(camera), rays, seeds)`, warm-started at the detections (`_project_from`, chosen by
  the camera's type: no `is_grid_lens` branch). Every caller outside the image formation moved: the self-check (`fpx_of`,
  `bearing_px`, `_imaged`, `residual_vectors`, `board_hits`), the focus start and its support, the front ends' pose fits, the drift
  monitor, `grid_angular_error`, the record readers, focus-ab; the experiments' evaluation, ground truth, texture recovery and
  focus-surface script.
- The image formation keeps the backward spline's VALUE (`_chief_ray`, internal: Core's lattice evaluation at K⁻¹𝐩, not
  normalized) — linear in the lattice's controls, which the per-tile Jacobian's fan-out onto them relies on; the parametric
  cameras' is Core's `backproject`. A Core follow-up names the raw backward value; then `_chief_ray` goes.
- Numbers: a measured comparison against the reference (the SplineLens back-projection now normalized and NaN off its present
  windows; the self-check's projections warm-started at the detections, so rim points where the cold start failed become defined).

### New: `masks.lens_coverage_scale` (default 1.0: no number changes; owner, 2026-11-15)

- The photometric pixels are sampled inside the lens coverage scaled about its centre by this factor (an image circle's semi-axes;
  the sensor's rectangle is not scaled): sampling only — the camera's coverage, its lattice's present nodes and the vignetting chart
  stay the measured ones (`_sampling_region`, start/masks.jl). The default 1.0 is the imaged region itself, bit for bit; 0.9 is
  pending the owner (it cut the rim's bias on the fisheye 10 cm but left ρ 0.9–1.0 of the fisheye 30 cm without photometric data,
  its nodes there drifting; configs/default.toml states both).
- experiments: the ablation `configs/experiments/radiometric_refinement_ablation_lens_coverage_scale.toml` — the config of record plus
  the override `masks.lens_coverage_scale = [1.0, 0.9]` (`expand_overrides`, DrWatson's `dict_list`): each case at 1.0 is the case
  itself (its run of record reused), at 0.9 `<case>__masks.lens_coverage_scale=0.9`. Not a stage of the regeneration (REPRODUCE.md).

### The reorganization (owner rulings D1–D16, 2026-11-14/15; no number changes: the golden and the regression goldens bit-identical)

- **The source tree by component** (T1, pure moves): `calibration_pipeline/`, `pipeline/` and `board/` are gone — `target/` (D12),
  `image_formation/` (+ the flux), `calibration/` (state, closed form, `problem_spec.jl`, `lm/per_view.jl`, `lm/per_tile.jl`,
  `device/{workers,flux_kernels,jacobian_kernels}.jl`, `calibrate.jl`), `start/` (front ends, poses, masks, the radiometric start, the
  start record), `diagnostics/`, `options/{device,tables,toml,retired}.jl`, `io/` (photographs, run outputs, records), `experiments/`
  (focus-ab, until it leaves). `src/api.jl` states the surface.
- **Breaking: the verbs** (D1, D3 as revised): `start_calibration(geometry::StartGeometry, photographs; options, device)` (or
  `start_calibration(camera, poses, observations, photographs; target, options)`) lifts the geometric calibration into a
  `CalibrationState`; `refine_calibration!(state; options)` returns Core's `FitResult{CalibrationEstimate, CalibrationMetadata}`
  (estimate: camera, poses, target, light; the LM's ReturnCode / StopRule; metadata: the trace, costs, seconds, the solver's record,
  the scrubbed counts); `calibrate(…; options)` = both; `start_geometry(options)` (the front end's typed product, `StartGeometry`),
  `save_calibration_start` / `load_calibration_start` (the start record), `save_radiometric_calibration(state, result; options)`
  (the run's records), `selfcheck(geometry; options)` (the diagnostic, out of the start). **Retired, gone (no alias)**:
  `prepare_start`, `prepare_start_refinement`, `prepare_refinement`, `write_refinement`, `calibrate!`, RC's method of Core's `refine!`.
- **Breaking: the API** (D13, D14): 16 names exported (the verbs above, their inputs and results, `CalibrationState`,
  `RadiometricCameraModel`, `RadiometricTarget`, `PipelineOptions`, `pipeline_options`); every other user-facing name is `public`
  (the paper's symbols among them: demoted, names kept); Core's `lens`, `chart`, `cost`, `residuals` are no longer re-exported.
- **Breaking: the options** (D8, D10, D11): `configs/default.toml` is THE statement of every default (the structs' keyword defaults read
  it, baked at precompile; docs/pipeline_config.toml is gone); the held blocks are one top-level `fixed = [...]` (poses, camera, focus,
  aperture, material, vignetting, lighting, inplane) — `"grid"` renamed `"camera"`, `"edge"` refused; a held aperture is
  `fixed = ["aperture"]`. **Retired keys, refused by name with their migration**: `lm.fixed`, `camera.r_min`, `camera.r_max` (inert:
  `ApertureVar`'s bounds were never read; gone with the builders' `r_bounds` keyword), `rendering.psf_sigma`, `edge.camera_sigma_px`,
  `warping.points`; `warping.heights_mm` takes the nodal heights only (the legacy 3 / 7 coefficient form is gone). Every config of
  record migrated in the same commit; run keys recompute (D11: no key migration).
- **Every input through IO** (D4, D6): the blob front end reads its detections only as the session's `observations.json`
  (vgio/observations; the sightings.json hand parse is deleted, a session without the record refused by name); the printed edge's
  distance map is a vgio/raster record (`load_distance_map`, public), converted once from the generator's sdf.f32 + sdf.json by
  experiments/scripts/make_distance_maps.jl into new input entries (siemens_1m/distance_map/, aprilgrid/distance_map_8ppm/).
- **The route by the camera's type** (T3): `lm_route(camera)` (per tile for a SplineLens camera, per view otherwise) replaces every
  `parametric::Bool`; **no module-level run state** (T2): the LM's final-iteration mark, the scrubbed counts and the device calls live
  on `CalibrationState.run`.
- **Photograph listings** skip macOS `._*` / `.DS_Store` (`is_raw_file`, `is_png_file`, `photo_files`).
- **Moved to experiments** (D6, D15): the pipeline golden and its tools (experiments/test/pipeline/), bin/material_grid.jl
  (experiments/scripts/), bin/loss_curve.jl (experiments/scripts/plots_loss_curve.jl; bin/refine.jl draws no figure). The package's tests
  read no paper config.
- **Gates** (T13): explicit imports only (ExplicitImports), the stdlibs' [compat], the argorder check (`draws(sampler, [T,] key, i)`:
  the type after the sampler), Aqua (one check at a time), the JET ratchet (34) — test/unit/quality_gates_tests.jl.
- **experiments**: an experiment is the config of record + `[overrides]` (DrWatson's `dict_list`, `expand_overrides`); the fisheye
  Siemens 27.10 set is on the exclusion list (configs/excluded.toml), kept as data, run by no driver.

- **Breaking: the refinement's start is a step of its own** (the owner's rule of 2026-11-14: every step of a pipeline caches its
  intermediate, keyed by only what it reads). `prepare_start(opts, dir)` computes the start — the front end (the start lens with the
  run's image circle, its initialization region, the PnP / page poses), the masks, the vignetting centre, the focus start, the
  closed-form radiometric start and the GCV weights — from `start_options(opts)` (the refinement-only options,
  `REFINE_ONLY_KEYS`, at their defaults: a start cannot read one) and writes a START RECORD (`save_start`: start.json
  `rc/refinement_start/v1.0`, state.json, grid.json | lens.json and frontend/, masks/, pipeline_config.toml);
  `prepare_refinement(opts, load_refinement_start(dir))` rebuilds the state from it, nothing of the start computed again, and refuses
  a start of other start options (`check_start_options`). Every refinement starts from a start record: `bin/refine.jl --stage=start`
  writes one into `io.out`, `--start=<dir>` refines from one, and without either the start is written into `io.out/start/` and the
  refinement continues from it (within 0.11.0 these entry points became the reorganization's verbs above: `start_calibration`, `save_calibration_start` / `load_calibration_start`, `refine_calibration!`). **Retired, refused by name**: `io.save_init` (the start record replaces it).
  the experiments' drivers run every stage as a keyed product (VisualGeometryToolkitEval 0.9's `Product` / `produce`, DrWatson's
  `produce_or_load`, `datadir("exp_pro", <experiment>, <stage>, <key>)`): a case's start (stage start) keyed without the
  refinement-only options, its refinement by those options and the start's key — a refine-only change (`lm.max_iter`) loads the start.
- **experiments: every FX3 chrome set on the FX3's characterisation of record.** Five of the seven chrome configs
  (14mm_central30cm, fisheye_10cm_ag, fisheye_2710_si, tamron_17cm_si, tamron_20261108_ag) read the superseded single-Dark record
  `fx3.sensor_characterisation.gain-exifband.json` (vgio v1.0); all seven, like the refinements, now read
  `fx3.sensor_characterisation.gain-exifband-alldarks.json` (v2.0, a Dark per measured exposure; sha256 add8fc12…5737) — path,
  sha256 and provenance. Their detect products are re-keyed (the input's pin keys them); a test holds every set to the record.
- **experiments: a raw_green chrome set is prepared as its photographs, not decoded; its inert `noise_sigma` retired.** `[detect]
  image = "raw_green"` now reads a `[prepare.<set>]` of mode `"photosites"` (refused on a decoding, by name): the prepare product
  is the photograph list and each photograph's EXIF (Features' `load_raw_metadata`, the headers alone; photosites.json, every
  photosites set's), and detect reads each mosaic (`load_raw_mosaic`) — the full green-cells decode of every set is gone. The seven
  FX3 chrome configs: `mode = "photosites"` (their `gain` / `srgb` dropped), their photographs named by their RAW files (.ARW).
  `[detect] noise_sigma` is a luma set's key alone (`DETECT_LUMA_KEYS`, required there): under the FX3 characterisation SFOP's λ₂
  test reads the sensor model's noise law, so a raw_green set refuses the key by name instead of keying its detect product on it.
- **experiments: the LM cap of every refinement case, `lm.max_iter`** (owner 2026-11-14; radiometric_refinement.toml's parametric
  cases were at 48 since 2026-11-05, the spline ones and radiometric_refinement_vignetting.toml at 50; the pixels as of record):
  **25** — the 14 mm si30_spline_warped run (Schöps start, photosites + FX3 all-darks) flattened by ~20 iterations (rel-RMS 0.2901
  at 10, 0.2876 at 21, 0.2872 at 47), and doubling the pixels left the trajectory unchanged to 3–4 digits — and **50 for every
  fisheye case** (ag10fe_*, si27fe_*, si_fisheye_spline_20261027_warped, the vignetting config's f4_fisheye_spline_warped): the
  fisheye 10 cm spline run was still improving at 25 (rel-RMS 0.3915 at 20, 0.3890 at 25) while its chrome median improved 27 %.
  The derivation on each line; it re-keys those refinements (a refine-only key: their starts are reused). A test holds every case.
- **/simplify of RC 0.11.0** (one change of results: the focus-surface figure's bug fix).
  - **Fix: the experiments' focus-surface figure** draws RadiometricCamera's focus point `𝔣(View(pose, camera), 𝐩)` and `t̂(view, 𝐩)`: the
    hand copy read the sensor-TPS focus kind alone, so a direction-chart focus would have drawn another surface.
  - **bin/refine.jl's default run reads its photographs once**: `prepare_start_refinement(opts, dir)` writes the start record and
    builds the refinement's state from it as the record path does (its state, poses and lenses as the record holds them), the
    start's streamed observations and masks handed over; `--repeat` reads the start record once, outside its repeats.
    `--start-out=<dir>` names where that run writes its start (a driver's start product).
  - **One Julia child per case when its start and refinement are both to be produced** (the experiments' `run_refine`, scripts/calibrate.jl):
    the start written into its own product (the same key) by the refinement's child; two children only when the start is loaded.
  - **The refinement-start stage is named `start`** (was `init`) in radiometric_refinement, radiometric_refinement_vignetting,
    detector_bias and calibrate — `--stage=start`, the products' `start/<key>/` directories, the refinements' `upstream.start`, the
    Slurm `STAGES` — re-keying them; the front ends' and texture_recovery's own `init` stages are other steps and keep their name.
  - **scripts/chrome_evaluation_paper.jl**: the chrome paper stages (sets, figures; `figures_product()` for scripts/numbers.jl) are a
    driver of their own, chosen by the script, not by the config's shape; the paper config's `[experiment] name` is
    `chrome_evaluation_paper` (scripts/chrome_evaluation.jl refuses it by that name). Tables' generator lines name it.
  - Smaller: the blob front end computes no bearing errors for a stored start's y sign; `start_is_grid` reads the options alone;
    `_stored_poses` one check (the record's views are the run's); `case_init` folded into `_start_init`; `_scoring_at` dropped;
    `driver_product(...; threads)` one knob (a thread count, or `nothing` for a thread-invariant product; `at_threads` gone); one
    m → mm factor in the chrome evaluation (`_MM_PER_M`); the chrome tables share one parsed summary per set.

- **Breaking: one smoothness vocabulary for every spline component** (the owner, 2026-11-14), in the options and in the code. A
  component's smoothness is a table dispatched by its `type` (the stack's `{type, …}` convention): `[grid.smoothness] type =
  "pspline", order = 3, weight = 0.1`, `[vignetting.smoothness] type = "pspline", order = 3, weight = 0.0`, `[focus.smoothness] type =
  "bending", weight = 0.0`; `type` one of `"pspline"` (its `order`-th differences), `"bending"`, `"none"`, `order` a pspline's only,
  `weight` a number β ≥ 0 or `"gcv"` (the grid and the vignetting). The orders are the code's of record — the grid's P-spline was
  and is its Δ³ (order 3, not 2), the vignetting's Δ³ — so every penalty is the same and no number moves; the run keys re-key. The
  vignetting takes a pspline of order 3 or none, the focus a bending (of a number) or none. **Retired, refused by name with the
  replacement:** `grid.prior`, `grid.prior_weight`, `vignetting.smoothness_weight`, `focus.prior`. The code: `PSplineSmoothness`,
  `BendingSmoothness`, `NoSmoothness` (src/pipeline/smoothness.jl, `smoothness_line`, `smoothness_weight`), `grid_smoothness` /
  `GridSmoothness` / `grid_smoothness_cost` / `select_grid_smoothness_weight` (grid_prior.jl → grid_smoothness.jl;
  `GRID_PRIORS` gone), `focus_smoothness!` and the state's `focus_smoothness`, `build_problem_grid(...; smoothness)`; the log lines
  name the smoothness (`smoothness pspline order 3, weight 0.1`). Every config of record migrated (43 grid tables).
  **The records follow** (the owner, 2026-11-14): `rc/run_parameters/v3.0` — the grid's `smoothness` {type, order, weight, cost,
  beta, selection} (was `prior`, `prior_weight`, `prior_cost`, `prior_beta`, `prior_gcv`) and `vignetting_options.smoothness` {type,
  order, weight, selection} (was `smoothness_weight`, `smoothness_gcv`) — and `rc/focus_ab/v2.0`'s `smoothness` {type, order,
  weight, row_weight} (was `prior_beta`, `prior_weight`); a v2.0 / v1.0 record reads through its version step (VisualGeometryToolkitIO's
  `migrate_schema`), its values as stored. The code's last `prior` names of the concept renamed (`FocusSmoothnessResidual`).

- **/code-review of RC 0.11.0** (eight findings, each with its test):
  - `--stage=start,refine --force` produces the start again with the refinement (`start_reused`: a produced start is read only when
    this run does not force the start stage); `--stage=refine --force` refines from the produced start.
  - The front ends' products (decode, Schöps run, fit, comparison) are keyed by the code their stage runs (`FRONTEND_CODE`: Core,
    IO, Features / Estimation; a Schöps run's container by its image's sha256 among its inputs), not every tree: a commit of the
    drivers or of RadiometricCamera re-keys none, so a `[runs.<id>]` reference stays produced. (Eval's `Product` states the rule.)
  - The chrome index's scorer: `refine!` of a pose the scorer did not score assigns it on demand (no error), and one photograph's
    failed registration refuses that photograph alone (recorded, warned), never the index product.
  - The start record's GCV selections round-trip bit for bit (their field order, Bool flags, Int counts and the smoothness's
    `kind` a Symbol): a refinement from the record writes the computed start's parameters.json byte for byte.
  - A driver's start is keyed by what RadiometricCamera's start reads (`start_pipeline_table`, `start_options`' one definition: no
    refinement-only key and no output — `io.out`, `lm.log`), in the experiments' drivers and `calibration_products` alike.
  - The start's streamed observations are kept only by a start computed now (`meta.observations` is `nothing` on a record's).
  - The one-child path checks the case's inputs and resolves its table once; `prepare_start` and `prepare_start_refinement`
    share one helper (`_compute_start`); a start record's `fpx`, `f_m`, `d_c` and bearing errors are checked against its stored
    lens and poses (`_recorded`: to the record's rounding, else refused by name).

- **Folded in: Ties Harder's PRs #10 and #11** (owner decision 2026-11-14), one breaking release with the
  above — the non-central spline model, the camera's in-focus blur and the printed edge's optimisation removed, the pipeline
  radial-vignetting only, the spline model named apart from Schöps's pipeline:
- **The spline camera model is no longer called Schöps's** (2026-11-14; the owner). Schöps et al. contributed the calibration
  pipeline; the model it calibrates is the central generic spline camera model. Where a comment, docstring, message or page named
  the model after them ("Schöps's model", "Schöps's central spline", "Schöps' generic models"), it now names the model and the
  pipeline apart ("a Schöps run's spline model", "the central spline of Schöps's pipeline"). The names of the pipeline, its runs and
  its files stay (`frontend.type = "schoeps"`, `frontend.schoeps`, the `schoeps:` spec, `schoeps_gridlens`, `schoeps_run_file`, the
  model kind `"schoeps"`): no identifier, config key or config value changes. Three comment lines of two configs of record are
  reworded; their parsed content is unchanged, so no run key moves.
- **Breaking: a refinement fits the radial vignetting profile alone** (2026-11-14; the owner: no case of record uses the
  tensor-product surface). The pipeline's option `vignetting.model = "tensor_product"` and its key `vignetting.size` are gone: the tag
  is refused by name ("a refinement fits \"radial\" alone …") and `vignetting.size` is a retired key. `vignetting.model = "radial"`
  stays accepted: every config of record states it, so no run key moves. `_vignetting_spec` builds the radial profile about the
  principal point, and a resumed state's tensor surface is never taken over (`_same_field`). **What stays:** the renderer, the
  closed-form solves and the LM take any vignetting field of Core's (`AbstractVignetting`), the tensor-product surface included —
  the unit tests and the unit goldens' scenes are built on it and are untouched — and the experiments' panorama comparison and ground
  truth use Core's type themselves. No number moves. The options' header comment named the radial profile "about the sensor
  centre"; it is about the camera's principal point.
- **Breaking: the printed edge is never a parameter** (2026-11-14; the owner: every case of record holds it, the ink spread at 0).
  The printed-edge model stays — the ink spread δ and the print's blur σ_print of `BoardSurface.edge`, constants from
  `edge.ink_spread_mm` and `edge.print_sigma_mm` — and what could learn δ goes: the material block's printed-edge column
  (`MaterialVar`'s `edge` keyword and `NE` parameter, `EDGE_NFREE`, `n_fd`, `n_edge`), its forward differences of the flux
  (`_fd_perturbed`, `EDGE_FD_H_MM`, the batch's extra flux launch, the tile layout's `NP` argument and `fd_cols`, the row kernel's
  `fd` argument), `build_problem_grid` / `run_lm_grid!`'s `edge_block`, `:edge` in the builders' `fixed` (`SURFACE_HOLDS` is
  `:height`, `:inplane`) and the refusal of a free edge on a parametric start. **`"edge"` in `lm.fixed` is no longer read** and stays
  accepted: every config of record states it, so no run key moves. A config that leaves it out used to learn δ on a board with a
  distance map; it now holds δ like every other. The retired key `edge.free` is refused with that text. **Record:**
  `parameters.json`'s `printed_edge` no longer holds `free`. **Numbers:** none moves — the golden scenes are the previous commit's bit
  for bit, the printed-edge tile's Jacobian without its δ column (`tile_edge_jacobian`: 267 columns, spliced in as text;
  test/regression/README.md). docs/pipeline_config.toml's `lm.fixed` line no longer lists `origins` (gone with the non-central model).
- **Breaking: the camera's in-focus blur is removed** (2026-11-14; the owner: every paper case holds it at zero). The sensor rule
  is the pixel box and a printed edge's width is the print's blur σ_print alone. Gone: `InFocusBlur`, `NO_BLUR`, `in_focus_blur`,
  `blur_sigmas`, `with_in_focus_blur`, `with_blur`, the fifth field and argument of `RadiometricCameraModel` and of `View`,
  `RenderSettings`'s `in_focus_blur` keyword, `SensorQuadrature`'s `psf_σ` (the Gauss–Hermite rule) and its `σ`, the printed
  edge's camera floor (`_edge_footprint`, the pixel's footprint on the board, and the distance map's orientation tensor:
  `PrintedTarget.sdf` is the `H × W` distance map itself), the material block's blur and its three σ_cam columns (`EDGE_NFREE` is 1:
  the ink spread δ), and `set_camera!` with `with_sensor_rule` and `_check_sensor_rule` (a state's camera is assigned like its
  other fields). **Config keys:** `rendering.psf_sigma` and `edge.camera_sigma_px` are no longer read; they are accepted at `0` and
  `[0, 0, 0]` (every config of record states them so: no run key moves) and refused by name otherwise. The default of
  `edge.camera_sigma_px` was `[1, 1, 1]`, learned: a config that does not state it now runs without a blur. **Records:** the
  camera body no longer holds `in_focus_blur`, the state's settings no `sensor.in_focus_blur` flag and `parameters.json` no
  `printed_edge.start.camera_blur`; the schemas' names are as they were. A record written before reads when its camera's blur is
  zero in every channel (the stored paper runs) and is refused by name otherwise ("the record holds a camera with an in-focus
  blur …"). **Numbers:** a run that held the blur at zero renders as before — a texture board bit for bit (the golden scenes
  without a distance map: 9000 numbers, CPU), a distance-map board to rounding: the compiled flux changed with the footprint gone,
  so on the CPU the flux moves in its last one or two ulps for under 1 % of the values (the tile Jacobian bit for bit), and on a
  device (Float32) the flux and the Jacobian in their last digit for 3–4 % of the values, the cost by 3e-9 relative (a two-view
  test scene, the edge held).
  **Goldens:** the unit goldens' printed-edge scene (σ_cam 0.9 px, three σ_cam columns) now has σ_print 3 mm and no
  blur — `tile_edge_residual` and `tile_edge_jacobian` are re-recorded and spliced in as text, every other entry untouched
  (test/regression/README.md). **The pipeline golden `siemens_coarse_full` must be re-recorded:** its config loses
  `edge.camera_sigma_px = [1, 1, 1]`, so its run renders a sharper edge; `test/pipeline/golden/siemens_coarse_full/` is the recording
  with the blur until then. experiments: the ground truth's `psf_px` and the three σ_cam entries of a set's `edge` are accepted at
  zero and refused otherwise.
- **Breaking: the non-central spline model is removed** (2026-11-14; the owner: the paper no longer uses it, and it was the suite's
  longest part). Every camera is central. Gone: `ApertureSpline`, `NoncentralLens`, `noncentral_gridlens`, `ApertureSplineVar`,
  `r_of`, the `aperture` field and keyword of `RadiometricCameraModel` and `View`, the non-central per-tile blocks
  (`src/calibration/per_tile_noncentral.jl`: the patches with origins, the r(𝐩) spline's columns, `chief_origin_node`),
  `build_problem_grid`'s `origin_gauge` and its `:origins` hold, `load_camera` / `load_model`'s `noncentral:` spec, and the
  two-argument `start_is_grid(init, noncentral)` (now `start_is_grid(init)`). **Refused by name:** the config key
  `frontend.noncentral` ("the non-central spline model is removed …; delete the key"), and a camera record whose aperture holds a
  `radius_grid` (a non-central camera's record: "regenerate the record from a central start"); `"origins"` is no `lm.fixed` block.
  The flux divides its aperture integral by πr² for every lens (the per-position average of a non-central lens is gone), bit for
  bit the central arithmetic. The golden case `fisheye_nc_selfcheck` (its config and `test/pipeline/golden/fisheye_nc_selfcheck/`) and
  the unit items `noncentral_tests.jl` and `noncentral_truth_tests.jl` are deleted; the unit goldens lose their three non-central
  entries (cut from `golden_migration_snd_try.json` as text, nothing re-recorded). The paper project: no `noncentral` model kind in
  the chrome evaluation (`CHROME_MODEL_KINDS`), the pipeline checks and the golden stage run `siemens_coarse_full` alone.
  **The hooks the model left behind are removed too:** `π_origin` (it was always `nothing`; no longer exported), the origin and
  r(𝐩) seed arguments of `Φ_pixel`, of the printed edge's footprint and of the device kernels (`_JacAt`'s `obase` / `rbase`), the
  tile layout's origin and radius-spline slots (`_tile_layout`'s `NO` and `NR` arguments, `off_o`, `r_cols`, `held_origin_node`),
  `AbstractApertureVar` (the aperture block is `ApertureVar`, documented now: the docs gate's backlog is empty) and the identity
  `_central` — `load_grid` and `load_model`'s `schoeps:` refuse a non-central record by name ("holds a non-central model; this
  package's cameras are central"). No number changes: the removed seeds were zero and the removed slots empty. **A run's record
  changes in one field:** `parameters.json` no longer writes `noncentral` (null in every run; `rc/run_parameters/v2.0` otherwise
  as it was) — the golden `siemens_coarse_full/parameters.json` loses that one line, cut as text, nothing re-recorded.
- **The golden calibration case `blob_kb` is deleted** (2026-11-14; the owner: no blob-board calibration is left in the paper):
  `experiments/configs/calibration/blob_kb.toml` and `test/pipeline/golden/blob_kb/`. The pipeline checks
  (`test/pipeline/run_pipeline_checks.sh`) run `siemens_coarse_full` and `fisheye_nc_selfcheck`; the reproduction's golden stage and
  `GOLDEN_CASES` default to `siemens_coarse_full` (`configs/reproduce.toml`, `scripts/slurm/common.sh`, `golden.sbatch --array=0-0`).
  The tests that took `blob_kb` as their example case (`experiments/test/calibration_tests.jl`, the options and comparator items) take
  `siemens_coarse_full`; experiments/README.md's first commands are a paper driver's (`scripts/radiometric_refinement.jl`). **The
  package's pipeline checks no longer run a `frontend.type = "blob"` case end to end** (the BlobBoards session's consumers stay
  covered by the unit items and `test/integration_blobboards`). The case's inputs stay in the tier `rc_calibration`.
- **experiments: the macro defocus experiment is removed** (2026-11-14; the owner: it is no part of the paper): `scripts/macro_defocus.jl`,
  `configs/experiments/macro_defocus.toml` (46 cases), the reproduction's stages `macro_defocus_prepare` and `macro_defocus` with
  their selection `MACRO_CASES` (configs/reproduce.toml, scripts/slurm/common.sh, case_task.sbatch, prepare.sbatch), its figure
  `Figures.figure_accuracy_vs_n` and the REPRODUCE.md row. The macro set's photographs, starts and board stay in their tiers (the
  tiers' accounting unchanged), read by no config. Test: no driver, config or reproduction stage of that name.
- **experiments: the frozen front ends leave the working tree** (2026-11-14; the owner's clean-up of the paper project):
  `experiments/frontends_history/` (the chains of 2026-11-05 … 09 as they ran: 55 files) is deleted. It stays the historical reference
  for the starts of record of tier `rc_baselines` where it already was: the repository's history (tag `v0.10.0`:
  `git show v0.10.0:experiments/frontends_history/README.md`) and `frontends/` beside the tier on the share. The pages and config
  comments that name it say so; a citation `frontends_history/<file>:<line>` in a provenance is a path of that tree. DUPLICATION.toml
  loses the three `srgb_transfer` exceptions of its Python converters (the files are gone).
- **experiments: the handover notes are removed** (2026-11-14; the owner's clean-up of the paper project): `notes/handover/`
  held two notes of 2026-11-02 and 2026-11-03 for a successor — a worked Brown–Conrady run, written against config and driver names
  that no longer exist (`refinement_chrome.toml`), and the Tamron panorama's rotation bundle. Nothing read them.

## 0.10.2

**RC 0.10.2 is RC 0.10.1 with the paper project's THORLABS chrome evaluation on the owner's Hungarian index, every FX3 chrome set
read from its RAW photographs, the chrome evaluation writing the paper's chrome tables and figure itself, and the owner's clean-up of
the paper project** (the floors: Core 0.10.6, Estimation 0.11.5, Features 0.9.6). A patch: the package's code is unchanged; no public name, key or
iterate of the package changes.

- **experiments chrome evaluation: the Hungarian index** (owner, 2026-11-14; one code path, every step Estimation's RANSAC framework on
  the `GridPoseProblem`). Each hypothesis is scored in board space: the dots' rays met with the board, matched one-to-one to the
  17 × 17 window's nodes by the Hungarian algorithm (Hungarian.jl) with a dustbin at half a pitch, MSAC at that half pitch (the
  threshold derived from the target record, never the table's). The local optimization is plain least squares on the scorer's
  matches, frozen, in board millimetres. After RANSAC: the Hungarian under the final pose and the pixel `refine_absolute_pose` on
  its matches until they repeat; one 8-connected component of matched nodes (ImageMorphology) or the photograph is refused. The
  gated lattice growth, its seed block and prediction gate are deleted; `[index] assignment_level` is retired (refused by name);
  `[index.refine_strategy]` is added. Distributions is dropped from the paper project; Hungarian and ImageMorphology are added.
  The LO's refit is accepted under the Gaussian shape (Estimation 0.11.2+: non-finite residuals weigh 0; the test that was
  `@test_broken` on 0.11.1 is a `@test`). A photograph with no circular detection, or whose pixel refine returns no pose, is
  refused, never a MethodError aborting the stage; a grazing ray is a miss.
- **experiments chrome evaluation: the RAW sets are detected on Features 0.9.6's bilinear green on the raw grid** (`[detect] image =
  "raw_green"`, retiring `"green_cells"`): each photograph's RAW mosaic (Features' `load_raw_mosaic`) under its frame's sensor model
  goes to `detect_sfop(m::RawMosaic; sensor_model)` (SFOP's 2× scale space for the seeds) and the seeded
  `detect_foerstner(m, seeds; sensor_model)` — the green photosites at the raw resolution, a red or blue site the mean of its usable
  green 4-neighbours, saturated and defect photosites masked, the covariances propagated from the photosites. The code path is the
  data's type (`detect_photo`, `detection_frame`, `photograph_size` on a `(RawMosaic, SensorModel)`); the rows are raw COLMAP
  pixels, no frame map. Features' polarity gate is configured: `[detect] polarity = "light"` (the backlit dots; SFOP's gate on the
  seeds, once; required, refused unless light | dark | none). One RAW path: a prepare set is detected on its RAW mosaics only (a
  decoded image of a RAW, `luma` on a prepare set, is refused), so the detections are always the photograph's own pixels — the
  frame-map plumbing of the detect stage goes (`detection_frame`, `to_photograph`, `photograph_size`'s map, the detect run's
  `image_frame`); a RAW frame its characterisation gives no noise law is refused by name. `image = "green_cells"` is refused by name with its replacement (one path).
  The detector's scales carry to the raw grid: `derivative_sigma = 2.0` and `precision_threshold = 1.0` raw px
  (1.0 and 0.5 green cells, the values of record). A raw_green detect names its sensor characterisation (`"none"` refused); under
  the FX3's (a gain) SFOP's λ₂ test reads the photosites' noise and `noise_sigma` is not read.
- **experiments: the chrome evaluation writes the paper's chrome outputs itself** (owner, 2026-11-14). Its paper config of record,
  `configs/experiments/chrome_evaluation_paper.toml` (`[chrome.<key>]` the seven sets, `[figures.color_wheels]`, `[tables.*]`,
  `[render]` — moved verbatim from geometric_figures.toml), runs `scripts/chrome_evaluation.jl`'s paper stages
  (src/chrome_evaluation_paper.jl): `sets` (each pending set's stages, the driver on the set's config) and `figures`
  (tab:refinement-chrome → table_radiometric_refinement.tex, tab:defocus → table_defocus.tex, chrome_all; numbers.tex with the
  `\defocus…` and `\nonConverged…` macros, names unchanged; fig:color-wheels), gated like every driver's figures stage
  (`figures_done`). The .tex files are byte-identical to
  what scripts/geometric_figures.jl wrote from the same evaluate products but for their generator comment, which names
  scripts/chrome_evaluation.jl (a test on fixture products, experiments/test/fixtures/chrome_paper, its golden written by the driver
  of record before the move). fig:focus-surfaces (it reads the refinements, not the chrome board) is drawn by
  `scripts/focus_surfaces.jl` (was scripts/geometric_figures.jl) on `configs/experiments/focus_surfaces.toml`
  (`[figures.focus_surfaces]`), its numbers fragment empty. configs/figures
  (color-wheels, table_defocus, table_radiometric_refinement), numbers.toml (the chrome paper config, then focus_surfaces),
  reproduce.toml (`chrome_index_grid` and `regenerate_paper` read the sets from the paper config), README and REPRODUCE.md follow;
  the chrome tables' tests move to experiments/test/chrome_evaluation_paper_tests.jl.
- **experiments chrome evaluation: `[index.ransac] min_trials = 200`** in every FX3 chrome config (owner, 2026-11-14: twice the 100
  every photograph of after4 stopped at, as margin; the confidence rule is met far earlier); `success_prob` and `max_trials` state
  their sources. It re-keys the index runs.
- **experiments chrome evaluation: a refinement's evaluation reaches the paper's table** — a test: a fixture run of a refinement case
  (its lens.json at case_run's keyed directory) is resolved by the set's `[evaluate]` model of that case, scored by the evaluate
  stage beside its start, and printed by the tables stage into tab:refinement-chrome (the refinement's retcode read from its run).
- **experiments chrome evaluation: the frozen correspondences are found by pose to 1e-12** (`_same_pose`: a re-representation of a pose,
  a few ulps, is the same pose; it was bitwise equality), and **`densest_window` reads 2-D prefix sums** over the nodes' distinct columns
  and rows (coordinate-compressed, however far a stray ray meets the plane) on every hypothesis. The evaluate stage's comments say
  what the rotation-centre refine minimises: the BEARING residual, whitened into pixels (owner ruling 2026-11-12 (B)(ii)).
- **experiments chrome evaluation: no mislabelled dot** (owner, 2026-11-14; precision over recall): `register_grid` refuses a
  photograph with a labelled dot nearer another node's image than its own under the final pose (`mislabelled_dots`, the 5 × 5
  neighbourhood of its node, px; `MislabelledDots`, a WARNING line and photos.json's reason). Tests: a synthetic labelling with two
  neighbours swapped and one with its outer column shifted are caught at exactly those dots; the synthetic grid and DSC06962 have
  none.
- **experiments chrome evaluation: every FX3 chrome set is read from its RAW photographs** (owner, 2026-11-14): the five sets that read
  decoded luma PNGs (fisheye_10cm_ag, fisheye_2710_si, tamron_20261108_ag, tamron_17cm_si, 14mm_central30cm) read their RAW folders
  (tier rc_raw_photos) through Features' raw module, as tamron_largeaperture and zoom48_f10 do; the decoded copies are no longer
  inputs (configs/data_artifacts.toml). A test holds every config of record to it.
- **experiments chrome evaluation: the evaluate stage reads the index camera through `config_source`** (a `[runs]` entry or an
  `[inputs]` entry).
## 0.10.1 (2026-11-14)

**RC 0.10.1 is RC 0.10.0 with the paper project's THORLABS chrome evaluation on Features 0.9.4's detectors**, two front-end fixes and
a RAW-path LM test (the floors: Features 0.9.4, IO 0.10.3; the package's code is unchanged). A patch: no public name, key or
iterate of the package changes.

- **experiments chrome evaluation: SFOP detection and a gated lattice index.** `[detect]` runs Features' `detect_sfop` (SFOP's scale
  selection for circular points) and localises each seed at full resolution by the seeded `detect_foerstner(T, img, seeds)` in the
  window of its own scale — on a RAW set, on the green cells (`LinearRaw`) under the frame's sensor model (Features 0.9.4); the
  window F-test at its effective degrees of freedom (Features 0.9.4) labels the seeds between dots, on line artefacts and on letter
  strokes unknown. `[detect]` keys: `noise_sigma`, `significance_level`, `precision_threshold` (SFOP's), `derivative`,
  `derivative_sigma` (the seeded windows'). `[index]`: the dots are assigned by a lattice grown from a RANSAC seed block, each node's
  prediction gated by the F quantile of its prediction interval (`assignment_level`) and mutual-nearest (Features' `MutualNN`), the
  next ring competing; the pose refitted on every grown dot; the board matched against an infinite lattice (no placement refusal:
  the densest window, any whole-pitch offset), the offsets and symmetries chosen by the set's one rotation centre (moves recorded as
  "reoffset"); a photograph whose labelled dots form more than one connected component is refused with the components' sizes
  (`LatticeComponents`, a WARNING line, photos.json). `[evaluate]`: a photograph the model's start projects no dot of is refused per
  model (`start_refusals!`, recorded in `eval_<label>.json` "refused" and summary.md), never looked up (it was a KeyError on the
  fisheye 10 cm set). New dependency of the paper project: Distributions (the assignment gate's F quantile).
- **experiments chrome evaluation: the evaluate stage reads an index camera that is a `[runs]` entry** (the Tamron f/8 and 14 mm sets'
  Kannala–Brandt fits, frontends.toml `[fits.*_kb]`): `index_camera_size` reads it by `config_source`; it was a KeyError "camera"
  (`config_input` reads `[inputs]` only).
- **experiments front ends: a Schöps view with no features counts 0** (`view_feature_counts`, views.txt); it was a `MethodError` of
  `sum` over no values, which stopped `import_schoeps_run` on a set with an image Schöps's tool found no pattern in.
- **Tests: the RAW path's LM.** An LM over photosites under a Poisson–Gaussian sensor model, its stops' residual resolution
  `lm_abstol(s) > 0` (no RC test ran one before): Stalled at a step-size stop away from the minimum, Success at the minimum and at a
  step-size stop restarted there (its cost far above the quantisation's square: the stationarity postcondition decides).

## 0.10.0 (2026-11-14)

**RC 0.10.0 is RC 0.9.0 with the stack's sensor model** (each RAW photograph's Core `SensorModel` by its EXIF, the residuals whitened
under its noise), **VisualGeometryToolkitFeatures 0.9.2's image circle**, Levenberg's λ·I damping on every LM again, no run-configured
globals, the paper project on PanoramaStitching 0.8 and the findings of RC 0.9.0's final review (the floors: Core 0.10.3, IO 0.10.2, Features
0.9.3; the paper project PanoramaStitching 0.8.0, BlobBoards 0.7.1). RC 0.9.0's two known limitations are lifted. A minor release: it breaks config keys, public names and every LM's iterates.

- **New: `camera.sensor_characterisation`** — the body's sensor characterisation, an IO `vgio/sensor_characterisation` record
  (Core's `SensorCharacterisation`, read by IO's `load`), or `"none"` (the camera has none). Each RAW photograph's Core `SensorModel`
  is resolved from it by its own EXIF capture settings (Features' `load_raw_metadata` → `extract_capture_settings` → Core's
  resolver `sensor_model`, a lookup) and held per view on the calibration state (`CalibrationState.sensor_models`): its black
  (the file's, a measured channel black or a dark frame's per photosite) is the photographs' black, and its defects are masked out
  of the fit (Core 0.10.3's public `defect_mask`). `parameters.json` records the key and, per photograph, the resolver's reason
  (`Matched`, `NoCharacterisation`, `UnmatchedISO`, `OtherBody`, `DarkOutOfBand`), each slot's source and the model's
  `content_digest` (`sensor_record`). **Breaking:** a run reading RAW photographs refuses an unstated key; `"none"` is the documented
  fallback (the file's black, recorded `NoCharacterisation`, never silent). A decoded image has the absent model; naming a record
  for decoded photographs is refused. New public: `sensor_characterisation`, `photo_sensor_models`, `sensor_record`,
  `NO_SENSOR_CHARACTERISATION`, `names_input_path` (the drivers' one test of an input path: `"none"` names none).
- **The residuals' weighting follows the camera's sensor model** (owner ruling 2026-11-14; D14), no option: a RAW photograph whose
  model has a noise with a variance law (a gain: PoissonGaussian, or a RelativeNoise) is read as Core's uncertain image (each value
  with its variance, Core's law) and every residual and its Jacobian row are divided by the value's σ — whitened least squares — in
  both LM routes (the per-view host residual and device row kernel, the per-tile blocks) and the closed-form blocks (their
  reductions and QR systems), through ONE weighting (`_whiten`, src/calibration/weighting.jl). No characterisation, a model without
  a variance law (a ReadNoise alone) or a decoded image: plain values, the rows as measured — bit for bit today's arithmetic (the
  rows' weights are the type `Unweighted()`, never a flag). The penalties' data scale (the grid and focus priors, the vignetting
  smoothness) and the relative RMS are in the cost's norm (`data_norm2`). New: `CalibrationState(…; weights)`,
  `masked_observations`, `photometric_residuals` (the unweighted Φ − ϕ, the diagnostics'); `residuals` and `cost` are the whitened
  ones under a variance law. **Refused by name:** `refine.weighting` (an option of the development branch: "weighting follows the
  camera's sensor model").
- **Every LM states the residual norm the data resolve** (Core's `LmStops(; residual_resolution)`, was 0): `lm_abstol(s)`,
  the RAW quantisation, never the noise level (owner ruling 2026-11-14) — each row's code step `q` (one RAW code in its units,
  `1/(white − black[c])`, Core 0.10.3's public `linear_quantum`) over √12: `Σᵢ (qᵢ/σᵢ)²/12` for whitened rows, `Σᵢ qᵢ²/12` as measured, `0` for decoded images. A fit at the
  noise level is decided by Core's stationarity postcondition. New: `CalibrationState(…; quanta)`; `masked_observations` returns
  the rows' code steps too.

- **The measured image circle is the lens coverage's OUTER edge.** `run_image_circle` (the one call path: `camera.image_circle_frames`
  → Features' `detect_image_circle`) now gets Features 0.9.2's measurement: the frames' per-pixel maximum resampled log-polar about
  the image centre, each angle's significant log-brightness drops the rim's candidates, and the RANSAC ellipse through them — no
  dark-region hull (no MSER) on that path. The return type is unchanged; the measurement runs on the run's device (`prepare_refinement`'s
  `backend`, given as `run_image_circle(opts, lens0; backend)`, by default Core's `default_device()`). The imaged region is built as
  before (the bounded camera's lens coverage ∩ the sensor, Core's `lens_coverage` / `imaged_region`, the saturation and black masks);
  there is no rim margin (the vignetting models the ramp). A run that measures its image circle gets a different coverage ellipse — on a soft rim
  at the ramp's foot, where the hull fit landed mid-ramp (the paper's 14 mm at f/4: semi-axes ≈ 1719 / 1717 raw px, were ≈ 1597 / 1596) — and with it a different
  data region and calibration; the pipeline reference cases measure none (their goldens are unchanged by it).
- **Breaking: retired `camera.image_circle_min_hole_area_px`**, refused by name ("delete the key"): the hull's smallest kept hole has no
  meaning on the rim's profiles (Features 0.9.2 ignores `min_hole_area`), an entry of the one retired-key table. `CameraOptions`
  loses its field `image_circle_min_hole_area_px` (the owner's ruling: no inert knob, even in a patch). No other key changes;
  `camera.image_circle_outlier_halfwidth_px` is still required with `camera.image_circle_frames`. Migration: delete the key from
  every config and the keyword from every `CameraOptions(…)` call (the paper project's configs of record are updated).
- **New: `start_is_grid(opts)`** (public): whether a refinement's start is a grid camera — every start but a parametric record's
  (`io.init = "lens:…"`) without a non-central front end (`frontend.noncentral` replaces any start by its non-central grid lens). THE decision of the start's kind: `prepare_refinement` takes it (and refuses a front end whose start
  disagrees), and the paper project's drivers read it to know what an init's `io.save_init` writes (`grid.json`, else `lens.json`)
  instead of testing the `io.init` prefix themselves.
- **Refused at load: `grid.prior_weight = "gcv"` with `"grid"` in `lm.fixed`** (with a grid prior, on a grid route): the grid is
  held, so GCV has no grid column to weigh (it linearized the held patches). Migration: state a number, or free the grid.
- **Breaking: `prepare_refinement`'s default device is Core's `default_device()`** (GPU first; was the CPU), and the image circle
  measures on it. Migration: give `backend = KernelAbstractions.CPU()` to keep the CPU.
- **New: `write_pipeline_config(dir, opts; header)`** (public): a run's options file, its `io.out` absolute — bin/refine.jl's and
  bin/material_grid.jl's one writer. Each LM problem's kept NLLSsolver cache is built for the solve's damping (it was rebuilt on
  every solve).
- **Breaking: the run-configured globals `FREEZE_MATERIAL`, `with_freeze_material`, `MU_BOUNDS`, `JAC_CHUNK`, `LM_LOG` are deleted**;
  their settings travel with the problem: `"material"` in `fixed` (Θ_m alone held while a surface parameter rides the block),
  `CalibrationState(...; mu_bounds, jacobian_chunk)`, `run_lm!`/`run_lm_grid!(...; lm_log)`. A run no longer leaks its settings into
  the next problem of a session (the grid prior's GCV holds Θ_m through its own problem's `fixed`, as the refinement does).
- **Fixed: a regeneration hint names only its writer's options.** A record of an earlier run (or a grid start without its
  `initialization_region.json`) names `bin/refine.jl --config=<dir>/pipeline_config.toml` only when that file's `io.out` is the
  record's directory (`realpath`; bin/refine.jl writes its options into its own `io.out`, absolute); in a shared directory (a saved
  initialization's), with a relative `io.out` or a malformed file, no options are named.
- **Breaking: every LM of RC damps with Levenberg's λ·I again** (Core 0.10.2's `LevenbergDamping`, RC 0.8.0's damping; the floor: Core
  0.10.2): the per-view and per-tile grid refinements and the closed-form blocks' joint LM, one value (`LM_DAMPING`), no option
  (the owner's ruling). RC 0.9.0's λ·DᵀD stalled the non-central origins and was slower on the Tamron and grid cases; RC 0.9.0's
  known limitation is lifted (the five non-central truth asserts hold again at the tolerances measured before it). Every refinement's
  iterates change back to 0.8.0's: the pipeline goldens are re-recorded and match RC 0.8.0's (blob_kb's final cost 0.0684538 and
  bearing median 0.0867 px, siemens_coarse_full's rel-RMS 0.04003 — both capped runs; the unit golden's three-iteration LM cost
  8.79e-8), the linear-solve stage unchanged. Measured against 0.9.0 (20 iterations, dgx01): the Tamron kb reaches 1 % of its
  final cost 1.8× sooner (313 s against 568 s); the fisheye bc5 converges more slowly under λ·I, as Core measured.
  Under Core 0.10.2's success rule a restart from an exact fit is `Stalled` (its resolution is relative to the start; the
  residual norm RC now states, above, is the RAW quantisation, `0` for decoded images).
- **The stack floors:** Core 0.10.2, IO 0.10.1 (a length reads back with its identity: one unit of its 15th significant digit,
  where 0.10.0 read it back within an ulp or two; the records' round-trip tests state that bound).
- **The paper project on PanoramaStitching 0.8** (compat `0.8.0`; RC 0.9.0's known limitation lifted, its runtime version guard
  deleted): near-duplicate pairs by Core's view overlap, `near_duplicate_overlap = 0.90` stated in the three panorama configs of
  record (no `overlap_deg` or `max_lag` is set: `pairs = "all"`). A retired `[example.registration]` key is refused again AT LOAD,
  by PanoramaStitching's one public list (`PanoramaStitching.refuse_retired_registration`), so such a config neither plans nor keys
  a run; that load is the one refusal point (the stages' entry no longer reads a list through a runtime `getglobal` into the
  example, which failed in the caller's world).
  The panorama registrations change with 0.8's always-on near-duplicates (fewer tracks; the bundle's rotations differ).

## 0.9.0 (2026-11-14)

**RC 0.9.0 is RC 0.8.0 on the registered wave-10 stack** (Core 0.10.1, IO 0.10.0, Estimation 0.11.1, Features 0.9.1; the paper
project also Eval 0.8.0, PanoramaStitching 0.7.0, BlobBoards 0.7.0, LocalOptimizationDoneRight 0.7.0), with every breaking change of
the wave carried through. No latest exception: `.release/latest-exception.toml` is deleted.

- **Breaking: one LM, Core's** (Core 0.10's "NO DUPLICATIONS" ruling). Both calibration routes (the per-view parametric lenses and
  the per-tile grid lenses) and the closed-form blocks' joint LM run through Core's one adapter, `lmsolve_chart!(::LmProblem,
  ::LmStops; callback)` and `lmsolve_chart!(ews, lm, stops)`; no NLLSsolver `optimize!`/`NLLSProblem`/`OptimizeCache` outside Core.
  The device normal equations and the concurrent views are kept. `run_lm!` / `run_lm_grid!` return Core's termination record
  `(; code, rule, iterations, cost)` (`cost` = ‖r‖², was NLLSsolver's `bestcost` = ‖r‖²/2) in place of an `NLLSResult`;
  `lm_problem` / `lm_problem_grid` are removed (`build_problem*` take `fixed`); `lm_stationary`, `lm_stationary_at` and
  `STATIONARITY_TOL` are removed (Core decides the stop's success). Core's success rule: a cost-decrease stop short of
  stationarity is `Stalled` (it was `Success`). `lm.iterator` (and `RC_LM_ITERATOR`) is retired, refused by name: the damping is
  Core's one (λ·DᵀD). `build_problem` / `build_problem_grid` return Core's `LmProblem`; `lm_termination(res)` takes the one record.
  `fix_origins` is retired: a non-central lens's chief-ray origins are held by `fixed = (:origins,)` (`build_problem_grid`,
  `run_lm_grid!`; the pipeline's `lm.fixed = ["origins"]` as before). Under the one `fixed` rule every hold goes through `fixed`,
  never a boolean: `edge_block`, `height_degree` and `inplane` choose which surface parameters the model HAS (its structure, the
  same keywords on both routes: the per-view `height_free` / `inplane_free` are `height_degree` / `inplane`), and `:edge`, `:height`,
  `:inplane` in `fixed` hold them (`SURFACE_HOLDS`); `tie_f` is removed (`:aspect` in `fixed` ties the focal lengths, as `lens_free`
  already read it); `origin_gauge` / `rotation_gauge` fix gauges exactly (structural), so they stay keywords. The two routes share
  one per-iteration callback (`_lm_callback`: the trace, `iter_callback`, `lm.log`) and one read-back (`read_back!`); `run_lm!`'s raw
  NLLSsolver `callback` keyword is removed (neither route takes one).
  Needs VisualGeometryToolkitCore ≥ 0.10.1 (`LmProblem(…; fixed)`). Migration: read `.cost` (`res.bestcost` → `res.cost / 2`);
  drop `lm.iterator`; `fix_origins = true` → `fixed = (:origins,)`; `tie_f = true` → `:aspect` in `fixed`; `height_free = true` →
  `height_degree = 1`, `inplane_free = true` → `inplane = true` (held: `fixed = (…, :height)` / `(…, :inplane)`).
- **Breaking: RC's records on IO 0.10's envelope.** The kinds move up a major version: `rc/radiometric_camera/v3.0`,
  `rc/radiometric_target/v2.0`, `rc/calibration_state/v3.0`, `rc/run_parameters/v2.0` — one `units` and one `conventions` object
  (no top-level `length_unit`, `pixel_frame`, pose declarations), written by IO's `write_record(…; style)`, read by
  `VgioStyle(record)`. The blur blocks name their plane, IO 0.10's blur body: the in-focus blur `"pixel_frame": "colmap"`, the
  print blur `"frame": "target:board"` (`PRINT_PLANE`); unlabeled in memory as before. A record of an earlier RC version is
  refused by name ("written by RC ≤ 0.8 on IO 0.9: regenerate it"): a derived result, regenerated by its run (the owner's ruling
  of 2026-11-14, as IO 0.10's for derived results). The one exception: **an `rc/calibration_state` v1 or v2 reads through its
  version step to v3.0** (a state of record no run regenerates: the detector-bias ground truth), IO's `migrate_envelope` and IO's
  own steps for every embedded body (`migrate_record`). `READ_SCHEMAS` is removed. Migration: re-run the stage that wrote an RC ≤ 0.8
  camera, target or run record; compare a schema with `RECORD_SCHEMAS.<kind>`.
- **Breaking: the terminology taxonomy** (Core 0.10, no aliases): `Csponds` → `Correspondences`; the Schöps detections are
  IO 0.10's `load(TargetObservations, …)` (`TargetPoints2D` is gone); no bound name holds "measured" (a provenance value only):
  `measured_channels` → `fit_channels`, `measured_material` → `material_sources` (and the private `_measured*` names).
- **Breaking: Core's `extent`** for a camera's pixel extent (Estimation 0.11 deleted `image_size`): `load_session`'s field
  `image_size` is `extent` (a `Size2{Int}`).
- **Breaking: a non-central refinement chooses its vignetting surface.** `frontend.noncentral` with `vignetting.model = "radial"`
  (the default) is refused at config load: a radial profile is centred on the principal point (`vignetting_center`), which a
  non-central lens has not (it aborted in the refinement's preparation, a MethodError). Migration: set `vignetting.model =
  "tensor_product"` (a self-check, `selfcheck.only`, fits no vignetting and is accepted).
- **Breaking: the retired config keys are refused by name with their migration** (`RETIRED_KEYS`): `edge.free` (RC 0.8.0: the
  printed edge is free whenever `edge.sdf` is set; hold it with `lm.fixed = ["edge"]`), `lm.fix` (renamed `lm.fixed`),
  `lm.iterator`, `io.photosites`. Before, `edge.free` and `lm.fix` were refused only as unknown keys.
- **Breaking: every grid the pipeline writes is a valid start.** The start's initialization region is a record of its own beside
  the grids, `initialization_region.json` (`rc/initialization_region/v1.0`, `save_initialization_region`), written with every grid
  the pipeline writes (the result and its start, `io.save_init`'s `grid.json`, the checkpoints `grid_ckpt.json` / `grid_it<N>.json`);
  `load_start("grid:…")` and `stored_initialization_region(grid_path)` read it there. It is the region's ONE record: any of Core's
  region types (IO 0.10's region body: `{"rect": …}`, `{"intersection": […]}`, …), so a run whose region is an intersection reloads
  its own grid as a start; `rc/run_parameters/v2.0` (parameters.json) holds no `initialization_region` (nor
  `initialization_region_unrecorded`). Before, the region was read from the run's `parameters.json`, which a saved initialization
  (`start_from_init`) and a crashed run's checkpoint lack, so neither could start a run. A grid written without a region (a run from
  a parametric start) removes a region another run left in its directory. A grid without the record is refused, naming the command
  that regenerates it (a refinement's grid: `bin/refine.jl --config=<its run directory>/pipeline_config.toml`; a saved initialization's: the refinement whose `io.save_init` wrote it); the experiments' init runs that save a grid start re-key on the record's
  schema, so cached ones regenerate. Migration: `stored_initialization_region(dir/parameters.json)` →
  `stored_initialization_region(dir/grid.json)`; re-run a refinement to start from its grid.
- **A non-central start reads its run through the one resolver** (`schoeps_run_file`: the run's import camera.json, else its
  intrinsics0.yaml), as the central start does: `noncentral_gridlens` and `load_camera("noncentral:…")` take a run directory; a
  green-cell front end's run starts from its raw-grid import.
- **The Schöps front end opens no photograph for its image size** (it decoded the whole first RAW only to compare its size, and read
  a stored start twice): the size is the start camera's, the stored start is read once, and every photograph — a decoded image
  as a RAW mosaic already was — is checked against it where it is read (`photo_observations`, its one decode), refused otherwise.
  The check comes after the preparation, at the photograph's decode: Features' RAW metadata (`load_raw_metadata`) states no image
  size, so a wrong-size photograph is refused there rather than before the start is built.
- **Breaking: a free printed edge on a parametric lens is refused** (`edge.sdf` without `"edge"` in `lm.fixed`): the per-view LM
  has no edge block, so the edge was recorded free (`printed_edge.free`) and kept the material block variable while never solved.
  Migration: `lm.fixed = ["edge"]` (every parametric config of record has it) or a grid start.
- `dataset_features` refuses a detections file of more than one target (their 0-based point ids collided in the flattened list,
  in the Dict's order): the Schöps front end reads one board's detections.
- An RC ≤ 0.8 record is refused with the command that regenerates it (`bin/refine.jl --config=<its run directory>/pipeline_config.toml`).
- **Known limitation: the paper project's panorama experiments need PanoramaStitching 0.8** (under PanoramaStitching < 0.8 experiments refuses
  to run PanoramaStitching's example, naming the need: every stage of panorama_vignetting and panorama_stitching that runs it; the
  plans, the configs and the run keys read on any version). The owner's ruling of 2026-11-14 leaves
  `near_duplicate_deg` unset (0.7 decides near-duplicate pairs by the optical axes' angle, rejected; 0.8 decides them by Core's view
  overlap), and PanoramaStitching 0.7's example requires the key. Restored in RC 0.10.0 with PanoramaStitching 0.8. Every stage
  that runs the example calls one entry first (`require_panorama_example`), before it removes or writes anything: the version
  guard, then PanoramaStitching's own `refuse_retired_registration` (a retired key named with its successor); a config without
  `[example.registration]` is refused at load. the experiments' compat stays PanoramaStitching 0.7; RC 0.10.0 raises it to 0.8 and
  deletes the guard.
- **Known limitation: non-central grid calibrations converge more slowly** under Core 0.10.1's λ·DᵀD damping, which stalls the
  chief-ray origins: on the synthetic non-central truths the cost ratio plateaus near 1e-8, where it reached about 1e-18 before
  (repro: dgx01 ~/logs/dgx-agent/runs/rc09lm/nc_damping_repro). The fix is expected in Core 0.10.2 and an RC patch release; the five
  affected test assertions (noncentral_truth_tests, noncentral_tests) are `@test_broken` at their unchanged tolerances until then.
- The goldens re-recorded on the wave-10 stack (Core's success rule, the LM's own stops, the records' new envelope).

## 0.8.0 (2026-11-13)

**RC 0.8.0 targets the VGTK 0.9 stack** (Core 0.9.9, Estimation 0.10.3, IO 0.9.5, Features 0.8.4, Eval 0.7.4, PanoramaStitching
0.6.5). Wave 10 (Core 0.10.0, IO 0.10.0, Estimation 0.11.x, Features 0.9.0, Eval 0.8.0, BlobBoards 0.7.0, LocalOptimizationDoneRight
0.7.0, PanoramaStitching 0.7.0) registered during
its gate; the cascade is RC 0.9 (owner exception 2026-11-13 to the latest-only rule, recorded in .release/latest-exception.toml).
The exception covers, at exact versions (resolved / registry latest): VisualGeometryToolkitCore 0.9.9 / 0.10.0, IO 0.9.5 / 0.10.0,
Estimation 0.10.3 / 0.11.1, Features 0.8.4 / 0.9.0, Eval 0.7.4 / 0.8.0, BlobBoards 0.6.3 / 0.7.0, LocalOptimizationDoneRight 0.6.2 /
0.7.0, PanoramaStitching 0.6.5 / 0.7.0. The gate's fresh-clone leg (cau-release's release_leg.sh, dgx01 job 15324) succeeded on its suite
(15619 succeeded / 1 broken, then 45 / 45) and then stopped on release_leg.sh's own latest-only check on
exactly these eight pairs, which does not read the exception: counted green on its tests (owner ruling (a), 2026-11-14); a follow-up
teaches release_leg.sh the exception.

- **Breaking: RAW or decoded is the photograph's TYPE, never a flag** (owner ruling 2026-11-13): `io.photosites` is retired and
  refused by name. `photo_file` finds a photograph's file (its RAW file when the folder holds one), `load_photo` is the one loader
  (a RAW file → Core's `RawMosaic`, a decoded image → its array), and `photo_observations` dispatches on the type (a mosaic's green
  photosites on the raw grid; a decoded image's pixels). A RAW photograph requires `io.channel = "G"`. Migration: delete
  `io.photosites`; state `io.channel = "G"` for RAW photographs.

- **Breaking: nine exported functions with no caller are removed** (owner ruling 2026-11-13, the reorg audit): `load_mask`,
  `with_values`, `ω_of_ray`, `views_of`, `set_intrinsics!`, `with_intrinsics`, `trace_callback`, `grid_model_from_samples` (and its
  helper `_controls_of_samples`). Migration: `Focus(tps, t̂_i)` for `with_values`; `[view_of(s, k) for k in 1:n_views(s)]` for
  `views_of`; `set_camera!(s, RadiometricCameraModel(lens, focus, r, c.vignetting, c.in_focus_blur; aperture))` for
  `set_intrinsics!`; `v.pose.R * π̂⁻¹(v, 𝐩, 𝐚)` for `ω_of_ray`.
- **Breaking: Core's `grid_nodes` and `hammersley` are no longer re-exported** (one public name per function): callers call Core's,
  `VisualGeometryToolkitCore.grid_nodes` / `.hammersley`.
- **Breaking: one holding keyword, `fixed`** (the fixed-keyword rule): `build_problem`, `lm_problem`, `run_lm!` and `calibrate!` no
  longer take `distortion_fixed`; `fixed` names the blocks held and the lens model's parameters held, by Core's names. Migration:
  `run_lm!(s; fixed = (:poses,), distortion_fixed = (:cx, :cy))` → `run_lm!(s; fixed = (:poses, :cx, :cy))`. An entry that is
  neither a block nor a parameter of the lens is an `ArgumentError`. The gauge helpers are renamed: `hard_pin_basis` →
  `gauge_basis`, `pin_center_value` → `gauge_center_value`, `rotation_pin_coordinates` → `rotation_gauge_coordinates`.

**Breaking (a minor bump): on the released VisualGeometryToolkit 0.9 stack** — Core 0.9.4, IO 0.9.2, Estimation 0.10.0,
Features 0.8.2, Eval 0.7.1; the paper project also PanoramaStitching 0.6.2, BlobBoards 0.6.1 and LocalOptimizationDoneRight 0.6.1
(Estimation 0.10's retired free-SplineLens image-space refine of `refine_camera_and_poses` is reached by no RC or experiments call:
the chrome evaluation's rotation-centre refine holds the camera, `:controls` for a SplineLens, and the parametric starts refine
parametric lenses). The breaking items, each below:
the 0.9 stack and its SplineLens grid camera; Core's §14 termination record in parameters.json (`retcode` names, `stop_rule`);
the parameter-holding keyword `fixed` (`[lm] fix`, `[edge] free`, `[material] freeze`, `distortion_free`; `unfixed` is no keyword); the
vignetting centre fixed at the principal point (the free centre's keys refused); the device Jacobian's internals; the board
warping's points; the GCV curve's owner; and the rc-dedup round (the vignetting block, `xtol`, `sweeps`, `rc_backend`, the one RAW
path, the paper numbers, the chrome registration records, the renamed config keys).

- **The LM's Jacobian and its Gauss–Newton blocks on the device** (owner ruling 2026-11-13: GPU first). Both LM routes (the per-view
  parametric lenses and the per-tile grid lenses) keep their kernels' outputs on the state's backend: the dual chunks' partials are
  gathered into one slot array, ONE KernelAbstractions row kernel (`_jacobian_row!`) writes the measured channels' rows only (a
  single-channel state no longer assembles, copies and drops two thirds of them) with the focus chain, the patch fan-out, the parts', the
  heights' and the printed edge's columns, and `JᵀJ`/`Jᵀr` are the backend's products (`_gauss_newton_blocks`; the channel ties, the
  charts' column scales and the held focus nodes are one column map on the n × n blocks): only the blocks reach the host
  (`computecostgradhess`, now also the per-view residual's). The host scatters (`Threads.@threads`), the dense host Jacobians and
  their host BLAS products are deleted; `computeresjac` keeps its host values for the tests and diagnostics. The sums run on the
  device, so the blocks match the host products to rounding, not bit for bit.
- **The closed-form radiometry on the device, and the backend tag `gpu`** (owner ruling 2026-11-13). The parts of the closed-form
  blocks are rendered once on the state's backend (`all_parts` → `RadiometryParts`, Core's `pointwise!` of `pixel_parts`, Float64 on
  every backend); their design rows (the QR solves of `Θ_l` and `Θ_v`, kept: column-pivoted, the GCV and penalty unchanged) and the
  reductions of the cost and the normal equations (`parts_cost`, `solve_μ!`, `solve_radiometry_gn!`) are KernelAbstractions kernels
  there, Estimation's `refine_camera` precedent. `solve_radiometry_gn!` is Core's chart LM (NLLSsolver) on that reduction and returns
  the stack's termination record `(ReturnCode, StopRule)`; its hand-written Levenberg–Marquardt is deleted, its keywords are
  `max_iter`/`ftol` (`iters`/`rtol` retired), μ moves on the LM's bounded chart. The LM's parts kernel `_PartsAt` is `pixel_parts`'s
  one path. `view_parts`, `light_response` and `inner_integral` are deleted. `compute.backend = "gpu"` (`GPUCompute`, any
  KernelAbstractions GPU backend through Core's `default_device`) replaces `"cuda"` (`CUDACompute`): the old tag is
  refused by name (the config tag; the type's old name is gone); the default is `"gpu"`. The quality item "quality: GPU first" fails on a config or Slurm script pinning the CPU
  outside the listed test oracles, the tag `"cuda"` in a config, or a `Threads.@threads` loop in src/calibration_pipeline outside
  its reasoned allow-list.
- **Breaking (the IO audit, 2026-11-13: every model through VisualGeometryToolkitIO).** The radiometric camera's body (in
  `rc/radiometric_camera/v1` and `rc/calibration_state/v1`) is IO's BOUNDED camera body (`camera_json(::BoundedCamera, frame;
  length_unit)`): the lens coverage (the image circle, θ included) and the sensor pitch now round-trip (they were dropped). Both kinds
  step to `rc/radiometric_camera/v2` and `rc/calibration_state/v2` (written); a v1 record reads as before (`READ_SCHEMAS`). The
  in-plane print error is written by IO's target-calibration (composition) writer (the same bytes). `parameters.json` is an `rc/run_parameters/v1`
  record (`save_run_parameters`/`load_run_parameters`): what the run was given and measured, never a value the model records hold,
  ASCII snake_case keys without units (lengths in metres, its `length_unit`); `load_pipeline_metrics` returns `bearing_error` and
  `camera_moved` (were `bearing_px`, `camera_moved_mm`). the experiments' run tables (chrome summary and registration, the paper tables'
  long form, texture curves and per-cell) are VisualGeometryToolkitEval's `.tsv` tables. On VisualGeometryToolkitIO 0.9.4: the chrome evaluation's Förstner
  detections go through IO's detection codec (`detections/<photo>.json`, the PointFrame case), and the panorama experiment's fits
  are `vgio/panorama/v1` files (`panorama_<subset>_<c>.json`, `panorama_2d_G.json`, `panorama_<slug>.json`). `parameters.json` no longer writes `image_circle` (its `{center, semi_axes}` lost θ): the
  coverage is the camera record's. The retired `radial_projective` vignetting reader is deleted. experiments: the panorama experiment
  stores every vignetting as IO's `vgio/vignetting/v1` (the learned V `V_<map>.json`, each rooftop fit `vignetting_<subset>_<c>.json`,
  the 2-D fit `vignetting_2d_G.json`; the stored map input `V_lat24.json`), maps rendered from the models when needed —
  `export_vignetting` and the figures' CSV parser are deleted; the chrome evaluation's photograph size is the camera of record's
  extent (not the detect run's summary CSV); the detector-bias scores, the chrome index grid's rows/table and the parametric starts'
  fit report are IO records (`scores.json`, `rows.json`, `table.json`, `.fit.json`); the stitching log's tables are
  VisualGeometryToolkitEval's tab-separated tables (`.tsv`); the texture sheet is IO's target record (`target.json`, made
  once by scripts/make_sheet_target.jl); the ground truth's field stop is gt/camera.json's lens coverage (no image_circle.json). The affected stages re-run (the audit report lists them).
- **The refinement compares at the GREEN PHOTOSITES of the raw mosaic** (owner ruling 2026-11-12): `io.photosites = true` reads
  each photograph's RAW file in `io.images` (Features' `load_raw_mosaic`) on the raw grid — the camera's own pixels, nothing
  resampled, no geometry changed — ONE residual per green photosite (`io.channel` "G", the default under `io.photosites`; anything
  else refused by name), its linear value Core's `linearview(m)` (lazy). The residual mask is the public `green_photosite_mask(m)`,
  broadcasting over Core's `photosite_colors` and `saturationview` — both greens, the censored photosites dropped at both ends (raw
  = 0, the ADC floor; raw ≥ white, Core's saturation; a value below the black level is noise and stays) — AND the existing masks; the
  edge selection reads a green photosite's diagonal (green) neighbours. No gain between the two greens is modelled.
- **The /simplify review (2026-11-13: the stack's functions for RC's copies, one name per concept).** `vignetting.pin` is
  retired (refused by name: its one value, the exact gauge, is the only gauge); the `unfixed` keyword's refusal is deleted;
  the experiments' texture recovery takes its image-circle mask from Core's `pixel_mask` — the pixels whose CENTRES lie in the circle
  (it tested the pixel's corner (u, v), half a pixel off); the chrome evaluation reads Core's `nearest_point`, `intersect(::Plane3D,
  …)` and `model`, and Features' `load_raw_metadata`; the five per-case Slurm launchers are one `case_task.sbatch` (its
  `EXPERIMENT` and `STAGES` from configs/reproduce.toml), and the decode-and-detect launcher `prepare.sbatch` requests a GPU (detect
  on the device). A run key of a config of record whose `[pipeline.vignetting]` stated `pin = "hard"` changes (the key is gone).
- **A result change: the auto gain's statistic** (experiments `decode_raw`, `gain = "auto"`): each RAW is decoded once, and the
  99.99th percentile is read from the green photosites' linear values (Core's `linearview`, `photosite_colors` 1 and 3: the one RAW
  path's green rule), no longer from a second `load_raw(...; mode = :bin2)` decode of all three channels. On the one config of
  record with an auto gain, `chrome_evaluation_fx3_tamron_largeaperture`, the gain moves from 6.530 to 4.842: that set's decoded
  photographs, detections and registrations change (8, 97 and 88 of 190 photographs under the chrome index grid's three variants),
  and its evaluate rows with them (sitla_spline_start's median 0.1893 → 0.1883 px). The prepare run's key holds `"auto"`, not the
  value, so an existing run is reused as it is; a fresh run is the new decode.
- **experiments efficiency (no result change; bit for bit where measured).** The texture test's page rows are one KernelAbstractions
  kernel, its blocks built directly, its RAW decoded once and its CGLS solves concurrent; RC's `compute_mask` is one kernel on a
  required `backend` (the texture mask 274 s → 12 s on the GPU); the chrome index and evaluate stages are one loop in src
  (`register_photos`, `evaluate_models`), every photograph registered once with its local optimization counted (photos.json), the
  index grid an array of one task per set; geometric_figures reads the chrome configs through one include.
- **Breaking (the /simplify review, phase 2).** The non-central lens is Core's bounded `NoncentralSplineModel` itself; its
  aperture radius r(𝐩) is an `ApertureSpline` carried beside it (`RadiometricCameraModel`'s and `View`'s `aperture`;
  `NonCentralAperture` is gone, `noncentral_gridlens` returns `(lens, aperture, c*)`, `NoncentralLens` names the lens types). The
  focus block is Core's `restrict(ParameterManifold(focus; box), free)` (`focus_block`, was `FocusVar`; focus-ab's held nodes are the
  state's `focus_fixed`, `FOCUS_FREE` is gone). `load_camera` returns every record as it is (a non-central `grid:` record keeps its
  chief-ray origins; `lens:` the bounded camera). The image-circle literals are absent rather than 0 / −1 when not stated; a
  render-settings record no longer writes `normalize_aperture` (an older one's is ignored); Julia 1.13 only.
- **New public names, one statement each of what was written out several times** (the final duplication audit):
  `green_photosite_mask(m)`, `is_raw_file(path)`, `raw_file(dir, image)` (was `_raw_file`), `raw_image_size(path)` (a RAW
  photograph's raw-grid size) and the exported `luma_image(img)` (the Rec. 709 plane of an H × W × 3 image). The green cells'
  mode-level frame map is Core 0.9.4's `raw_frame_map(LinearRaw{:green_cells})`; the experiments' `texel_weights` reads Core's
  `is_inside_texel_centres`.
- **The stack's public names for what RC copied or called privately** (VisualGeometryToolkit 0.9.2): the raw mosaic's names
  (`RawMosaic`, `LinearRaw`, `raw_frame_map`, `photosite_colors`, `cfa_string`, `raw_provenance`) from Core (Features 0.8 re-exports
  none); the board relief by Core's `with_parameters(::MappedSurface, θ)` (RC's `_with_heights` deleted); the lattice point by Core's
  `normalized_point` (RC's `_lattice_point` deleted: K⁻¹𝐩 by the inverse instead of RC's division, an ulp-level change); IO's
  `metres` and `with_provenance`; Estimation's `image_size` (RC's method for `NonCentralAperture`); the experiments' `texel_weights` by
  Core's `bilinear_weights`; the README test by Core's `julia_blocks`.
- **Fixed: a single-channel run counted every measurement three times** (found and fixed 2026-11-12, owner ruling). `io.channel`
  "R"/"G"/"B" (since 2026-11-04), the green-cell photographs and the photosite plane were copied into the three channels
  of the residual, so each measurement was three residuals: the minimiser was the same, the cost, the observation count and every
  statistic three times too large (a GCV's observation count among them). Now a state has a `channel` (0: the three; `c`: channel `c`
  alone, `measured_channels`) and ONE residual per measured pixel or photosite: `masked_vector(img, px; channel)`, the per-view and
  per-tile residuals' `nres`, the closed-form solves and the parts' joint solve. The kernels (one implementation, CPU and GPU) still
  evaluate the three channels of the flux; the residual takes the measured channel's rows and the Jacobian its columns
  (`_measured_jacobian`). The other channels' lighting coefficients and refractive indices η are no parameters of a single-channel
  fit (`LightingVar` J coordinates instead of 3J, `MaterialVar` 4 nonlinear instead of 8): they follow the measured channel
  (`tie_channels`), so no free parameter has a zero column. A colour run is bit for bit as before. The cost of a single-channel run
  is a third of its former value at the same parameters, and a fit reaches the same parameters (a GCV-selected weight may differ: its
  observation count is now right). `_take_channel` (the copy) is deleted.
- **Breaking: `detect_image_circle` is VisualGeometryToolkitFeatures'** (0.8.0, its fit through the Estimation extension): RC's
  copy, its export and its tests are deleted (no re-export: `using VisualGeometryToolkitFeatures`), and with them RC's ImageSmooth and
  DelaunayTriangulation dependencies. `run_image_circle` stays RC's, the pipeline's caller.
- **Breaking: the image circle is measured on the GREEN CELLS** (owner ruling 2026-11-12: it is geometry). `camera.image_circle_frames`
  may list RAW files, measured by Features on their green cells (`load_raw(f; mode = :green_cells)`), the circle returned in raw
  pixels; full-resolution decoded `.png` frames are read as before. The detector's literals
  `camera.image_circle_outlier_halfwidth_px` and `image_circle_min_hole_area_px` are in green-cell px: the configs of record restate
  the owner's 1000 px / 256 px² (chosen on full-resolution frames) as 500 / 64, and a full-resolution frame takes them back by the
  map's scale (× 2, × 4), so its measurement is unchanged.
- **Breaking (rc-dedup, 2026-11-12: one implementation per concept; the stack's names).**
  - The LM holds blocks with `fixed = (names…)` (`run_lm!`, `run_lm_grid!`, `lm_problem`, `lm_problem_grid`, `calibrate!`); `unfixed`
    is no keyword (a `MethodError`; its by-name refusal is deleted in the /simplify review). Its increment tolerance keyword is `xtol` (was `dstep`, Core's `nlls_options` name);
    `solve_linear_blocks!(…; sweeps)` names its sweep count.
  - The vignetting block is Core's `restrict(ParameterManifold(vig), Z)` (`LinearSubspaceManifold`); `VignettingVar` is refused by
    name and `n_vig_params` is `tangent_size`.
  - The stall postcondition is Core's block-form `is_stationary(blocks, variables, free)`; the two-row Jacobian is deleted.
  - One device resolution, `rc_backend(backend; inline, fallback)` (on Core's `default_device`), for bin/refine.jl and every driver.
  - experiments: one RAW path — every `[prepare.<set>]` decodes through Features' `load_raw`; `mode = "fullres"` (a demosaic) is
    refused by name, radiometric work reads the green cells (`"green"`); `convert_raw`, scripts/py/convert_raw.py and the pixi
    environment are deleted. The paper numbers are Eval's `PaperNumbers` (RC's `tex_macro`/`write_numbers`/`tex_number` deleted).
    One `residual_summary` record `(n, median_px, mean_px, rms_px, p95_px, max_px)`. The chrome registration keeps its points
    (`points/<stem>.json`), its pose (`poses/<stem>.json`) and its scorer's scale (`registration/<stem>_scale.json`) once
    (`_pose.json` deleted). Renamed config keys, refused by name: `spacing_px`, `[evaluate] spacing`, `diagonal_spacing_px`;
    `[evaluate] centre_rho`/`edge_rho` added; the panorama_vignetting configs' `outlier_halfwidth_px` (PanoramaStitching 0.6.0).
- **The refinement's stall postcondition is evaluated, never assumed** (the uniform convergence record). A step-size or rejection
  stop of the LM records `Success` only where VisualGeometryToolkitCore's `is_stationary` holds at the final state, at Estimation's
  `[chart_lm] stationarity_tol` (read through `parse_defaults` / `toml_default`: `STATIONARITY_TOL`), and `Stalled` otherwise. The
  check (`lm_stationary`, `lm_stationary_at`) costs one residual-and-Jacobian evaluation of the whole problem after the LM, accumulated
  block by block (the photographs' Jacobian is never held). `lm_termination(res, stationary)` takes the postcondition; `refine!`
  returns the record (its third value's `termination`), which `write_refinement` writes. The problem and free blocks of `run_lm!` /
  `run_lm_grid!` are `lm_problem` / `lm_problem_grid`, one construction for the LM and the check. No estimate moves; a run whose LM
  stopped on its step size away from a stationary point now reads `Stalled`.
- **Every termination is VisualGeometryToolkitCore's §14 record** (robust-core 5d61c18b: a `ReturnCode` and a `StopRule` in the
  FitResult core). **Breaking (records):** the refinement's parameters.json `retcode` is the name of its LM's `ReturnCode`
  (`"Success"`, `"MaxIters"`, `"MaxTime"`, `"Failure"`, …; `"Default"` without an LM iteration), beside a new `stop_rule`, both from
  Core's `nlls_termination` of NLLSsolver's termination bits (`lm_termination`; the private `lm_retcode(nlls_stop_code(…))` reading is
  gone). Its stall postcondition is taken as held, as before: the run keeps no Jacobian. `load_pipeline_metrics` reads both, and
  leaves out a record without `stop_rule` (before this change). The P-spline fits are checked by Core's `converged`, and the image
  circle's RANSAC by `found_model` (its former `converged`: a model was found). experiments: the chrome evaluation's records and
  summary.csv `retcode` are `ReturnCode` names (with `stop_rule` in eval_<label>.json); the held-out variant's two fits combine by
  Core's `worst_retcode`; the index RANSAC and the parametric starts' pose RANSACs test `found_model`; the paper tables' convergence
  flags (†) are Core's `successful_retcode` of the recorded `ReturnCode` (`recorded_retcode`), and a retcode Symbol of an earlier
  record is refused by name. No estimate, residual or iteration count moves.
- **Ported to the VisualGeometryToolkit 0.9 robust-estimation contract (its rename head, port-pin-1da1ef16)** (2026-11-11).
  **Breaking:** the stack's one parameter-holding keyword is `fixed`. `[lm] fix` is `[lm] fixed`, and `[edge] free` is deleted:
  the printed edge (δ, σ_cam,R/G/B) is free on a board with a distance map (`edge.sdf`) unless `"edge"` is in `lm.fixed`
  (`FIXABLE_BLOCKS`; `_edge_free`). Every config of record that had `free = false` now holds `"edge"`, so its run is unchanged
  but re-keyed. The old keys are refused by name (`RETIRED_KEYS`, which also holds the retired vignetting-centre keys).
  `lens_free(camera; fixed)` / `lens_manifold(camera; fixed)` and `build_problem` / `run_lm!` / `calibrate!(…; distortion_fixed)`
  name the parameters HELD, expanded by Core's `free_parameter_names` (groups such as `:k`, `:p` accepted). They replace `free` /
  `distortion_free`, which named the free set. experiments: `scale_multiple` is `efficiency_constant` (all Huber, so the values are
  unchanged). `[index.local_optimization]` is `[index.local_optimization_strategy]` and `[evaluate] local_optimization` is
  `[evaluate] optimization_strategy` (the fit record's key too). The readers are `optimization_strategy_from_toml`. The
  scalar MADs (stitching_log, bad_pose_flags) are StatsBase's `mad(x; normalize = true)`, since Core's `mad_scale` now takes
  residual vectors. The rotation-bundle test's grid lens is a SplineLens camera.
- **Ruling compliance (2026-11-11 audit).** **Breaking:** `[material] freeze` is deleted. `"material"` in `lm.fixed` holds Θ_m
  (8 nonlinear + μ), and the block stays variable with its columns zero (`FREEZE_MATERIAL`) when it carries a free board
  warping, printed edge or in-plane print error (`_material_rides`); otherwise it holds the block as before. The old key is refused
  by name. Every config of record had `freeze = false`, so they are unchanged. `detect_image_circle`'s `outlier_halfwidth` is
  a required keyword with no computed default (it was `0.1 · hypot(W, H)`). The pipeline reads it from
  `camera.image_circle_outlier_halfwidth_px`, which is required (> 0) with `camera.image_circle_frames`. Every config of record
  that runs the stage states the owner's 1000 px (ruling 2026-11-12: every set is single-basin for 750–2000 with
  min_hole_area = 256).
- **The refine contract (VisualGeometryToolkit 0.9 B1, robust-core 17ab6cc2).** A refine's default is now `Triggs{Welsch}`,
  which needs exactly one of `scale` or `width`. The bearing-pose refits, which were least squares under the former default
  (`_fit_page_pose`, the per-view refit, central and non-central), state `optimization_strategy = nothing`: plain least
  squares, as before. the experiments' texture-recovery pose refit and the parametric starts' `refine_camera_and_poses` do the
  same. The rotation-bundle test runs PanoramaStitching's `refine_rotations` with a strategy and a scale.
- **The image circle's rim drops small holes** (2026-11-12, owner ruling, variant (iii)). Step 4 of `detect_image_circle` keeps
  every dark region's outer boundary and its holes, except holes of |area| < `min_hole_area` px² (Core's `area`). A hot pixel
  punches a 1–5 px hole in the dark surround, and its corners pulled the hull off the rim. `min_hole_area` is a required keyword,
  and `camera.image_circle_min_hole_area_px` is required with `camera.image_circle_frames`. The configs of record state the owner's 256
  (ruling 2026-11-12: inner-fringe pockets go too; the rim's own hole is 3.5–5.6 Mpx; T ≥ 1024 starts losing true rim pockets).
- **The vignetting centre is the camera's principal point, fixed; the gauge V(𝐜) = 1 always held** (2026-11-11, owner rulings (a), (b)).
  `vignetting_center(camera)`: K's principal point for a parametric camera, the principal point of Estimation's `optical_axis` (the
  pixel of the optical axis, over the camera's bound) for a SplineLens camera; a non-central camera is refused. **Breaking:** the
  free centre of 2026-11-10 is deleted (`vignetting.center`, `center_px`, `center_bound_px` refused by name; `VignettingVar` has no
  centre parameters, the parts kernel no centre columns, `center_gradient` and `CalibrationState`'s `center_free` / `center_bound`
  gone), and so is `CalibrationState(…; pin_center = false)`. The projective chart is Core's `RadialVignetting(…; chart = :projective)`
  (RC's `ProjectiveRadialVignetting`, `RadialProfile`, `RADIAL_CHARTS`, `projective_knots`, `radial_chart`, `with_center` deleted; a
  record of the retired `radial_projective` body reads back as Core's profile). parameters.json's `vignetting_center` is
  `{chart, center, source}`. Results of every radial-vignetting run move (the centre was the sensor centre, or free).
- **experiments: one robust schema** (2026-11-11, owner rulings (c), (d)). The chrome evaluation's motion fit is `[evaluate] scoring` +
  `[evaluate] local_optimization` (`[evaluate] robust` retired): ONE Triggs stage from the index poses, no least squares first, its
  width scale_multiple × ŝ of the partition the Student-t scorer selects among the dots' pixel residuals at the start (Estimation's
  partition anchoring); `motion_scale_s2` deleted. `[index] scoring` and the parametric starts' `[fit] scoring` (was
  `ransac_threshold_px`) are scorer tables read by Estimation's `scoring_from_toml`. The GPU chrome detect resolves its device at the
  driver's top level (it hit a world-age MethodError in KA's `allocate`).
- **experiments: one command reproduces the paper** (2026-11-11, plan 3 Task 4e). `experiments/configs/reproduce.toml` is the stage graph of
  record (each Slurm launcher of `experiments/scripts/slurm/`, its driver and config of record, what it runs after); VisualGeometryToolkitEval's
  runner runs it on a site, `bash experiments/scripts/reproduce/reproduce_results_dgx.sh` (dgx01, Slurm: sbatch arrays chained by
  `--dependency=afterok`, setup instantiating the environment inside Slurm) or `reproduce_results_desktop.sh` (one RTX 4090, no Slurm,
  device 0), with `--dry-run`, `--only=…`, `--no-deps` and `status`; the inputs are checked first and the missing ones reported as data
  (path, tier, used by); each launch writes a run manifest and reuses the stages its run keys finished; it ends with
  `scripts/regenerate_paper.jl --gpu --results-branch` (nothing deployed into `paper/`; `numbers.jl --results-branch` copies
  `numbers.tex` beside the tables in `plots/latex/`) and the paper's results branch. **Retired:** `experiments/scripts/slurm/submit_all.sh`
  (its `SUBMIT_ONLY` is `--only`, its `SBATCH_SITE` and `GPU_LIMIT` a site profile's `cpus_per_task`, `mem_per_cpu` and `gpu_limit`).
- **experiments: the non-central spline is out of the configs of record** (2026-11-11; the owner removed it from the paper): the six
  `*_nc` cases of `refinement_chrome.toml` and their non-central inputs, the six `thorlabs_distortion_*_nc.toml` chrome evaluations,
  their `[chrome.*_nc]` and table rows in `geometric_figures.toml`; the catalog keeps the non-central calibrations as an
  `[[unpacked.rc_baselines]]` entry no config reads. The package's non-central model and the golden case `fisheye_nc_selfcheck` stay.

- **The pose's Jacobian columns in Float64 on every backend** (2026-11-11, owner: the heights' rule for the pose). The chunks that
  carry the pose's six slots (`_pose_chunk`: the first two chunks, slots 1–8 at the chunk width 4) run in `_pose_float` = Float64 on the device with the
  nodal heights' Float64 inputs and radiance record (`_tile_chunk`, `_view_chunk`); the other chunks keep Float32. On the
  paper-like Siemens case the device's pose columns were 0.37–0.67 % off the CPU's stacked and 1.2–1.6 % in a tile (Float32);
  Float64 chunk duals with the Float32 record (mixed) still 0.38 % stacked and 1.6 % in a tile after the LM; now ≤ 1.2e-12 stacked
  and 1.8e-12 in a tile. `test/pipeline/jacobian_parity.jl` gates the pose like the heights (0.4 %, stacked and per tile), the
  GPU item at 1e-9.
- **The nodal heights' ray hit is Core's** (2026-11-11, owner ruling: the hoist, no waiver). `HeightBoardRays` is deleted: the height
  mode's rays are `BoardRays(b, hd)` again, and Core's `intersection` of the dual relief now takes the height field's 1-D Newton hit
  and its partials by one implicit-function step (VisualGeometryToolkitCore hoists-09-intersect) — the derivative RC's IFT gave, to
  rounding, at the same cost. Needs that Core: on an older Core the height kernel is correct but takes the generic 2 × 2 root (≈ 6×
  slower).
- **The nodal heights' Jacobian columns in Float64 on every backend** (2026-11-11, owner: correctness first). The height kernel and a
  radiance record of its own run in `_height_float` = Float64 on the device too (the view, the scene, the relief, the estimator); the
  Jacobian chunks keep Float32 and their record. On the paper-like Siemens case the device's height columns were 0.2–1.3 % off the
  CPU's (Float32 board points 0.1 µm off on the print's 0.03 mm edge; Float64 rays with a Float32 estimator still 0.44 % in a tile);
  now ≤ 2.2e-12. The height mode meets the PLAIN relief by Core's height-field Newton and takes the ray parameter's height partials by
  the implicit function theorem at that hit (then RC's own step, now Core's: see above), not by Core's root of a
  dual chart map — the height kernel 220 → 37 ms (Float32) on a 12 048-px view, 83 ms in Float64. `test/pipeline/jacobian_parity.jl`
  gates the height columns at 0.4 %, stacked and per tile.
- **One Jacobian implementation on every backend** (2026-11-11, owner policy: a GPU branch's CPU branch is the same
  KernelAbstractions code on the CPU backend). The per-view and the per-tile `computeresjac`, `computeresidual`, `render_view` /
  `render_all` / `residuals` and the printed-edge finite differences are the pixel functors of gpu.jl (`_FluxAt`, `_RecordAt`,
  `_JacAt`, `_HeightAt`, `_PartsAt`) run by Core's `pointwise!` on the state's backend: Float32 on a device as before (its
  residuals and Jacobians bitwise unchanged), Float64 on the CPU backend (`_kernel_float`; one chunk of all the dual slots,
  `_jac_chunk`). The CPU Jacobian is the device's chain rule in Float64 — the radiance estimator's local Jacobian once per ray, the
  nodal heights through the ray parameter — and agrees with the deleted threaded-dual one to 1e-12 (the paper-like Siemens case);
  its residual is now bitwise `computeresidual`'s. `pixel_parts` and the parts kernel share one aperture quadrature
  (`_sample_parts`). Two backend differences are gone with the twin: the device now zeroes a frozen focus node's column
  (`FOCUS_FREE`, focus-ab) and takes a direction-chart focus's weights through the lens (`focus_weights`; it took the sensor TPS's —
  the same for the sensor chart of the paper's runs); and its dual focus keeps the focus's chart (`Focus{D, N}(…, kind)`: it was
  evaluated as a sensor TPS, so a direction-chart or sphere focus had columns 1.5–4× off on the device). **Breaking (internal):** `_computeresjac_device(_tile)`,
  `_computeresidual_device(_tile)`, `_fd_columns_cpu!`, `_analytic_columns!`, `_sample_partials`, `_tile_view`, the sensor-only
  `_nodal_weights_at` deleted; a grid problem is registered for the view batches on every backend.
- **The board warping is Core's posed height field** (2026-11-11, plan 3 Task 4b). The board's warping is Core 0.9's `HeightField` over
  the page in units of its width on the placed chart (`warping_nodes(aspect)`: `(u, −aspect·v)`), placed by a `SimilarityMap{3}`
  (`warping_placement`) and posed by the board: the render's surface is Core's `MappedSurface(pose, field, placement, page)`
  (`PrintedTarget.warping` holds the flat one, `board_relief` rebuilds its field from the nodal heights by Core's `with_parameters`), the
  same parts `posed_target` gives Core's posed target; plain rays take Core's 1-D height-field Newton, duals its implicit 2 × 2 root.
  Board features on the warped page (the Schöps front end, the ground truths) are Core's `points` of that posed target
  (`page_points`); `warping.points` fits the nodal heights by Core's scattered `HeightField(nodes, points .=> heights)`. **Breaking:**
  `BoardWarping`, `board_height_grad`, `height_weights`, `page_widths`, `ReliefMap` deleted; `board_height(b, h, 𝐛)`,
  `height_basis(b)`, `height_manifold(b)`, `placed_chart`/`design_chart`/`physical_chart(b, …)` take the board; `board_surface(aspect, t)`;
  `warping_aspect(b)`. Hits and heights move at rounding (≤ 1.3e-15); the golden outputs are unchanged.
- **One penalised least squares: the stack's** (2026-11-11, plan 3 Tasks 4g and 4d). The GCV curve and choice are Estimation's
  `gcv_curve`/`gcv_select`, the penalty weight Core's `penalty_scale` and `mean_positive_eigenvalue`, the grid prior's rows Core's
  `row_stencils` of `difference_rows` (or of the lattice's `bending_stencils`), the focus prior Core's `bending_rows`, a weight check
  Estimation's `check_penalty_weight`, the grid start Estimation's `pspline_grid` at its call. **Breaking:** RC's `_gcv_curve`,
  `gcv_select`, `_brent_minimize`, `GCV_LOG10_BETA_BRACKET`/`GCV_SCAN_POINTS`/`GCV_LOG10_TOL`, `_check_penalty_weight`,
  `_mean_positive_eigenvalue`, `term_rows`, `_pspline_fit` and `bending_rows(::FocusSpline)` deleted; a GCV selection's fields are
  `β`, `λ`, `log10_β_bracket` (were `beta`, `lambda`, `log10_beta_bracket`). **The grid prior `:tikhonov_to_start` is deleted** (no penalty
  holds a gauge; `grid.prior = "tikhonov_to_start"` is refused) and **the default prior is `:pspline`**; `solve_Θ_l!`'s `ridge_rel` is
  deleted. The experiments configs of record that stated `tikhonov_to_start` state `pspline`: their run keys change.
- **The radial vignetting profile's centre can be a block of the joint LM, and the profile has a chart** (2026-11-10).
  `[vignetting] center = "free"` adds the centre's two parameters to the vignetting block
  (`VignettingVar{TV, NF, NC}`, `NC = 2`): a tanh box of `center_bound_px` (default 100) about `center_px` (default the sensor centre),
  its Jacobian columns the dipole `Σ_s w_s ∂V/∂𝐯(𝐩_s) I_{s,c}` (`center_gradient`, a two-partial Dual through `with_center`, in
  the parts kernel and on the host); the hard pin V(𝐯) = 1 and the pole tie move with the centre (V at the centre is the first control
  point on every chart), the support rule is the start's. `[vignetting] chart = "projective"` puts the profile on the chart
  ρ = d / (r_max + d) (`ProjectiveRadialVignetting`, RC's decorator of Core's `RadialVignetting`, its chart Core's 1-D `HomographyMap` dehomogenized
  by `PerspectiveMap` (`projective_chart`), its own record body `radial_projective`) with pixel-uniform knots up to r_max and one tail span beyond, so no pixel is clamped for any centre; on the
  linear chart a free centre grows r_max by the bound instead. parameters.json gains `vignetting_center` (chart, box, start, refined);
  a resumed state's moved centre is the run's when the centre is free. **The free centre is the default** (`center = "free"`): a
  radial run that does not state `center = "fixed"` optimises the centre and, on the linear chart, lays its knots over r_max + the
  bound; its run key changes. The two golden pipeline cases (and their configs of record in experiments) state `center = "fixed"` and
  stay bit for bit the same; a tensor surface ignores `center`. **The radial profile is the default vignetting model** (`model = "radial"`,
  16 coefficients; was `tensor_product`): the model of record, which wins the panorama seams on every lens. Every config of record states
  its model, so none changes. The chart's default stays `linear`.


- **Breaking: on the VisualGeometryToolkit 0.9 stack** (2026-11-10; Core 0.9, IO 0.9, Estimation and Features at the 0.9 tree's
  versions, through TEMPORARY `[sources]` until the 0.9 release is registered, with RANSACScoringDoneRight's Core-0.9 port; compat
  moves to the released versions at the cascade). The grid lens is Core's SplineLens camera: `grid_model(src, G_x, G_y; coverage)`
  returns a `BoundedCamera` of `CameraModel(K, SplineLens)`, K the initializing camera's own (Estimation's `pspline_grid` on the
  given lattice), and the non-central lens's geometry a bounded `NoncentralSplineModel`. The image circle is the bounded camera's
  `lens_coverage` (`domain` is gone); every consumer reads Core's `imaged_region(camera)`, and `in_domain` of a bounded camera is
  its lens coverage and a finite back-projection. `schoeps_gridlens`/`noncentral_gridlens` take no nominal pinhole (the Schöps
  model's gauge is the K). Grid files are `vgio/cameras/v5` records (`save_grid`); `load_grid`/`load_lens` read an old
  `vgio/generic_camera` file through IO's read migration, never rewriting it. The warped board's relief is Core's `MappedSurface`
  of its chart map (`HeightSurface`, `ImplicitRoot` and the scene's derivative choice are gone: Core's root is differentiated by
  the implicit function theorem), and a posed target's placement Core's `SimilarityMap{3}`. Deleted: the `CentralGrid`,
  `NoncentralGrid` and `_CoreGrid` aliases, `_nominal`, `_focal_px` (`calibration_matrix(cam)[1, 1]`), the grid's own `in_domain`,
  `_fit_source`/`grid_fit_region`, `_image_rect` and `_nominal_pinhole`.

- **The in-plane print error of the board is a block of the refinement** (2026-11-10): Core's gauge-free factors of an affine print
  error under a jointly estimated pose, `Aspect ∘ Skew` — the log-stretch `a` of the page's x against its y (det 1) and the shear `u`
  — in the chart `posed_target` places the height field in (the page in units of its width, origin top-left, y up), so the posed
  target's record is `HeightField ∘ Aspect ∘ Skew` in Core's canonical order and the render samples the texture (and the edge map) at
  the DESIGN point a physical surface point shows, `inv(Aspect ∘ Skew)` of its chart point (`design_chart`; the height field stays a
  function of the physical point). `BoardSurface` carries the two parameters (`inplane`, `with_inplane`, `inplane_on`,
  `inplane_calibration`, `physical_chart`); they ride in the material block as dual-number columns (`MaterialVar(…; inplane)`,
  `build_problem(…; inplane_free)`, `build_problem_grid` / `run_lm_grid!(…; inplane)`), additive and unbounded. Options: `[warping]
  inplane = "aspect_skew"` frees them (`"none"`, the default), `inplane_start = [a, u]` starts them, `"inplane"` in `lm.fix` holds
  them; the pipeline places the board features with `physical_chart` and reports the error per iteration and in parameters.json
  (`inplane_print_error`). `RadiometricTarget` holds it (`inplane`; its record `inplane_print_error` is IO's target-calibration
  record of the composition, absent when the design is as printed; an error without the page's aspect is refused as a warping is).
- experiments, the paper run of 2026-11-07 (every experiment again on the merged stack): every set on ALL its registered photographs
  (the every-N subsets of 05–07.11 replaced: the front ends re-run on every photograph, views ≥ 20 stars / ≥ 16 corners; the start
  fits of the 14 mm AprilGrid400 on its views with ≥ 60 corners, as on 05.11), the Double Sphere of the 14 mm sets, the Tamron f/8
  starts evaluated on the chrome board, the geometric-accuracy sets of the author's cleaned layout (the 14 mm central_30cm, the
  fisheye at 10 cm, the Tamron at 17 cm), `scripts/texture_recovery.jl` (the 14 mm f/2.8 texture inverse rendering: pinhole vs the
  learned aperture and focus surface on the photographs the refinement never used), and PanoramaStitching 0.4's key names in the
  panorama_vignetting configs.
- **The radial profile's dependent control points follow Core's `BoundaryRule`** (2026-11-07), the tensor surface's rule, on the
  profile's N × 1 lattice: a control point below `vignetting.support_min` of the mean data weight continues the supported ones inward
  with a zero `vignetting.boundary_order`-th difference — 2 the natural Δ²c = 0 (`P_k = 2 P_{k−1} − P_{k−2}`, the last fitted slope
  carried past the data), 3 the curvature-preserving Δ³c = 0. From 2026-10-30 it replicated the last supported one, so V was constant
  past the data: the f/4 diagonals of the paper's vignetting figure showed it as plateaus inside the frame (the 14 mm's last two of 16
  points, the Tamron's last three). `RadialSupportRule` holds the rule; `vignetting_basis` is its `boundary_rule_jacobian` with the
  pole tie, `tie_pole` applies it to a resumed state's values; the pole pair P_0 = P_1 is always a parameter and counted as supported
  (the Δ³ smoothness windows include it). The refine log names the order. Every refinement of a radial profile with a dependent point
  changes; the experiments' f/4 vignetting cases name `boundary_order = 2`, so their runs are keyed apart from the 2026-11-05 ones.
- **`focus_ab.json` counts the masked pixels the lens gives no ray** (2026-11-08) instead of failing: a masked pixel whose
  back-projection is not finite (a parametric lens refined past its invertible range at a corner) is in no node's count and not in
  `theta_max_masked_px`, and the refine log names how many. Before, the NaN reached the record writer, which refused it after the
  last iteration. The refinement is unchanged.
- the experiments' chrome evaluation takes `[evaluate] max_unassigned` (optional, default 0; Ties Harder, 2026-11-06): a photograph enters
  when at most this many dots of the grid are unassigned, where before every dot had to be (`_all_dots(r, g, max_unassigned)`). It
  must stay below the grid's shorter side (`check_max_unassigned`), since a labelling shifted by a whole spacing leaves a whole row
  or column unassigned, so the labelling stays pinned. For the fisheye at ≈ 10 cm, whose central camera of record leaves a few rim
  dots > 8 px off in most photographs (9 of 143 complete; 84 with `max_unassigned = 8`). Absent, the evaluation is unchanged.

## 0.7.0 (2026-11-07)

**Breaking: on the VisualGeometryToolkit 0.7 stack**, from General: Core 0.7, IO 0.7, Features 0.6 and Estimation 0.7; the
paper project also Eval 0.6 and PanoramaStitching 0.5 (CauRegistry). NLLSsolver's compat floor is 5.3.2 (its precompile fix).

- **A generic model's back-projection is NaN where it images nothing** (Core 0.7), never `nothing`: `load_model`'s `schoeps:` and
  `noncentral:` specs and the experiments' `load_ray_model` test `all(isfinite, d)` (for a non-central model, of the ray's direction); they
  still return `nothing` there.
- **Breaking: the vignetting field is a tag of Core's `VIGNETTING_TAGS`.** `vignetting.model` is `"tensor_product"` (was `"tensor"`,
  Core's retired tag, which the reader now refuses naming its successor) or `"radial"`; the option holds Core's type itself
  (`TensorProductVignetting`, `RadialVignetting`). `TensorVignettingModel`, `RadialVignettingModel`, `AbstractVignettingModel` and the
  package's `VIGNETTING_TAGS` are removed; `VIGNETTING_CONFIG_TAGS` is Core's registry with the configs' key path (`vignetting.model`),
  for `config_tag_violations`. docs/pipeline_config.toml and the experiments' panorama configs (`compare_models`) and fit2d stage carry the
  current tag, so those configs of record get new run keys.
- **Breaking: the generic models through Core's set accessors and accessors.** `with_region` → Core's `with_domain`, `central_part` →
  Core's `central_model`, the package's `with_present` → Core's (all three with a `NonCentralAperture` method, which carries the aperture
  spline); they are no longer exported. The private rebuilds (`_with_nominal`, `_with_controls`, `_with_directions`, `_absent_nan`,
  `_dirs`, `_full_size`, `_first_node`, `_lattice_spacing_px` and the field-by-field rebuilds of a non-central model) are Core's
  `with_nominal`, `with_directions`, `with_points`, `directions`, `grid_size`, `window_offset` and `lattice_spacing`. `load_grid` reads
  IO's `load` of the generic-model union instead of matching the error text of a typed load (the experiments' `load_ray_model` likewise).
- The P-spline rows are Core's `difference_rows(mask, 3)` (`third_difference_rows` and camera/bspline.jl are removed; the grid prior
  keeps them as term lists, `term_rows`). Core's stencil is the negation of the former rows: the penalty `DᵀD`, the vignetting solve and
  the grid prior's LM are bit for bit unchanged (every row and its target flip sign together).
- The pipeline options' reader is Core's: `read_table` (its `toml_key` walk, the writer's) and the keyed `convert_field` (a table
  refused, a tuple's length checked, a string refused for an array); `_from_toml_value` is removed. Same messages.
- The focus surface evaluates and differentiates through Core's spline systems: `spherical_spline_evaluate`, `nodal_weights` of the
  chart's and the sphere's systems, `tps_coefficients` / `spherical_spline_coefficients`. The sphere's nodal weights read Core's
  normalized node directions (≤ 1 ulp from the chart's `ae_dir`). The padded Float32 coefficient map stays for the device evaluations.
- The camera's in-focus blur in a record is IO's named-blur body (`JSON.lower` / `JSON.lift` of a `Dict{Symbol, GaussianBlur}`):
  the blocks are written in name order (B, G, R; read by name, so earlier records read unchanged). The records' pixel frame is IO's
  `pixel_frame_name` of COLMAP's frame.
- The sensor rule's σ_eff is the in-focus blur convolved with the pixel box (Core's `convolve`, an exact 1/12 px² box: bit for bit
  √(σ² + 1/12)); the device's in-focus blur is Core's `convert_eltype` (its Float32 moment `Float32(σ²)`, was `Float32(σ)²`).
- Element-type conversion is Core's `convert_eltype` for any `T <: Real` (a pixel camera, and a pose with `unlabel`); `_lens_eltype`
  is removed. `_real_type` is Core's `primal_float`. `_nominal_pinhole` is Core's `LogicalIntrinsics(f, pitch, pp)` of the quantities
  (the same K). The tile Jacobian buffer is Core's `task_buffers` (was a locked global pool). `load_mask` reads the package's `luma`.
- **Breaking:** `refine_view_pose` is removed: the front-ends call Estimation's `refine_absolute_pose` on `EuclideanMap`s, with the
  bearing correspondences of `_bearing_csponds`. `lens_type_name` is Core's `lens_label` (`"rational"` → `"division"`; an OpenCV
  14-coefficient lens is labelled by its record's leaf tags), which reaches the run labels and the lens records' provenance.
- the experiments' `GridPoseProblem` answers the RANSAC protocol's `mahalanobis_distances!` (was `residuals!`; its values were already
  distances ≥ 0).

## 0.6.0 (2026-11-06)

**Breaking: on the VisualGeometryToolkit 0.6 stack**, from General: VisualGeometryToolkitCore 0.6, VisualGeometryToolkitIO 0.6,
VisualGeometryToolkitFeatures 0.5 and VisualGeometryToolkitEstimation 0.6.1 (compat floors; the paper project also
VisualGeometryToolkitEval 0.5 and PanoramaStitching 0.4). No stack package is sourced from a local checkout. Folds in the
unreleased 0.5.1 below.

- **Model correction: the board warping is the page's physical bending, on a page of any aspect.** Its thin-plate spline was over
  the page fraction 𝐛 ∈ [0, 1]², x and y scaled apart — for a non-square page a bending that depended on the page's aspect (a
  thin-plate spline is invariant only under a similarity of its domain), which is why the render path refused the warping on such a
  page. It is now over the page in units of its width, `𝐜 = (b₁, α b₂)`, α = height / width (`BoardWarping(α)`: its nodes and its
  weights map; a `PrintedTarget` holds its page's, `b.warping`, adapted to a device like the texture). The 25 nodal heights stay
  values at the same page points (records unchanged); a square page's warping is the previous one bit for bit (every test board and
  the pipeline pages are square). **Breaking:** `board_height(h, 𝐛)` / `board_height_grad(h, 𝐛)` → `board_height(b.warping, h, 𝐛)` /
  `board_height_grad(wp, h, 𝐛)`; `square_page` and the square-page refusals (`Scene`, `posed_target`) are removed; `HEIGHT_CW`,
  `HEIGHT_CW32`, `HEIGHT_NODES32` are gone (the warping's `cw`; `HEIGHT_NODES` are the nodes as page fractions, `HEIGHT_TPS` the
  square page's spline). The free-height basis is each page's: `height_basis(wp)`, Core's `retract_jacobian` of
  `height_manifold(wp)` (the `TargetManifold` of a `HeightField` over THAT warping's nodes) — the null space of the warping's own
  affine rows, which depend on the aspect; the material block (`MaterialVar(mat, surface, blur, warping; height)`) carries its
  board's. `HEIGHT_BASIS` and `HEIGHT_MANIFOLD` (the square page's, whose columns carry an affine part on any other page) are
  removed; a square page's basis is theirs bit for bit.
- **Breaking: a target's warping knows its page's aspect.** The same nodal heights are another surface on a page of another
  aspect. `RadiometricTarget(target, warping, aspect, radiometry)`; the `rc/radiometric_target/v1` body (and a state's target)
  holds `"warping_aspect"`, required unless every nodal height is zero (a flat warping fits any page). `board_surface(t)` →
  `board_surface(board, t)` / `board_surface(wp, t)`, an `ArgumentError` on a board of another aspect; `set_radiometric_target!`
  and `load_state_into!` (before any pose moves) and the ground truth's state and render go through it. `parameters.json`'s
  `board_warping` states its `aspect` and describes the domain, `[0, 1] × [0, H/W]` in page widths.
- `set_camera!` is the only writer of a `CalibrationState`'s camera (`s.camera = c` is an `ArgumentError`), so every route that
  changes the camera's blur — the LM's read-back, a rejected sweep's or joint solve's restore, `set_vignetting!`,
  `set_intrinsics!` — rebuilds the sensor rule with it.
- A non-finite LM step of the material block (a NaN σ_cam, which Core's `GaussianBlur` refuses) keeps the block's blur and marks it
  invalid (`is_valid(m)`), which every residual of the block checks, CPU and device: NaN residuals, the step rejected, the state
  finite. The NaN no longer rides in the ink spread δ, and the device paths' per-field finite checks of the block are that flag.

- **Breaking: the light is VisualGeometryToolkitCore's `EnvironmentLight`** (a `SphericalHarmonicMap` on a sphere about the board
  centre), hoisted from this package's `Lighting` bit for bit; `Lighting`, `y_basis`, `y_lm`, `K_lm`, `P_lm`, `n_sh`, `sh_index`,
  `ω̂`, `L_SH`, `L_SVSH`, `y_SVSH` and the package's `with_coefficients` are removed. Migration: `Lighting(𝔟, r, l, a)` →
  `EnvironmentLight(SphericalHarmonicMap(a, l), 𝔟, r)`, `Lighting(𝔟, r, l; radiance)` → `EnvironmentLight(SphericalHarmonicMap(radiance,
  l), 𝔟, r)`, `light.a` → `coefficients(light)`, `n_sh(l_max(light))` → `sh_count(max_degree(light))`, `L_SVSH` → `radiance`,
  `y_SVSH` → `coefficient_jacobian`, `ω̂` → `exit_direction` (all Core's). `sphere_radius(𝔟, centres)` stays the package's.
- **Breaking: the radiometric camera is one value, `RadiometricCameraModel`** — the geometric camera (lens), the focus surface, the
  aperture radius, the vignetting and the camera's blur at perfect focus (`in_focus_blur`: one Core `GaussianBlur` per channel, px;
  `InFocusBlur`, `in_focus_blur(σ)`, `blur_sigmas`, `NO_BLUR`). `CalibrationState` holds it and one pose per view (`s.camera`,
  `s.poses`) instead of `s.views` (each with its copy of the lens, focus and r) and `s.vig`: `view_of(s, k)`, `views_of(s)`,
  `set_vignetting!`, `with_vignetting`, `with_intrinsics`, `with_in_focus_blur`. `CalibrationState(scene, settings, views, vig, …)` still
  builds it from views that share one camera (an `ArgumentError` otherwise). A `View` is one photograph as the renderer evaluates it:
  `View(pose, camera)` or `View(pose, lens, focus, r[, blur])`.
- **Breaking: no image distance.** `View`'s `image_distance` (s′) is deleted: nothing in the model read it (the cos⁴/s′² obliquity is in
  V; the defocus is the focus surface along each chief ray, which holds for a non-central lens too); `sensor_distance` is removed.
  `parameters.json` no longer has `image_distance_mm` (nothing reads it). `load_session(…; image_distance)` still places the thin-lens
  focus start.
- **Breaking: one source for the camera's blur.** The printed edge's σ_cam,R/G/B (`BoardSurface.edge[3:5]`) and the sensor rule's
  point spread (`RenderSettings(; psf_σ)`) are the camera's `in_focus_blur`: a distance-map board folds it into its edge profile (the
  view's `blur`, `edge_albedo`), a texture board's sensor rule integrates it (`RenderSettings(; in_focus_blur)`, one σ for every
  channel); a state checks that its settings' rule is its camera's, and the rule is derived from the camera wherever a state's camera
  is set or read (`set_camera!`, `with_sensor_rule`: a loaded or resumed state renders its own camera's blur), so the state record holds
  the blur once; a per-channel blur under a rule that integrates one is refused where the settings or the record are built. A NaN step
  of σ_cam is rejected by the LM rather than aborting it. `BoardSurface.edge` is the print's `(δ, σ_print)` (`EDGE_NC = 2`);
  the material block still carries σ_cam (finite-difference columns in the same order) and returns it to the camera. The config keys
  (`rendering.psf_sigma`, `edge.camera_sigma_px`) and `parameters.json` are unchanged. Bit for bit: the goldens move by 0.
- **`TargetRadiometry`**: the target's radiometry as one record — the material Θ_m and the print's edge (the ink spread δ and the print
  blur as quantities, a Core `GaussianBlur` on the print; plain values are an `ArgumentError`); `target_radiometry(s)`, `material`,
  `edge_parameters`.
- **`RadiometricTarget`**: the calibration target as a refinement knows it — the geometric target (Core's, as the session's target
  record holds it), the board warping h and its `TargetRadiometry`; `radiometric_target(s)`, `set_radiometric_target!` (replaces
  `set_target_radiometry!`), `board_surface`.
- **Breaking: every file is read and written through VisualGeometryToolkitIO.** Serialization and JSON3 are no longer dependencies.
  RadiometricCamera owns three kinds, each nesting IO's own records of Core's values (`write_record` / `read_record` on IO's
  `VgioStyle`): `rc/radiometric_camera/v1` (`save_radiometric_camera`, `load_radiometric_camera`), `rc/radiometric_target/v1`
  (`save_radiometric_target`, `load_radiometric_target`) and `rc/calibration_state/v1` (`save_state`, `load_state`). The state files
  are `state.json`, `state_it<N>.json` and the `io.save_init` start state (formerly `.jls`); a refinement also writes
  `radiometric_camera.json` and `radiometric_target.json`. `load_state` returns `(; camera, target, poses, names, light, settings,
  page_flip)`; the `.jls` record's unread fields (`poses0`, `trace`, `px`, `mask_frac`, `focus_info`) are not kept, its labels
  (iteration, costs) are the record's `provenance`. `state_record` is removed. The reports (`parameters.json`, `selfcheck*.json`,
  `focus_ab.json`) are written by IO's `write_record`. A `.jls` state of 0.5 is converted once (a converter run in an RC 0.5.0
  environment; not part of the package).
- Core's `TensorVignetting` is `TensorProductVignetting` at every call site. The pipeline's option type `TensorVignettingModel` and
  its config tag `tensor` are kept (the configs of record name the tag).

## 0.5.1 (unreleased, folded into 0.6.0)

- The Schöps front end's page frame is one labelling, rows along +y (the world "printed page, origin top-left, x right, y down"),
  with the camera on the printed side (cheirality: Core's `signed_distance` to the page plane, `_fit_page_pose`). The ±y search
  is gone: rows along −y with the camera behind is the same scene rotated 180° about x, the two scored alike, and rounding broke
  the tie (Julia 1.12 and 1.13 chose differently, so the siemens goldens could not hold on both). A mirrored texture — the print
  seen from behind — has no pose on the printed side and is an error naming it, not a silent fit. A state written before this
  release with `page_flip = −1` is still rendered and resumed in its own frame.
- Core compat 0.5.7 (`signed_distance`).

The pipeline's inputs and light sphere (Ties Harder, PR #3, 2026-11-04; the numbers change only where a camera stands closer than
the page's half-diagonal):

- **`io.channel`: a refinement on one colour channel.** `"R"`, `"G"` or `"B"` replaces all three channels of every photograph by
  that one before anything reads it — the edge selection, the masks' observations, the closed-form start and the LM all see the
  three identical copies — so the per-channel lighting and η come out equal and the fit is to that channel alone; `"RGB"` (the
  default) is as photographed. The image circle is still measured on the frames' Rec. 709 luma. For the single-channel
  refinements (the G-channel runs of the 2026-11-02 and 2026-11-03 sets; not yet configs of record).
- **The light sphere holds the page's corners.** The pipeline's start sphere about the board centre 𝔟 has the radius
  `sphere_radius(board, centres)` = max(farthest camera, farthest page corner), the three logged; it was the farthest camera alone, so
  a board whose cameras stand closer than its half-diagonal (the 27.10 set: 0.61 m against 0.71 m) had corners outside the sphere,
  where a light ray from the page misses it and `ω̂` falls back to the ray's closest point (finite partials since the fix below, but
  not the sphere's exit point). Runs whose cameras all stand farther than the half-diagonal are unchanged bit for bit; a resumed
  state keeps its own sphere (`load_state_into!`). `sphere_radius(𝔟, centres)` keeps the camera rule for the callers that give a
  centre (the experiments' ground truths of record, the tests' scenes).

The paper project has three experiments (Ties Harder, PR #3, 2026-11-05): geometric accuracy, vignetting and the macro defocus.

- **The retired material is gone:** the archive, the ablations, the night stars, the 8 mm rooftop vignetting, the HX400V chrome
  prototype (kept as a test fixture of the driver), the five figure configs the paper no longer read and ten unread figure files
  of `paper/src/fig`.
- **Geometric accuracy:** `configs/experiments/refinement_chrome.toml` holds the refinements of record (central_30cm 14 mm on the
  Siemens board and on the AprilGrid, five models each; the 27.10 Tamron AprilGrid and fisheye Siemens rows), scored on the chrome
  board by one `thorlabs_distortion_fx3_<set>[_<model>].toml` per model pair; the synthetic detector bias (`detector_bias.toml`) renders
  its truth from a Double Sphere lens and the radiometry of a real refinement's state at that refinement's own cameras
  (`src/ground_truth.jl`: a `radiometry` input may be a state, taking its light as it is, SVBSDF, vignetting and board surface, the
  true stars on its warping; the board kind `siemens_posed`, the Siemens board at the poses of a report — the real views and
  perturbed copies — as the AprilGrid already was; designed poses rendered the real light's ink too bright for Schöps's detector;
  both sets behind the real fisheye's image circle, where cv2.fisheye's and Kalibr's starts need the corners; `apply_field_stop`
  now stops the sRGB photographs and their records too, so a stop on existing renders equals rendering with it); `scripts/geometric_figures.jl` draws the
  colour wheels and the non-central figure and writes the table's macros.
- **Vignetting:** `panorama_stitching.jl` reads its learned maps from the cases of a refinement config (`[upstream] refinement`,
  the f/4 runs of `refinement_chrome_vignetting.toml`) instead of the night stars, and its device from its upstream chain (no `ENV`);
  one config per lens on the panorama_good sweeps; `scripts/vignetting_figures.jl` draws the diagonal / ln-quotient figure in the
  submitted layout (a column per lens: the diagonal inside the image circle, ln(V̂ / V) against the rotation estimate with its error
  band) and the 3 × 3 panoramas (`src/figures.jl` `figure_vignetting_lenses`, `figure_panorama_grid`). The panoramas' PNGs are written
  as PanoramaStitching's example chain writes its own renders — one exposure for every candidate (1 / the 99.5th percentile of the G
  of the corrected panorama, `[panoramas] exposure_percentile`, `exposure_candidate`), daylight white balance, the example's
  `save_srgb` — instead of `tone_map`'s own white per panorama, which hid the corrections' brightening.
- **Regeneration from the two entry points** (`scripts/slurm/submit_all.sh`, then `scripts/regenerate_paper.jl --gpu`): `configs/figures/numbers.toml` lists the per-lens `panorama_stitching` configs (their
  macros off: an empty `numbers_prefix`), so `scripts/regenerate_paper.jl` runs them; `scripts/geometric_figures.jl` has an `evaluate`
  stage that runs every chrome pair's `scripts/thorlabs_distortion.jl` stages; `scripts/slurm/submit_all.sh` submits the f/4
  refinements too (`refinement_chrome_vignetting.sbatch`) and the detector bias's every case. The detector bias's derived inputs
  (the real poses, the η_ink state, the field stop's frames) are written by the AUTHOR-SIDE `scripts/make_detector_bias_inputs.jl`
  (`configs/inputs/detector_bias.toml`); `data/DATA_LOCATIONS.md` says where the inputs not packed yet live.
- **Review of the above (2026-11-05):** the detector bias renders with the ink's η matched to the real photographs (η_ink 2.0) behind
  the real f/22 image circle (r 1309.6 px, measured on the 27.10 photographs); the night-sky evaluation left behind by the night
  stars is gone (`src/stars.jl`, `Evaluation.evaluate_star_models`, their fixtures); the vignetting figure's curve styles and
  limits and the table's percent digits are keys of their configs; the inherited-value marks are back in the configs written from
  main's; the launchers' headers state today's cases, `panorama_vignetting.sbatch` runs at the other launchers' thread count (28:
  the count keys the run `panorama_stitching.jl` looks for; it was 32), `submit_all.sh` takes two site options (`SBATCH_SITE`,
  `GPU_LIMIT`) and `regenerate_paper.sbatch` runs the regeneration where every run goes through Slurm; `regenerate_paper.jl`
  names its stages by config, so one lens can be selected; `--force` reaches the chrome scoring; the `[[unpacked]]` table's
  explanation is main's again and its entries carry no machine path; the tests narrowed for the new experiment set are restored
  (a cosmetic `[render]` edit never re-keys a run, the `--plan` of the two figure drivers, the JPEG driver run, the per-lens
  configs in the path and tier checks). REPRODUCE.md maps every item the paper shows to its driver, config and upstream runs.
- **Fixes:** `require_stage_inputs` returns a `Vector{String}` for a stage without inputs of its own (an empty comprehension's
  abstract eltype made `write_run_record` refuse every detector-bias init); the detector bias's evaluation takes a case's start from
  its init run's grid when the case names no `io.init` (Schöps's own start).
- **Merged onto 0.5.0's stack (PanoramaStitching 0.3, Core's vignetting):** the fisheye and Tamron `panorama_vignetting` configs
  state `gain_correction = "channel"` and the 0.3 weights keys (they held 0.2's `frame_gains = true`, which 0.3's example does not
  read), as the 14 mm config already did (its Sampson key is 0.3's `sampson_iterations`, it was `sampson_solves`); the panoramas
  stage keeps the frames' gains of PanoramaStitching's `fit_vignetting` with V held fixed (`[panoramas] gains = "fit"`), so the
  RC copy `frame_gains_fixed_v` stays deleted and the three `panorama_stitching_fx3_<lens>.toml` drop `gain_directions` and
  `min_pair_directions`, which only it read (all three set `gains = "none"`: no run changes); the ground truths' flat vignetting is
  Core's `TensorVignetting`.

## 0.5.0 (2026-11-05)

The vignetting model is VisualGeometryToolkitCore's (0.5.6): RC's copy is deleted. Every result is bit for bit 0.4.0's.

- **Breaking**: `Vignetting`, `RadialVignetting`, `AbstractVignetting`, `V`, `basis_row`, `with_control_points`,
  `n_control_points`, `control_points` and `sensor_center` are no longer RC's. Core's `TensorVignetting(n, box; degree)` (the
  sensor box a `Rect`) and `RadialVignetting(n, center, radius; degree)` are the models, callable `v(u)`, with
  `coefficient_jacobian` (the former `basis_row`), `center` (the former `sensor_center`), `field_radius`, `pole_regular_basis`
  (the radial profile's P₁ = P₂ tie) and `convert_eltype`. `clamped_knots` is Core's. RC keeps what is its own: `VignettingVar`, `vignetting_rule`,
  `vignetting_basis`, `center_constraint` (still exported), `tie_pole`, `RadialSupportRule`, the hard and soft pins and the
  smoothness rows. BSplineKit is no longer a dependency. Compat VisualGeometryToolkitCore 0.5.6 (`center`).
- **Breaking**: a state file written before (`state.jls`, `state_it<N>.jls`, an `io.save_init` start state) holds RC's former
  vignetting types and no longer deserializes; regenerate it from its run's records (the ground-truth states of the paper project
  are regenerated by its prepare stage, `design_ground_truth`).
- experiments: PanoramaStitching 0.3 (its fit on Core's `LogVignetting`, one linear solve in log V; the frames' gains per colour
  plane, `gain_correction = "channel"`, a `K × 3` matrix in the records). The fit stage records a fit's `log_control_points`.
  `frame_gains_fixed_v` is gone: the gains under a fixed V are PanoramaStitching's `fit_vignetting` with no free direction.
- **A precompile workload** (PrecompileTools, `src/precompile.jl`): every route's problem and its cost blocks' residuals and
  Jacobians (the double sphere with hard and soft pin and a free warping, Kannala–Brandt, OpenCV's 14 coefficients, the division
  lens, a fixed lens, the grid lens's per-tile blocks with and without warping), the render, the closed-form radiometry and the
  pipeline options are compiled into the package image, on a synthetic 24 × 24 scene: CPU only, no data files, no output. A fresh
  process's first calls of the entry points take 68 s instead of 276 s (`run_lm!` 114 → 23 s, `run_lm_grid!` 69 → 31 s); the
  package precompiles in ~130 s instead of 15 s. No NLLSsolver solve runs in it (NLLSsolver's `optimize!` evaluates into its own
  module, which a precompiling package may not do), so the LM loop compiles on first use; the non-central grid lens is left out
  (its items are dominated by its 125-parameter LM). Every result is unchanged.

## 0.4.0 (2026-11-04)

The grid lens holds only the nodes its imaged region reaches (VisualGeometryToolkitCore 0.5.4's `present`, VisualGeometryToolkitIO
0.5.2's `vgio/generic_camera/v3`). Every result is bit for bit 0.3.0's.

- **No dependent nodes.** A grid run's lens holds the nodes whose B-spline support meets the imaged region (`grid_present`: Core's
  footprint rule `node_mask(model, imaged_region(model))`, the set that was interior before). The other nodes are absent, with
  NaN in their slots, and are never read: Core evaluates a pixel only when its whole 4 × 4 window is present. They were
  dependent before (Core's `BoundaryRule`, extrapolated from the interior after every iteration), and no imaged pixel read them
  either. Measured on the typed-circle fisheye (grid_prior.toml's blob_circ: 506 of 875 nodes outside the circle, the
  tikhonov_to_start, bending and GCV-weighted P-spline priors) and on the three pipeline goldens: every output `==`.
- **No residual reads an absent node**: `build_problem_grid` refuses a pixel whose sensor sample reads one (`ArgumentError`).
- `with_present(lens, present)` (exported) holds a mask; `with_region`, the patch assembly and the start perturbation keep it,
  and the grid files store the present nodes only (`save_grid`, Core's runs per lattice row). A v2 grid file loads with every
  node present.
- **Breaking**: the option `grid.boundary_order` is retired (it set the dependent nodes' extrapolation, which nothing reads);
  `build_problem_grid` / `run_lm_grid!` lose `boundary` (the mask is the lens's own) and `select_grid_prior_weight` its `rule`
  argument; `grid_boundary_rule` is `grid_present` (a mask, not a rule). The vignetting tensor keeps Core's `BoundaryRule`
  (`vignetting.boundary_order`): V is a field over the whole sensor, read outside the imaged region.
- **A departure from the spec's decision 4**: `grid.nodes = "all"` stays, as "every node held" (the grid A/B's other arm, which
  the experiments' ablations run); the spec retired it with the dependent nodes.
- `parameters.json` loses its `grid.boundary_order` key (always null). Over a non-central lens's origins, `origin_offset_mm` and
  `origin_change_um` are taken over the held nodes (they were over the dependent nodes too); the log's "model drift" median is over
  the probes the lens evaluates.
- On a device, a tile's finiteness gate reads the present nodes only (an absent node is NaN by design); `grid_angular_error`
  compares where the grid evaluates.

The capture's camera settings (Core 0.5.4's `CaptureSettings`, Features 0.4.2's `extract_capture_settings`):

- `load_session(…; raw_dir, lens_identity)`: with `raw_dir`, each frame's RAW file (the image's stem, `RAW_EXTENSIONS` in any case;
  an `ArgumentError` naming the image when there is none or more than one) read by `load_raw_metadata` into the session's `settings`
  (`nothing` without `raw_dir`); `lens_identity`, a `DeviceIdentity`, overrides the lens EXIF claims (an adapted lens).
- Options `io.raw` (a path key) and the optional table `io.lens_identity = {make, model, serial}` (`LensIdentityOptions`); the blob
  front-end hands them on and logs one line (`settings_line`, public): frames, body, lens and its source, and the exposure,
  "constant 1/60 s f/4 ISO 100" or its ranges.
- the experiments' `decode_raw` writes `capture_settings_<set>.json` (IO's `vgio/capture_settings/v1`, body and lens from EXIF) beside
  `exif_<set>.json`, from the metadata it already reads, and logs the line.
- Compat: VisualGeometryToolkitCore 0.5.4, VisualGeometryToolkitFeatures 0.4.2, VisualGeometryToolkitIO 0.5.2.

## Unreleased (port onto VisualGeometryToolkitCore)

The grid fit (2026-11-04; the numbers of a non-central fit change):

- **The non-central origins move in 3-D.** Every node's chief-ray origin is a free (Δx, Δy, Δz) (Core's `AdditivePoints`; was
  (Δx, Δy) with Δz held at the start's). A slide of a node's point along its own ray is not a gauge of the spline model (the
  samples' lines blend points and directions separately), so no per-node coordinate is held; the one gauge, a common translation
  of the origins that the poses' translations absorb exactly (cost invariant to 5e-30; a common rotation is not one, 5e-7), is
  held by pinning the chief node's origin (`chief_origin_node`). Each sensor sample's origin is seeded in all three components
  and fanned out as w·∂Φ/∂o (exact: ∂Φ/∂o has a component along the chief ray, since the aperture cone slides along its axis).
  Synthetic truths with origins on no plane are recovered (rays to 103.5°: to 3e-27 of the start cost), which the held z cannot
  reach (1.5e-8 and 3.1e-6). The real 8 mm fisheye start (3 views): cost +0.9 %, bearing p90 0.295 → 0.284 px.
- **The light sphere's exit point had a NaN derivative** for a board point outside the sphere (`ω̂`'s `sqrt(max(Δ, 0))`):
  `scrub_jacobian!` zeroed it, so every such pixel's Jacobian row was zero and a wide view's fit stalled at the truth. Fixed
  (`Δ > 0 ? sqrt(Δ) : 0`, values unchanged); `scrub_jacobian!` now counts what it zeroes by parameter block and `run_lm!` /
  `run_lm_grid!` warn with the counts. The real fisheye start has no such pixel (0 of 72,900).
- **The device kernels are Core's `pointwise!`** (`OnBackend`, Core 0.5.2): the flux, the linear blocks' parts and the dual
  Jacobian are functors of a pixel id (`_FluxAt`, `_PartsAt`, `_JacAt`); RC has no kernel code. Values bit for bit.

Repository:

- Cleanup (2026-11-04), every value bit for bit: `noncentral_model_from_samples` removed (no caller since the grid starts are
  `pspline_grid`); `lattice_boundary_rule` removed (callers take Core's `BoundaryRule(m, region; order)`, which refuses another
  order); the Δ³ rows read Core's `difference_stencil(3)`; one mean-positive-eigenvalue normalization for both penalties; the GCV
  pencil and the vignetting solve factor in place (no extra n × n / design-matrix copies); the GPU Jacobians convert V and the
  lighting to Float32 once; the experiments' inputs resolve through one `config_input(cfg, id)` and `frame_gains_fixed_v` takes each
  frame's log ratio once. experiments drops RANSACScoringDoneRight and LocalOptimizationDoneRight (their one reader, the rotation
  registration, is PanoramaStitching's example now) and takes PanoramaStitching 0.2 (its compat still named 0.1.2, which does
  not resolve on Core 0.5).
- The stack's 0.5 series (2026-11-04): VisualGeometryToolkitCore 0.5, IO 0.5, Estimation 0.6, Features 0.4.1 (package,
  experiments, the stage C/D environment). Core 0.5's API: a grid lens's seeded projection is `project(m, X, 𝐩₀)` (the seed
  positional, `project_pixel`); the OpenCV 14-coefficient lens is `Tilt ∘ Additive` (a CoordinateTransformations
  `ComposedTransformation`, `lens_type_name`'s "opencv14"; CoordinateTransformations is a new direct dependency); the run's region
  is Core's `imaged_region(camera, circle)` (`WholePlane()` without a circle) and its pixel mask Core's `pixel_mask(region, W, H)`
  — `run_region` and `region_mask` are removed (the audit's EXACT swaps: the same predicate and pixel layout).
- The repository's persisted records migrated by VisualGeometryToolkitIO's `vgio_migrate_tree` (2026-11-04): cameras v3 → v4 (the
  goldens' and the experiments evaluation fixtures' lenses, the lenssim session's camera), rig v5 → v6 (its targets, content id
  unchanged); every migration read back as the same value, the numbers unchanged (provenance keys reordered by the writer). The
  inline camera record of the Thorlabs test and the comments naming schema versions follow. The data of record under data_root()
  is not migrated here (the owner's step).
- The board warping on Core 0.5 (2026-11-04): `posed_target` builds its `HeightField` on the page's dimensionless chart — the
  reflected nodes `(u, −v)`, heights in chart units — placed by Core's `ChartPlacement(origin, side)` at the page's top-left corner
  (lengths when the target has units), so the heights are kept bit for bit and the posed surface is the warping, piston included;
  `HEIGHT_MANIFOLD` has no gauge (Core's pure-bending kinds: the free heights have no affine term, an oblique rule). The relief of a
  render is formed once per pixel's quadrature (`BoardRays`: its TPS weights and nodes), not per ray, and Float32 heights read
  Float32 copies of `HEIGHT_CW` / `HEIGHT_NODES` — both bit-identical.
- The grid starts (2026-11-04) are VisualGeometryToolkitEstimation's `pspline_grid` — one penalized least-squares solve on Core's
  full-image lattice of the configs' fixed grid size (`grid.size`): `grid_model` (over `grid_fit_region(src, domain)`, the sensor ∩
  the domain for a pixel camera) and `schoeps_gridlens` (over Schöps's calibrated box) for `CentralGenericModel`, `noncentral_gridlens`
  for `NoncentralGenericModel`, its lines anchored at their points closest to c* (Core's `nearest_point`; c* from the fit's
  metadata, RC's camera origin as before). Replaces Core's quasi-interpolation at the nodes and, for the non-central start, the lines
  sampled at clamped nodes and moved to the pupil plane. `π_origin` / `pupil_origins` are the full 3-D stored points (were (x, y, 0)).
  Measured: the start's direction error falls 25–30× on synthetic central lenses (35 × 25), and on the real FX3 fisheye's
  non-central Schöps model (35 × 25) from 0.114 / 19.6 px (median / max) to 0.031 / 1.37 px, the line offsets from 146 µm / 203 mm
  to 44 µm / 2.0 mm. The golden scene keeps its recorded quasi-interpolated grid (test/regression/golden_scenes.jl).
- the experiments' `scripts/panorama_vignetting.jl` (2026-11-04): PanoramaStitching's example chain (examples/panorama_vignetting:
  load_raw → register_rotations → fit_vignetting, interpolation free → score_seams → render_panorama), called from configs of record
  with their inputs in the data tiers — the FX3 + Sigma 14 mm indoor sweep (`panorama_vignetting_fx3_14mm.toml`, its frames and
  migrated lens `[[unpacked]]`) and the FX3 + 8 mm rooftop sweep (`panorama_vignetting_fx3_8mm_rooftop.toml`); radial V.
  `src/panorama.jl` loads on PanoramaStitching's new API (the removed cell imports dropped); the rooftop chain's port is separate.
- the experiments' RAW decoding (2026-11-04): the prepare stage's modes bin2 and green decode in Julia, VisualGeometryToolkitFeatures'
  `load_raw` on LibRaw_jll (`decode_raw`, `write_decoded`: convert_raw.py's files and layout); fullres (a demosaic, which Features
  does not provide) stays scripts/py/convert_raw.py. On an FX3 ARW of the 2026-11-02 Thorlabs set the two decoders' linear PNGs,
  masks and sRGB copies are identical. LibRaw_jll joins the experiments' deps.
- `load_camera(spec)` (2026-11-04): the Core camera of a model spec (`grid:`, `lens:`, `schoeps:`, `noncentral:`), the object
  Core's `backproject_with_jacobian` takes. the experiments' night-stars px residual (`Stars.residual_px`) is through that analytic
  Jacobian instead of central differences of 0.5 px: `px_jacobian` and the `[star_fits]` key `jacobian_step_px` are removed
  (night_stars.toml, grid_prior.toml). The stored-run residuals move by at most 2.2e-7 px (the differences' error).
- the experiments' Thorlabs evaluation (2026-11-04): `robust_lm!`'s Jacobian is ForwardDiff's (was forward differences of 1e-7), the
  local scale and the residual vectors are through Core's analytic `backproject_with_jacobian` (were central differences of
  `jacobian_step_px`, a key removed from `[evaluate]`, thorlabs_distortion_fx3_14mm.toml); ForwardDiff joins the experiments' deps.

- The stack's 0.4 series (2026-11-03): VisualGeometryToolkitCore, IO, Features and Estimation 0.4 (from 0.2) for the package and
  the migration harness's stage C/D environment; experiments also VisualGeometryToolkitEval 0.4, RANSACScoringDoneRight 0.3,
  LocalOptimizationDoneRight 0.3 and PanoramaStitching 0.1.2. No source change: nothing here used the Sampson and whitening API that
  Core 0.3/0.4 reworked. The golden scenes evaluate bit-identically on 0.2.1 and 0.4; the known issues below are unchanged.
  `test/integration_blobboards/frontend_blobboards` stays on the recorded legacy stack until BlobBoards is released on 0.4.
- `docs/handover/` (2026-11-02): two notes for a successor — the open Tamron panorama rotation bundle (all 35 frames; the paper's
  row is 18 frames without the bundle) and a worked refinement of the table (`si_14mm_bc5`: inputs, the run log read line by
  line, the chrome-board evaluation with one rotation centre and its numbers).
- The paper project `experiments/` is complete (2026-10-29): README with the publish and handover steps (Zenodo record, GPU run,
  first-deploy sign-off, Blens subtree sync). The golden gate on Julia 1.13 in exact mode: blob_kb 248 and siemens_coarse_full 5551
  quantities match; fisheye_nc_selfcheck not run to completion (see the commit).
- `experiments/` and Git LFS retired (2026-10-29). The experiments' inputs are the paper project's published data: four tiers of lazy
  Julia artifacts (`experiments/configs/data_artifacts.toml`, `experiments/Artifacts.toml`; `experiments/scripts/fetch_data.jl`), their
  recipes are the staged drivers of `experiments/`, the renderers of the synthetic ground truth `experiments/src/ground_truth.jl` (the
  detector-bias driver's prepare stage), the board records' writer `experiments/src/boards.jl`, the A/Bs (the grid prior and its
  free nodes, the focus surface, the LM iterator, the image circle) the paper project's ablations (`experiments/scripts/ablation.jl`,
  `experiments/configs/ablations/`); `.gitattributes` is gone. The package's golden gate (`test/pipeline/run_pipeline_checks.sh`) reads
  the cases from the data root (`VGTK_DATA_ROOT`, the tier `rc_calibration`), its configs' paths relative to it. The migration
  harness's stages C and D run in their own environment, `test/integration_blobboards/Project.toml`.

Added:

- Smoothness penalties of the spline regressions and their weights (2026-11-04; measured 2026-11-03 on a noise-free forward-model
  regression of a real fisheye grid). `grid.prior = "pspline"`: the third differences Δ³ of the correction d̂ − d̂₀ along both lattice
  axes over the windows of 4 consecutive interior nodes, normalized as the other kinds (β relative to Σϕ² per node and the
  operator's mean positive eigenvalue) — it holds the nodes whose support merely grazes the region without biasing the
  well-determined ones. `vignetting.smoothness_weight` (default 0 = none): the same Δ³ penalty over the free control points in the
  closed-form vignetting solve (`solve_Θ_v!`; refused with `sweep.joint`). `"gcv"` for either weight (and `grid.prior_weight`):
  β by generalized cross-validation, a scan of log₁₀ β ∈ [−8, 4] refined by Brent's method (`gcv_select`) — for V on each solve's
  own linear system, for the grid on the linearization of the grid problem at the start, after the closed-form radiometry
  (`select_grid_prior_weight`); the chosen β and the curve's minimum go to `parameters.json` (`grid.prior_gcv`,
  `vignetting_options.smoothness_gcv`). `grid.boundary_order` / `vignetting.boundary_order` (default 2, the natural Δ²c = 0): 3
  extrapolates the dependent nodes with Δ³c = 0 (`c₀ = 3c₁ − 3c₂ + c₃`, about half the edge error), Core's
  `BoundaryRule(m, region; order)` of VisualGeometryToolkitCore 0.5 (`lattice_boundary_rule`; until RC adopted Core 0.5 on
  2026-11-04 an `ArgumentError` naming that requirement). At the defaults every run is unchanged and `parameters.json` gains no field.

- `grid.start_perturbation = {sigma_px, seed}` (2026-11-02; grid-prior-ab's RC_GRID_PERTURB, ported, not merged): every direction
  control of the start grid rotated on its sphere by Core's control chart (`ControlPatchManifold`), σ / f px rad RMS per tangent
  axis, draws from `StableRNG(seed)` (`perturb_start` / `perturb_directions`, grid_prior.jl). Applied after the masks and before the
  boundary rule and the prior, so the prior's reference d̂₀ and `grid_init.json` are the perturbed start. Absent = none; when present
  both keys are required and others refused; a parametric start lens is refused. `parameters.json` records it under `grid`.
- Two knobs of the pre-port code ported (2026-10-31; paper-finalization 4883fab): `rendering.hammersley` (default `false`) draws the
  N radiance samples from the Hammersley set shifted per key — `RadianceSampler`'s flag, present but never exposed — the same N at
  about half the per-pixel noise of the pseudo-random draws; `masks.edge_fraction` q (default 0 = uniform) makes the pixel subset
  edge-selective: the Bernoulli draw becomes lm_fraction / q and `_edge_select` keeps the q share with the strongest |∇ϕ|² of the
  photo (four neighbours, summed over the channels), the per-view cap applies after it as before. The photos are decoded once
  (selection, cap and observations in one sweep). At the defaults every pixel set and render is unchanged. Motivation: the
  low-distortion truth test (poses and the 35 × 25 grid free) — the drift from Schöps's start was the Monte Carlo floor, noise ÷ √pixels;
  with Hammersley, q = 0.37 and 1.9 M pixels it fell from 0.0065 to 0.0015 px (Schöps 0.0012), with 7.7 M pixels to 0.0008 px.
- The closed-form start runs on the LM's own pixels (2026-10-31): the lighting-only Θ_l solve and the one sweep of Θ_l, Θ_v, μ use
  the run's masked pixels, their parts rendered once (`solve_linear_blocks!(s; parts)`), and the sweep is always kept — every solve is
  exact on these pixels, so the cost cannot rise. Before, the start ran on a separate sparse Bernoulli subset (`masks.linear_fraction`,
  its own seed) and was kept only if it lowered the cost on the LM's pixels (a test that fired in none of the runs of record).
  `masks.linear_fraction` is accepted but not read, so the configs of record still load; the "masks:" line no longer reports a
  linear-init count. The start of every run changes (the solves see other pixels), the pipeline goldens with it.
- The pixel masks of `prepare_refinement` are built threaded over the views (2026-10-31): the Bernoulli candidate draws stay in view
  order on the run's one RNG stream (the same pixels as before), and only the board test of the candidates (`compute_mask`) runs under
  `Threads.@threads`. Verified bit-identical on a Tamron Brown–Conrady AprilGrid start (cost 1.2066e+01 both ways); the start of a
  139-view set took 303 s instead of 1531 s on 8 threads.
- The closed-form V solve's bending-energy penalty is gone (2026-10-31): `vignetting.smoothing`, `vignetting_bending_rows` and the
  `smooth_rel` of `solve_Θ_v!`, `solve_radiometry_gn!` and `solve_linear_blocks!`. Measured on the Tamron and the fisheye starts
  (smoothing 1e-3 against 0): V changed by ≤ 4e-4, μ and the lighting not at all — the support rule already replicates the control
  points without data, which the penalty was for. The key is refused like any other unknown one (no config of record sets it).
- `run_image_circle` seeds `detect_image_circle` with `StableRNG(masks.seed)` (2026-10-31). The RANSAC ellipse fit ran on Julia's
  default RNG, so a run with a measured image circle (`camera.image_circle_frames`) got a slightly different circle, imaged region
  and pixel set every time (two identical fisheye starts differed by 1 % in cost). Runs with a measured circle change; runs on the
  whole sensor do not.
- The focus surface's TPS nodes are the nodes of the 5 × 5 grid over the imaged region's bounding box that the start lens images inside
  the region (`focus_grid_nodes`, `_check_focus_nodes`; 2026-10-31) — the ruling for the grid's and the vignetting's nodes outside the
  imaged region applied to the focus surface. Before, the grid spanned the sensor and every node was a parameter: the FX3 fisheye runs
  had 5 of 25 nodes in their image circle and 20 free nodes with no pixel of their own. A lens that lights the whole sensor keeps its
  25 nodes and its numbers; the pipeline goldens of the cases with an image circle (blob_kb, fisheye_nc_selfcheck) change.
- The aperture radius is no longer boxed (2026-10-31): the scalar r keeps its log chart (r ← r eᵘ) without the clamp to
  `[camera.r_min, camera.r_max]`, and the non-central r(𝐩) spline's controls step the same way, c ← c eᵘ, with their Jacobian
  columns scaled by the control (`_log_r_column!(…, ::ApertureSplineVar)`; before: additive steps clamped to the box). Positivity
  comes from the chart; `camera.r_min == camera.r_max` still fixes the aperture (the block is removed). Why: the box had pinned the
  f/4 runs at 3 mm and once clamped a start above it into a worse state (f4_tamron, 30.10); the non-central golden changes.
- The material block steps on charts (2026-10-31): α = eᵘ, η = 1 + eᵘ (a dielectric in air), μ = lo + (hi − lo) σ(w) on
  `material.mu_bounds` — the Jacobian's nine material columns scaled accordingly (`_chart_material_columns!`, beside
  `_log_r_column!` at every Jacobian return); before, additive steps clamped to `bounds(::Material)` (α ∈ [0.01, 1], η ∈ [1.0001, 3]).
  Why: the paper's η sat on its lower bound in most runs of record (the fits make the paper Lambertian) and ran to the upper bound
  under DᵀD; the charts make the limits approachable without projected steps and drop the arbitrary upper bounds. The closed-form μ
  solves clamp to `mu_bounds` instead of [0, 1]. Every golden with a free material changes.
- `rendering.normalize_aperture` (default `true`, 2026-10-31): the central lenses' flux is the aperture integral divided by the
  aperture area, Φ = ∫ E / (π r²), as the paper's Φ_c states and as the non-central lens has always done per position
  (`RenderSettings.normalize_aperture`, honoured by the CPU and device renders and the closed-form start). Until now the pipeline left
  the flag at `false`: Φ ∝ r², the scale absorbed by the lighting, and ∂Φ/∂r carried the 2Φ/r term. `false` restores that. Every
  pipeline golden changes.
- `grid.nodes` (`GridOptions.nodes` ∈ `:interior`, `:all`; `GRID_NODE_RULES`; `grid_boundary_rule(lens; nodes)`): the grid's
  parameters are its interior nodes with the others dependent (the default, the behaviour so far) or every lattice node — the
  boundary A/B's other arm, until now an experiment-only override (experiments/grid_prior/refine_allfree.jl).
- `detect_image_circle([rng,] [T = Ellipse,] frames; smoothing_weight, threshold)` (`T` ∈ {`Ellipse`, `Circle`}: Estimation's
  `find_ellipse` / `find_circle` on the contour): the image circle MEASURED from a session's photographs — their
  per-pixel maximum, ImageSmooth's L0 smoothing, VisualGeometryToolkitFeatures' dark MSERs, the surround (the region holding the
  extent's four corners and not its centre), its hole's contour and the model's RANSAC — in COLMAP pixels
  (the lens's lit region, not necessarily round: an `Ellipse` by default); it assumes a dark surround (a field stop) and checks none. New dependencies ImageSmooth
  and GeometryBasics.
- The run's image circle is MEASURED (`run_image_circle`): `detect_image_circle` on the photographs of `RC_IMAGE_CIRCLE_FRAMES`
  (`RefineOptions.image_circle_frames`), else the region a grid start's record stores, else none (the whole sensor). A grid stores it
  as its region (`with_region`: Core's generic model's `domain`, persisted by its record), which `in_domain`, its interior nodes and
  its prior read; `run_region(circle, W, H)` is the run's support for the masks, the focus nodes and the vignetting.

Removed from the public API, with no deprecation shim:

- The grid lens's free-node option and the typed image circle (`RC_GRID_FREE_NODES`, `RC_IMAGE_CIRCLE`, `RefineOptions.image_circle`):
  the run's region is the sensor's rect ∩ the MEASURED image circle (`run_image_circle`), once per run — never predicted from the
  start lens (its fold is Core's `valid_domain`, where the model is valid; `start_region` and the grid's fold-derived domain are
  gone: `grid_model(src, G_x, G_y; domain)` stores the region it is given, the whole plane by default); the grid's PARAMETERS are its interior nodes (Core's `free_nodes` over the region:
  the nodes whose support meets it) and every other node is DEPENDENT, defined from the interior by Core's `BoundaryRule` (the natural
  boundary condition f'' = 0 of BSplineKit's recombined basis, the linear extrapolation of the correction outward) — no frozen and no
  unconstrained node. The same for the tensor vignetting surface: its control points whose knot span meets the region are the unknowns
  of the linear and LM solves (`vignetting_basis`: one column per interior point), the others follow them through the same rule
  (`vignetting_rule`, the surface opting into Core's support trait). The radial profile's ρ = 1 is the region's radius seen from the
  sensor centre (`vignetting_radius`: the farthest imaged point, no extrapolation on an imaged pixel). The masks, the focus nodes'
  support and the vignetting read the one region.
- The grid prior and the vignetting penalty as second differences of the control values (P-spline): both are the ANALYTIC bending
  energy ∫∫ f_xx² + 2 f_xy² + f_yy² of the spline — VisualGeometryToolkitCore's `bending_stencils` / `bending_gram` (BSplineKit's
  Galerkin matrices) — of the correction d̂ − d̂₀ per component for the grid lens (`RefineOptions.grid_prior` ∈ `:bending`,
  `:tikhonov_to_start`, `:none`; `grid_prior_weight`), of the control points for the vignetting (`vignetting_bending_rows`).
  The default is `:tikhonov_to_start` at `grid_prior_weight = 1e-1`, measured (the grid A/B: `experiments/configs/ablations/grid_prior.toml`).
  `second_difference_rows` is gone; `clamped_uniform_knots` is BSplineKit's knot vector (`clamped_knots`, the static vector a kernel
  takes).

- The sensor chart ℐ = [−½, ½]² and every unit of the image's size in it: a sensor position 𝐩 is a COLMAP pixel of the image,
  ℐ = [0, W] × [0, H] (VisualGeometryToolkitCore's `Rect(sensor)`), everywhere — the cameras' K, the pixel footprints and quadrature
  (`𝐩_of_pixel(u, v)`, `𝐩_sample(q, k, u, v)`, `ℐ_𝐮(u, v)`: no image size), the focus spline's nodes (`grid_nodes(n, Rect(0, 0, W, H))`,
  `Focus(n, d, W, H)`), the vignetting (`Vignetting(n, W, H)`, `RadialVignetting(n, 𝐜, r_max)` with the centre in pixels;
  `sensor_center(v)` is the gauge pin's point), the masks, the detections and the self-check's residual vectors (`residual_vectors(lens,
  pose, feats)`, `project_pixel`). `chart_of_colmap`, `colmap_of_chart`, `ℐ_lo`, `ℐ_hi`, `pixel_area` are gone.
- `ParametricLens` (and `GeometricModel`, `chart_intrinsics`, `camera(lens, ext)`, `pixel_parameters`, `pixel_camera`, `LENS_LM_BOX`): a
  parametric model is VisualGeometryToolkitCore's pixel camera, a `BoundedCamera` of a `CameraModel(LogicalIntrinsics(K), lens)` with K in
  the image's pixels — isbits (Core's `BoundedCamera{C, S}`), its π⁻¹ Core's `backproject`, its element type Core's `convert_eltype`.
  The per-view lens block is Core's `ParameterManifold` of the camera's model with the box `lm_box(camera, W, H)` (in pixels: f in 0.1 …
  50 image widths, the principal point within 0.3 of the image of its centre) and the coordinate scale `lm_scale(W, H)` (Core's new
  `scale` option: the principal point and the skew one coordinate per image width / height), so the LM's λ·I damping steps the pixel
  camera exactly as it stepped the camera of the normalized image. `save_lens(path, camera)`, `load_parametric_lens(path) -> camera`,
  `grid_model(camera, G_x, G_y)` (the size from the camera), `schoeps_gridlens` / `noncentral_gridlens` / `grid_model_from_samples` /
  `noncentral_model_from_samples` take the nominal pinhole as Core's `LogicalIntrinsics` in pixels. `load_session` returns the record's
  camera bounded by its `Sensor` and moved to the scene's pixels when the photographs are box-averaged (K scaled by the sampling).

- `angle_between`: VisualGeometryToolkitCore's `angle_between(u, v)`, `atan(‖u × v‖, u · v)` — accurate near 0 and π, where the
  former `acos` of the dot product lost half the digits (an angle of 1e-8 rad came back as 0); scale-free. The pipeline's bearing
  numbers move at the 1e-8 relative level (the pipeline goldens are re-recorded with it).
- `schoeps_central`, `schoeps_noncentral`: read Schöps's `intrinsics0.yaml` with VisualGeometryToolkitIO's
  `load(CentralGenericModel, yaml)` / `load(NoncentralGenericModel, yaml; file_unit = m)` — the point grid's unit, the calibration
  pattern's, stated at the call (metres).
- `HEIGHT_Z`: the board warping's free height directions are VisualGeometryToolkitCore's `TargetManifold` of the height field
  (`HEIGHT_MANIFOLD`), which fixes the same gauge — the piston and the two tilts; `HEIGHT_BASIS` is its `retract_jacobian`, an
  orthonormal basis of the same subspace (another basis of it, so the free height coordinates are rotated).
- The Schöps board records (`experiments/lib/board_records.jl`, since 2026-10-29 the paper project's `experiments/src/boards.jl`): the target is VisualGeometryToolkitCore's
  `PlanarTarget(SchoepsPattern)`, its points in the pattern's design chart (the pattern's lower-left corner at the origin) and the page
  the canvas with its origin; `schoeps_lattice` and `schoeps_feature_mm` are gone. The two tracked records are rewritten (every point
  the same on the page to 2.3e-13 mm).
- The experiments' `refine_lens_and_poses` and `bundle_residuals` (experiments/lib/lens_tools.jl, retired 2026-10-29): the joint LM of a camera and its
  poses is VisualGeometryToolkitEstimation's `refine_camera_and_poses(camera, poses, views)` on the pixel camera (Core's
  `BoundedCamera`, the model's own camera); `views_in_pixels` builds its views.
- `CameraToWorld`, `camera_to_world`: a view's pose is Core's `EuclideanMap` (`EuclideanMap(R_cw, 𝐜; check = false)`); `View` takes
  the pose's type as a fourth parameter. `PoseChart`: the pose block is Core's `LeftRigidManifold6{Float64}`. `gsqrt`, `lsq`: inlined
  into their callers (the light sphere's exit point; LinearAlgebra's pivoted QR). `SensorQuadrature` / `ApertureQuadrature` are built
  from Core's `square_rule` / `unit_disc_rule` (the same nodes and weights).
- `GridPatchVar`, `NCGridPatchVar` (the per-tile grid blocks): a patch is VisualGeometryToolkitCore's `ControlPatchManifold` of the
  block's controls — central: the directions on Core's direction chart; non-central: directions and point controls `restrict`ed to
  (δ₁, δ₂, Δx, Δy) per node (Δz held, the origins in the pupil plane) — an NLLSsolver variable through Core's manifold adapter. The same
  step arithmetic and coordinate order.
- `LensVar` (the per-view lens block) and the `_boxed` step: the block is VisualGeometryToolkitCore's `ParameterManifold` of the chart
  camera with the LM's search box (Core's `box` option, which clamps every step into it) `restrict`ed to the free parameters —
  `lens_manifold(lens; free, tie_f)`, an NLLSsolver variable through Core's manifold adapter; `lens_free(lens; free, tie_f)` names the
  free parameters and says whether the block is held fixed. `lm_box(camera)` is now the box by Core's parameter names (the
  `ParameterManifold`'s `box`); `parameter_box` of that manifold is the former `(lo, hi)`. The material and aperture-spline blocks
  step with Base's `clamp`.
- The printed and reported millimetre, micrometre and milliradian values of `refine!` / `write_refinement` are converted through
  FlexUnits (`ustrip(mm, …)`, `ustrip(μm, …)`) instead of numeric factors; the report's fields and values are unchanged. The model drift
  of the per-iteration line is printed in degrees (`rad2deg` of the radians), not mrad.
- `DoubleSphere` (the type alias of `ParametricLens` over Core's double sphere, its six-number and vector constructors): the model is
  `ParametricLens(VgtkCore.Lens(VgtkCore.DoubleSphere(ξ, α)), f_x, f_y, c_x, c_y)`, like every other Core lens; the family is
  `core_lens(l)`'s. The `lens_type` tag `double_sphere` on disk is unchanged.
- `KannalaBrandtLens`, `RationalLens`, `OpenCVLens` (type aliases and their positional constructors) and `chart_lens`: a parametric
  model is `ParametricLens` over its Core lens — `ParametricLens(VgtkCore.KannalaBrandt(k₁, k₂, k₃, k₄), f_x, f_y, c_x, c_y)`,
  `ParametricLens(VgtkCore.Division(k₁, k₂), …)` — or, from pixel data, `ParametricLens(camera, W, H)` of a Core `CameraModel` with K
  in COLMAP pixels (the one pixel path; an OpenCV calibration is VisualGeometryToolkitIO's `opencv_camera(K, dist)`, and K / dist back
  out are its `opencv_parameters(pixel_camera(l, W, H))`). A method that needs the family dispatches on `core_lens(l)`. The
  pixel-intrinsics `ParametricLens(lens, f_px, c_px, W, H)` method is gone. The `lens_type` tags on disk are unchanged.
- `RefineOptions` fields `sensor_mm`, `f_mm`, `page_mm`, `height0` (plain millimetres): now `sensor_width`, `focal_length`,
  `page_size` and `height_start`, lengths (FlexUnits quantities) read from the same `RC_*` variables.
- `gauss_legendre(n)`: use QuadGK's `gauss(n)` (same rule, nodes ascending).
- `hat`, `expSO3`, `leftJacobianSO3`: use VisualGeometryToolkitCore's `skew`, `exp_so3` and `left_jacobian(SO3, ω)` (the same
  maps; Sophus' left Jacobian, the transpose of GTSAM's `ExpmapDerivative`).
- `GridLens`, `NonCentralGridLens`, `AbstractGridLens`, `grid_lens_from_samples`, `noncentral_lens_from_samples`, `tangent_basis`,
  `gridlens_from_json`, `ncgridlens_from_json`: the grid lenses ARE VisualGeometryToolkitCore's `CentralGenericModel` /
  `NoncentralGenericModel` (per-tile `GenericModelWindow`s; controls `[gy, gx]` over the image in COLMAP pixels). Builders:
  `grid_model(lens, W, H, G_x, G_y)` (Core's `init = :quasi_interpolate`), `grid_model_from_samples(F, θ, W, H)`,
  `noncentral_model_from_samples(F, O, θ, rgrid, W, H)`; reading `grid_model_from_json` / `noncentral_model_from_json` (the grid JSON
  format is unchanged, `image_size` now written by `write_grid_json`); a non-central lens is `NonCentralAperture(model, rgrid)`, the
  Core model with the aperture-radius spline; `is_grid_lens`, `grid_dims`, `schoeps_model` (a Schöps yaml as Core's model). The
  patch blocks step in Core's `direction_tangents` (another orthonormal tangent basis than the former `tangent_basis`: the same LM
  under λ·I damping, the per-tile Jacobian's patch columns rotated per node).
- `Board` (and its fields `plane`, `𝐨`, `𝐮_B`, `𝐯_B`, `𝐧`, `inv_u²`, `inv_v²`), `chart_dir`: the board's geometry is
  VisualGeometryToolkitCore's — a `PosedTarget` of a `PlanarTarget` whose `Board` is the page — and the package attaches only the
  print: `PrintedTarget(𝐨, 𝐮_B, 𝐯_B, texture; sdf, page_mm, facing, design)` holds the target (`posed_target`), its `surface` (a
  `Quad`, what every ray meets), the texture, the edge map and `page_mm`. `board_normal(b)` is the cameras' side of the normal.
  `posed_target(b, surface)` is the warped board as Core's target with a `HeightField` calibration. `board_relief(b, h)` is the
  warping as Core's `HeightSurface` (the chart's thin-plate spline re-expressed exactly in metres: square pages only, an
  `ArgumentError` otherwise); `board_plane_hit` is the unbounded plane hit (the self-check's board residuals).

Tests:

- The suite runs on ReTestItems (the stack's pattern): `test/unit/<x>_tests.jl` and `test/regression/golden_tests.jl` hold
  `@testitem`s, run in three runs by tag — CPU items on a worker pool, `:gpu` items on one worker, `:golden` on one worker with 8
  threads; `Pkg.test(; test_args = ["pattern"])` selects items by name, `RETESTITEMS_NWORKERS=0` runs serially.

Changed behaviour:

- The Schöps front end's ±y choice of the page rows is stored in the state — `page_flip` of `state_record`, of every checkpoint and of
  the `io.save_init` start state — and holds for a render (`io.render_state`) or a resume (`resume.state`) of that state:
  `_frontend(…; page_flip)` scores the stored direction only (2026-11-02). Before, both directions were scored again on the run's
  views; the two scores are equal to rounding when the texture's features are symmetric enough, and a subset of the views could
  break the tie the other way, rendering the board mirrored (the known issue of 2026-11-01). A state without the field (written
  before) is read as before, with a warning.
- `RenderSettings(; normalize_aperture)` defaults to `true` (2026-11-02), as the pipeline's `rendering.normalize_aperture` does.
  Before, the constructor's default was `false` while every run used `true`, so a settings object built by hand rendered a run's
  state ≈ πr² (≈ 1e-5) too dark — the synthetic replica renderer of the fisheye AprilGrid set did (02.11). The pipeline gives the
  option explicitly, so no run changes; the tests that need the paper's integral (the flux and non-central tests, the golden
  scenes, the VGC contract) and the paper project's synthetic ground truth (`experiments/src/ground_truth.jl`; its tiers of record
  were rendered so) say `normalize_aperture = false`.
- The radial vignetting profile's first two control points are tied, `P_0 = P_1` (2026-10-29; `vignetting_basis` of a
  `RadialVignetting`): V is the surface of revolution of the profile about the sensor centre, smooth at the pole only when the
  profile meets the axis at a right angle (V′(0) = 0, Piegl & Tiller eq. 3.7); the free end slope of the clamped spline made the
  field a cone at the centre, where few pixels constrain it. One parameter fewer; a resumed state of the untied profile gets
  `P_1 := P_0` (`tie_pole`). The joint closed-form solve's pin row is projected onto the free directions like its penalty (it
  assumed `Z = I`, true for the untied profile only). The golden gate's radial cases (blob_kb, siemens_coarse_full) change and need
  re-recording.
- the 14-coefficient lens (formerly `OpenCVLens`): `π⁻¹` / `in_domain` are Core's chain inverse on the imaged branch (the region connected to the axis where det J > 0
  and the rational factor is before its first fold or pole); `project` is NaN behind the camera and past that fold or pole, where it
  used to return a finite (wrong) pixel.
- A ray that misses the page (off its edges, as Core's `Quad` bounds it) is a miss — `intersect_board` gives NaN and the flux skips
  it — where it used to read the texture's edge texel on the infinite plane. The masks already kept every aperture ray on the page.
- The page's edges `𝐮_B`, `𝐯_B` must be orthogonal (an `ArgumentError` otherwise; `𝐯_B` is orthogonalised to rounding).

Known issues (2026-11-03):

- Resolved 2026-11-03: the unit suite's 18 failures and 1 error, the GPU twins and the stale package goldens, all from
  the hard vignetting pin (and, for the goldens, also the material charts and the log r(𝐩) chart). The
  tests expect 15 vignetting columns and give the truth in the pin's gauge (`pin_center_value`); the 0.007 start residual
  was (1/V(𝐜) − 1) ϕ of a truth with V(𝐜) = 0.978 rendered unpinned, correct for the pinned start, not a regression; the
  Richardson item's BoundsError was its hard-coded column 16. `golden_migration_snd_try.json` re-recorded with the measured
  reasons (test/regression/README.md).
- The pinned state does not rescale the lighting when it normalises V to V(𝐜) = 1 (`pin_center_value`), so a state built or
  resumed from a field with V(𝐜) ≠ 1 starts off its rendered prediction by the factor V(𝐜); the closed-form lighting start
  absorbs it in the pipeline. Whether the constructor should move the factor into the lighting is open.
- A non-central refinement with the aperture-radius spline r(𝐩) free stalls: the LM's damped normal matrix turns non-finite in the
  r(𝐩) block (`ApertureSplineVar`, 25 controls) and its couplings, while every tile's Jacobian, its r-columns and their JᵀJ block are
  finite — the NaN enters between the tiles' Jacobians and NLLSsolver's Hessian of that variable (seen on the 14 mm and the fisheye
  Siemens sets of 27.10; the sparse LDLᵀ path then crawls for hours, the dense Cholesky fails). Cause open. Workaround: hold the
  aperture, `[lm] fix = ["aperture"]` (at f/22 r is barely observable). With it NLLSsolver's own sparse path runs normally — a reduced
  14 mm case (61 views, 3000 px/view, 6 its) gave rel-RMS 0.3461 → 0.2117 in 438 s, the same as a forced dense solve (449 s).
