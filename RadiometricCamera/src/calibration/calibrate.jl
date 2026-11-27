# ---------------------------------------------------------------------------
# The refinement of a radiometric calibration (paper §Second Order Optimization): Levenberg–Marquardt over every block of the started
# calibration's parameters on its data (`RadiometricRefinement`'s solve of `RefineProblem(f, u0)`, `RadiometricCalibration`'s after its
# start: the route's LM, `_lm_views!` per view or `_lm_tiles!` per tile, by the camera's type; in chunks with a
# closed-form sweep between them, with the live monitor), its result a Core FitResult whose estimate is the `CalibrationEstimate`
# (the calibrated camera, the poses, the radiometric target, the light). The problem's structure is its data's (ruling R1:
# `f.calibration_settings`); a run's I/O is the driver's — the checkpoints and the optimizer's log a `callback` it builds
# (`calibration_checkpoints`), a resume a problem it builds from a state (`load_calibration_state!`, calibration/resume.jl).
# ---------------------------------------------------------------------------

"""
    CalibrationEstimate(camera, poses, target, light)

The estimate of a radiometric calibration ([`RadiometricCalibration`](@ref)): the calibrated camera (a
[`CalibratedCamera`](@ref): the `RadiometricCameraModel` — the geometric camera, the focus surface, the aperture radius and
the vignetting — with the observation model its photographs were read under and its render settings), the poses (camera →
world, one per photograph), the radiometric target (the printed target, its warping, its material and print) and the light.
"""
struct CalibrationEstimate{C, P, T, L}
    camera::C
    poses::Vector{P}
    target::T
    light::L
end

"""
    CalibrationMetadata

A radiometric calibration's diagnostics (a Core `FitResult`'s `metadata`): the LM's `trace` (one entry per LM iteration and per
closed-form sweep), the cost and relative RMS of the start (`linear_cost`, `linear_rms`: after the closed-form start) and at the end
(`final_cost`, `final_rms`), the LM's wall time (`seconds`), the last LM chunk's result (`solver`: Core's `LevenbergMarquardt`'s
`FitResult` — its termination record, `stats.nsteps`, `metadata.objective` — `nothing` without an iteration), the non-finite Jacobian entries replaced, by block
(`scrubbed`), and the calibrated camera's folds (`folds`: Core's `folded_pixels` over its coverage, [`calibrated_folds`](@ref);
`nothing` for a parametric camera), and the focus nodes of the run's support that ran away (`focus_runaway`: t̂ ≤ 0 or non-finite,
`focus_runaway`; empty when none).
"""
struct CalibrationMetadata{S}
    trace::Vector{NamedTuple}
    linear_cost::Float64
    linear_rms::Float64
    final_cost::Float64
    final_rms::Float64
    seconds::Float64
    solver::S
    scrubbed::Dict{Symbol, Int}
    folds::Union{Nothing, NamedTuple}
    focus_runaway::Vector{Int}
end

"""
    RadiometricRefinement(; optimizer = LevenbergMarquardt(; damping = LevenbergDamping()), radiometry = RadiometryBlockDescent(),
                          radiometry_stops = ToleranceStops(; max_iter = sweeps, rtol = 1e-4), lm_chunk)

The refinement of a radiometric calibration (paper §Second Order Optimization), a refine (`RefineKind`): it solves
`RefineProblem(f, u0)` of the calibration's data `f` (a [`CalibrationData`](@ref): the problem's structure its
`calibration_settings`, [`CalibrationSettings`](@ref); the start's record its `start`, `_start_state`'s, which
[`RadiometricCalibration`](@ref)'s `init` or [`load_calibration_start`](@ref) lift a geometric calibration into) from the
parameters `u0` (a [`CalibrationParameters`](@ref): refined in place, the result's estimate built from them), no context (ruling
R1). `optimizer` the refinement's (VisualGeometryToolkitCore's `LevenbergMarquardt`; its damping Levenberg's λ·I by default, which
converged on the origins of a non-central lens where λ·DᵀD stalled at 1.2e-8 of its start cost, and reaches a slightly lower cost
1.6–1.8× faster on the Tamron and grid cases), `radiometry` the closed-form blocks' refine ([`RadiometryBlockDescent`](@ref), or
[`JointRadiometry`](@ref) of the optimizer: `[sweep] radiometry`; the start's closed-form solve one sweep of it; a joint solve takes
no vignetting smoothness, refused at `init`), `radiometry_stops` the stops of its solve between LM chunks (Core's `ToleranceStops`:
`max_iter` the sweeps, `[sweep] iterations`; `rtol` the relative cost decrease of record, 1e-4), `lm_chunk` the LM iterations
between two such solves (`[sweep] lm_chunk`; 0: the LM in one run), the defaults configs/default.toml's `[sweep]`. The solve's
keywords: `stops = LmStops(…)` (Core's `LmStops`: `max_iter`, `max_time`, `ftol`, `xtol_rel`, `xtol`; the defaults
configs/default.toml's `[levenberg_marquardt.stops]`) and `callback` (Core's LM contract, `callback(state::LmIterationState) ->
Bool`, `true` stopping the solve: `state.iteration` the refinement's iteration across its LM chunks, from 1, `state.u` the
iterate's estimate decoded, `(; camera, poses, mat, surface, light)`, `state.linear_solves` and `state.elapsed` the refinement's
so far; the checkpoints and the optimizer's log of a run are [`calibration_checkpoints`](@ref)'s), as `init`'s. Its live monitor
(a line per iteration) is its printed output.

The solve is the Levenberg–Marquardt over every block the settings leave free (`fixed`), on the route the camera's type picks
(`lm_route`: per view, in a parametric camera's own parameters; per tile for a SplineLens camera), optionally in chunks with a
closed-form sweep of the lighting, the vignetting and μ between them (`lm_chunk`), up to the stops' `max_iter` iterations and
`max_time` seconds. The result is a `FitResult{CalibrationEstimate, CalibrationMetadata}`: `u` the refined
[`CalibrationEstimate`](@ref); `retcode` / `stop_rule` the LM's termination record (VisualGeometryToolkitCore's: a tolerance stop
converged, the rejection cap judged by the stall postcondition at the final state) — `(Failure, ShapeGate)` when the calibrated
SplineLens camera still folds over its coverage (`fold_termination`: no calibration; the camera is held to the coverage assertion
too, [`calibrated_folds`](@ref)) or a focus node of its support runs to t̂ ≤ 0 (`focus_runaway`, the same `validity_gate`);
`stats.nsteps` the LM iterations; `rms` the final relative RMS over the `n` measured values.
"""
struct RadiometricRefinement{O <: VgtkCore.LevenbergMarquardt, R <: RadiometrySolve} <: VgtkCore.AbstractEstimationAlgorithm
    optimizer::O
    radiometry::R
    radiometry_stops::VgtkCore.ToleranceStops
    lm_chunk::Int
end
function RadiometricRefinement(; optimizer::VgtkCore.LevenbergMarquardt = VgtkCore.LevenbergMarquardt(; damping = VgtkCore.LevenbergDamping()),
                               radiometry::RadiometrySolve = RadiometryBlockDescent(),
                               radiometry_stops::VgtkCore.ToleranceStops = _radiometry_stops(SweepTable().iterations),
                               lm_chunk::Integer = SweepTable().lm_chunk)
    lm_chunk >= 0 || throw(ArgumentError("RadiometricRefinement: lm_chunk ≥ 0 (0: no closed-form solve between LM chunks); got $lm_chunk"))
    return RadiometricRefinement(optimizer, radiometry, radiometry_stops, Int(lm_chunk))
end
VgtkCore.estimator_kind(::RadiometricRefinement) = VgtkCore.RefineKind()
"""The closed-form blocks' algorithm of the refinement `alg` on data of the settings `settings`: its `radiometry` with the
settings' vignetting smoothness (`vignetting.smoothness`; a joint solve takes none, refused naming the keys) and the printing
`verbose` — the sweeps between LM chunks and on a resumed state."""
_scheduled_radiometry(alg::RadiometricRefinement, settings::CalibrationSettings; verbose::Bool = true) =
    _with_schedule(alg.radiometry; verbose, vignetting_smoothness = smoothness_weight(settings.vignetting.smoothness))

"""
    RadiometricCalibration(refinement::RadiometricRefinement)

The radiometric calibration of the paper, a fit (`FitKind`: it seeds itself, from the geometric calibration): the START, which
lifts the geometric camera into a radiometric calibration's data and parameters with its data's start options, then the
[`RadiometricRefinement`](@ref) `refinement` of them. A run's TOML gives the run's options, the settings, it and the solve's stops,
[`read_calibration`](@ref). Solved on `EstimationProblem(CalibrationEstimate, CalibrationPhotographs(geometry, photographs; run,
settings, device))`, no context (ruling R1): the data a [`CalibrationPhotographs`](@ref) — a [`StartGeometry`](@ref), the
photographs' files, the start options of the run's [`RunOptions`](@ref) and [`CalibrationSettings`](@ref) and the compute backend
(`device`, [`default_backend`](@ref) by default). The start: the focus surface through the mean target
centre, the aperture from the f-number, the image circle measured where the run names its frames, the masks, the photographs'
observations streamed, the vignetting and the lighting by the closed-form start, the grid smoothness.

`init` returns the start: its cache is the refinement stage's `init(RefineProblem(f, u0), refinement; stops, callback)` (an
`EstimationCache` of a [`CalibrationCache`](@ref); its `prob`, the started `RefineProblem(f, u0)`); `solve!` of it, as `solve`, the
refinement — its result [`RadiometricRefinement`](@ref)'s, recorded by the front end for the stage and relabelled for the
calibration (spec §9 G-l). A started calibration ([`compute_calibration_start`](@ref)'s, `load_calibration_start`'s, `load_calibration_state!`'s, or a
caller's `RefineProblem(f, u0)`) is solved by the refinement itself: `solve(RefineProblem(f, u0), refinement; stops, callback)`. A
run's checkpoints and optimizer log are the `callback` its driver builds from the started problem, [`calibration_checkpoints`](@ref).

```julia
run, settings, alg, stops = read_calibration("run.toml")
c = init(EstimationProblem(CalibrationEstimate, CalibrationPhotographs(geometry, photographs; run, settings)), alg; stops)   # the start
r = solve!(c)                                                                                                             # the refinement
```

or, the start and the refinement in one call, `solve(EstimationProblem(CalibrationEstimate, CalibrationPhotographs(geometry,
photographs; run, settings, device)), alg; stops)`. The data and parameters refined are the ones a start record rebuilds
(`save_calibration_start` → `load_calibration_start`, bin/refine.jl's path): the start reads no refinement-only key
(`REFINE_ONLY_KEYS`), and a record carries the start's values bit for bit.

Every input is a value: the pipeline's front ends read theirs through VisualGeometryToolkitIO ([`start_geometry`](@ref)); a caller
with a Schöps run, a Kalibr or an OpenCV calibration of its own loads its camera and observations with IO's importers itself
(`VisualGeometryToolkitIO.load(BoundedCamera, …)`, `load(TargetObservations, …)`) and builds the [`StartGeometry`](@ref) — RC reads
no foreign format.
"""
struct RadiometricCalibration{A <: RadiometricRefinement} <: VgtkCore.AbstractEstimationAlgorithm
    refinement::A
end
VgtkCore.estimator_kind(::RadiometricCalibration) = VgtkCore.FitKind()

"""
    CalibrationCache

The cache of a [`RadiometricRefinement`](@ref) (its `init`'s; Core's `EstimationCache` holds it as its `cache`, and
[`RadiometricCalibration`](@ref)'s `init` returns that refinement stage's `EstimationCache` after its start): the refinement's problem
`prob`, `RefineProblem(f, u0)` of the calibration's data `f` ([`CalibrationData`](@ref): its `calibration_settings` the problem's
structure, its `start` what the refinement and its records read — the start views, the observations, the costs of the start) and
its parameters `u0` ([`CalibrationParameters`](@ref)), the `refinement`, the solve's `stops` and its `callback` (`nothing`: none).
`solve!` refines `prob.u0` in place.
"""
mutable struct CalibrationCache{P <: VgtkCore.RefineProblem{<:CalibrationData, <:CalibrationParameters, VgtkCore.NullParameters}, A <: RadiometricRefinement, C}
    prob::P
    refinement::A
    stops::VgtkCore.LmStops
    callback::C
end

"""
    _calibration_stops(; max_iter, ftol, xtol, xtol_rel, stationarity_tol, max_time, residual_resolution) -> LmStops

The stop settings of a calibration's solve when it names none: configs/default.toml's `[levenberg_marquardt.stops]`
(`read_calibration` of an empty config), each keyword given replacing its value (the package's tests and benchmarks state an
iteration cap so).
"""
function _calibration_stops(; kw...)
    s = last(read_calibration(Dict{String, Any}(VgtkCore.CONFIG_VERSION_KEY => VgtkCore.current_config_version(RC_CONFIG_VERSIONS))))
    return VgtkCore.LmStops(; s.max_iter, s.ftol, s.xtol, s.xtol_rel, s.stationarity_tol, s.max_time, s.residual_resolution, kw...)
end

"""
    compute_calibration_start(alg::RadiometricCalibration, data::CalibrationPhotographs) -> RefineProblem

The START of the calibration `alg` on its data `data` (a [`CalibrationPhotographs`](@ref)): the geometric calibration lifted into the
refinement's problem, `RefineProblem(f, u0)` of the data `f` ([`CalibrationData`](@ref), its `start` the start's record) and the
started parameters `u0` ([`CalibrationParameters`](@ref)) — the focus surface, the aperture, the image region, the masks, the
photographs' observations streamed, the closed-form start of the lighting, the vignetting and μ (by `alg.refinement`'s radiometry),
the grid smoothness. What [`RadiometricCalibration`](@ref)'s `init` refines and [`save_calibration_start`](@ref) records; a stored
start is [`load_calibration_start`](@ref)'s. A driver that hands the refinement a run's callback starts here
(`prob = compute_calibration_start(alg, data)`; `solve(prob, alg.refinement; stops, callback =
calibration_checkpoints(prob, run))`: the checkpoints read the started problem). A joint radiometry with a vignetting smoothness
is refused before the start.
"""
function compute_calibration_start(alg::RadiometricCalibration, data::CalibrationPhotographs)
    refinement = alg.refinement
    _scheduled_radiometry(refinement, data.calibration_settings)   # (a joint solve takes no smoothness: refused before the start)
    f, θ = _start_state(data.geometry, data.photographs, data.run_options, data.calibration_settings, nothing; backend = data.device, refinement.radiometry)
    return VgtkCore.RefineProblem(f, θ)
end
VgtkCore.__init(prob::VgtkCore.EstimationProblem{CalibrationEstimate, <:CalibrationPhotographs, VgtkCore.NullParameters},
                alg::RadiometricCalibration; stops::VgtkCore.LmStops = _calibration_stops(), callback = nothing) =
    VgtkCore.init(compute_calibration_start(alg, prob.f), alg.refinement; stops, callback)   # the refinement stage through the front end (spec §9 G-l)
function VgtkCore.__init(prob::VgtkCore.RefineProblem{<:CalibrationData, <:CalibrationParameters, VgtkCore.NullParameters}, alg::RadiometricRefinement;
                         stops::VgtkCore.LmStops = _calibration_stops(), callback = nothing)
    _scheduled_radiometry(alg, prob.f.calibration_settings)   # (a joint solve takes no smoothness)
    return CalibrationCache(prob, alg, stops, callback)
end
VgtkCore.solve!(c::CalibrationCache) = _calibrate!(c.prob.f, c.prob.u0, c.refinement, c.stops, c.callback)

"""The refinement of the started calibration — data `f`, parameters `θ`, refined in place — by `alg` at the stop settings `stops`
with the solve's `callback` (`nothing`: none; [`RadiometricRefinement`](@ref)'s `solve!`)."""
function _calibrate!(f::CalibrationData, θ::CalibrationParameters, alg::RadiometricRefinement, stops::VgtkCore.LmStops, callback)
    f.start === nothing && throw(ArgumentError("RadiometricRefinement: the data have no start (RadiometricCalibration's `init` on a StartGeometry, or load_calibration_start, lift a geometric calibration into one)"))
    res, trace, costs = _run_refinement!(f, θ, f.start, alg, stops, callback)
    # the calibrated camera on output (the fold policy, 2026-11-15): held to the coverage assertion, and a remaining fold flags the
    # calibration non-converged — no calibration is accepted with a folded camera
    folds = calibrated_folds(θ.camera.geometric; device = f.backend)
    folds === nothing || @printf("calibrated camera: folds %d of %d samples over its coverage (%d unimaged)%s\n", length(folds.folded), folds.samples,
                                 folds.unimaged, isempty(folds.folded) ? "" : " — NOT CONVERGED: a folded camera is no calibration")
    # the calibrated focus surface on output: a support node run to t̂ ≤ 0 (focus at infinity) or a non-finite one flags the
    # calibration non-converged, as a fold does (2026-11-15)
    fs = focus(θ)
    runaway = focus_runaway(fs, f.start.in_support, f.start.views0[1].focus)
    isempty(runaway) || @printf("calibrated focus: %d of the %d support nodes ran away (t̂ ≤ 0 or non-finite: %s) — NOT CONVERGED: a focus at infinity is no calibration\n",
                                length(runaway), count(f.start.in_support, fs.nodes),
                                join([@sprintf("node %d at (%.0f, %.0f) px, t̂ %.3g", i, fs.nodes[i]..., fs.t̂_i[i]) for i in runaway], "; "))
    code, rule = validity_gate(fold_termination(costs.termination..., folds)..., !isempty(runaway))
    meta = CalibrationMetadata(convert(Vector{NamedTuple}, trace), costs.linear[1], costs.linear[2], costs.final[1], costs.final[2],
                               Float64(costs.seconds), res, copy(f.run.scrubbed), folds, runaway)
    estimate = CalibrationEstimate(calibrated_camera(f, θ), copy(θ.poses), radiometric_target(f, θ), θ.light)
    return VgtkCore.FitResult(estimate, code, rule, VgtkCore.SolveStats(; nsteps = count(t -> t.phase == :lm, trace)), costs.final[2],
                              sum(length, f.ϕ; init = 0), meta)
end

"""
    read_calibration(path::AbstractString; set = ()) -> (run, settings, algorithm, stops)
    read_calibration(table::AbstractDict; set = ()) -> (run, settings, algorithm, stops)

The radiometric calibration a run's TOML states: `run` its [`RunOptions`](@ref) and `settings` its [`CalibrationSettings`](@ref)
([`pipeline_options`](@ref): the capture's facts, the run's I/O and the problem's structure, the data's), `algorithm` the
[`RadiometricCalibration`](@ref) of the [`RadiometricRefinement`](@ref) of its `[optimizer]` table (`type`, a tag of
[`OPTIMIZER_TAGS`](@ref): Core's `LevenbergMarquardt` at Levenberg's damping λ·I) and of its `[sweep]` table (`radiometry`: the
closed-form blocks' algorithm, a tag of [`RADIOMETRY_TAGS`](@ref) — [`JointRadiometry`](@ref) of the optimizer or
[`RadiometryBlockDescent`](@ref); `iterations` the sweeps of their stops, `radiometry_stops`; `lm_chunk`), and `stops` its solve's
stop settings (Core's `LmStops`: `max_iter`, `max_time`, `ftol`, `xtol_rel` of its `[levenberg_marquardt.stops]` table, Core's
absolute step tolerance of record) — `solve(EstimationProblem(CalibrationEstimate, CalibrationPhotographs(geometry, photographs;
run, settings)), algorithm; stops)`. A version-1 config's `[lm]` and a version-3 config's `[optimizer]` stops are the
`[levenberg_marquardt.stops]` table ([`RC_CONFIG_VERSIONS`](@ref)).
"""
read_calibration(path::AbstractString; set = ()) = _read_config_file(read_calibration, path, set)
function read_calibration(d::AbstractDict; set = ())
    run, settings = pipeline_options(d; set)
    d = _current_config(d, set)
    t = _read_table(OptimizerTable, _pop_table!(d, _table_name(OptimizerTable)), _table_name(OptimizerTable))
    st = _read_table(LevenbergMarquardtStopsTable, _pop_table!(d, _table_name(LevenbergMarquardtStopsTable)), _table_name(LevenbergMarquardtStopsTable))
    sw = _read_table(SweepTable, _pop_table!(d, _table_name(SweepTable)), _table_name(SweepTable))
    optimizer = VgtkCore.LevenbergMarquardt(; damping = VgtkCore.LevenbergDamping())   # the optimizer `t.type` names (one tag, OPTIMIZER_TAGS)
    radiometry = _radiometry_algorithm(sw.radiometry, optimizer)   # the algorithm `sw.radiometry` names (RADIOMETRY_TAGS)
    algorithm = RadiometricCalibration(RadiometricRefinement(; optimizer, radiometry, radiometry_stops = _radiometry_stops(sw.iterations), sw.lm_chunk))
    _scheduled_radiometry(algorithm.refinement, settings)          # a joint solve with a smoothness refused, naming the keys
    return run, settings, algorithm, VgtkCore.LmStops(; max_iter = st.max_iter, max_time = st.max_time, ftol = st.ftol, xtol_rel = st.xtol_rel, xtol = CORE_LM_XTOL)
end

function config_dict(run::RunOptions, settings::CalibrationSettings, algorithm::RadiometricCalibration, stops::VgtkCore.LmStops)
    r = algorithm.refinement
    d = config_dict(run, settings)
    d["optimizer"] = toml_table(OptimizerTable(; type = VgtkCore.LevenbergMarquardt))
    d["levenberg_marquardt"] = Dict{String, Any}("stops" => toml_table(LevenbergMarquardtStopsTable(; stops.max_iter, stops.max_time, stops.ftol, stops.xtol_rel)))
    d["sweep"] = toml_table(SweepTable(; r.lm_chunk, iterations = r.radiometry_stops.max_iter, radiometry = _radiometry_type(r.radiometry)))
    return d
end
# the closed-form blocks' algorithm of its tag's type (`[sweep] radiometry`), and the type of an algorithm (its tag's)
_radiometry_algorithm(::Type{JointRadiometry}, optimizer::VgtkCore.LevenbergMarquardt) = JointRadiometry(optimizer)
_radiometry_algorithm(::Type{RadiometryBlockDescent}, ::VgtkCore.LevenbergMarquardt) = RadiometryBlockDescent()
_radiometry_type(::JointRadiometry) = JointRadiometry
_radiometry_type(::RadiometryBlockDescent) = RadiometryBlockDescent
write_pipeline_config(dir::AbstractString, run::RunOptions, settings::CalibrationSettings, algorithm::RadiometricCalibration, stops::VgtkCore.LmStops;
                      header::AbstractString = "") =
    _write_pipeline_config(dir, config_dict(run, settings, algorithm, stops), run; header)

"""
    _run_refinement!(f, θ, meta, alg, stops, callback) -> (result, trace, (; linear, final, seconds, termination))

The Levenberg–Marquardt over Θ̄ (`θ`, in place) on the data `f` (the structure `f.calibration_settings`) from the prepared start
`meta` (a resumed one's state and cost, `meta.resumed`: [`load_calibration_state!`](@ref)) by `alg`'s optimizer at the solve's
`stops`: runs `stops.max_iter` iterations (in chunks of `alg.lm_chunk` with a closed-form sweep between them when set), prints the
live monitor after every iteration and calls `callback` (Core's `callback(state::LmIterationState) -> Bool`, `nothing`: none) with
the refinement's state: its iteration across the chunks, the iterate's estimate decoded (`(; camera, poses, mat, surface, light)`),
its linear solves and seconds so far, the chunk's cost, best cost, damping, scaling and `trial_ready`; `true` stops the refinement.
Returns the LM's result (the last chunk's; `nothing` without an iteration), the trace, and the costs after the linear solve (a
resumed start's: at the resumed state) and at the end, the LM's seconds and its termination record (`lm_termination`:
VisualGeometryToolkitCore's verdict at the final state, the one written).
"""
function _run_refinement!(f::CalibrationData, θ::CalibrationParameters, meta, alg::RadiometricRefinement, stops::VgtkCore.LmStops, callback)
    settings = f.calibration_settings
    names = meta.names; K = length(names); W₀ = meta.W₀; H₀ = meta.H₀; fpx = meta.fpx
    lens0 = meta.lens0; route = lm_route(lens0); views0 = meta.views0; features = meta.features
    c1, rms1 = hasproperty(meta, :resumed) ? meta.resumed.costs : meta.costs.linear   # a resumed start's cost: at the resumed state
    ϕ² = data_norm2(f)
    probe = [SVector(x * W₀, y * H₀) for x in 0.05:0.05:0.95, y in 0.05:0.05:0.95]   # a 19 × 19 probe grid of the image (pixels)
    lens_init = geometric(θ)
    rays_init = [normalize(backproject(lens_init, p)) for p in probe]   # the start camera's probe rays, once (the drift monitor)
    pinned_material = "material" in settings.fixed
    damping = alg.optimizer.damping
    lp = refinement_structure(settings, route; smoothness = meta.grid_smoothness)   # THE problem: its blocks and holds (a parametric start has no grid smoothness)
    pinned_material && lp.blocks.material && println("material Θ_m FIXED (\"material\" in fixed): its columns are zero; the board warping / in-plane print error of the block stay free")
    function monitor(it, cst, camera, poses, mat, surf, light)          # the live monitor: one line per iteration (the algorithm's printed output)
        zs = settings.warping.degree > 0 && it > 0 ? board_height(_board(f, θ), surf.h, HEIGHT_NODES) : nothing   # the surface at the nodes, one field
        settings.warping.degree > 0 && it > 0 && @printf("      board warping: TPS nodal heights %s mm (rms change from the start %.3f mm; surface at the nodes %+.2f … %+.2f mm)\n",
                                                   join([@sprintf("%+.2f", _mm(c)) for c in surf.h], " "), _mm(sqrt(sum(abs2, surf.h .- meta.height0) / HEIGHT_NC)),
                                                   _mm(minimum(zs)), _mm(maximum(zs)))
        _inplane_modelled(settings, surf) && it > 0 &&
            @printf("      in-plane print error: a %+.4e (x by %+.4f %%, y by %+.4f %%)  shear u %+.4e (x += u·y in the placed chart, y up)\n",
                    surf.inplane[1], 100 * (exp(surf.inplane[1]) - 1), 100 * (exp(-surf.inplane[1]) - 1), surf.inplane[2])
        feats = _board_features(meta, surf, features)   # the board features follow the refined warping and print error
        lens = camera.geometric
        dr = [angle_between(normalize(backproject(lens, p)), r0) for (p, r0) in zip(probe, rays_init)]   # radians; NaN where a lens images no ray
        drift = any(isfinite, dr) ? median(filter(isfinite, dr)) : NaN
        it > 0 && _print_camera_parameters(route, lens)
        be = detection_bearings(lens, poses, feats, fpx)
        fv = _support_focus_values(camera.focus, meta.in_support)   # the focus nodes in the run's support (fixed at the start)
        @printf("it %3d  rel-RMS %.4f  model drift %.4f°  bearing(detections, own poses) median %.3f px  r %.3f mm  focus 1/t̂ mean over the support nodes %.2f m [%.2f, %.2f]\n",
                it, sqrt(2cst / ϕ²), rad2deg(drift), isempty(be) ? NaN : median(be), _mm(camera.aperture_radius), 1 / mean(fv), 1 / maximum(fv), 1 / max(minimum(fv), 1e-9)); flush(stdout)
        return nothing
    end
    monitor(0, c1, θ.camera, θ.poses, θ.mat, board_surface(θ), θ.light)
    blocks_held_r = !lp.blocks.aperture
    blocks_held_r && println("aperture FIXED at r₀ (\"aperture\" in fixed): the block is removed from the solve (a clamped free block rejects every LM step)")
    blocks = lp.blocks                                           # true: the block is solved
    inplane_free = _surface_free(settings, :inplane)                 # the in-plane print error (a, u): two dual columns of the material block
    @printf("free blocks: %s; %s; %d iterations max%s%s\n",
            join([String(k) == "grid" ? _lens_block_label(route, lens0) : String(k) for k in keys(blocks) if getfield(blocks, k)], ","),
            !blocks_held_r ? "r free (log chart, no box)" : @sprintf("r fixed at %.3f mm", _mm(meta.R0)), stops.max_iter,
            settings.warping.degree > 0 ? "; board warping h ($(HEIGHT_NFREE) free nodal-height parameters)" : height_on(board_surface(θ)) ? "; board warping FIXED" : "",
            inplane_free ? "; in-plane print error (a, u: Core's Aspect ∘ Skew) free" : _inplane_modelled(settings, board_surface(θ)) ? "; in-plane print error FIXED" : ""); flush(stdout)
    trace = NamedTuple[]; t2 = time()
    it_done = Ref(0)                                             # the refinement's iterations before the current chunk
    solves_done = Ref(0); chunk_solves = Ref(0)                  # its linear solves before the chunk, the chunk's so far
    stopped = Ref(false)                                         # the callback asked to stop
    function on_iteration(state)                                 # a chunk's iteration, its iterate decoded (`_lm_callback`)
        it = it_done[] + state.iteration
        u = state.u
        monitor(it, state.cost, u.camera, u.poses, u.mat, u.surface, u.light)
        chunk_solves[] = state.linear_solves
        callback === nothing && return false
        stop = callback(VgtkCore.LmIterationState(it, state.cost, state.best_cost, u, state.damping, solves_done[] + state.linear_solves,
                                                  state.scaling, state.trial_ready, time() - t2))::Bool
        stop && (stopped[] = true)
        return stop
    end
    # the problem of every chunk (`build_problem` / `build_problem_grid`'s keywords)
    problem_kw = (; lp.kw..., damping)
    function run_chunk(n)                                    # n LM iterations from the current state (the problem is rebuilt per chunk, λ restarts)
        chunk_solves[] = 0
        r = _lm_route!(f, θ, route; stops = _chunk_stops(stops, n, max(stops.max_time - (time() - t2), 1.0)), trace,
                       trace_meta = (; ϕ²), callback = on_iteration, problem_kw...)
        solves_done[] += chunk_solves[]
        return r
    end
    res = if alg.lm_chunk <= 0
        run_chunk(stops.max_iter)
    else                                                     # LM chunks with the closed-form (lighting, vignetting, μ) sweep between them
        r_last = nothing
        while it_done[] < stops.max_iter && time() - t2 < stops.max_time
            n_before = length(trace)
            r_last = run_chunk(min(alg.lm_chunk, stops.max_iter - it_done[]))
            it_done[] += length(trace) - n_before
            length(trace) == n_before && break               # the LM did not iterate (converged / out of time)
            stopped[] && break                               # the callback stopped the refinement
            ts = time(); cb, rmsb = cost(f, θ); camera_b, light_b, mat_b = θ.camera, θ.light, θ.mat
            _update_radiometry!(f, θ, _scheduled_radiometry(alg, settings),
                                (; light = blocks.lighting, vignetting = blocks.vignetting, mu = blocks.material); stops = alg.radiometry_stops,
                                gcv_record = meta.vignetting_gcv)
            ca, rmsa = cost(f, θ)
            if ca < cb
                @printf("      sweep after it %d (%.0f s): cost %.4e → %.4e, rel-RMS %.4f → %.4f; μ %.4f; V(31/48/58°) %s → %s\n", it_done[], time() - ts, cb, ca, rmsb, rmsa, θ.mat.diffuse_weight, _Vprof(camera_b.vignetting), _Vprof(θ.camera.vignetting))
                push!(trace, (; ϕ², phase = :sweep, cost = ca, time = time(), snap = Θ̄_snapshot(f, θ)))
            else
                @printf("      sweep after it %d (%.0f s) REJECTED: cost %.4e ≥ %.4e — state kept\n", it_done[], time() - ts, ca, cb)
                θ.camera = camera_b; θ.light = light_b; θ.mat = mat_b
            end
            flush(stdout)
        end
        r_last
    end
    c2, rms2 = cost(f, θ)
    @printf("%s LM: %.0f s, %d its%s; rel-RMS %.4f → %.4f; stopped: %s\n", _route_label(route), time() - t2, count(t -> t.phase == :lm, trace),
            alg.lm_chunk > 0 ? @sprintf(" + %d sweeps", count(t -> t.phase == :sweep, trace)) : "", rms1, rms2, res === nothing ? "no LM iteration" : string(res.retcode, " (", res.stop_rule, ")")); flush(stdout)
    length(trace) >= 3 && @printf("steady-state ≈ %.1f s/it\n", median(diff([t.time for t in trace])[2:end]))
    termination = lm_termination(res)
    return res, trace, (; linear = (c1, rms1), final = (c2, rms2), seconds = time() - t2, termination)
end

"""
    calibration_checkpoints(prob, run) -> callback

The checkpoints and the optimizer's log of a run of the refinement of `prob` (a started `RefineProblem(f, u0)`:
[`compute_calibration_start`](@ref)'s, [`load_calibration_start`](@ref)'s or [`load_calibration_state!`](@ref)'s) in the
outputs of `run` (its [`RunOptions`](@ref)): the `callback` its driver hands the solve, `solve(prob, refinement; stops, callback)`
(ruling R1: a run's I/O is the driver's, never the algorithm's or the problem's). After the refinement's iteration `it`
(`state.iteration`, numbered from `resume.iteration_offset`: a resumed run continues its numbering) it writes, into `io.out`, the
camera's checkpoint `lens_ckpt.json` | `grid_ckpt.json` every 5 iterations; with `io.checkpoint_each` `lens_it<N>.json` |
`grid_it<N>.json`, `poses_it<N>.json` and `state_it<N>.json` (the iteration's cost and rel-RMS in its provenance) after every one;
with `io.optimizer_log` a line (tab-separated) appended to that file: the iteration, the cost, the rel-RMS (√(2 cost / ϕ²)), the
best cost before the iteration, λ after its update, the linear solves of the iteration (1 = accepted at once; each further solve is
a rejected step that raised λ), the DᵀD scaling's min / max / number of entries at its floor (1e-8 × max; empty under Levenberg's
damping) and the time. It never stops the solve.
"""
function calibration_checkpoints(prob::VgtkCore.RefineProblem{<:CalibrationData, <:CalibrationParameters}, run::RunOptions)
    f, θ = prob.f, prob.u0
    meta = f.start
    meta === nothing && throw(ArgumentError("calibration_checkpoints: the data have no start (compute_calibration_start, load_calibration_start or load_calibration_state! build one)"))
    out = run.io.out
    isempty(out) && throw(ArgumentError("calibration_checkpoints: io.out required (the run's checkpoints are written there)"))
    mkpath(out)
    route = lm_route(meta.lens0); fname = _camera_file(route); tag = _frontend_tag(run); names = meta.names
    offset = run.resume.iteration_offset; each = run.io.checkpoint_each; log = run.io.optimizer_log
    ϕ² = data_norm2(f)
    prev = Ref(0)                                                # the linear solves at the previous iteration
    return function (state)
        it = state.iteration + offset
        (; camera, poses, mat, surface, light) = state.u
        lens = camera.geometric
        it % 5 == 0 && _save_run_camera(route, joinpath(out, "$(fname)_ckpt.json"), lens, meta.init_region;
                                        provenance = (; model = "radiometric_$(_model_label(route, lens))_$(tag)_ckpt", iteration = it))
        if each                                                  # per-iteration camera, poses + state: the learned-pose evaluations and renders between iterations
            _save_run_camera(route, joinpath(out, "$(fname)_it$(it).json"), lens, meta.init_region;
                             provenance = (; model = "radiometric_$(_model_label(route, lens))_$(tag)_it$(it)", iteration = it))
            save_poses(joinpath(out, "poses_it$(it).json"), poses, names; world = meta.world)
            save_state(joinpath(out, "state_it$(it).json"),
                       (; camera, target = RadiometricTarget(_geometric_target(f.board.target), surface.h, warping_aspect(_board(f, θ)), TargetRadiometry(mat, surface); inplane = surface.inplane),
                          poses, names, light, settings = f.settings, observation = observation_model(f), page_flip = meta.page_flip);
                       provenance = (; iteration = it, cost = state.cost, rel_rms = sqrt(2state.cost / ϕ²)))   # the iteration's cost, as the monitor's line
        end
        if !isempty(log)
            sc = state.scaling
            open(log, "a") do io
                println(io, join((it, state.cost, sqrt(2 * state.cost / ϕ²), state.best_cost, state.damping, state.linear_solves - prev[],
                                  sc === nothing ? "" : minimum(sc), sc === nothing ? "" : maximum(sc),
                                  sc === nothing ? "" : count(<=(1e-8 * maximum(sc) * (1 + 1e-12)), sc), time()), '\t'))
            end
        end
        prev[] = state.linear_solves
        return false
    end
end

"""
    load_calibration_state!(prob, refinement, path; sweep = false, load_state = RadiometricCamera.load_state) -> prob

A resume: the started refinement problem `prob` (`RefineProblem(f, u0)`: [`compute_calibration_start`](@ref)'s or
[`load_calibration_start`](@ref)'s) continued from the finished state `path` (a `state.json`, [`load_state`](@ref)), in place: the
state's poses, camera, target and — when their sizes and geometry are the run's — vignetting and lighting loaded into `u0`
(`load_state_into!`, its observation model the run's: the start's `camera.sensor_model` decides whether a record present is used),
the vignetting's dependent points tied to its rule and its gauge `V(𝐜) = 1` set, the printed edge's start set when the run models the
edge and the state carries none. Then the closed-form sweep of the blocks the settings leave free (lighting, vignetting, μ) by
`refinement`'s radiometry at its `radiometry_stops` (one sweep at least), when the driver asks for it (`sweep`, bin/refine.jl's
`resume.sweep`) and always when the state's vignetting is not the run's (V flat otherwise: no start of it). Returns `prob`, refined
from there by `solve(prob, refinement; stops, callback)` (ruling R1: a resume is a problem built from the loaded state; its
iteration numbering is the driver's, `resume.iteration_offset` in [`calibration_checkpoints`](@ref)). The data record the resume
(`f.start.resumed`: the state's path, whether its vignetting was loaded, the cost at the resumed — and swept — state); the sweep's
GCV selections join the start's.
"""
function load_calibration_state!(prob::VgtkCore.RefineProblem{<:CalibrationData, <:CalibrationParameters}, refinement::RadiometricRefinement,
                                 path::AbstractString; sweep::Bool = false, load_state = RadiometricCamera.load_state)
    f, θ = prob.f, prob.u0
    meta = f.start
    meta === nothing && throw(ArgumentError("load_calibration_state!: the data have no start (compute_calibration_start or load_calibration_start build one)"))
    settings = f.calibration_settings
    sst = load_state(path)
    smode = sensor_model_mode(meta.sensor_model.mode)                  # the start's `camera.sensor_model`
    # a resumed state's record present and not used, the run naming none (else the run's resolution warned): the warning, here
    smode isa OffSensorModel && sst.observation.sensor isa SensorRecord && meta.sensor_model.present === nothing &&
        warn_sensor_model_off(_record_line(sst.observation.sensor))
    vig_ok, _ = load_state_into!(f, θ, sst, meta.names; sensor_model = smode)
    set_vignetting!(θ, tie_pole(θ.camera.vignetting, f.vig_rule))   # a state of the untied radial profile (before 2026-10-29): P_1 := P_0, V(𝐜) kept; the rule's dependent points set
    set_vignetting!(θ, gauge_center_value(θ.camera.vignetting))   # the gauge's start: V(𝐜) = 1 exactly
    if !isempty(settings.edge.sdf) && iszero(board_surface(θ).edge)                 # resumed without the edge model: start it (the print's)
        set_surface!(θ, with_edge(board_surface(θ), collect(_edge_start(settings))))
    end
    c1, rms1 = cost(f, θ)
    sv = sst.camera.vignetting
    @printf("resumed from %s: rel-RMS %.4f%s\n", path, rms1,
            vig_ok ? "" : "; the state's vignetting ($(nameof(typeof(sv))), $(join(size(control_points(sv)), " × "))$(sv isa RadialVignetting ? ", r_max $(round(VrtkCore.field_radius(sv); digits = 1)) px" : "")) is not the run's ($(meta.vig_desc)) → V flat"); flush(stdout)
    if !vig_ok || sweep                                         # the closed-form start of the linear blocks on the resumed geometry (and material)
        tsw = time()
        rs = refinement.radiometry_stops                        # at least one sweep on the resumed geometry
        _update_radiometry!(f, θ, _scheduled_radiometry(refinement, settings),
                            (; light = !("lighting" in settings.fixed), vignetting = !("vignetting" in settings.fixed), mu = !("material" in settings.fixed));
                            stops = VgtkCore.ToleranceStops(; max_iter = max(rs.max_iter, 1), rs.rtol, rs.atol), gcv_record = meta.vignetting_gcv)
        c1, rms1 = cost(f, θ)
        @printf("closed-form sweep on the resumed geometry (%.0f s): rel-RMS %.4f; μ %.4f; V(31/48/58°) %s\n", time() - tsw, rms1, θ.mat.diffuse_weight, _Vprof(θ.camera.vignetting)); flush(stdout)
    end
    f.start = (; meta..., resumed = (; state = String(path), vignetting = vig_ok, costs = (c1, rms1)))
    return prob
end
