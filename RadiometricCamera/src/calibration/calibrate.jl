# ---------------------------------------------------------------------------
# The refinement of a radiometric calibration (paper §Second Order Optimization): Levenberg–Marquardt over every block of the started
# calibration's parameters on its data (`RadiometricRefinement`'s solve of `RefineProblem(f, u0, run)`, `RadiometricCalibration`'s after its
# start: the route's LM, `_lm_views!` per view or `_lm_tiles!` per tile, by the camera's type; in chunks with a
# closed-form sweep between them, with the live monitor and the per-iteration checkpoints), its result a Core FitResult whose estimate
# is the `CalibrationEstimate` (the calibrated camera, the poses, the radiometric target, the light); `calibrate` = its solve.
# ---------------------------------------------------------------------------

"""
    CalibrationEstimate(camera, poses, target, light)

The estimate of a radiometric calibration ([`RadiometricCalibration`](@ref), [`calibrate`](@ref)): the calibrated camera (a
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
`FitResult` — its termination record, `stats.nsteps`, `metadata.cost` — `nothing` without an iteration), the non-finite Jacobian entries replaced, by block
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
    RadiometricRefinement(settings::CalibrationSettings; optimizer = LevenbergMarquardt(; damping = LevenbergDamping()),
                          radiometry = RadiometryBlockDescent(; sweeps), lm_chunk, log = "")

The refinement of a radiometric calibration (paper §Second Order Optimization), a refine (`RefineKind`): it solves
`RefineProblem(f, u0, run)` of the calibration's data `f` (a [`CalibrationData`](@ref)) from the parameters `u0` (a
[`CalibrationParameters`](@ref): refined in place, the result's estimate built from them; the start's record in `f.start`,
`_start_state`'s, which [`RadiometricCalibration`](@ref)'s `init` or [`load_calibration_start`](@ref) lift a geometric
calibration into) in the run `run` (its [`RunOptions`](@ref), the problem's context: the outputs, the resume, the capture's facts).
`settings` the calibration's ([`CalibrationSettings`](@ref), [`pipeline_options`](@ref)), `optimizer` the refinement's (VisualGeometryToolkitCore's
`LevenbergMarquardt`; its damping Levenberg's λ·I by default, which converged on the origins of a non-central lens where λ·DᵀD
stalled at 1.2e-8 of its start cost, and reaches a slightly lower cost 1.6–1.8× faster on the Tamron and grid cases), `radiometry`
the closed-form blocks' solve ([`RadiometryBlockDescent`](@ref), or [`JointRadiometry`](@ref) at the optimizer's damping:
`[sweep] joint`; its `sweeps` those of a solve between LM chunks, `[sweep] iterations`; the start's closed-form solve one sweep of
it), `lm_chunk` the LM iterations between two such solves (`[sweep] lm_chunk`; 0: the LM in one run), the defaults
configs/default.toml's `[sweep]`; `log` a file the LM appends a line per iteration to (empty: none). The stop settings are the
solve's: `solve(problem, refinement; stops = LmStops(…))` (and `init`'s), Core's `LmStops` (`max_iter`, `max_time`, `ftol`,
`xtol_rel`, `xtol`; the defaults configs/default.toml's `[optimizer]`).

The solve is the Levenberg–Marquardt over every block the settings leave free (`fixed`), on the route the camera's type picks
(`lm_route`: per view, in a parametric camera's own parameters; per tile for a SplineLens camera), optionally in chunks with a
closed-form sweep of the lighting, the vignetting and μ between them (`sweep.lm_chunk`), up to the stops' `max_iter` iterations and
`max_time` seconds. The result is a `FitResult{CalibrationEstimate, CalibrationMetadata}`: `u` the refined
[`CalibrationEstimate`](@ref); `retcode` / `stop_rule` the LM's termination record (VisualGeometryToolkitCore's: a tolerance stop
converged, the rejection cap judged by the stall postcondition at the final state) — `(Failure, ShapeGate)` when the calibrated
SplineLens camera still folds over its coverage (`fold_termination`: no calibration; the camera is held to the coverage assertion
too, [`calibrated_folds`](@ref)) or a focus node of its support runs to t̂ ≤ 0 (`focus_runaway`, the same `validity_gate`);
`stats.nsteps` the LM iterations; `rms` the final relative RMS over the `n` measured values.
"""
struct RadiometricRefinement{O <: VgtkCore.LevenbergMarquardt, R <: RadiometrySolve} <: VgtkCore.AbstractEstimationAlgorithm
    settings::CalibrationSettings
    optimizer::O
    radiometry::R
    lm_chunk::Int
    log::String
end
function RadiometricRefinement(settings::CalibrationSettings; optimizer::VgtkCore.LevenbergMarquardt = VgtkCore.LevenbergMarquardt(; damping = VgtkCore.LevenbergDamping()),
                               radiometry::RadiometrySolve = RadiometryBlockDescent(; sweeps = SweepTable().iterations),
                               lm_chunk::Integer = SweepTable().lm_chunk, log::AbstractString = "")
    lm_chunk >= 0 || throw(ArgumentError("RadiometricRefinement: lm_chunk ≥ 0 (0: no closed-form solve between LM chunks); got $lm_chunk"))
    _with_schedule(radiometry; vignetting_smoothness = smoothness_weight(settings.vignetting.smoothness))   # (a joint solve takes no smoothness)
    return RadiometricRefinement(settings, optimizer, radiometry, Int(lm_chunk), String(log))
end
VgtkCore.estimator_kind(::RadiometricRefinement) = VgtkCore.RefineKind()

"""
    RadiometricCalibration(refinement::RadiometricRefinement)

The radiometric calibration of the paper, a fit (`FitKind`: it seeds itself, from the geometric calibration): the START, which
lifts the geometric camera into a radiometric calibration's data and parameters with the run's start options
(`start_options(run, refinement.settings)`, the run's output directory kept), then the [`RadiometricRefinement`](@ref) `refinement` of
them in the run. A run's TOML gives the run, it and the solve's stops, [`read_calibration`](@ref). Solved on
`EstimationProblem(CalibrationEstimate, (geometry, photographs), (; run[, device]))` (`run` the run's [`RunOptions`](@ref), `device`
the compute backend, [`default_backend`](@ref) by default): a [`StartGeometry`](@ref) and the photographs' files, `photographs[k]` the file of view `k`
(a RAW photograph by its file, its green photosites, or a decoded image). The start: the focus surface through the mean target
centre, the aperture from the f-number, the image circle measured where the run names its frames, the masks, the photographs'
observations streamed, the vignetting and the lighting by the closed-form start, the grid smoothness.

`init` returns the start: its cache is the refinement stage's `init(RefineProblem(f, u0, run), refinement; stops)` (an `EstimationCache` of
a [`CalibrationCache`](@ref); its `prob`, the started `RefineProblem(f, u0, run)`); `solve!` of it, as `solve`, the refinement — its result
[`RadiometricRefinement`](@ref)'s, recorded by the front end for the stage and relabelled for the calibration (spec §9 G-l). A started calibration (`load_calibration_start`'s, or a caller's
`RefineProblem(f, u0, run)`) is solved by the refinement itself: `solve(RefineProblem(f, u0, run), refinement; stops)`.

```julia
run, alg, stops = read_calibration("run.toml")
c = init(EstimationProblem(CalibrationEstimate, (geometry, photographs), (; run)), alg; stops)   # the start
r = solve!(c)                                                                           # the refinement
```

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
[`RadiometricCalibration`](@ref)'s `init` returns that refinement stage's `EstimationCache` after its start): the refinement's problem `prob`, `RefineProblem(f, u0, run)` of the calibration's data `f`
([`CalibrationData`](@ref): its `start` holds what the refinement and its records read — the start views, the observations, the
costs of the start), its parameters `u0` ([`CalibrationParameters`](@ref)) and the run `run` ([`RunOptions`](@ref)), the `refinement` and
the solve's `stops`. `solve!`
refines `prob.u0` in place.
"""
mutable struct CalibrationCache{P <: VgtkCore.RefineProblem{<:CalibrationData, <:CalibrationParameters, RunOptions}, A <: RadiometricRefinement}
    prob::P
    refinement::A
    stops::VgtkCore.LmStops
end

"""
    _calibration_stops(; max_iter, ftol, xtol, xtol_rel, stationarity_tol, max_time, residual_resolution) -> LmStops

The stop settings of a calibration's solve when it names none: configs/default.toml's `[optimizer]` (`read_calibration` of an
empty config), each keyword given replacing its value (the package's tests and benchmarks state an iteration cap so).
"""
function _calibration_stops(; kw...)
    s = last(read_calibration(Dict{String, Any}(VgtkCore.CONFIG_VERSION_KEY => VgtkCore.current_config_version(RC_CONFIG_VERSIONS))))
    return VgtkCore.LmStops(; s.max_iter, s.ftol, s.xtol, s.xtol_rel, s.stationarity_tol, s.max_time, s.residual_resolution, kw...)
end

# the context of a calibration problem: the run (`(; run)`) and the compute backend (`(; run, device)`; the default backend without one)
_calibration_device(::NamedTuple{(:run,)}) = default_backend()
_calibration_device(p::NamedTuple{(:run, :device)}) = p.device

function VgtkCore.__init(prob::VgtkCore.EstimationProblem{CalibrationEstimate, <:Tuple{StartGeometry, AbstractVector{<:AbstractString}},
                                                          <:Union{NamedTuple{(:run,), <:Tuple{RunOptions}}, NamedTuple{(:run, :device), <:Tuple{RunOptions, Any}}}},
                         alg::RadiometricCalibration; stops::VgtkCore.LmStops = _calibration_stops())
    geometry, photographs = prob.f
    run = prob.p.run
    refinement = alg.refinement
    f, θ = _start_state(geometry, photographs, _start_run_options(run, refinement.settings)..., nothing; backend = _calibration_device(prob.p),
                        refinement.radiometry)
    return VgtkCore.init(VgtkCore.RefineProblem(f, θ, run), refinement; stops)   # the refinement stage through the front end (spec §9 G-l)
end
VgtkCore.__init(prob::VgtkCore.RefineProblem{<:CalibrationData, <:CalibrationParameters, RunOptions}, alg::RadiometricRefinement;
                stops::VgtkCore.LmStops = _calibration_stops()) =
    CalibrationCache(prob, alg, stops)
VgtkCore.solve!(c::CalibrationCache) = _calibrate!(c.prob.f, c.prob.u0, c.prob.p, c.refinement, c.stops)

"""The refinement of the started calibration — data `f`, parameters `θ`, refined in place, in the run `run` — by `alg` at the stop settings `stops`
([`RadiometricRefinement`](@ref)'s `solve!`); `load_state` the resume's state reader (a test's hook)."""
function _calibrate!(f::CalibrationData, θ::CalibrationParameters, run::RunOptions, alg::RadiometricRefinement, stops::VgtkCore.LmStops; load_state = RadiometricCamera.load_state)
    f.start === nothing && throw(ArgumentError("RadiometricRefinement: the data have no start (RadiometricCalibration's `init` on a StartGeometry, or load_calibration_start, lift a geometric calibration into one)"))
    res, trace, costs = _run_refinement!(f, θ, f.start, run, alg, stops; load_state)
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
    read_calibration(path::AbstractString; set = ()) -> (run, algorithm, stops)
    read_calibration(table::AbstractDict; set = ()) -> (run, algorithm, stops)

The radiometric calibration a run's TOML states: `run` its [`RunOptions`](@ref) (the problem's context), `algorithm` the
[`RadiometricCalibration`](@ref) of the [`RadiometricRefinement`](@ref) of its [`CalibrationSettings`](@ref) ([`pipeline_options`](@ref)), of its `[optimizer]` table (`type`, a tag of [`OPTIMIZER_TAGS`](@ref): Core's
`LevenbergMarquardt` at Levenberg's damping λ·I; `log`, its per-iteration log) and of its `[sweep]` table (`joint`: the closed-form
blocks by [`JointRadiometry`](@ref), else [`RadiometryBlockDescent`](@ref), of `iterations` sweeps; `lm_chunk`), and `stops` its solve's stop settings (Core's
`LmStops`: `max_iter`, `max_time`, `ftol`, `xtol_rel` of the table, Core's absolute step tolerance of record) —
`solve(EstimationProblem(CalibrationEstimate, (geometry, photographs), (; run)), algorithm; stops)`. A version-1 config's `[lm]` is the `[optimizer]` table ([`RC_CONFIG_VERSIONS`](@ref)).
"""
read_calibration(path::AbstractString; set = ()) = _read_config_file(read_calibration, path, set)
function read_calibration(d::AbstractDict; set = ())
    run, settings = pipeline_options(d; set)
    d = _current_config(d, set)
    t = _read_table(OptimizerTable, get(d, "optimizer", Dict{String, Any}()), "optimizer")
    sw = _read_table(SweepTable, get(d, "sweep", Dict{String, Any}()), "sweep")
    optimizer = VgtkCore.LevenbergMarquardt(; damping = VgtkCore.LevenbergDamping())   # the optimizer `t.type` names (one tag, OPTIMIZER_TAGS)
    radiometry = sw.joint ? JointRadiometry(; optimizer.damping, sweeps = sw.iterations) : RadiometryBlockDescent(; sweeps = sw.iterations)
    algorithm = RadiometricCalibration(RadiometricRefinement(settings; optimizer, radiometry, sw.lm_chunk, log = t.log))
    return run, algorithm, VgtkCore.LmStops(; max_iter = t.max_iter, max_time = t.max_time, ftol = t.ftol, xtol_rel = t.xtol_rel, xtol = CORE_LM_XTOL)
end

function config_dict(run::RunOptions, algorithm::RadiometricCalibration, stops::VgtkCore.LmStops)
    r = algorithm.refinement
    d = config_dict(run, r.settings)
    d["optimizer"] = toml_table(OptimizerTable(; type = VgtkCore.LevenbergMarquardt, stops.max_iter, stops.max_time, stops.ftol, stops.xtol_rel,
                                               r.log))
    d["sweep"] = toml_table(SweepTable(; r.lm_chunk, iterations = r.radiometry.sweeps, joint = r.radiometry isa JointRadiometry))
    return d
end
write_pipeline_config(dir::AbstractString, run::RunOptions, algorithm::RadiometricCalibration, stops::VgtkCore.LmStops; header::AbstractString = "") =
    _write_pipeline_config(dir, config_dict(run, algorithm, stops), run; header)

"""
    calibrate(geometry::StartGeometry, photographs, run::RunOptions, algorithm::RadiometricCalibration; stops, device = default_backend())
        -> FitResult{CalibrationEstimate, CalibrationMetadata}

A radiometric calibration in one call (a task verb): `solve(EstimationProblem(CalibrationEstimate, (geometry, photographs),
(; run, device)), algorithm; stops)` of the run `run` — the start, then its refinement ([`RadiometricCalibration`](@ref); `stops` its solve's,
configs/default.toml's `[optimizer]` by default).
The geometric calibration is data: a [`StartGeometry`](@ref)`(camera, poses, observations, target; names)`. The data and
parameters refined are the ones a start record rebuilds (`save_calibration_start` → `load_calibration_start`, bin/refine.jl's path): the start
reads no refinement-only key (`REFINE_ONLY_KEYS`), and a record carries the start's values bit for bit.
"""
calibrate(geometry::StartGeometry, photographs::AbstractVector{<:AbstractString}, run::RunOptions, algorithm::RadiometricCalibration;
          stops::VgtkCore.LmStops = _calibration_stops(), device = default_backend()) =
    VgtkCore.solve(VgtkCore.EstimationProblem(CalibrationEstimate, (geometry, photographs), (; run, device)), algorithm; stops)
"""The start's options of the run `run` and the settings `settings` (`start_options`: the refinement-only keys at their defaults), written
to the run's `io.out`: `(run, settings)`."""
_start_run_options(run::RunOptions, settings::CalibrationSettings) =
    (d = config_dict(start_options(run, settings)...); pipeline_options(merge(d, Dict{String, Any}("io" => merge(d["io"], Dict{String, Any}("out" => run.io.out))))))

"""
    _run_refinement!(f, θ, meta, run, alg, stops; load_state = RadiometricCamera.load_state) -> (result, trace, (; linear, final, seconds, termination))

The Levenberg–Marquardt over Θ̄ (`θ`, in place) on the data `f` from the prepared start in the run `run` (`alg`'s optimizer at the solve's `stops`): resumes from `resume.state` when given, runs `stops.max_iter` iterations (in chunks
of `sweep.lm_chunk` with a closed-form sweep between them when set), prints the live monitor after every iteration and writes the checkpoints
(`lens_ckpt.json` / `grid_ckpt.json` every 5 iterations; with `io.checkpoint_each` `lens_it<N>.json` | `grid_it<N>.json`, `poses_it<N>.json`,
`state_it<N>.json` after every iteration). Returns the NLLSsolver result (the last chunk's; `nothing` without an iteration), the trace,
and the costs after the linear solve and at the end, the LM's seconds and its termination record (`lm_termination`: VisualGeometryToolkitCore's
verdict at the final state, the one written).
"""
function _run_refinement!(f::CalibrationData, θ::CalibrationParameters, meta, run::RunOptions, alg::RadiometricRefinement, stops::VgtkCore.LmStops; load_state = RadiometricCamera.load_state)
    settings = alg.settings
    names = meta.names; K = length(names); W₀ = meta.W₀; H₀ = meta.H₀; fpx = meta.fpx
    lens0 = meta.lens0; route = lm_route(lens0); views0 = meta.views0; features = meta.features
    c1, rms1 = meta.costs.linear
    ϕ² = data_norm2(f)
    probe = [SVector(x * W₀, y * H₀) for x in 0.05:0.05:0.95, y in 0.05:0.05:0.95]   # a 19 × 19 probe grid of the image (pixels)
    lens_init = geometric(θ)
    rays_init = [normalize(backproject(lens_init, p)) for p in probe]   # the start camera's probe rays, once (the drift monitor)
    pinned_material = "material" in settings.fixed
    damping = alg.optimizer.damping
    lp = _refinement_lm(settings, route; smoothness = meta.grid_smoothness)   # THE problem: its blocks and holds (a parametric start has no grid smoothness)
    pinned_material && lp.blocks.material && println("material Θ_m FIXED (\"material\" in fixed): its columns are zero; the board warping / in-plane print error of the block stay free")
    if !isempty(run.resume.state)                           # continue from a finished state (views, material, surface, vignetting, light)
        sst = load_state(run.resume.state)
        smode = sensor_model_mode(run.camera.sensor_model)
        # a resumed state's record present and not used, the run naming none (else the run's resolution warned): the warning, here
        smode isa OffSensorModel && sst.observation.sensor isa SensorRecord && (!hasproperty(meta, :sensor_model) || meta.sensor_model.present === nothing) &&
            warn_sensor_model_off(_record_line(sst.observation.sensor))
        vig_ok, light_ok = load_state_into!(f, θ, sst, names; sensor_model = smode)
        set_vignetting!(θ, tie_pole(θ.camera.vignetting, f.vig_rule))   # a state of the untied radial profile (before 2026-10-29): P_1 := P_0, V(𝐜) kept; the rule's dependent points set
        set_vignetting!(θ, gauge_center_value(θ.camera.vignetting))   # the gauge's start: V(𝐜) = 1 exactly
        if !isempty(settings.edge.sdf) && iszero(board_surface(θ).edge)                 # resumed without the edge model: start it (the print's)
            set_surface!(θ, with_edge(board_surface(θ), collect(_edge_start(settings))))
        end
        c1, rms1 = cost(f, θ)
        sv = sst.camera.vignetting
        @printf("resumed from %s at iteration %d: rel-RMS %.4f%s\n", run.resume.state, run.resume.iteration_offset, rms1,
                vig_ok ? "" : "; the state's vignetting ($(nameof(typeof(sv))), $(join(size(control_points(sv)), " × "))$(sv isa RadialVignetting ? ", r_max $(round(VrtkCore.field_radius(sv); digits = 1)) px" : "")) is not the run's ($(meta.vig_desc)) → V flat"); flush(stdout)
        if !vig_ok || run.resume.sweep                      # the closed-form start of the linear blocks on the resumed geometry (and material)
            tsw = time()
            _update_radiometry!(f, θ, _with_schedule(alg.radiometry; sweeps = max(alg.radiometry.sweeps, 1), verbose = true,
                                                  vignetting_smoothness = smoothness_weight(settings.vignetting.smoothness)),
                                (; light = !("lighting" in settings.fixed), vignetting = !("vignetting" in settings.fixed), mu = !pinned_material);
                                gcv_record = meta.vignetting_gcv)
            c1, rms1 = cost(f, θ)
            @printf("closed-form sweep on the resumed geometry (%.0f s): rel-RMS %.4f; μ %.4f; V(31/48/58°) %s\n", time() - tsw, rms1, θ.mat.diffuse_weight, _Vprof(θ.camera.vignetting)); flush(stdout)
        end
    end
    function monitor(it0, cst, camera, poses, mat, surf, light)
        it = it0 + run.resume.iteration_offset
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
        fname = _camera_file(route)                                       # lens | grid: the route's camera record (`_save_run_camera`)
        it > 0 && it % 5 == 0 && _save_run_camera(route, joinpath(run.io.out, "$(fname)_ckpt.json"), lens, meta.init_region;
                                                  provenance = (; model = "radiometric_$(_model_label(route, lens))_$(_frontend_tag(run))_ckpt", iteration = it))
        run.io.checkpoint_each && it > 0 && _save_run_camera(route, joinpath(run.io.out, "$(fname)_it$(it).json"), lens, meta.init_region;
                                                              provenance = (; model = "radiometric_$(_model_label(route, lens))_$(_frontend_tag(run))_it$(it)", iteration = it))
        if run.io.checkpoint_each && it > 0                              # per-iteration poses + state: the learned-pose evaluations and renders between iterations
            save_poses(joinpath(run.io.out, "poses_it$(it).json"), poses, names; world = meta.world)
            save_state(joinpath(run.io.out, "state_it$(it).json"),
                       (; camera, target = RadiometricTarget(_geometric_target(f.board.target), surf.h, warping_aspect(_board(f, θ)), TargetRadiometry(mat, surf); inplane = surf.inplane),
                          poses, names, light, settings = f.settings, observation = observation_model(f), page_flip = meta.page_flip);
                       provenance = (; iteration = it, cost = cst, rel_rms = sqrt(2cst / ϕ²)))   # the iteration's cost, as the monitor's line
        end
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
    it_done = Ref(0)
    monitor_chunk(it0, args...) = monitor(it0 + it_done[], args...)
    # the problem of every chunk (`build_problem` / `build_problem_grid`'s keywords)
    problem_kw = (; lp.kw..., damping)
    function run_chunk(n)                                    # n LM iterations from the current state (the problem is rebuilt per chunk, λ restarts)
        _lm_route!(f, θ, route; stops = _chunk_stops(stops, n, max(stops.max_time - (time() - t2), 1.0)), trace,
                 trace_meta = (; ϕ²), iter_callback = monitor_chunk, lm_log = alg.log, problem_kw...)
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
            ts = time(); cb, rmsb = cost(f, θ); camera_b, light_b, mat_b = θ.camera, θ.light, θ.mat
            _update_radiometry!(f, θ, _with_schedule(alg.radiometry; verbose = true, vignetting_smoothness = smoothness_weight(settings.vignetting.smoothness)),
                                (; light = blocks.lighting, vignetting = blocks.vignetting, mu = blocks.material); gcv_record = meta.vignetting_gcv)
            ca, rmsa = cost(f, θ)
            if ca < cb
                @printf("      sweep after it %d (%.0f s): cost %.4e → %.4e, rel-RMS %.4f → %.4f; μ %.4f; V(31/48/58°) %s → %s\n", it_done[] + run.resume.iteration_offset, time() - ts, cb, ca, rmsb, rmsa, θ.mat.diffuse_weight, _Vprof(camera_b.vignetting), _Vprof(θ.camera.vignetting))
                push!(trace, (; ϕ², phase = :sweep, cost = ca, time = time(), snap = Θ̄_snapshot(f, θ)))
            else
                @printf("      sweep after it %d (%.0f s) REJECTED: cost %.4e ≥ %.4e — state kept\n", it_done[] + run.resume.iteration_offset, time() - ts, ca, cb)
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
