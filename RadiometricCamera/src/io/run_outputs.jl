"""
    save_poses(path, poses, names; world)

The views' poses (camera → world, one per view) as VisualGeometryToolkitIO's trajectory record (`vgio/trajectory/v3`): world → camera
(`X_cam = R X_world + t`, the inverse of each camera → world pose) per frame k with its image `names[k]`, lengths in mm on disk;
`world` names the calibration's world frame in the provenance.
"""
function save_poses(path::AbstractString, poses::AbstractVector{<:EuclideanMap}, names; world::AbstractString)
    poses = Dict(k => VgtkCore.with_unit(inv(_pose_eltype(Float64, poses[k])), VgtkCore.m) for k in eachindex(names))
    tr = VgtkCore.Trajectory(VgtkCore.WorldFrame, VgtkCore.ViewFrame{:cam}, poses; images = Dict(k => String(names[k]) for k in eachindex(names)))
    VgtkIO.save(path, tr; provenance = (; world))
    return nothing
end

# ---- outputs -------------------------------------------------------------------------------------------------------------------
"""
    _save_run_grid(path, lens, init_region; provenance)

A grid camera the pipeline writes (`save_grid`: the result, its start, the saved initialization, a checkpoint) with its start's
initialization region `init_region` beside it (`save_initialization_region`): every grid the pipeline writes is a valid start
(`load_start`). A run whose start records no region (`nothing`: a parametric start) writes none — and removes one a former run left
in the directory — and its grids are refused as starts.
"""
function _save_run_grid(path::AbstractString, g, init_region; provenance::NamedTuple)
    save_grid(path, g; provenance)
    dir = dirname(abspath(path))                     # the grid's own directory, absolute (a relative path's dirname is "")
    if init_region === nothing                       # no region: none beside the grid, never another run's left in the directory
        rm(joinpath(dir, INITIALIZATION_REGION_FILE); force = true)
    else
        save_initialization_region(dir, init_region)
    end
    return nothing
end

"""
    lm_termination(res) -> (ReturnCode.T, StopRule)

The termination record of the refinement's LM: `res` is the last chunk's record (VisualGeometryToolkitCore's `lmsolve_chart!` on the
LM's `LmProblem`: `(; code, rule, iterations, cost)`), decided at the point the chunk returned (VisualGeometryToolkitCore's rule): a cost-decrease or
step-size tolerance stop (`[optimizer] ftol`, `xtol`) is `Success` — converged by definition, no stationarity postcondition; the rejection
cap is `Success` only at a stationary point and `Stalled` otherwise; the iteration and time caps `MaxIters` / `MaxTime`. A run of chunks ends on one
that did not iterate, at the final state. `nothing` (the LM did not run): `(ReturnCode.Default, NoRule)`.
"""
lm_termination(res) = res === nothing ? (VgtkCore.ReturnCode.Default, VgtkCore.NoRule) : (res.code, res.rule)

"""
    save_radiometric_calibration(f, θ, result; options)

The records of a radiometric calibration (a [`RadiometricCalibration`](@ref) solve's `result` of the state `state`, the run `options`) in the run's
directory `io.out`, each through VisualGeometryToolkitIO:

`grid.json` / `lens.json` (+ `grid_init.json` / `lens_init.json`; the model spec of the evaluation scripts is `grid:<file>` | `lens:<file>`),
`state.json`, `radiometric_camera.json`, `radiometric_target.json`, `poses_{init,refined}.json` and `parameters.json` in `io.out`.
"""
function save_radiometric_calibration(f::CalibrationData, θ::CalibrationParameters, result::VgtkCore.FitResult{<:CalibrationEstimate, <:CalibrationMetadata}; options::PipelineOptions)
    m = result.metadata
    costs = (; linear = (m.linear_cost, m.linear_rms), final = (m.final_cost, m.final_rms), seconds = m.seconds,
               termination = (result.retcode, result.stop_rule), m.folds)
    return _write_calibration(f, θ, f.start, options, m.solver, m.trace, costs)
end
function _write_calibration(f::CalibrationData, θ::CalibrationParameters, meta, opts::PipelineOptions, res, trace, costs)
    names = meta.names; K = length(names); W₀ = meta.W₀; H₀ = meta.H₀; lens0 = meta.lens0; views0 = meta.views0; route = lm_route(lens0)
    rcam = θ.camera; gfin = rcam.geometric
    (c0, rms0) = meta.costs.init; (c1, rms1) = costs.linear; (c2, rms2) = costs.final
    _write_run_cameras(route, opts, meta, gfin, lens0, trace, costs, (rms0, rms1, rms2))
    save_state(joinpath(opts.io.out, "state.json"), f, θ, names; page_flip = meta.page_flip,
               provenance = (; iterations = length(trace), μ_lin = meta.μ_lin, cost = (; init = (c0, rms0), linear = (c1, rms1), final = (c2, rms2))))
    save_radiometric_camera(joinpath(opts.io.out, "radiometric_camera.json"), calibrated_camera(f, θ); provenance = (; views = names))
    save_radiometric_target(joinpath(opts.io.out, "radiometric_target.json"), radiometric_target(f, θ))
    save_poses(joinpath(opts.io.out, "poses_init.json"), [v.pose for v in views0], names; world = meta.world)
    save_poses(joinpath(opts.io.out, "poses_refined.json"), θ.poses, names; world = meta.world)
    surf = board_surface(θ)
    feats1 = _board_features(meta, surf, meta.features)
    be1 = detection_bearings(gfin, θ.poses, feats1, meta.fpx)
    moved_m = [norm(θ.poses[k].t - views0[k].pose.t) for k in 1:K]; moved = _mm.(moved_m)
    save_run_parameters(joinpath(opts.io.out, "parameters.json"), _run_parameters(opts, meta, f, θ, lens0, gfin, rcam, surf, be1, moved_m, route, costs, trace, res,
                                                                                    (c0, rms0), (c1, rms1), (c2, rms2));
                        provenance = (; window_width = something(_device_entry(f, :window_width), 0)))   # an execution setting (`_window_width`)
    @printf("bearing error of the detections: init %.3f px → refined model + refined poses %.3f px; camera centres moved %.1f–%.1f mm\n",
            _med(meta.be0), _med(be1), minimum(moved), maximum(moved))
    println("written to ", opts.io.out, "; REFINE ", uppercase(_camera_file(route)), " DONE"); flush(stdout)
    return nothing
end

"""The run's rows' weights from each view's (`masked_observations`): `Unweighted()` when every view's rows are as measured, the views'
1/σ vectors when every view's are whitened (one weighting per run: a mix is an error)."""
function _run_weights(ω::AbstractVector)
    all(w -> w isa Unweighted, ω) && return Unweighted()
    all(w -> w isa AbstractVector, ω) || error("the photographs' rows mix whitened and unweighted views: one weighting per run")
    return Vector{Vector{Float64}}(ω)
end

"The photographs' count per resolver reason (Core's `ModelReason` of each model, `component_sources`): the run log's sensor line."
_count_reasons(models) = (d = Dict{Any, Int}(); foreach(m -> (r = VgtkCore.component_sources(m).reason; d[r] = get(d, r, 0) + 1), models); d)

# ---- the run's parameters (rc/run_parameters/v1) -----------------------------------------------------------------------------------
"""
    _run_parameters(opts, meta, f, θ, lens0, gfin, rcam, surf, be1, moved_m, route, costs, trace, res, cost0, cost1, cost2) -> NamedTuple

The body of a refinement's `rc/run_parameters/v1` record ([`save_run_parameters`](@ref)): what the run was given and what it measured,
never what its model records hold (the camera, the target, the poses, the light and the settings are radiometric_camera.json,
radiometric_target.json, poses_*.json and state.json). ASCII snake_case keys, no unit in a key: every length a plain metre value here
(the writer puts it in the file's `length_unit`), image quantities in the photographs' (COLMAP) pixels, the rest dimensionless.
"""
function _run_parameters(opts, meta, f::CalibrationData, θ::CalibrationParameters, lens0, gfin, rcam, surf, be1, moved_m, route::LmRoute, costs, trace, res, cost0, cost1, cost2)
    m(x) = VgtkCore.ustrip(VgtkCore.m, x)                                     # a length quantity → plain metres
    sp = opts.grid.start_perturbation
    fv = _support_focus_values(rcam.focus, meta.in_support); fm = 1 / mean(fv)
    return (; frontend = _frontend_tag(opts), session = opts.frontend.session, schoeps = opts.frontend.schoeps, images = opts.io.images,
        views = [String(n) for n in meta.names], image_size = [meta.W₀, meta.H₀],
        # (the initialization region is the grids' own record, initialization_region.json beside them: `_save_run_grid`)
        grid_size = route isa PerViewRoute ? nothing : collect(Int, grid_dims(lens0)),   # the start's lattice (its exact resample's, or fitted over the image region at grid.node_count)
        grid = route isa PerViewRoute ? nothing : (; n_interior = count(meta.grid_int), n_nodes = length(meta.grid_int), interior = [collect(c) for c in eachcol(meta.grid_int)],
                                         # [grid.smoothness] as the run applied it (rc/run_parameters v3.0): its type, order and weight
                                         # (a number or "gcv"), its cost at the start and the end, the weight GCV chose and its selection
                                         smoothness = (; smoothness_record(opts.grid.smoothness)...,
                                                         cost = (; init = grid_smoothness_cost(meta.grid_smoothness, lens0), final = grid_smoothness_cost(meta.grid_smoothness, gfin)),
                                                         beta = smoothness_weight(opts.grid.smoothness) isa VgtkEstimation.GeneralizedCrossValidation && meta.grid_smoothness !== nothing ? meta.grid_smoothness.β : nothing,
                                                         selection = meta.grid_gcv),
                                         start_perturbation = sp === nothing ? nothing : (; sigma = sp.sigma_px, seed = sp.seed)),
        vignetting_options = !smoothing(opts.vignetting.smoothness) && opts.vignetting.boundary_order == 2 ? nothing :
            (; smoothness = (; smoothness_record(opts.vignetting.smoothness)..., selection = isempty(meta.vignetting_gcv) ? nothing : meta.vignetting_gcv),
               boundary_order = opts.vignetting.boundary_order),
        # [focus.smoothness] as the run applied it: its type and weight β, and the row weight β Σϕ²/N of its bending rows
        # (`focus_smoothness!`); nothing without a focus smoothness
        focus_options = meta.focus_row_weight > 0 ? (; smoothness = (; smoothness_record(opts.focus.smoothness)..., row_weight = meta.focus_row_weight)) : nothing,
        resolution_used = [_image_width(θ), _image_height(θ)], samples = meta.samples, masked_pixels = meta.npx,
        sensor_width = m(_sensor_width(opts)), focal_length = m(_focal_length(opts)), nominal_focal = meta.fpx,
        aperture_start = meta.R0, focus_distance_start = meta.d_c, linearize = opts.io.linearize, material_fixed = "material" in opts.fixed,
        material_mu_start = opts.material.mu, mu_linear = meta.μ_lin,
        cost = (; init = (; cost = cost0[1], rel_rms = cost0[2]), after_linear_solve = (; cost = cost1[1], rel_rms = cost1[2]),
                  after_refinement = (; cost = cost2[1], rel_rms = cost2[2])),
        printed_edge = isempty(opts.edge.sdf) ? nothing :
            (; sdf = opts.edge.sdf,
               start = (; ink_spread = m(opts.edge.ink_spread_mm * VgtkCore.mm), print_blur = m(opts.edge.print_sigma_mm * VgtkCore.mm))),
        board_warping = (opts.warping.degree > 0 || height_on(surf)) ? (; free = opts.warping.degree > 0, start = collect(Float64, meta.height0)) : nothing,
        inplane_print_error = _inplane_modelled(opts, surf) ? (; free = _surface_free(opts, :inplane), start = collect(Float64, meta.inplane0)) : nothing,
        focus_distance = isfinite(fm) ? fm : nothing, focus_support_nodes = length(fv),   # 1/mean t̂ over the support nodes (Inf: at infinity)
        bearing_error = isempty(meta.be0) ? nothing : (; init_median = _med(meta.be0), init_p90 = _q90(meta.be0), refined_median = _med(be1), refined_p90 = _q90(be1)),
        sensor_characterization = opts.camera.sensor_characterization, sensor_models = sensor_record(meta.names, f.sensor_models),
        sensor_model = hasproperty(meta, :sensor_model) ? meta.sensor_model : nothing,   # mode, used, the record present (`sensor_model_record`): "off" + present = a record not used
        camera_moved = moved_m, iterations = length(trace), seconds = round(costs.seconds),
        stopped = res === nothing ? "no LM iteration" : string(res.rule),
        retcode = string(costs.termination[1]), stop_rule = string(costs.termination[2]),        # the termination record (Core's §14): the LM's, or the fold check's
        calibrated_folds = costs.folds === nothing ? nothing :                                    # the calibrated camera's folds over its coverage (`calibrated_folds`)
            (; folded = length(costs.folds.folded), samples = costs.folds.samples, unimaged = costs.folds.unimaged))
end

# ---- the run's metrics, read back from its output files -------------------------------------------------------------------------
"""
    load_pipeline_metrics(dir) -> Dict{String, Any}

The metrics a pipeline run wrote in `dir`, as one TOML-printable table, read from the run's own files:

- a refinement (`parameters.json`, its `rc/run_parameters/v1` record, [`load_run_parameters`](@ref)): `cost` (the photometric cost
  and its relative RMS at the start, `init`, after the closed-form solve, `after_linear_solve`, and after the LM, `after_refinement`),
  `bearing_error` (the bearing error of the detections through the start model, `init_*`, and through the refined model and poses,
  `refined_*`: median and p90, pixels), `camera_moved` (per view, metres: the record's length unit), `iterations`, `stopped` (the LM's
  termination bits by name) and its termination record: `retcode` (the name of its VisualGeometryToolkitCore `ReturnCode`) and
  `stop_rule` (of its `StopRule`);
- a `selfcheck.only` run (`selfcheck.json`): `selfcheck` with `n`, `median`, `p90` and `mean` of the bearing error over every
  detection (px).

A value the pipeline wrote as null is left out. A directory holding neither file is an `ArgumentError`.
"""
function load_pipeline_metrics(dir::AbstractString)
    parameters = joinpath(dir, "parameters.json"); selfcheck = joinpath(dir, "selfcheck.json")
    if isfile(parameters)
        p = load_run_parameters(parameters)
        return Dict{String, Any}("cost" => _metric_value(p.cost), "bearing_error" => _metric_value(p.bearing_error),
                                 "camera_moved" => _metric_value(p.camera_moved), "iterations" => Int(p.iterations), "stopped" => String(p.stopped),
                                 "retcode" => String(p.retcode), "stop_rule" => String(p.stop_rule))
    elseif isfile(selfcheck)
        sc = _metric_value(VgtkIO.read_record(selfcheck; allownan = true).all)
        sc["n"] = Int(sc["n"])
        return Dict{String, Any}("selfcheck" => sc)
    end
    throw(ArgumentError("$dir holds no pipeline output (parameters.json of a refinement, selfcheck.json of a self-check)"))
end
# parsed JSON values → plain Dict / Vector / scalars, nulls dropped (TOML has none).
_metric_value(x::AbstractDict) = Dict{String, Any}(String(k) => _metric_value(v) for (k, v) in pairs(x) if v !== nothing)
_metric_value(x::AbstractVector) = [_metric_value(v) for v in x if v !== nothing]
_metric_value(x::AbstractString) = String(x)
_metric_value(x) = x

"""The refined camera and its start in the run's directory, as the route records them: `lens.json`, `lens_init.json` (a parametric
camera, `save_lens`, its start spec `io.init` in the provenance) or `grid.json`, `grid_init.json` (a grid, `_save_run_grid`: the
start's initialization region beside them). `rms = (init, linear, final)`."""
function _write_run_cameras(::PerViewRoute, opts::PipelineOptions, meta, gfin, lens0, trace, costs, rms)
    rms0, rms1, rms2 = rms; names = meta.names
    save_lens(joinpath(opts.io.out, "lens.json"), gfin;
              provenance = (; model = "radiometric_$(lens_type_name(gfin))_$(_frontend_tag(opts))", rel_rms = (; init = rms0, linear = rms1, final = rms2), iterations = length(trace),
                            seconds = round(costs.seconds), frontend = _frontend_tag(opts), views = names, init = opts.io.init))
    save_lens(joinpath(opts.io.out, "lens_init.json"), lens0; provenance = (; model = "init_$(lens_type_name(lens0))_$(_frontend_tag(opts))", frontend = _frontend_tag(opts), init = opts.io.init))
    return nothing
end
function _write_run_cameras(::PerTileRoute, opts::PipelineOptions, meta, gfin, lens0, trace, costs, rms)
    rms0, rms1, rms2 = rms; names = meta.names
    _save_run_grid(joinpath(opts.io.out, "grid.json"), gfin, meta.init_region; provenance = (; model = "radiometric_grid_$(_frontend_tag(opts))", rel_rms = (; init = rms0, linear = rms1, final = rms2), iterations = length(trace),
                                                     seconds = round(costs.seconds), frontend = _frontend_tag(opts), views = names))
    _save_run_grid(joinpath(opts.io.out, "grid_init.json"), lens0, meta.init_region; provenance = (; model = "init_grid_$(_frontend_tag(opts))", frontend = _frontend_tag(opts)))
    return nothing
end
