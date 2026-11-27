#!/usr/bin/env julia
# Radiometric refinement of a geometric calibration on photographs of a printed board (paper §Calibration Pipeline): the generic grid
# model (a SplineLens start on its own lattice extended over the lens coverage, `grid.lattice = "derived"`, m = 1; a parametric start's
# lattice fitted over the image region at `grid.node_count` supported nodes, `grid.lattice = "node_count"`) or, with `io.init = "lens:<record>"`, a
# parametric model (a Core pixel camera: DoubleSphere, KannalaBrandt, the 14-coefficient chain, …; a VisualGeometryToolkitIO camera
# record) refined in its own parameters. The options are ONE TOML file
# (`RunOptions` and `CalibrationSettings`, src/options/; configs/default.toml lists every key and its default); two front-ends
# give the start geometry:
#   frontend.type = "blob"     a calibration session's VisualGeometryToolkitIO records (`frontend.session`: camera.json, targets_1.json,
#                              trajectory_1.json, target_poses.json, textures/ — `load_session`) and its observations record observations.json
#   frontend.type = "schoeps"  a Schöps central-generic run on a printed page (`frontend.schoeps`: intrinsics0.yaml, dataset.bin;
#                              `frontend.texture`, `frontend.texture_features`, `frontend.page_mm` describe the print)
# Outputs in `io.out`: grid.json | lens.json (+ *_init.json: VisualGeometryToolkitIO generic camera | camera records),
# poses_{init,refined}.json (trajectory records), state.json, radiometric_camera.json, radiometric_target.json (RadiometricCamera's
# records), parameters.json, the run's options as pipeline_config.toml, and with `io.checkpoint_each` the per-iteration
# lens_it<N>.json | grid_it<N>.json, poses_it<N>.json, state_it<N>.json. Diagnostic modes: `selfcheck.only` (held-out detections through
# `io.init`'s model), `io.render_init`, `io.render_state = "<state.json>"`.
# THE START is a step of its own (the owner's rule of 2026-11-14): `--stage=start` computes it from the options without the
# refinement-only keys (RadiometricCamera.REFINE_ONLY_KEYS, `start_options`) and writes the start record into `io.out` (start.json,
# state.json, grid.json | lens.json, masks/, pipeline_config.toml) and exits; `--start=<dir>` refines from the start record `dir`
# (refused when it is of other start options); with neither, the start is written into `io.out/start/` and the refinement continues
# from it in the same process (the photographs read once; `--start-out=<dir>` writes that start into `dir` instead: a driver's start
# product) — every refinement starts from a start record, so a start loaded and a start computed are one code path.
#   julia --project=. bin/refine.jl --config=<run.toml> [--set table.key=value …] [--plan] [--stage=start | --start=<dir>]
#     --set   overrides one key; repeatable. The value is parsed as ONE TOML value (2, 1e-10, true, [1, 2], "quoted string"); a value
#             that does not parse as TOML is taken as a bare string (--set io.out=/some/dir), so quote a string that looks like a number
#             or a boolean: --set 'io.out="2024"'.
#     --plan  prints the resolved options as TOML and exits without running
#     --repeat=N  runs the refinement N times in this one process (a timing measurement: the first repeat compiles, the later ones
#             are warm), each repeat's prepare and refine wall times, allocations and GC printed as a `REPEAT` line; every repeat writes
#             io.out, so the files are the last repeat's (the runs are deterministic: the same files). Each repeat also prints a
#             `TIMING r/N key=value …` line (the refinement's and the write's wall seconds, its LM iterations and seconds: what
#             experiments/scripts/timing.jl reads), and, after the last, a `PARAMETERS key=value …` line (the free parameters of the LM per block)
using RadiometricCamera
import VisualGeometryToolkitCore as VgtkCore
import TOML
const RC = RadiometricCamera
const USAGE = "usage: bin/refine.jl --config=<toml> [--set table.key=value …] [--plan] [--repeat=N] [--stage=start | --start=<dir> | --start-out=<dir>]"

function _cli(args)
    config = nothing; set = String[]; plan = false; repeat = 1; start_only = false; start = ""; start_out = ""; i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--config=");  config = a[(length("--config=") + 1):end]
        elseif a == "--config";          i += 1; i <= length(args) || throw(ArgumentError("--config needs a file; $USAGE")); config = args[i]
        elseif startswith(a, "--set=");  push!(set, a[(length("--set=") + 1):end])
        elseif a == "--set";             i += 1; i <= length(args) || throw(ArgumentError("--set needs table.key=value; $USAGE")); push!(set, args[i])
        elseif a == "--plan";            plan = true
        elseif startswith(a, "--repeat="); repeat = parse(Int, a[(length("--repeat=") + 1):end]); repeat >= 1 || throw(ArgumentError("--repeat: N ≥ 1; $USAGE"))
        elseif a == "--stage=start";     start_only = true
        elseif startswith(a, "--start="); start = a[(length("--start=") + 1):end]
        elseif startswith(a, "--start-out="); start_out = a[(length("--start-out=") + 1):end]
        else throw(ArgumentError("unknown argument $(repr(a)); $USAGE"))
        end
        i += 1
    end
    config === nothing && throw(ArgumentError("--config=<toml> is required; $USAGE"))
    !start_only || isempty(start) || throw(ArgumentError("--stage=start writes a start, --start=<dir> reads one: not both; $USAGE"))
    isempty(start_out) || (!start_only && isempty(start)) ||
        throw(ArgumentError("--start-out=<dir> names where a default run (start and refinement in one process) writes its start; $USAGE"))
    return config, set, plan, repeat, start_only, start, start_out
end
config, set, plan, repeat, start_only, start_dir, start_out = _cli(ARGS)
run_options, alg, stops = read_calibration(config; set)       # the run's options, the calibration (its settings and optimizer) and its solve's stops
if plan
    TOML.print(stdout, RC.config_dict(run_options, alg, stops); sorted = true)
    exit(0)
end
# the device (RC's one resolver, inlined): flux and Jacobian share this one backend, so the residual stays bitwise the one
# computeresidual returns
backend = RC.rc_backend(run_options.compute.backend)
isempty(run_options.io.out) && throw(ArgumentError("$config: io.out is required"))   # before the run record
isempty(start_dir) || abspath(start_dir) != abspath(run_options.io.out) ||
    throw(ArgumentError("--start=$start_dir is the run's io.out: a refinement writes its outputs beside its start, never into its start record"))
isempty(start_out) || abspath(start_out) != abspath(run_options.io.out) ||
    throw(ArgumentError("--start-out=$start_out is the run's io.out: the start record is written beside the refinement, never into its outputs"))
isempty(run_options.io.images) && !run_options.selfcheck.only &&
    throw(ArgumentError("$config: io.images is required (only a self-check, selfcheck.only, runs without photographs)"))
RC.write_pipeline_config(run_options.io.out, run_options, alg, stops;
                         header = "# the options of this run (bin/refine.jl --config=$config$(isempty(set) ? "" : " --set " * join(set, " --set ")))")
# the start: written (--stage=start: into io.out, and done; else into io.out/start/), or read (--start=<dir>)
# a default run (neither flag): the start computed, written into `--start-out=<dir>` (a driver's start product) or io.out/start/, and the
# refinement's state built from it in this process, no photograph read twice (the start's streamed observations handed over: the record
# path's state). The start reads the start options alone (`start_options`: no refinement option), its files in the run's io.out.
srun, ssettings = RC._start_run_options(run_options, alg.refinement.settings)
if srun.selfcheck.only                                                     # diagnostic: the start geometry's self-check, no photograph read
    RC.selfcheck(RC.start_geometry(srun, ssettings); run = srun)
    exit(0)
end
computed = nothing
if isempty(start_dir)
    start_dir = start_only ? run_options.io.out : isempty(start_out) ? joinpath(run_options.io.out, "start") : start_out
    ts = @timed begin
        g = RC.start_geometry(srun, ssettings)
        VgtkCore.init(VgtkCore.EstimationProblem(CalibrationEstimate, RC.CalibrationPhotographs(g, RC.start_photographs(srun, g.names)), (; run = run_options, device = backend)), alg; stops).cache.prob
    end
    s0 = ts.value
    if !isempty(srun.io.render_state)                                        # diagnostic: the masked pixels rendered with a finished state
        RC.load_state_into!(s0.f, s0.u0, RC.load_state(srun.io.render_state), s0.f.start.names)
        RC.render_dump!(s0.f, s0.u0, s0.f.start.names, srun.io.out, srun.io.render_state)
        exit(0)
    elseif srun.io.render_init                                               # diagnostic: the start state rendered
        RC.render_dump!(s0.f, s0.u0, s0.f.start.names, srun.io.out, "init (linear sweep)")
        exit(0)
    end
    save_calibration_start(start_dir, s0.f, s0.u0; run = srun, settings = ssettings)
    println("START written to ", start_dir, " in ", round(ts.time; digits = 1), " s"); flush(stdout)
    start_only && exit(0)
    computed = s0.f.start.observations; s0 = nothing; ts = nothing; GC.gc()  # the start released before the refinement's is built
end
"""
    _print_free_parameters(s, alg)

A timing measurement's context (`--repeat`): the free parameters of the refinement's LM over the loaded start `s` (its `RefineProblem`), per block of THE
problem the refinement builds (`_refinement_lm`, the per-tile route's `build_problem_grid`; the coordinates of its free variables), as
one `PARAMETERS key=value …` line. The grid lens's are its present direction controls' 2 tangent coordinates each, less the 3 that hold
the common rotation with the poses (`rotation_gauge_coordinates`); its patch blocks' solver coordinates (`lens_solver_coordinates`, which
also carry the absent nodes of a partly present patch) beside them. A parametric (per-view) start prints its route alone.
"""
function _print_free_parameters(s, alg)
    settings = alg.refinement.settings
    meta = s.f.start
    route = RC.lm_route(meta.lens0)
    if !(route isa RC.PerTileRoute)
        println("PARAMETERS route=$(nameof(typeof(route)))"); flush(stdout); return nothing
    end
    lp = RC._refinement_lm(settings, route; smoothness = meta.grid_smoothness)
    problem, idx = RC.build_problem_grid(s.f, s.u0; lp.kw...)
    vars = problem.variables
    n(is) = sum(i -> idx.free[i] ? Int(RC.NLLSsolver.nvars(vars[i])) : 0, is; init = 0)
    present = count(VgtkCore.present_nodes(VgtkCore.model(s.u0.camera.geometric)))
    gauge = lp.blocks.poses ? 3 : 0                                        # the rotation gauge: three direction coordinates held while the poses are free
    lensfree = lp.blocks.grid ? 2present - gauge : 0                       # a held lens ("camera" in fixed) has no free coordinate
    println("PARAMETERS route=PerTileRoute grid_nodes_present=$present lens=$lensfree lens_solver_coordinates=$(n(vec(idx.patches))) ",
            "focus=$(n(idx.focus:idx.focus)) aperture=$(n(idx.aperture:idx.aperture)) vignetting=$(n(idx.vignetting:idx.vignetting)) ",
            "poses=$(n(idx.poses)) views=$(length(idx.poses)) material=$(n(idx.material:idx.material)) lighting=$(n(idx.lighting:idx.lighting)) ",
            "samples=$(sum(length, s.f.ϕ; init = 0))")
    flush(stdout)
    return nothing
end
for r in 1:repeat
    if r > 1                                                                # the previous repeat's state released and collected BEFORE this
        global s = nothing; global result = nothing; GC.gc(true)           # repeat's spans: no finaliser of it lands in a timed span
    end
    obs = computed; global computed = nothing                               # the start's streamed observations: handed to repeat 1 only, then released
    tp = @timed load_calibration_start(start_dir; run = run_options, settings = alg.refinement.settings, device = backend, observations = obs); obs = nothing
    global s = tp.value
    tr = @timed VgtkCore.solve(s, alg.refinement; stops)
    global result = tr.value
    tw = @timed save_radiometric_calibration(s.f, s.u0, result; run = run_options, settings = alg.refinement.settings)
    println("TIMING $r/$repeat refine_s=$(tr.time) save_s=$(tw.time) refine_gc_s=$(tr.gctime) refine_compile_s=$(tr.compile_time) ",
            "save_compile_s=$(tw.compile_time) iterations=$(result.stats.nsteps) lm_seconds=$(result.metadata.seconds)")
    println(join(("REPEAT $r/$repeat", "load_calibration_start $(round(tp.time; digits = 1)) s ($(round(tp.bytes / 2^30; digits = 2)) GiB allocated, gc $(round(tp.gctime; digits = 1)) s, compile $(round(tp.compile_time; digits = 1)) s)",
                  "refinement $(round(tr.time; digits = 1)) s ($(round(tr.bytes / 2^30; digits = 2)) GiB allocated, gc $(round(tr.gctime; digits = 1)) s, compile $(round(tr.compile_time; digits = 1)) s)"), "; "))
    flush(stdout)
end
repeat > 1 && _print_free_parameters(s, alg)                               # a timing measurement's context, after every timed span
