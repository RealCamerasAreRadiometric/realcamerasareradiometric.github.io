#!/usr/bin/env julia
# A STUDY (research, moved out of the package's bin/ by RC 0.11.0, owner D6). The material grid search of a refinement's start (the chains of record since 2026-10-27; the user: "a grid search for the material
# parameters and the lighting"): from an init state (a start record's state.json, bin/refine.jl --stage=start: the start geometry, focus and board warping; V flat)
# every candidate (α_ink, α_¬ink, η_ink, η_¬ink; η grey over the channels) of the α × α × η × η grid is scored on the config's masked
# pixels — the parts rendered once on the CPU (threaded), μ scanned over 0:0.1:1 with the closed-form lighting Θ_l at each value, then from
# the best μ solve_Θ_l! / solve_μ! alternating until the cost changes < 1e-7 relative: the exact cost of the candidate (parts_cost).
# Writes <out>/grid.tsv (one row per candidate), <out>/state.json (the init state with the best material, μ and lighting; V flat) — the
# start of the joint LM (bin/refine.jl's `resume.state`) — and the options as pipeline_config.toml.
#   julia --project=experiments experiments/scripts/material_grid.jl --config=<run.toml> --init=<state.json> --out=<dir> [--set table.key=value …]
#                                          [--alphas=0.03,0.15,0.6] [--etas=1.3,2.0,2.7] [--limit=<n>]
#     --config, --set   the run's options (pipeline_options): the candidates are scored on THESE options' masks and radiance samples (a
#                       sparse set — masks.samples_per_view, rendering.N small) and the init state's geometry
#     --alphas, --etas  the candidate values, comma lists (the defaults are the values of record)
#     --limit           evaluate only the first n candidates (a timing probe)
using RadiometricCamera, Printf, StaticArrays
using RadiometricCamera: all_parts, cost, Material, material, set_vignetting!, solve_Θ_l!, solve_μ!, with_μ
import VisualGeometryToolkitCore as VgtkCore
import VisualRadiometryToolkitCore as VrtkCore
const RC = RadiometricCamera
const USAGE = "usage: experiments/scripts/material_grid.jl --config=<toml> --init=<state.json> --out=<dir> [--set table.key=value …] [--alphas=…] [--etas=…] [--limit=<n>]"

function _cli(args)
    config = nothing; init = nothing; out = nothing; set = String[]; alphas = "0.03,0.15,0.6"; etas = "1.3,2.0,2.7"; limit = typemax(Int)
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--config=");      config = a[(length("--config=") + 1):end]
        elseif startswith(a, "--init=");    init = a[(length("--init=") + 1):end]
        elseif startswith(a, "--out=");     out = a[(length("--out=") + 1):end]
        elseif startswith(a, "--set=");     push!(set, a[(length("--set=") + 1):end])
        elseif a == "--set";                i += 1; i <= length(args) || throw(ArgumentError("--set needs table.key=value; $USAGE")); push!(set, args[i])
        elseif startswith(a, "--alphas=");  alphas = a[(length("--alphas=") + 1):end]
        elseif startswith(a, "--etas=");    etas = a[(length("--etas=") + 1):end]
        elseif startswith(a, "--limit=");   limit = parse(Int, a[(length("--limit=") + 1):end])
        else throw(ArgumentError("unknown argument $(repr(a)); $USAGE"))
        end
        i += 1
    end
    (config === nothing || init === nothing || out === nothing) && throw(ArgumentError("--config, --init and --out are required; $USAGE"))
    return config, init, out, set, parse.(Float64, split(alphas, ",")), parse.(Float64, split(etas, ",")), limit
end
config, initpath, out, set, αs, ηs, limit = _cli(ARGS)
opts = pipeline_options(config; set)
mkpath(out)
RC.write_pipeline_config(out, opts;
                         header = "# the options of this material grid (experiments/scripts/material_grid.jl --config=$config$(isempty(set) ? "" : " --set " * join(set, " --set ")))")
init = RC.load_state(initpath)
# the state's masks and photographs: the init's start record when it is one (its state.json), else a start of these options written
# into <out>/start (bin/refine.jl --stage=start's); a diagnostic mode of the options writes no start
(opts.selfcheck.only || opts.io.render_init || !isempty(opts.io.render_state)) && error("$config: a diagnostic mode (selfcheck.only, io.render_*) writes no start")
sdir = if isfile(joinpath(dirname(initpath), RC.START_RECORD_FILE))
    dirname(initpath)
else
    so = RC._start_run_options(opts); g = RC.start_geometry(so)
    RC.save_calibration_start(joinpath(out, "start"), RC.start_calibration(g, RC.start_photographs(so, g.names); options = so, device = RC.KernelAbstractions.CPU()); options = so)
end
s = RC.load_calibration_start(sdir; options = opts, device = RC.KernelAbstractions.CPU()); meta = s.start
RC.load_state_into!(s, init, meta.names)                                 # the init's poses and camera (lens, focus, r), surface, lighting, V
let v = s.camera.vignetting                                              # V flat: the grid scores the material and the lighting alone
    set_vignetting!(s, VrtkCore.with_control_points(v, ones(size(VrtkCore.control_points(v)))))
end
cands = vec([(ai, an, ei, en) for ai in αs, an in αs, ei in ηs, en in ηs])
cands = cands[1:min(length(cands), limit)]
ϕ² = sum(sum(abs2, p) for p in s.ϕ)
relrms(c) = sqrt(2c / ϕ²)
@printf("material grid: %d candidates (α %s; η %s) on %d pixels of %d views, V flat\n", length(cands), join(αs, "/"), join(ηs, "/"),
        sum(length.(s.px)), length(s.px)); flush(stdout)

"""μ scanned over 0:0.1:1 with the closed-form lighting at each value, then the alternation from the best μ; returns the cost."""
function fit_light_mu!(s, parts)
    best = (Inf, 0.5)
    for μ in 0.0:0.1:1.0
        s.mat = with_μ(s.mat, μ); solve_Θ_l!(s, parts); c = RC.parts_cost(s, parts)
        c < best[1] && (best = (c, μ))
    end
    s.mat = with_μ(s.mat, best[2]); solve_Θ_l!(s, parts); c = RC.parts_cost(s, parts)
    for _ in 1:100
        solve_μ!(s, parts); solve_Θ_l!(s, parts); c1 = RC.parts_cost(s, parts)
        done = c - c1 < 1e-7 * c; c = min(c, c1); done && break
    end
    return c
end

best = (; c = Inf, mat = s.mat, light = s.light)
open(joinpath(out, "grid.tsv"), "w") do io
    println(io, "α_ink\tα_noink\tη_ink\tη_noink\tμ\trel_RMS\tseconds"); flush(io)
    for (i, (ai, an, ei, en)) in enumerate(cands)
        t = time()
        s.mat = Material(; roughness_ink = ai, roughness_paper = an, refractive_index_ink = ei, refractive_index_paper = en, diffuse_weight = 0.5)
        parts = all_parts(s)
        c = fit_light_mu!(s, parts); parts = nothing
        dt = time() - t
        @printf(io, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.6f\t%.1f\n", ai, an, ei, en, s.mat.diffuse_weight, relrms(c), dt); flush(io)
        c < best.c && (global best = (; c, mat = s.mat, light = s.light))
        @printf("%4d/%d  α %.3f/%.3f  η %.2f/%.2f  μ %.3f  rel-RMS %.5f  (%.0f s)%s\n", i, length(cands), ai, an, ei, en, s.mat.diffuse_weight, relrms(c), dt,
                c == best.c ? "  *" : ""); flush(stdout)
        i % 16 == 0 && GC.gc()
    end
end
η₂(x) = x isa Number ? Float64(x) : Float64(x[2])
r0 = init.target.radiometry
RC.save_state(joinpath(out, "state.json"),
              (; init..., target = RC.RadiometricTarget(init.target.target, init.target.warping, init.target.aspect, RC.TargetRadiometry(best.mat, r0.ink_spread, r0.print_blur)),
                 light = best.light, camera = s.camera);
              provenance = (; iteration = 0, material_grid = (; alphas = αs, etas = ηs, candidates = length(cands), rel_rms = relrms(best.c))))
@printf("best: α %.3f/%.3f  η %.2f/%.2f  μ %.3f  rel-RMS %.5f → %s\n", best.mat.roughness_ink, best.mat.roughness_paper, η₂(best.mat.refractive_index_ink), η₂(best.mat.refractive_index_paper),
        best.mat.diffuse_weight, relrms(best.c), joinpath(out, "state.json"))
