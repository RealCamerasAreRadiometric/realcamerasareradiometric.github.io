# JACOBIAN PARITY on the paper-like Siemens case: the one Jacobian implementation on the device against the same on the CPU backend
# (Float64), tile by tile, at the start of the paper's f4_14mm_spline_warped refinement (experiments
# configs/experiments/radiometric_refinement_vignetting.toml: Sigma 14 mm f/4, the 1 m Siemens print with its distance-map edge, n_p 2,
# N 16 Hammersley, ≤ 12 048 px per view, warping degree 1, radial V) on its first NVIEWS photographs, and after a 3-iteration grid LM
# from there. The package's GPU items use small scenes whose height columns are benign; on this case the warping's columns are
# sensitive (the print's 0.03 mm edge): the 0.9 port before perf-lm (25 nodal-height slots through the Float32 radiance estimator)
# was 6 % off in a column and 14–94 % in single tiles, a Float32 height path 0.2–1.3 % stacked and ≤ 5.6 % in a tile, the
# Float64 rays with a Float32 estimator 0.30 % stacked and 0.44 % in a tile — all of them fail this check's gates.
#
#   julia --project=RadiometricCamera experiments/test/pipeline/jacobian_parity.jl DATA_ROOT IMAGES [NVIEWS = 2]
# DATA_ROOT: the data root of the experiments configs (its relative input paths); IMAGES: the decoded linear photographs of the set
# (`[prepare.fx3_14mm_siemens_20261028]`'s run, <run>/fx3_14mm_siemens_20261028_linear). Needs a functional CUDA device. Exit 1 on
# a gate's failure. Gates: the heights' columns stacked over the tiles AND the worst column of any tile with ≥ 300 px within 0.4 % of
# the CPU's (the device's Float64 height path: ≤ 2.2e-12, on a B200, 2026-11-11); the pose's the same (its chunks in Float64,
# `HIGH_FLOAT`: ≤ 1.8e-12; Float32 chunks were 0.37–0.67 % stacked and 1.6 % in a tile); the residuals bitwise the flux kernel's.
using RadiometricCamera, CUDA, LinearAlgebra, TOML, Printf
const RC = RadiometricCamera
const NL = RC.NLLSsolver

function jacobian_parity(root, images, nviews = 2)
    CUDA.functional() || error("jacobian_parity.jl: no functional CUDA device")
    cfgpath = joinpath(@__DIR__, "..", "..", "configs", "experiments", "radiometric_refinement_vignetting.toml")
    cfg = TOML.parsefile(cfgpath); t = cfg["cases"]["f4_14mm_spline_warped"]
    p = deepcopy(t["pipeline"])
    for key in RC.PIPELINE_PATH_KEYS
        tab, k = split(key, ".")
        (haskey(p, tab) && haskey(p[tab], k) && RC.names_input_path(key, p[tab][k]) && !isabspath(p[tab][k])) || continue   # "none" names no path
        p[tab][k] = joinpath(root, p[tab][k])
    end
    p["io"]["views"] = cfg["views"][t["views"]][1:nviews]; p["io"]["images"] = images
    p["io"]["out"] = mktempdir(); p["io"]["checkpoint_each"] = "off"
    opts = RC.pipeline_options(p)
    s = redirect_stdout(devnull) do
        so = RC._start_run_options(opts); g = RC.start_geometry(so); dir = joinpath(opts.io.out, "start")
        RC.save_calibration_start(dir, RC.start_calibration(g, RC.start_photographs(so, g.names); options = so, device = CUDA.CUDABackend()); options = so)
        RC.load_calibration_start(dir; options = opts, device = CUDA.CUDABackend())
    end
    meta = s.start
    # the problem as refine_calibration! builds it (the one derivation, `_refinement_lm`: its blocks, the structure, the holds — Θ_m's among them — and the prior)
    lp = RC._refinement_lm(opts, RC.PerTileRoute(); smoothness = meta.grid_smoothness); kw = lp.kw
    problem(st) = RC.build_problem_grid(st; kw...)
    host(st) = (h = typeof(st)(ntuple(i -> getfield(st, i), fieldcount(typeof(st)))...); h.backend = RC.KernelAbstractions.CPU(); h.device_cache = nothing; h)

    failed = false
    function check!(label)
        sc = host(s)
        pc, _ = problem(sc); pg, _ = problem(s)
        acc = Dict{Symbol, Vector{Tuple{Float64, Float64}}}()
        worst_tile = Dict{Symbol, Float64}()
        for (rc, rg) in zip(RC.tile_costs(pc), RC.tile_costs(pg))
            vc = NL.getvars(rc, pc.variables); vg = NL.getvars(rg, pg.variables)
            r0c, Jc = NL.computeresjac(RC.static(1023), rc, vc...)
            r0g, Jg = NL.computeresjac(RC.static(1023), rg, vg...)
            (r0c == NL.computeresidual(rc, vc...) && r0g == NL.computeresidual(rg, vg...)) || (failed = true; println("FAIL $label: a residual is not the flux kernel's"))
            nv = [Int(NL.nvars(v)) for v in vc]
            off(b) = sum(nv[1:(b - 1)]; init = 0)
            blocks = (pose = off(1) .+ (1:6), heights = off(8) .+ (10:(9 + RC.HEIGHT_NFREE)))
            for (name, cols) in pairs(blocks)
                a = get!(acc, name, [(0.0, 0.0) for _ in cols])
                for (q, c) in enumerate(cols)
                    a[q] = (a[q][1] + sum(abs2, Jg[:, c] .- Jc[:, c]), a[q][2] + sum(abs2, Jc[:, c]))
                end
            end
            length(rc.pix) >= 300 && for (name, cols) in pairs(blocks)
                worst_tile[name] = max(get(worst_tile, name, 0.0), maximum(c -> norm(Jg[:, c] .- Jc[:, c]) / norm(Jc[:, c]), cols))
            end
        end
        gates = (pose = 4e-3, heights = 4e-3)
        for (name, a) in acc
            e = [sqrt(x / y) for (x, y) in a]
            ok = maximum(e) <= gates[name]
            ok || (failed = true)
            @printf("%s %-8s %s: stacked column errors %.2e … %.2e (gate %.0e)\n", ok ? "ok  " : "FAIL", label, name, minimum(e), maximum(e), gates[name])
        end
        for (name, w) in worst_tile
            ok = w <= gates[name]
            ok || (failed = true)
            @printf("%s %-8s %s: worst column of a tile (≥ 300 px) %.2e (gate %.0e)\n", ok ? "ok  " : "FAIL", label, name, w, gates[name])
        end
    end
    check!("start")
    redirect_stdout(devnull) do
        RC.run_lm_grid!(s; maxiters = 3, kw...)
    end
    check!("lm3")
    return failed ? 1 : 0
end

abspath(PROGRAM_FILE) == (@__FILE__) && exit(jacobian_parity(ARGS[1], ARGS[2], length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2))
