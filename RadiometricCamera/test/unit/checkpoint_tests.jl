# The per-iteration checkpoint (`io.checkpoint_each`): a refinement writes poses_it<N>.json and state_it<N>.json after every LM
# iteration, the state through save_state with the run's observation model (RC 0.11.1 made it required; the monitor omitted it, and
# every checkpointed refinement crashed at iteration 1 until 0.11.2). A small decoded scene through `calibrate`, 2 LM iterations.
# A small decoded scene through `calibrate`: two 24 × 24 renders of a board through a grid camera, the start the rendering's own
# geometry (`run_scene(dir, lm, tex, cam)`: the fit and the run's io.out under `dir`, `lm` the [lm] table).
@testsetup module CheckpointScene
    using RadiometricCamera, StaticArrays, LinearAlgebra, PNGFiles, ImageCore
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: EnvironmentLight, SphericalHarmonicMap
    const RC = RadiometricCamera
    function run_board(dir, lm::AbstractDict, tex, cam; checkpoint = "on", tables = Dict{String, Any}(), start_only = false)   # tex, cam: RCFixtures' fixture_board(), cam
        board = RadiometricCamera.with_facing(RC.TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex), [0.0, 0.0, 1.5])
        W = H = 24
        glens = RC._image_start(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H), (10, 10)).camera
        poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                                       SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
        focus = RC.Focus(RC.FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), fill(1 / 1.5, 9))
        vig = RC.gauge_center_value(VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2))
        camera = RC.RadiometricCameraModel(glens, focus, 0.0386, vig)
        b = RC.VgtkCore.centroid(board); a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8)
        light = EnvironmentLight(SphericalHarmonicMap(a, 2), b, RC.sphere_radius(b, [p.t for p in poses]))
        mat = RC.Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
        rs = RC.RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
        masks = [RC.compute_mask(camera, p, board, rs; δ = 0.02, backend = RC.KernelAbstractions.CPU()) for p in poses]
        imgs = [RC.render_view(k, camera, poses[k], mat, light, board, rs, masks[k]) for k in 1:2]
        photos = [joinpath(dir, "V0$k.png") for k in 1:2]
        for k in 1:2                                                     # the renders as 16-bit linear photographs
            PNGFiles.save(photos[k], [RGB{N0f16}(clamp.(imgs[k][i, j, :], 0, 1)...) for i in 1:H, j in 1:W])
        end
        out = joinpath(dir, "out")
        run, alg, stops = RC.read_calibration(Dict("io" => Dict("out" => out, "checkpoint_each" => checkpoint, "channel" => "RGB"),
                                        "lm" => lm, "masks" => Dict("samples_per_view" => 10^9, "seed" => 1),
                                        "rendering" => Dict("N" => 4, "n_rho" => 2, "n_phi" => 6, "n_p" => 2), tables...))
        obs = [Tuple{SVector{2, Float64}, SVector{3, Float64}}[] for _ in 1:2]
        start_only && return VgtkCore.init(VgtkCore.EstimationProblem(RC.CalibrationEstimate, (RC.StartGeometry(glens, poses, obs, board; names = basename.(photos)), photos),
                                                                      (; run, device = RC.KernelAbstractions.CPU())), alg; stops).cache.prob
        fit = RC.calibrate(RC.StartGeometry(glens, poses, obs, board; names = basename.(photos)), photos, run, alg; stops, device = RC.KernelAbstractions.CPU())
        return (; fit, out, photos)
    end
    export run_board
end

@testitem "checkpoint: a checkpointed refinement runs 2 LM iterations; each state round-trips through save_state / load_state" setup = [RCFixtures, CheckpointScene] begin
    mktempdir() do dir
        (; fit, out, photos) = run_board(dir, Dict("max_iter" => 2, "ftol" => 0.0, "xtol_rel" => 0.0), fixture_board(), cam)
        @test fit.stats.nsteps == 2
        for it in 1:2
            f = joinpath(out, "state_it$(it).json")
            @test isfile(f) && isfile(joinpath(out, "poses_it$(it).json"))
            st = RC.load_state(f)                                         # the checkpoint round-trips, with the run's observation model
            @test st.observation == RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB)
            @test st.names == basename.(photos) && length(st.poses) == 2
            prov = VgtkIO.read_record(f)["provenance"]                    # the iteration's LM cost and rel-RMS (RC 0.11.4), as the monitor's line
            @test prov["iteration"] == it && prov["cost"] > 0 && 0 < prov["rel_rms"] < 1
        end
    end
end

# The LM's stop (RC 0.11.4, owner 2026-11-15): [lm] ftol / xtol_rel are the run's, from its TOML (Core's LmStops, no hard-coded value).
# A run whose relative cost decrease falls below ftol stops before the cap, its termination record (Success, CostDecrease): Core 0.11.0,
# a tolerance stop is converged; a run whose tolerances never fire reaches the cap: MaxIters / IterationCap, not converged.
@testitem "LM stop: [lm] ftol stops a converged run before the cap; tolerances that never fire reach it (MaxIters)" setup = [RCFixtures, CheckpointScene] begin
    mktempdir() do dir
        early = run_board(dir, Dict("max_iter" => 6, "ftol" => 0.5), fixture_board(), cam; checkpoint = "off").fit           # the start is the truth: the first step decreases < 50 %
        @test early.stats.nsteps < 6 && early.stop_rule == VgtkCore.CostDecrease
        @test early.retcode == VgtkCore.ReturnCode.Success
    end
    mktempdir() do dir
        capped = run_board(dir, Dict("max_iter" => 3, "ftol" => 0.0, "xtol_rel" => 0.0), fixture_board(), cam; checkpoint = "off").fit   # no tolerance can fire
        @test capped.stats.nsteps == 3 && capped.stop_rule == VgtkCore.IterationCap
        @test capped.retcode == VgtkCore.ReturnCode.MaxIters && !VgtkCore.successful_retcode(capped)
    end
    @test RC._calibration_stops().ftol == 5.0e-4 && RC._calibration_stops().xtol_rel == 1.0e-8     # configs/default.toml's (the owner's, 2026-11-15)
    @test RC._calibration_stops().max_iter == 50                                                     # the safety cap (owner ruling 2026-11-11)
    src = join(read.(joinpath.(pkgdir(RadiometricCamera), "src", "calibration", "lm", ("per_view.jl", "per_tile.jl")), String))
    @test !occursin("LM_RELDCOST", src) && !occursin("xtol = 1e-12", src)                    # no hard-coded tolerance
end

# The relative step stop (owner ruling 2026-11-16): `[lm] xtol_rel`, Core 0.10.11's `LmStops(; xtol_rel)` (Ceres' parameter tolerance,
# ‖Δx‖ ≤ xtol_rel·(‖x‖ + xtol_rel)), is the run's; the absolute `[lm] xtol` of 0.11.4's drafts is refused by name, `xtol_rel` named.
@testitem "LM stop: [lm] xtol_rel, Core's relative step test, stops an RC fit (IncrementSize); [lm] xtol is refused, xtol_rel named" setup = [RCFixtures, CheckpointScene] begin
    mktempdir() do dir      # a relative tolerance every first step meets: the run stops at it, before the cap, by the step-size rule
        f = run_board(dir, Dict("max_iter" => 6, "ftol" => 0.0, "xtol_rel" => 0.5), fixture_board(), cam; checkpoint = "off").fit
        @test f.stats.nsteps < 6 && f.stop_rule == VgtkCore.IncrementSize
    end
    @test RC._calibration_stops().xtol_rel == 1.0e-8 && !hasfield(RC.OptimizerTable, :xtol)
    e = try RC.pipeline_options(Dict("lm" => Dict("xtol" => 1.0e-8))); nothing catch err; err end
    @test e isa ArgumentError && occursin("xtol_rel", sprint(showerror, e))
end

# The focus support rule's record (owner ruling 2026-11-16): after `focus.support_min` drops the weakly seen nodes and the TPS is rebuilt
# on the supported ones, the start's `focus_support` describes the REBUILT node set (its weights and support, one entry per node of the
# start's focus), and names how many candidate nodes were dropped — never the pre-rebuild set.
@testitem "focus support: the record describes the rebuilt node set" setup = [RCFixtures, CheckpointScene] begin
    mktempdir() do dir
        s = run_board(dir, Dict("max_iter" => 1), fixture_board(), cam; checkpoint = "off", start_only = true,
                      tables = Dict("focus" => Dict("support_min" => 0.9)))
        sup = s.f.start.focus_support
        f = s.f.start.views0[1].focus
        @test sup.dropped > 0                                                  # the rule dropped nodes (else this test says nothing)
        @test length(sup.weight) == length(sup.supported) == length(f.t̂_i)
        @test sup.support_min == 0.9
    end
end
