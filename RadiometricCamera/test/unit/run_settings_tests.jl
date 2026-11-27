# A run's settings travel with its problem, never through module globals (the owner's rule, 2026-11-14: "These global constants
# should not exist"): the material hold (`"material"` in `lm.fixed`: the problem's `fixed`, the MaterialVar's type), the μ bounds
# (`material.mu_bounds`: the state's), the Jacobian chunk (`compute.jacobian_chunk`: the state's) and the LM log (`io.optimizer_log`,
# version 1 `lm.log`: the run's `calibration_checkpoints` callback). A run that sets them leaves the next problem of the same session untouched.
@testitem "run settings: no leak into the next problem" setup = [RCFixtures] begin
    using RadiometricCamera: build_problem, load_state, material, _lm_views!
    using StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: exp_so3

    module RunSettingsTest
    using Test, RadiometricCamera, StaticArrays, LinearAlgebra
    const OLD_STACK = false
    include(joinpath(@__DIR__, "..", "regression", "golden_scenes.jl"))
    const RC = RCg
    const NL = RC.NLLSsolver

    W = H = 32
    kb = cam(RC.VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0), 41.216, (16.32, 15.36), W, H)
    fresh() = (sc = gold_board(W, H; h = gold_h()); gold_state(sc..., kb))
    material_var(prob) = only(v for v in prob.variables if v isa RC.MaterialVar)
    # a step of every coordinate of the material block, μ's far below its chart's centre
    increment(mv) = (u = fill(0.01, Int(NL.nvars(mv))); u[9] = -50.0; u)

    mktempdir() do dir
        log = joinpath(dir, "lm.tsv")
        run, settings, alg, stops = RC.read_calibration(Dict("fixed" => ["material"], "lm" => Dict("log" => log), "warping" => Dict("degree" => 1),
                                        "material" => Dict("mu_bounds" => [0.6, 1.0]), "compute" => Dict("jacobian_chunk" => 7),
                                        "resume" => Dict("state" => joinpath(dir, "state.json")),
                                        "io" => Dict("out" => mkpath(joinpath(dir, "out")), "images" => dir),
                                        "selfcheck" => Dict("only" => "on")))
        @test run.io.optimizer_log == log                                                     # the version-1 `lm.log`, the run's output
        # a run of these settings, stopped after it took them in: the preparation (its resume state is absent) and the
        # resume (its state reader)
        @test_throws Exception RC.start_geometry(run, settings)
        s = fresh()
        meta = (; names = ["a", "b"], W₀ = W, H₀ = H, fpx = 41.216, lens0 = s.u0.camera.geometric, views0 = nothing,
                features = nothing, costs = (; linear = (0.0, 0.0)), grid_smoothness = nothing)
        s.f.start = meta
        @test_throws "stop" RC.load_calibration_state(RC.VgtkCore.RefineProblem(s.f, s.u0), run.resume.state; load_state = _ -> error("stop"))

        # the next problem of the session: the material free, μ on (0, 1), no LM log
        s2 = fresh()
        mv = material_var(first(build_problem(s2.f, s2.u0; height_degree = 1)))
        m2 = NL.update(mv, increment(mv))
        @test RC.nonlinear_parameters(m2.mat) != RC.nonlinear_parameters(mv.mat)                           # Θ_m moves
        @test m2.mat.diffuse_weight < 0.5                                                                 # μ's chart is (0, 1), not the run's (0.6, 1)
        _lm_views!(s2.f, s2.u0; stops = RC._calibration_stops(; max_iter = 1), damping = VgtkCore.LevenbergDamping(), height_degree = 1)
        @test !isfile(log)                                                                   # the run's log is the run's
        # … and the run's own problem has them: Θ_m held inside the riding block, μ on the run's bounds
        lp = RC.refinement_structure(settings, RC.PerViewRoute())
        s3 = fresh()
        prob3, idx3 = build_problem(s3.f, s3.u0; lp.kw...)
        mv3 = material_var(prob3)
        @test idx3.free[idx3.material]                                                       # the block stays variable (the warping rides)
        m3 = NL.update(mv3, increment(mv3))
        @test RC.nonlinear_parameters(m3.mat) == RC.nonlinear_parameters(mv3.mat) && m3.mat.diffuse_weight == mv3.mat.diffuse_weight && m3.surface.h != mv3.surface.h
        s4 = fresh(); s4.f.mu_bounds = settings.material.mu_bounds
        mv4 = material_var(first(build_problem(s4.f, s4.u0)))
        @test NL.update(mv4, increment(mv4)).mat.diffuse_weight >= 0.6                                         # the chart's lower end, the run's 0.6
        s4.f.start = meta
        _lm_views!(s4.f, s4.u0; stops = RC._calibration_stops(; max_iter = 1), damping = VgtkCore.LevenbergDamping(),
                   callback = RC.calibration_checkpoints(RC.VgtkCore.RefineProblem(s4.f, s4.u0), run))   # the run's log: its driver's callback
        @test isfile(log) && length(readlines(log)) == 1
    end
    # the settings' globals are gone (deleted, not scoped)
    @test !any(n -> isdefined(RC, n), (:FREEZE_MATERIAL, :MU_BOUNDS, :JAC_CHUNK, :LM_LOG, :with_freeze_material))
    end
end
