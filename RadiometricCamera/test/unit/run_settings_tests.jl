# A run's settings travel with its problem, never through module globals (the owner's rule, 2026-11-14: "These global constants
# should not exist"): the material hold (`"material"` in `lm.fixed`: the problem's `fixed`, the MaterialVar's type), the μ bounds
# (`material.mu_bounds`: the state's), the Jacobian chunk (`compute.jacobian_chunk`: the state's) and the LM log (`lm.log`: run_lm!'s
# keyword). A run that sets them leaves the next problem of the same session untouched.
@testitem "run settings: no leak into the next problem" setup = [RCFixtures] begin
    using RadiometricCamera: build_problem, load_state, material, run_lm!
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
    fresh() = (sc = gold_scene(W, H; h = gold_h()); gold_state(sc..., kb))
    material_var(prob) = only(v for v in prob.variables if v isa RC.MaterialVar)
    # a step of every coordinate of the material block, μ's far below its chart's centre
    increment(mv) = (u = fill(0.01, Int(NL.nvars(mv))); u[9] = -50.0; u)

    mktempdir() do dir
        log = joinpath(dir, "lm.tsv")
        opts = RC.pipeline_options(Dict("fixed" => ["material"], "lm" => Dict("log" => log), "warping" => Dict("degree" => 1),
                                        "material" => Dict("mu_bounds" => [0.6, 1.0]), "compute" => Dict("jacobian_chunk" => 7),
                                        "resume" => Dict("state" => joinpath(dir, "state.json")),
                                        "io" => Dict("out" => mkpath(joinpath(dir, "out")), "images" => dir),
                                        "selfcheck" => Dict("only" => "on")))
        # a run of these settings, stopped after it took them in: the preparation (its resume state is absent) and the
        # refinement (its resume reader)
        @test_throws Exception RC.start_geometry(opts)
        s = fresh()
        meta = (; names = ["a", "b"], W₀ = W, H₀ = H, fpx = 41.216, lens0 = s.camera.geometric, views0 = nothing,
                features = nothing, costs = (; linear = (0.0, 0.0)), grid_smoothness = nothing)
        s.start = meta
        @test_throws "stop" RC.refine_calibration!(s; options = opts, load_state = _ -> error("stop"))

        # the next problem of the session: the material free, μ on (0, 1), no LM log
        s2 = fresh()
        mv = material_var(first(build_problem(s2; height_degree = 1)))
        m2 = NL.update(mv, increment(mv))
        @test RC.Θ_m_nonlinear(m2.mat) != RC.Θ_m_nonlinear(mv.mat)                           # Θ_m moves
        @test m2.mat.μ < 0.5                                                                 # μ's chart is (0, 1), not the run's (0.6, 1)
        run_lm!(s2; maxiters = 1, height_degree = 1)
        @test !isfile(log)                                                                   # the run's log is the run's
        # … and the run's own problem has them: Θ_m held inside the riding block, μ on the run's bounds
        lp = RC._refinement_lm(opts, RC.PerViewRoute())
        s3 = fresh()
        prob3, idx3 = build_problem(s3; lp.kw...)
        mv3 = material_var(prob3)
        @test idx3.free[idx3.material]                                                       # the block stays variable (the warping rides)
        m3 = NL.update(mv3, increment(mv3))
        @test RC.Θ_m_nonlinear(m3.mat) == RC.Θ_m_nonlinear(mv3.mat) && m3.mat.μ == mv3.mat.μ && m3.surface.h != mv3.surface.h
        s4 = fresh(); s4.mu_bounds = opts.material.mu_bounds
        mv4 = material_var(first(build_problem(s4)))
        @test NL.update(mv4, increment(mv4)).mat.μ >= 0.6                                         # the chart's lower end, the run's 0.6
        run_lm!(s4; maxiters = 1, lm_log = log)
        @test isfile(log) && length(readlines(log)) == 1
    end
    # the settings' globals are gone (deleted, not scoped)
    @test !any(n -> isdefined(RC, n), (:FREEZE_MATERIAL, :MU_BOUNDS, :JAC_CHUNK, :LM_LOG, :with_freeze_material))
    end
end
