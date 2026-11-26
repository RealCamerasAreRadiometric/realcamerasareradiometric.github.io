# The refinement's start as its own step (start/start_record.jl): the start options (the refinement-only keys at their
# defaults, so a refine-only change never changes a start), the start record's round trip, the refusal of a start of other options,
# and the retired one-step entry points.
@testitem "refinement start: start options, the start record, refusals" setup = [RCFixtures] begin
    using RadiometricCamera: material, start_options
    using StaticArrays, LinearAlgebra

    module RefinementStartTest
    using Test, RadiometricCamera, StaticArrays, LinearAlgebra
    const OLD_STACK = false
    include(joinpath(@__DIR__, "..", "regression", "golden_scenes.jl"))
    const RC = RCg

    @testset "start_options: exactly the refinement-only keys (and the outputs) at their defaults" begin
        d0 = RC.config_dict(RC.RadiometricCalibration(RC.PipelineOptions()), RC._calibration_stops())   # the options, the optimizer's table
        for k in RC.REFINE_ONLY_KEYS                                    # every key is an option of the pipeline
            t, key = split(k, '.'; limit = 2)
            @test haskey(d0, t) && haskey(d0[t], key)
        end
        o = RC.pipeline_options(Dict("lm" => Dict("max_iter" => 50, "max_time" => 99.0), "fixed" => ["material"],
                                     "sweep" => Dict("lm_chunk" => 3, "iterations" => 2), "io" => Dict("out" => "/tmp/x", "checkpoint_each" => "on"),
                                     "resume" => Dict("iteration_offset" => 4, "sweep" => "on", "state" => "s.json"),
                                     "grid" => Dict("lattice" => "node_count", "node_count" => 875)))
        so = RC.start_options(o)
        @test so == RC.start_options(RC.pipeline_options(Dict("fixed" => ["material"], "grid" => Dict("lattice" => "node_count", "node_count" => 875),
                                                              "resume" => Dict("state" => "s.json"))))
        @test so.io.out == "" && !so.io.checkpoint_each                    # (the optimizer's table is the algorithm's, no option: RC 0.4.0)
        @test so.fixed == ["material"] && so.grid.node_count == 875 && so.resume.state == "s.json"   # every other key kept
        # a refine-only change: the same start options; a start change: others
        @test RC.start_options(RC.pipeline_options(Dict("lm" => Dict("max_iter" => 24)))) == RC.start_options(RC.pipeline_options(Dict("lm" => Dict("max_iter" => 50))))
        # the LM's stop tolerances are the refinement's alone (RC 0.11.4, code review): a change never re-keys or refuses a start
        @test RC.start_options(RC.pipeline_options(Dict("lm" => Dict("ftol" => 1e-3, "xtol_rel" => 1e-6)))) == RC.start_options(RC.PipelineOptions())
        @test all(k -> k in RC.REFINE_ONLY_KEYS, ("optimizer.ftol", "optimizer.xtol_rel", "optimizer.max_iter", "optimizer.max_time", "optimizer.type"))
        @test RC.start_options(RC.pipeline_options(Dict("grid" => Dict("lattice" => "node_count", "node_count" => 900)))) != RC.start_options(RC.PipelineOptions())
    end

    @testset "the start record round-trips; a start of other options is refused" begin
        W = H = 32
        kb = cam(RC.VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0), 41.216, (16.32, 15.36), W, H)
        sc = gold_board(W, H; h = gold_h())
        s = gold_state(sc..., kb)
        names = ["V01.png", "V02.png"]
        meta = (; names, W₀ = W, H₀ = H, fpx = 41.216, lens_start = s.camera.geometric, init_region = nothing, R0 = 0.0386,
                d_c = 1.31, f_m = 0.0041, be0 = [0.1, 0.25], page_flip = 1.0, board_y_sign = -1.0, vig_c = SVector(16.0, 15.5),
                costs = (; init = (2.0, 0.3), linear = (1.0, 0.2)), μ_lin = 0.61,
                vignetting_gcv = NamedTuple[(; β = 0.5, gcv = 1.25, dof = 3.5, n_free = 7, at_bracket_end = false)], grid_gcv = nothing,
                resample = nothing)
        o = RC.pipeline_options(Dict("lm" => Dict("max_iter" => 50), "io" => Dict("init" => "lens:x.json")))
        dir = mktempdir()
        RC.save_start(dir, s, meta, RC.start_options(o))
        st = RC.load_refinement_start(dir)
        @test st.names == names && (st.W₀, st.H₀) == (W, H) && st.parametric && st.fpx == 41.216 && st.R0 == 0.0386
        @test st.d_c == 1.31 && st.f_m == 0.0041 && st.be0 == [0.1, 0.25] && st.board_y_sign == -1.0 && st.vig_c == SVector(16.0, 15.5)
        @test st.costs == meta.costs && st.μ_lin == 0.61 && st.grid_gcv === nothing
        @test only(st.vignetting_gcv).β == 0.5 && only(st.vignetting_gcv).n_free == 7 && only(st.vignetting_gcv).at_bracket_end === false
        @test st.masks == [BitMatrix(m) for m in s.masks]                                      # the masks bit for bit
        # the start's own masks handed over (prepare_start_refinement): taken, not read from the PNGs; a wrong count refused
        given = [BitMatrix(m) for m in s.masks]
        @test RC.load_refinement_start(dir; masks = given).masks == given
        @test_throws ArgumentError RC.load_refinement_start(dir; masks = given[1:1])
        @test st.state.names == names
        @test all(k -> isapprox(st.state.poses[k].t, s.poses[k].t; atol = 1e-12) && isapprox(Matrix(st.state.poses[k].R), Matrix(s.poses[k].R); atol = 1e-12), eachindex(names))
        @test st.options == RC.start_options(o) && isfile(joinpath(dir, "lens.json")) && isfile(joinpath(dir, "frontend", "lens.json"))
        # a GCV selection round-trips bit for bit (code review 0.11.0, finding 4): its field order, a Bool flag, an Int count and the
        # smoothness's `kind` a Symbol — what a grid run whose weight is "gcv" records in its start and its parameters.json
        sel = (; β = 0.0123, gcv = 1.5e-3, dof = 41.25, at_bracket_end = false, kind = :pspline, n_observations = 1200, n_columns = 98,
                 linearization = "the grid's free patch blocks at the start")
        dir2 = mktempdir()
        RC.save_start(dir2, s, merge(meta, (; grid_gcv = sel, vignetting_gcv = NamedTuple[merge(only(meta.vignetting_gcv), (; m̄ = 2.5, rows = 14))])), RC.start_options(o))
        st2 = RC.load_refinement_start(dir2)
        @test typeof(st2.grid_gcv) == typeof(sel) && st2.grid_gcv == sel
        @test typeof(only(st2.vignetting_gcv)) == typeof(merge(only(meta.vignetting_gcv), (; m̄ = 2.5, rows = 14)))
        @test st.resample === nothing                                                            # a start that was not resampled
        # the exact resample's record (`exact_start`, RC 0.11.0) round-trips through start.json: its lattice, marks, folds, fidelity
        res = (; grid_size = [9, 8], subdivision = [1, 1], lattice_spacing = [110.0, 106.66666666666667], extension_offset = [1, 1],
                 extrapolated = [[true, false, true], [false, false, true]], extrapolated_fraction = 0.5417,
                 folds = (; folded = [[12.0, 40.0], [20.0, 40.0]], samples = 4941, unimaged = 0), max_px = 2.76e-13, max_px_edge = 2.18e-13,
                 rms = 7.53e-14, axis_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dir3 = mktempdir()
        RC.save_start(dir3, s, merge(meta, (; resample = res)), RC.start_options(o))
        r3 = RC.load_refinement_start(dir3).resample
        @test keys(r3) == keys(res) && all(getfield(r3, k) == getfield(res, k) for k in keys(res))
        @test r3.folds.samples === 4941 && r3.subdivision isa Vector{Int} && r3.lattice_spacing isa Vector{Float64}
        @test RC.check_start_options(st, RC.pipeline_options(Dict("lm" => Dict("max_iter" => 24), "io" => Dict("init" => "lens:x.json")))) === nothing
        e = try RC.check_start_options(st, RC.pipeline_options(Dict("io" => Dict("init" => "lens:y.json")))); nothing catch err; err end
        @test e isa ArgumentError && occursin("io.init", e.msg) && occursin("other start options", e.msg)
    end

    @testset "a start's streamed observations are kept only by a start computed now (code review 0.11.0, finding 6)" begin
        psrc = read(joinpath(pkgdir(RadiometricCamera), "src", "start", "prepare.jl"), String)
        @test occursin("observations = start === nothing ? (; masks = copy(masks), ϕ, ω, qs, sensor) : nothing", psrc)
        @test count("observations = (; masks = copy(masks)", psrc) == 0                   # never on a start record's refinement
    end

    @testset "a start record's derived values agree with its stored lens and poses (code review 0.11.0, finding 8)" begin
        rec = (; fpx = 1000.0, d_c = 0.31, be0 = [0.1, 0.2], dir = "the record")
        @test RC._recorded(rec, :fpx, 1000.0 * (1 + 1e-9)) === 1000.0                     # the record's value, to its rounding
        @test RC._recorded(rec, :be0, [0.1 + 1e-9, 0.2]) == [0.1, 0.2]
        e = try RC._recorded(rec, :fpx, 1001.0); nothing catch err; err end
        @test e isa ArgumentError && occursin("fpx", e.msg) && occursin("produce its start again", e.msg)
        @test_throws ArgumentError RC._recorded(rec, :be0, [0.1])                         # another count of bearing errors
        @test_throws ArgumentError RC._recorded(rec, :d_c, 0.35)
    end

    @testset "the retired one-step entry points are refused by name" begin
        e = try RC.pipeline_options(Dict("io" => Dict("save_init" => "s.json"))); nothing catch err; err end
        @test e isa ArgumentError && occursin("io.save_init` is retired", e.msg) && occursin("--stage=start", e.msg)
        # the RC 0.11.0 verbs replace the start-and-refinement driver API: the retired names are gone (an UndefVarError, no alias),
        # and RC adds no method to Core's refine! any more
        for n in (:prepare_refinement, :prepare_start, :prepare_start_refinement, :write_refinement, :calibrate!)
            @test !isdefined(RC, n)
        end
        @test !any(m -> m.module === RC, methods(RC.VgtkCore.refine!))
        @test isdefined(RC, :RadiometricCalibration) && isdefined(RC, :CalibrationCache) && isdefined(RC, :calibrate)
        @test !(:save_init in fieldnames(RC.IOOptions)) && !("io.save_init" in RC.PIPELINE_OUTPUT_KEYS)
    end
    end
end
