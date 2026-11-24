# The geometric-accuracy refinements (scripts/radiometric_refinement.jl, configs/experiments/radiometric_refinement.toml) and their chrome
# evaluation (scripts/chrome_evaluation.jl, one config per chrome set: configs/experiments/chrome_evaluation_fx3_<set>.toml):
# the configs load and plan; the case si30_bc5_warped carries the 03.11 run's values (the run of record); the evaluation names the
# case's refined model and its start, scores them on the set's own detect and index runs, and is keyed by the refinement's run; what is
# not the evaluate stage's own is refused; a missing upstream is produced on demand. No run: the refinements need the GPU and the published data.
@testitem "radiometric_refinement and its chrome evaluation" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    import VisualGeometryToolkitIO as VgtkIO
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("chrome_evaluation.jl"))

    rpath = projectdir("configs", "experiments", "radiometric_refinement.toml")
    tpath = projectdir("configs", "experiments", "chrome_evaluation_fx3_14mm_central30cm.toml")
    rc = load_radiometric_refinement_config(rpath)
    tc = load_chrome_evaluation_config(tpath)

    @testset "the refine stage produces its set's decode on demand (owner 2026-11-15: an upstream is produced, never refused)" begin
        # an empty results tree and an empty data root: the case's decode is missing, so the refinement makes it — its stage prepare,
        # which reads the set's RAW input, not there: the input's error, never "run its stage first"
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_DATA_ROOT" => mktempdir()) do
            err = try run_refine(rc, "si30_bc5_warped"; driver = "scripts/radiometric_refinement.jl", config_path = rpath); nothing catch e; e end
            @test err isa Exception && !occursin("run its stage", sprint(showerror, err))
            set = rc["cases"]["si30_bc5_warped"]["images"]
            @test !produced(prepare_product(rc, set)) && !produced(refine_product(rc, "si30_bc5_warped"))
            @test require_prepared(rc, Dict{String,Any}("kind" => "lens"), "a table without images"; driver = "d", config_path = rpath) === nothing   # reads no decode
        end
    end

    @testset "the case si30_bc5_warped: the 03.11 run's values" begin
        t = rc["cases"]["si30_bc5_warped"]
        @test t["kind"] == "lens"
        o = RadiometricCamera.pipeline_options(t["pipeline"])
        @test o.frontend.type === RadiometricCamera.SchoepsFrontend && o.frontend.page_mm == (1000.0, 1000.0)
        @test o.frontend.texture_features_size == (4000, 4000)
        @test o.camera.sensor_mm == 35.6 && o.camera.f_mm == 14.6776 && o.camera.fnumber == 22.0 && !("aperture" in o.fixed)
        @test o.masks.samples_per_view == 8192 && o.masks.gradient_bias == 0.0 && o.masks.edge_fraction == 0.0 && o.masks.addon_samples == 0
        @test o.rendering.N == 16 && o.rendering.n_p == 2 && o.rendering.hammersley
        @test (o.material.alpha_ink, o.material.alpha_noink, o.material.eta_ink, o.material.eta_noink, o.material.mu) == (0.6, 0.6, 2.0, 1.3, 0.244)
        @test o.vignetting.model === RadiometricCamera.VrtkCore.RadialVignetting && o.vignetting.node_count == 19 && o.vignetting.chart === :projective
        @test o.warping.degree == 1 && o.edge.print_sigma_mm == 0.0277 && o.lm.max_iter == 25 && isempty(o.fixed)
        @test o.io.checkpoint_each && o.io.channel == "G" && o.compute.jacobian_chunk == 4 && runs_on_gpu(t)
        # the inputs the case's table reads key its run, each a declared input; the photographs are the prepare run's (absolute)
        p = resolve_pipeline(rc, t; produce = false)                     # the paths alone: nothing produced (the run inputs are produced only by a run)
        @test table_inputs(rc, p) == ["central30cm_14mm_si_green_bc5_cv2_json", "fx3_sensor_characterisation", "siemens_1m_board"]   # the FX3 record: keyed by its sha256
        @test table_runs(t["pipeline"]) == ["fx3_14mm_central30cm_schoeps_green",   # its Schöps start: the front ends' run (frontends.toml), keyed by its key
                                            "siemens_1m_distance_map"]               # its edge.sdf: the board's distance map (distance_maps.toml), keyed by its key
        # the photographs are the RAW input (a photosites set: its prepare run links the RAW files): RC reads them as RAW by their files
        # and observes their green photosites by their type — no flag (owner ruling 2026-11-13); one measurement per photosite: "G"
        @test !haskey(p["io"], "photosites") && startswith(p["io"]["images"], results_root()) && endswith(p["io"]["images"], "raw_14mm_siemens_central30cm_frames")
        @test RadiometricCamera.pipeline_options(p).io.channel == "G"         # RC's single-channel form, stated by the driver for a RAW set
        t2 = deepcopy(t); t2["pipeline"]["io"]["channel"] = "RGB"
        @test resolve_pipeline(rc, t2; produce = false)["io"]["channel"] == "RGB"              # a stated channel is the case's (RC refuses it at the RAW photograph)
        @test all(t -> t["mode"] == "photosites" && !haskey(t, "gain") && !haskey(t, "srgb"), values(rc["prepare"]))
        @test p["io"]["init"] == "lens:" * rc["inputs"]["central30cm_14mm_si_green_bc5_cv2_json"]["path"]
        @test rc["inputs"]["raw_14mm_siemens_central30cm"]["tier"] == "rc_raw_photos"
        @test length(rc["views"]["fx3_14mm_siemens_central30cm_si"]) == 132   # every view with ≥ 20 stars in its front end's start (frontends.toml [schoeps.central30cm_14mm_siemens], owner ruling 2026-11-13) …
        # … but the two the project's Schöps run gives 0 features (no start pose: RadiometricCamera.require_featured_views; owner 2026-11-14)
        @test isdisjoint(rc["views"]["fx3_14mm_siemens_central30cm_si"], ["DSC06034.png", "DSC06129.png"])
        # a view list keys the run; a provenance text does not
        c2 = deepcopy(rc); c2["views"]["fx3_14mm_siemens_central30cm_si"] = ["DSC06030.png"]
        @test refine_product(c2, "si30_bc5_warped").key != refine_product(rc, "si30_bc5_warped").key
        c2 = deepcopy(rc); c2["inputs"]["siemens_1m_board"]["provenance"] = "edited"
        @test refine_product(c2, "si30_bc5_warped").key == refine_product(rc, "si30_bc5_warped").key
        @test isempty(get(rc, "pending", Dict()))
        c2 = deepcopy(rc); c2["pending"] = Dict{String,Any}("x" => "y")     # a [pending] note is description: it never re-keys the run
        @test refine_product(c2, "si30_bc5_warped").key == refine_product(rc, "si30_bc5_warped").key
        # the defocus ablation's pinhole: sitla_spline_pinhole_warped is the Tamron f/8 central spline as the plain pinhole (r₀ = f / 2N = 1 µm,
        # f = 21.1975 mm, held: "aperture" in fixed) with the focus surface held. The 14 mm f/22 control (si30_spline_pinhole_warped) left the
        # experiment (owner 2026-11-15: tab:defocus is the Tamron and the zoom lens)
        @test !haskey(rc["cases"], "si30_spline_pinhole_warped") && !haskey(rc["cases"]["si30_spline_warped"]["pipeline"], "fixed")
        tp = rc["cases"]["sitla_spline_pinhole_warped"]["pipeline"]
        @test tp["camera"]["f_mm"] / (2 * tp["camera"]["fnumber"]) ≈ 1e-3 atol = 1e-6
        @test tp["fixed"] == ["focus", "aperture"] && !haskey(tp["camera"], "r_min")
        kp, ka = refine_product(rc, "sitla_spline_pinhole_warped").key, refine_product(rc, "sitla_spline_warped").key
        @test kp != ka && kp == refine_product(deepcopy(rc), "sitla_spline_pinhole_warped").key   # its own run key, deterministic
    end

    @testset "a refinement config refuses what is not its own" begin
        dir = mktempdir()
        good = TOML.parsefile(rpath)
        function refused(edit!)
            c = deepcopy(good); edit!(c)
            p = joinpath(dir, "radiometric_refinement.toml")
            open(io -> TOML.print(io, c), p, "w")
            return try load_radiometric_refinement_config(p); nothing catch err; err end
        end
        @test refused(identity) === nothing
        e = refused(c -> c["cases"]["si30_bc5_warped"]["kind"] = "spline");        @test e isa ArgumentError && occursin("spline", e.msg)
        e = refused(c -> c["cases"]["si30_bc5_warped"]["views"] = "nope");         @test e isa ArgumentError && occursin("views", e.msg)
        e = refused(c -> c["cases"]["si30_bc5_warped"]["images"] = "nope");        @test e isa ArgumentError && occursin("prepare.nope", e.msg)
        e = refused(c -> delete!(c["cases"]["si30_bc5_warped"], "kind"));          @test e isa ArgumentError && occursin("kind", e.msg)
        e = refused(c -> c["cases"]["si30_bc5_warped"]["pipeline"]["masks"]["edge_fractio"] = 0.3); @test e isa ArgumentError && occursin("edge_fractio", e.msg)
        e = refused(c -> c["cases"]["si30_bc5_warped"]["pipeline"]["io"]["out"] = "o"); @test e isa ArgumentError && occursin("io.out", e.msg)
        e = refused(c -> c["prepare"]["fx3_14mm_siemens_central30cm"]["raw"] = "siemens_1m_board"); @test e isa ArgumentError && occursin("stage prepare", e.msg)
        e = refused(c -> c["pending"] = Dict("x" => 1));                    @test e isa ArgumentError && occursin("pending", e.msg)
        # a [pending] entry makes the refine stage refuse
        c2 = deepcopy(rc); c2["pending"] = Dict{String,Any}("x" => "a value the run of record had")
        e = try refuse_pending(c2, rpath); nothing catch err; err end
        @test e isa ArgumentError && occursin("x", e.msg) && refuse_pending(rc, rpath) === nothing
    end

    @testset "the chrome evaluation: the case's refined model against its start" begin
        s = evaluate_settings(tc, tpath)
        @test s.index == "stage" && s.target == "target"
        @test (s.optimization_strategy_table, s.iterations, s.lens_coverage) == (Dict{String, Any}("type" => "triggs", "loss" => "huber"),   # at the efficiency table's constant
                                                                             80, "sensor") && !haskey(s, :scoring)    # the radial frame: the set's coverage (src/image_region.jl)
        @test profile_frame(VgtkCore.Rect2{Float64}(0, 0, 4256, 2848), 4256, 2848).centre == [2128.0, 1424.0]
        @test s.max_unassigned == 4                                                              # the two dots under the target's logo may be missing
        @test length(s.models) == 20 && [m["label"] for m in s.models[1:2]] == ["si30_bc5_start", "si30_bc5_refined"]   # one config per set: every model of the set a row (20 since the pinhole control left, 2026-11-15)
        @test haskey(tc, "detect") && haskey(tc, "index")                                        # the set's own registration
        # the start model is the case's own start (one file, the same input path in both configs)
        @test tc["inputs"]["bc5_si30"]["path"] == rc["inputs"]["central30cm_14mm_si_green_bc5_cv2_json"]["path"]
        # the set is one config: no per-model chrome config is left (2026-11-12)
        @test !any(f -> occursin(r"^chrome_evaluation_fx3_14mm_central30cm_", f), readdir(projectdir("configs", "experiments")))
        # the evaluation is keyed by the refinement's run and the inputs it reads
        eff = evaluate_product(tc).record
        @test eff["upstream"]["radiometric_refinement"]["si30_bc5_warped"] == refine_product(rc, "si30_bc5_warped").key
        @test eff["upstream"]["index"] == index_product(tc).key
        @test issubset(["bc5_si30", "target"], keys(eff["inputs"])) && all(c -> eff["upstream"]["radiometric_refinement"][c] == refine_product(rc, c).key,
                                                                         [m["case"] for m in s.models if haskey(m, "case")])
    end

    @testset "an evaluate config refuses what is not its own" begin
        dir = mktempdir()
        good = TOML.parsefile(tpath)
        function refused(edit!)
            c = deepcopy(good); edit!(c)
            p = joinpath(dir, "chrome_evaluation_fx3_14mm_central30cm.toml")
            open(io -> TOML.print(io, c), p, "w")
            return try load_chrome_evaluation_config(p); nothing catch err; err end
        end
        @test refused(identity) === nothing
        e = refused(c -> c["evaluate"]["models"][2]["case"] = "nope");         @test e isa ArgumentError && occursin("nope", e.msg)
        e = refused(c -> c["evaluate"]["max_unassigned"] = -1);             @test e isa ArgumentError && occursin("max_unassigned", e.msg)
        @test refused(c -> c["evaluate"]["max_unassigned"] = 3) === nothing                   # a tolerance of a few rim dots parses
        e = refused(c -> c["evaluate"]["models"][2]["kind"] = "grid");         @test e isa ArgumentError && occursin("writes a lens", e.msg)
        e = refused(c -> c["evaluate"]["models"][1]["case"] = "si30_bc5_warped");     @test e isa ArgumentError && occursin("one of input, case", e.msg)
        e = refused(c -> c["evaluate"]["models"][1]["spec"] = "lens:/abs.json"); @test e isa ArgumentError && occursin("spec", e.msg)
        e = refused(c -> c["evaluate"]["models"][2]["label"] = "si30_bc5_start"); @test e isa ArgumentError && occursin("unique", e.msg)
        e = refused(c -> c["evaluate"]["image_size"] = [4256, 2848]);          @test e isa ArgumentError && occursin("image_size", e.msg)
        e = refused(c -> delete!(c["evaluate"], "profile_bins"));              @test e isa ArgumentError && occursin("profile_bins", e.msg)
        e = refused(c -> delete!(c, "upstream"));                              @test e isa ArgumentError && occursin("radiometric_refinement", e.msg)
        e = refused(c -> c["upstream"]["radiometric_refinement"] = "configs/x.toml"); @test e isa ArgumentError && occursin("configs/x.toml", e.msg)
        e = refused(c -> c["upstream"]["night_stars"] = "configs/experiments/night_stars.toml"); @test e isa ArgumentError && occursin("night_stars", e.msg)
    end

    @testset "--plan of both drivers lists their stages" begin
        for (script, config, stages) in (("radiometric_refinement", rpath, ("prepare", "detect", "start", "material_grid", "refine")),
                                         ("chrome_evaluation", tpath, ("detect", "index", "evaluate")))
            cmd = `$(Base.julia_cmd()) --project=$(projectdir()) $(projectdir("scripts", "$script.jl")) --config=$config --plan`
            out = IOBuffer()
            ok = success(pipeline(addenv(cmd, "VGTK_DATA_ROOT" => mktempdir(), "VGTK_RESULTS_ROOT" => mktempdir()); stdout = out, stderr = out))
            o = String(take!(out))
            ok || @info o
            @test ok
            for st in stages
                @test occursin(Regex("^\\s+$st\\s+(check|\\d+/\\d+ produced)", "m"), o)
            end
        end
    end
end
