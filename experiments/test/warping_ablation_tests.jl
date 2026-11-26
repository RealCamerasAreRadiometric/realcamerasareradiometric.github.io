# The board warping ablation (scripts/warping_ablation.jl, configs/studies/warping_ablation.toml, src/warping_ablation.jl): the
# config of record loads; its cases are ENUMERATED from the paper's chrome sets (every global-model case of a capture set that has one,
# and its central spline), each paired with its start; each mode is an override of the case of record (`override_variant`: the
# reference mode IS the case, so its run key is the run of record's; the others name their keys, the in-plane start deleted); the
# products are keyed by what they read; the record round-trips through VisualGeometryToolkitIO and its table and figure are written.
# No real run: the refinements need the GPU and the published data.
# SKIPPED (rc-0116, a study): the ablation enumerates every global-model case as a case "with the heights and the in-plane print
# error", the board model of record when it was written; since the owner's decision of 2026-11-16 (heights only for the global
# models, `decided` in configs/studies/warping_ablation.toml) the configs of record refuse that enumeration, and the study never ran.
# To run it, base it on the configs of v0.11.5 (or widen its enumeration): kept as code, skipped as a test.
@testitem "warping ablation: the cases enumerated, the modes as overrides, the products, the record through IO, the figure" default_imports = false skip = true begin
    using Test
    using DrWatson
    using TOML: TOML
    import VisualGeometryToolkitIO as VgtkIO
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("chrome_evaluation.jl"))
    include(srcdir("figure_style.jl"))
    include(srcdir("warping_ablation.jl"))

    path = projectdir("configs", "studies", "warping_ablation.toml")
    cfg = load_warping_ablation_config(path)
    @test sort(collect(keys(cfg["modes"]))) == ["both", "heights", "rigid"] && cfg["ablation"]["reference"] == "both"
    @test occursin("# RATIONALE", read(path, String))                         # the owner's rationale block, comments only

    @testset "a reference that sets keys, a key both set and unset, an unknown key, refused" begin
        bad(f) = (c = TOML.parsefile(path); f(c); p = joinpath(mktempdir(), "warping_ablation.toml"); open(io -> TOML.print(io, c), p, "w"); p)
        @test_throws ArgumentError load_warping_ablation_config(bad(c -> c["ablation"]["reference"] = "rigid"))
        @test_throws ArgumentError load_warping_ablation_config(bad(c -> c["modes"]["heights"]["unset"] = ["warping.inplane"]))
        @test_throws ArgumentError load_warping_ablation_config(bad(c -> c["ablation"]["colour"] = "red"))
        @test_throws ArgumentError load_warping_ablation_config(bad(c -> c["ablation"]["stats"] = ["max_px"]))
    end

    rc = warping_refinements(cfg)
    cases = warping_cases(cfg, rc)
    @testset "the cases: enumerated from the paper's chrome sets, never listed" begin
        # every global model of the capture sets with one, and their central spline; not the Tamron f/8 or the zoom (spline only), not the excluded 27.10 fisheye
        @test sort(unique([c.prefix for c in cases])) == ["ag08ta", "ag10fe", "ag30", "si17ta", "si30"]
        @test sort(unique([c.chrome for c in cases])) == ["fisheye_tenth", "fourteen", "tamron_ag08", "tamron_si17"]
        for p in ("ag08ta", "ag10fe", "ag30", "si17ta", "si30")
            @test sort([c.family for c in cases if c.prefix == p]) == ["bc5", "ds", "kb", "ocv14", "spline"]
        end
        @test all(c -> (c.family == "spline") == (c.kind == "grid"), cases)
        @test all(c -> c.start.label == "$(c.prefix)_$(c.family)_start" && c.case == "$(c.prefix)_$(c.family)_warped", cases)
        @test only(c for c in cases if c.case == "si30_spline_warped").start.kind == "schoeps"   # the spline's start is Schöps's own camera
    end

    plan = warping_plan(cfg; rc, cases)
    @testset "the modes: overrides of the case of record; the reference IS the case" begin
        @test length(plan.rows) == 3 * length(cases)
        r(case, m) = only(x for x in plan.rows if x.case == case && x.mode == m)
        @test r("si17ta_kb_warped", "both").variant == "si17ta_kb_warped"
        @test plan.rc["cases"]["si17ta_kb_warped"] == rc["cases"]["si17ta_kb_warped"]
        @test refinement_product(plan.rc, "si17ta_kb_warped").key == refinement_product(rc, "si17ta_kb_warped").key   # the run of record, by its key
        a = r("si17ta_kb_warped", "rigid"); b = r("si17ta_kb_warped", "heights")
        @test a.variant == "si17ta_kb_warped__warping.degree=0__warping.inplane=none__warping.inplane_start=unset"
        @test b.variant == "si17ta_kb_warped__warping.inplane=none__warping.inplane_start=unset"
        wa = plan.rc["cases"][a.variant]["pipeline"]["warping"]; wb = plan.rc["cases"][b.variant]["pipeline"]["warping"]
        @test wa["degree"] == 0 && wa["inplane"] == "none" && !haskey(wa, "inplane_start")
        @test wb["degree"] == rc["cases"]["si17ta_kb_warped"]["pipeline"]["warping"]["degree"] && wb["inplane"] == "none" && !haskey(wb, "inplane_start")
        o = pipeline_options(plan.rc["cases"][a.variant]["pipeline"])
        @test o.warping.degree == 0 && o.warping.inplane == "none"
        @test length(unique(refinement_product(plan.rc, x.variant).key for x in plan.rows)) == length(plan.rows)
        @test a.label == "si17ta_kb_rigid" && r("si30_spline_warped", "heights").label == "si30_spline_heights"
        # nothing else of the case changes
        for x in (a, b)
            t = deepcopy(plan.rc["cases"][x.variant]); t0 = deepcopy(rc["cases"]["si17ta_kb_warped"])
            delete!(t["pipeline"], "warping"); delete!(t0["pipeline"], "warping")
            @test t == t0
        end
        # the slow lattice (a node count subdividing Schöps's lattice) and a case of record without heights are refused
        t = deepcopy(rc["cases"]["si30_spline_warped"]); t["pipeline"]["grid"]["lattice"] = "node_count"; t["pipeline"]["grid"]["node_count"] = 4000
        @test_throws ArgumentError check_case_of_record(t, "si30_spline_warped")
        t = deepcopy(rc["cases"]["si17ta_kb_warped"]); t["pipeline"]["warping"]["degree"] = 0
        @test_throws ArgumentError check_case_of_record(t, "si17ta_kb_warped")
    end

    @testset "the products: keyed by the protocol, the refinements and the index; the comparisons against the reference and the start" begin
        p = warping_evaluate_product(cfg, plan, "tamron_si17")
        cc = plan.sets["tamron_si17"]
        @test p.name == "evaluate" && p.record["upstream"]["index"] == index_product(cc).key
        rr = p.record["upstream"]["radiometric_refinement"]
        @test length(rr) == 15 && rr["si17ta_kb_warped"] == refinement_product(rc, "si17ta_kb_warped").key
        @test !haskey(p.record["chrome"]["tamron_si17"], "models") && p.record["chrome"]["tamron_si17"]["holdout"] == "on"
        cmp = [(c["model"], c["reference"]) for c in p.record["comparisons"]]
        @test ("si17ta_kb_rigid", "si17ta_kb_both") in cmp && ("si17ta_kb_heights", "si17ta_kb_both") in cmp && ("si17ta_kb_both", "si17ta_kb_start") in cmp
        @test !any(c -> c[1] == "si17ta_kb_both" && c[2] == "si17ta_kb_both", cmp) && length(cmp) == 5 * 5
        @test length(p.record["models"]) == 5 + 15
        @test p.key != warping_evaluate_product(cfg, plan, "tamron_ag08").key
        @test warping_figure_product(cfg, plan).record["upstream"]["record"] == warping_record_product(cfg, plan).key
    end

    @testset "the run summary: the LM's run, the board's heights and in-plane terms through IO" begin
        rd, sd = mktempdir(), mktempdir()
        VgtkIO.write_record(joinpath(rd, "parameters.json"),
            Dict{String, Any}("schema" => "rc/run_parameters/v3.0", "iterations" => 38, "retcode" => "Success", "stop_rule" => "CostDecrease", "seconds" => 744.0,
                              "cost" => Dict("init" => Dict("rel_rms" => 0.47), "after_refinement" => Dict("rel_rms" => 0.22)),
                              "board_warping" => Dict("free" => true), "inplane_print_error" => Dict("free" => true)))
        VgtkIO.write_record(joinpath(rd, "radiometric_target.json"),
            Dict{String, Any}("schema" => "rc/radiometric_target/v2.0", "warping" => [1.0, -2.0, 0.0, 1.0],
                              "inplane_print_error" => Dict("type" => "composition", "compose" => [Dict("type" => "aspect", "parameters" => [6e-4]), Dict("type" => "skew", "parameters" => [-2e-4])])))
        s = warping_run_summary(rd, sd, "lens")
        @test s.iterations == 38 && s.retcode == "Success" && s.rel_rms_final == 0.22 && s.rel_rms_start == 0.47
        @test s.heights_rms_mm ≈ sqrt(6 / 4) && s.heights_max_mm == 2.0 && s.inplane_aspect == 6e-4 && s.inplane_skew == -2e-4
    end

    @testset "the record round-trips through IO; the table and the figure" begin
        sc(m) = (; photos = 100, dots = 3000, median_px = m, mean_px = 1.2m, p95_px = 2m, rms_px = 1.3m, all_median_px = m, all_mean_px = 1.2m,
                 all_p95_px = 2m, all_rms_px = 1.3m, coverage_dots = 3000, coverage_fraction = 1.0, retcode = "Success")
        d(x) = (; difference = x, lo = x - 0.01, hi = x + 0.01, excludes_zero = abs(x) > 0.01)
        ds(x) = (; median_px = d(x), mean_px = d(x), p95_px = d(x), rms_px = d(x))
        run = (; rel_rms_start = 0.4, rel_rms_final = 0.2, iterations = 10, retcode = "Success", stop_rule = "CostDecrease", seconds = 1.0,
               heights_rms_mm = nothing, heights_max_mm = nothing, inplane_aspect = nothing, inplane_skew = nothing)
        med = Dict("rigid" => 0.20, "heights" => 0.24, "both" => 0.26)
        runs = [(; chrome = "tamron_si17", capture = "c", prefix = "si17ta", family = "kb", kind = "lens", case = "si17ta_kb_warped", mode = m,
                 variant_case = "v_$m", label = "si17ta_kb_$m", start_label = "si17ta_kb_start",
                 scores = Dict("motion_heldout" => sc(med[m]), "motion" => sc(med[m])), start_scores = Dict("motion_heldout" => sc(0.184), "motion" => sc(0.184)),
                 beats_start = false, vs_reference = m == "both" ? nothing : ds(med[m] - 0.26), vs_start = ds(med[m] - 0.184), run) for m in ("both", "heights", "rigid")]
        rec = (; schema = EXPERIMENT_RECORD_SCHEMAS.warping_ablation, headline = "motion_heldout", in_sample = "motion", reference = "both",
               stats = ["median_px", "mean_px", "p95_px", "rms_px"], modes = Dict("rigid" => "rigid board", "heights" => "heights only", "both" => "of record"), runs,
               winners = [(; chrome = "tamron_si17", prefix = "si17ta", family = "kb", best = "rigid", best_median_px = 0.20, start_median_px = 0.184,
                           beats_start = String[], best_vs_reference = d(-0.06))])
        f = joinpath(mktempdir(), WARPING_RECORD_FILE)
        VgtkIO.write_record(f, rec)
        r = load_warping_record(f)
        @test r["headline"] == "motion_heldout" && length(r["runs"]) == 3 && r["runs"][1]["vs_reference"] === nothing
        @test r["runs"][3]["scores"]["motion_heldout"]["median_px"] == 0.20 && r["winners"][1]["best"] == "rigid"
        rows = warping_table_rows(r)
        @test length(rows) == 3 && all(x -> length(x) == length(WARPING_TABLE_COLUMNS), rows)
        out = mktempdir()
        w = write_warping_figure(out, r, cfg["figure"])
        tex = read(w.fragment, String)
        @test occursin("symbolic x coords={si17ta-kb}", tex) && occursin("(si17ta-kb,0.20000)", tex) && occursin("(si17ta-kb,0.18400)", tex)
        @test all(k -> occursin("rc warping $k", tex), ("start", "rigid", "heights", "both")) && !occursin("width=", tex) && !occursin("height=", tex)
        tab = read(w.table, String)
        @test occursin("\\label{tab:warping-ablation}", tab) && occursin("-0.060 [-0.070, -0.050]*", tab) && occursin("& rigid \\\\", tab)
        @test isfile(w.preview) && occursin("rc warping both/.style", read(w.styles, String))
    end
end
