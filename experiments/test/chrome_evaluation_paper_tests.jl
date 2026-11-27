# The chrome evaluation's paper stages (scripts/chrome_evaluation_paper.jl on configs/experiments/chrome_evaluation_paper.toml;
# src/chrome_evaluation_paper.jl, the table writer src/figures.jl summary_rows / summary_table / write_summary_tables): the paper
# config of record — every chrome config with an [evaluate] table is a [chrome.<key>] of it and the other way round, every table row
# names a [chrome] key and labels its config evaluates (one row per line, so rows are appended mechanically) —, the table writer on
# a synthetic summary.tsv, tab:defocus's number macros, and the tables stage on fixture products (test/fixtures/chrome_paper)
# byte-identical to what the driver of record before the move (scripts/geometric_figures.jl) wrote from them
# (test/fixtures/chrome_paper/golden). The driver's --plan is the drivers item's (drivers_tests.jl).

@testitem "chrome evaluation paper config: the config of record and its tables" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("figures.jl"))
    using .Figures: SUMMARY_STATS

    E = projectdir("configs", "experiments")
    path = joinpath(E, "chrome_evaluation_paper.toml")
    cfg = TOML.parsefile(path); text = read(path, String)
    include(srcdir("experiments.jl")); include(srcdir("drivers.jl")); include(srcdir("radiometric_refinement.jl"))
    include(srcdir("chrome_evaluation.jl")); include(srcdir("chrome_evaluation_paper.jl"))
    @test load_chrome_paper_config(path) isa AbstractDict                         # the driver's own checks: keys, rows, stats, panels
    @test occursin("\n# --- end of [chrome] ---\n", text)
    excluded = Set(basename(c) for e in values(excluded_sets()) for c in get(e, "configs", String[]))   # an excluded set's config: in no table
    evaluating = Set(f for f in readdir(E) if startswith(f, "chrome_evaluation_fx3_") && endswith(f, ".toml") && !(f in excluded) &&
                                                occursin("\n[evaluate]\n", read(joinpath(E, f), String)))
    @test Set(basename(t["config"]) for t in values(cfg["chrome"])) == evaluating      # every evaluating chrome config, and no other
    @test all(isfile(projectdir(t["config"])) for t in values(cfg["chrome"]))
    @test length(cfg["chrome"]) == 7                                               # one per chrome set (2026-11-12; the zoom lens's, 2026-11-13; fisheye 27.10 excluded 2026-11-15; fisheye 30 cm, 2026-11-16)
    models(key) = Set(String(m["label"]) for m in TOML.parsefile(projectdir(cfg["chrome"][key]["config"]))["evaluate"]["models"])
    @test "chrome_all" in keys(cfg["tables"]) && "defocus" in keys(cfg["tables"])
    for (name, t) in cfg["tables"]
        @test haskey(t, "rows") && !isempty(t["rows"])
        @test all(st -> st in SUMMARY_STATS, get(t, "stats", collect(SUMMARY_STATS)))
        seen = Set{Tuple{String,String}}()
        for r in t["rows"]
            @test setdiff(Set(keys(r)), ["macro", "camera"]) == Set(["set", "model", "chrome", "start", "refined"])   # (`macro`: a numbered table's rows; `camera`: optional)
            @test haskey(cfg["chrome"], r["chrome"])
            labels = models(r["chrome"])
            @test r["refined"] in labels
            @test isempty(r["start"]) || r["start"] in labels
            @test !((r["set"], r["model"]) in seen)
            push!(seen, (r["set"], r["model"]))
        end
        m = match(Regex("\\[tables\\.$name\\].*?rows = \\[\\n((?:    \\{[^\\n]*\\},\\n)+)\\]\\n", "s"), text)   # one row per line, `]` alone
        @test m !== nothing && count("\n", m.captures[1]) == length(t["rows"])
    end
    # the chrome_all table reaches every set the configs evaluate; the defocus table pairs the Tamron f/8 refinements
    sets = Set(r["set"] for r in cfg["tables"]["chrome_all"]["rows"])
    # the geometric-accuracy sets and the Tamron f/8 of the defocus experiment
    @test sets == Set(["14 mm · central_30cm · Siemens stars", "14 mm · central_30cm · AprilGrid", "fisheye · 10 cm · AprilGrid", "Tamron · f/8 · Siemens stars",
                "Tamron · 17 cm · Siemens stars", "Tamron · AprilGrid · 2026-11-08",
                "fisheye · 30 cm · Siemens stars"])   # the fisheye 30 cm set (2026-11-16; its OpenCV-14 and Double Sphere starts 2026-11-17)
    tam = [r for r in cfg["tables"]["defocus"]["rows"] if r["chrome"] == "tamron_la"]
    # every Tamron f/8 row names its geometric start (sitla_dm_<model>_start, the same for the pipeline and the pinhole case); since
    # the central spline's two refinements alone (the author's choice), the paper's
    # tab:defocus (layout blocks)
    @test length(tam) == 2 && all(r -> r["start"] == replace(replace(r["refined"], "_pinhole_refined" => "_start"), "_refined" => "_start"), tam) &&
          count(r -> endswith(r["refined"], "_pinhole_refined"), tam) == 1 && all(r -> r["chrome"] == "tamron_la", tam)
    # the zoom lens's pair (Ties 2026-11-13: Schöps's start, ours with the aperture and the focus surface, the pinhole): one start
    zoom = [r for r in cfg["tables"]["defocus"]["rows"] if r["chrome"] == "zoom48"]
    # and, since 2026-11-17 (owner GO 05:10, Table 3 as T. Harder defined it), the same pair from OpenCV's Brown–Conrady start
    @test length(zoom) == 4 && Set(r["start"] for r in zoom) == Set(["sizl_spline_start", "sizl_bc5_start"]) &&
          all(s -> count(r -> r["start"] == s && endswith(r["refined"], "_pinhole_refined"), zoom) == 1, ("sizl_spline_start", "sizl_bc5_start"))
    @test cfg["tables"]["defocus"]["layout"] == "blocks" && cfg["tables"]["defocus"]["label"] == "tab:defocus"
end

@testitem "chrome tables: the writer on a synthetic summary" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("figures.jl"))
    using .Figures: SUMMARY_STATS, TABLE_COLUMNS, summary_rows, summary_table, write_summary_tables, row_retcodes, row_converged

    dir = mktempdir()
    cols = ("model", "variant", "photos", "dots", "coverage_dots", "coverage_fraction", "median_px", "mean_px", "p95_px", "rms_px",   # EVALUATE_SUMMARY_COLUMNS
            "median_mrad", "centre_median_px", "centre_mean_px", "edge_median_px", "edge_mean_px",
            "all_photos", "all_median_px", "all_mean_px", "all_p95_px", "all_rms_px", "c_mm", "d_mm", "retcode")
    csv = joinpath(dir, "summary.tsv")                                      # VisualGeometryToolkitEval's table, as the evaluate stage writes it
    tsv(r) = split(r, ',')
    # the head-to-head on the common dots, each model's coverage (the dots it scores, their fraction) and its score on all of them
    Figures.write_tsv(csv, cols, tsv.(["a_start,motion,40,11000,12000,0.8,0.5,0.6,1.2,0.8,0.1,0.4,0.5,0.7,0.8,41,0.55,0.65,1.3,0.85,1.0/2.0/3.0,4.0/5.0/6.0,Success",
                                       "a_refined,motion,42,11500,15000,1.0,0.25,0.3,0.6,0.4,0.05,0.2,0.25,0.35,0.4,42,0.26,0.31,0.62,0.41,1.0/2.0/3.0,4.0/5.0/6.0,Success",
                                       "a_start,free,40,11000,12000,0.8,0.45,0.55,1.1,0.75,0.1,,,,,40,0.5,0.6,1.2,0.8,,,Success",
                                       "b_refined,motion,10,2000,2000,0.5,2.0,2.5,4.0,3.0,0.3,,,,,10,2.1,2.6,4.2,3.1,,,MaxIters"]))
    rows = summary_rows(csv)
    @test Set(keys(rows)) == Set([("a_start", "motion"), ("a_refined", "motion"), ("a_start", "free"), ("b_refined", "motion")])
    @test rows[("a_start", "motion")]["rms_px"] == 0.8 && rows[("a_start", "motion")]["photos"] == 40 && rows[("a_start", "free")]["centre_median_px"] == ""
    tr = [(; set = "S", model = "A", variant = "motion", start = rows[("a_start", "motion")], refined = rows[("a_refined", "motion")], refinement_retcode = "Success"),
          (; set = "T", model = "B", variant = "motion", start = nothing, refined = rows[("b_refined", "motion")], refinement_retcode = "Success")]
    # the convergence record (2026-11-12): B's chrome evaluation stopped at its iteration cap — recorded (tsv/json), never printed (owner 2026-11-16)
    @test row_retcodes(tr[1]) == ("Success", "Success", "Success") && row_converged(tr[1])
    @test row_retcodes(tr[2]) == ("", "MaxIters", "Success") && !row_converged(tr[2])
    @test !row_converged(merge(tr[1], (; refinement_retcode = "MaxTime")))            # a refinement that did not converge: its row recorded not converged
    @test row_converged(merge(tr[1], (; refinement_retcode = "Terminated")))          # Core's successful_retcode, not a name match
    @test !row_converged(merge(tr[1], (; refinement_retcode = "unrecorded")))         # no record: not converged
    @test row_converged(merge(tr[1], (; refinement_retcode = "converged")))           # a legacy spelling (before Core 0.14): IO's termination_from_legacy
    @test !row_converged(merge(tr[1], (; refinement_retcode = "max_iterations")))
    @test_throws ArgumentError row_converged(merge(tr[1], (; refinement_retcode = "no_model")))   # no ReturnCode, no legacy spelling: refused
    long = summary_table(tr)
    @test length(long) == 8 && [t.stat for t in long[1:4]] == collect(SUMMARY_STATS)
    @test long[4].start == 0.8 && long[4].refined == 0.4 && long[4].change_pct ≈ -50 && long[4].photos_start == 40 && long[4].photos_refined == 42 && long[4].dots_refined == 11500 &&
          (long[4].coverage_dots_start, long[4].coverage_dots_refined, long[4].coverage_fraction_start, long[4].coverage_fraction_refined) == (12000, 15000, 0.8, 1.0) &&
          long[4].refined_all == 0.41 && long[4].photos_refined_all == 42
    @test isnan(long[5].start) && isnan(long[5].change_pct) && long[5].refined == 2.0 && long[5].photos_start == 0 && long[5].photos_refined == 10
    @test length(summary_table(tr; stats = ("rms_px",))) == 2
    out = mktempdir()
    fmt(x) = string(round(x; sigdigits = 3)); pct(x) = string(round(x; sigdigits = 2))
    paths = write_summary_tables(out, "demo", tr; number = fmt, change = pct)
    @test paths == (joinpath(out, "table_demo.tsv"), joinpath(out, "table_demo.md"), joinpath(out, "table_demo.tex"))
    @test readline(paths[1]) == join(TABLE_COLUMNS, '\t')
    A = Figures.read_summary(paths[1])
    @test length(A) == 8
    @test [A[4][c] for c in TABLE_COLUMNS] == Any["S", "A", "motion", "rms_px", 0.8, 0.4, -50.0, 40, 42, 11000, 11500, "Success", "Success", "Success", "true",
                                                      12000, 15000, 0.8, 1.0, 0.41, 42]
    @test [A[5][c] for c in TABLE_COLUMNS[1:4]] == Any["T", "B", "motion", "median_px"] && isnan(A[5]["start"]) && isnan(A[5]["change_pct"]) && A[5]["refined"] == 2.0 &&
          [A[5][c] for c in TABLE_COLUMNS[8:11]] == Any[0, 10, 0, 2000] && [A[5][c] for c in TABLE_COLUMNS[12:15]] == Any["", "MaxIters", "Success", "false"] &&
          A[5]["coverage_dots_start"] == 0 && isnan(A[5]["coverage_fraction_start"]) && A[5]["coverage_fraction_refined"] == 0.5 && A[5]["refined_all"] == 2.1
    md = read(paths[2], String)
    @test occursin("## S\n", md) && occursin("## T\n", md)
    @test occursin("| A | 0.5 | 0.25 | -50.0 | 0.26 | 0.6 | 0.3 | -50.0 | 0.31 | 1.2 | 0.6 | -50.0 | 0.62 | 0.8 | 0.4 | -50.0 | 0.41 | 40 / 42 | 11000 / 11500 | 80.0 / 100.0 |", md)
    @test occursin("| B |  | 2.0 |  | 2.1 |  | 2.5 |  | 2.6 |  | 4.0 |  | 4.2 |  | 3.0 |  | 3.1 | 10 | 2000 | 50.0 |", md)
    tex = read(paths[3], String)
    @test count("\\begin{table}", tex) == 2 && occursin("\\caption{S:", tex) && occursin("\\caption{T:", tex)
    @test occursin("A & 0.5 & 0.25 & -50.0 & 0.26 & 0.6 & 0.3 & -50.0 & 0.31 & 1.2 & 0.6 & -50.0 & 0.62 & 0.8 & 0.4 & -50.0 & 0.41 & 40 / 42 & 11000 / 11500 & 80.0 / 100.0 \\\\", tex)
    @test occursin("B &  & 2.0 &  & 2.1 &  & 2.5 &  & 2.6 &  & 4.0 &  & 4.2 &  & 3.0 &  & 3.1 & 10 & 2000 & 50.0 \\\\", tex)
    @test !occursin("converge", tex) && !occursin("†", md) && !occursin("\\dag", tex)                  # the paper reports no convergence
    @test occursin("\\multicolumn{4}{c}{RMS [px]}", tex) && occursin("\\toprule", tex) && occursin("\\bottomrule", tex) && occursin("\\cmidrule(lr){14-17}", tex) &&
          occursin("start & refined & \\% & all", tex) && occursin("coverage [\\%]", tex)
    # LaTeX specials in the labels are escaped; the md and tex of a run are what plotsdir() gets too (the driver writes both)
    tr2 = [(; set = "set_2 & more", model = "m%", variant = "motion", start = nothing, refined = rows[("a_refined", "motion")], refinement_retcode = "Success")]
    tex2 = read(write_summary_tables(mktempdir(), "esc", tr2; number = fmt, change = pct)[3], String)
    @test occursin("m\\% &", tex2) && occursin("\\caption{set\\_2 \\& more:", tex2)
    # the paper's layout: ONE table* with a header row per set, no photos / dots, its caption and label
    tex3 = read(write_summary_tables(mktempdir(), "blk", tr; number = fmt, change = pct, layout = "blocks", label = "tab:x", caption = "C.")[3], String)
    @test count("\\begin{table*}", tex3) == 1 && count("\\end{table*}", tex3) == 1 && !occursin("\\begin{table}", tex3)
    @test occursin("\\multicolumn{18}{l}{\\textit{S}} \\\\", tex3) && occursin("\\multicolumn{18}{l}{\\textit{T}} \\\\", tex3)
    @test occursin("A & 0.5 & 0.25 & -50.0 & 0.26 & 0.6 & 0.3 & -50.0 & 0.31 & 1.2 & 0.6 & -50.0 & 0.62 & 0.8 & 0.4 & -50.0 & 0.41 & 80.0 / 100.0 \\\\", tex3)
    @test occursin("\\caption{C.}", tex3) && occursin("\\label{tab:x}", tex3) && !occursin("photos", tex3)
    @test occursin("\\toprule", tex3) && occursin("\\midrule", tex3) && occursin("\\bottomrule", tex3) && !occursin("\\hline", tex3) && count("\\addlinespace", tex3) == 1   # booktabs rules, a space between the sets
    @test_throws ArgumentError write_summary_tables(mktempdir(), "bad", tr; layout = "wide")
    # a non-improvement marked (worse_color): row A improves (no mark); a row whose refinement is worse gets its refined value and change marked
    tw = [(; set = "S", model = "W", variant = "motion", start = rows[("a_refined", "motion")], refined = rows[("a_start", "motion")], refinement_retcode = "Success")]
    tex4 = read(write_summary_tables(mktempdir(), "worse", vcat(tr[1:1], tw); number = fmt, change = pct, layout = "blocks", worse_color = "red")[3], String)
    @test occursin("W & 0.25 & \\textcolor{red}{\\textbf{0.5}} & \\textcolor{red}{\\textbf{100.0}} &", tex4)
    @test occursin("A & 0.5 & 0.25 & -50.0 &", tex4) && count("\\textcolor{red}{\\textbf{", tex4) == 8
    # the held-out headline (owner 2026-11-15, ruling on the columns): a table with an in-sample variant prints, per statistic, start,
    # refined and Δ % of the headline variant, then the refinement's in-sample score and the gap (headline − in-sample, px), then all
    using .Figures: IN_SAMPLE_COLUMNS, has_in_sample, in_sample_retcode
    Figures.write_tsv(csv, cols, tsv.(["a_start,motion_heldout,40,11000,12000,0.8,0.5,0.6,1.2,0.8,0.1,0.4,0.5,0.7,0.8,41,0.55,0.65,1.3,0.85,1.0/2.0/3.0,4.0/5.0/6.0,Success",
                                       "a_refined,motion_heldout,42,11500,15000,1.0,0.3,0.36,0.7,0.5,0.05,0.2,0.25,0.35,0.4,42,0.31,0.37,0.72,0.51,1.0/2.0/3.0,4.0/5.0/6.0,Success",
                                       "a_refined,motion,42,11500,15000,1.0,0.25,0.3,0.6,0.4,0.05,0.2,0.25,0.35,0.4,42,0.26,0.31,0.62,0.41,1.0/2.0/3.0,4.0/5.0/6.0,Success"]))
    hrows = summary_rows(csv)
    th = [(; set = "S", model = "A", variant = "motion_heldout", start = hrows[("a_start", "motion_heldout")], refined = hrows[("a_refined", "motion_heldout")],
             in_sample = hrows[("a_refined", "motion")], refinement_retcode = "Success")]
    @test has_in_sample(th) && !has_in_sample(tr) && in_sample_retcode(th[1]) == "Success" && in_sample_retcode(tr[1]) == ""
    @test !row_converged(merge(th[1], (; in_sample = merge(hrows[("a_refined", "motion")], Dict("retcode" => "MaxIters")))))   # the in-sample fit counts too
    lh = summary_table(th)
    @test lh[4].refined == 0.5 && lh[4].in_sample == 0.4 && lh[4].gap ≈ 0.1 && lh[4].variant_in_sample == "motion" && lh[4].retcode_in_sample == "Success"
    @test !haskey(summary_table(tr)[1], :gap)                                      # a table without an in-sample variant: the columns of before
    ph = write_summary_tables(mktempdir(), "held", th; number = fmt, change = pct, layout = "blocks", label = "tab:h", caption = "H.")
    @test readline(ph[1]) == join((TABLE_COLUMNS..., IN_SAMPLE_COLUMNS...), '\t') && readline(paths[1]) == join(TABLE_COLUMNS, '\t')
    texh = read(ph[3], String); mdh = read(ph[2], String)
    # … in the long form and the md, never a column of the tex (owner 2026-11-15: Ties's table format; the paper table's columns are
    # start, refined and change, and all)
    @test occursin("A & 0.5 & 0.3 & -40.0 & 0.31 & 0.6 & 0.36 & -40.0 & 0.37 & 1.2 & 0.7 & -42.0 & 0.72 & 0.8 & 0.5 & -38.0 & 0.51 & 80.0 / 100.0 \\\\", texh)
    @test occursin("\\multicolumn{4}{c}{RMS [px]}", texh) && occursin("\\cmidrule(lr){14-17}", texh) && !occursin("in-sample", texh) &&
          occursin("\\multicolumn{18}{l}{\\textit{S}} \\\\", texh)
    @test occursin("| RMS start | RMS refined | RMS Δ % | RMS in-sample | RMS gap | RMS all |", mdh) && occursin("variant motion_heldout", mdh) && occursin("in-sample: the refinement scored", mdh)
    # the paired bootstrap's interval of refined − start beside each change (`confidence` "on"), * where it excludes 0
    using .Figures: CI_COLUMNS, has_ci
    q(d, lo, hi) = (; difference = d, lo, hi, excludes_zero = lo > 0 || hi < 0)
    ci = Dict("median_px" => q(-0.2, -0.3, -0.1), "mean_px" => q(-0.24, -0.3, 0.02), "p95_px" => q(-0.5, -0.7, -0.3), "rms_px" => q(-0.3, -0.4, -0.2))
    tc = [merge(th[1], (; ci)), (; set = "T", model = "B", variant = "motion_heldout", start = nothing, refined = hrows[("a_refined", "motion_heldout")],
                                 in_sample = hrows[("a_refined", "motion")], ci = nothing, refinement_retcode = "Success")]
    @test has_ci(tc) && !has_ci(th)
    lc = summary_table(tc)
    @test (lc[1].difference, lc[1].ci_lo, lc[1].ci_hi, lc[1].ci_excludes_zero) == (-0.2, -0.3, -0.1, true) && !haskey(lc[5], :ci_lo)
    pc = write_summary_tables(mktempdir(), "ci", tc; number = fmt, change = pct, layout = "blocks")
    @test readline(pc[1]) == join((TABLE_COLUMNS..., IN_SAMPLE_COLUMNS..., CI_COLUMNS...), '\t')
    texc = read(pc[3], String); mdc = read(pc[2], String)
    @test occursin("A & 0.5 & 0.3 & -40.0 & 0.31 & 0.6 & 0.36 & -40.0 & 0.37 &", texc) && !occursin("[-0.3", texc) && !occursin("CI", texc)
    @test occursin("| RMS start | RMS refined | RMS Δ % | RMS 95 % CI | RMS in-sample | RMS gap | RMS all |", mdc) && occursin("[-0.3, -0.1]*", mdc) &&
          occursin("paired photograph bootstrap", mdc)
    @test occursin("B &  & 0.3 &  & 0.31 &", texc) && occursin("| B |  | 0.3 |  |  | 0.25 | 0.05 | 0.31 |", mdc)   # a row without a start: blank change (and interval)
    # the best value per camera marked (mark_best, owner 2026-11-15): of the rows of one camera the lowest start or refinement of each
    # statistic is bold, and a non-improvement is then red without bold; the rows of another camera have their own best; the
    # refinement on all of its own dots is never marked
    tb = vcat(tr[1:1], [(; r..., camera = "S") for r in tw],
              [(; set = "U", model = "X", variant = "motion", camera = "other", start = rows[("a_start", "motion")], refined = rows[("a_refined", "motion")], refinement_retcode = "Success")])
    tex5 = read(write_summary_tables(mktempdir(), "best", tb; number = fmt, change = pct, layout = "blocks", worse_color = "red", mark_best = true)[3], String)
    @test occursin("A & 0.5 & \\textbf{0.25} & -50.0 & 0.26 &", tex5)                                           # camera "S" (A's set): its best is A's refinement …
    @test occursin("W & \\textbf{0.25} & \\textcolor{red}{0.5} & \\textcolor{red}{100.0} & 0.55 &", tex5)        # … and W's start (the same value); W's non-improvement red, not bold
    @test occursin("X & 0.5 & \\textbf{0.25} & -50.0 & 0.26 &", tex5) && count("\\textbf{", tex5) == 12 && !occursin("\\textcolor{red}{\\textbf{", tex5)
    # start → refined (change %) per statistic (inline_change, Ties's format, owner 2026-11-15; layout "blocks"): the arrow a column of its
    # own, the signed change in brackets behind the refinement, ONE header row (no start / refined row), no `all` and no coverage column,
    # @{} at the outer edges, the tabular scaled down to the text width
    tex6 = read(write_summary_tables(mktempdir(), "inline", tb; number = fmt, change = pct, layout = "blocks", worse_color = "red", mark_best = true, inline_change = true)[3], String)
    @test occursin("A & 0.5 & \$\\rightarrow\$ & \\textbf{0.25}\\,(\$-50.0\$\\%) & ", tex6) && !occursin("\\%) & 0.26", tex6)
    @test occursin("W & \\textbf{0.25} & \$\\rightarrow\$ & \\textcolor{red}{0.5\\,(\$+100.0\$\\%)} & ", tex6)
    @test occursin("\\setlength{\\tabcolsep}{3.2pt}", tex6) && occursin("\\begin{tabular}{@{}l" * repeat("r@{\\hspace{2pt}}c@{\\hspace{2pt}}l", 4) * "@{}}", tex6)
    @test occursin("\\toprule\nmodel & \\multicolumn{3}{c}{median [px]} & \\multicolumn{3}{c}{mean [px]} & \\multicolumn{3}{c}{p95 [px]} & \\multicolumn{3}{c}{RMS [px]} \\\\\n" *
                   "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}\n", tex6)
    @test !occursin("coverage", tex6) && !occursin("refined (\\%)", tex6) && !occursin("& all", tex6) && !occursin("\\midrule", tex6)
    @test occursin("\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}", tex6) && occursin("\\end{tabular}}\n", tex6)
    @test occursin("\\multicolumn{13}{l}{\\textit{S}} \\\\", tex6) && count("\\textbf{", tex6) == 12
    # a row without a start: no arrow and no bracket; the layout per_set has no inline form
    tex7 = read(write_summary_tables(mktempdir(), "nostart", tr; number = fmt, change = pct, layout = "blocks", inline_change = true)[3], String)
    @test occursin("B &  &  & 2.0 &  &  & 2.5 &", tex7) && !occursin("converge", tex7)   # no marker, no footnote row
    @test_throws ArgumentError write_summary_tables(mktempdir(), "bad", tr; number = fmt, change = pct, inline_change = true)
end

@testitem "tab:defocus number macros: names, values and provenance through Eval's writer" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("figures.jl"))
    using .Figures: summary_table, table_numbers!
    import VisualGeometryToolkitEval.Reproduce as Reproduce
    # two refinements from one start, as tab:defocus: the aperture row improves every statistic, the pinhole row less (its p95 worse)
    model_stats(m, me, p, r) = Dict{String, Any}("photos" => 40, "dots" => 11000, "median_px" => m, "mean_px" => me, "p95_px" => p, "rms_px" => r, "retcode" => "Success",
                                                 "coverage_dots" => 11000, "coverage_fraction" => 1.0, "all_photos" => 40,
                                                 "all_median_px" => m, "all_mean_px" => me, "all_p95_px" => p, "all_rms_px" => r)
    start = model_stats(0.5, 0.6, 1.2, 0.8)
    rows = [(; set = "central spline", model = "with aperture + focus surface", variant = "motion", name_part = "aperture", start, refined = model_stats(0.25, 0.3, 0.6, 0.4),
               refinement_retcode = "Success"),
            (; set = "central spline", model = "pinhole (no defocus)", variant = "motion", name_part = "pinhole", start, refined = model_stats(0.4, 0.54, 1.32, 0.72),
               refinement_retcode = "Success")]
    nums = Reproduce.PaperNumbers(; driver = "scripts/chrome_evaluation_paper.jl", config = "configs/experiments/chrome_evaluation_paper.toml",
                                  run_key = "abc123", commit = "deadbee")
    table_numbers!(nums, "defocus", rows; sigdigits = 3, change_sigdigits = 2, label = "tab:defocus")
    path = Reproduce.write_numbers(joinpath(mktempdir(), "numbers.tex"), nums)
    vals = Dict(Reproduce.tex_macro_values(path))
    macro_names = Set(keys(vals))
    for r in ("Aperture", "Pinhole"), s in ("Median", "Mean", "PNineFive", "Rms"), w in ("Start", "Refined", "Change")
        @test "defocus$r$s$w" in macro_names
    end
    for s in ("Median", "Mean", "PNineFive", "Rms")
        @test "defocus$(s)ImprovementGain" in macro_names
    end
    @test length(macro_names) == 2 * 4 * 3 + 4
    num(n) = parse(Float64, replace(vals[n], r"[^0-9.\-]" => ""))
    @test num("defocusApertureMedianStart") == 0.5 && num("defocusApertureMedianRefined") == 0.25
    @test num("defocusApertureMedianChange") == -50 && num("defocusPinholeMedianChange") == -20
    @test num("defocusPinholePNineFiveChange") == 10                         # a non-improvement keeps its sign
    @test num("defocusMedianImprovementGain") == 30                          # 50 − 20 percentage points
    @test num("defocusPNineFiveImprovementGain") == 60                       # 50 − (−10)
    text = read(path, String)
    for n in ("defocusApertureMedianRefined", "defocusMedianImprovementGain")
        line = only(l for l in split(text, '\n') if startswith(l, "% provenance \\$n:"))
        @test occursin("driver=scripts/chrome_evaluation_paper.jl", line) && occursin("config=configs/experiments/chrome_evaluation_paper.toml", line)
        @test occursin("run=abc123", line) && occursin("commit=deadbee", line) && occursin("label=tab:defocus", line)
    end
    @test occursin("better=lower", only(l for l in split(text, '\n') if startswith(l, "% provenance \\defocusApertureMedianRefined:")))
    @test occursin("better=higher", only(l for l in split(text, '\n') if startswith(l, "% provenance \\defocusMedianImprovementGain:")))
    # both lenses (the zoom lens's rows after the Tamron's), each row's numbers with its own run's provenance
    rows4 = vcat(rows, [merge(rows[1], (; name_part = "zoomAperture")), merge(rows[2], (; name_part = "zoomPinhole"))])
    n4 = Reproduce.PaperNumbers(; driver = "d", config = "c", run_key = "per-number", commit = "per-number")
    table_numbers!(n4, "defocus", rows4; sigdigits = 3, change_sigdigits = 2, provenance = r -> startswith(r.name_part, "zoom") ? (; run_key = "rZ", commit = "cZ") : (; run_key = "rT", commit = "cT"))
    p4 = Reproduce.write_numbers(joinpath(mktempdir(), "numbers.tex"), n4); t4 = read(p4, String)
    @test length(Reproduce.tex_macro_values(p4)) == 4 * 4 * 3 + 2 * 4
    @test occursin("run=rZ", only(l for l in split(t4, '\n') if startswith(l, "% provenance \\defocusZoomApertureRmsRefined:")))
    @test occursin("run=rZ", only(l for l in split(t4, '\n') if startswith(l, "% provenance \\defocusZoomMedianImprovementGain:")))
    @test occursin("run=rT", only(l for l in split(t4, '\n') if startswith(l, "% provenance \\defocusMedianImprovementGain:")))
    @test_throws ArgumentError table_numbers!(Reproduce.PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x"), "defocus",
                                              [merge(rows[1], (; start = nothing))]; sigdigits = 3, change_sigdigits = 2)
    # a pair in the other order (the pinhole row first, as tab:defocus prints it): the same macros and values; two rows of one kind are no pair
    n5 = Reproduce.PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x")
    table_numbers!(n5, "defocus", reverse(rows); sigdigits = 3, change_sigdigits = 2, label = "tab:defocus")
    @test Dict(Reproduce.tex_macro_values(Reproduce.write_numbers(joinpath(mktempdir(), "numbers.tex"), n5))) == vals
    @test_throws ArgumentError table_numbers!(Reproduce.PaperNumbers(; driver = "d", config = "c", run_key = "k", commit = "x"), "defocus",
                                              [rows[1], merge(rows[1], (; name_part = "zoomAperture"))]; sigdigits = 3, change_sigdigits = 2)
end

@testitem "chrome evaluation paper: the tables stage on fixture products is byte-identical to the driver of record's" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("chrome_evaluation.jl"))
    include(srcdir("figures.jl"))
    include(srcdir("chrome_evaluation_paper.jl"))
    import VisualGeometryToolkitEval.Reproduce as Reproduce
    # the fixture products: each set's evaluate summary.tsv (every label the tables read, both variants; one chrome evaluation at its
    # iteration cap, si17ta_ds_refined) and the refinements' retcodes (one MaxTime, one without a record); the golden is what this
    # driver of record wrote from the same products after the fisheye 27.10 set's exclusion (owner 2026-11-15: make_golden.jl); before,
    # scripts/geometric_figures.jl's (the tables' driver of record before they moved here; the bytes this driver matched)
    F = projectdir("test", "fixtures", "chrome_paper"); G = joinpath(F, "golden")
    path = projectdir("configs", "experiments", "chrome_evaluation_paper.toml")
    P = load_chrome_paper_config(path)
    # the fixture products predate the held-out headline (they hold the variants motion and free): the tables are rendered here at the
    # variant they were recorded with, no in-sample column and no interval (no paired.tsv: the golden's bytes); the held-out columns: the synthetic summary's item above
    @test all(t -> t["variant"] == "motion_heldout" && t["in_sample_variant"] == "motion" && t["confidence"] == "on", values(P["tables"]))
    foreach(t -> (t["variant"] = "motion"; delete!(t, "in_sample_variant"); t["confidence"] = "off"), values(P["tables"]))
    # the paper config is its own driver's ([experiment] name, scripts/chrome_evaluation_paper.jl); a set's driver refuses it by that name
    @test P["experiment"]["name"] == "chrome_evaluation_paper"
    @test_throws ArgumentError load_chrome_evaluation_config(path)
    # the rows added after the fixtures and the golden were recorded: the fisheye 30 cm set (2026-11-16) and the zoom's Brown–Conrady
    # start (2026-11-17) — no fixture product of theirs, so the tables are rendered without them, as the golden was
    later(r) = get(r, "chrome", "") == "fisheye_si30" || occursin("sizl_bc5", string(get(r, "start", ""), get(r, "refined", "")))
    delete!(P["chrome"], "fisheye_si30")
    foreach(t -> haskey(t, "rows") && filter!(!later, t["rows"]), values(P["tables"]))
    ret = TOML.parsefile(joinpath(F, "refinement_retcodes.toml"))
    products = (; summary = k -> joinpath(F, k, "summary.tsv"), vectors = (k, l, v) -> error("no vectors in the fixture"), retcode = (k, l) -> ret[k][l],
                provenance = k -> (; run_key = "fixture-key-$k", commit = String(TOML.parsefile(joinpath(F, k, "_provenance.toml"))["gitcommit"])))
    out, out2 = mktempdir(), mktempdir()
    rows = chrome_tables_rows(P, products)
    files = write_chrome_tables(P, (out, out2), rows)
    @test length(files) == 2 * 3 * 3                                                    # both directories, three files per table
    @test sort(unique(basename.(files))) == sort([f for f in readdir(G) if startswith(f, "table_")])
    for f in files
        @test read(f) == read(joinpath(G, basename(f)))                              # byte for byte: .tex, .md and .tsv (the .tex's
    end                                                                              # generator comment names this driver: make_golden.jl)
    @test Set(chrome_tables(P)) == Set(["chrome_all", "defocus", "refinement_chrome"])
    # the number macros: the same names and values as the golden numbers.tex (its provenance lines name the driver, which moved)
    nums = chrome_table_macros(P, path, rows, products.provenance; driver = "scripts/chrome_evaluation_paper.jl")
    mine = Reproduce.write_numbers(joinpath(out, "numbers.tex"), nums)
    @test Dict(Reproduce.tex_macro_values(mine)) == Dict(Reproduce.tex_macro_values(joinpath(G, "numbers.tex")))
    @test occursin("driver=scripts/chrome_evaluation_paper.jl", read(mine, String)) && occursin("run=fixture-key-tamron_la", read(mine, String))
    # a missing evaluation fails the stage
    @test_throws ArgumentError chrome_tables_rows(P, merge(products, (; summary = k -> joinpath(F, "fourteen", "summary.tsv"))))
end
