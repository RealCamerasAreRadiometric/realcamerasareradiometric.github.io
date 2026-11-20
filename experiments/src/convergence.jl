# =============================================================================
# src/convergence.jl — the convergence experiment (owner 2026-11-15; configs/experiments/convergence.toml, scripts/convergence.jl):
# the bias of T. Schöps's method, read off our radiometric refinement started from his camera. At every LM iteration of the
# refinement (its checkpoints, `io.checkpoint_each`: state_it<N>.json and grid_it<N>.json; iteration 0 the start record) two
# series on the same iterations:
#   1. the BEARING DIVERGENCE (degrees): the angle between each of Schöps's own detections' bearings through the refinement's camera
#      of that iteration, under its own poses of that iteration, and its target point's predicted bearing (the target point on the
#      iteration's own surface) — RadiometricCamera's `state_bearings`, the refinement monitor's `bearing(detections, own poses)`
#      quantity in radians, here in degrees; its median and p95 over the detections. It grows as the photometric fit leaves his
#      detections;
#   2. the THORLABS REPROJECTION ERROR (px): the camera of that iteration scored on the independent chrome target by the chrome
#      evaluation's own scorer (src/chrome_evaluation.jl `score_chrome_models`, the set's config of record): the ball-head motion
#      (one rotation centre, `RotationCenterViews`), every iteration one model of ONE scoring so the statistics are on the dots every
#      iteration scores (`common_dots`), the held-out variant (motion_heldout: the parity-interleaved fits, the headline) beside
#      the in-sample one (motion).
# Beside them the LM's rel-RMS of each iteration (the checkpoint's provenance; the start record's after the linear solve at 0).
# Nothing here logs the LM: every value is computed after the run from the states the refinement saved (RC's checkpoint
# mechanism), keyed by the refinement's and the chrome index's products.
# The record (`convergence.json`, EXPERIMENT_RECORD_SCHEMAS.convergence) is the figure stage's one input (`write_convergence_figure`).
# =============================================================================
using Printf: Printf, @printf
using Statistics: median
include("convergence_panels.jl")             # the paper figure's PGFPlots text (rms_figure.py's style)

"The tables of configs/experiments/convergence.toml."
const CONVERGENCE_TABLES = ("experiment", "inputs", "upstream", "convergence", "cases", "figure", "panels")
"The keys of `[convergence]`, every one required: the iteration stride."
const CONVERGENCE_KEYS = ("stride",)
"""The keys of a `[cases.<name>]` table (its name a case of `[upstream] radiometric_refinement`), every one required: the chrome set's
config of record that scores it (`chrome`, a path relative to the project) and the panel's caption label (`label`)."""
const CONVERGENCE_CASE_KEYS = ("chrome", "label")
"The chrome evaluation's variants the experiment scores, the headline first (the held-out fit, then the in-sample one)."
const CONVERGENCE_VARIANTS = ("motion_heldout", "motion")
"The record's file (the convergence product's output)."
const CONVERGENCE_RECORD_FILE = "convergence.json"

"""
    load_convergence_config(path) -> Dict

The config of record of the convergence experiment, every table and key checked: `[convergence] stride` a positive integer;
`[upstream] radiometric_refinement` a refinement config of record; every `[cases.<name>]` a case of it that writes a grid model and
checkpoints every iteration (`io.checkpoint_each`), its `chrome` a chrome evaluation config of record whose `[evaluate]` models
score that case (the pairing of record) and whose index is its own stage.
"""
function load_convergence_config(path::AbstractString)
    c = load_driver_config(path, "convergence"; tables = CONVERGENCE_TABLES, case_keys = CONVERGENCE_CASE_KEYS)
    for t in ("upstream", "convergence", "cases", "figure")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    VgtkCore.check_keys(c["upstream"], ("radiometric_refinement",), "$path [upstream]")
    check_required_keys(c["convergence"], CONVERGENCE_KEYS, "$path [convergence]")
    st = c["convergence"]["stride"]
    st isa Integer && st >= 1 || throw(ArgumentError("$path [convergence] stride must be a positive integer, got $(repr(st))"))
    check_figure_table(c["figure"], "$path [figure]")
    haskey(c, "panels") && check_panels_table(c, "$path [panels]")
    rc = convergence_refinements(c)
    for (n, t) in c["cases"]
        where = "$path [cases.$n]"
        check_required_keys(t, CONVERGENCE_CASE_KEYS, where)
        rt = get(rc["cases"], n, nothing)
        rt === nothing && throw(ArgumentError("$where: no case $n in $(c["upstream"]["radiometric_refinement"])"))
        rt["kind"] == "grid" || throw(ArgumentError("$where: the case writes a $(rt["kind"]) model; the iterations are scored as grid cameras"))
        RadiometricCamera.pipeline_options(rt["pipeline"]).io.checkpoint_each ||
            throw(ArgumentError("$where: the refinement does not checkpoint every iteration (its [pipeline.io] checkpoint_each): no iteration to read"))
        cc = convergence_chrome(c, n)
        any(m -> get(m, "case", "") == n, cc["evaluate"]["models"]) ||
            throw(ArgumentError("$where: chrome = $(t["chrome"]) scores no model of case $n (its [evaluate] models): not the set of record of the case"))
        cc["evaluate"]["index"] == "stage" || throw(ArgumentError("$where: chrome = $(t["chrome"]) reads an index run of record; the experiment reads the index stage's"))
    end
    return c
end

"The refinement config of record `[upstream] radiometric_refinement` names (src/radiometric_refinement.jl)."
convergence_refinements(cfg::AbstractDict) = load_radiometric_refinement_config(projectdir(cfg["upstream"]["radiometric_refinement"]))
"The chrome evaluation config of record of case `case` (`[cases.<case>] chrome`, src/chrome_evaluation.jl)."
convergence_chrome(cfg::AbstractDict, case::AbstractString) = load_chrome_evaluation_config(convergence_chrome_path(cfg, case))
"The path of the chrome evaluation config of record of case `case`."
convergence_chrome_path(cfg::AbstractDict, case::AbstractString) = projectdir(cfg["cases"][case]["chrome"])

"""
    convergence_protocol(cc) -> Dict

The chrome set `cc`'s scoring protocol as it keys the convergence: its `[evaluate]` table without what the experiment does not read —
the models (the iterations are this experiment's), the paired bootstrap and its comparisons (between the set's models: none of an
iteration's; `CONVERGENCE_UNREAD_EVALUATE`) — and with the held-out variant on (`holdout`: the headline, beside the in-sample fit).
"""
convergence_protocol(cc::AbstractDict) =
    merge(Dict{String, Any}(k => v for (k, v) in cc["evaluate"] if !(k in CONVERGENCE_UNREAD_EVALUATE)), Dict{String, Any}("holdout" => "on"))   # the switch as the config spells it (RC 0.11.4)
"The chrome set's `[evaluate]` keys the convergence does not read (its key leaves them out, its scoring compares no models)."
const CONVERGENCE_UNREAD_EVALUATE = ("models", "bootstrap", "comparisons")

"""
    convergence_product(cfg, case) -> Product

The convergence record of `case` (its output convergence.json): keyed by `[convergence]`, the case's table, the chrome set's scoring
protocol (`convergence_protocol`) and target input, and the keys of the products it reads (`upstream`: the refinement and its start,
the chrome set's detect and index, its image region).
"""
function convergence_product(cfg::AbstractDict, case::AbstractString)
    rc = convergence_refinements(cfg); cc = convergence_chrome(cfg, case)
    up = Dict{String, Any}("refine" => refine_product(rc, case).key, "start" => start_product(rc, case).key,
                           "detect" => detect_product(cc).key, "index" => index_product(cc).key)
    n = lens_coverage_name(cc["evaluate"]["lens_coverage"])
    n === nothing || (up["image_region"] = image_region_product(cc, n).key)
    rec = Dict{String, Any}("experiment" => cfg["experiment"], "convergence" => cfg["convergence"], "case" => Dict(case => cfg["cases"][case]),
                            "chrome" => convergence_protocol(cc), "inputs" => inputs_record(cc, [cc["evaluate"]["target"]]), "upstream" => up,
                            "record" => EXPERIMENT_RECORD_SCHEMAS.convergence)
    return driver_product("convergence", rec; output = CONVERGENCE_RECORD_FILE)
end

"""
    convergence_iterations(refine_dir, stride) -> Vector{Int}

The LM iterations the convergence reads of the refinement in `refine_dir`: 0 (the start) and every checkpointed iteration (both
`grid_it<N>.json` and `state_it<N>.json` written) that is a multiple of `stride`, the last checkpoint always among them.
"""
function convergence_iterations(refine_dir::AbstractString, stride::Integer)
    its = [i for i in checkpoint_iterations(refine_dir, "grid") if isfile(joinpath(refine_dir, "state_it$i.json"))]
    isempty(its) && throw(ArgumentError("$refine_dir: no checkpointed iteration (grid_it<N>.json with state_it<N>.json)"))
    return [0; sort!(unique!([filter(i -> i % stride == 0, its); last(its)]))]
end

"""
    bearing_divergence(geometry, state) -> (; median_deg, p95_deg, n)

The bearing divergence of the start geometry's detections at the refinement state `state` (`RadiometricCamera.load_state`'s): the
median and the 95th percentile, in degrees, of RadiometricCamera's `state_bearings` (each detection's bearing through the state's
camera against its target point's under the state's poses), and the number of detections the camera images.
"""
function bearing_divergence(g, st)
    b = rad2deg.(RadiometricCamera.state_bearings(g, st))
    isempty(b) && throw(ArgumentError("bearing_divergence: the state's camera images none of the detections"))
    s = residual_summary(b)                                              # THE project's summary of a residual vector (its p95 the 0.95 quantile)
    return (; median_deg = s.median_px, p95_deg = s.p95_px, n = s.n)    # (its `_px` names; here degrees)
end

"""
    state_rel_rms(path) -> Float64

The LM's rel-RMS at a checkpointed iteration: the provenance `rel_rms` of its state record (`state_it<N>.json`, RadiometricCamera
0.11.4's checkpoint); an `ArgumentError` for a state that records none.
"""
function state_rel_rms(path::AbstractString)
    p = get(VgtkIO.read_record(path), "provenance", nothing)
    p isa AbstractDict && haskey(p, "rel_rms") || throw(ArgumentError("$path: no rel_rms in its provenance (a checkpoint before RadiometricCamera 0.11.4)"))
    return Float64(p["rel_rms"])
end

"""
    bearing_series(start_dir, refine_dir, iterations; geometry = nothing) -> Vector{NamedTuple}

The bearing divergence (`bearing_divergence`) and the LM's rel-RMS at each of `iterations` of the refinement in `refine_dir` from
the start record `start_dir`: iteration 0 the start record's state (its rel-RMS after the linear solve), iteration N its checkpoint
state_it<N>.json. The detections are the start record's front end's (`start_geometry` of the record's options, read under
data_root(), where the options' relative inputs are; `geometry` given: that one).
"""
function bearing_series(start_dir::AbstractString, refine_dir::AbstractString, iterations::AbstractVector{<:Integer}; geometry = nothing)
    start = RadiometricCamera.load_refinement_start(start_dir)
    g = geometry === nothing ? cd(() -> RadiometricCamera.start_geometry(start.options; record = start), data_root()) : geometry
    return map(iterations) do i
        f = i == 0 ? joinpath(start_dir, "state.json") : joinpath(refine_dir, "state_it$i.json")
        (; iteration = Int(i), bearing_divergence(g, RadiometricCamera.load_state(f))..., rel_rms = i == 0 ? start.costs.linear[2] : state_rel_rms(f))
    end
end

"""
    iteration_models(start_dir, refine_dir, iterations) -> Vector{Tuple{String, String}}

The chrome scorer's models of the iterations: `("it<N>", "grid:<file>")`, the start record's grid.json at 0, grid_it<N>.json after.
"""
iteration_models(start_dir::AbstractString, refine_dir::AbstractString, iterations) =
    [("it$i", "grid:" * (i == 0 ? joinpath(start_dir, "grid.json") : joinpath(refine_dir, "grid_it$i.json"))) for i in iterations]

"""
    thorlabs_series(summaries, iterations) -> NamedTuple

Per variant of `CONVERGENCE_VARIANTS`, the chrome scorer's head-to-head (`common`: the dots every iteration scores) of each
iteration's model `it<N>`: `median`, `rms`, `p95` (px), `n` (dots) and `photos`.
"""
function thorlabs_series(summaries, iterations)
    row(v, i) = (k = findfirst(s -> s.label == "it$i" && s.variant == v, summaries);
                 k === nothing && throw(ArgumentError("thorlabs_series: no $v score of iteration $i (its model did not load)")); summaries[k].common)
    one(v) = (rs = [row(v, i) for i in iterations];
              (; median = [r.median_px for r in rs], rms = [r.rms_px for r in rs], p95 = [r.p95_px for r in rs], n = [r.n for r in rs], photos = [r.photos for r in rs]))
    return NamedTuple{Symbol.(CONVERGENCE_VARIANTS)}(Tuple(one(v) for v in CONVERGENCE_VARIANTS))
end

"""
    convergence_record(; case, label, iterations, bearings, thorlabs, keys) -> NamedTuple

The convergence record of a case (EXPERIMENT_RECORD_SCHEMAS.convergence): per iteration the bearing divergence's median and p95 (deg) and
its detections, the Thorlabs error per variant (the headline `motion_heldout` and the in-sample `motion`: median, RMS, p95 px, the
common dots), the LM rel-RMS, the iteration of the headline's lowest median (`thorlabs_lowest`: where the Thorlabs error stops
falling), and the keys of the runs it was computed from (`keys`).
"""
function convergence_record(; case, label, iterations, bearings, thorlabs, keys)
    h = thorlabs.motion_heldout.median
    j = argmin(h)
    return (; schema = EXPERIMENT_RECORD_SCHEMAS.convergence, case = String(case), label = String(label), headline = first(CONVERGENCE_VARIANTS),
            iterations = collect(Int, iterations),
            bearing_divergence_deg = (; median = [b.median_deg for b in bearings], p95 = [b.p95_deg for b in bearings], n = [b.n for b in bearings]),
            thorlabs_px = thorlabs, rel_rms = [b.rel_rms for b in bearings],
            thorlabs_lowest = (; iteration = iterations[j], median_px = h[j]), runs = keys)
end

"""
    load_convergence_record(path) -> NamedTuple

The convergence record at `path` (`convergence_record`'s fields), read through VisualGeometryToolkitIO.
"""
function load_convergence_record(path::AbstractString)
    r = VgtkIO.read_record(path)
    r["schema"] == EXPERIMENT_RECORD_SCHEMAS.convergence || throw(ArgumentError("$path: schema $(r["schema"]), not $(EXPERIMENT_RECORD_SCHEMAS.convergence)"))
    f(v) = Float64.(v)
    var(t) = (; median = f(t["median"]), rms = f(t["rms"]), p95 = f(t["p95"]), n = Int.(t["n"]), photos = Int.(t["photos"]))
    b = r["bearing_divergence_deg"]
    return (; schema = String(r["schema"]), case = String(r["case"]), label = String(r["label"]), headline = String(r["headline"]),
            iterations = Int.(r["iterations"]), bearing_divergence_deg = (; median = f(b["median"]), p95 = f(b["p95"]), n = Int.(b["n"])),
            thorlabs_px = NamedTuple{Symbol.(CONVERGENCE_VARIANTS)}(Tuple(var(r["thorlabs_px"][v]) for v in CONVERGENCE_VARIANTS)),
            rel_rms = f(r["rel_rms"]), thorlabs_lowest = (; iteration = Int(r["thorlabs_lowest"]["iteration"]), median_px = Float64(r["thorlabs_lowest"]["median_px"])),
            runs = Dict{String, Any}(String(k) => v for (k, v) in r["runs"]))
end

"""
    run_convergence(cfg, case; driver, config_path) -> dir

Stage convergence of `case`: its record produced or loaded (`convergence_product`). Reads the refinement's checkpoints and start
record (stage refine), the chrome set's index (stage chrome), each produced on demand (`refinement_upstream`, `chrome_upstream`); scores every iteration's camera with the chrome scorer
into `chrome/` of the product (eval_it<N>.json, summary.tsv: the scorer's own records) and writes convergence.json last.
"""
function run_convergence(cfg::AbstractDict, case::AbstractString; driver::AbstractString, config_path::AbstractString)
    rc = convergence_refinements(cfg); cc = convergence_chrome(cfg, case); cpath = convergence_chrome_path(cfg, case)
    produce(convergence_product(cfg, case); product_records(driver, config_path)...) do dir
        rp = refine_product(rc, case); sp = start_product(rc, case)
        rdir = refinement_upstream(rc, cfg["upstream"]["radiometric_refinement"], case, "refine", "the refinement of case $case")
        sdir = refinement_upstream(rc, cfg["upstream"]["radiometric_refinement"], case, "start", "the start of case $case")
        chrome_upstream(cc, cpath, "index", "the chrome index of $(cfg["cases"][case]["chrome"])")
        its = convergence_iterations(rdir, cfg["convergence"]["stride"])
        @printf(">> convergence %s: %d iterations (%s … %s) of %s\n", case, length(its), first(its), last(its), rdir); flush(stdout)
        bearings = bearing_series(sdir, rdir, its)
        E = merge(evaluate_settings(cc, cpath), (; holdout = true, comparisons = NamedTuple[]))   # the set's protocol, held out on, no comparison of the set's models
        sc = score_chrome_models(cc, cpath, nothing, joinpath(dir, "chrome"); settings = E, models = iteration_models(sdir, rdir, its))
        th = thorlabs_series(sc.summaries, its)
        keys = (; refine = rp.key, start = sp.key, detect = detect_product(cc).key, index = index_product(cc).key, chrome_config = cfg["cases"][case]["chrome"])
        rec = convergence_record(; case, label = cfg["cases"][case]["label"], iterations = its, bearings, thorlabs = th, keys)
        for (k, i) in enumerate(its)
            @printf("   it %3d  bearing divergence median %.4f° p95 %.4f°  Thorlabs held-out median %.4f px (in-sample %.4f)  rel-RMS %.4f\n", i,
                    rec.bearing_divergence_deg.median[k], rec.bearing_divergence_deg.p95[k], th.motion_heldout.median[k], th.motion.median[k], rec.rel_rms[k])
        end
        VgtkIO.write_record(joinpath(dir, CONVERGENCE_RECORD_FILE), merge(rec, (; schema = EXPERIMENT_RECORD_SCHEMAS.convergence)))
        Produced(; inputs = [sc.inputs; rdir; sdir])
    end
end

# ---- the figure: the dual-axis pgfplots fragment (src/figure_style.jl: its rows, `dual_axis`, the style sheet and the preview) -------
"The keys of `[figure]`, every one required: the assets' directory under plotsdir() and the axis labels (LaTeX)."
const CONVERGENCE_FIGURE_KEYS = ("asset_dir", "xlabel", "divergence_ylabel", "thorlabs_ylabel")

"`[figure]` checked: every key of `CONVERGENCE_FIGURE_KEYS`, a non-empty string."
function check_figure_table(t::AbstractDict, where::AbstractString)
    check_required_keys(t, CONVERGENCE_FIGURE_KEYS, where)
    all(k -> t[k] isa AbstractString && !isempty(t[k]), CONVERGENCE_FIGURE_KEYS) || throw(ArgumentError("$where: every key is a non-empty string"))
    return t
end

"""
    convergence_figure_product(cfg, case) -> Product

The figure of `case`: keyed by `[figure]` and the convergence record's key; its output the fragment's copy `figure.tex`, written last.
"""
convergence_figure_product(cfg::AbstractDict, case::AbstractString) =
    driver_product("figure", Dict{String, Any}("experiment" => cfg["experiment"], "figure" => cfg["figure"], "case" => case,
                                               "upstream" => Dict{String, Any}("convergence" => convergence_product(cfg, case).key));
                   output = "figure.tex", threads = nothing)

"""
    convergence_fragment(r, t) -> String

The dual-axis figure of the convergence record `r` (`load_convergence_record`) under `[figure]` `t`: src/figure_style.jl's
`dual_axis` on the LM iterations — the bearing divergence (deg: its median and p95, the rows `divergence median` / `divergence p95`)
on the left axis, the Thorlabs error (px: the held-out median, the headline, and the in-sample median; `thorlabs heldout` /
`thorlabs in-sample`) on the right — its data inline, a comment header naming the record's runs.
"""
function convergence_fragment(r, t::AbstractDict)
    head = "% convergence_$(r.case).tex — $(r.label) (case $(r.case)): the bearing divergence of T. Sch\\\"ops's detections (left, deg) and the Thorlabs\n" *
           "% reprojection error (right, px; $(r.headline) the headline) per LM iteration. Generated by experiments src/convergence.jl from convergence.json\n" *
           "% (refine $(r.runs["refine"]), chrome index $(r.runs["index"])). \\input figure_styles.tex first (src/figure_style.jl).\n"
    return head * dual_axis((; opts = "ylabel={$(t["divergence_ylabel"])}, ymin=0",
                             curves = [("divergence median", r.bearing_divergence_deg.median), ("divergence p95", r.bearing_divergence_deg.p95)]),
                            (; opts = "ylabel={$(t["thorlabs_ylabel"])}",
                             curves = [("thorlabs heldout", r.thorlabs_px.motion_heldout.median), ("thorlabs in-sample", r.thorlabs_px.motion.median)]);
                            x = r.iterations, xlabel = t["xlabel"])
end

"""
    convergence_caption(r) -> String

The caption snippet of the convergence figure of the record `r` (the house rule: the figure's facts in the caption, no text inside
the axes): what each curve is, the case, and the record's start, end and the iteration where the headline Thorlabs error is lowest.
"""
function convergence_caption(r)
    b = r.bearing_divergence_deg.median; h = r.thorlabs_px.motion_heldout.median
    return "% convergence_$(r.case)_caption.tex — generated by experiments src/convergence.jl from convergence.json\n" *
           Printf.format(Printf.Format("Our refinement from T.~Sch\\\"ops's camera, %s: the bearing divergence of his detections through our camera under our poses " *
                    "(left; median solid, 95th percentile dashed) and our camera's reprojection error on the Thorlabs chrome target " *
                    "(right; held-out median solid, in-sample median dashed) per LM iteration. The median divergence goes from " *
                    "\$%.3f^\\circ\$ to \$%.3f^\\circ\$ over %d iterations while the held-out Thorlabs error goes from %.3f\\,px to %.3f\\,px, lowest " *
                    "(%.3f\\,px) at iteration %d.\n"), r.label, first(b), last(b), last(r.iterations), first(h), last(h),
                    r.thorlabs_lowest.median_px, r.thorlabs_lowest.iteration)
end

"""
    write_convergence_figure(dir, r, t) -> (; fragment, caption, styles, preview)

The convergence figure of the record `r` under `[figure]` `t` into `dir`: the fragment `convergence_<case>.tex`, its caption snippet
`convergence_<case>_caption.tex`, the project's style sheet figure_styles.tex (`write_figure_styles`) and a standalone preview.tex
inputting both (`write_figure_preview`; preview.pdf where latexmk is installed) — vector output only.
"""
function write_convergence_figure(dir::AbstractString, r, t::AbstractDict)
    mkpath(dir)
    base = "convergence_$(r.case)"
    frag = joinpath(dir, base * ".tex"); cap = joinpath(dir, base * "_caption.tex")
    write(frag, convergence_fragment(r, t)); write(cap, convergence_caption(r))
    sty = write_figure_styles(joinpath(dir, "figure_styles.tex"))
    write_figure_preview(dir, base)
    return (; fragment = frag, caption = cap, styles = sty, preview = joinpath(dir, "preview.tex"))
end

# ---- the paper's figure (fig:convergence-rms): the detections' reprojection error beside the held-out Thorlabs error, in px -------
"""
    detection_reprojection_series(start_dir, refine_dir, iterations) -> (; median, rms, n)

The detections' TRUE reprojection error per iteration, px: for each of the start geometry's detections, |proj(X; camera_it,
pose_it) − detection| (RadiometricCamera's `residual_vectors` of the iteration's camera under its own poses, the target points on the
iteration's own surface: `_board_features`), its median and RMS over ONE fixed set — the detections every iteration projects (a
port of T. Harder's convergence/tools/per_detection_values.jl and rms_figure.py's stored statistic); `n` that set's size.
"""
function detection_reprojection_series(start_dir::AbstractString, refine_dir::AbstractString, iterations::AbstractVector{<:Integer})
    RC_ = RadiometricCamera
    start = RC_.load_refinement_start(start_dir)
    g = cd(() -> RC_.start_geometry(start.options; record = start), data_root())
    nd = sum(length, g.observations)
    D = fill(NaN, nd, length(iterations))
    for (m, it) in enumerate(iterations)
        f = it == 0 ? joinpath(start_dir, "state.json") : joinpath(refine_dir, "state_it$it.json")
        st = RC_.load_state(f)
        idx = [findfirst(==(n), st.names) for n in g.names]
        any(isnothing, idx) && throw(ArgumentError("$f lacks a view of the start's"))
        feats = RC_._board_features(g, RC_.board_surface(g.target, st.target), g.observations)
        poses = st.poses[idx]; j = 0
        for k in eachindex(g.names)
            length(feats[k]) == length(g.observations[k]) && all(feats[k][i][1] == g.observations[k][i][1] for i in eachindex(feats[k])) ||
                throw(ArgumentError("$f view $k: the board features are not the detections, in order"))
            for v in RC_.residual_vectors(st.camera.geometric, poses[k], feats[k])
                j += 1; D[j, m] = hypot(v[3], v[4])
            end
        end
    end
    ok = [all(isfinite, view(D, q, :)) for q in 1:nd]
    E = D[ok, :]
    return (; median = [median(view(E, :, m)) for m in axes(E, 2)], rms = [sqrt(sum(abs2, view(E, :, m)) / size(E, 1)) for m in axes(E, 2)],
            n = size(E, 1))
end

"The keys of `[panels]`: the figure's left and right cases (cases of [cases]), its assets' directory under plotsdir(), the macro prefix and the iteration the detections' rise is quoted from."
const CONVERGENCE_PANELS_KEYS = ("left", "right", "asset_dir", "macro_prefix", "rise_from_iteration")
"The keys of a `[panels.<case>]` table: the panel's title and its target (`aprilgrid` | `siemens`: the detections' colour) and its macro name part."
const CONVERGENCE_PANEL_KEYS = ("title", "target", "macro")

"`[panels]` checked: its keys, its two cases among [cases], each with its `[panels.<case>]` table."
function check_panels_table(c::AbstractDict, where::AbstractString)
    p = c["panels"]
    VgtkCore.check_keys(p, (CONVERGENCE_PANELS_KEYS..., get(p, "left", ""), get(p, "right", "")), where)
    check_required_keys(p, (CONVERGENCE_PANELS_KEYS..., p["left"], p["right"]), where)
    for s in (p["left"], p["right"])
        haskey(c["cases"], s) || throw(ArgumentError("$where: $s is not a case of [cases]"))
        check_required_keys(p[s], CONVERGENCE_PANEL_KEYS, "$where.$s")
        haskey(CONVERGENCE_PANEL_COLOURS, p[s]["target"]) || throw(ArgumentError("$where.$s target: one of $(sort(collect(keys(CONVERGENCE_PANEL_COLOURS))))"))
    end
    return p
end

"""
    convergence_panels_product(cfg) -> Product

The paper's convergence figure (its output numbers.tex, written last; the panels kept in `assets/`): keyed by `[panels]` and, per
panel, the convergence record's key and the refinement's and its start's (the detections' reprojection reads their states).
"""
function convergence_panels_product(cfg::AbstractDict)
    rc = convergence_refinements(cfg); p = cfg["panels"]
    up = Dict{String, Any}(s => Dict("convergence" => convergence_product(cfg, s).key, "refine" => refine_product(rc, s).key,
                                     "start" => start_product(rc, s).key) for s in (p["left"], p["right"]))
    return driver_product("panels", Dict{String, Any}("experiment" => cfg["experiment"], "panels" => p, "upstream" => up);
                          output = "numbers.tex", threads = nothing)
end

"""
    convergence_panel_numbers!(nums, p, case, its, det, heldout) -> nums

The macros of one panel the paper's sentence cites (prefix `[panels] macro_prefix`, the panel's `macro`): the detections' median at the
start, at `rise_from_iteration` (where the sentence starts their rise) and at the end, the held-out median at the start and the end.
"""
function convergence_panel_numbers!(nums, p::AbstractDict, case::AbstractString, its, det, heldout)
    m(parts...) = Reproduce.number_macro_name(String(p["macro_prefix"]), String(p[case]["macro"]), parts...)
    r = findfirst(==(p["rise_from_iteration"]), its)
    r === nothing && throw(ArgumentError("[panels] rise_from_iteration $(p["rise_from_iteration"]) is not an iteration of $case"))
    for (n, v) in (("detections start", first(det)), ("detections rise", det[r]), ("detections end", last(det)),
                   ("heldout start", first(heldout)), ("heldout end", last(heldout)))
        Reproduce.add_number!(nums, m(n), v; digits = 2, label = "$case: $n median [px]", better = :lower)
    end
    return nums
end
