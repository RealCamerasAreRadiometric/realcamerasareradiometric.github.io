# =============================================================================
# src/detector_bias.jl — the detector-bias experiment (scripts/detector_bias.jl; its config of record
# configs/experiments/detector_bias.toml): fig:detector-bias. Per case — a set of renders with a known camera at known poses, a
# detector-based geometric calibration and its radiometric refinement — the evaluation against the truth, at the start (iteration 0)
# and at the refinement's last checkpoint:
#
#   learned_it<k>.json    every true board point projected with the camera under the run's own poses, minus its true pixel
#   truepose_it<k>.json   the same under the TRUE poses, one global rotation of all rays removed first ([evaluate] gauge)
#   scores.json           the two records' summaries per iteration, and the last iteration
#
# and the paper's figure from the cases' evaluations, the render pipeline's detections of each whole set and one render per row
# (src/paper_figures/detector_bias.jl). The projections and their records are src/evaluation.jl's.
#
# Included after src/drivers.jl, src/evaluation.jl and src/paper_figures/PaperFigures.jl.
# =============================================================================
using .Evaluation: poses_json_of_state, reprojection_vectors, truepose_vectors

"The tables of configs/experiments/detector_bias.toml."
const DETECTOR_BIAS_TABLES = ("experiment", "inputs", "runs", "image_regions", "views", "cases", "evaluate", "figure")
"""The keys of a `[cases.<name>]` table, every one required: the model kind it refines (`kind`: lens | grid), its renders (`views`, a
`[views]` list), its truth report (`truth`, an `[inputs]` id of stage detect: poses.json, the true poses beside it) and its
`[pipeline]` table, whose `io.images` names the folder of its renders."""
const DETECTOR_BIAS_CASE_KEYS = ("kind", "views", "truth", "pipeline")
"""The keys of `[figure]` beside its rows' tables, every one required: the figure's name under plotsdir() (`asset_dir`), the rows
from the top (`order`, each a `[figure.<row>]` table), the sensor bins of a field (`bins`, columns × rows), a field image's pixels
(`field_px`), the shown render's pixels and JPEG quality (`render_px`, `render_jpeg_quality`) and the fewest detections and board
points a drawn bin holds (`bias_min_features`, `calibration_min_points`)."""
const DETECTOR_BIAS_FIGURE_KEYS = ("asset_dir", "order", "bins", "field_px", "render_px", "render_jpeg_quality", "bias_min_features", "calibration_min_points")
"""The keys of a row's table `[figure.<row>]`, every one required: the case its calibration panels show (`case`), the `[inputs]` id
of its set's record of all detections (`detections`, of stage figures, pinned by its sha256), the view shown (`render`, one of the
case's views), the name of its files (`files`), its names in the figure (`detector`, `target`, `model`, `camera`: LaTeX text), what
its detector finds (`features`: corners | stars), the quantile that is the radius of its calibration pairs' keys (`key_quantile`)
and its notes (`notes`: panel = text)."""
const DETECTOR_BIAS_ROW_KEYS = ("case", "detections", "render", "files", "detector", "target", "model", "camera", "features", "key_quantile", "notes")

"""
    detector_bias_renders(cfg, case) -> String

The `[inputs]` id of the renders of `[cases.<case>]`: the one input whose path is the case's `[pipeline.io] images`. A case without
`io.images`, or whose folder is no input (or several), is an `ArgumentError`.
"""
function detector_bias_renders(cfg::AbstractDict, case::AbstractString)
    images = get(get(cfg["cases"][case]["pipeline"], "io", Dict{String, Any}()), "images", "")
    ids = sort([id for (id, t) in cfg["inputs"] if t["path"] == images])
    length(ids) == 1 || throw(ArgumentError("[cases.$case.pipeline.io] images = $(repr(images)): the folder of the case's renders is the path of one input of the config, found $(length(ids))"))
    return only(ids)
end

"""
    detector_bias_rows(cfg) -> Vector{NamedTuple}

The figure's rows from the top as the drawing code takes its settings (src/paper_figures/detector_bias.jl): per `[figure.<row>]`
`(; files, detector, target, model, camera, features, key_quantile, notes)`.
"""
detector_bias_rows(cfg::AbstractDict) =
    [(; files = String(t["files"]), detector = String(t["detector"]), target = String(t["target"]), model = String(t["model"]), camera = String(t["camera"]),
        features = String(t["features"]), key_quantile = Float64(t["key_quantile"]), notes = Dict{String, String}(String(k) => String(v) for (k, v) in t["notes"]))
     for t in (cfg["figure"][r] for r in cfg["figure"]["order"])]

"""
    load_detector_bias_config(path) -> Dict

The config of record of the detector-bias experiment, every table and key checked (`load_driver_config`): every case key required,
its kind lens | grid, its views a `[views]` list, its truth an input of stage detect, its renders an input (`detector_bias_renders`);
`[evaluate] gauge` a switch; `[figure]` exactly its keys and a table per row of `order` — each row a case of the config, a
detections input of stage figures pinned by its sha256, a render among its case's views, and the settings the drawing code takes
(checked by it: `db_check_rows`).
"""
function load_detector_bias_config(path::AbstractString)
    c = load_driver_config(path, "detector_bias"; tables = DETECTOR_BIAS_TABLES, case_keys = DETECTOR_BIAS_CASE_KEYS)
    for t in ("views", "cases", "evaluate", "figure")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    check_tables(c, path, "cases", DETECTOR_BIAS_CASE_KEYS; required = DETECTOR_BIAS_CASE_KEYS)
    for (n, v) in c["views"]
        v isa AbstractVector && !isempty(v) && all(x -> x isa AbstractString, v) && allunique(v) || throw(ArgumentError("$path [views] $n must be a list of distinct image names"))
    end
    for (n, t) in c["cases"]
        where = "$path [cases.$n]"
        t["kind"] in REFINEMENT_KINDS || throw(ArgumentError("$where: kind $(repr(t["kind"])) is not one of $(REFINEMENT_KINDS)"))
        t["views"] isa AbstractString && haskey(c["views"], t["views"]) || throw(ArgumentError("$where: views must name a [views] list"))
        i = get(c["inputs"], t["truth"], nothing)
        i !== nothing && i["stage"] == "detect" || throw(ArgumentError("$where: truth = $(repr(t["truth"])) is not an input of stage detect"))
        detector_bias_renders(c, n)
    end
    e = check_required_keys(c["evaluate"], ("gauge",), "$path [evaluate]")
    RadiometricCamera.read_switch(e["gauge"], "$path [evaluate] gauge")
    f = c["figure"]; where = "$path [figure]"
    order = get(f, "order", nothing)
    order isa AbstractVector && !isempty(order) && all(x -> x isa AbstractString, order) && allunique(order) && isdisjoint(order, DETECTOR_BIAS_FIGURE_KEYS) ||
        throw(ArgumentError("$where order = $(repr(order)): the rows from the top, each once, each a [figure.<row>] table"))
    check_required_keys(f, (DETECTOR_BIAS_FIGURE_KEYS..., order...), where)
    f["asset_dir"] isa AbstractString && !isempty(f["asset_dir"]) || throw(ArgumentError("$where asset_dir: the figure's name, a non-empty string"))
    for k in ("bins", "field_px", "render_px")
        v = f[k]
        v isa AbstractVector && length(v) == 2 && all(x -> x isa Integer && x > 0, v) || throw(ArgumentError("$where $k: [columns, rows], two positive integers"))
    end
    f["render_jpeg_quality"] isa Integer && 1 <= f["render_jpeg_quality"] <= 100 || throw(ArgumentError("$where render_jpeg_quality: an integer from 1 to 100"))
    for k in ("bias_min_features", "calibration_min_points")
        f[k] isa Integer && f[k] >= 1 || throw(ArgumentError("$where $k: a positive integer"))
    end
    for r in order
        t = check_required_keys(f[r], DETECTOR_BIAS_ROW_KEYS, "$where.$r")
        haskey(c["cases"], t["case"]) || throw(ArgumentError("$where.$r: case = $(repr(t["case"])) is no case of the config"))
        d = get(c["inputs"], t["detections"], nothing)
        d !== nothing && d["stage"] == "figures" && haskey(d, "sha256") ||
            throw(ArgumentError("$where.$r: detections = $(repr(t["detections"])) is not an input of stage figures pinned by its sha256"))
        t["render"] in c["views"][c["cases"][t["case"]]["views"]] || throw(ArgumentError("$where.$r: render = $(repr(t["render"])) is not a view of case $(t["case"])"))
        all(k -> t[k] isa AbstractString && !isempty(t[k]), ("files", "detector", "target", "model", "camera", "features")) ||
            throw(ArgumentError("$where.$r: files, detector, target, model, camera and features are non-empty strings"))
        t["features"] in ("corners", "stars") || throw(ArgumentError("$where.$r: features = $(repr(t["features"])) is not corners or stars"))
        t["key_quantile"] isa Real || throw(ArgumentError("$where.$r: key_quantile is a number in (0, 1]"))
        t["notes"] isa AbstractDict && all(v -> v isa AbstractString && !isempty(v), values(t["notes"])) || throw(ArgumentError("$where.$r: notes is a table, panel = its text"))
    end
    try
        PaperFigures.DetectorBias.db_check_rows(detector_bias_rows(c))           # the drawing code's own check of what it takes
    catch err
        err isa ErrorException ? throw(ArgumentError("$where: $(err.msg)")) : rethrow()
    end
    return c
end

"""
    detector_bias_evaluate_product(cfg, case) -> Product

The evaluation of `[cases.<case>]` (its output scores.json, written last): keyed by `[evaluate]`, the case's truth report and the
keys of its start and refine products.
"""
function detector_bias_evaluate_product(cfg::AbstractDict, case::AbstractString)
    t = cfg["cases"][case]
    return driver_product("evaluate", Dict{String, Any}("experiment" => cfg["experiment"], "evaluate" => cfg["evaluate"], "case" => case,
                                                        "truth" => input_record(cfg["inputs"][t["truth"]]),
                                                        "upstream" => Dict{String, Any}("refine" => refine_product(cfg, case).key, "start" => start_product(cfg, case).key));
                          output = "scores.json")
end

"""
    detector_bias_start_camera(cfg, case, start) -> String

The camera `[cases.<case>]` starts from, as RadiometricCamera's lens spec (`load_lens`): its `io.init` (a global model's record under
data_root()), else the spline its start wrote (`grid:<start>/grid.json`; `start` is the directory of its start product).
"""
function detector_bias_start_camera(cfg::AbstractDict, case::AbstractString, start::AbstractString)
    v = get(get(cfg["cases"][case]["pipeline"], "io", Dict{String, Any}()), "init", "")
    isempty(v) && return "grid:" * joinpath(start, "grid.json")
    m = match(r"^(lens|grid):(.*)$", v)
    m === nothing && throw(ArgumentError("[cases.$case.pipeline.io] init = $(repr(v)): \"lens:<record>\" or \"grid:<record>\""))
    return m.captures[1] * ":" * joinpath(data_root(), m.captures[2])
end

"""
    evaluate_detector_bias(cfg, case, dir; start, refine, key) -> Dict

The evaluation of `[cases.<case>]` into `dir`, from the directories of its start product (`start`: the start's state, the spline
of a grid case) and its refine product (`refine`: the checkpoints `<kind>_it<N>.json`, `poses_it<N>.json`). At iteration 0 (the
start camera, `detector_bias_start_camera`, under the start's poses, written here as poses_it0.json) and at the last checkpoint:
`truepose_vectors` (truepose_it<k>.json; the gauge removed as `[evaluate] gauge` states) and `reprojection_vectors`
(learned_it<k>.json) against the case's truth report. scores.json, written last, holds their summaries per iteration and
`last_iteration`; `key` is the run's key, named in its note. Returns that record's body.
"""
function evaluate_detector_bias(cfg::AbstractDict, case::AbstractString, dir::AbstractString; start::AbstractString, refine::AbstractString, key::AbstractString)
    t = cfg["cases"][case]; kind = t["kind"]
    truth = config_input(cfg, t["truth"])
    gauge = RadiometricCamera.read_switch(cfg["evaluate"]["gauge"], "[evaluate] gauge")
    its = checkpoint_iterations(refine, kind)
    isempty(its) && error("case $case: no checkpoint $(kind)_it<N>.json in $refine")
    n = maximum(its)
    summary(s) = Dict{String, Any}(String(k) => v for (k, v) in pairs(s))
    scores = Dict{String, Any}()
    for (it, camera, poses) in ((0, detector_bias_start_camera(cfg, case, start), poses_json_of_state(joinpath(start, "state.json"), joinpath(dir, "poses_it0.json"))),
                                (n, "$kind:" * joinpath(refine, "$(kind)_it$n.json"), joinpath(refine, "poses_it$n.json")))
        tv = truepose_vectors(truth, camera, joinpath(dir, "truepose_it$it.json"); label = "$case it $it, true poses", gauge)
        lv = reprojection_vectors(truth, camera, poses, joinpath(dir, "learned_it$it.json"); label = "$case it $it, learned poses")
        scores["it$it"] = Dict{String, Any}("iteration" => it, "lens" => basename(camera), "truepose" => summary(tv), "learned" => summary(lv))
    end
    scores["last_iteration"] = n
    scores["note"] = "the evaluation of case $case (run $key): the camera alone under the true poses and the reprojection error under the run's own " *
                     "poses, at the start (it0) and at the last checkpoint; px"
    VgtkIO.write_record(joinpath(dir, "scores.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.detector_bias_scores), scores); allownan = true)
    return scores
end

"The cases the figure's rows show, sorted."
detector_bias_figure_cases(cfg::AbstractDict) = sort(unique(String(cfg["figure"][r]["case"]) for r in cfg["figure"]["order"]))

"""
    detector_bias_figures_product(cfg) -> Product

The figure's product (its output numbers.tex, written last; the tables in `tables/`, the figure's files in `assets/`): keyed by
`[figure]`, the rows' detections inputs (tier, path, sha256) and the keys of the evaluations it reads. One result at any thread
count.
"""
detector_bias_figures_product(cfg::AbstractDict) =
    driver_product("figures", Dict{String, Any}("experiment" => cfg["experiment"], "figure" => cfg["figure"],
                                                "inputs" => inputs_record(cfg, sort(unique(String(cfg["figure"][r]["detections"]) for r in cfg["figure"]["order"]))),
                                                "upstream" => Dict{String, Any}(c => detector_bias_evaluate_product(cfg, c).key for c in detector_bias_figure_cases(cfg)));
                   output = "numbers.tex", threads = nothing)

"""
    detector_bias_render(cfg, row) -> String

The render the row `[figure.<row>]` shows: its view in the folder of its case's renders (that input, checked: `config_input`).
"""
detector_bias_render(cfg::AbstractDict, row::AbstractString) =
    (t = cfg["figure"][row]; joinpath(config_input(cfg, detector_bias_renders(cfg, t["case"])), t["render"]))

"""
    write_detector_bias_figure(cfg, dir, evaluations; generator) -> Vector{String}

The paper's figure into `dir`, by its drawing code (src/paper_figures/detector_bias.jl, both parts): the tables in `dir`/tables —
per row of `[figure]` its case's evaluation (`evaluations`: case => the directory of its evaluate product), its set's detections
(the row's input, its sha256 checked) and its render (`detector_bias_render`) — and the figure's files
in `dir`/assets, with the settings of `[figure]` and `[evaluate] gauge`. `generator` is what the files name as their generator.
Returns the files written.
"""
function write_detector_bias_figure(cfg::AbstractDict, dir::AbstractString, evaluations::AbstractDict; generator::AbstractString)
    F = cfg["figure"]
    rows = [(; s..., evaluate = String(evaluations[F[r]["case"]]), detections = config_input(cfg, F[r]["detections"]), render = detector_bias_render(cfg, r))
            for (s, r) in zip(detector_bias_rows(cfg), F["order"])]
    return PaperFigures.generate("detector-bias-mitsuba"; part = "all", data = joinpath(dir, "tables"), out = joinpath(dir, "assets"), generator,
                                 asset_dir = F["asset_dir"], rows, bins = F["bins"], field_px = F["field_px"], render_px = F["render_px"],
                                 render_jpeg_quality = F["render_jpeg_quality"], bias_min_features = F["bias_min_features"],
                                 calibration_min_points = F["calibration_min_points"],
                                 camera_gauge = RadiometricCamera.read_switch(cfg["evaluate"]["gauge"], "[evaluate] gauge"))
end
