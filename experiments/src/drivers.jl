# =============================================================================
# src/drivers.jl — what every staged driver of the project shares, after the
# prelude (src/experiments.jl): the declared inputs of a config of record (catalog
# tier + path under data_root()), the keyed steps of every driver (Eval's Product:
# DrWatson's produce_or_load, datadir("exp_pro", <experiment>, <stage>, <key>);
# the thread count keys a step: it is part of the arithmetic), the pipeline run
# (RadiometricCamera's bin/refine.jl in a child process at data_root()), the RAW
# decoder (VisualGeometryToolkitFeatures' load_raw on LibRaw_jll, in Julia, every
# mode: the one RAW path) and the numbers.tex fragments.
#
# A driver's config of record names every input as a table
#
#   [inputs.<id>]
#   tier = "rc_baselines"   # the catalog tier the input belongs to (configs/data_artifacts.toml packs the tiers)
#   path = "<relative to its tier's root, data_root(; tier)>"
#   sha256 = "…"            # optional, an input of one file: its content, checked wherever it is read (Eval's input_path)
#   stage = "detect"        # the stage that requires it (prepare | detect | index | init | refine | evaluate | figures)
#   provenance = "…"        # how it was made: the tool and every setting; quoted when it is missing and recorded with the run
#
# and nothing else names a file under data_root(). A Git LFS pointer is not data:
# `require_data` refuses one.
# =============================================================================
using RadiometricCamera: RadiometricCamera, pipeline_options
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitEval: VisualGeometryToolkitEval, cli_option
using VisualGeometryToolkitEval: inspect_input
using VisualGeometryToolkitEval.Experiments: assert_not_input, input_path
import VisualGeometryToolkitEval.Reproduce as Reproduce                 # the stack's one number-macro writer (driver_numbers)
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
import PNGFiles
using ColorTypes: RGB, N0f8, N0f16
import StaticArrays
using Statistics: quantile, mean

"The catalog tiers an input belongs to (spec §6): the raw photographs, the synthetic renders, the baseline tools' outputs, the calibration targets and cases, the chrome evaluation target and its photographs, the FX3 14 mm capture of 2026-11-02."
const CATALOG_TIERS = ("rc_raw_photos", "rc_synthetic", "rc_baselines", "rc_calibration", "rc_thorlabs", "rc_fx3_14mm")
"""Every tier an input may name: the catalogued ones and VisualGeometryToolkitEval's reserved `unpublished` (data of record no
artifact publishes, kept on one machine and pinned by its `sha256`: the external calibrators' container images; its root the site's
`[sites.<site>.tiers]` of configs/reproduce.toml)."""
const INPUT_TIERS = (CATALOG_TIERS..., Reproduce.UNPUBLISHED_TIER)
"The stages of a driver, in run order; an input names the one that requires it."
const DRIVER_STAGES = ("prepare", "detect", "index", "init", "start", "refine", "evaluate", "figures")
"The keys of an `[inputs.<id>]` table."
const INPUT_KEYS = ("tier", "path", "stage", "provenance")
"""The optional content key of an input of one file: `sha256`, the file's SHA-256 (64 hex digits). VisualGeometryToolkitEval checks
it wherever the input is read (`Experiments.input_path`, the runner's `missing_inputs`), and it keys every run that reads the input
([`input_record`](@ref))."""
const INPUT_CONTENT_KEYS = ("sha256",)
"""The optional keys of an input of raw photographs, a SELECTION of its folder ([`selected_raw_dir`](@ref)): `frames`, exactly the
named files, or `exclude`, the folder less the named files — one of them at most, each a non-empty array of distinct file names. A
config states the subset it uses beside the published folder it comes from, so the folder stays whole; the selection keys every run
that reads the input ([`input_record`](@ref))."""
const INPUT_SELECTION_KEYS = ("frames", "exclude")

"""
    execution_threads() -> String

This process's thread count as the pipeline receives it, `"<default>,<interactive>"`. The closed-form start sums over thread chunks,
so the thread count is part of the arithmetic: it keys every run and is recorded in its provenance.
"""
execution_threads() = "$(Threads.nthreads()),$(Threads.nthreads(:interactive))"

"""
    with_execution(record) -> Dict

The effective config of a run: `record` plus `[execution] threads` ([`execution_threads`](@ref)). Its `run_key` names the run's
directory, and `_config.toml` holds it, so a run at another thread count is another run.
"""
with_execution(record::AbstractDict, threads::AbstractString = execution_threads()) =
    merge(Dict{String,Any}(record), Dict{String,Any}("execution" => Dict{String,Any}("threads" => threads)))

"""
    driver_product(stage, record; experiment = product_experiment(record["experiment"]["name"]), output, threads = execution_threads()) -> Product

The product of stage `stage` of `experiment` (the driver's name, a config of record's `[experiment] name`, under its group's directory:
[`product_experiment`](@ref)), keyed by `record` — ONLY
what the product reads: its own table, its inputs (their tier, path and pinned sha256), the keys of the products it reads
(`"upstream"`) — by the code it runs (`CODE_VERSION`) and, with `threads`, by the thread count (`with_execution`: a computation
whose floating-point sums run over thread chunks — bin/refine.jl's closed-form start — is another one at another count; a
thread-invariant computation (one result per photograph, or a parallel map equal to the serial one) takes `threads = nothing`): VisualGeometryToolkitEval's
`Product`, produced or loaded by `produce` (DrWatson's `produce_or_load`) in `datadir("exp_pro", experiment, stage, key)`;
`output` is the file it writes last (the product's result). `threads` is the thread count that keys it: this process's by default,
another for a child run at another count.
"""
driver_product(stage::AbstractString, record::AbstractDict; experiment::AbstractString = product_experiment(record["experiment"]["name"]), output::AbstractString,
               threads::Union{Nothing, AbstractString} = execution_threads()) =
    Product(experiment, stage, threads === nothing ? Dict{String, Any}(record) : with_execution(record, threads); output, code = CODE_VERSION)

"""The groups of experiments beside configs/experiments/ (owner 2026-11-18): `supplement` (experiments of record whose assets appear in
the supplement alone) and `studies` (ideas tried that never came to fruition, each with the decision of record it backed). A group's
configs lie in configs/<group>/ and its products in results_root()/<group>/…; the location keys nothing."""
const EXPERIMENT_GROUPS = ("supplement", "studies")

"""
    config_group(path) -> Union{Nothing, String}

The group of the config at `path`: the [`EXPERIMENT_GROUPS`](@ref) entry whose configs/<group>/ holds it, else `nothing`.
"""
function config_group(path::AbstractString)
    p = normpath(abspath(path))
    for g in EXPERIMENT_GROUPS
        startswith(p, normpath(projectdir("configs", g)) * "/") && return g
    end
    return nothing
end

"""Per experiment (an `[experiment] name`), its group, for the names whose configs all lie in one group's directory (an override
config, `[experiment] base`, names its base's driver and is not counted: its variants are located by [`variant_location`](@ref));
a name with configs in a group and elsewhere is an error."""
const GROUP_OF_EXPERIMENT = let groups = Dict{String, Set{String}}()
    for dir in ("experiments", EXPERIMENT_GROUPS...)
        d = projectdir("configs", dir)
        isdir(d) || continue
        for f in readdir(d; join = true)
            endswith(f, ".toml") || continue
            e = get(TOML.parsefile(f), "experiment", Dict{String, Any}())
            haskey(e, "name") && !haskey(e, "base") && push!(get!(Set{String}, groups, String(e["name"])), dir)
        end
    end
    for (n, g) in groups
        length(g) > 1 && any(in(EXPERIMENT_GROUPS), g) &&
            error("experiment $n has configs in $(sort(collect(g))): a group's experiment lies in one directory (its products' location)")
    end
    Dict{String, String}(n => only(g) for (n, g) in groups if only(g) in EXPERIMENT_GROUPS)
end

"""
    product_experiment(name) -> String

Where the products of experiment `name` lie under results_root(): `<group>/<name>` for an experiment of a group
([`GROUP_OF_EXPERIMENT`](@ref)), else `<name>`. A location, never a key: moving a config between the groups re-keys nothing.
"""
product_experiment(name::AbstractString) = haskey(GROUP_OF_EXPERIMENT, name) ? "$(GROUP_OF_EXPERIMENT[name])/$name" : String(name)

"""
    variant_location(cfg, case) -> String

Where the start and refine products of `[cases.<case>]` of `cfg` lie: a variant an override config of a group made
(`expand_overrides` records it in `cfg["_locations"]`) under `<group>/<that config's stem>/<experiment>`, every other case where its
experiment's products lie ([`product_experiment`](@ref)). A location, never a key.
"""
variant_location(cfg::AbstractDict, case::AbstractString) =
    get(get(cfg, "_locations", Dict{String, Any}()), case, product_experiment(cfg["experiment"]["name"]))

"""The code version every product of this process is keyed by (`Product`'s `code`): VisualGeometryToolkitEval's
`current_code_trees` of the project — every stack package's source tree (a path checkout's git trees, a registered version's
Manifest tree hash), RadiometricCamera's (a path dependency) and the project's `src` — read once when the drivers load."""
const CODE_VERSION = VisualGeometryToolkitEval.Experiments.current_code_trees(projectdir())

"""
    config_product(stage, cfg, tables; upstream = (), output) -> Product

The step of a stage keyed by the config of record `cfg`'s `[experiment]` and its tables `tables` (by name), and the keys of the steps
it reads (`upstream`: `name => product`): [`driver_product`](@ref) of that record. A `[render]` table keys it as `render_files`
(VisualGeometryToolkitEval's `Product` refuses a top-level `render`, which a run key excludes: the rendering keys the product here).
"""
config_product(stage::AbstractString, cfg::AbstractDict, tables; upstream = (), output::AbstractString, threads::Union{Nothing, AbstractString} = execution_threads()) =
    driver_product(stage, Dict{String, Any}("experiment" => cfg["experiment"], ((t == "render" ? "render_files" : t) => cfg[t] for t in tables)...,
                                            "upstream" => Dict(k => s.key for (k, s) in upstream)); output, threads)

"The records of a driver's steps: the driver and its config of record (`produce`'s `driver` and `config_source`)."
product_records(driver::AbstractString, config_path::AbstractString) = (; driver, config_source = config_path)

# ---- the config of record ---------------------------------------------------------------------------------------------------

"""
    check_input_table(t, where; stage = true)

One `[inputs.<id>]` table, the tiered input schema's one check: its keys `INPUT_KEYS` (without `stage` when `stage = false`: an
input no driver stage reads, a record an author-side script writes), each a non-empty string, and the optional `INPUT_CONTENT_KEYS`
(and, of a stage's input, `INPUT_SELECTION_KEYS`: at most one, a non-empty array of distinct file names); `tier` one of
`INPUT_TIERS`, `stage` one of `DRIVER_STAGES`, `path` relative to its tier's root and inside it.
"""
function check_input_table(t, where::AbstractString; stage::Bool = true)
    t isa AbstractDict || throw(ArgumentError("$where must be a table"))
    required = stage ? INPUT_KEYS : filter(!=("stage"), INPUT_KEYS)
    VgtkCore.check_keys(t, (required..., INPUT_CONTENT_KEYS..., (stage ? INPUT_SELECTION_KEYS : ())...), where)
    for k in required
        VgtkCore.required_key(t, k, where) isa AbstractString && !isempty(t[k]) || throw(ArgumentError("$where: `$k` must be a non-empty string"))
    end
    t["tier"] in INPUT_TIERS || throw(ArgumentError("$where: tier \"$(t["tier"])\" is not one of $(INPUT_TIERS)"))
    stage && (t["stage"] in DRIVER_STAGES || throw(ArgumentError("$where: stage \"$(t["stage"])\" is not one of $(DRIVER_STAGES)")))
    p = t["path"]
    (isabspath(p) || startswith(p, "~") || ".." in splitpath(p)) &&
        throw(ArgumentError("$where: path \"$p\" must be relative to data_root() and stay inside it"))
    (p == RETIRED_IMAGE_CIRCLE_INPUTS || startswith(p, RETIRED_IMAGE_CIRCLE_INPUTS * "/")) &&
        throw(ArgumentError("$where: path \"$p\" is retired — the shared image-circle photographs; a capture set's image region is " *
                            "measured on its own frames ([image_regions.<name>], src/image_region.jl; owner ruling 2026-11-15)"))
    sel = [k for k in INPUT_SELECTION_KEYS if haskey(t, k)]
    length(sel) <= 1 || throw(ArgumentError("$where: one of $(join(INPUT_SELECTION_KEYS, ", ")), not both"))
    for k in sel
        v = t[k]
        v isa AbstractVector && !isempty(v) && all(x -> x isa AbstractString && !isempty(x) && !occursin('/', x), v) && allunique(v) ||
            throw(ArgumentError("$where: `$k` must be a non-empty array of distinct file names (no paths)"))
    end
    return t
end

"""
    load_driver_config(path, name; tables, case_keys = ()) -> Dict

Read the config of record of driver `name` at `path`: the top-level tables must be among `tables` (each driver lists its own; an
unknown one is an `ArgumentError` naming it), `[experiment] name` must be `name` and the file `<name>.toml` or `<name>_<variant>.toml`
(another config of record of the same driver), every `[inputs.<id>]` is
checked ([`INPUT_KEYS`](@ref), a known tier and stage, a relative path inside data_root()), and every `[cases.<name>]` takes only
`case_keys`, its `pipeline` table read through `pipeline_options` (refused keys name the case).
"""
function load_driver_config(path::AbstractString, name::AbstractString; tables, case_keys = ())
    isfile(path) || throw(ArgumentError("config not found: $path"))
    ex = excluded_sets()
    for (set, e) in ex                                                    # an excluded set's own config: refused by name
        any(c -> normpath(joinpath(projectdir(), c)) == normpath(abspath(path)), get(e, "configs", String[])) &&
            throw(ArgumentError("$path is the config of the excluded set $set (configs/excluded.toml): $(e["reason"])"))
    end
    cfg = skip_excluded(expand_overrides(inherit_kinds!(TOML.parsefile(path), path), path), ex)
    VgtkCore.check_keys(cfg, (tables..., "_locations"), path)             # `_locations`: where a group's variants lie (`variant_location`)
    for t in ("experiment", "inputs")
        haskey(cfg, t) && cfg[t] isa AbstractDict || throw(ArgumentError("$path: no [$t] table"))
    end
    VgtkCore.check_keys(cfg["experiment"], ("name",), "$path [experiment]")
    get(cfg["experiment"], "name", "") == name || throw(ArgumentError("$path: [experiment] name must be \"$name\""))
    stem = splitext(basename(path))[1]
    stem == name || startswith(stem, name * "_") ||
        throw(ArgumentError("$path: a config of driver $name is configs/experiments/$name.toml (or $(name)_<variant>.toml)"))
    foreach(((id, t),) -> check_input_table(t, "$path [inputs.$id]"), cfg["inputs"])
    check_runs_table(cfg, path)
    check_image_regions(cfg, path)
    for (c, t) in get(cfg, "cases", Dict{String,Any}())
        where = "$path [cases.$c]"
        t isa AbstractDict || throw(ArgumentError("$where must be a table"))
        VgtkCore.check_keys(t, case_keys, where)
        haskey(t, "pipeline") || continue
        for k in RadiometricCamera.PIPELINE_OUTPUT_KEYS                  # io.out, io.save_init, lm.log: the driver's (the run's keyed directory)
            tb, f = split(k, '.')
            isempty(get(get(t["pipeline"], tb, Dict()), f, "")) || throw(ArgumentError("$where: $k is the driver's (the run's keyed directory)"))
        end
        try
            pipeline_options(t["pipeline"])
        catch err
            err isa ArgumentError ? throw(ArgumentError("$where [pipeline]: $(err.msg)")) : rethrow()
        end
    end
    for tb in ("init", "cases"), (c, t) in get(cfg, tb, Dict{String, Any}())          # every run states its set's lens coverage
        t isa AbstractDict && haskey(t, "pipeline") || continue
        cam = get(t["pipeline"], "camera", Dict{String, Any}())
        haskey(cam, "lens_coverage") || throw(ArgumentError("$path [$tb.$c.pipeline.camera]: no lens_coverage — every config of record states its " *
                                                             "set's coverage: \"sensor\" or \"image_region:<name>\" (src/image_region.jl)"))
        check_lens_coverage(cfg, cam["lens_coverage"], "$path [$tb.$c.pipeline.camera]")
    end
    return cfg
end

"""
    cli_subset(argv, option, all; what = option) -> Vector{String}

The names of `all` (sorted) a run treats: all of them, or those `--<option>=a,b` names, sorted (an unknown name is an `ArgumentError`
naming the `what`s).
"""
function cli_subset(argv, option::AbstractString, all::AbstractVector{<:AbstractString}; what::AbstractString = option)
    sel = cli_option(argv, option, "")
    isempty(sel) && return collect(String, all)
    want = String.(split(sel, ','))
    bad = setdiff(want, all)
    isempty(bad) || throw(ArgumentError("unknown $what(s) $(bad); the config's $(what)s are $(collect(all))"))
    return sort(want)
end

"""
    cli_shard(argv, items) -> Vector

The items of shard `k` of `n` (`--shard=k/n`, `0 ≤ k < n`): every `n`-th item from the `k+1`-th, in their order, so `n` array
tasks of a launch (configs/reproduce.toml, one GPU each) split a stage's items between them and together run each once; all
of them without `--shard`. A malformed or out-of-range shard is an `ArgumentError`.
"""
function cli_shard(argv, items::AbstractVector)
    s = cli_option(argv, "shard", "")
    isempty(s) && return collect(items)
    m = match(r"^([0-9]+)/([0-9]+)$", s)
    m === nothing && throw(ArgumentError("--shard=$s is not k/n"))
    k, n = parse(Int, m[1]), parse(Int, m[2])
    0 <= k < n || throw(ArgumentError("--shard=$s: k must be in 0:$(n - 1)"))
    return collect(items)[(k + 1):n:end]
end

"""
    case_names(cfg, argv = ARGS) -> Vector{String}

The cases of `cfg` a run treats, sorted: all of them, or those `--case=a,b` names (an unknown name is an `ArgumentError`).
"""
case_names(cfg::AbstractDict, argv = ARGS) = cli_subset(argv, "case", sort(collect(keys(get(cfg, "cases", Dict{String,Any}())))); what = "case")

# ---- the declared inputs ------------------------------------------------------------------------------------------------------

"The ids of the inputs and `[runs]` of `cfg` that stage `stage` requires, sorted."
stage_inputs(cfg::AbstractDict, stage::AbstractString) =
    sort([id for tb in ("inputs", "runs") for (id, t) in get(cfg, tb, Dict{String, Any}()) if t["stage"] == stage])

const LFS_POINTER_HEAD = "version https://git-lfs.github.com/spec"

"Whether `file` is a Git LFS pointer (a stub that names the data, not the data)."
function is_lfs_pointer(file::AbstractString)
    isfile(file) && filesize(file) < 1024 || return false
    head = open(io -> String(read(io, length(LFS_POINTER_HEAD))), file)
    return head == LFS_POINTER_HEAD
end

"""
    require_data(path; what, provenance) -> path

`require_input` (a missing input is an error quoting its provenance), and a Git LFS pointer is refused: the file itself, or any
file directly inside the directory `path`.
"""
function require_data(path::AbstractString; what::AbstractString, provenance::AbstractString)
    require_input(path; what, provenance)
    stubs = isdir(path) ? filter(is_lfs_pointer, [joinpath(path, f) for f in readdir(path)]) : filter(is_lfs_pointer, [path])
    isempty(stubs) || error("INPUT IS A GIT LFS POINTER [$what]: $(first(stubs))$(length(stubs) > 1 ? " (and $(length(stubs) - 1) more)" : "")\n" *
                            "  a pointer names the data, it is not the data; provenance: $provenance")
    return path
end

"""
    require_stage_inputs(cfg, stage) -> Vector{String}

Every input of `stage`, resolved under data_root() and checked by [`require_data`](@ref); the absolute paths, in id order.
"""
require_stage_inputs(cfg::AbstractDict, stage::AbstractString) = String[config_source(cfg, id) for id in stage_inputs(cfg, stage)]   # typed: a stage without inputs of its own gives an empty comprehension of an inferred, abstract eltype (produce's inputs: a Vector{String})

"""
    excluded_sets() -> Dict{String, Any}

The capture sets excluded from the paper, configs/excluded.toml's `[excluded.<set>]` tables (`reason`, `configs`, `cases`): the one list
every driver reads (through [`load_driver_config`](@ref)); the data and catalogue entries stay.
"""
excluded_sets() = (f = projectdir("configs", "excluded.toml"); isfile(f) ? get(TOML.parsefile(f), "excluded", Dict{String, Any}()) : Dict{String, Any}())

"""
    expand_overrides(cfg, path) -> cfg

An experiment as the config of record plus keyword OVERRIDES of the hyperparameters it varies, never a copied case (owner,
2026-11-15): a config whose `[experiment]` names a `base` (a config of record, project-relative) and its `cases` (of the base) and
whose `[overrides]` maps dotted `[pipeline]` keys to the values tried — `"masks.lens_coverage_scale" = [1.0, 0.9]` — is the base config
with those cases alone, each expanded into one case per combination of the values (DrWatson's `dict_list`). A variant whose values are
the case's own resolved ones (configs/default.toml ← the case ← the override) IS the case — its name, its table, so its run key: the run
of record is reused; another is the case with the keys set, named `<case>__<table.key>=<value>[…]`. The base's other tables (inputs, runs,
prepare, views) are kept as they are; `[experiment]` is the base's (the runs are keyed by the base's experiment). A config without a
`base` is returned as it is.
"""
function expand_overrides(cfg::AbstractDict, path::AbstractString)
    e = get(cfg, "experiment", Dict{String, Any}())
    haskey(e, "base") || return cfg
    VgtkCore.check_keys(cfg, ("experiment", "overrides"), path)
    VgtkCore.check_keys(e, ("name", "base", "cases"), "$path [experiment]")
    base = inherit_kinds!(TOML.parsefile(projectdir(e["base"])), e["base"])
    get(base["experiment"], "name", "") == e["name"] || throw(ArgumentError("$path: base $(e["base"]) is not a config of $(e["name"])"))
    ov = get(cfg, "overrides", Dict{String, Any}())
    ov isa AbstractDict && !isempty(ov) && all(v -> v isa AbstractVector && !isempty(v), values(ov)) ||
        throw(ArgumentError("$path [overrides]: each entry is `\"<table>.<key>\" = [values…]`"))
    variants = dict_list(Dict{String, Any}(k => collect(x) for (k, x) in ov))
    cases = Dict{String, Any}(); locations = Dict{String, Any}()
    g = config_group(path)                                          # a group's variant lies in its directory (`variant_location`)
    for c in e["cases"]
        t = get(base["cases"], c, nothing)
        t === nothing && throw(ArgumentError("$path: no case $c in $(e["base"])"))
        for v in variants
            n, t2 = override_variant(c, t, v)
            cases[n] = t2
            g === nothing || n == c || (locations[n] = "$g/$(splitext(basename(path))[1])/$(base["experiment"]["name"])")
        end
    end
    out = deepcopy(base); out["cases"] = cases
    isempty(locations) || (out["_locations"] = locations)
    return out
end
"""
    inherit_kinds!(cfg, path) -> cfg

The settings a config states once per model kind (owner 2026-11-16 23:30, samples_per_view: one place per setting, the cases inherit):
every key of `[kinds.<kind>.pipeline.<table>]` set in the [pipeline] of each `[cases.<c>]` of that `kind` (`"lens"` | `"grid"`); a
case that states a key its kind sets is refused (a case differs only inside a named experiment's override). `[kinds] source =
"<config>"` (project-relative) reads that config's `[kinds]` instead (the vignetting refinements inherit the refinements' of record).
The table is consumed: the cases carry the resolved values, so a case's run key is its resolved table's, whichever file states them.
"""
function inherit_kinds!(cfg::AbstractDict, path::AbstractString)
    k = pop!(cfg, "kinds", nothing)
    k === nothing && return cfg
    k isa AbstractDict || throw(ArgumentError("$path [kinds] must be a table"))
    if haskey(k, "source")
        length(k) == 1 || throw(ArgumentError("$path [kinds]: `source` names another config's [kinds]; nothing beside it"))
        src = String(k["source"])
        k = get(TOML.parsefile(projectdir(src)), "kinds", nothing)
        k isa AbstractDict && !haskey(k, "source") ||
            throw(ArgumentError("$path [kinds] source: $src states no [kinds.<kind>] tables of its own"))
    end
    for (kind, kt) in k
        where = "$path [kinds.$kind]"
        kind in REFINEMENT_KINDS || throw(ArgumentError("$where: a kind is one of $(REFINEMENT_KINDS)"))
        kt isa AbstractDict && keys(kt) == Set(["pipeline"]) && all(v -> v isa AbstractDict, values(kt["pipeline"])) ||
            throw(ArgumentError("$where: exactly a [kinds.$kind.pipeline.<table>] table per [pipeline] table it sets"))
    end
    for (c, t) in get(cfg, "cases", Dict{String, Any}())
        kt = get(k, get(t, "kind", ""), nothing)
        kt === nothing && continue
        pipe = get!(t, "pipeline", Dict{String, Any}())
        for (tb, kv) in kt["pipeline"], (key, v) in kv
            dst = get!(pipe, tb, Dict{String, Any}())
            haskey(dst, key) && throw(ArgumentError("$path [cases.$c.pipeline.$tb] $key: [kinds.$(t["kind"]).pipeline.$tb] sets it for every " *
                                                    "$(t["kind"]) case (one place per setting)"))
            dst[key] = deepcopy(v)
        end
    end
    return cfg
end
"The model kinds a refinement case writes (`[cases.<c>] kind`, `last_model`): a global (parametric) camera, or the central spline."
const REFINEMENT_KINDS = ("lens", "grid")

"""
    override_variant(case, t, v; unset = ()) -> (name, table)

The variant of the case `case` (its table `t`) under the overrides `v` (`"<table>.<key>" => value`) with the keys `unset` deleted
from its [pipeline] (a key the overrides make unusable: `INAPPLICABLE_KEYS`' rule, a value the run cannot use is never stated):
THE one rule of an experiment's variant ([`expand_overrides`](@ref); the warping ablation's modes, src/warping_ablation.jl). A variant
whose values are the case's own resolved ones (configs/default.toml ← the case ← the override) and that unsets nothing the case states
IS the case — its name, its table, so its run key: the run of record is reused; another is the case with the keys set, named
`<case>__<table.key>=<value>[…]` (`<table.key>=unset` for a deleted key), in key order.
"""
function override_variant(case::AbstractString, t::AbstractDict, v; unset = ())
    resolved = RadiometricCamera.config_dict(pipeline_options(t["pipeline"]))
    stated = [k for k in unset if _override_stated(t["pipeline"], k)]
    all(kx -> _override_value(resolved, kx[1]) == kx[2], v) && isempty(stated) && return (String(case), t)
    t2 = deepcopy(t)
    foreach(kx -> _set_override!(t2["pipeline"], kx...), v)
    foreach(k -> _unset_override!(t2["pipeline"], k), stated)
    parts = sort([[(String(k), "$k=$(_override_label(x))") for (k, x) in v]; [(String(k), "$k=unset") for k in stated]]; by = first)
    return (String(case) * "__" * join(last.(parts), "__"), t2)
end
# whether the [pipeline] table `pipe` states the key `k` (`table.key` or a top-level key); the key deleted (its table dropped when emptied)
_override_stated(pipe, k) = (p = split(k, '.'; limit = 2); length(p) == 1 ? haskey(pipe, k) : haskey(get(pipe, p[1], Dict()), p[2]))
_unset_override!(pipe, k) = (p = split(k, '.'; limit = 2); length(p) == 1 ? delete!(pipe, k) :
                             (delete!(pipe[p[1]], p[2]); isempty(pipe[p[1]]) && delete!(pipe, p[1])); pipe)
# an override's key: `table.key` (a key of a [pipeline] table) or a top-level key (`fixed`); its resolved value in `config_dict`'s form
_override_value(resolved, k) = (p = split(k, '.'; limit = 2); length(p) == 1 ? get(resolved, k, nothing) : get(get(resolved, p[1], Dict()), p[2], nothing))
# an override value in a variant's name: a table (an inline TOML table, e.g. `masks.edge_detector`) as `{k=v,…}` in its keys' order,
# so the name is the same whatever the Dict's iteration order; any other value as it prints
_override_label(x) = x isa AbstractDict ? "{" * join(("$k=$(_override_label(x[k]))" for k in sort!(collect(keys(x)))), ",") * "}" : string(x)
_set_override!(pipe, k, x) = (p = split(k, '.'; limit = 2); length(p) == 1 ? (pipe[k] = x) : (get!(pipe, p[1], Dict{String, Any}())[p[2]] = x); pipe)

"""
    skip_excluded(cfg) -> cfg

`cfg` without the cases (and the front ends' runs: `[cases]`, `[schoeps]`, `[fits]`, `[compare]` entries) an excluded set names
([`excluded_sets`](@ref)), each skipped with a line naming it and its reason.
"""
function skip_excluded(cfg::AbstractDict, excluded = excluded_sets())
    for (set, e) in excluded, tb in ("cases", "schoeps", "fits", "compare"), c in get(e, "cases", String[])
        t = get(cfg, tb, nothing)
        t isa AbstractDict && haskey(t, c) || continue
        delete!(t, c); println("skipped [$tb.$c]: the excluded set $set (configs/excluded.toml) — $(e["reason"])")
    end
    return cfg
end

"""
    config_input(cfg, id) -> String

The input `id` of `cfg`: VisualGeometryToolkitEval's `Experiments.input_path` (its `path` under its tier's root, `data_root(; tier)`;
refused when absent or, with a `sha256`, of another content: the error quotes the input's provenance), then checked by
[`require_data`](@ref) (a Git LFS pointer is refused). An id that is no input is refused by name (a `[runs]` entry: read by
[`config_source`](@ref))."""
function config_input(cfg::AbstractDict, id::AbstractString)
    if !haskey(get(cfg, "inputs", Dict()), id)
        haskey(get(cfg, "runs", Dict()), id) && throw(ArgumentError("$id is a [runs] entry, not an input: read it with config_source"))
        throw(ArgumentError("$id is not an input of the config ([inputs])"))
    end
    t = cfg["inputs"][id]
    p = try
        input_path(cfg, id)
    catch err
        err isa ArgumentError ? throw(ArgumentError("$(err.msg)\n  provenance: $(t["provenance"])")) : rethrow()
    end
    return require_data(p; what = "$id ($(t["tier"]))", provenance = t["provenance"])
end

"""
    selected_raw_dir(cfg, id, dir) -> String

The folder of raw photographs the input `id` of `cfg` stands for: its folder itself ([`config_input`](@ref)), or, for an input with a
selection (`frames` | `exclude`, [`INPUT_SELECTION_KEYS`](@ref)), the folder `dir/<id>_frames` of links to the selected files of it,
made here (idempotent: the run `dir` is keyed by the selection). Every named file must be in the folder, and a selection must leave
a file. The decoders and the panorama example read a folder, so a config's subset of a published folder is a folder of links in its run.
"""
function selected_raw_dir(cfg::AbstractDict, id::AbstractString, dir::AbstractString)
    t = cfg["inputs"][id]
    src = config_input(cfg, id)
    any(k -> haskey(t, k), INPUT_SELECTION_KEYS) || return src
    files = raw_files(src)
    names = Set(basename.(files))
    if haskey(t, "frames")
        bad = filter(n -> !(n in names), t["frames"])
        isempty(bad) || error("input $id: frames not in $src: $(join(bad, ", "))")
        wanted = Set(String.(t["frames"])); keep = [f for f in files if basename(f) in wanted]
    else
        bad = filter(n -> !(n in names), t["exclude"])
        isempty(bad) || error("input $id: exclude names files not in $src: $(join(bad, ", "))")
        dropped = Set(String.(t["exclude"])); keep = [f for f in files if !(basename(f) in dropped)]
    end
    isempty(keep) && error("input $id: the selection leaves no raw photograph of $src")
    out = joinpath(dir, "$(id)_frames")
    assert_not_input(out)
    mkpath(out)
    for f in keep
        dst = joinpath(out, basename(f))
        islink(dst) || symlink(f, dst)
    end
    n = count(e -> islink(joinpath(out, e)), readdir(out))
    n == length(keep) || error("input $id: $out holds $n links, the selection $(length(keep)) files — another selection under the same run key? remove the folder")
    return out
end

# ---- the runs of other configs a config reads: [runs.<id>] -------------------------------------------------------------------------
#
# A derived thing another driver makes — the front ends' starts and index cameras (scripts/frontends.jl) — is a RESULT: a keyed run
# under results_root(), never an input under data_root(). A config that reads one names it as
#
#   [runs.<id>]
#   config = "configs/experiments/<config>.toml"   # the config of record whose run makes it
#   run = "<table>.<name>"                         # its run: the table and the entry of that config (e.g. "schoeps.zoom48_f10_siemens_w40")
#   path = "camera.json"                           # the file inside the run ("" for the run's directory)
#   stage = "detect"                               # the stage that requires it (DRIVER_STAGES)
#   provenance = "…"
#
# found by the key of that run (`RUN_RESOLVERS`: the driver's own keying), so a changed upstream config is another run, as an
# upstream refinement is for the chrome evaluations. An input path of a [pipeline] table names it as `run:<id>`.

"The keys of a `[runs.<id>]` table."
const RUN_INPUT_KEYS = ("config", "run", "path", "stage", "provenance")
"""Per driver (its `[experiment] name`), how a run of its config is found and made: `(cfg, run, config_path) -> (; product, make,
command[, gpu])` for `run` = "<table>.<name>": the run's keyed product, `make()` producing it (and, recursively, what it reads: the
upstream driver in a module of its own, `driver_run`), the command line that produces it, and whether making it needs a GPU. A driver whose runs other
configs read registers itself here (src/frontends_runs.jl)."""
const RUN_RESOLVERS = Dict{String, Function}()
"An input path naming a `[runs.<id>]`: `[lens:|grid:]run:<id>`."
const RUN_REF = r"^((?:lens|grid):)?run:([A-Za-z0-9_]+)$"

"""
    check_runs_table(cfg, path)

Every `[runs.<id>]` of `cfg` takes exactly `RUN_INPUT_KEYS`: a config of this project whose driver registered a resolver
(`RUN_RESOLVERS`), a run `<table>.<name>` of it, a relative path inside the run, a stage of `DRIVER_STAGES`; and no id is also an input.
"""
function check_runs_table(cfg::AbstractDict, path::AbstractString)
    for (id, t) in get(cfg, "runs", Dict{String, Any}())
        where = "$path [runs.$id]"
        t isa AbstractDict || throw(ArgumentError("$where must be a table"))
        VgtkCore.check_keys(t, RUN_INPUT_KEYS, where)
        foreach(k -> haskey(t, k) && t[k] isa AbstractString || throw(ArgumentError("$where lacks `$k` (a string)")), RUN_INPUT_KEYS)
        isfile(projectdir(t["config"])) || throw(ArgumentError("$where config: $(repr(t["config"])) is not a config of this project"))
        occursin(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+$", t["run"]) || throw(ArgumentError("$where run: \"<table>.<name>\", got $(repr(t["run"]))"))
        (isabspath(t["path"]) || ".." in splitpath(t["path"])) && throw(ArgumentError("$where path: relative to the run's directory"))
        t["stage"] in DRIVER_STAGES || throw(ArgumentError("$where stage: one of $(DRIVER_STAGES)"))
        haskey(get(cfg, "inputs", Dict()), id) && throw(ArgumentError("$where: $id is also an [inputs] id"))
    end
    return nothing
end

"The resolution of `[runs.<id>]` of `cfg` by its driver's resolver (`RUN_RESOLVERS`): `(; product, make, command)`."
function run_resolution(cfg::AbstractDict, id::AbstractString)
    t = cfg["runs"][id]
    up = TOML.parsefile(projectdir(t["config"]))
    name = up["experiment"]["name"]
    haskey(RUN_RESOLVERS, name) || throw(ArgumentError("[runs.$id]: no resolver for the runs of driver $name (include its runs file)"))
    return RUN_RESOLVERS[name](up, String(t["run"]), String(t["config"]))
end
"The step of `[runs.<id>]` of `cfg` (its keyed product), by its driver's resolver."
run_source(cfg::AbstractDict, id::AbstractString) = run_resolution(cfg, id).product
"The file of `[runs.<id>]` (its run's directory joined with `path`), absolute; not checked."
run_source_path(cfg::AbstractDict, id::AbstractString) =
    (t = cfg["runs"][id]; d = product_dir(run_source(cfg, id)); isempty(t["path"]) ? d : joinpath(d, t["path"]))

"""
    with_upstream(run_upstream, p) -> p

The consumer's side of produce_or_load (owner 2026-11-15; rc-texture-0111): when `p` is not produced, `run_upstream()`
produces it — or loads, by its key — through its own stage's producer (Eval's `produce`, DrWatson's `produce_or_load`), which
recurses the same way; the keys stay chained. A produced `p` is not touched.
"""
with_upstream(run_upstream, p) = (produced(p) || run_upstream(); p)

"""
    config_source(cfg, id) -> String

What `id` names, checked: an input ([`config_input`](@ref)), or a `[runs.<id>]` file — produced on demand when missing (owner
2026-11-15): the run's resolver makes it under its own key (`with_upstream`), and a run its maker did not produce is an
`ArgumentError` naming the run, its key, its directory and the command that produces it.
"""
function config_source(cfg::AbstractDict, id::AbstractString)
    haskey(get(cfg, "inputs", Dict()), id) && return config_input(cfg, id)
    t = cfg["runs"][id]
    r = run_resolution(cfg, id)
    produce_upstream(r.product, "[runs.$id]: the run $(t["run"]) of $(t["config"])"; r.make, r.command, gpu = get(r, :gpu, false),
                     note = "\n  provenance: $(t["provenance"])")
    return require_data(run_source_path(cfg, id); what = "$id (run $(t["run"]) of $(t["config"]))", provenance = t["provenance"])
end
"""
    produce_upstream(p, what; make, command, gpu = false, has = nothing, note = "") -> String

The directory of the upstream product `p` a consumer reads (`what` names it), PRODUCED ON DEMAND when missing (owner 2026-11-15: an
upstream is produced, never refused with "run its stage first"): `make()`, the upstream's own producer (its stage function, or its
driver in a module of its own: [`driver_run`](@ref)), produces it — or loads, by its key — under its own key (`with_upstream`),
recursing the same way. A GPU stage (`gpu`) is made only where the process holds a GPU ([`require_gpu`](@ref), `has` overriding the
device list): from a CPU job an `ArgumentError` naming `command`. A maker that did not produce `p` is an `ArgumentError` naming it,
its key, its directory and `command` (`note` appended). THE one path from a consumer to an upstream product (test/upstream_tests.jl
locks that no other refusal of one comes back).
"""
function produce_upstream(p, what::AbstractString; make, command::AbstractString, gpu::Bool = false, has = nothing, note::AbstractString = "")
    with_upstream(p) do
        gpu && (has === nothing ? require_gpu(what, command) : require_gpu(what, command; has))
        println(">> $what (key $(p.key)) is not produced: producing it ($command)"); flush(stdout)
        make()
    end
    produced(p) || throw(ArgumentError("$what (key $(p.key), $(product_dir(p))) is not produced, and producing it did not make it: " *
                                       "produce it with `$command`$note"))
    return product_dir(p)
end
"""
    driver_run(driver, config, stages; case = nothing) -> (; make, command)

The maker of a product of another driver (or of the consumer's own driver, a stage it has not run): `driver` (project-relative,
`scripts/<name>.jl`) on `config` through `stages` (`"prepare,detect"`: each stage producing or loading its own upstream by its key),
optionally restricted to one `case`, in a module of its own (Eval's `run_isolated`), and the command line that does the same.
"""
function driver_run(driver::AbstractString, config::AbstractString, stages::AbstractString; case = nothing)
    args = ["--config=$(isabspath(config) ? config : projectdir(config))", "--stage=$stages"]
    case === nothing || push!(args, "--case=$case")
    command = "julia --project=experiments experiments/$driver $(join(args, ' '))"
    return (; make = () -> (run_isolated(projectdir(driver); args); nothing), command)
end
"""
    require_gpu(what, command; has = gpu_listed())

A GPU-only upstream (`what`, e.g. a Schöps run: its camera_calibration needs CUDA) is made only where the process holds a GPU (`has`:
`nvidia-smi -L` lists one — a Slurm task's device cgroup lists exactly the GPUs it was given, none in a CPU job): an `ArgumentError`
naming it and the `command` to run in a GPU job otherwise — never run on the CPU (owner: GPU first).
"""
function require_gpu(what::AbstractString, command::AbstractString; has::Bool = gpu_listed())
    has || throw(ArgumentError("$what needs a GPU, and this process holds none (a CPU job): produce it in a GPU job with `$command`"))
    return nothing
end
"Whether `nvidia-smi -L` lists a GPU this process may use (read from the device list, never the environment: the paper project's data policy)."
gpu_listed() = Sys.which("nvidia-smi") !== nothing && occursin(r"^GPU \d+"m, try read(pipeline(`nvidia-smi -L`; stderr = devnull), String) catch; "" end)
"The `[runs]` ids a [pipeline] table names (`run:<id>` input paths), sorted."
table_runs(p::AbstractDict) = sort(unique([String(match(RUN_REF, v).captures[2]) for (tb, kv) in p if kv isa AbstractDict for (k, v) in kv
                                           if "$tb.$k" in PIPELINE_PATH_KEYS && v isa AbstractString && match(RUN_REF, v) !== nothing]))
"The `[runs]` ids of `cfg` as they key a run: `id => (config, run, path, key)` — the run's key, never its machine path."
runs_record(cfg::AbstractDict, ids) =
    Dict{String, Any}(id => Dict{String, Any}("config" => cfg["runs"][id]["config"], "run" => cfg["runs"][id]["run"], "path" => cfg["runs"][id]["path"],
                                              "key" => run_source(cfg, id).key) for id in ids)


# ---- the pipeline run -----------------------------------------------------------------------------------------------------------

"""
    run_pipeline(table, run_dir; source, label) -> String

RadiometricCamera's own entry point, `bin/refine.jl` (with `args`: `--stage=start`, `--start=<dir>`), on the `[pipeline]` table
`table` into `run_dir`: the table is written to
`run_dir/pipeline_input.toml` with `io.out = run_dir` and run in a child process (the package's environment, this process's Julia
and thread count) whose working directory is data_root(), the directory the table's relative input paths are relative to; an
absolute path (an earlier stage's result) is read where it is. The child's output goes to `run_dir/refine.log`. Returns that log.
"""
function run_pipeline(table::AbstractDict, run_dir::AbstractString; source::AbstractString, label::AbstractString, args = String[])
    assert_not_input(run_dir)
    mkpath(run_dir)
    t = deepcopy(Dict{String,Any}(table))
    get!(t, "io", Dict{String,Any}())["out"] = run_dir
    input = joinpath(run_dir, "pipeline_input.toml")
    open(input, "w") do io
        println(io, "# The [pipeline] table of $label from $source as bin/refine.jl read it: io.out set by the driver; relative input")
        println(io, "# paths are relative to data_root(), the working directory of the run.")
        TOML.print(io, t; sorted = true)
    end
    pkg = pkgdir(RadiometricCamera)
    threads = execution_threads()
    rep = REPEAT > 1 ? ["--repeat=$REPEAT"] : String[]
    cmd = `$(Base.julia_cmd()) --threads=$threads --project=$pkg $(joinpath(pkg, "bin", "refine.jl")) --config=$input $rep $args`
    log = joinpath(run_dir, "refine.log")
    println(">> bin/refine.jl: $label → ", run_dir, "   (threads ", threads, "; in data_root() = ", data_root(), "; log refine.log)")
    flush(stdout)
    ok = open(log, "w") do io
        success(pipeline(Cmd(cmd; dir = data_root()); stdout = io, stderr = io))
    end
    ok || error("bin/refine.jl failed on $label: see $log\n" * join(last(readlines(log), 15), "\n"))
    return log
end

"""
    last_model(run_dir, kind) -> String

The model a refinement ended with: `<kind>.json` (kind `lens` or `grid`), else its newest checkpoint `<kind>_it<N>.json` (a run the
convergence rule or the time budget stopped writes no final file); "" when there is neither.
"""
function last_model(run_dir::AbstractString, kind::AbstractString)
    f = joinpath(run_dir, "$kind.json")
    isfile(f) && return f
    its = checkpoint_iterations(run_dir, kind)
    return isempty(its) ? "" : joinpath(run_dir, "$(kind)_it$(last(its)).json")
end

"The checkpoint iterations a refinement wrote (`<kind>_it<N>.json`), sorted."
function checkpoint_iterations(run_dir::AbstractString, kind::AbstractString)
    its = Int[]
    isdir(run_dir) || return its
    for n in readdir(run_dir)
        m = match(Regex("^$(kind)_it(\\d+)\\.json\$"), n)
        m === nothing || push!(its, parse(Int, m.captures[1]))
    end
    return sort(its)
end

# ---- the RAW decoder -------------------------------------------------------------------------------------------------------------

"""
The RAW decodings, a `[prepare.<set>] mode` each, stated in the config of record so that its intention is explicit:
- `bin2`: 2 × 2 Bayer binning at half size (a COLMAP coordinate scales by exactly ½): the per-colour radiometric stages' image (the
  panorama vignetting sweeps), never a geometry image;
- `raw_green`: the GEOMETRY image (owner 2026-11-16: one geometric image for every detector) — VisualGeometryToolkitFeatures'
  bilinear green on the raw grid at full resolution (RadiometricCamera's `raw_green_image`: the image the chrome detect's
  `detect_sfop(m::RawMosaic)` reads, the same vocabulary as its `[detect] image`), each frame under its sensor model resolved from the
  set's characterisation ([`raw_sensor_models`](@ref)); one channel in the raw's COLMAP pixels (its frame map the identity). No sRGB
  copy.
The half-resolution green cells (Features' green-cell mode, the decode mode "green" until 2026-11-16) are retired for geometry, and no other
stage decodes them. `fullres` (a demosaic: interpolated, for display only) is not a mode.
Every mode is VisualGeometryToolkitFeatures' raw module (LibRaw_jll), [`decode_raw`](@ref): the one RAW path. Each decode's conversion
record states its frame and its map onto the raw (`frame_map`, [`decoded_frame`](@ref)), and a reader maps a position through it
([`prepared_frame_map`](@ref)).
"""
const RAW_MODES = ("bin2", "raw_green")

"""A `[prepare.<set>]` decoding checked: `mode` one of `RAW_MODES`, and `srgb = "off"` with `raw_green` (it has no colour to encode).
`photosites` is not decoded: the set is its RAW photographs, which a refinement compares at their green photosites on the raw grid
(RadiometricCamera reads a RAW photograph as one by its file: `load_photo`); see [`check_prepare_table`](@ref)."""
function check_raw_mode(mode::AbstractString, srgb::Bool, where::AbstractString)
    mode == "green" && throw(ArgumentError("$where: mode \"green\" (the green cells) is retired for geometry (owner 2026-11-16): mode \"raw_green\", the bilinear green on the raw grid"))
    mode in RAW_MODES || throw(ArgumentError("$where: mode $(repr(mode)) is not one of $(join(RAW_MODES, ", "))"))
    mode == "raw_green" && srgb && throw(ArgumentError("$where: mode \"raw_green\" writes no sRGB copy; set srgb = \"off\""))
    return nothing
end

"""
    photosites_set(t) / photosites_set(cfg, set) -> Bool

Whether the `[prepare.<set>]` table `t` (of `cfg`) is a photosites set: its RAW photographs read, not decoded (`mode = "photosites"`),
compared at their green photosites on the raw grid: what the prepare run makes (links to the RAW files, not decoded photographs). The
readers take the photographs' type from their files (RadiometricCamera's `observation_model` / `photo_set`), never from this test.
"""
photosites_set(t::AbstractDict) = t["mode"] == "photosites"
photosites_set(cfg::AbstractDict, set::AbstractString) = photosites_set(cfg["prepare"][set])

"""
    check_prepare_table(t, where)

A `[prepare.<set>]` table's rule (its keys are `check_prepare_tables`'): `mode` a decoding of `RAW_MODES` (`check_raw_mode`; it
states `gain`, a positive number or `"auto"`, and `srgb`, a switch: `prepare_srgb`) or `"photosites"`: the set read, not decoded, so `gain` and `srgb` are refused by name.
"""
function check_prepare_table(t::AbstractDict, where::AbstractString)
    if photosites_set(t)
        for k in ("gain", "srgb")
            haskey(t, k) && throw(ArgumentError("$where $k: a photosites set is read, not decoded (its RAW photographs compared at their green photosites on the raw grid): delete the key"))
        end
    else
        foreach(k -> VgtkCore.required_key(t, k, "$where (mode $(repr(t["mode"])) decodes)"), ("gain", "srgb"))
        t["gain"] == "auto" || (t["gain"] isa Real && t["gain"] > 0) || throw(ArgumentError("$where gain must be a positive number or \"auto\""))
        t["mode"] isa AbstractString || throw(ArgumentError("$where mode must be a string"))
        check_raw_mode(t["mode"], prepare_srgb(t, where), where)
    end
    return nothing
end

"""`[prepare.<set>]` `t`'s `srgb`, a switch (`"on"` | `"off"`: RadiometricCamera's `read_switch`, RC 0.11.4): whether a decoding also
writes its sRGB copy."""
prepare_srgb(t::AbstractDict, where::AbstractString) = RadiometricCamera.read_switch(t["srgb"], "$where srgb")

"The keys of a `[prepare.<set>]`: its raw input and its mode (required), and a decoding's gain and sRGB copy."
const PREPARE_KEYS = ("raw", "gain", "mode", "srgb")
"""
    check_prepare_tables(cfg, path)

Every `[prepare.<set>]` of `cfg` checked, the ONE check of the drivers' loaders: its keys of `PREPARE_KEYS` (`raw` and `mode`
required: `check_tables`) and its rule (`check_prepare_table`: a decoding states `gain` and `srgb`, a photosites set refuses them).
"""
function check_prepare_tables(cfg::AbstractDict, path::AbstractString)
    check_tables(cfg, path, "prepare", PREPARE_KEYS; required = ("raw", "mode"))
    foreach(((s, t),) -> check_prepare_table(t, "$path [prepare.$s]"), get(cfg, "prepare", Dict{String, Any}()))
    return nothing
end

"The raw photographs of `src`, sorted."
raw_files(src::AbstractString) =   # RadiometricCamera's RAW extensions (a session finds its RAW files by them)
    filter(f -> RadiometricCamera.is_raw_file(f) && isfile(f), RadiometricCamera.photo_files(src; join = true))

"""The auto gain's statistic of a mosaic: the 99.99th percentile of its green photosites' linear values (Core's `linearview`, the one
linearisation; the greens of Core's `photosite_colors`, 1 and 3)."""
green_quantile(m::VgtkCore.RawMosaic) = (c = VgtkCore.photosite_colors(m); quantile(VgtkCore.linearview(m)[(c .== 1) .| (c .== 3)], 0.9999))

"""
    raw_sensor_models(files, characterisation; what) -> Vector{SensorModel}

Each RAW photograph's sensor model, resolved by its EXIF from the sensor characterisation `characterisation` (a record's path)
through RadiometricCamera's ONE entry point (`photo_set` under the observation model of RAW photosites under that record,
`observation_model`: the resolution, and the refusals, of the calibration and of every prediction). The geometry image
(`raw_green`) is read under it — the chrome detect's and the front ends' decode alike. Precision over recall: a frame the
characterisation gives no noise law (another body, an unmatched ISO) is refused by name (`what`), never read at a guessed level.
"""
function raw_sensor_models(files::AbstractVector{<:AbstractString}, characterisation::AbstractString; what::AbstractString)
    om = RadiometricCamera.observation_model(files, characterisation)
    models = RadiometricCamera.photo_set(om, files; characterisation).models
    bad = [basename(f) for (f, m) in zip(files, models) if !(m.noise isa VgtkCore.PoissonGaussian)]
    isempty(bad) || throw(ArgumentError("$what: the sensor characterisation gives no noise law (gain) for $(length(bad)) photographs: $(join(bad, ", "))"))
    return models
end

"""
    raw_exif(md, f, root) -> Dict

The EXIF of RAW photograph `f` a prepare record keeps (its metadata `md`, Features' `load_raw_metadata` or a mosaic's): capture
time, exposure, f-number, focal length, ISO, body, and `raw` the file's path relative to `root`.
"""
raw_exif(md, f::AbstractString, root::AbstractString) =
    Dict{String, Any}("datetime_original" => md.datetime_original, "subsec" => something(md.subsec, ""), "exposure_s" => md.exposure_s,
                      "fnumber" => md.fnumber, "focal_mm" => md.focal_mm, "iso" => md.iso, "model" => md.model, "raw" => relpath(f, root))

"""
    decode_raw(src, root, set; gain, mode, srgb, characterisation = nothing) -> String

The raw photographs in `src` decoded in Julia by VisualGeometryToolkitFeatures' `load_raw` (LibRaw_jll), every mode of `RAW_MODES`:
`root/<set>_linear/*.png` (16-bit linear, camera raw colour, white
balance 1; `round(clamp(x, 0, 1) · 65535)`), `root/<set>_srgb/*.png` (bin2 unless `srgb = "off"`: IEC 61966-2-1 of the clamped linear
values, 8-bit), `root/<set>_linear/conversion.json` (with `frame_map`, the images' frame and its map onto the raw, [`frame_record`](@ref): one for
the set), `root/exif_<set>.json` and `root/capture_settings_<set>.json` (the capture's camera settings,
VisualGeometryToolkitIO's `vgio/capture_settings`: Features' `extract_capture_settings` of the same metadata, frame index the
file's position in `raw_files(src)`, body and lens as EXIF states them). `gain` is a number or `"auto"` (the set's 99.99th percentile of the
green photosites' linear values → 0.9, `green_quantile`). Each RAW is decoded once (Features' `load_raw_mosaic`; `"auto"` holds the
set's mosaics between its statistic and the decode), the photographs concurrently, one per thread, each its own LibRaw handle: the
files are the serial decode's, byte for byte. Returns `root/<set>_linear`. The front ends' images are GEOMETRIC (corner and star
positions): decoded under the file's black, no sensor model — exempt from the observation model (RC 0.11.1; every photograph a
calibrated camera is calibrated on or predicts goes through RadiometricCamera's `photo_set`).
"""
function decode_raw(src::AbstractString, root::AbstractString, set::AbstractString; gain, mode::AbstractString, srgb::Bool,
                    characterisation::Union{Nothing, AbstractString} = nothing)
    check_raw_mode(mode, srgb, "decode_raw ($set)")
    (mode == "raw_green") == (characterisation !== nothing) ||
        throw(ArgumentError("decode_raw ($set): $(mode == "raw_green" ? "mode \"raw_green\" reads each frame under its sensor model: name the characterisation" : "mode $(repr(mode)) reads no sensor characterisation")"))
    assert_not_input(root)
    files = raw_files(src)
    isempty(files) && throw(ArgumentError("decode_raw ($set): no raw files in $src"))
    # every photograph on its own thread: each load_raw_mosaic opens its own LibRaw handle (Features' LibRaw extension: libraw_init /
    # libraw_close per call), and each writes its own files; the records are assembled in the files' order afterwards
    mosaics = gain == "auto" ? Vector{Any}(undef, length(files)) : nothing   # decoded once: the statistic and the decode read them
    gain == "auto" && Threads.@threads for i in eachindex(files)
        mosaics[i] = VgtkFeatures.load_raw_mosaic(files[i])
    end
    g = gain == "auto" ? 0.9 / max(maximum(green_quantile, mosaics), 1e-6) : Float64(gain)
    gain == "auto" && println("auto gain $(round(g; digits = 3))")
    models = mode == "raw_green" ? raw_sensor_models(files, characterisation; what = "decode_raw ($set)") : nothing
    device = mode == "raw_green" ? RadiometricCamera.rc_backend("gpu"; inline = false) : nothing   # GPU first (a loud fallback)
    dirs = (linear = joinpath(root, "$(set)_linear"), srgb = srgb ? joinpath(root, "$(set)_srgb") : nothing)
    foreach(d -> d === nothing || mkpath(d), dirs)
    done = Vector{Any}(undef, length(files))
    Threads.@threads for i in eachindex(files)                # frame index: the file's position in raw_files(src)
        f = files[i]
        name = first(splitext(basename(f))) * ".png"
        m = mosaics === nothing ? VgtkFeatures.load_raw_mosaic(f) : mosaics[i]
        md = m.metadata
        ex = merge(raw_exif(md, f, root), Dict{String, Any}("raw_size" => [size(m.raw, 2), size(m.raw, 1)],
                               "cfa" => VgtkCore.cfa_string(m), "black" => collect(m.black),
                               "white" => m.white, "daylight_wb" => md.daylight_wb === nothing ? nothing : collect(md.daylight_wb)[1:3]))
        if mode == "raw_green"                                 # the geometry image: Features' bilinear green under the frame's model
            img = Base.invokelatest(RadiometricCamera.raw_green_image, Float32, m; sensor_model = models[i], device)
            done[i] = (; name, md, ex, frame = frame_record(m), rec = write_raw_green(img, g, name, dirs))
        else
            L = VgtkCore.LinearRaw(m; mode = raw_mode(mode), gain = g)
            done[i] = (; name, md, ex, frame = frame_record(L), rec = write_decoded(L, name, dirs))
        end
    end
    meta = Dict{String, Any}(d.name => d.rec for d in done); exif = Dict{String, Any}(d.name => d.ex for d in done)
    frames = Dict{Int, Any}(i => d.md for (i, d) in enumerate(done)); frame_maps = Dict{String, Any}[d.frame for d in done]
    write_conversion(dirs.linear, mode, frame_maps, meta)
    VgtkIO.write_record(joinpath(root, "exif_$set.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.exif), exif))
    # the capture's camera settings (Features' extract_capture_settings of the same metadata; body and lens as EXIF states them)
    cs = VgtkFeatures.extract_capture_settings(frames)
    VgtkIO.save(joinpath(root, "capture_settings_$set.json"), VgtkCore.Attributed(cs, (; identity_sources = (body = "exif", lens = "exif"))))
    println("camera settings ($set): ", RadiometricCamera.settings_line(cs))
    return dirs.linear
end

"""
    write_conversion(linear_dir, mode, frame_maps, photos) -> path

[`decode_raw`](@ref)'s conversion record `linear_dir/conversion.json`: the `mode`, a note on the values, the decoder, the images' frame
and its map onto the raw (`frame_map`: the one of `frame_maps`, each photograph's [`frame_record`](@ref); an error when they differ),
and per photograph its record (`photos`, [`write_decoded`](@ref)'s). [`prepared_frame_map`](@ref) reads the frame back.
"""
function write_conversion(linear_dir::AbstractString, mode::AbstractString, frame_maps, photos::AbstractDict)
    fm = unique(frame_maps)
    length(fm) == 1 || error("write_conversion ($linear_dir): the photographs' frame maps differ (or there is none): $fm")
    note = mode == "raw_green" ?
        "the bilinear green on the raw grid (VisualGeometryToolkitFeatures, the chrome detect's image; full resolution, the raw's COLMAP pixels: frame_map the identity), each frame under its sensor model: linear × gain, 16-bit; a pixel with no usable green in its 3×3 (a defect cluster) written 0 and counted" :
        "linear = (raw − black)/(white − black) × gain, camera raw colour, wb 1, in the cell grid's pixels (frame_map onto the raw); sRGB copy = IEC 61966-2-1 of the same data"
    path = joinpath(linear_dir, "conversion.json")
    VgtkIO.write_record(path, Dict("schema" => EXPERIMENT_RECORD_SCHEMAS.conversion, "mode" => mode, "note" => note, "decoder" => "VisualGeometryToolkitFeatures.load_raw (LibRaw_jll)",
                                   "frame_map" => only(fm), "photos" => photos))
    return path
end
"""[`write_conversion`](@ref)'s record of the decoded photographs in `linear_dir` (strict JSON, as written): the one reader of
`conversion.json` ([`prepared_frame_map`](@ref), the drivers' decode gain)."""
read_conversion(linear_dir::AbstractString) = VgtkIO.read_record(joinpath(linear_dir, "conversion.json"))

"The Core `LinearRaw` mode of a `[prepare.<set>]` cell-grid decoding (`check_raw_mode`'s): `\"bin2\"` is `:bin2`; `\"raw_green\"` is no
`LinearRaw` (the bilinear green on the raw grid, [`decode_raw`](@ref)); never `:demosaic` (display only)."
function raw_mode(mode::AbstractString)
    mode in RAW_MODES || throw(ArgumentError("raw_mode: mode $(repr(mode)) is not one of $(join(RAW_MODES, ", "))"))
    mode == "raw_green" && throw(ArgumentError("raw_mode: \"raw_green\" is the bilinear green on the raw grid, not a Core LinearRaw mode"))
    return Symbol(mode)
end

"""
    frame_record(L) -> Dict{String, Any}

The frame of a decoded image (Features' `LinearRaw`) and its map onto the visible raw's COLMAP pixels as plain data: Features'
`raw_provenance(L).frame_map` (`from`, `to`, `scale`, `translation`), as a conversion record and a run key hold it.
"""
frame_record(L) = Dict{String, Any}(String(k) => v for (k, v) in pairs(VgtkCore.raw_provenance(L).frame_map))
"""The frame of the raw-grid geometry image of mosaic `m` (`raw_green`): Core's `raw_frame_map(m)` onto the visible raw (the identity
as decoded, a window's offset), as plain data in the form of [`frame_record`](@ref)'s."""
frame_record(m::VgtkCore.RawMosaic) = (S = VgtkCore.raw_frame_map(m);
    Dict{String, Any}("from" => "PixelFrame{:colmap}", "to" => "PixelFrame{:colmap}", "scale" => Float64(VgtkCore.scale(S)),
                      "translation" => collect(Float64, VgtkCore.translation(S))))
"""The raw-grid frame of a `raw_green` decode of a whole mosaic: the identity onto the visible raw's COLMAP pixels (Core's
`raw_frame_map(m)` of an unwindowed mosaic)."""
const RAW_GRID_FRAME_MAP = VgtkCore.relabel(VgtkCore.PixelFrame{:colmap}, VgtkCore.PixelFrame{:colmap},
                                            VgtkCore.SimilarityMap(VgtkCore.CoordinateTransformations.Translation(StaticArrays.SVector(0.0, 0.0))))   # as Core builds raw_frame_map(m)


"""
    decoded_frame(mode) -> (; map, record) | nothing

The pixel frame of the images a `[prepare.<set>]` decoding of `mode` writes and its map onto the visible raw's COLMAP pixels, as
VisualGeometryToolkitFeatures states it: `map` its `raw_frame_map` (the Bayer cell grid of "bin2": `BayerCellFrame{:G}`; "raw_green" the raw grid itself, the identity
→ `PixelFrame{:colmap}`) and `record` the same map as plain data ([`frame_record`](@ref)), the form a conversion record and a run key
hold. The map depends on the mode alone, not on the photograph (the same for every Bayer phase and visible origin): Core's
mode-level `raw_frame_map(LinearRaw{mode})`. A mode not of `RAW_MODES` is refused (`raw_mode`).
"""
function decoded_frame(mode::AbstractString)
    if mode == "raw_green"                                 # the raw grid itself: the identity (an unwindowed mosaic's raw_frame_map)
        S = RAW_GRID_FRAME_MAP
        return (; map = S, record = Dict{String, Any}("from" => "PixelFrame{:colmap}", "to" => "PixelFrame{:colmap}",
                                                     "scale" => Float64(VgtkCore.scale(S)), "translation" => collect(Float64, VgtkCore.translation(S))))
    end
    T = VgtkCore.LinearRaw{raw_mode(mode)}                # the mode's frame, without a photograph (Core's mode-level map)
    return (; map = VgtkCore.raw_frame_map(T), record = frame_record(T))
end

"""
    prepared_frame_map(linear_dir) -> SimilarityMap

The map of a prepare run's decoded images (`linear_dir`, its `<set>_linear`) onto the raw's COLMAP pixels: the frame its conversion
record states (`frame_map`, written by [`decode_raw`](@ref)), which must be Features' frame of the record's mode
([`decoded_frame`](@ref)). Positions move through the map, covariances through its linear part (Core's `propagate`). A record that
states no frame — a decode made before the frame was recorded, Features 0.8's raw-grid "green" among them — or another frame is
refused (`ArgumentError`), never guessed: decode the set again (stage prepare).
"""
function prepared_frame_map(linear_dir::AbstractString)
    path = joinpath(linear_dir, "conversion.json")
    rec = read_conversion(linear_dir)
    mode = String(rec["mode"])
    f = decoded_frame(mode)                                              # a mode not of RAW_MODES (fullres) is refused
    got = get(rec, "frame_map", nothing)
    got === nothing &&
        throw(ArgumentError("$path states no frame_map: its images' frame is unknown (decoded before prepare recorded it); decode the set again (--stage=prepare --force)"))
    Dict{String, Any}(String(k) => v for (k, v) in pairs(got)) == f.record ||
        throw(ArgumentError("$path: frame_map $got is not Features' frame of a \"$mode\" decode $(f.record); decode the set again (--stage=prepare --force)"))
    return f.map
end

"""
    write_raw_green(img, gain, name, dirs) -> Dict

One `raw_green` photograph (`img`, RadiometricCamera's `raw_green_image`, linear) as `decode_raw`'s 16-bit grey PNG under
`dirs.linear` (× `gain`, clamped to [0, 1]; a NaN pixel — no usable green in its 3×3 — written 0) and its conversion record.
"""
function write_raw_green(img::AbstractMatrix{<:AbstractFloat}, gain::Real, name::AbstractString, dirs)
    nan = isnan.(img)
    x = ifelse.(nan, 0.0, Float64.(img) .* gain)
    g16 = round.(UInt16, clamp.(x, 0.0, 1.0) .* 65535)
    PNGFiles.save(joinpath(dirs.linear, name), reinterpret.(N0f16, g16))
    return Dict{String, Any}("size" => [size(img, 2), size(img, 1)], "gain" => Float64(gain), "max_linear" => maximum(g16) / 65535,
                             "mean_linear" => mean(g16) / 65535, "clipped_frac" => count(>(1), x) / length(x), "no_green_pixels" => count(nan))
end

"""
    write_decoded(L, name, dirs) -> Dict

One decoded photograph (Features' `LinearRaw`, the gain applied) as `decode_raw`'s files under `dirs` (`linear`, `srgb` or nothing)
and its conversion record.
"""
function write_decoded(L, name::AbstractString, dirs)
    u16(x) = round(UInt16, clamp(x, 0.0, 1.0) * 65535)
    rgb = L.image
    lin = clamp.(rgb, 0.0, 1.0)
    rgb16 = u16.(rgb)
    PNGFiles.save(joinpath(dirs.linear, name), [RGB{N0f16}(reinterpret(N0f16, rgb16[i, j, 1]), reinterpret(N0f16, rgb16[i, j, 2]),
                                                           reinterpret(N0f16, rgb16[i, j, 3])) for i in axes(rgb, 1), j in axes(rgb, 2)])
    if dirs.srgb !== nothing
        PNGFiles.save(joinpath(dirs.srgb, name), VgtkFeatures.srgb_image(N0f8, lin))     # Features' one display encoding
    end
    return Dict{String, Any}("size" => [size(rgb, 2), size(rgb, 1)], "gain" => L.gain, "max_linear" => maximum(lin), "mean_linear" => mean(lin),
                             "clipped_frac" => count(>(1), rgb) / length(rgb))
end

# ---- the runs of a config: prepare sets, init runs, cases -------------------------------------------------------------------------
#
# Each run is keyed by what determines it — never by a machine path: its own table, the inputs (tier and path; not their
# provenance text), and the keys of the runs it reads (`upstream`). A product's directory is `driver_product`'s: <results_root()>/<experiment>/<stage>/<key>.

"An input table as it keys a run: its tier and path, its content hash ([`INPUT_CONTENT_KEYS`](@ref)) and its selection
([`INPUT_SELECTION_KEYS`](@ref)) when it has them — never its provenance text."
input_record(t::AbstractDict) =
    merge(Dict{String,Any}("tier" => t["tier"], "path" => t["path"]),
          Dict{String,Any}(k => String(t[k]) for k in INPUT_CONTENT_KEYS if haskey(t, k)),
          Dict{String,Any}(k => String.(t[k]) for k in INPUT_SELECTION_KEYS if haskey(t, k)))

"The inputs of `cfg` as they key a run: `id => (tier, path[, frames | exclude])` ([`input_record`](@ref)), a `[runs]` id as `runs_record`'s; all the inputs, or `ids`."
inputs_record(cfg::AbstractDict, ids = keys(cfg["inputs"])) =
    merge(Dict{String,Any}(id => input_record(cfg["inputs"][id]) for id in ids if haskey(cfg["inputs"], id)),
          runs_record(cfg, [id for id in ids if !haskey(cfg["inputs"], id)]))

"""
    table_inputs(cfg, p) -> Vector{String}

The ids of the inputs of `cfg` that the [pipeline] table `p` reads: an input whose path is one of the table's input paths
(`table_paths`, the relative ones), contains one (a directory input) or lies inside one; sorted. They, not every input of the config, key a
run of that table.
"""
function table_inputs(cfg::AbstractDict, p::AbstractDict)
    paths = filter(!isabspath, table_paths(p))                      # relative: under data_root() (no data needed to key a run)
    inside(a, b) = a == b || startswith(a, b * "/")
    return sort([id for (id, t) in cfg["inputs"] if any(q -> inside(q, t["path"]) || inside(t["path"], q), paths)])
end

"The case-table keys a start reads (`start_table`): its photographs, its views and its [pipeline]."
const START_KEYS = ("images", "views", "pipeline")

"""
    start_settings(t) -> Dict

The settings of the table `t` that determine its start: `START_KEYS`, the [pipeline] table as its start reads it
(RadiometricCamera's `start_pipeline_table`, `start_options`' one definition: no refinement-only key, no output).
"""
function start_settings(t::AbstractDict)
    out = Dict{String,Any}(k => deepcopy(t[k]) for k in START_KEYS if haskey(t, k))
    haskey(out, "pipeline") && (out["pipeline"] = RadiometricCamera.start_pipeline_table(out["pipeline"]))
    return out
end

"""
    refine_settings(t) -> Dict

The settings of the case table `t` only its refinement reads: its [pipeline] table's `REFINE_ONLY_KEYS` (`split_refine_only`;
everything else keys its start, which the refinement reads by key).
"""
refine_settings(t::AbstractDict) = Dict{String, Any}("pipeline" => last(RadiometricCamera.split_refine_only(get(t, "pipeline", Dict{String, Any}()))))

"""
    prepare_product(cfg, set) -> Product

The step of `[prepare.<set>]` (the raw photographs of its `raw` input decoded): keyed by the set's table, its input and, for the
decodings Features makes on the Bayer cell grid (bin2, green), the frame of their images ([`decoded_frame`](@ref)): a decode of
another frame — Features 0.8's raw-grid "green" — is another step. Its output: the decoded set's conversion record, or a
photosites set's record.
"""
function prepare_product(cfg::AbstractDict, set::AbstractString)
    t = cfg["prepare"][set]
    raw = cfg["inputs"][t["raw"]]
    rec = Dict{String,Any}("experiment" => cfg["experiment"], "prepare" => Dict(set => t), "input" => input_record(raw))
    photosites_set(t) || (rec["frame_map"] = decoded_frame(t["mode"]).record)   # photosites: the raw grid, nothing decoded
    return driver_product("prepare", rec; output = photosites_set(t) ? "photosites.json" : joinpath("$(set)_linear", "conversion.json"),
                          threads = nothing)                               # one decode per photograph: thread-invariant
end

"The decoded linear photographs of `[prepare.<set>]`: `<its step>/<set>_linear`."
prepared_images(cfg::AbstractDict, set::AbstractString) =
    photosites_set(cfg, set) ? photosite_images(cfg, set) : joinpath(product_dir(prepare_product(cfg, set)), "$(set)_linear")
"""The RAW photographs of a `photosites` set: the prepare step's folder of links to them (`<id>_frames`: every photograph of the
input, or its selection's, `selected_raw_dir`), in the results like a decoded set's folder — a run key or a plan never reads the
data root for it."""
photosite_images(cfg::AbstractDict, set::AbstractString) = joinpath(product_dir(prepare_product(cfg, set)), "$(cfg["prepare"][set]["raw"])_frames")

"""
    start_table(cfg, name) -> Dict

The table whose start `name` is: `[init.<name>]` (a start other cases read, `init = "<name>"`), else `[cases.<name>]` (a case's own).
"""
function start_table(cfg::AbstractDict, name::AbstractString)
    haskey(get(cfg, "init", Dict()), name) && return cfg["init"][name]
    haskey(get(cfg, "cases", Dict()), name) && return cfg["cases"][name]
    throw(ArgumentError("no start $(repr(name)): neither [init.$name] nor [cases.$name]"))
end


# The upstream keys of a table: its prepare set's step.
function _upstream(cfg, t)
    up = Dict{String,Any}()
    haskey(t, "images") && (up["prepare"] = prepare_product(cfg, t["images"]).key)
    return up
end

"""A case `t`'s `start_from_init`, a switch (`"on"` | `"off"`: RadiometricCamera's `read_switch`, RC 0.11.4; `"off"` when absent):
whether its start reads its grid from its `[init.<name>]` start's run."""
start_from_init(t::AbstractDict, where::AbstractString) = RadiometricCamera.read_switch(get(t, "start_from_init", "off"), "$where start_from_init")

"""The `[init.<name>]` start the start of `name` reads its grid from: a case's `init = "<name>"` with `start_from_init` (its grid
the case's `io.init`), else `nothing` — the case's own start (`init = "self"` or none), or an `[init.<name>]` start itself."""
function _start_init(cfg, name)
    t = get(get(cfg, "cases", Dict()), name, nothing)
    (t === nothing || !start_from_init(t, "[cases.$name]")) && return nothing
    i = String(get(t, "init", ""))
    return isempty(i) || i == "self" ? nothing : i
end

"""
    start_product(cfg, name) -> Product

The start of `name` (`start_table`: a case's, or an `[init.<name>]`'s): bin/refine.jl --stage=start's start record. Keyed by its
settings (`start_settings`: never the refinement's), its views list, the inputs its table reads (`table_inputs`), its prepare step's
key, the `[init.<name>]` start it reads its grid from (`start_from_init`) and the start record's schema (RadiometricCamera's
`rc/refinement_start`): a new record format re-keys every start. Its output: the record's start.json.
"""
function start_product(cfg::AbstractDict, name::AbstractString)
    t = start_table(cfg, name)
    st = start_settings(t)                                         # once: the record, the inputs and the start's kind read the one table
    up = _upstream(cfg, t)
    i = _start_init(cfg, name)
    i === nothing || (up["init"] = start_product(cfg, i).key)
    p = resolve_pipeline(cfg, st; produce = false)                 # the table's inputs, keying only: nothing produced (the init's grid keys it by the init's key, not a path)
    rec = Dict{String,Any}("experiment" => cfg["experiment"], "start" => Dict(name => st),
                           "inputs" => inputs_record(cfg, table_inputs(cfg, p)), "upstream" => up,
                           "start_record" => RadiometricCamera.RECORD_SCHEMAS.refinement_start)
    ids = table_runs(get(st, "pipeline", Dict{String, Any}()))
    isempty(ids) || (rec["runs"] = runs_record(cfg, ids))
    lc = get(get(get(st, "pipeline", Dict{String, Any}()), "camera", Dict{String, Any}()), "lens_coverage", nothing)
    lc === nothing || lens_coverage_name(lc) === nothing || (up["image_region"] = lens_coverage_key(cfg, lc))   # the image region's step
    haskey(t, "views") && t["views"] isa AbstractString && (rec["views"] = cfg["views"][t["views"]])
    return driver_product("start", rec; output = RadiometricCamera.START_RECORD_FILE, experiment = variant_location(cfg, name))
end

"""
    refine_product(cfg, case) -> Product

The refinement of `[cases.<case>]`: keyed by its refine-only settings (`refine_settings`) and the key of its start step
(`start_product`, which holds every other setting, its inputs and its views) — a refine-only change re-keys the refinement alone and
loads the start. Its output: the refinement's parameters.json.
"""
function refine_product(cfg::AbstractDict, case::AbstractString)
    t = cfg["cases"][case]
    rec = Dict{String,Any}("experiment" => cfg["experiment"], "refine" => Dict(case => refine_settings(t)),
                           "upstream" => Dict{String, Any}("start" => start_product(cfg, case).key))
    return driver_product("refine", rec; output = "parameters.json", experiment = variant_location(cfg, case))
end

"""
    case_characterisation(cfg, case) -> Union{Nothing, String}

The location of the sensor characterisation the refinement case `[cases.<case>]` read its photographs under — its `[pipeline.camera]
sensor_characterisation`, the one input of `cfg` at that path, read through [`config_input`](@ref) (its sha256 checked) — or `nothing`
when it names none (empty: decoded images, or RAW photographs read without a record). What a consumer of the case's camera hands RadiometricCamera's `photo_set` as the
`characterisation` (a location only: the camera's record states which file it must be).
"""
function case_characterisation(cfg::AbstractDict, case::AbstractString)
    spec = get(get(get(cfg["cases"][case], "pipeline", Dict{String, Any}()), "camera", Dict{String, Any}()), "sensor_characterisation", "")
    isempty(spec) && return nothing
    ids = sort([id for (id, t) in cfg["inputs"] if t["path"] == spec])
    length(ids) == 1 || throw(ArgumentError("case $case: its sensor characterisation $spec is $(isempty(ids) ? "no input" : "the path of $(length(ids)) inputs ($(join(ids, ", ")))") of the config: one input states it"))
    return config_input(cfg, only(ids))
end

"""
    resolve_pipeline(cfg, t; init_dir = nothing, produce = true) -> Dict

The `[pipeline]` table of the init or case table `t` as bin/refine.jl reads it: `io.views` from its `views` (a `[views]` name or a
list), `io.images` from its `images` (the decoded photographs of that prepare set, an absolute path under results_root()) and,
with `init_dir` (an `[init.<name>]` start a case starts from, `start_from_init`), `io.init = "grid:<init_dir>/grid.json"`; an input path `run:<id>` names the file of `[runs.<id>]`
(checked and produced on demand, [`config_source`](@ref); with `produce = false` — a product's keying — its path alone,
[`run_source_path`](@ref), nothing produced).
"""
function resolve_pipeline(cfg::AbstractDict, t::AbstractDict; init_dir = nothing, produce::Bool = true)
    p = deepcopy(Dict{String,Any}(get(t, "pipeline", Dict{String,Any}())))
    io = get!(p, "io", Dict{String,Any}())
    if haskey(t, "views")
        io["views"] = t["views"] isa AbstractString ? cfg["views"][t["views"]] : t["views"]
    end
    haskey(t, "images") && (io["images"] = prepared_images(cfg, t["images"]))
    # RAW or decoded is the photographs' TYPE, read by RadiometricCamera from their files (observation_model / photo_set; owner ruling
    # 2026-11-13): no flag. A RAW photograph is measured at its green photosites, one value each: `io.channel` "G" unless the case
    # states it (RadiometricCamera refuses another channel for a RAW photograph). The set's photographs are RAW files exactly when its
    # prepare run links them (`photosites_set`: what prepare made, not a reader's switch).
    haskey(t, "images") && photosites_set(cfg, t["images"]) && get!(io, "channel", "G")
    if init_dir !== nothing                                     # a case's start from an [init.<name>] start (`start_from_init`)
        io["init"] = "grid:" * joinpath(init_dir, "grid.json")
    end
    # an input path `run:<id>` (a `lens:` / `grid:` prefix kept): that `[runs.<id>]`'s file, an absolute path under results_root(),
    # checked and produced on demand (`config_source`: never a bare SystemError at its first read)
    for tb in collect(keys(p)), k in (p[tb] isa AbstractDict ? collect(keys(p[tb])) : ())
        v = p[tb][k]
        "$tb.$k" in PIPELINE_PATH_KEYS && v isa AbstractString || continue
        m = match(RUN_REF, v)
        m === nothing || (p[tb][k] = something(m.captures[1], "") * (produce ? config_source(cfg, String(m.captures[2])) : run_source_path(cfg, String(m.captures[2]))))
    end
    # the set's lens coverage: "sensor", or its image region's record (an absolute path under results_root(), src/image_region.jl)
    cam = get(p, "camera", nothing)
    cam isa AbstractDict && haskey(cam, "lens_coverage") && (cam["lens_coverage"] = lens_coverage_path(cfg, cam["lens_coverage"]))
    return p
end

"""The [pipeline] keys whose values are input paths (relative to data_root(), or absolute: an earlier stage's result): the
package's declaration of its options, every input path among them (`RadiometricCamera.PIPELINE_PATH_KEYS`)."""
const PIPELINE_PATH_KEYS = RadiometricCamera.PIPELINE_PATH_KEYS
"The input paths the [pipeline] table `p` names, as written (a `lens:` / `grid:` prefix dropped), sorted."
table_paths(p::AbstractDict) =
    sort([replace(v, r"^(lens|grid):" => "") for (t, kv) in p if kv isa AbstractDict for (k, v) in kv
          if RadiometricCamera.names_input_path("$t.$k", v)])
"The inputs the [pipeline] table `p` names (relative paths: under data_root()), absolute; for a run's provenance."
pipeline_input_paths(p::AbstractDict) = sort([isabspath(q) ? q : joinpath(data_root(), q) for q in table_paths(p)])

"""
    run_inputs_through(cfg, t, stage) -> Vector{String}

Every input of the stages up to and including `stage` that the run of table `t` (a case or an [init.<name>]) reads, checked: the
config's [inputs] of those stages, and of its [runs] only those `t`'s [pipeline] names (`table_runs`) — a case resolves and
produces ONLY its own upstreams (owner 2026-11-15), each produced on demand ([`config_source`](@ref)).
"""
function run_inputs_through(cfg::AbstractDict, t::AbstractDict, stage::AbstractString)
    upto = DRIVER_STAGES[1:findfirst(==(stage), DRIVER_STAGES)]
    own = Set(table_runs(get(t, "pipeline", Dict{String, Any}())))
    runs = get(cfg, "runs", Dict{String, Any}())
    return String[config_source(cfg, id) for s in upto for id in stage_inputs(cfg, s) if !haskey(runs, id) || id in own]
end

"Every input of the stages up to and including `stage` (the run order of DRIVER_STAGES), checked by `require_data`."
function require_inputs_through(cfg::AbstractDict, stage::AbstractString)
    upto = DRIVER_STAGES[1:findfirst(==(stage), DRIVER_STAGES)]
    return reduce(vcat, [require_stage_inputs(cfg, s) for s in upto]; init = String[])
end

"""
    check_tables(cfg, path, table, allowed; required = ())

Every entry of `cfg[table]` (a table of tables) takes only `allowed` keys and has every `required` one; `ArgumentError` naming the
entry otherwise.
"""
function check_tables(cfg::AbstractDict, path::AbstractString, table::AbstractString, allowed; required = ())
    for (n, t) in get(cfg, table, Dict{String,Any}())
        where = "$path [$table.$n]"
        t isa AbstractDict || throw(ArgumentError("$where must be a table"))
        VgtkCore.check_keys(t, allowed, where)
        foreach(k -> VgtkCore.required_key(t, k, where), required)
    end
    return cfg
end

"""
    check_required_keys(t, keys, where) -> t

`t` a table of exactly `keys`, every one required (no code default): Core's `check_keys` (an unknown key refused, the allowed
listed) and its `required_key` of each (a missing one named). `where` names the file and the table.
"""
function check_required_keys(t, keys, where::AbstractString)
    t isa AbstractDict || throw(ArgumentError("$where must be a table"))
    VgtkCore.check_keys(t, keys, where)
    foreach(k -> VgtkCore.required_key(t, k, where), keys)
    return t
end

"""
    excluded_table(t, where) -> Dict{String, String}

A stage's `excluded` table `t` (photographs left out): file name = its reason, each a non-empty string; an `ArgumentError` naming
`where` otherwise.
"""
function excluded_table(t, where::AbstractString)
    t isa AbstractDict && all(v -> v isa AbstractString && !isempty(v), values(t)) ||
        throw(ArgumentError("$where excluded must be a table: file name = its reason"))
    return Dict{String, String}(String(k) => String(v) for (k, v) in t)
end

"""
    driver_numbers(cfg_path, driver, key) -> Reproduce.PaperNumbers

A driver's numbers.tex fragment, empty: VisualGeometryToolkitEval's `PaperNumbers` of the `driver` and its config `cfg_path`
(project-relative), the run key `key` of the run that writes it and the checkout's commit (`checkout_commit`). Filled with
`add_number!` (each name `number_macro_name`'s), written with `write_numbers`; scripts/numbers.jl gathers the fragments.
"""
driver_numbers(cfg_path::AbstractString, driver::AbstractString, key) =
    Reproduce.PaperNumbers(; driver, config = relpath(cfg_path, projectdir()), run_key = String(key), commit = Reproduce.checkout_commit())

# ---- the stages every driver shares ------------------------------------------------------------------------------------------------

"""With `--repeat=N` (a timing measurement) bin/refine.jl runs each refinement N times in its one process (its `REPEAT` lines: the
first repeat compiles, the later ones are warm); the files are the last repeat's, the same run. It keys nothing."""
const REPEAT = parse(Int, cli_option(ARGS, "repeat", "1"))
"""
    run_prepare(cfg, set; driver, config_path) -> dir

Stage prepare of `[prepare.<set>]`: its `raw` input decoded by [`decode_raw`](@ref) (Features' `load_raw`, every mode) into the set's
step (`prepare_product`), produced or loaded (`produce`). An input with a selection (`frames` | `exclude`) is decoded through its folder
of links ([`selected_raw_dir`](@ref)); the record names the input's folder.
"""
function run_prepare(cfg::AbstractDict, set::AbstractString; driver::AbstractString, config_path::AbstractString)
    t = cfg["prepare"][set]
    produce(prepare_product(cfg, set); product_records(driver, config_path)...) do dir
        raw = config_input(cfg, t["raw"])
        src = selected_raw_dir(cfg, t["raw"], dir)
        if photosites_set(t)                                       # read, not decoded: links to the RAW photographs and their record
            out = photosite_images(cfg, set)
            if src != out                                          # the whole input (no selection): every photograph linked, as a selection's
                assert_not_input(out); mkpath(out)
                for f in raw_files(src)
                    d = joinpath(out, basename(f)); islink(d) || symlink(f, d)
                end
            end
            # the photograph list and each one's EXIF (its headers alone, Features' load_raw_metadata): the mosaics are read by their consumer
            files = raw_files(out)
            VgtkIO.write_record(joinpath(dir, "photosites.json"), Dict("schema" => EXPERIMENT_RECORD_SCHEMAS.photosites, "mode" => "photosites", "raw" => src,
                                                                       "photos" => basename.(files),
                                                                       "exif" => Dict(basename(f) => raw_exif(VgtkFeatures.load_raw_metadata(f), f, dir) for f in files)))
        else
            decode_raw(src, dir, set; gain = t["gain"], mode = t["mode"], srgb = prepare_srgb(t, "[prepare.$set]"))
        end
        Produced(; inputs = [raw])
    end
end

"""
    require_prepared(cfg, t, what; driver, config_path)

The decode of the set a start or case table names (`images = "<set>"`, a `[prepare.<set>]` table), produced on demand
(`produce_upstream`: its stage prepare, [`run_prepare`](@ref)) before bin/refine.jl reads its photographs. A table without `images`
reads no decode.
"""
function require_prepared(cfg::AbstractDict, t::AbstractDict, what::AbstractString; driver::AbstractString, config_path::AbstractString)
    haskey(t, "images") || return nothing
    set = t["images"]
    produce_upstream(prepare_product(cfg, set), "$what: the decode of its set $set";
                     make = () -> run_prepare(cfg, set; driver, config_path), command = driver_run(driver, config_path, "prepare").command)
    return nothing
end
"""The start an `[init.<name>]`-starting table reads (`i`, or `nothing`), produced on demand (its stage start, [`run_start`](@ref);
a GPU stage when its [pipeline] runs on one)."""
init_start_dir(cfg::AbstractDict, i, what::AbstractString; driver::AbstractString, config_path::AbstractString) =
    i === nothing ? nothing :
    produce_upstream(start_product(cfg, i), "the start $i that $what starts from"; make = () -> run_start(cfg, i; driver, config_path),
                     command = driver_run(driver, config_path, "prepare,start").command, gpu = runs_on_gpu(start_table(cfg, i)))

"""
    run_start(cfg, name; driver, config_path, after = nothing) -> dir

Stage start of `name` (`start_table`): bin/refine.jl --stage=start on its settings (`start_settings`: what keys it) into its start
step (`start_product`), produced or loaded; a case starting from an `[init.<name>]` start (`start_from_init`) reads that start's grid,
which must be produced first. `after(dir)`, when given, writes what the driver derives from the start record before its records.
"""
function run_start(cfg::AbstractDict, name::AbstractString; driver::AbstractString, config_path::AbstractString, after = nothing)
    s = start_product(cfg, name)
    t = start_table(cfg, name)
    produce(s; product_records(driver, config_path)...) do dir
        require_prepared(cfg, t, "start $name"; driver, config_path)
        require_lens_coverage(cfg, t, "start $name")
        idir = init_start_dir(cfg, _start_init(cfg, name), name; driver, config_path)
        p = resolve_pipeline(cfg, start_settings(t); init_dir = idir)
        inputs = [run_inputs_through(cfg, t, "start"); pipeline_input_paths(p)]
        run_pipeline(p, dir; source = relpath(config_path, projectdir()), label = "start $name (product $(s.key))", args = ["--stage=start"])
        after === nothing || after(dir)
        Produced(; inputs)
    end
end
"The driver of the refinements' config of record (configs/experiments/radiometric_refinement.toml)."
const REFINEMENT_DRIVER = "scripts/radiometric_refinement.jl"
"""
    refinement_upstream(rc, rc_path, case, stage, what) -> dir

The start or refine product (`stage`) of `[cases.<case>]` of the refinements' config `rc` (at `rc_path`) another driver (`what`)
reads, produced on demand (`produce_upstream`): its driver on that case through the stage (`driver_run`: prepare, start[, refine]),
a GPU's when its [pipeline] runs on one (`runs_on_gpu`).
"""
function refinement_upstream(rc::AbstractDict, rc_path::AbstractString, case::AbstractString, stage::AbstractString, what::AbstractString)
    p = stage == "start" ? start_product(rc, case) : stage == "refine" ? refine_product(rc, case) :
        throw(ArgumentError("refinement_upstream: stage start or refine, got $(repr(stage))"))
    return produce_upstream(p, what; driver_run(REFINEMENT_DRIVER, rc_path, stage == "start" ? "prepare,start" : "prepare,start,refine"; case)...,
                            gpu = runs_on_gpu(rc["cases"][case]))
end
"""
    run_refine(cfg, case; driver, config_path) -> dir

Stage refine of `[cases.<case>]` on the case's resolved [pipeline] table into its refine product (`refine_product`), produced or
loaded: with its start produced, bin/refine.jl --start=<the start product> (bin/refine.jl refuses a start of other start options);
with neither, ONE bin/refine.jl run (`--start-out=<the start product>`) writes the start record into the start product
(`start_product`, the same key) and refines from it in the same process — one Julia child, the photographs read once.
"""
function run_refine(cfg::AbstractDict, case::AbstractString; driver::AbstractString, config_path::AbstractString)
    s = refine_product(cfg, case); st = start_product(cfg, case)
    R = product_records(driver, config_path); source = relpath(config_path, projectdir())
    produce(s; R...) do dir
        require_prepared(cfg, cfg["cases"][case], "case $case"; driver, config_path)
        require_lens_coverage(cfg, cfg["cases"][case], "case $case")
        idir = init_start_dir(cfg, _start_init(cfg, case), case; driver, config_path)
        p = resolve_pipeline(cfg, cfg["cases"][case]; init_dir = idir)
        label = "case $case (product $(s.key))"
        paths = pipeline_input_paths(p)                              # the table's inputs, resolved once (the start reads the same)
        sdir = if start_reused(st)                                   # the start loaded: the refinement's own child
            run_pipeline(p, dir; source, label, args = ["--start=$(product_dir(st))"])
            product_dir(st)
        else                                                         # start and refinement in one child, the start in its own product
            produce(st; R..., force = true) do sd
                run_pipeline(p, dir; source, label, args = ["--start-out=$sd"])
                Produced(; inputs = [run_inputs_through(cfg, cfg["cases"][case], "start"); paths])
            end
        end
        Produced(; inputs = [run_inputs_through(cfg, cfg["cases"][case], "refine"); paths; sdir])
    end
end

"""
    start_with_refine(cfg, case; argv = ARGS) -> Bool

Whether the start of `[cases.<case>]` is left to its refinement's one child (`run_refine`): this run's refine stage runs (`--stage`
lists it, or none is given) and the case's refinement is to be produced (not produced, or `--force`). The start stage then runs
nothing for the case; two children only when the start is produced and the refinement is not.
"""
start_with_refine(cfg::AbstractDict, case::AbstractString; argv = ARGS) =
    stage_runs("refine"; argv) && ("--force" in argv || !produced(refine_product(cfg, case)))

"Whether this run runs stage `name`: `--stage` lists it, or none is given (Eval's `run_stages` rule)."
stage_runs(name::AbstractString; argv = ARGS) = (s = cli_option(argv, "stage", ""); isempty(s) || name in split(s, ','))

"""
    start_reused(st; argv = ARGS) -> Bool

Whether a refinement reads its start product `st` as produced: it is produced, and this run does not force the start (`--force` with
the start stage among those it runs) — `--stage=start,refine --force` produces both again, `--stage=refine --force` the refinement
alone, from its start.
"""
start_reused(st::Product; argv = ARGS) = produced(st) && !("--force" in argv && stage_runs("start"; argv))

"""
    produce_figures(f, product, assets; driver, config_path, inputs = String[]) -> dir

A figures product: `f(dir)` writes the `assets` (paths relative to DrWatson's `plotsdir()`, a final output) — or writes an asset
straight into `dir/assets/<its path under plotsdir()>` — and the product's output into `dir`; the assets are kept in the product
(`dir/assets/…`) and every one of them is published to `plotsdir()` whether the product was produced now or loaded, so the figures
of the plotsdir are always this product's.
"""
function produce_figures(f, p, assets; driver::AbstractString, config_path::AbstractString, inputs = String[])
    dir = produce(p; product_records(driver, config_path)..., inputs) do d
        f(d)
        for a in assets
            mkpath(dirname(joinpath(d, "assets", a))); cp(plotsdir(a), joinpath(d, "assets", a); force = true)
        end
        nothing
    end
    kept = joinpath(dir, "assets")
    for (root, _, files) in walkdir(kept), file in files
        a = relpath(joinpath(root, file), kept)
        mkpath(dirname(plotsdir(a))); cp(joinpath(kept, a), plotsdir(a); force = true)
    end
    return dir
end

"""
    runs_on_gpu(t) -> Bool

Whether bin/refine.jl runs the `[pipeline]` table of the init, case or calibration table `t` on the GPU: its `compute.backend`
tag, RadiometricCamera's default (`ComputeOptions`) when the table names none.
"""
function runs_on_gpu(t::AbstractDict)
    tag = get(get(get(t, "pipeline", Dict{String,Any}()), "compute", Dict{String,Any}()), "backend", nothing)
    backend = tag === nothing ? RadiometricCamera.ComputeOptions().backend : RadiometricCamera.BACKEND_TAGS[tag]
    return backend === RadiometricCamera.GPUCompute
end

"""
    gpu_stages(stage => tables, ...) -> Vector{Symbol}

A driver's declaration of its GPU stages (`const GPU_STAGES`, read by scripts/regenerate_paper.jl): each `stage` paired with the
init, case or calibration tables whose `[pipeline]` it hands bin/refine.jl, kept when any of them `runs_on_gpu`. A stage that runs
no pipeline is not named.
"""
gpu_stages(stages::Pair...) = Symbol[s for (s, tables) in stages if any(runs_on_gpu, tables)]

"The `[figures.*]` assets of `cfg` (paths relative to `plotsdir()`)."
figure_assets(cfg::AbstractDict) = sort([String(f["asset"]) for f in values(cfg["figures"])])

# the runs of the front ends, which other configs read through their [runs] tables
include("frontends_runs.jl")
# the lens coverage of a capture set: one keyed step per set's image region, which every consumer reads (`lens_coverage`)
include("image_region.jl")
# the printed boards' edge distance maps: one keyed step per map, which every edge.sdf / distance_map consumer reads through [runs]
include("distance_maps.jl")
