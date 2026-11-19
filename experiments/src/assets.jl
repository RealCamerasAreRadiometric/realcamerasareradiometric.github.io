# =============================================================================
# src/assets.jl — this paper's glue around VisualGeometryToolkitEval's paper-asset
# stages (Reproduce: verify_assets, deploy_assets, gather_numbers,
# uncited_macros; Experiments: run_isolated): the figure configs of record
# (configs/figures/<asset>.toml) and the configs of scripts/numbers.jl and
# scripts/regenerate_paper.jl, where the paper reads its figures and numbers,
# the driver of an experiment config and its numbers fragment, and the GPU gate
# of a driver's stages.
#
# A driver renders into plotsdir() (never into the paper); Eval's
# `deploy_assets` copies a render into ../paper/fig — and the gathered
# numbers.tex into ../paper/latex — only when that changes nothing or the
# config of record's [deploy] replace_submitted = "on" signs the difference off.
#
# Included after the prelude (src/experiments.jl) by scripts/regenerate_paper.jl,
# scripts/numbers.jl and the suite.
# =============================================================================
using TOML: TOML
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitEval.Experiments: run_isolated
using VisualGeometryToolkitEval.Reproduce: deploy_assets
using RadiometricCamera: read_switch

"The directory the paper reads its generated figures from."
paper_figdir() = projectdir("..", "paper", "fig")

"The paper's generated number macros (input from main.tex)."
paper_numbers_path() = projectdir("..", "paper", "latex", "numbers.tex")

"The directory the paper reads its generated LaTeX tables from (and numbers.tex)."
paper_latexdir() = projectdir("..", "paper", "latex")

"The paper's compile directory (the manuscript ../paper is built there, main.tex at its top): where a manuscript path such as
`sec/experiments` resolves (Reproduce `paper_asset_graph` `search_dirs`, relative to the project)."
const PAPER_SEARCH_DIRS = ["../paper"]

# ---- the figure configs of record --------------------------------------------------------------------------------------------------

"The tables of a figure config and the keys of each; every key is required (no default)."
const FIGURE_CONFIG_KEYS = Dict(
    "experiment" => ("name", "generator"),   # the asset's name; the driver that renders it (project-relative)
    "runs" => ("experiment",),               # the config of record of that driver's runs (project-relative)
    "render" => ("asset",),                  # "figures.<name>": the [figures.<name>] table of that config that draws a PNG (../paper/fig);
                                             # "tables.<name>": its [tables.<name>] table that writes table_<name>.tex (../paper/latex)
    "deploy" => ("replace_submitted",))      # true only once the author signed off a differing regeneration

function _check_tables(cfg::AbstractDict, keys_of::AbstractDict, path::AbstractString)
    VgtkCore.check_keys(cfg, collect(keys(keys_of)), path)
    for (table, ks) in keys_of
        t = get(cfg, table, nothing)
        t isa AbstractDict || throw(ArgumentError("$path lacks the table [$table]"))
        VgtkCore.check_keys(t, ks, "$path [$table]")
        for k in ks
            haskey(t, k) || throw(ArgumentError("$path [$table] lacks `$k`"))
        end
    end
    return cfg
end

"""
    load_config_tables(path, keys_of) -> Dict

A config of record of this file's scripts: exactly the tables of `keys_of` (`table => keys`), each with exactly its keys, every
one present (no default). An unknown or missing table or key is an `ArgumentError`; every value of a key `experiment`,
`experiments` or `numbers` (a project-relative config) must exist, and a `[deploy] replace_submitted` is a switch ([`replace_submitted`](@ref)).
"""
function load_config_tables(path::AbstractString, keys_of::AbstractDict)
    cfg = _check_tables(TOML.parsefile(path), keys_of, path)
    for (table, t) in cfg, key in ("experiment", "experiments", "numbers")
        haskey(t, key) && table != "experiment" || continue
        for c in (t[key] isa AbstractVector ? t[key] : [t[key]])
            isfile(projectdir(c)) || throw(ArgumentError("$path [$table] $key: $c does not exist"))
        end
    end
    haskey(cfg, "deploy") && replace_submitted(cfg, path)
    return cfg
end

"""
    replace_submitted(cfg, where = "") -> Bool

A config's `[deploy] replace_submitted`, a switch (`"on"` | `"off"`: RadiometricCamera's `read_switch`, RC 0.11.4): `"on"` only once
the author signed off a differing regeneration.
"""
replace_submitted(cfg::AbstractDict, where::AbstractString = "") =
    read_switch(cfg["deploy"]["replace_submitted"], strip("$where [deploy] replace_submitted"))

"The tables and keys of configs/figures/numbers.toml (scripts/numbers.jl): its [runs] experiments are THE list of drivers."
const NUMBERS_CONFIG_KEYS = Dict("experiment" => ("name", "generator"), "runs" => ("experiments",), "deploy" => ("replace_submitted",))
"The tables and keys of configs/figures/regenerate_paper.toml (scripts/regenerate_paper.jl)."
const REGENERATE_CONFIG_KEYS = Dict("experiment" => ("name", "generator"), "runs" => ("numbers",), "gpu" => ("min_free_gib",))

"""
    driver_configs(regenerate_cfg) -> Vector{String}

The experiment configs whose drivers a regeneration runs, in order: the [runs] experiments of the numbers config it names (one
list, read where the numbers are gathered from).
"""
driver_configs(cfg::AbstractDict) =
    String.(load_config_tables(projectdir(cfg["runs"]["numbers"]), NUMBERS_CONFIG_KEYS)["runs"]["experiments"])

"""
    load_figure_config(path) -> Dict

A figure config of record (configs/figures/<asset>.toml): exactly the tables and keys of `FIGURE_CONFIG_KEYS`, every one present;
the generator and the runs' config exist and `replace_submitted` is a switch. An unknown or missing key is an `ArgumentError`.
"""
function load_figure_config(path::AbstractString)
    cfg = load_config_tables(path, FIGURE_CONFIG_KEYS)
    isfile(projectdir(cfg["experiment"]["generator"])) ||
        throw(ArgumentError("$path [experiment] generator: $(cfg["experiment"]["generator"]) does not exist"))
    return cfg
end

"Every figure config of record: the configs under `dir` with a [render] table, sorted."
figure_configs(dir::AbstractString = projectdir("configs", "figures")) =
    sort([joinpath(dir, f) for f in readdir(dir) if endswith(f, ".toml") && haskey(TOML.parsefile(joinpath(dir, f)), "render")])

"""
    asset_kind(cfg) -> (kind, name)

The kind of the asset a figure config deploys and the table of its runs' config that makes it: `[render] asset = "figures.<name>"`
→ `(:figure, name)` (a PNG), `"tables.<name>"` → `(:table, name)` (a LaTeX table). Anything else is an `ArgumentError`.
"""
function asset_kind(cfg::AbstractDict)
    m = match(r"^(figures|tables)\.([A-Za-z0-9_]+)$", String(cfg["render"]["asset"]))
    m === nothing && throw(ArgumentError("[render] asset = $(repr(cfg["render"]["asset"])): \"figures.<name>\" or \"tables.<name>\" of the runs' config"))
    return (m.captures[1] == "figures" ? :figure : :table), String(m.captures[2])
end

"""
    figure_asset(cfg) -> String

The file name of the asset a figure config deploys: for a figure the `asset` of the [figures.<name>] table of its runs' config
(named once, there, where the driver reads it); for a table `table_<name>.tex`, the file `write_summary_tables` writes for its
[tables.<name>] table.
"""
function figure_asset(cfg::AbstractDict)
    src = cfg["runs"]["experiment"]; kind, name = asset_kind(cfg)
    tables = get(TOML.parsefile(projectdir(src)), kind === :figure ? "figures" : "tables", Dict{String,Any}())
    t = get(tables, name, nothing)
    t === nothing && throw(ArgumentError("$src has no [$(kind === :figure ? "figures" : "tables").$name] table"))
    return kind === :figure ? String(t["asset"]) : "table_$name.tex"
end

"""
    asset_dirs(cfg) -> (outdir, paperdir)

Where a figure config's asset is rendered and where the paper reads it: a figure in `plotsdir()` → `paper_figdir()`, a table in
`plotsdir("latex")` → `paper_latexdir()`.
"""
asset_dirs(cfg::AbstractDict) = asset_kind(cfg)[1] === :figure ? (plotsdir(), paper_figdir()) : (plotsdir("latex"), paper_latexdir())

"""
    deploy_figures(outdir, figdir, figures; latex_outdir = joinpath(outdir, "latex"), latexdir = nothing) -> Vector{String}

Deploy the render of every figure config of record in `figures` (`(cfg, path)` pairs): the figures from `outdir` into `figdir`, the
tables from `latex_outdir` into `latexdir` (required when `figures` holds a table config), each kind in ONE call of
VisualGeometryToolkitEval's `deploy_assets` (one verify report per kind, every row kept), each asset signed off by its own config's
[deploy] replace_submitted. A refusal names the configs to sign off. Returns the assets copied.
"""
function deploy_figures(outdir::AbstractString, figdir::AbstractString, figures; latex_outdir::AbstractString = joinpath(outdir, "latex"),
                        latexdir::Union{Nothing, AbstractString} = nothing)
    copied = String[]
    for (kind, out, dst) in ((:figure, outdir, figdir), (:table, latex_outdir, latexdir))
        group = [f for f in figures if asset_kind(f.cfg)[1] === kind]
        isempty(group) && continue
        dst === nothing && throw(ArgumentError("deploy_figures: table configs ($(join([relpath(f.path, projectdir()) for f in group], ", "))) need `latexdir`"))
        assets = [figure_asset(f.cfg) for f in group]
        signed = [a for (f, a) in zip(group, assets) if replace_submitted(f.cfg, relpath(f.path, projectdir()))]
        try
            append!(copied, deploy_assets(out, dst, assets; replace_submitted = signed))
        catch err
            err isa ErrorException || rethrow()
            configs = [relpath(f.path, projectdir()) for (f, a) in zip(group, assets) if occursin("  $a:", err.msg)]
            error(err.msg * "\nThe figure configs to sign off: " * join(configs, ", "))
        end
    end
    return copied
end

# ---- a driver in a module of its own (Eval's run_isolated) ------------------------------------------------------------------------------------------

"""
    driver_script(experiment_config) -> path

The driver of an experiment config of record: `scripts/<name>.jl` for its `[experiment] name` (a driver and its configs share the
name: scripts/focus_surfaces.jl reads configs/experiments/focus_surfaces.toml, scripts/texture_recovery.jl each
configs/experiments/texture_recovery_<set>.toml).
"""
function driver_script(config::AbstractString)
    name = TOML.parsefile(config)["experiment"]["name"]
    path = projectdir("scripts", "$name.jl")
    isfile(path) || throw(ArgumentError("$config: no driver scripts/$name.jl for [experiment] name = $(repr(name))"))
    return path
end

"""
    numbers_fragment(experiment_config) -> path

The numbers.tex fragment the figures stage of `experiment_config`'s driver writes: the driver loaded in a module of its own with
`--plan` (it runs nothing) and asked for its keyed figures run (`figures_product()`).
"""
function numbers_fragment(config::AbstractString)
    m = run_isolated(driver_script(config); args = ["--plan", "--config=$config"])
    return joinpath(product_dir(Base.invokelatest(Base.invokelatest(getfield, m, :figures_product))), "numbers.tex")
end

"""
    host_stage(stage; driver, gpu_stages, gpu) -> Stage

A stage of an isolated driver (`run_isolated`), to run from the caller's older world: its `items`, `product` and `run` (or its `check`)
through `Base.invokelatest`. A producing stage in `gpu_stages` (the driver's declared `GPU_STAGES`) without `gpu` is not run: a
pending product is an error naming `--gpu`; a produced one is reused.
"""
function host_stage(s::Stage; driver::AbstractString, gpu_stages, gpu::Bool)
    s.items === nothing && return Stage(s.name, s.desc; check = () -> Base.invokelatest(s.run))
    items = () -> Base.invokelatest(s.items)
    product = x -> Base.invokelatest(s.product, x)
    (gpu || !(s.name in gpu_stages)) && return Stage(s.name, s.desc; items, product, run = x -> Base.invokelatest(s.run, x))
    return Stage(s.name, s.desc * " [GPU: not run without --gpu]"; items, product,
                 run = x -> (p = product(x); produced(p) || error("stage $(s.name) of $driver is pending ($(product_dir(p))) and runs on the GPU; rerun with --gpu"); nothing))
end

"""
    driver_gpu_stages(m, driver) -> Vector{Symbol}

The GPU stages the isolated driver module `m` declares (its `GPU_STAGES`, from the compute backend of the [pipeline] tables its
stages hand bin/refine.jl: src/drivers.jl `gpu_stages`); a driver that declares none is an error.
"""
function driver_gpu_stages(m::Module, driver::AbstractString)
    Base.invokelatest(isdefined, m, :GPU_STAGES) || error("$driver declares no GPU_STAGES (src/drivers.jl gpu_stages)")
    return collect(Symbol, Base.invokelatest(getfield, m, :GPU_STAGES))
end

"""
    gpu_free_mib(device = 0) -> Float64

The free memory of GPU `device` in MiB, as nvidia-smi reports it (`-i device`: nvidia-smi's index, which is CUDA's device 0 when
no device list restricts the process). An error when nvidia-smi is missing or fails.
"""
function gpu_free_mib(device::Integer = 0)
    Sys.which("nvidia-smi") === nothing && error("--gpu: nvidia-smi is not on the PATH; no GPU memory to check")
    out = readchomp(`nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $device`)
    return parse(Float64, strip(first(split(out, '\n'))))
end

"""
    require_gpu_memory(min_free_gib; device = 0, free_mib = gpu_free_mib(device)) -> free_mib

Refuse a `--gpu` regeneration on a device without `min_free_gib` GiB free (the config of record's [gpu] min_free_gib): an error
stating the free and the required memory. Returns the free MiB.
"""
function require_gpu_memory(min_free_gib::Real; device::Integer = 0, free_mib::Real = gpu_free_mib(device))
    free_mib / 1024 >= min_free_gib ||
        error("--gpu: GPU $device has $(round(free_mib / 1024; digits = 1)) GiB free; the GPU stages need at least $min_free_gib GiB. " *
              "Free the device and rerun.")
    return free_mib
end
