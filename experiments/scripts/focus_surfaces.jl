#!/usr/bin/env julia
# reproduce: writes ../paper/fig/focus-surfaces.tex
# reproduce: writes plots/focus-surfaces.png (plotsdir(); an earlier rendering of the same surfaces, not in the paper)
# reproduce: reads configs/experiments/focus_surfaces.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/focus_surfaces.jl — the paper's focus-surface figure: the focus surfaces two refinements learned as surfaces of in-focus
# points in camera space (fig:focus-surfaces — the fisheye at f/4 a sphere about the camera, the Tamron 20 mm at f/8 a plane; their
# radiometric_camera.json). Config of record: configs/experiments/focus_surfaces.toml. The
# chrome-board figures and tables (fig:color-wheels, tab:refinement-chrome, tab:defocus, chrome_all) are the chrome evaluation's
# own stages (scripts/chrome_evaluation_paper.jl, configs/experiments/chrome_evaluation_paper.toml).
#
# STAGES (each a product keyed by its config record and the keys of the products it reads; datadir("exp_pro", <experiment>, <stage>, <key>))
#   panels    the paper's figure (fig:focus-surfaces) as PGFPlots: the [panels] cases' in-focus points on the pixel lattice (dump/),
#             the figure's tables from them (tables/) and its files (src/paper_figures/focus_surfaces.jl: both parts) →
#             plotsdir(<asset_dir>.tex, <asset_dir>/<lens>.tex, <asset_dir>-caption.tex), numbers.tex (empty: the figure cites no number)
#   figures   the same surfaces as a PNG, an earlier rendering ([figures.<asset_dir>] kind = "focus", each a gridded surface) →
#             plotsdir(), numbers.tex (empty: the figure cites no number)
#
#   julia --project=experiments experiments/scripts/focus_surfaces.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--force]
#
# The refinements must exist (scripts/radiometric_refinement.jl; scripts/reproduce/reproduce_results_vcai.sh (configs/reproduce.toml)).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))                                  # the refinement configs the focus-surface panels name (refine_product)
include(srcdir("figures.jl"))
include(srcdir("paper_figures", "PaperFigures.jl"))                           # the paper figure's PGFPlots text (its generator: focus_surfaces.jl there)
using .Figures: focus_surface_figure
import VisualGeometryToolkitEval.Reproduce as Reproduce
using StaticArrays: SVector
import VisualGeometryToolkitCore as VgtkCore

const DRIVER = "scripts/focus_surfaces.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "focus_surfaces.toml")))   # default: the config of record

"The keys of `[panels]`, every one required: the figure's name under plotsdir(), the px between lattice samples and the panels."
const PANELS_KEYS = ("asset_dir", "spacing_px", "cases")
"""The keys of a `[panels] cases` entry, every one required: the refinement config (project-relative) and its case, the panel's
file name, its title, its view (PGFPlots' azimuth and elevation, degrees) and the spacing of its drawn lattice lines (one every
`grid_every` lattice samples)."""
const PANEL_CASE_KEYS = ("config", "case", "lens", "label", "azimuth_deg", "elevation_deg", "grid_every")

"""
    check_panels_table(t, where) -> t

`[panels]` checked: exactly its keys (`PANELS_KEYS`); `asset_dir` a name and `spacing_px` an integer ≥ 1; `cases` the two panels,
left and right (the figure's frame and its caption are written for two), each exactly the keys of `PANEL_CASE_KEYS` — its config a
file of this project, its `lens` a file name of its own, its `label` "<lens name>, f/<f-number>" (the panel's title is typeset from
the two), its view one from the front left above (the axes' numbers stand on the three edges of the box that such a view shows)
and its `grid_every` an integer ≥ 1. An `ArgumentError` naming `where` otherwise.
"""
function check_panels_table(t, where::AbstractString)
    check_required_keys(t, PANELS_KEYS, where)
    t["asset_dir"] isa AbstractString && !isempty(t["asset_dir"]) || throw(ArgumentError("$where asset_dir: the figure's name, a non-empty string"))
    t["spacing_px"] isa Integer && t["spacing_px"] >= 1 || throw(ArgumentError("$where spacing_px: an integer ≥ 1"))
    t["cases"] isa AbstractVector && length(t["cases"]) == 2 || throw(ArgumentError("$where cases: an array of two panels, the left and the right one"))
    for c in t["cases"]
        check_required_keys(c, PANEL_CASE_KEYS, "$where cases entry")
        all(k -> c[k] isa AbstractString && !isempty(c[k]), ("config", "case", "lens", "label")) ||
            throw(ArgumentError("$where cases: an entry names config, case, lens and label (strings)"))
        isfile(projectdir(c["config"])) || throw(ArgumentError("$where cases: no config $(c["config"])"))
        length(split(c["label"], ", f/")) == 2 || throw(ArgumentError("$where cases: a panel's label is \"<lens name>, f/<f-number>\", got $(repr(c["label"]))"))
        all(k -> c[k] isa Real && 0 < c[k] < 90, ("azimuth_deg", "elevation_deg")) ||
            throw(ArgumentError("$where cases: a panel's azimuth_deg and elevation_deg, a view from the front left above: each between 0 and 90 degrees"))
        c["grid_every"] isa Integer && c["grid_every"] >= 1 || throw(ArgumentError("$where cases: a panel's grid_every, an integer ≥ 1"))
    end
    allunique(c["lens"] for c in t["cases"]) || throw(ArgumentError("$where cases: each panel has a lens of its own (the name of its file)"))
    return t
end

const CFG = let c = load_driver_config(CFG_PATH, "focus_surfaces"; tables = ("experiment", "inputs", "figures", "panels"))
    for (f, t) in c["figures"]
        # the focus surfaces of refined cameras (fig:focus-surfaces: src/figures.jl focus_surface_figure): a panel names the refinement
        # config and case whose final radiometric_camera.json it draws, sampled every spacing_px pixels, a grid line along every
        # grid_every-th row and column of that lattice, seen from the panel's elevation_deg above the x–y plane
        VgtkCore.check_keys(t, ("kind", "asset", "size_in", "dpi", "spacing_px", "grid_every", "title", "panels"), "$CFG_PATH [figures.$f]";
                           renamed = ("step_px" => "spacing_px",))
        get(t, "kind", "") == "focus" || throw(ArgumentError("$CFG_PATH [figures.$f]: kind = \"focus\" (the chrome residual maps are configs/experiments/chrome_evaluation_paper.toml's)"))
        t["size_in"] isa AbstractVector && length(t["size_in"]) == 2 && all(x -> x isa Real && x > 0, t["size_in"]) || throw(ArgumentError("$CFG_PATH [figures.$f]: size_in = [width, height] inches"))
        t["spacing_px"] isa Integer && t["spacing_px"] >= 1 || throw(ArgumentError("$CFG_PATH [figures.$f]: spacing_px, an integer ≥ 1"))
        haskey(t, "grid_every") && t["grid_every"] isa Integer && t["grid_every"] >= 2 || throw(ArgumentError("$CFG_PATH [figures.$f]: grid_every, an integer ≥ 2"))
        t["panels"] isa AbstractVector && !isempty(t["panels"]) || throw(ArgumentError("$CFG_PATH [figures.$f]: panels, a non-empty array"))
        for p in t["panels"]
            VgtkCore.check_keys(p, ("config", "case", "label", "elevation_deg"), "$CFG_PATH [figures.$f] panel")
            all(k -> haskey(p, k) && p[k] isa AbstractString && !isempty(p[k]), ("config", "case", "label")) ||
                throw(ArgumentError("$CFG_PATH [figures.$f]: a panel names config, case and label (strings)"))
            haskey(p, "elevation_deg") && p["elevation_deg"] isa Real && 0 <= p["elevation_deg"] <= 90 ||
                throw(ArgumentError("$CFG_PATH [figures.$f]: a panel's elevation_deg, the view's elevation in [0, 90] degrees"))
            isfile(projectdir(p["config"])) || throw(ArgumentError("$CFG_PATH [figures.$f]: no config $(p["config"])"))
        end
    end
    # the paper's figure (stage panels): its name, its lattice and its panels; a config without the table draws the PNG alone
    haskey(c, "panels") && check_panels_table(c["panels"], "$CFG_PATH [panels]")
    c
end

"The entries of the config that name a refinement (`config` and `case`), each with the table that holds it: the `[figures.<f>]` panels and the `[panels]` cases."
named_refinements(c::AbstractDict) = [[("[figures.$f]", p) for (f, t) in c["figures"] for p in t["panels"]];
                                      [("[panels]", p) for p in get(get(c, "panels", Dict{String, Any}()), "cases", Any[])]]
"The `[figures.<f>]` panels of the config: the entries whose refinements stage figures reads."
figure_panels() = [p for t in values(CFG["figures"]) for p in t["panels"]]
"The `[panels]` cases of the config: the entries whose refinements stage panels reads (none in a config without the table)."
panel_cases() = haskey(CFG, "panels") ? collect(CFG["panels"]["cases"]) : Any[]
"""The `[panels]` cases as the generator's part `tables` takes them, left to right: per panel its file name, the refinement's case,
its title, its view and the spacing of its drawn lattice lines."""
panel_settings() = [(; lens = String(c["lens"]), case = String(c["case"]), label = String(c["label"]), azimuth_deg = Float64(c["azimuth_deg"]),
                       elevation_deg = Float64(c["elevation_deg"]), grid_every = Int(c["grid_every"])) for c in panel_cases()]

"The refinements the config names, `(config, case) => product` (`refine_product` of the refinement config), each config read and each case resolved once."
const FOCUS_CASES = let cases = Dict{Tuple{String,String}, Product}(), configs = Dict{String, Any}()
    for (table, p) in named_refinements(CFG)
        k = (String(p["config"]), String(p["case"]))
        haskey(cases, k) && continue
        rc = get!(() -> load_radiometric_refinement_config(projectdir(k[1])), configs, k[1])
        haskey(rc["cases"], k[2]) || throw(ArgumentError("$CFG_PATH $table: $(k[1]) has no case $(repr(k[2]))"))
        cases[k] = refine_product(rc, k[2])
    end
    cases
end
"The refine product of an entry that names a refinement (`config` and `case`)."
refinement(p::AbstractDict) = FOCUS_CASES[(String(p["config"]), String(p["case"]))]
"The refine products the entries `ps` name as they key a product that reads them: `\"<config>::<case>\" => key`."
refinement_keys(ps) = Dict("$(p["config"])::$(p["case"])" => refinement(p).key for p in ps)

"The figures product (its output numbers.tex): keyed by [figures] and the keys of the refine products its panels draw."
figures_product() = driver_product("figures", Dict{String,Any}("experiment" => CFG["experiment"], "figures" => CFG["figures"],
                                                               "focus" => refinement_keys(figure_panels())); output = "numbers.tex", threads = nothing)
"The panels product (its output numbers.tex; the dumps, the tables and the figure's files kept beside it): keyed by [panels] and the keys of the refine products its cases name."
panels_product() = driver_product("panels", Dict{String,Any}("experiment" => CFG["experiment"], "panels" => CFG["panels"],
                                                             "upstream" => refinement_keys(panel_cases())); output = "numbers.tex", threads = nothing)

"""
    focus_surface_points(camera, W, H; spacing) -> (X, Y, Z, D)

The in-focus points of a `RadiometricCameraModel` on the `spacing`-px lattice of its `W × H` sensor (COLMAP pixels): for a pixel 𝐩 the lens
images, RadiometricCamera's focus point 𝔣(𝐩) = 𝔱(𝐩) d̂(𝐩) of the camera in its own frame (`𝔣(camera, 𝐩)`:
the chief ray's unit direction from the centre of projection and the focus distance 𝔱 = 1 / t̂ of the camera's focus, whatever its
kind — the sensor TPS or a direction chart), metres; NaN where the lens images nothing (`in_domain`) or t̂ ≤ 0. `D` = 𝔱.
"""
function focus_surface_points(camera, W::Integer, H::Integer; spacing::Integer)
    xs = (spacing / 2):spacing:W; ys = (spacing / 2):spacing:H
    X = fill(NaN, length(xs), length(ys)); Y = copy(X); Z = copy(X); D = copy(X)
    for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        p = SVector(Float64(x), Float64(y))
        RadiometricCamera.in_domain(camera.geometric, p) || continue
        all(isfinite, VgtkCore.backproject(camera.geometric, p)) || continue
        t̂ = RadiometricCamera.t̂(camera, p); (isfinite(t̂) && t̂ > 0) || continue   # the camera's t̂: the focus kind's (sensor TPS or a direction chart)
        X[i, j], Y[i, j], Z[i, j] = RadiometricCamera.𝔣(camera, p); D[i, j] = 1 / t̂  # RC's focus point 𝔣(𝐩) = d̂(𝐩) / t̂(𝐩), camera frame
    end
    return X, Y, Z, D
end

"The sensor size [px] of a refined camera: its lens's width and height where the lens carries them, else Core's `extent` of the bounded camera (as scripts/texture_recovery.jl reads it)."
_sensor_size(lens) = hasproperty(lens, :width) && hasproperty(lens, :height) ? (Int(lens.width), Int(lens.height)) :
    (e = VgtkCore.extent(lens); (Int(e.width), Int(e.height)))

"""
    camera_record(p, what) -> path

The record of the refined camera the entry `p` names (`config` and `case`): its refine product's radiometric_camera.json. `what`
names the reader; a refinement that wrote none (it is not complete) is an `ArgumentError`.
"""
function camera_record(p::AbstractDict, what::AbstractString)
    dir = product_dir(refinement(p))
    rec = joinpath(dir, "radiometric_camera.json")
    isfile(rec) || throw(ArgumentError("$what $(p["case"]): $dir has no radiometric_camera.json (the refinement is not complete)"))
    return rec
end

"Stage figures: `[figures.<f>]` of kind focus — each panel's refined camera (its refine product's radiometric_camera.json) sampled every spacing_px pixels (`focus_surface_points`), drawn by `Figures.focus_surface_figure` into `dir`."
function figure_focus_surfaces(t, out)
    panels = map(t["panels"]) do p
        rec = camera_record(p, "[figures] focus panel")
        cam = RadiometricCamera.load_radiometric_camera(rec).model          # (geometric: the focus surface of the model)
        W, H = _sensor_size(cam.geometric)
        X, Y, Z, D = focus_surface_points(cam, W, H; spacing = Int(t["spacing_px"]))
        n = count(isfinite, D); n > 0 || throw(ArgumentError("[figures] focus panel $(p["case"]): no pixel of $W × $H has a focus point"))
        println("   $(p["case"]): $n of $(length(D)) lattice points in focus, distance $(round(minimum(filter(isfinite, D)); digits = 3)) … $(round(maximum(filter(isfinite, D)); digits = 3)) m")
        (String(p["label"]), X, Y, Z, D)
    end
    fig = focus_surface_figure(panels; size_in = Tuple(Float64.(t["size_in"])), dpi = Int(t["dpi"]), title = String(t["title"]),
                               grid_every = Int(t["grid_every"]), elevations_deg = [Float64(p["elevation_deg"]) for p in t["panels"]])
    return Figures._save(fig, String(t["asset"]), out)
end

"The figures, kept in the product (`assets/`) and published to plotsdir() (`produce_figures`); numbers.tex, the output, written last."
figures(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH, inputs = [product_dir(refinement(q)) for q in figure_panels()]) do dir
    for (f, t) in sort(collect(CFG["figures"]); by = first)
        println(">> $f → ", figure_focus_surfaces(t, joinpath(dir, "assets")))
    end
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
end

"""
    write_focus_dump(path, X, Y, Z, D) -> path

The in-focus points `X`, `Y`, `Z` of a camera on its pixel lattice and their focus distance `D` (`focus_surface_points`) as the dump
the generator's part `tables` reads (src/paper_figures/focus_surfaces.jl): two Int64, the lattice's size nx and ny, then the four
arrays as Float64, nx × ny each, column-major, everything little-endian; a NaN stays a NaN (the lens images nothing there).
"""
function write_focus_dump(path::AbstractString, X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix, D::AbstractMatrix)
    size(X) == size(Y) == size(Z) == size(D) || throw(ArgumentError("write_focus_dump: the four arrays differ in size"))
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, htol(Int64(size(D, 1))), htol(Int64(size(D, 2))))
        foreach(A -> write(io, htol.(Matrix{Float64}(A))), (X, Y, Z, D))
    end
    return path
end

"""
    focus_panels(records, dir; asset_dir, spacing, panels) -> Vector{String}

The paper's figure `asset_dir` from refined cameras. `panels` are the figure's panels, left to right (`panel_settings`), and `records`
their `case => radiometric_camera.json`, in the same order. Per case the camera's in-focus points on the lattice of `spacing` px
(`focus_surface_points`) are written to `dir/dump/<case>.f64` (`write_focus_dump`); the generator's part `tables` writes the
figure's tables from the dumps into `dir/tables` (<lens>.tsv, meta.toml), and its part `figure` draws the figure from the tables
into `dir/assets` (<asset_dir>.tex, <asset_dir>/<lens>.tex, <asset_dir>-caption.tex); the files name this stage as their generator. Returns the
figure's files. Other cases than the panels', or a camera without a focus point, are an `ArgumentError`.
"""
function focus_panels(records, dir::AbstractString; asset_dir::AbstractString, spacing::Integer, panels)
    drawn = [p.case for p in panels]
    [String(first(r)) for r in records] == drawn ||
        throw(ArgumentError("focus_panels: the panels draw the cases $(join(drawn, ", ")), in this order, got $(join(first.(records), ", "))"))
    dump = joinpath(dir, "dump"); tables = joinpath(dir, "tables")
    for (case, rec) in records
        cam = RadiometricCamera.load_radiometric_camera(rec).model
        W, H = _sensor_size(cam.geometric)
        X, Y, Z, D = focus_surface_points(cam, W, H; spacing)
        d = filter(isfinite, D)
        isempty(d) && throw(ArgumentError("[panels] case $case: no pixel of $W × $H has a focus point"))
        write_focus_dump(joinpath(dump, case * ".f64"), X, Y, Z, D)
        println("   $case: $(size(D, 1)) × $(size(D, 2)) lattice samples of $spacing px on $W × $H, $(length(d)) with a focus point, " *
                "distance $(round(minimum(d); digits = 4)) … $(round(maximum(d); digits = 4)) m")
    end
    writer = "experiments/$DRIVER (stage panels; T. Harder's focus_surfaces.py)"   # what the figure's files name as their generator: this stage, and whose generator it runs
    PaperFigures.generate("focus-surfaces"; part = "tables", data = dump, out = tables, spacing, panels, generator = writer)
    return PaperFigures.generate("focus-surfaces"; part = "figure", data = tables, out = joinpath(dir, "assets"), asset_dir, generator = writer)
end

"""Stage panels: the paper's figure (fig:focus-surfaces) from the `[panels]` cases' refined cameras (`focus_panels`: their dumps,
the figure's tables and its files), kept in the product (`assets/`) and published to plotsdir() (`produce_figures`); numbers.tex,
the output, written last."""
panels(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH, inputs = [product_dir(refinement(c)) for c in panel_cases()]) do dir
    files = focus_panels([String(c["case"]) => camera_record(c, "[panels] case") for c in panel_cases()], dir;
                         asset_dir = String(CFG["panels"]["asset_dir"]), spacing = Int(CFG["panels"]["spacing_px"]), panels = panel_settings())
    foreach(f -> println(">> panels → ", f), files)
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
end

"Neither stage draws on a device (the refined cameras are records)."
const GPU_STAGES = Symbol[]

const STAGES = [
    Stage(:panels, "the paper's figure fig:focus-surfaces (plotsdir(<asset_dir>.tex, <asset_dir>/<lens>.tex, <asset_dir>-caption.tex)), its dumps and tables, numbers.tex";
          items = () -> haskey(CFG, "panels") ? [panels_product()] : Product[], product = identity, run = panels),
    Stage(:figures, "the focus surfaces → plotsdir(), numbers.tex"; items = () -> [figures_product()], product = identity, run = figures),
]

run_stages("focus_surfaces", STAGES)
