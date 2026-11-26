#!/usr/bin/env julia
# reproduce: writes plots/focus-surfaces.png (plotsdir(); the paper's Fig. 7 is the external focus-surfaces.tex)
# reproduce: reads configs/experiments/focus_surfaces.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/focus_surfaces.jl — the paper's focus-surface figure: the focus surfaces two refinements learned as surfaces of in-focus
# points in camera space ([figures.<name>] kind = "focus": fig:focus-surfaces — the fisheye at f/4 a sphere about the camera, the
# Tamron 20 mm at f/8 a plane; their radiometric_camera.json), each a gridded surface. Config of record: configs/experiments/focus_surfaces.toml. The
# chrome-board figures and tables (fig:color-wheels, tab:refinement-chrome, tab:defocus, chrome_all) are the chrome evaluation's
# own stages (scripts/chrome_evaluation_paper.jl, configs/experiments/chrome_evaluation_paper.toml).
#
# STAGES (each a product keyed by its config record and the keys of the products it reads; datadir("exp_pro", <experiment>, <stage>, <key>))
#   figures   the focus surfaces → plotsdir(), numbers.tex (empty: the figure cites no number)
#
#   julia --project=experiments experiments/scripts/focus_surfaces.jl [--config=<toml>] [--plan] [--force]
#
# The refinements must exist (scripts/radiometric_refinement.jl; scripts/reproduce/reproduce_results_vcai.sh (configs/reproduce.toml)).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))                                  # the refinement configs the focus-surface panels name (refinement_product)
include(srcdir("figures.jl"))
using .Figures: focus_surface_figure
import VisualGeometryToolkitEval.Reproduce as Reproduce
using StaticArrays: SVector
import VisualGeometryToolkitCore as VgtkCore

const DRIVER = "scripts/focus_surfaces.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "focus_surfaces.toml")))   # default: the config of record
const CFG = let c = load_driver_config(CFG_PATH, "focus_surfaces"; tables = ("experiment", "inputs", "figures"))
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
    c
end

"The focus-surface panels' refinements, `(config, case) => product` (`refinement_product` of the refinement config), resolved once."
const FOCUS_CASES = Dict{Tuple{String,String}, Product}(
    (String(p["config"]), String(p["case"])) => let rc = load_radiometric_refinement_config(projectdir(p["config"]))
        haskey(rc["cases"], p["case"]) || throw(ArgumentError("$CFG_PATH [figures]: $(p["config"]) has no case $(repr(p["case"]))"))
        refinement_product(rc, p["case"])
    end
    for t in values(CFG["figures"]) for p in t["panels"])

"The figures product (its output numbers.tex): keyed by [figures] and the keys of the refine products its panels draw."
figures_product() = driver_product("figures", Dict{String,Any}("experiment" => CFG["experiment"], "figures" => CFG["figures"],
                                                               "focus" => Dict("$c::$k" => v.key for ((c, k), v) in FOCUS_CASES)); output = "numbers.tex", threads = nothing)

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

"Stage figures: `[figures.<f>]` of kind focus — each panel's refined camera (its refine product's radiometric_camera.json) sampled every spacing_px pixels (`focus_surface_points`), drawn by `Figures.focus_surface_figure` into `dir`."
function figure_focus_surfaces(t, out)
    panels = map(t["panels"]) do p
        dir = product_dir(FOCUS_CASES[(String(p["config"]), String(p["case"]))])
        rec = joinpath(dir, "radiometric_camera.json")
        isfile(rec) || throw(ArgumentError("[figures] focus panel $(p["case"]): $dir has no radiometric_camera.json (the refinement is not complete)"))
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
figures(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH, inputs = [product_dir(v) for v in values(FOCUS_CASES)]) do dir
    for (f, t) in sort(collect(CFG["figures"]); by = first)
        println(">> $f → ", figure_focus_surfaces(t, joinpath(dir, "assets")))
    end
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
end

"The figures stage draws on no device (the refined cameras are records)."
const GPU_STAGES = Symbol[]

const STAGES = [
    Stage(:figures, "the focus surfaces → plotsdir(), numbers.tex"; items = () -> [figures_product()], product = identity, run = figures),
]

run_stages("focus_surfaces", STAGES)
