# =============================================================================
# src/panorama_stitching.jl — the config of record of scripts/panorama_stitching.jl (one lens per config:
# configs/experiments/panorama_stitching_fx3_<lens>.toml), the two configs it reads ([upstream]: the lens's large-aperture refinement
# and its sweep's chain) and its keyed products, shared by that driver and the drivers that read its panoramas
# (scripts/panorama_figure.jl). Included after src/drivers.jl, src/panorama_example.jl and src/radiometric_refinement.jl.
# =============================================================================

"The tables of a panorama_stitching config."
const PANORAMA_STITCHING_TABLES = ("experiment", "inputs", "upstream", "maps", "evaluate", "lens", "panoramas")
"""The keys of `[panoramas]`, every one required: the frames' gains (`gains`: "none", one exposure, or "fit", fitted per candidate
with its V held fixed), the ring of the field angle a learned map is divided by its mean over (`unit_ring_deg`), the seam score's
raster and chunk (`seam_deg_per_pixel`, `chunk`), the panoramas' raster (`deg_per_pixel`) and the exposure the PNGs are written at
(`exposure_percentile` of the green plane of the panorama of the learned map `exposure_candidate`)."""
const PANORAMAS_KEYS = ("gains", "unit_ring_deg", "seam_deg_per_pixel", "chunk", "deg_per_pixel", "exposure_percentile", "exposure_candidate")

"""
    load_panorama_stitching_config(path) -> Dict

A config of record of scripts/panorama_stitching.jl, every table and key checked: `[upstream]` the refinement config whose cases
learn the maps and the sweep's panorama_vignetting config (both of this project); `[maps.<name>]` a learned vignetting — its `case`
of the refinement config, the `state` file of that case's refine product and its `label`; `[evaluate]` the grid a learned map is
sampled on (`spacing`, `size`); `[lens]` the focal length of the rings' field angle and the sweep's frame size (`f_px`,
`image_size`); `[panoramas]` exactly `PANORAMAS_KEYS`.
"""
function load_panorama_stitching_config(path::AbstractString)
    c = load_driver_config(path, "panorama_stitching"; tables = PANORAMA_STITCHING_TABLES)
    u = check_required_keys(get(c, "upstream", nothing), ("refinement", "panorama_vignetting"), "$path [upstream]")
    for k in ("refinement", "panorama_vignetting")
        u[k] isa AbstractString && isfile(projectdir(u[k])) || throw(ArgumentError("$path [upstream] $k: no config $(repr(u[k]))"))
    end
    get(c, "maps", nothing) isa AbstractDict && !isempty(c["maps"]) || throw(ArgumentError("$path: no [maps.<name>] table (a learned vignetting)"))
    check_tables(c, path, "maps", ("case", "state", "label"); required = ("case", "state", "label"))
    check_required_keys(get(c, "evaluate", nothing), ("spacing", "size"), "$path [evaluate]")
    check_required_keys(get(c, "lens", nothing), ("f_px", "image_size"), "$path [lens]")
    p = check_required_keys(get(c, "panoramas", nothing), PANORAMAS_KEYS, "$path [panoramas]")
    p["gains"] in ("fit", "none") || throw(ArgumentError("$path [panoramas] gains must be \"fit\" or \"none\""))
    r = p["unit_ring_deg"]
    r isa AbstractVector && length(r) == 2 && all(x -> x isa Real, r) && 0 <= r[1] < r[2] ||
        throw(ArgumentError("$path [panoramas] unit_ring_deg: [from, to], a ring of the field angle in degrees, 0 ≤ from < to"))
    haskey(c["maps"], p["exposure_candidate"]) ||
        throw(ArgumentError("$path [panoramas] exposure_candidate = $(repr(p["exposure_candidate"])) is not a learned map ($(join(sort(collect(keys(c["maps"]))), ", ")))"))
    return c
end

"""
    panorama_stitching_upstream(cfg; path = "the config") -> (; refinement_path, refinement, pv_path, pv)

The two configs a panorama_stitching config `cfg` reads, loaded: its `[upstream] refinement` (the refinement config whose cases
learn the maps; a map whose case it does not hold is an `ArgumentError` naming `path`) and its `[upstream] panorama_vignetting`
(the sweep's chain), each with its absolute path.
"""
function panorama_stitching_upstream(cfg::AbstractDict; path::AbstractString = "the config")
    refinement_path = projectdir(cfg["upstream"]["refinement"])
    refinement = load_radiometric_refinement_config(refinement_path)
    for (m, t) in cfg["maps"]
        haskey(refinement["cases"], t["case"]) || throw(ArgumentError("$path [maps.$m]: $(t["case"]) is not a case of $refinement_path"))
    end
    pv_path = projectdir(cfg["upstream"]["panorama_vignetting"])
    return (; refinement_path, refinement, pv_path, pv = load_panorama_vignetting_config(pv_path))
end

"""The evaluate product of a panorama_stitching config (its output maps.json; V_<map>.json per learned map): keyed by `[evaluate]`,
the maps' cases and states and the keys of the refine products it reads (`up`: `panorama_stitching_upstream`)."""
panorama_maps_product(cfg::AbstractDict, up) =
    driver_product("evaluate", Dict{String,Any}("experiment" => cfg["experiment"], "evaluate" => cfg["evaluate"],
                                                "maps" => Dict(m => Dict("case" => t["case"], "state" => t["state"]) for (m, t) in cfg["maps"]),
                                                "upstream" => Dict(m => refine_product(up.refinement, t["case"]).key for (m, t) in cfg["maps"])); output = "maps.json")

"""The panoramas product of a panorama_stitching config (its output panoramas.json; panorama_<candidate>.png per candidate): keyed
by `[panoramas]`, `[lens]`, `[evaluate]`, the maps' labels and the keys of the two products it reads — the sweep's run
(`panorama_vignetting_product`) and the learned maps (`panorama_maps_product`)."""
panoramas_product(cfg::AbstractDict, up) =
    driver_product("panoramas", Dict{String,Any}("experiment" => cfg["experiment"], "panoramas" => cfg["panoramas"], "lens" => cfg["lens"], "evaluate" => cfg["evaluate"],
                                                 "labels" => Dict(m => t["label"] for (m, t) in cfg["maps"]),
                                                 "upstream" => Dict{String,Any}("panorama_vignetting" => panorama_vignetting_product(up.pv).key,
                                                                                "maps" => panorama_maps_product(cfg, up).key)); output = "panoramas.json")

"""
    panoramas_upstream(config, what) -> dir

The directory of the panoramas product of the panorama_stitching config `config` (project-relative) another driver (`what`)
reads, produced on demand (`produce_upstream`): its driver through its stages, in a process that holds a GPU when the sweep's
registration states LoMa's "cuda" provider (the stage reads the registration through the example, which resolves the provider).
"""
function panoramas_upstream(config::AbstractString, what::AbstractString)
    cfg = load_panorama_stitching_config(projectdir(config)); up = panorama_stitching_upstream(cfg; path = config)
    return produce_upstream(panoramas_product(cfg, up), what; driver_run("scripts/panorama_stitching.jl", config, "refine,evaluate,rooftop,panoramas")...,
                            gpu = up.pv["example"]["registration"]["execution_provider"] == "cuda")
end
