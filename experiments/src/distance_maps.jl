# =============================================================================
# src/distance_maps.jl — the printed boards' edge distance maps (RadiometricCamera's `edge.sdf`, `load_distance_map`) as keyed steps
# (owner 2026-11-16: a derived file is PRODUCED, never an input of record). Each map of configs/experiments/distance_maps.toml is ONE
# step, keyed by only its two inputs — the board generator's own output, a raw row-major float32 file `sdf.f32` and its header
# `sdf.json` (width, height, page_mm, clip_mm, sign, unit), each pinned by its sha256 — and the code of the packages it runs
# (`DISTANCE_MAP_CODE`): two configs naming the same files share it. The step reads them once (the one importer of that format) and
# writes a VisualGeometryToolkitIO raster record (vgio/raster/v2.0: float32 on the page chart's rectangle, the values bit for bit in
# its TIFF payload; owner D6, 2026-11-15: every external file through IO), whose provenance names the sources by SHA-256 with the
# header's descriptive entries (clip_mm, sign, method: RC computes with none of them). A consumer — a case's [pipeline] `edge.sdf`, a
# ground-truth set's `distance_map` — names it as a `[runs.<id>]` of this config (`run = "maps.<name>"`, `path = "distance_map.json"`)
# and reads it through `config_source` (`RUN_RESOLVERS["distance_maps"]`), produced on demand when missing. Included at the end of
# src/drivers.jl.
# =============================================================================

"The driver of the distance maps and its config of record."
const DISTANCE_MAPS_DRIVER = "scripts/distance_maps.jl"
const DISTANCE_MAPS_CONFIG = joinpath("configs", "experiments", "distance_maps.toml")
"The tables of the config of record."
const DISTANCE_MAPS_TABLES = ("experiment", "inputs", "maps")
"The keys of a `[maps.<name>]` table, every one required: the `[inputs]` ids of the generator's `sdf.f32` and its header `sdf.json`."
const DISTANCE_MAP_KEYS = ("values", "header")
"The record a map's step writes (vgio/raster/v2.0, its payload `distance_map.tiff` beside it), the file every consumer reads."
const DISTANCE_MAP_FILE = "distance_map.json"
"The packages the step runs (the per-package rule of the code a product is keyed by): IO's raster record on Core."
const DISTANCE_MAP_CODE = ("VisualGeometryToolkitCore", "VisualGeometryToolkitIO")

"""
    load_distance_maps_config(path) -> Dict

The config of record of scripts/distance_maps.jl: `[experiment] name = "distance_maps"`, its `[inputs]` and `[maps.<name>]` tables,
each exactly `DISTANCE_MAP_KEYS`, each value an input id of one file pinned by its `sha256`; an `ArgumentError` naming the table
otherwise.
"""
function load_distance_maps_config(path::AbstractString)
    c = load_driver_config(path, "distance_maps"; tables = DISTANCE_MAPS_TABLES)
    maps = get(c, "maps", nothing)
    maps isa AbstractDict && !isempty(maps) || throw(ArgumentError("$path: no [maps.<name>] table"))
    for (n, t) in maps
        where = "$path [maps.$n]"
        t isa AbstractDict || throw(ArgumentError("$where must be a table"))
        VgtkCore.check_keys(t, DISTANCE_MAP_KEYS, where)
        for k in DISTANCE_MAP_KEYS
            id = get(t, k, nothing)
            id isa AbstractString && haskey(c["inputs"], id) || throw(ArgumentError("$where $k: an [inputs] id, got $(repr(id))"))
            haskey(c["inputs"][id], "sha256") || throw(ArgumentError("$where $k: the input $id pins no sha256 (a map is keyed by its sources' content)"))
        end
    end
    return c
end

"""
    distance_map_product(cfg, name) -> Product

The step of `[maps.<name>]` of `cfg`: keyed by its two inputs (`values`, `header`: tier, path, sha256) and the code of
`DISTANCE_MAP_CODE` — not by the config, the map's name or the thread count (one conversion). Its output: `DISTANCE_MAP_FILE`.
"""
function distance_map_product(cfg::AbstractDict, name::AbstractString)
    t = cfg["maps"][name]
    rec = Dict{String, Any}("distance_map" => Dict{String, Any}(k => input_record(cfg["inputs"][t[k]]) for k in DISTANCE_MAP_KEYS))
    return Product("distance_maps", "prepare", rec; output = DISTANCE_MAP_FILE, code = code_of(DISTANCE_MAP_CODE))
end

"""
    run_distance_map(cfg, name; driver, config_path) -> dir

The step of `[maps.<name>]`, produced or loaded: the generator's `sdf.f32` (row-major float32, row 1 the texture's top) and its header
`sdf.json` (both read through [`config_input`](@ref): their sha256 checked) written as an IO raster record on the page chart's
rectangle (metres), read back and checked bit for bit, its page's widths as the header states.
"""
function run_distance_map(cfg::AbstractDict, name::AbstractString; driver::AbstractString, config_path::AbstractString)
    produce(distance_map_product(cfg, name); product_records(driver, config_path)...) do dir
        t = cfg["maps"][name]
        f32, hdr = config_input(cfg, t["values"]), config_input(cfg, t["header"])
        dirname(f32) == dirname(hdr) || throw(ArgumentError("[maps.$name]: values and header in two folders ($f32, $hdr)"))
        meta = VgtkIO.read_record(hdr)
        raw = Vector{Float32}(undef, meta.height * meta.width); read!(f32, raw)
        values = permutedims(reshape(raw, meta.width, meta.height))           # row = texture y (down): the raster's row 1 on top
        page = VgtkCore.Rect2(0.0, 0.0, Float64(meta.page_mm[1]) / 1000, Float64(meta.page_mm[2]) / 1000)   # the design chart, metres
        src = dirname(cfg["inputs"][t["values"]]["path"])                    # data-root-relative: no machine path in a record
        path = VgtkIO.save_raster(joinpath(dir, DISTANCE_MAP_FILE), values; frame = (chart = :page, area = page), nan = :none,
                                  provenance = (; source = "the generator's sdf.f32 + sdf.json in $(src), converted by $(driver)",
                                                sdf_f32_sha256 = cfg["inputs"][t["values"]]["sha256"], sdf_json_sha256 = cfg["inputs"][t["header"]]["sha256"],
                                                clip_mm = Float64(meta.clip_mm), sign = String(meta.sign), unit = String(meta.unit),
                                                method = haskey(meta, :method) ? String(meta.method) : ""))
        back = VgtkIO.load_raster(path)                                      # the record reads back bit for bit, and its page as stated
        isequal(back.values, values) || error("$path: the raster does not read back bit for bit")
        (1000 * back.frame.area.widths[1], 1000 * back.frame.area.widths[2]) == (Float64(meta.page_mm[1]), Float64(meta.page_mm[2])) ||
            error("$path: the page's widths do not read back as $(meta.page_mm) mm")
        println("distance map $name: $(meta.width) × $(meta.height) px over $(meta.page_mm[1]) × $(meta.page_mm[2]) mm → $path")
        Produced(; inputs = [f32, hdr])
    end
end

# A map of the config by "maps.<name>" (a `[runs]` entry of another config), with its maker (owner 2026-11-15: a consumer produces
# its missing upstream): the step itself, in this process — a CPU conversion of two files, no driver of its own to isolate.
RUN_RESOLVERS["distance_maps"] = function (cfg::AbstractDict, run::AbstractString, config_path::AbstractString)
    table, name = split(run, '.'; limit = 2)
    table == "maps" && haskey(get(cfg, "maps", Dict()), name) || throw(ArgumentError("distance_maps: a run is maps.<name> of $config_path, got $run"))
    product = distance_map_product(cfg, name)
    command = driver_run(DISTANCE_MAPS_DRIVER, config_path, "prepare").command
    return (; product, make = () -> (run_distance_map(cfg, name; driver = DISTANCE_MAPS_DRIVER, config_path = projectdir(config_path)); nothing), command)
end
