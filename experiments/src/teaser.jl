# =============================================================================
# src/teaser.jl — the paper's teaser figure (fig:teaser; scripts/teaser.jl, its config of record configs/experiments/teaser.toml):
# a part of one lens's panorama, composed from the frames as they are and from the frames with the learned vignetting divided out,
# over one strip of the learned-texture figure's bands. Nothing is estimated here. The stage reads two products by their keys —
# the lens's panoramas (scripts/panorama_stitching.jl, stage panoramas: panorama_<candidate>.png) and the learned-texture figure's
# tables (scripts/texture_recovery.jl, stage bands: tables/<band>.png) —, cuts the panorama's part out of each panorama
# (`crop_panorama`) and hands the crops and the tables to the drawing code (src/paper_figures/teaser.jl).
#
# Included after src/drivers.jl, src/panorama_example.jl, src/radiometric_refinement.jl and src/panorama_stitching.jl; the driver
# brings `run_isolated` (the texture driver is loaded in a module of its own for its bands product).
# =============================================================================

"The tables of configs/experiments/teaser.toml."
const TEASER_TABLES = ("experiment", "inputs", "upstream", "teaser")
"The keys of `[upstream]`, every one required: the lens's panorama_stitching config and the texture_recovery config whose stage bands writes the learned-texture figure's tables (both relative to the project)."
const TEASER_UPSTREAM_KEYS = ("panorama_stitching", "texture_recovery")
"""The keys of `[teaser]`, every one required: the figure's name under plotsdir() (`asset_dir`); the part of the panorama shown
(`box`: `[x0, y0, x1, y1]`, pixels of the panorama, first and past-last column and row) and the factor both crops' 8-bit values are
multiplied by (`gain`); the lens as the caption names it (`lens`, LaTeX); the panels of row 1 from the left (`panels`, each a table
of `TEASER_PANEL_KEYS`); the strip's parts from the left (`segments`, each a band of the texture figure's tables with its table
`[teaser.<segment>]`), the band whose image is the page's texel grid (`texel_grid`) and the pixels a texel is drawn with
(`pixels_per_texel`)."""
const TEASER_KEYS = ("asset_dir", "box", "gain", "lens", "panels", "segments", "texel_grid", "pixels_per_texel")
"The keys of one entry of `[teaser] panels`: its files' name (`panel`), the candidate of the panorama stage it shows (`candidate`: panorama_<candidate>.png) and its title (`title`, LaTeX)."
const TEASER_PANEL_KEYS = ("panel", "candidate", "title")
"The keys of a `[teaser.<segment>]` table: the part's name under the strip (`label`, LaTeX) and the caption's words for it (`caption`)."
const TEASER_SEGMENT_KEYS = ("label", "caption")

"""
    load_teaser_config(path) -> Dict

The config of record of the teaser, every table and key checked: `[upstream]` two configs of this project; `[teaser]` exactly
`TEASER_KEYS` and a table per segment — `asset_dir` and `lens` non-empty strings, `box` four integers with 0 ≤ x0 < x1 and
0 ≤ y0 < y1, `gain` a positive number, `panels` two tables of `TEASER_PANEL_KEYS` (the caption is written for two: the frames as
they are, then the frames with the learned vignetting divided out), each with a panel of its own, `segments` a non-empty array of
distinct band names (none a key of `[teaser]`), each with its `[teaser.<segment>]` of `TEASER_SEGMENT_KEYS`, `texel_grid` a
non-empty string and `pixels_per_texel` a positive integer.
"""
function load_teaser_config(path::AbstractString)
    c = load_driver_config(path, "teaser"; tables = TEASER_TABLES)
    u = check_required_keys(get(c, "upstream", nothing), TEASER_UPSTREAM_KEYS, "$path [upstream]")
    for k in TEASER_UPSTREAM_KEYS
        u[k] isa AbstractString && isfile(projectdir(u[k])) || throw(ArgumentError("$path [upstream] $k: no config $(repr(u[k]))"))
    end
    t = get(c, "teaser", nothing); w = "$path [teaser]"
    t isa AbstractDict || throw(ArgumentError("$path: no [teaser] table"))
    segments = get(t, "segments", nothing)
    segments isa AbstractVector && !isempty(segments) && all(s -> s isa AbstractString && !isempty(s) && !(s in TEASER_KEYS), segments) && allunique(segments) ||
        throw(ArgumentError("$w segments = $(repr(segments)): a non-empty array of distinct band names (none a key of [teaser]), from the left"))
    check_required_keys(t, (TEASER_KEYS..., segments...), w)
    for k in ("asset_dir", "lens", "texel_grid")
        t[k] isa AbstractString && !isempty(t[k]) || throw(ArgumentError("$w $k: a non-empty string"))
    end
    b = t["box"]
    b isa AbstractVector && length(b) == 4 && all(x -> x isa Integer, b) && 0 <= b[1] < b[3] && 0 <= b[2] < b[4] ||
        throw(ArgumentError("$w box: [x0, y0, x1, y1], pixels of the panorama (first and past-last column and row), 0 ≤ x0 < x1 and 0 ≤ y0 < y1"))
    t["gain"] isa Real && t["gain"] > 0 || throw(ArgumentError("$w gain: the factor the crops are brightened by, a positive number"))
    panels = t["panels"]
    panels isa AbstractVector && length(panels) == 2 || throw(ArgumentError("$w panels: two panels — the frames as they are, then the frames with the learned vignetting divided out"))
    for (i, p) in enumerate(panels)
        check_required_keys(p, TEASER_PANEL_KEYS, "$w panels[$i]")
        all(x -> p[x] isa AbstractString && !isempty(p[x]), TEASER_PANEL_KEYS) || throw(ArgumentError("$w panels[$i]: every key is a non-empty string"))
    end
    allunique(p["panel"] for p in panels) || throw(ArgumentError("$w panels: every panel names files of its own"))
    for s in segments
        check_required_keys(t[s], TEASER_SEGMENT_KEYS, "$path [teaser.$s]")
        all(x -> t[s][x] isa AbstractString && !isempty(t[s][x]), TEASER_SEGMENT_KEYS) || throw(ArgumentError("$path [teaser.$s]: every key is a non-empty string"))
    end
    t["pixels_per_texel"] isa Integer && t["pixels_per_texel"] > 0 || throw(ArgumentError("$w pixels_per_texel: a positive integer"))
    return c
end

"""
    teaser_panoramas(cfg; path = "the config") -> (; config, product)

The panoramas product the teaser of `cfg` crops: its `[upstream] panorama_stitching` config (the path as the config states it) and
that config's panoramas product (`panoramas_product`). A candidate of `[teaser] panels` that config does not render (neither
"original", the frames as they are, nor one of its `[maps]`) is an `ArgumentError` naming `path`.
"""
function teaser_panoramas(cfg::AbstractDict; path::AbstractString = "the config")
    config = String(cfg["upstream"]["panorama_stitching"]); pc = load_panorama_stitching_config(projectdir(config))
    for p in cfg["teaser"]["panels"]
        p["candidate"] == "original" || haskey(pc["maps"], p["candidate"]) ||
            throw(ArgumentError("$path [teaser] panels: $config renders no candidate $(repr(p["candidate"])) (original, $(join(sort(collect(keys(pc["maps"]))), ", ")))"))
    end
    return (; config, product = panoramas_product(pc, panorama_stitching_upstream(pc; path = config)))
end

"""
    teaser_bands(cfg; path = "the config") -> (; config, product)

The learned-texture figure's product the teaser of `cfg` takes its strip from: its `[upstream] texture_recovery` config and the
bands product of scripts/texture_recovery.jl on it (its tables: tables/<band>.png). That driver keys its products itself, so it
is loaded in a module of its own for its plan alone (`run_isolated` with --plan: nothing runs) and asked for its `bands_product`.
A texture config without a `[bands]` table is an `ArgumentError` naming `path`.
"""
function teaser_bands(cfg::AbstractDict; path::AbstractString = "the config")
    config = String(cfg["upstream"]["texture_recovery"])
    m = redirect_stdout(() -> run_isolated(scriptsdir("texture_recovery.jl"); args = ["--plan", "--config=$(projectdir(config))"]), devnull)
    haskey(Base.invokelatest(getglobal, m, :CFG), "bands") || throw(ArgumentError("$path [upstream] texture_recovery: $config has no [bands] table (the learned-texture figure's stage)"))
    return (; config, product = Base.invokelatest(Base.invokelatest(getglobal, m, :bands_product)))
end

"""
    teaser_product(cfg, panoramas, bands) -> Product

The teaser's product (its output numbers.tex, written last; the crops in `tables/`, the figure's files in `assets/`): keyed by
`[teaser]` and the keys of the two products it reads (`teaser_panoramas`, `teaser_bands`). One result at any thread count.
"""
teaser_product(cfg::AbstractDict, panoramas, bands) =
    driver_product("teaser", Dict{String, Any}("experiment" => cfg["experiment"], "teaser" => cfg["teaser"],
                                               "upstream" => Dict{String, Any}("panoramas" => panoramas.product.key, "bands" => bands.product.key));
                   output = "numbers.tex", threads = nothing)

"""
    crop_panorama(image, box, gain) -> Array{UInt8, 3}

The part `box = [x0, y0, x1, y1]` of the 8-bit image `image` (rows × columns × 3; the box in pixels, first and past-last column
and row, counted from 0), every value times `gain` in single precision, kept within 0 … 255 and cut to 8 bit (not rounded). A box
that is not inside the image is an `ArgumentError`.
"""
function crop_panorama(image::AbstractArray{UInt8, 3}, box, gain::Real)
    x0, y0, x1, y1 = box; H, W = size(image)[1:2]
    0 <= x0 < x1 <= W && 0 <= y0 < y1 <= H || throw(ArgumentError("crop_panorama: the box $(collect(box)) is not inside the image ($W x $H px)"))
    g = Float32(gain)
    return [trunc(UInt8, clamp(Float32(v) * g, 0.0f0, 255.0f0)) for v in @view image[y0+1:y1, x0+1:x1, :]]
end

"""
    write_teaser_crops(dir, panoramas, t) -> dir

The crops of row 1 of the `[teaser]` table `t`, written into `dir`: per panel `<panel>.png`, the part `box` of
`panoramas`/panorama_<candidate>.png brightened by `gain` (`crop_panorama`), an 8-bit RGB PNG (the drawing code's `save_png`).
"""
function write_teaser_crops(dir::AbstractString, panoramas::AbstractString, t::AbstractDict)
    for p in t["panels"]
        src = joinpath(panoramas, "panorama_$(p["candidate"]).png")
        isfile(src) || throw(ArgumentError("[teaser] panels: $panoramas has no panorama_$(p["candidate"]).png"))
        PaperFigures.save_png(joinpath(dir, p["panel"] * ".png"), crop_panorama(PaperFigures.load_png(src), t["box"], t["gain"]))
    end
    return dir
end

"""
    teaser_settings(cfg) -> NamedTuple

The settings of `[teaser]` of `cfg` as the drawing code takes them (src/paper_figures/teaser.jl `figure_teaser`): `asset_dir`, the
`panels` (`(; panel, title)`), `gain`, `lens`, the `segments` (`(; name, label, caption)`), `texel_grid` and `pixels_per_texel`.
"""
function teaser_settings(cfg::AbstractDict)
    t = cfg["teaser"]
    return (; asset_dir = String(t["asset_dir"]), panels = [(; panel = String(p["panel"]), title = String(p["title"])) for p in t["panels"]],
            gain = t["gain"], lens = String(t["lens"]),
            segments = [(; name = String(s), label = String(t[s]["label"]), caption = String(t[s]["caption"])) for s in t["segments"]],
            texel_grid = String(t["texel_grid"]), pixels_per_texel = t["pixels_per_texel"])
end
