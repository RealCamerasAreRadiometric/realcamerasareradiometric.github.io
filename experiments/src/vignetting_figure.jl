# =============================================================================
# src/vignetting_figure.jl — the tables of the paper's vignetting figure (fig:vignetting; scripts/vignetting_figure.jl, its config of
# record configs/experiments/vignetting_figure.toml). Per lens, along the sensor's diagonal through the centre 𝔠 of the vignetting its
# large-aperture refinement learned, three curves, each over its value at 𝔠:
#
#   flat      the flat field: the lens's diffuser photograph, read as the refinement read its photographs (RadiometricCamera's entry
#             point `photo_set` under the learned camera's own observation model, `observe`: the usable green photosites of the RAW
#             photograph on the raw grid, the sensor record's black subtracted, its defects left out), at a sample the mean of the
#             photosites in a box about it, over the mean of the photosites within a disc about 𝔠
#   learned   the learned vignetting V (the refined camera's record), V(𝐩) / V(𝔠)
#   cos4      the thin lens model, cos⁴α with tan α = distance from 𝔠 / focal length — the refinement case's own nominal focal
#             length (its [pipeline.camera] f_mm), in pixels of the run's sensor (`case_focal_length`)
#
# The diagonal runs from the sensor's top left to its bottom right corner, t the signed distance from 𝔠 in pixels, and ends where the
# camera stops imaging: inside its lens coverage less a rim (Core's `lens_coverage(camera, rim)` and `imaged_region`: the camera is
# asked, no image circle is carried here) and inside the frame. The tables <lens>_diagonal.tsv and meta.toml are what
# src/paper_figures/vignetting.jl draws the figure from.
#
# Included after src/drivers.jl and src/radiometric_refinement.jl.
# =============================================================================
using Printf: @printf
using Statistics: mean
using StaticArrays: SVector

"The tables of configs/experiments/vignetting_figure.toml."
const VIGNETTING_FIGURE_TABLES = ("experiment", "inputs", "lenses", "diagonal", "figure")
"""The keys of a `[lenses.<lens>]` table, every one required: the refinement config of record and its case whose learned vignetting
is drawn (`config`, relative to the project; `case`), the input that holds the lens's diffuser photograph (`diffuser`, an
`[inputs.<id>]` of stage figures), the lens's name over its panel and inside the caption's sentence (`label`, `caption_label`:
"<name>, f/<number>") and the name of its panel's file (`panel`: <asset_dir>/<panel>-diagonal.tex). The focal length of the thin
lens model's curve is no key: it is the refinement case's own (`case_focal_length`)."""
const VIGNETTING_LENS_KEYS = ("config", "case", "diffuser", "label", "caption_label", "panel")
"""The keys of `[diagonal]`, every one required: the distance between two samples (`spacing_px`), the side of the box the flat field
is averaged over at a sample (`blur_px`, odd), the radius of the disc about 𝔠 whose mean is the flat field's value at 𝔠
(`centre_px`) and the rim of the lens coverage the diagonal stays clear of (`rim_px`)."""
const VIGNETTING_DIAGONAL_KEYS = ("spacing_px", "blur_px", "centre_px", "rim_px")
"""The keys of `[figure]`, every one required: the figure's name under plotsdir() — its file <asset_dir>.tex, the folder of its panels
and its caption's file (`asset_dir`), the lenses left to right (`order`, every lens of `[lenses]` once), the upper limit of the y axis
(`ymax`) and about how many of a diagonal's samples a curve is drawn through (`samples`)."""
const VIGNETTING_FIGURE_KEYS = ("asset_dir", "order", "ymax", "samples")

"""
    load_vignetting_figure_config(path) -> Dict

The config of record of the vignetting figure, every table and key checked: each `[lenses.<lens>]` names a config of this project
(its case is looked up by `vignetting_refinements`), an input of stage figures as its diffuser photograph, a label and a caption
label of the form "<name>, f/<number>" and a panel no other lens names; `[diagonal]` positive numbers, `blur_px` an odd integer;
`[figure] asset_dir` a non-empty string, `order` every lens once, `ymax` positive and `samples` a positive integer.
"""
function load_vignetting_figure_config(path::AbstractString)
    c = load_driver_config(path, "vignetting_figure"; tables = VIGNETTING_FIGURE_TABLES)
    for t in ("lenses", "diagonal", "figure")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    for (k, t) in c["lenses"]
        where = "$path [lenses.$k]"
        check_required_keys(t, VIGNETTING_LENS_KEYS, where)
        all(x -> t[x] isa AbstractString && !isempty(t[x]), VIGNETTING_LENS_KEYS) || throw(ArgumentError("$where: every key is a non-empty string"))
        isfile(projectdir(t["config"])) || throw(ArgumentError("$where: no config $(t["config"])"))
        d = get(c["inputs"], t["diffuser"], nothing)
        d !== nothing && d["stage"] == "figures" || throw(ArgumentError("$where: diffuser = $(repr(t["diffuser"])) is not an input of stage figures"))
        for x in ("label", "caption_label")
            occursin(r"^.+, f/[0-9]+(\.[0-9]+)?$", t[x]) || throw(ArgumentError("$where: $x is \"<name>, f/<number>\", got $(repr(t[x]))"))
        end
    end
    allunique(t["panel"] for t in values(c["lenses"])) || throw(ArgumentError("$path [lenses]: every lens names a panel of its own (the panel's file)"))
    d = check_required_keys(c["diagonal"], VIGNETTING_DIAGONAL_KEYS, "$path [diagonal]")
    all(k -> d[k] isa Real && d[k] > 0, VIGNETTING_DIAGONAL_KEYS) || throw(ArgumentError("$path [diagonal]: every key is a positive number"))
    d["blur_px"] isa Integer && isodd(d["blur_px"]) || throw(ArgumentError("$path [diagonal] blur_px: the side of a box centred on a pixel, an odd integer"))
    f = check_required_keys(c["figure"], VIGNETTING_FIGURE_KEYS, "$path [figure]")
    f["asset_dir"] isa AbstractString && !isempty(f["asset_dir"]) || throw(ArgumentError("$path [figure] asset_dir: the figure's name, a non-empty string"))
    f["order"] isa AbstractVector && sort(String.(f["order"])) == sort(collect(keys(c["lenses"]))) ||
        throw(ArgumentError("$path [figure] order: every lens of [lenses] once, left to right ($(join(sort(collect(keys(c["lenses"]))), ", ")))"))
    f["ymax"] isa Real && f["ymax"] > 0 || throw(ArgumentError("$path [figure] ymax: the upper limit of the y axis, a positive number"))
    f["samples"] isa Integer && f["samples"] > 0 || throw(ArgumentError("$path [figure] samples: about how many samples a curve is drawn through, a positive integer"))
    return c
end

"""
    vignetting_refinements(cfg; path = "the config") -> Dict

Per lens of `cfg`, `lens => (; config, rc, case, product)`: its refinement config of record (the path as the config states it and the
loaded config, each file loaded once), its case and the case's refine product (`refine_product`). A case the refinement config does
not hold is an `ArgumentError` naming `path`, the file `cfg` was read from.
"""
function vignetting_refinements(cfg::AbstractDict; path::AbstractString = "the config")
    loaded = Dict{String, Any}()
    return Dict{String, Any}(k => begin
            rc = get!(() -> load_radiometric_refinement_config(projectdir(t["config"])), loaded, String(t["config"]))
            haskey(rc["cases"], t["case"]) || throw(ArgumentError("$path [lenses.$k]: $(t["config"]) has no case $(repr(t["case"]))"))
            (; config = String(t["config"]), rc, case = String(t["case"]), product = refine_product(rc, t["case"]))
        end for (k, t) in cfg["lenses"])
end

"""
    vignetting_figure_product(cfg, refinements = vignetting_refinements(cfg)) -> Product

The figure's product (its output numbers.tex, written last; the tables in `tables/`, the figure's files in `assets/`): keyed by the
config's `[lenses]`, `[diagonal]` and `[figure]`, its inputs (the diffuser photographs) and the keys of the refine products it reads.
The computation is serial, so the thread count keys nothing.
"""
vignetting_figure_product(cfg::AbstractDict, refinements = vignetting_refinements(cfg)) =
    driver_product("figures", Dict{String, Any}("experiment" => cfg["experiment"], "lenses" => cfg["lenses"], "diagonal" => cfg["diagonal"],
                                                "figure" => cfg["figure"], "inputs" => inputs_record(cfg),
                                                "upstream" => Dict{String, Any}(k => r.product.key for (k, r) in refinements));
                   output = "numbers.tex", threads = nothing)

# ---- the thin lens model's focal length -----------------------------------------------------------------------------------------------

"""
    thin_lens_focal_px(options, W, H) -> Float64

The nominal focal length of a refinement case in pixels of its `W` × `H` photographs: `camera.f_mm` of the case's [pipeline] table
(`options`, RadiometricCamera's `PipelineOptions`) over the pixel pitch of the run's sensor — RadiometricCamera's `run_sensor`, the
sensor width `camera.sensor_mm` over `W` pixels: the sensor the refinement itself builds for these photographs and converts a focal
length with, so no pixel size is stated here.
"""
function thin_lens_focal_px(options::RadiometricCamera.PipelineOptions, W::Integer, H::Integer)
    sensor = RadiometricCamera.run_sensor(options, W, H)
    pitch_mm = VgtkCore.ustrip(VgtkCore.mm, VgtkCore.px * sensor.pitch.width)   # the length of one pixel along the rows
    return options.camera.f_mm / pitch_mm
end

"""
    case_focal_length(rc, case, camera) -> (; f_mm, f_px)

The thin lens model's focal length of a lens, taken from its refinement: `f_mm`, the nominal focal length of `[cases.<case>]` of the
refinement config `rc` (its `[pipeline.camera] f_mm`, read as the refinement reads its [pipeline] table: `pipeline_options`), and
`f_px`, the same in pixels of the refined `camera`'s photographs (`thin_lens_focal_px` with the size of the camera's sensor).
"""
function case_focal_length(rc::AbstractDict, case::AbstractString, camera)
    options = pipeline_options(rc["cases"][case]["pipeline"])
    e = VgtkCore.extent(camera.model.geometric)
    return (; f_mm = options.camera.f_mm, f_px = thin_lens_focal_px(options, Int(e.width), Int(e.height)))
end

# ---- the diagonal ------------------------------------------------------------------------------------------------------------------

"""
    box_mean(A, i, j, h) -> Float64

The mean of the finite values of `A` in the box of `2h + 1` rows and columns about row `i`, column `j`, cut at the border of `A`;
NaN when the box holds no finite value. An image that has a value only at some pixels (NaN elsewhere: a RAW photograph's green
photosites) is averaged over the pixels it has.
"""
function box_mean(A::AbstractMatrix, i::Integer, j::Integer, h::Integer)
    s = 0.0; n = 0
    for ii in max(1, i - h):min(size(A, 1), i + h), jj in max(1, j - h):min(size(A, 2), j + h)
        v = A[ii, jj]
        isfinite(v) || continue
        s += v; n += 1
    end
    return n == 0 ? NaN : s / n
end

"""
    disc_mean(A, cx, cy, radius)

The mean of the finite values of `A` at the pixels whose centres lie within `radius` of the point (`cx`, `cy`) (COLMAP pixels: the
centre of the pixel in row `i`, column `j` is (j − 0.5, i − 0.5)), in the element type of `A`. An error when there is none.
"""
function disc_mean(A::AbstractMatrix, cx::Real, cy::Real, radius::Real)
    H, W = size(A)
    rows = max(1, floor(Int, cy - radius)):min(H, ceil(Int, cy + radius) + 1)
    cols = max(1, floor(Int, cx - radius)):min(W, ceil(Int, cx + radius) + 1)
    near(i, j) = (j - 0.5 - cx)^2 + (i - 0.5 - cy)^2 < radius^2 && isfinite(A[i, j])
    any(near(i, j) for i in rows, j in cols) || throw(ArgumentError("disc_mean: no finite value within $radius px of ($cx, $cy)"))
    return mean(A[i, j] for i in rows, j in cols if near(i, j))
end

"""
    vignetting_diagonal(flat, vignetting, f_px; inside, spacing_px, blur_px, centre_px) -> NamedTuple

The three curves of one lens along its sensor's diagonal. `flat` is the flat field as an image of the sensor (rows × columns; NaN
where it has no value), `vignetting` the learned vignetting (a function of a pixel position, with Core's `center`), `f_px` the thin
lens model's focal length in pixels.

The samples lie every `spacing_px` on the line through 𝔠 = `center(vignetting)` in the direction of the sensor's diagonal (from the
top left to the bottom right corner), at the signed distances t = −R, −R + spacing_px, … ≤ R from 𝔠, R the half diagonal. A sample
is kept where `inside(x, y)` holds and it lies in the frame (0 ≤ x < W − 1, 0 ≤ y < H − 1). Per kept sample:

- `flat`: the mean of `flat` over the box of `blur_px` × `blur_px` pixels about the sample's pixel (`box_mean`), over the mean of
  `flat` within `centre_px` of 𝔠 (`disc_mean`);
- `learned`: `vignetting` at the sample over its value at 𝔠;
- `cos4`: cos⁴(atan(|t| / f_px)).

Returns `(; centre, t, flat, learned, cos4)`, `centre` = 𝔠 as a tuple. No kept sample is an error.
"""
function vignetting_diagonal(flat::AbstractMatrix, vignetting, f_px::Real; inside, spacing_px::Real, blur_px::Integer, centre_px::Real)
    H, W = size(flat)
    c = VgtkCore.center(vignetting); cx, cy = Float64(c[1]), Float64(c[2])
    at(x, y) = Float64(vignetting(SVector(Float64(x), Float64(y))))
    v0 = at(cx, cy); f0 = disc_mean(flat, cx, cy, centre_px)
    R = hypot(W, H) / 2
    t = collect(-R:spacing_px:R); xs = cx .+ t .* (W / (2R)); ys = cy .+ t .* (H / (2R))
    keep = [inside(x, y) && 0 <= x < W - 1 && 0 <= y < H - 1 for (x, y) in zip(xs, ys)]
    any(keep) || throw(ArgumentError("vignetting_diagonal: no sample of the diagonal through ($cx, $cy) is inside"))
    h = blur_px ÷ 2
    return (; centre = (cx, cy), t = t[keep],
            flat = [box_mean(flat, clamp(floor(Int, y), 0, H - 1) + 1, clamp(floor(Int, x), 0, W - 1) + 1, h) / f0 for (x, y) in zip(xs[keep], ys[keep])],
            learned = [at(x, y) / v0 for (x, y) in zip(xs[keep], ys[keep])],
            cos4 = [cos(atan(abs(s) / f_px))^4 for s in t[keep]])
end

"""
    write_diagonal_table(path, d) -> path

The curves `d` of a lens (`vignetting_diagonal`) as the table `path`: a header line, then per sample t (one decimal) and flat,
learned, cos4 (six decimals), tab-separated.
"""
function write_diagonal_table(path::AbstractString, d)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "t\tflat\tlearned\tcos4")
        for i in eachindex(d.t)
            @printf(io, "%.1f\t%.6f\t%.6f\t%.6f\n", d.t[i], d.flat[i], d.learned[i], d.cos4[i])
        end
    end
    return path
end

# ---- a lens: its camera, its diffuser photograph -----------------------------------------------------------------------------------

"""
    green_photosite_image(camera, photograph; characterisation) -> Matrix{Float64}

The RAW `photograph` as the calibrated `camera` (a `CalibratedCamera`) observes it, an image of its sensor (rows × columns): at every
usable green photosite its linear value, NaN everywhere else. The photograph is read through RadiometricCamera's entry point under
the camera's own observation model (`photo_set(camera, [photograph]; characterisation)`, `observe`; the values `masked_observations`
returns, the refinement's own reader): the sensor record's black is subtracted, the photosites the record lists as defects and the
censored ones are left out. `characterisation` is the location of the camera's sensor record (`case_characterisation`; `nothing` for a
camera calibrated without one). A decoded image, or another record than the camera's, is refused there; a camera calibrated on
another channel than the green is refused here.
"""
function green_photosite_image(camera, photograph::AbstractString; characterisation::Union{Nothing, AbstractString})
    camera.observation.channel === :G ||
        throw(ArgumentError("green_photosite_image: the camera was calibrated on the channel $(camera.observation.channel); the flat field is read at its green photosites"))
    e = VgtkCore.extent(camera.model.geometric); W, H = Int(e.width), Int(e.height)
    photos = RadiometricCamera.photo_set(camera, [photograph]; characterisation)
    return photosite_image(RadiometricCamera.observe(photos, 1, trues(H, W))...)
end

"""
    photosite_image(plane, usable) -> Matrix{Float64}

The observations of a photograph (`observe`: its `plane` of values and the mask `usable` of the photosites that carry one) as an
image: at every usable photosite its value (`masked_observations`, the green channel's reader), NaN everywhere else.
"""
function photosite_image(plane::AbstractMatrix, usable::AbstractMatrix{Bool})
    size(plane) == size(usable) || throw(ArgumentError("photosite_image: the plane is $(size(plane)), the mask $(size(usable))"))
    px = RadiometricCamera.masked_pixels(usable)
    values, _, _ = RadiometricCamera.masked_observations(plane, px; channel = 2)
    image = fill(NaN, size(usable))
    for (k, (u, v)) in enumerate(px)
        image[v, u] = values[k]
    end
    return image
end

"""
    diffuser_photograph(dir, what) -> path

The one photograph of the folder `dir` (RadiometricCamera's `photo_files`), a lens's flat field; `what` names the folder in the
`ArgumentError` raised when it holds none or several.
"""
function diffuser_photograph(dir::AbstractString, what::AbstractString)
    files = RadiometricCamera.photo_files(dir; join = true)
    length(files) == 1 || throw(ArgumentError("$what: one photograph is expected in $dir, found $(length(files))"))
    return only(files)
end

"""
    imaged_away_from_rim(lens, rim_px) -> region

Where the bounded camera `lens` images, `rim_px` clear of the edge of its lens coverage: Core's `imaged_region` of the camera with
its `lens_coverage(lens, rim_px)` — an image circle or ellipse loses `rim_px` at its rim, a lens that lights the whole sensor loses
nothing. A point is inside when `Point2(x, y) ∈ region`. A rim that leaves nothing of the coverage is an `ArgumentError`.
"""
function imaged_away_from_rim(lens::VgtkCore.BoundedCamera, rim_px::Real)
    c = VgtkCore.lens_coverage(lens, rim_px)
    c === nothing && throw(ArgumentError("imaged_away_from_rim: the lens coverage less rim_px = $rim_px leaves nothing"))
    return VgtkCore.imaged_region(VgtkCore.with_lens_coverage(lens, c))
end

"""
    lens_diagonal(camera, photograph, f_px, diagonal; characterisation) -> NamedTuple

The curves of one lens (`vignetting_diagonal`) from its refined `camera` (a `CalibratedCamera`: its learned vignetting, its lens
coverage, its observation model) and its diffuser `photograph` (a RAW file, `green_photosite_image`), with the settings `diagonal`
(the config's `[diagonal]` table); with them the number of green photosites read (`photosites`).
"""
function lens_diagonal(camera, photograph::AbstractString, f_px::Real, diagonal::AbstractDict; characterisation::Union{Nothing, AbstractString})
    flat = green_photosite_image(camera, photograph; characterisation)
    region = imaged_away_from_rim(camera.model.geometric, diagonal["rim_px"])
    d = vignetting_diagonal(flat, camera.model.vignetting, f_px; inside = (x, y) -> VgtkCore.Point2{Float64}(x, y) ∈ region,
                            spacing_px = diagonal["spacing_px"], blur_px = diagonal["blur_px"], centre_px = diagonal["centre_px"])
    return (; d..., photosites = count(isfinite, flat))
end

"""
    write_vignetting_tables(dir, cfg, lenses) -> dir

The figure's data folder `dir`, as src/paper_figures/vignetting.jl reads it: per lens `<lens>_diagonal.tsv` (`write_diagonal_table`)
and meta.toml — `diagonal_ymax` and `order` of the config's `[figure]`; per lens its `label` (the config's), the thin lens model's
focal length `f_mm` and `f_px` (`case_focal_length`), the centre 𝔠 and the ends `t_min`, `t_max` of its diagonal; under
`[sources.<lens>]` where its curves come from (the refinement config, the case and the key of its refine product, the diffuser
photograph's file name). `lenses` maps each lens to `(; curves, f_mm, f_px, refine_key, photograph)`, `curves` its `lens_diagonal`
with that `f_px`.
"""
function write_vignetting_tables(dir::AbstractString, cfg::AbstractDict, lenses::AbstractDict)
    mkpath(dir)
    meta = Dict{String, Any}("diagonal_ymax" => cfg["figure"]["ymax"], "order" => String.(cfg["figure"]["order"]),
                             "lenses" => Dict{String, Any}(), "sources" => Dict{String, Any}())
    for k in cfg["figure"]["order"]
        t = cfg["lenses"][k]; l = lenses[k]; d = l.curves
        write_diagonal_table(joinpath(dir, "$(k)_diagonal.tsv"), d)
        meta["lenses"][k] = Dict{String, Any}("label" => t["label"], "f_mm" => Float64(l.f_mm), "f_px" => Float64(l.f_px),
                                              "centre" => collect(d.centre), "t_min" => minimum(d.t), "t_max" => maximum(d.t))
        meta["sources"][k] = Dict{String, Any}("config" => t["config"], "case" => t["case"], "refine" => l.refine_key, "diffuser" => l.photograph)
    end
    open(joinpath(dir, "meta.toml"), "w") do io
        println(io, "# The tables of the vignetting figure (src/paper_figures/vignetting.jl), written by scripts/vignetting_figure.jl: per lens")
        println(io, "# <lens>_diagonal.tsv — t, px along the sensor's diagonal from the centre of the learned vignetting; the diffuser's flat field,")
        println(io, "# the learned vignetting and cos(atan(|t| / f_px))^4, each over its value at the centre. [lenses.<lens>]: the lens's name, the")
        println(io, "# thin lens model's focal length (f_mm, and f_px in the sensor's pixels), the centre and the ends of its diagonal;")
        println(io, "# [sources.<lens>]: where its curves come from.")
        TOML.print(io, meta; sorted = true)
    end
    return dir
end
