# ---------------------------------------------------------------------------
# The chief-ray models' files — every one read and written by VisualGeometryToolkitIO:
#
#   save_lens / load_parametric_lens   a parametric model (Core's pixel camera, a `BoundedCamera` of a `CameraModel`) ↔ a
#                                       vgio/cameras/v6.0 record of one camera; any Core lens — the record carries the lens's own type
#   save_grid / load_grid              a grid camera (a bounded SplineLens camera) ↔ a vgio/cameras/v6.0 record (IO 0.10's one camera
#                                       kind); old vgio/generic_camera files read through IO's migration
#   schoeps_gridlens                   Schöps et al.'s central intrinsics0.yaml (IO's `load`) resampled into our G_x × G_y grid
#   load_lens(spec)                    "grid:<record>" | "lens:<record>" → the lens object (the start of a refinement)
#   load_model(spec, W, H)             "grid:" | "lens:" | "schoeps:" spec → (𝐩 ↦ unit direction, label)
#
# A model in memory and on disk is in the COLMAP pixels of its image (frames rule 1).
# ---------------------------------------------------------------------------

# ---- Schöps et al.'s camera_calibration models ---------------------------------------------------------------------------------
# Read by VisualGeometryToolkitIO's `load(BoundedCamera, intrinsics0.yaml)` where they are used: a central model as a
# SplineLens camera, K the lattice's gauge (Core's); the YAML is in SCHOEPS_PIXEL_FRAME
# (COLMAP's, the frame Core's spline cameras are read into).

"""The spline model of the Schöps run's `yamlpath` (VisualGeometryToolkitIO's reader: a bounded SplineLens camera), refused unless it is a `W × H` model
(`who` names the caller)."""
function _schoeps_model(yamlpath::AbstractString, W::Integer, H::Integer, who::AbstractString; kwargs...)
    src = VgtkIO.load(BoundedCamera, yamlpath; kwargs...)                      # Schöps's intrinsics0.yaml, COLMAP pixels
    VgtkCore.extent(src) == VgtkCore.Size2(W, H) ||
        throw(ArgumentError("$who: $yamlpath: expected a model of the photographs' size $W × $H px (a RAW photograph's raw grid, the mosaic's), " *
                            "found $(join(VgtkCore.extent(src), " × ")) px (a green-cell or binned front end is half the raw size: refit it on the raw grid)"))
    return src
end

"""
    schoeps_gridlens(yaml, W, H; coverage = nothing, device = default_device()) -> BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}

A Schöps run's central spline model (VisualGeometryToolkitIO's reader: a SplineLens camera, K its lattice's gauge) over the `W × H`
image (its own) resampled EXACTLY by [`exact_start`](@ref): his lattice extended over the sensor ∩ the lens coverage (`coverage`, the
run's image circle; else his own) on his own spacing, never subdivided, turned onto his optical axis.
"""
schoeps_gridlens(yamlpath::AbstractString, W::Integer, H::Integer; kwargs...) = schoeps_start(yamlpath, W, H; kwargs...).lens

"""
    schoeps_start(yaml, W, H; coverage = nothing, device = default_device()) -> (; lens, init_region, resample)

[`schoeps_gridlens`](@ref)'s grid start `lens`, the source's [`initialization_region`](@ref) `init_region` (Schöps's calibrated
area: where his model is defined) and the resample's record `resample` ([`exact_start`](@ref)).
"""
function schoeps_start(yamlpath::AbstractString, W::Integer, H::Integer; coverage = nothing, device = VgtkCore.default_device())
    src = _schoeps_model(yamlpath, W, H, "schoeps_gridlens")
    src isa SplineCamera ||
        throw(ArgumentError("schoeps_gridlens: $yamlpath holds a non-central model; this package's cameras are central (a Schöps run's central-generic model)"))
    st = exact_start(src; coverage, device)
    return (; lens = st.camera, init_region = initialization_region(src), st.resample)
end

# ---- parametric models ↔ vgio/cameras/v6.0 ---------------------------------------------------------------------------------
"""The display name of a parametric model's lens (a label of reports and run names): Core's `lens_label`."""
lens_type_name(c::_PixelCamera) = VgtkCore.lens_label(VgtkCore.lens(c))

"""
    save_lens(path, camera; provenance = (;))

A parametric model (Core's pixel camera) as VisualGeometryToolkitIO's camera record (`vgio/cameras/v6.0`, one camera `:cam`): K in COLMAP
pixels, the lens Core's lens with its own type and coefficients, the image's bound and the lens coverage. `provenance`: the run's labels (the model's name,
the iteration, …).
"""
save_lens(path::AbstractString, c::_PixelCamera; provenance::NamedTuple = (;)) =
    VgtkIO.save(path, Dict(:cam => c); provenance)

"""
    load_parametric_lens(path) -> camera

The one camera of a VisualGeometryToolkitIO camera record: `load(BoundedCamera, path)`, the pixel camera of its image.
"""
load_parametric_lens(path::AbstractString) = VgtkIO.load(VgtkCore.BoundedCamera, path)

# ---- grid models ↔ vgio/cameras/v6.0 ----------------------------------------------------------------------------------------
"""
    save_grid(path, lens; provenance = (;))

A grid lens as VisualGeometryToolkitIO's camera record (`vgio/cameras/v6.0`, one camera `:cam`; IO 0.10's one camera kind holds
any bounded camera): the bounded SplineLens camera — its K, the lens's `{"spline": …}` body (the nodes it holds) and its lens
coverage.
"""
save_grid(path::AbstractString, g::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}};
          provenance::NamedTuple = (;)) = VgtkIO.save(path, Dict(:cam => g); provenance)

"""
    load_grid(path) -> BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}

The central grid camera of a VisualGeometryToolkitIO record (`load(BoundedCamera, path)`): a camera store's SplineLens camera, an old
`vgio/generic_camera` file through IO's read migration (never rewritten: its nominal pinhole the K, its `domain` the lens
coverage).
"""
function load_grid(path::AbstractString)
    g = VgtkIO.load(BoundedCamera, path)
    g isa BoundedCamera{<:CameraModel} ||
        throw(ArgumentError("load_grid: $path holds a non-central model; this package's cameras are central"))
    return g
end

"""
    load_lens(spec) -> BoundedCamera

The lens of a model spec that names a record: `grid:<record>` (a grid camera, `load_grid`) or `lens:<record>` (a camera record: Core's
pixel camera). The start of a refinement (`io.init` of the pipeline options).
"""
function load_lens(spec::AbstractString)
    kind, rest = split(spec, ":"; limit = 2)
    kind == "grid" && return load_grid(rest)
    kind == "lens" && return load_parametric_lens(rest)
    error("load_lens: spec must be grid:<record> or lens:<record> (got $spec)")
end

"""
    load_start(spec) -> (; lens, init_region)

A stored start (`io.init`, [`load_lens`](@ref)'s spec) and its initialization region: `lens:` a parametric camera, defined wherever
its own `valid_domain` says (`init_region` `nothing`); `grid:` a grid camera the pipeline wrote (a result, a saved initialization or a
checkpoint), with its initialization region from the record beside it (`initialization_region.json`,
[`stored_initialization_region`](@ref)) — never the sensor by default.
"""
function load_start(spec::AbstractString)
    lens = load_lens(spec)
    kind, rest = split(spec, ":"; limit = 2)
    kind == "lens" && return (; lens, init_region = nothing)
    return (; lens, init_region = stored_initialization_region(rest))
end

"""
    load_camera(spec) -> camera

The bounded Core camera of a model spec (as `load_model`'s), every record as it is: `grid:` the record's camera (IO's
`load(BoundedCamera, …)`: a bounded SplineLens camera), `lens:` the record's pixel camera, `schoeps:` a Schöps run's spline model as IO reads it
(a bounded SplineLens camera) — the object Core's `backproject` and `backproject_with_jacobian(camera, 𝐩)` take.
"""
function load_camera(spec::AbstractString)
    kind, rest = split(spec, ":"; limit = 2)
    kind == "grid" && return VgtkIO.load(BoundedCamera, rest)
    kind == "lens" && return load_parametric_lens(rest)
    kind == "schoeps" && return VgtkIO.load(BoundedCamera, _schoeps_yaml(rest))
    error("unknown model spec $spec")
end
"""A Schöps run's camera: the file itself, or the run directory's ([`schoeps_run_file`](@ref))."""
_schoeps_yaml(r::AbstractString) = isdir(r) ? schoeps_run_file(r, :camera) : r

"""
    schoeps_run_file(dir, what) -> String

The file of the Schöps run `dir` that holds `what` (`:camera` or `:detections`): the run's IMPORT when it has one —
`camera.json` / `detections.json`, VisualGeometryToolkitIO records in the raw grid's COLMAP pixels, written by a front-end stage
that imported the tool's run onto the raw grid (the experiments' scripts/frontends.jl; through its decode's frame map) — else the tool's own file,
`intrinsics0.yaml` / `dataset.bin`. IO's loaders read either form by its content (`load(BoundedCamera, …)`,
`load(TargetObservations, …)`), so every reader of a run takes its file from here.
"""
function schoeps_run_file(dir::AbstractString, what::Symbol)
    imported, native = what === :camera ? ("camera.json", "intrinsics0.yaml") :
                       what === :detections ? ("detections.json", "dataset.bin") :
                       throw(ArgumentError("schoeps_run_file: what is :camera or :detections, got $(repr(what))"))
    f = joinpath(dir, imported)
    return isfile(f) ? f : joinpath(dir, native)
end

"""
    load_model(spec, W, H) -> (unp, label)

`unp(𝐩::SVector{2})` = unit direction of the COLMAP pixel position 𝐩, or `nothing` where the model is undefined. Specs: `grid:<record>` | `lens:<record>` (a
camera record: a parametric model, e.g. BlobBoards' division camera of a session's camera.json) | `schoeps:<dir or yaml>`.
"""
function load_model(spec::AbstractString, W::Integer, H::Integer)
    kind, rest = split(spec, ":"; limit = 2)
    if kind == "grid" || kind == "lens"
        l = load_lens(spec); Wj, Hj = VgtkCore.extent(l)
        (Wj, Hj) != (W, H) && @warn "$kind model image size $(Wj)×$(Hj) ≠ $(W)×$(H)"
        return (𝐩 -> (in_domain(l, 𝐩) ? normalize(backproject(l, 𝐩)) : nothing)), "$kind:$(basename(rest))"
    elseif kind == "schoeps"
        m = load_camera(spec)                                                 # Schöps's YAML, COLMAP pixels
        m isa BoundedCamera{<:CameraModel} ||
            throw(ArgumentError("load_model: $spec holds a non-central model; this package's cameras are central (a Schöps run's central-generic model)"))
        unp(𝐩) = (d = backproject(m, SVector{2, Float64}(𝐩)); all(isfinite, d) ? SVector{3, Float64}(d) : nothing)   # NaN: not imaged
        return unp, "$kind:$(basename(isdir(rest) ? rest : dirname(rest)))"
    end
    error("unknown model spec $spec")
end
