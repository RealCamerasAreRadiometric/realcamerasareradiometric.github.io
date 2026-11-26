# ---------------------------------------------------------------------------
# The grid lenses ARE VisualGeometryToolkitCore's spline cameras (the central generic spline camera model, the one the pipeline of
# Schöps et al. calibrates, as VisualGeometryToolkitCore implements it): a central grid
# is a `BoundedCamera` of a `CameraModel(K, SplineLens)` — a uniform bicubic B-spline lattice of direction controls over the
# K-normalized plane x = K⁻¹𝐩, K a skew-0 gauge (the initializing camera's own K). The bound is the image's extent, and the lens coverage (where
# light falls: the capture set's image region, `camera.lens_coverage`: the sensor or its measured record) is the bounded camera's; every consumer reads Core's
# `imaged_region(camera)`. A per-tile window (Core's `window`) is a bare camera of a block of the controls. This file holds no model:
# only the chief-ray interface of models/geometric.jl for Core's types and the builders from this package's cameras and samples.
#
#   chief_ray(m, 𝐩)  VisualRadiometryToolkitCore's: the raw backward spline at K⁻¹𝐩 (Core's `cam.lens.backward(x)`): the unit chief direction is
#                    taken downstream (`chief_direction`); a control's norm is a weight, not a gauge
#   in_domain        Core's bounded camera's (pixel_camera.jl): inside the lens coverage, and the lens images 𝐩 (its window present)
# ---------------------------------------------------------------------------

using VisualGeometryToolkitCore: SplineLens, BoundedCamera, spline_window, directions, grid_size, with_lens_coverage

"""Every grid lens: Core's spline cameras (bounded, or a bare window)."""
const GridLike = Union{BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}, CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}
"""Whether `l` is a grid lens (its nodes are optimised per tile; never a Θ_d block)."""
is_grid_lens(l) = l isa GridLike

"""
    lm_route(camera) -> PerViewRoute() | PerTileRoute()

The refinement's LM route, by the camera's TYPE (never a flag): a SplineLens camera is refined per tile (its lattice's patches,
`_lm_tiles!`: [`PerTileRoute`](@ref)), any other pixel camera per view in its own parameters (`_lm_views!`: [`PerViewRoute`](@ref)).
The problem spec (`_refinement_lm`), the LM's call and the run's camera records dispatch on it.
"""
function lm_route end
"""The LM route of a camera ([`lm_route`](@ref)): `PerViewRoute` or `PerTileRoute`."""
abstract type LmRoute end
"""The per-view route: a parametric pixel camera, its parameters the lens block of every view's residual (`_lm_views!`)."""
struct PerViewRoute <: LmRoute end
"""The per-tile route: a SplineLens camera, its lattice refined per tile (`_lm_tiles!`)."""
struct PerTileRoute <: LmRoute end
lm_route(::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}) = PerTileRoute()
lm_route(::BoundedCamera{<:CameraModel}) = PerViewRoute()
"""The route's name in a run's log lines and records' model labels."""
_route_label(::PerViewRoute) = "parametric"
_route_label(::PerTileRoute) = "grid"



"""The backward (direction) spline of a spline model."""
_backward(m::CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}) = m.lens.backward
# (the image formation's chief ray of a SplineLens camera — the raw backward spline value at K⁻¹𝐩 — is VisualRadiometryToolkitCore's
# `chief_ray`)

_model_eltype(::Type, g::GridLike) = g                  # never seeded as a whole: the per-tile route seeds the chief directions
# the element type of the grid's controls: a Float32 device grid (`convert_eltype(Float32, ·)`) gives a Float32 view — not a Float64 one, whose
# pose, focus and r ran the device flux kernel in FP64 (1/64 of the FP32 rate on a GeForce; 16× slower measured on an RTX 4090)
_model_number_type(g::GridLike) = eltype(eltype(directions(VgtkCore.model(g))))
# (a bounded SplineLens camera is also a pixel camera, `_PixelCamera`: these name the grid's methods for it)
_model_eltype(::Type, g::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}) = g
_model_number_type(g::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}) = eltype(eltype(directions(VgtkCore.model(g))))

"""
    initialization_region(src) -> region

Where the start camera `src` (the source a grid start is fitted to: a Schöps run's spline model, a session's camera) is DEFINED: its image ∩ its
lens model's `valid_domain` (Schöps's calibrated area, a `Rect2`, for its SplineLens) — the region a [`GridStart`](@ref) fits over and
the re-gridding's sensor-wide lattice no longer carries. A stage input of the initialization (the run's `rc/run_parameters/v1`
record holds it), never a camera property: the initialization's steps take it explicitly ([`in_initialization`](@ref)).
"""
initialization_region(src) = (e = VgtkCore.extent(src);
                              VgtkCore.Rect(0.0, 0.0, Float64(e.width), Float64(e.height)) ∩ VgtkCore.valid_domain(VgtkCore.model(src)))

"""
    in_initialization(detections, region) -> detections

The detections (pixel, board point) of `detections` whose pixel lies in the initialization `region` ([`initialization_region`](@ref),
Core's region membership `∈`): the ones an initialization step (a pose fit, the self-check) may evaluate the start at. `nothing` (a
parametric start, defined wherever its own valid domain says) keeps them all.
"""
in_initialization(detections, ::Nothing) = detections
in_initialization(detections, region) = [d for d in detections if SVector{2, Float64}(first(d)) ∈ region]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


"""
    _image_start(src, G; coverage = nothing) -> (; camera, fit)

[`GridStart`](@ref)'s [`ImageLattice`](@ref): the SplineLens camera of `G = (G_x, G_y)` controls over the image of the parametric
camera `src` by Estimation's `CameraRegression(; grid_size = G, forward = false)` on `CameraRegressionProblem(SplineLens, src,
imaged_region(src))` (`_spline_regression`): the source turned onto its optical axis first (`OpticalAxis`, Core's `axis_gauged`: K's
principal point at the axis's pixel — the same camera, its rays rotated), then the backward fitted on Core's full-image lattice of
that size to the turned source's rays over its imaged region by the penalized least squares of `PSpline` (one solve, no forward
spline; a node the region misses held by the penalty). The camera is bounded by `src`'s image, its lens coverage `coverage` (the
capture set's image region — where light falls, its keyed step's record, never the source lens's fold) or, without one, `src`'s own.
`fit` the regression's `FitResult`. No tolerance is asked of the resample and its fidelity is not reported (Estimation's held-out
errors of a resample are not yet right). A start whose optical-axis fit did not converge (the regression's retcode) proceeds on the
lattice the regression returned (the stalled axis's rotation 0): the start is not axis-gauged for that run, and one line says so.
"""
function _image_start(src, G::Tuple{Integer, Integer}; coverage = nothing)
    (G[1] >= 5 && G[2] >= 5) || throw(ArgumentError("grid must be at least 5 × 5"))
    s = coverage === nothing ? src : with_lens_coverage(src, coverage)
    fit = _spline_regression(s, VgtkCore.imaged_region(s); grid_size = (Int(G[1]), Int(G[2])))
    _axis_line(fit, G)
    return (; camera = VgtkCore.with_model(s, fit.u), fit)
end
# A parametric start's fit (`_image_start`, `_region_start`) whose optical-axis fit did not converge proceeds on the lattice it returned: one line.
_axis_line(fit, G) = VgtkCore.successful_retcode(fit) ||
    (@printf("start resample onto %d × %d: the optical-axis fit did not converge (%s, %s): the start is not axis-gauged for this run\n",
             G..., fit.retcode, fit.stop_rule); flush(stdout))
"""
    _spline_regression(source, region; grid_size = nothing, lattice_spacing = nothing, settings...) -> FitResult

Every SplineLens camera of RC's starts from a camera `source` over `region`: Estimation's `CameraRegression` (no tolerance asked,
no forward spline) on `CameraRegressionProblem(SplineLens, source, region; settings...)` (`support`, `device`), at the lattice
`grid_size` (a parametric source) or by `lattice_spacing` (a SplineLens source's exact resample; `nothing`: its own cell).
"""
_spline_regression(source, region; grid_size = nothing, lattice_spacing = nothing, settings...) =
    VgtkCore.solve(VgtkCore.EstimationProblem(VgtkEstimation.CameraRegressionProblem(SplineLens, source, region; settings...)),
                   VgtkEstimation.CameraRegression(; tolerance = Inf, grid_size, lattice_spacing, forward = false))

"""The bounded SplineLens cameras: a grid start's source when it is a spline model (a Schöps run's), resampled exactly."""
const SplineCamera = BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}

"""
    coverage_region(camera) -> region

Where a camera must be defined: the sensor `Rect(extent(camera))` ∩ its lens coverage (where light falls) — Core's
`imaged_region` without the model's own valid domain (FOLLOW-UP Core: a public name for it). The region a SplineLens start is
resampled over and the coverage assertion ([`check_coverage`](@ref)) holds the start and the calibrated camera to.
"""
coverage_region(camera::BoundedCamera) = VgtkCore.Rect(0.0, 0.0, Float64.(Tuple(VgtkCore.extent(camera)))...) ∩ VgtkCore.lens_coverage(camera)

"""
    check_coverage(camera, region; who, device = default_backend()) -> region

The coverage assertion, set-level: `region ⊆ valid_domain(model(camera))` (Core's `check_region_support`, a
`RegionBeyondSupport` naming `who` otherwise) and every pixel of `region` imaged — the camera's batched `backproject` on `device`
of each pixel centre in it (Core's `pixel_mask` convention) finite; an `ArgumentError` naming `who` and the count otherwise.

The containment is tested on `region` itself: VisualGeometryToolkitCore decides `RegionIntersection ⊆ Rect` by the intersection (the
sensor ∩ a housing's image circle whose sensor reaches past the lattice on one axis and circle on the other), not by the region's
bounding box.
"""
function check_coverage(camera::BoundedCamera, region; who::AbstractString, device = default_backend())
    m = VgtkCore.model(camera)
    VgtkCore.check_region_support(m, region; who)                 # a RegionIntersection ⊆ Rect holds by its parts' intersection
    W, H = Tuple(VgtkCore.extent(camera))
    px = VgtkCore.pixel_centres(region, 1:Int(W), 1:Int(H))
    rays = Array(VgtkCore.backproject(m, px; device))
    n = count(r -> !all(isfinite, r), rays)
    n == 0 || throw(ArgumentError("$who: $n of the $(length(px)) pixels of the coverage region $(region) are not imaged (NaN rays): " *
                                  "the camera does not cover the sensor ∩ its lens coverage"))
    return region
end

# ---- the geometry lattice's mode: `[grid] lattice` ---
"""How a grid start's geometry lattice is made (`[grid] lattice`, parsed ONCE by `read_switch` through [`LATTICE_TAGS`](@ref)):
[`DerivedLattice`](@ref) or [`NodeCountLattice`](@ref). Every decision on it is a dispatch ([`GridStart`](@ref))."""
abstract type LatticeRule end
"""`grid.lattice = "derived"`: the start source's OWN lattice (T. Schöps's camera) extended over the sensor ∩ the lens coverage on its
own spacing, split exactly by the integer m whose present nodes come closest to `grid.node_count` without exceeding it (no count:
m = 1; [`GridStart`](@ref)'s exact resample, its internal `exact_subdivision`). A source without a lattice (a parametric camera) is refused."""
struct DerivedLattice <: LatticeRule end
"""`grid.lattice = "node_count"`: a lattice of square cells laid over the image region's bounding box, sized so the nodes the region
supports come nearest `grid.node_count` ([`GridStart`](@ref); a parametric source). A SplineLens source is refused."""
struct NodeCountLattice <: LatticeRule end
"""The spellings of `grid.lattice` and the rule each reads as (`read_switch`'s cases; the writer's `switch_spelling`)."""
const LATTICE_TAGS = VgtkCore.TagRegistry(:lattice, "Lattice", "derived" => DerivedLattice, "node_count" => NodeCountLattice; paths = [r"grid\.lattice"])

"""
    ImageLattice(G_x, G_y)

A lattice of `G_x × G_y` controls over the whole image of a PARAMETRIC source, Core's full-image lattice (each at least 5): a
[`GridStart`](@ref)'s for a camera sampled into the generic grid model at a stated size (the session's `lens = :grid`, a test
scene's). Not a `grid.lattice` of a run's TOML (a run states `"derived"` or `"node_count"`, [`LATTICE_TAGS`](@ref)).
"""
struct ImageLattice <: LatticeRule
    grid_size::Tuple{Int, Int}
end
function ImageLattice(G_x::Integer, G_y::Integer)
    (G_x >= 5 && G_y >= 5) || throw(ArgumentError("ImageLattice: a lattice of at least 5 × 5 controls; got $G_x × $G_y"))
    return ImageLattice((Int(G_x), Int(G_y)))
end

"""
    GridStart(; lattice::LatticeRule, node_count = nothing)

The grid start of a source camera, a fit: the SplineLens camera (Core's generic grid model) of a camera `source`, solved on
`EstimationProblem(SplineLens, source[, (; coverage, device)])` — `coverage` the capture set's lens coverage (its measured image
region; `nothing`, the source's own), `device` the compute backend of an exact resample ([`default_backend`](@ref) by default) — by
VisualGeometryToolkitEstimation's `CameraRegression` (no tolerance, no forward spline), its lattice by `lattice`:

- [`DerivedLattice`](@ref)`()`: a SplineLens source (a Schöps run's model) resampled EXACTLY on its own lattice, extended over the
  sensor ∩ the coverage, subdivided by the integer m whose present nodes come closest to `node_count` without exceeding it (`nothing`:
  m = 1), padded to the grid route's patch multiple; `metadata.resample` the start record's entry (`_exact_start`);
- [`NodeCountLattice`](@ref)`()`: a parametric source's lattice of square cells over the image region, fitted at `node_count` (≥ 16)
  supported nodes, the others absent, padded; `metadata.spacing`, `origin`, `supported` (`_region_start`);
- [`ImageLattice`](@ref)`(G_x, G_y)`: a parametric source's full-image lattice of that size (`_image_start`).

The source's kind and the lattice must agree (`lattice_refusal`: an `ArgumentError` naming the key and the fix). The `FitResult`:
`u` the bounded SplineLens camera; `retcode`, `stop_rule`, `stats`, `rms` and `n` the regression's; `metadata.regression` its
metadata. A start whose optical-axis fit did not converge proceeds unturned, and one line says so.
"""
struct GridStart{L <: LatticeRule} <: VgtkCore.AbstractEstimationAlgorithm
    lattice::L
    node_count::Union{Nothing, Int}
end
function GridStart(; lattice::LatticeRule, node_count::Union{Nothing, Integer} = nothing)
    lattice isa NodeCountLattice && node_count === nothing &&
        throw(ArgumentError("GridStart: a NodeCountLattice lattice is fitted at a `node_count` of supported nodes; give one"))
    lattice isa NodeCountLattice && check_node_count(node_count)
    lattice isa ImageLattice && node_count !== nothing &&
        throw(ArgumentError("GridStart: an ImageLattice has its own size ($(join(lattice.grid_size, " × "))); no `node_count`"))
    return GridStart(lattice, node_count === nothing ? nothing : Int(node_count))
end
VgtkCore.estimator_kind(::GridStart) = VgtkCore.FitKind()

# the problem's context: the lens coverage and the device (the default backend only when none is stated)
_grid_start_coverage(p) = p isa NamedTuple ? get(p, :coverage, nothing) : nothing
_grid_start_device(p) = p isa NamedTuple && haskey(p, :device) ? p.device : default_backend()

function VgtkCore.__solve(prob::VgtkCore.EstimationProblem{VgtkCore.SplineLens}, alg::GridStart)
    st = _grid_start(prob.f, alg.lattice, alg.node_count; coverage = _grid_start_coverage(prob.p),
                     device = alg.lattice isa DerivedLattice ? _grid_start_device(prob.p) : nothing)
    f = st.fit
    extras = alg.lattice isa DerivedLattice ? (; st.resample) :
             alg.lattice isa NodeCountLattice ? (; st.spacing, st.origin, st.supported) : (;)
    return VgtkCore.FitResult(st.camera, f.retcode, f.stop_rule, f.stats, f.rms, f.n, (; regression = f.metadata, extras...))
end

# THE grid start of a source by its lattice (the solve's, and the front ends' `_start_grid`): the source's kind and the lattice agree
# (`lattice_refusal`), then the start of that lattice
function _grid_start(src, lattice::LatticeRule, node_count; coverage, device)
    r = lattice_refusal(lattice, lattice_source(src))
    r === nothing || throw(ArgumentError(r * " (the start's source: $(_lattice_source_label(src)))"))
    return _lattice_start(src, lattice, node_count; coverage, device)
end
_lattice_start(src::SplineCamera, ::DerivedLattice, node_count; coverage, device) = _exact_start(src; node_count, coverage, device)
_lattice_start(src, ::NodeCountLattice, node_count; coverage, device) = _region_start(src, node_count; coverage)
_lattice_start(src, l::ImageLattice, _node_count; coverage, device) = _image_start(src, l.grid_size; coverage)

"""
    _start_grid(source, grid::GridOptions; coverage = nothing, device = default_backend()) -> (; camera, resample)

A run's grid start of a source camera ([`GridStart`](@ref)'s, by `grid.lattice` and `grid.node_count`): a SplineLens source under
`"derived"` resampled EXACTLY on its own lattice; a parametric source under `"node_count"` fitted over the image region
(`resample = nothing`). The other two combinations are refused, naming the key and the fix.
"""
function _start_grid(src, grid; coverage = nothing, device = default_backend())
    st = _grid_start(src, grid.lattice, grid.node_count; coverage, device)
    return (; st.camera, resample = get(st, :resample, nothing))
end

"""The kind of a start's source for its lattice: `:spline` (a SplineLens camera, T. Schöps's: its own lattice) or `:parametric`."""
lattice_source(::SplineCamera) = Val(:spline)
lattice_source(_) = Val(:parametric)
_lattice_source_label(::SplineCamera) = "a SplineLens camera"
_lattice_source_label(src) = VgtkCore.lens_label(VgtkCore.lens(src))
lattice_refusal(::ImageLattice, ::Val{:spline}) =
    "an ImageLattice: a SplineLens source is resampled on its own lattice — `DerivedLattice()` (Estimation takes no `grid_size` for one)"
"""
    lattice_refusal(rule::LatticeRule, source::Val) -> Union{String, Nothing}

THE rule of a lattice mode against its start's source kind (`lattice_source`; the config reader's `frontend.type = "schoeps"` is a
`:spline` source): a SplineLens / Schöps source is its own lattice ("node_count" refused), a parametric one has none ("derived"
refused) — the refusal's text, or `nothing`. Read by the config reader (`_check_lattice`) and by the start ([`GridStart`](@ref)).
"""
lattice_refusal(::NodeCountLattice, ::Val{:spline}) =
    "`grid.lattice = \"node_count\"`: a Schöps / SplineLens start is its own lattice — state `grid.lattice = \"derived\"` (m = 1) and no `grid.node_count`"
lattice_refusal(::DerivedLattice, ::Val{:parametric}) =
    "`grid.lattice = \"derived\"`: a parametric camera has no lattice of its own — state `grid.lattice = \"node_count\"` and `grid.node_count` " *
    "(e.g. 7875, the 105 × 75 lattice's nodes)"
lattice_refusal(::LatticeRule, ::Val) = nothing

# ---- a parametric start's lattice over the image region ----------------------------------------
# STOPGAP (the stack's no-duplication rule): Estimation's CameraRegression of a SplineLens on a parametric source (grid_size)
# lays Core's full-image lattice over the source's bound, its origin the image's (0, 0), and takes no lattice area. Until it does,
# the region's lattice is that fit on the source CROPPED to the lattice's box: the same camera in the pixels of the box (K's
# principal point moved by the box's origin: `_origin_shifted`; the region moved with it: `_in_frame`), the fitted camera moved back.
# A pixel-frame change is a translation, so the lens and every ray are unchanged (the rule of Core's convert_frame, which has no
# LogicalIntrinsics method). FOLLOW-UP Estimation: a lattice-area setting of CameraRegressionProblem(SplineLens, …) (Core's spline_lattice already
# takes `calib_area`); then `_origin_shifted`, `_in_frame` and the crop in `_region_start` and `_layout_lattice` are DELETED, not kept.

# The camera model `m` (Core's LogicalIntrinsics over its lens) in the pixels of the frame whose origin is the pixel `o` of `m`'s.
function _origin_shifted(m::CameraModel{<:VgtkCore.LogicalIntrinsics}, o)
    K = SMatrix{3, 3, Float64}(VgtkCore.calibration_matrix(m))
    K′ = SMatrix{3, 3, Float64}(K[1, 1], K[2, 1], K[3, 1], K[1, 2], K[2, 2], K[3, 2], K[1, 3] - o[1], K[2, 3] - o[2], K[3, 3])
    return CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(K′)), VgtkCore.lens(m))
end
_origin_shifted(m, _) = throw(ArgumentError("a parametric start's lattice: a $(nameof(typeof(m))) source is no CameraModel of LogicalIntrinsics " *
                                         "(K in pixels), whose pixel frame the lattice's box is laid in"))
# A region of the pixel plane in the frame whose origin is the pixel `o`: Core's similarity action on regions, the translation by −o —
# one method for every region Core has (rectangles, ellipses, circles, the whole plane, intersections part by part, and the
# star `PolygonRegion` of a lens that is not radially symmetric: Brown–Conrady, OpenCV, a tilt).
_in_frame(r, o) = VgtkCore.SimilarityMap(Translation(-SVector{2, Float64}(o)))(r)

"""
    region_layout(region, h) -> (; origin, grid_size)

The lattice of square cells `h` px (an integer) over the bounding box of `region` (Core's `Rect2` of it, out to whole pixels): Core's
node-count rule over the box (`⌈w / h⌉ + 3` nodes per axis, the cubic window's border nodes; `grid_size = (n_x, n_y)`) on the area
`(n − 3)·h` px per axis — the box widened to a whole number of cells, centred on it to the pixel — from the pixel `origin` (integers).
"""
function region_layout(region, h::Integer)
    h >= 1 || throw(ArgumentError("region_layout: a cell of $h px"))
    b = VgtkCore.Rect2(region)
    lo = floor.(Int, Tuple(b.origin)); hi = ceil.(Int, Tuple(b.origin) .+ Tuple(b.widths))
    w = hi .- lo
    n = VgtkCore.Cameras._node_counts(w..., h)                    # Core's ONE node-count rule (FOLLOW-UP Core: public)
    return (; origin = lo .- ((n .- 3) .* h .- w) .÷ 2, grid_size = n)
end

"""`check_node_count(n)`: `grid.node_count` (a parametric start's supported nodes) is at least Core's smallest lattice, 4 × 4 nodes."""
check_node_count(n::Integer) =
    n >= 16 ? n : throw(ArgumentError("`grid.node_count` = $n: a parametric start's lattice has at least 4 × 4 = 16 nodes"))

# The lattice camera (every control the axis: Core's spline_lattice, no fit) of `layout` with the cell `h`, in the source's pixels.
_layout_lattice(m, layout, h) =
    _origin_shifted(VgtkCore.spline_lattice(SplineLens, _origin_shifted(m, layout.origin), VgtkCore.Size2(((layout.grid_size .- 3) .* h)...); layout.grid_size), .-layout.origin)
# The nodes of the layout at the cell `h` that `region` supports (Core's present-node rule, `node_mask`: the closed footprint meets it).
_supported_nodes(m, region, h) = (layout = region_layout(region, h); count(VgtkCore.node_mask(_layout_lattice(m, layout, h), region)))

"""
    region_spacing(m, region, node_count) -> h

The cell (px, an integer, one for both axes) of the lattice over `region`'s bounding box ([`region_layout`](@ref)) whose nodes that
`region` supports (Core's `node_mask`) come nearest `node_count` FROM BELOW — `node_count` is the case's bound on its present nodes
(a slow-lattice regression impossible): the count falls as the cell grows, so the cell is bracketed from the box's
area over `node_count` and bisected to a cell whose count is at most `node_count` while the next finer one's exceeds it, then the
largest count at most `node_count` among the cells within two of that crossing (the count falls only up to the ceilings and the
centring; ties to the finer). `m` is the source camera model, whose pixel frame the lattice is laid in.
"""
function region_spacing(m, region, node_count::Integer)
    check_node_count(node_count)
    counts = Dict{Int, Int}()                                     # each cell's count once (a lattice and its node_mask per cell)
    n(h) = get!(() -> _supported_nodes(m, region, h), counts, h)
    b = VgtkCore.Rect2(region).widths
    hi = max(1, round(Int, sqrt(b[1] * b[2] / node_count)))
    while n(hi) > node_count; hi *= 2; end                        # the coarse end: at most node_count
    lo = hi
    while lo > 1 && n(lo) <= node_count; lo = max(1, lo ÷ 2); end   # the fine end: more than node_count (or the 1 px cell)
    n(lo) <= node_count && return lo
    while hi - lo > 1                                             # n(lo) > node_count ≥ n(hi)
        mid = (lo + hi) ÷ 2
        n(mid) > node_count ? (lo = mid) : (hi = mid)
    end
    # the count is monotone only up to the ceilings and the centring: the largest count ≤ node_count of the crossing's neighbourhood
    # (ties to the finer); `hi` is one of them
    return argmin(h -> (node_count - n(h), h), filter(h -> n(h) <= node_count, max(1, lo - 2):(hi + 2)))
end

"""
    _region_start(src, node_count; coverage = nothing) -> (; camera, spacing, origin, supported, fit)

A PARAMETRIC start's lattice: the SplineLens camera of the parametric camera `src` on a lattice of
square cells laid over the bounding box of the image region — the sensor ∩ the lens coverage `coverage` (the capture set's image
region; `nothing`, `src`'s own: [`coverage_region`](@ref)) — whose cell (`region_spacing`, an integer px) makes the nodes the
region supports (Core's present-node rule, `node_mask`) come nearest `node_count`. Its controls are Estimation's public
`CameraRegression(; grid_size, forward = false)` on `CameraRegressionProblem(SplineLens, src, imaged_region(src))` (`_spline_regression`; the source turned onto its optical axis, the
penalized least squares of `PSpline`, as `_image_start`) on the source in the pixels of the lattice's box, moved back;
the nodes the region does not support are ABSENT (never read on it, never a parameter); then the lattice is padded at its right and
bottom edges to the grid route's patch multiple (`pad_lattice`). `spacing` (px), `origin` (the pixel where the lattice's area begins: its first node + one cell) and
`supported` (the nodes the region supports, before the padding) describe the lattice; `fit` the regression's `FitResult`.
"""
function _region_start(src, node_count::Integer; coverage = nothing)
    s = coverage === nothing ? src : with_lens_coverage(src, coverage)
    m = VgtkCore.model(s)
    region = coverage_region(s)
    h = region_spacing(m, region, node_count)
    layout = region_layout(region, h)
    o = layout.origin; G = layout.grid_size
    # the fit in the box's pixels: the source's camera there, bounded by the box, its lens coverage moved with it, over its imaged
    # region (the sensor ∩ the coverage ∩ its valid domain, as `_image_start`'s) moved with it
    box = VgtkCore.with_lens_coverage(BoundedCamera(_origin_shifted(m, o), VgtkCore.Size2(((G .- 3) .* h)...)), _in_frame(VgtkCore.lens_coverage(s), o))
    # the support the fit checks the region against: the source's valid domain moved by the same translation, the very object the
    # region holds (Estimation re-deriving it from the moved K could differ by rounding and refuse a region on its edge)
    fit = _spline_regression(box, _in_frame(VgtkCore.imaged_region(s), o); grid_size = G, support = _in_frame(VgtkCore.valid_domain(s), o))
    _axis_line(fit, G)
    fitted = _origin_shifted(fit.u, .-o)
    supported = BitMatrix(VgtkCore.node_mask(fitted, region))
    count(supported) <= node_count ||                              # the guard: never more present nodes than the case's literal
        error("_region_start: $(count(supported)) supported nodes exceed grid.node_count = $node_count at the $h px cell (region_spacing's bound)")
    pad = pad_lattice(VgtkCore.with_model(s, VgtkCore.with_present(fitted, supported)), region)
    @printf("start: a parametric source's lattice over the image region's box from pixel (%d, %d): %d × %d nodes of %d px (square), %d supported by the region (target %d)%s\n",
            o..., G..., h, count(supported), node_count,
            pad.folds === nothing ? "" : @sprintf("; padded at its right and bottom edges by %d × %d nodes to %d × %d (the grid route's patch multiple, %s)", pad.nodes..., grid_dims(pad.camera)..., join(GRID_PATCH, " × ")))
    flush(stdout)
    return (; pad.camera, spacing = h, origin = collect(Int, o), supported = count(supported), fit)
end

"""
    pad_lattice(camera::SplineCamera, region; patch = GRID_PATCH) -> (; camera, nodes, folds)

The SplineLens `camera`'s lattice extended at its right and bottom edges (Core's `extend_lattice`, its own spacing and origin) to the
next multiple of `patch` nodes per axis: the grid route tiles the lattice with fixed patch blocks (`GRID_PATCH`), while an
exact start's node counts (m·cells + 3) are set by the data. Every node of `camera` keeps its control bit for bit and its presence, so
the camera is the unpadded one bit for bit on `region` (the region it covers, inside its lattice). The added nodes lie past the
lattice: those whose closed footprint meets `region` (Core's `node_mask`; only where the region's edge is a knot of the lattice, the
first added ring, whose basis is zero on the region) are present with Core's bending-energy continuation; every other added node is
ABSENT (NaN), so it is never a parameter under either `grid.nodes` rule. `nodes` = `[added x, added y]`; `folds` is the padded call's
report (Core's `folded_pixels` of the extended lattice over the padded box, `(; folded, samples, unimaged)`), `nothing` when nothing
is added.
"""
function pad_lattice(camera::SplineCamera, region; patch::Tuple{Int, Int} = GRID_PATCH)
    m = VgtkCore.model(camera)
    G = grid_dims(camera)                                        # (G_x, G_y)
    r = mod.(.-G, patch)                                         # the nodes added per axis
    r == (0, 0) && return (; camera, nodes = [0, 0], folds = nothing)
    h = Float64.(Tuple(VgtkCore.lattice_spacing(m)))
    first = Float64.(Tuple(VgtkCore.node_position(m, CartesianIndex(1, 1))))
    last = Float64.(Tuple(VgtkCore.node_position(m, CartesianIndex(G[2], G[1]))))
    # Core's needed nodes of a box [lo, hi] are those of cell coordinate t ∈ (q_lo − 2, q_hi + 2), the lattice's at −1 … g − 2:
    # lo = the lattice area's origin (its first node + one cell) needs none before it; hi = the last node + (r − 1.5) cells, the
    # middle of the interval that ends the needed nodes r past the last
    lo = first .+ h
    hi = last .+ (r .- 1.5) .* h
    ext = VgtkCore.extend_lattice(m, VgtkCore.Rect(lo..., (hi .- lo)...))
    (ext.offset == (0, 0) && size(VgtkCore.directions(ext.camera)) == (G[2] + r[2], G[1] + r[1])) ||
        error("pad_lattice: Core's extend_lattice grew the $(G[1]) × $(G[2]) lattice by offset $(ext.offset) to " *
              "$(reverse(size(VgtkCore.directions(ext.camera)))), not by $(r[1]) × $(r[2]) at the right and bottom")
    present = BitMatrix(VgtkCore.node_mask(ext.camera, region))  # an added node: present where its footprint meets the region
    present[1:G[2], 1:G[1]] .= VgtkCore.present_nodes(m)         # the camera's own nodes as they were
    f = ext.folds
    folds = (; folded = [collect(Float64, p) for p in f.folded], samples = Int(f.samples), unimaged = Int(f.unimaged))
    return (; camera = VgtkCore.with_model(camera, VgtkCore.with_present(ext.camera, present)), nodes = collect(Int, r), folds)
end

"""
    exact_subdivision(m, region, node_count) -> (; m, present, nodes)

The subdivision factor of an exact start (`grid.lattice = "derived"` with its target `grid.node_count`): the source
model `m`'s lattice extended over `region` (Core's `extend_lattice`) and split k × k (Core's `subdivide_lattice`, exact knot
insertion); the largest k whose present nodes over `region` (Core's `node_mask`) are at most `node_count` — closest to the target
without exceeding it. `node_count = nothing`: k = 1. Refused when even his own cell (k = 1) exceeds the target: the target is the
case's bound on its present nodes. `present` and `nodes` are k's counts (present over the region, all of the extended lattice).
"""
function exact_subdivision(m, region, node_count)
    ext = VgtkCore.extend_lattice(m, region)
    counts(k) = (c = VgtkCore.subdivide_lattice(ext.camera, k).camera; (count(VgtkCore.node_mask(c, region)), prod(size(VgtkCore.directions(c)))))
    p, n = counts(1)
    node_count === nothing && return (; m = 1, present = p, nodes = n)
    p <= node_count || throw(ArgumentError("`grid.node_count` = $node_count: the exact start's own lattice (m = 1) already has $p present nodes over " *
                                           "the region — the target is the bound on the present nodes; state at least $p"))
    k = 1
    while true
        pk, nk = counts(k + 1)
        pk <= node_count || break
        k += 1; p, n = pk, nk
    end
    return (; m = k, present = p, nodes = n)
end

"""
    _exact_start(source::SplineCamera; node_count = nothing, coverage = nothing, device = default_backend()) -> (; camera, resample, fit)

The start of a SplineLens `source` (Schöps's lattice, `grid.lattice = "derived"`), resampled exactly by Estimation's
`CameraRegression(; forward = false)` on `CameraRegressionProblem(SplineLens, source, region)` (`_spline_regression`) over `region` = the sensor ∩ the lens coverage
(`coverage`, else the source's: [`coverage_region`](@ref)): his lattice extended on his own spacing until every node whose support
meets the region exists (his present nodes bit for bit, the new ones by Core's bending-energy continuation, marked
`extrapolated`), subdivided by the integer m whose PRESENT nodes over the region come closest to `node_count` without exceeding it
(the target of present nodes doubles as their bound; `nothing`: m = 1, his own cell — a fixed 40 px subdivision, as an earlier
release had, cost the 14 mm cases ≈ 9.6 k nodes and a ≈ 200 s host factorisation per LM step; `exact_subdivision`), Core's exact
cubic B-spline knot insertion (the start stays his camera), turned onto
his optical axis (Core's exact `axis_gauged`). The start keeps the full region: a fold of the extension is REPORTED (`resample.folds`,
Core's `folded_pixels` over the region), nothing trimmed or masked — the refinement may unfold it, and the calibrated camera is
checked ([`calibrated_folds`](@ref)). The lattice is then padded at its right and bottom edges to the grid route's patch multiple
(`pad_lattice`: nodes outside the region, absent but where the region's edge is a knot; the camera unchanged on it). The start is held to the coverage
assertion ([`check_coverage`](@ref)).

`resample` is the start record's entry (`start.json`, `resample`): the start's lattice's `grid_size` (padded), the `subdivision`
`(m_x, m_y)`, the resulting `lattice_spacing` (px), the `extension_offset` (Core's 0-based `(row, col)` of his first control),
`extrapolated` (the nodes' marks, rows `[gy][gx]` of the padded lattice; a present padded node marked), `padding` (`from`: the resample's own
`grid_size`; `nodes`: the nodes added per axis; `folds`: the padded call's report, `nothing` when none is added), `extrapolated_fraction` (the share of the region his model images no ray in),
`folds` (`folded` pixels, `samples`, `unimaged`), the fidelity to his rays where he images one (`max_px`, `max_px_edge`, `rms`) and
the axis turn (`axis_rotation`, the rows of R: every control of his is R·c in the start).
"""
function _exact_start(src::SplineCamera; node_count::Union{Nothing, Integer} = nothing, coverage = nothing, device = default_backend())
    s = coverage === nothing ? src : with_lens_coverage(src, coverage)
    region = coverage_region(s)
    sub = exact_subdivision(VgtkCore.model(s), region, node_count)
    @printf("grid: %d present of %d, m = %d (target %s)\n", sub.present, sub.nodes, sub.m, node_count === nothing ? "none: his cell" : string(node_count)); flush(stdout)
    # no `support`: Estimation's default (nothing) extends past the source's domain, which is the point; a stated one is checked.
    # The cell h / m (Estimation's exact resample takes the integer factors whose cell is nearest: m itself)
    h = Tuple(Float64.(VgtkCore.lattice_spacing(VgtkCore.model(s))))
    fit = _spline_regression(s, region; lattice_spacing = sub.m == 1 ? nothing : (h[1] / sub.m * VgtkCore.px, h[2] / sub.m * VgtkCore.px), device)
    resampled = VgtkCore.with_model(s, fit.u)
    # the lattice padded at its right and bottom edges to the grid route's patch multiple (`pad_lattice`): nodes outside the region,
    # the resample's camera bit for bit on it
    pad = pad_lattice(resampled, region)
    camera = pad.camera
    check_coverage(camera, region; who = "the start (exact resample)", device)
    md = fit.metadata; f = md.family
    X = VgtkCore.present_nodes(VgtkCore.model(camera)); X[axes(f.extrapolated)...] .= f.extrapolated   # the resample's marks; a padded node present: extrapolated
    resample = (; grid_size = collect(Int, grid_dims(camera)), subdivision = collect(Int, f.subdivision), lattice_spacing = collect(Float64, f.lattice_spacing),
                  extension_offset = collect(Int, f.extension_offset), extrapolated = [Bool.(collect(r)) for r in eachrow(X)],
                  padding = (; from = collect(Int, f.grid_size), nodes = pad.nodes, folds = pad.folds),
                  extrapolated_fraction = Float64(f.extrapolated_fraction),
                  folds = (; folded = [collect(Float64, p) for p in f.folds.folded], samples = Int(f.folds.samples), unimaged = Int(f.folds.unimaged)),
                  max_px = md.max_px, max_px_edge = md.max_px_edge, rms = Float64(fit.rms),
                  axis_rotation = [collect(Float64, row) for row in eachrow(SMatrix{3, 3, Float64}(VgtkCore.rotation(md.rotation)))])
    @printf("start: the source's lattice extended (its first control at %s), subdivided %d × %d, to %d × %d nodes of %.2f × %.2f px; %d nodes extrapolated, %.2f %% of the region unimaged by the source; folds: %d of %d samples (%d unimaged); fidelity to the source max %.3g px (edge %.3g), rms %.3g px\n",
            Tuple(f.extension_offset), f.subdivision..., f.grid_size..., f.lattice_spacing..., count(f.extrapolated),
            100 * f.extrapolated_fraction, length(f.folds.folded), f.folds.samples, f.folds.unimaged, md.max_px, md.max_px_edge, fit.rms)
    pad.folds === nothing ||
        @printf("start: the lattice padded at its right and bottom edges by %d × %d nodes to %d × %d (the grid route's patch multiple, %s; %d present, outside the region); padding folds: %d of %d samples (%d unimaged)\n",
                pad.nodes..., grid_dims(camera)..., join(GRID_PATCH, " × "), count(VgtkCore.present_nodes(VgtkCore.model(camera))) - count(VgtkCore.present_nodes(VgtkCore.model(resampled))), length(pad.folds.folded), pad.folds.samples, pad.folds.unimaged)
    flush(stdout)
    return (; camera, resample, fit)
end

"""
    calibrated_folds(camera; device = default_backend()) -> Union{Nothing, NamedTuple}

The calibrated camera's check on output (the fold policy), by its type: a SplineLens camera is held to the
coverage assertion ([`check_coverage`](@ref)) over [`coverage_region`](@ref) and its folds are Core's `folded_pixels` over it
(`(; folded, samples, unimaged)`); a parametric camera has no lattice to fold (`nothing`). A remaining fold flags the calibration
non-converged (`fold_termination`).
"""
function calibrated_folds(camera::SplineCamera; device = default_backend())
    region = coverage_region(camera)
    check_coverage(camera, region; who = "the calibrated camera", device)
    return VgtkCore.folded_pixels(camera, region)
end
calibrated_folds(camera; device = default_backend()) = nothing

"""
    fold_termination(code, rule, folds) -> (ReturnCode.T, StopRule)

The refinement's termination record with the calibrated camera's fold check folded in: a camera that still folds
(`folds.folded` non-empty) is no calibration — `(ReturnCode.Failure, ShapeGate)` (Core's validity-gate rule: the fitted shape
failed a validity gate), so it is flagged non-converged in its record and the paper's tables; else the LM's record as given.
"""
fold_termination(code, rule, folds) = validity_gate(code, rule, folds !== nothing && !isempty(folds.folded))
"""
    validity_gate(code, rule, failed::Bool) -> (ReturnCode.T, StopRule)

The one rule that flags a calibration non-converged on a failed validity check of its output: `(ReturnCode.Failure, ShapeGate)` (Core's
validity-gate rule: the fitted shape failed a validity gate) when `failed`, else the record `(code, rule)` as given. Its checks: the
camera's folds (`fold_termination`) and the focus surface's runaway nodes (`focus_runaway`).
"""
validity_gate(code, rule, failed::Bool) = failed ? (VgtkCore.ReturnCode.Failure, VgtkCore.ShapeGate) : (code, rule)

"""Direction control of node `(g_x, g_y)` (0-based) of a grid lens (Core's `directions`)."""
@inline _direction_node(g, gx::Integer, gy::Integer) = directions(VgtkCore.model(g))[gy + 1, gx + 1]
"""`(G_x, G_y)` of a grid lens's grid: Core's `grid_size` of its geometry."""
grid_dims(g::GridLike) = grid_size(VgtkCore.model(g))
"""A start camera's lattice for a log line: `G_x × G_y`, or none for a parametric camera."""
_lattice_line(g::GridLike) = join(grid_dims(g), " × ")
_lattice_line(_) = "none (a parametric camera)"


"""Largest angular deviation (radians) between the grid's and a reference model's chief rays on an `n × n` probe grid of the image
(the `margin` of each side, a fraction of the image size, left out), where the grid evaluates (its windows of present nodes)."""
function grid_angular_error(g::GridLike, ref; n = 41, margin = 0.05)
    worst = 0.0
    W, H = VgtkCore.extent(g)
    for y in range(margin * H, (1 - margin) * H; length = n), x in range(margin * W, (1 - margin) * W; length = n)
        𝐩 = SVector(x, y)
        d = backproject(g, 𝐩); dr = backproject(ref, 𝐩)
        all(isfinite, d) && all(isfinite, dr) || continue   # either does not image 𝐩 (no present window, past a fold): no rays to compare
        worst = max(worst, angle_between(d, dr))
    end
    return worst
end
