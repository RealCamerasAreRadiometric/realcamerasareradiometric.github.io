# ---------------------------------------------------------------------------
# Parameter blocks and the calibration state.
#
#   Θ_e  extrinsics (per view)            Θ_d  lens (a pixel camera: Core's BoundedCamera of a CameraModel; grid lenses per tile)
#   Θ_f  focus surface (t̂ nodal values)   r    aperture radius
#   Θ_l  lighting (SH coefficients)       Θ_v  vignetting (nodes)
#   Θ_m  material {α, η, μ}                h    board warping (+ the print's edge parameters) in the board's state (`board.state`)
#   Θ_i = Θ_d ∪ Θ_f ∪ {r},  Θ_c = Θ_e ∪ Θ_i,  Θ_r = Θ_l ∪ Θ_m,  Θ̄ = all
#
# The camera (Θ_d, Θ_f, r and Θ_v) is ONE value, `RadiometricCameraModel`, shared by every view; a view adds its
# pose. The renderer's per-view `View` is built from the two (`view_of`).
# ---------------------------------------------------------------------------

"""
    _LmRun()

The record of a calibration state's runs, held by the state (the no-global-state rule: nothing of a run lives in a module-level
container, so two states in one process never see each other's): `final_iteration`, the grid LM's mark that its iteration cap's last
iteration has begun (its trial linearization is never used: `_final_iteration_callback`, per_tile.jl); `scrubbed`, the non-finite
Jacobian entries `_scrub_entries!` replaced, by block, since the fit began (`_reset_scrubbed!`); `device_calls`, the device flux and
Jacobian evaluations (instrumentation the batched-path tests read). `lock` guards the counts: the views' jobs run concurrently.
"""
mutable struct _LmRun
    final_iteration::Bool
    scrubbed::Dict{Symbol, Int}
    device_calls::Dict{Symbol, Int}
    lock::ReentrantLock
end
_LmRun() = _LmRun(false, Dict{Symbol, Int}(), Dict{Symbol, Int}(:flux => 0, :jac => 0), ReentrantLock())

"""
    CalibrationState

Everything the calibration works on: the fixed data (the board, a `TexturedTarget`; the quadrature, masks, observed images ϕ; the image
size is the camera's) and the current
parameters Θ̄ — the camera every view shares (`camera`, a `RadiometricCameraModel`), one pose per view (`poses`, camera → world),
the material and the lighting.
"""
mutable struct CalibrationState{N, SC <: TexturedTarget, ST <: RenderSettings, TC <: RadiometricCameraModel, TL <: EnvironmentLight, TP <: EuclideanMap}
    board::SC
    settings::ST
    camera::TC                                # Θ_d, Θ_f, r and Θ_v, shared by every view
    poses::Vector{TP}                         # Θ_e: camera → world per view
    mat::Material{Float64}                    # Θ_m
    light::TL                                 # Θ_l
    masks::Vector{BitMatrix}
    px::Vector{Vector{Tuple{Int, Int}}}       # masked pixel lists
    targets::Vector{Array{Float64, 3}}        # observed images
    ϕ::Vector{Vector{Float64}}                # masked observed vectors
    weights::Union{Unweighted, Vector{Vector{Float64}}}   # the rows' weights (weighting.jl): Unweighted() (as measured), or 1/σ per row of each view's ϕ (whitened)
    vig_rule::Union{Nothing, VgtkCore.BoundaryRule, RadialSupportRule}   # a tensor V's interior / dependent nodes under the imaged region, a radial V's supported / replicating ones under the masks (`vignetting_rule`)
    backend::Any                              # KernelAbstractions.Backend; CPU() = threaded Float64 path
    jacobian_chunk::Int                       # dual partials per Jacobian kernel launch on a device (`compute.jacobian_chunk`, `_jac_chunk`)
    device_cache::Any                         # per-view device data (device/jacobian_kernels.jl); nothing on CPU
    channel::Int                              # 0: the three channels measured (3 residuals per pixel); c ∈ 1:3: channel c alone (1)
    focus_fixed::Vector{Int}                  # the focus nodes the LM holds (`focus_block`'s `fixed`; EXPERIMENT focus-ab variant B), none by default
    focus_smoothness::Matrix{Float64}         # EXPERIMENT (focus-ab): the focus's bending smoothness rows (`focus_smoothness!`); 0 × 0 = none
    mu_bounds::NTuple{2, Float64}             # the interval (lo, hi) of μ (`material.mu_bounds`): the LM's chart, the closed-form solves' clamp
    quanta::Vector{Union{Nothing, Vector{Float64}}}   # per view, each row's RAW code step in its units (`_quanta`); nothing: a decoded image
    sensor_models::Vector{VgtkCore.SensorModel}   # per view, the photograph's sensor model (Core's, resolved from the camera's characterisation by its EXIF: `photo_set`); the absent model for a decoded image
    sensor::SensorStatement                   # the photographs' sensor statement (observation_model.jl): the characterisation they were read under, or none
    photo_kind::PhotoKind                     # the photographs' kind: RAW photosites or decoded images (with `channel`, the state's `observation_model`)
    run::_LmRun                               # the state's runs: the final-iteration mark, the scrubbed entries, the device calls (`_LmRun`)
    start::Any                                # the start this state was lifted from (`start_calibration`: its geometry, views, costs, masks' record); nothing for a bare state
end

"""
    fit_channels(s) -> Tuple

The channels a residual reads per pixel: `(1, 2, 3)` for a colour photograph (`channel = 0`), `(c,)` for a single-channel run
(`io.channel` "R"/"G"/"B", the green photosites): ONE residual per measured pixel or photosite. The kernels evaluate the three
channels of the flux (one kernel on every backend); a single-channel residual takes channel `c`'s rows, and the lighting and
the refractive indices η of the other channels are not estimated (they follow channel `c`: `LightingVar`, `MaterialVar`).
"""
fit_channels(s::CalibrationState) = _fit_channels(s.channel)
"""The observation model of a state's photographs (its `sensor`, `photo_kind` and `channel`): what its records state."""
observation_model(s::CalibrationState) = ObservationModel(s.sensor, s.photo_kind, _channel_symbol(s.channel))
"""The calibrated camera of a state: its camera, its photographs' observation model and its render settings."""
calibrated_camera(s::CalibrationState) = CalibratedCamera(s.camera, observation_model(s), s.settings)
_fit_channels(c::Integer) = c == 0 ? (1, 2, 3) : (Int(c),)
"""The residual rows per pixel (`length(fit_channels(s))`)."""
n_channels(s::CalibrationState) = length(fit_channels(s))
"""The rows of the measured channels of a flux vector `x` laid out three channels per pixel (pixel-major): `x` itself for a colour
photograph, every third entry from `c` for channel `c` alone."""
_channel_rows(x::AbstractVector, c::Integer) = c == 0 ? x : x[c:3:end]

"""
    CalibrationState(board, settings, camera, poses, mat, light, targets; masks = nothing, δ = 0.02,
                     external_masks = nothing, support_min = 0, region = nothing, boundary_order = 2, backend = CPU(), ϕ = nothing,
                     channel = 0, focus_fixed = Int[], mu_bounds = (0.0, 1.0), jacobian_chunk = 4,
                     sensor_models = fill(SensorModel(), length(poses)), weights = Unweighted(), quanta = fill(nothing, length(poses)),
                     sensor = NoSensorRecord(), photo_kind = Decoded(false))

`channel` (0: the three channels; 1–3: that channel alone, ONE residual per measured pixel, `fit_channels`): a
single-channel state ties the other channels' lighting coefficients and refractive indices η to channel `channel`'s
(`tie_channels`), which no residual reads. `focus_fixed`: the focus nodes the LM holds (indices; `focus_block`); the focus's bending smoothness rows, `focus_smoothness`, are set by
`focus_smoothness!` (none at construction). `mu_bounds` `(lo, hi)` (`material.mu_bounds`; e.g. (0.6, 1.0): paper at most 40 % specular):
the interval of the diffuse weight μ — the LM steps on the chart μ = lo + (hi − lo) σ(w) (`MaterialVar`), the closed-form solves
clamp to it. `jacobian_chunk` (`compute.jacobian_chunk`): the dual partials per Jacobian kernel launch on a device (`_jac_chunk`;
the CPU backend takes every slot in one chunk) — 4, not 15: local memory grows ~1.9 KB per partial, and from about 8 partials the
kernel is bound by spill traffic, so 14 re-traces at 4 beat 4 re-traces at 15 by 4.3× (25,564-px view, RTX 4090, 2026-11-04); 4
also cuts the driver's local-memory reservation (bytes per thread × resident threads) ~5×. `sensor` and `photo_kind`: the photographs'
[`ObservationModel`](@ref) with `channel` (`observation_model(s)`, what the state's records state; by default decoded images, no sensor
record: a state of synthetic renders). `sensor_models`: per view, the photograph's Core `SensorModel` (`photo_set`; the
absent model, `SensorModel()`, by default: a decoded image, or a RAW photograph of a camera stated to have no characterisation). `weights`:
the rows' weights (weighting.jl, `masked_observations`): `Unweighted()` (the default, rows as measured) or one 1/σ vector per view,
each of its `ϕ`'s length (whitened rows: the residuals and the Jacobians divided by the measured values' σ). `quanta`: per view,
each row's RAW code step in its units (`masked_observations`), or `nothing` (a decoded image): what the data resolve
(`lm_abstol`).

The state of the `RadiometricCameraModel` `camera` (one camera: geometric camera, focus surface, aperture radius and vignetting)
seen from `poses` (camera → world, one per photograph). `region` (the run's imaged region, pixels): a tensor vignetting surface's nodes outside it are dependent
(`vignetting_rule`, extrapolated with a zero `boundary_order`-th difference), its parameters the interior ones; a radial profile's nodes with less than `support_min` of the mean data
weight are dependent the same way, continuing the supported ones with that order (`RadialSupportRule`). Builds the masks from the initial views (∩ `external_masks`) unless `masks` is given. Float64 targets are
kept by reference (no copy). When the masked observation vectors `ϕ` are given (built one photo at a time
from `masked_vector`, e.g. for full-resolution photos that do not fit in memory together), `targets` may be
empty placeholder arrays.
"""
function CalibrationState(board::TexturedTarget, settings::RenderSettings, camera::RadiometricCameraModel{<:Any, <:Focus{Float64, N}}, poses::AbstractVector{<:EuclideanMap},
                          mat::Material, light::EnvironmentLight, targets::AbstractVector{<:AbstractArray{<:Real}};
                          masks = nothing, δ::Real = 0.02, external_masks = nothing, support_min::Real = 0.0, region = nothing,
                          boundary_order::Integer = 2, backend = KernelAbstractions.CPU(), ϕ = nothing, channel::Integer = 0,
                          focus_fixed = Int[], mu_bounds = (0.0, 1.0), jacobian_chunk::Integer = 4,
                          sensor_models = fill(VgtkCore.SensorModel(), length(poses)),
                          weights::Union{Unweighted, AbstractVector{<:AbstractVector{<:Real}}} = Unweighted(),
                          quanta = fill(nothing, length(poses)), sensor::SensorStatement = NoSensorRecord(),
                          photo_kind::PhotoKind = Decoded(false)) where {N}
    0 <= channel <= 3 || throw(ArgumentError("CalibrationState: channel = $channel (0: the three channels, 1–3: that channel alone)"))
    ObservationModel(sensor, photo_kind, _channel_symbol(channel))          # the three agree (a RAW photograph's photosites: channel G)
    K = length(poses)
    length(targets) == K || throw(DimensionMismatch("$(K) views but $(length(targets)) targets"))
    poses = [_pose_eltype(Float64, p) for p in poses]
    if masks === nothing
        masks = [compute_mask(camera, poses[k], board, settings; δ, backend,   # the state's device
                              external = external_masks === nothing ? nothing : external_masks[k]) for k in 1:K]
    end
    W, H = _image_size(camera)                                              # the image is the camera's (its geometric camera's extent): stated once
    all(m -> size(m) == (H, W), masks) ||
        throw(ArgumentError("CalibrationState: a mask is not of the camera's image, $H × $W (rows × columns): the image size is the camera's " *
                            "(`extent(geometric(camera))`), the masks and photographs are of it"))
    ϕ === nothing && !all(t -> (size(t, 1), size(t, 2)) == (H, W), targets) &&
        throw(ArgumentError("CalibrationState: a photograph is not of the camera's image, $H × $W (rows × columns): the image size is the " *
                            "camera's (`extent(geometric(camera))`)"))
    px = [masked_pixels(m) for m in masks]
    ϕ === nothing && (ϕ = [masked_vector(targets[k], px[k]; channel) for k in 1:K])   # a plane target (H × W): one measured value per pixel
    tg = Array{Float64, 3}[t isa Array{Float64, 3} ? t : Array{Float64, 3}(ndims(t) == 2 ? reshape(t, size(t)..., 1) : t) for t in targets]
    length(ϕ) == K || throw(DimensionMismatch("$(K) views but $(length(ϕ)) observation vectors"))
    length(sensor_models) == K || throw(DimensionMismatch("$(K) views but $(length(sensor_models)) sensor models"))
    weights isa Unweighted || (length(weights) == K && all(k -> length(weights[k]) == length(ϕ[k]), 1:K)) ||
        throw(DimensionMismatch("the rows' weights are one per measured value of each view"))
    weights = weights isa Unweighted ? weights : [Vector{Float64}(w) for w in weights]
    length(quanta) == K && all(k -> quanta[k] === nothing || length(quanta[k]) == length(ϕ[k]), 1:K) ||
        throw(DimensionMismatch("the code steps are one per measured value of each view (or nothing: a decoded image)"))
    camera.vignetting isa NoVignetting &&                                   # the method estimates V: Θ_v under the gauge V(𝐜) = 1, a block of every solve
        throw(ArgumentError("CalibrationState: the camera has no vignetting (`NoVignetting`: V ≡ 1, no parameters), and the calibration " *
                            "estimates V (Θ_v, under the gauge V(𝐜) = 1). Give the camera a field to estimate: " *
                            "`with_vignetting(camera, flat_vignetting(geometric(camera)))` (the flat 4 × 4 start) or a `RadialVignetting`. " *
                            "A camera without vignetting renders as it is (`render_view`, `Φ_pixel`)"))
    camera = with_vignetting(camera, gauge_center_value(camera.vignetting))   # the gauge V(𝐜) = 1, held exactly: the start's, kept by every step (`gauge_basis`)
    if channel > 0                                                          # the other channels' lighting and η follow channel `channel`
        light = tie_channels(light, channel); mat = tie_channels(mat, channel)
    end
    rule = vignetting_rule(camera.vignetting, region, px; support_min, order = boundary_order)
    return CalibrationState{N, typeof(board), typeof(settings), typeof(camera), typeof(light), eltype(poses)}(
        board, settings, camera, poses, VgtkCore.convert_eltype(Float64, mat), light, masks, px, tg, ϕ, weights, rule, backend, Int(jacobian_chunk), nothing, Int(channel), collect(Int, focus_fixed), zeros(0, 0),
        Float64.(Tuple(mu_bounds)), Union{Nothing, Vector{Float64}}[q === nothing ? nothing : Vector{Float64}(q) for q in quanta],
        collect(VgtkCore.SensorModel, sensor_models), sensor, photo_kind, _LmRun(), nothing)
end
"""
    n_views(s::CalibrationState) -> Int

The number of views (photographs) of the calibration.
"""
n_views(s::CalibrationState) = length(s.poses)
"""
    view_of(s::CalibrationState, k) -> View

Internal: the kernels' view `k`, the state's camera at its pose `k` (`View(s.poses[k], s.camera)`).
"""
view_of(s::CalibrationState, k::Integer) = View(s.poses[k], s.camera)
"""
    geometric(s::CalibrationState)

The geometric camera π⁻¹ (Θ_d) the views share: Core's pixel camera or a grid model.
"""
geometric(s::CalibrationState) = geometric(s.camera)
"""
    focus(s::CalibrationState) -> Focus

The focus surface (Θ_f) the views share.
"""
focus(s::CalibrationState) = focus(s.camera)
"""
    aperture_radius(s::CalibrationState)

The aperture radius r the views share, in metres.
"""
aperture_radius(s::CalibrationState) = aperture_radius(s.camera)
_image_size(s::CalibrationState) = _image_size(s.camera)   # the image is the camera's
"""The board's own state: its warping h, the print's edge parameters and the in-plane print error (VisualRadiometryToolkitCore's
`TargetState`, the state of the state's `TexturedTarget`)."""
board_surface(s::CalibrationState) = s.board.state
"""Replace the board's state (host precision)."""
function set_surface!(s::CalibrationState, surf::TargetState)
    s.board = with_state(s.board, surf)
    return s
end
"""
    target_radiometry(s::CalibrationState) -> TargetRadiometry

The target's radiometry of the state: its material and the print's edge parameters ([`TargetRadiometry`](@ref)).
"""
target_radiometry(s::CalibrationState) = TargetRadiometry(s.mat, board_surface(s))
"""
    radiometric_target(s::CalibrationState) -> RadiometricTarget

The state's calibration target: the board's geometric target (Core's target of its posed target; `nothing` on a board without one),
the warping h with the aspect of the board's page and the target's radiometry ([`RadiometricTarget`](@ref)).
"""
radiometric_target(s::CalibrationState) =
    RadiometricTarget(_geometric_target(s.board.target), board_surface(s).h, warping_aspect(s.board), target_radiometry(s);
                      inplane = board_surface(s).inplane)
_geometric_target(::Nothing) = nothing
_geometric_target(t::VgtkCore.PosedTarget) = VgtkCore.model(t)
"""
    set_radiometric_target!(s::CalibrationState, t::RadiometricTarget) -> s

The state's material, board warping and the print's edge parameters from `t` (its geometric target is the run's board's). A warping
learned on a page of another aspect than the run's board is an `ArgumentError` ([`board_surface`](@ref)).
"""
function set_radiometric_target!(s::CalibrationState, t::RadiometricTarget)
    surf = board_surface(s.board, t)
    s.mat = VgtkCore.convert_eltype(Float64, material(t.radiometry))
    return set_surface!(s, surf)
end
"""Replace the camera's vignetting V (Θ_v)."""
set_vignetting!(s::CalibrationState, v::AbstractVignetting) = (s.camera = with_vignetting(s.camera, v); s)

"""
    set_pose!(s::CalibrationState, k, pose::EuclideanMap) -> s

Replace the pose Θ_e of photograph `k` (Core's `EuclideanMap`, camera → world, metres); the camera is kept.
"""
function set_pose!(s::CalibrationState, k::Integer, pose::EuclideanMap)
    s.poses[k] = _pose_eltype(Float64, pose)
    return s
end

"""Render every view at the current parameters (`Vector` of `H × W × 3` images): the flux kernel on the state's backend."""
render_all(s::CalibrationState) = [_render_view_device(s, k) for k in 1:n_views(s)]

"""Residual vectors `𝔯_k = Φ_k − ϕ_k` in the measured values' units (unweighted). One view at a time, straight from the masked flux
vector of the flux kernel on the state's backend (`_flux_device`, same pixel order as `masked_vector`): a full-size H × W × 3 image per
view (290 MB each at 4256 × 2848) is what pushed the 16-view full-resolution runs over the container cap. The diagnostics' residuals
(the model is `ϕ + 𝔯`); the cost's are `residuals(s)` (Core's `residuals`)."""
photometric_residuals(s::CalibrationState) =
    [_channel_rows(_flux_device(s, k, view_of(s, k), s.camera.vignetting, s.mat, s.light), s.channel) - s.ϕ[k] for k in 1:n_views(s)]

"""The cost's residual vectors: [`photometric_residuals`](@ref) under the rows' weights (`_whiten`: whitened when the state's rows
carry 1/σ, as measured otherwise)."""
residuals(s::CalibrationState) = [_whiten(_view_weights(s.weights, k), r) for (k, r) in enumerate(photometric_residuals(s))]

"""Cost `½ Σ_k ‖𝔯_k‖²` of the cost's residual vectors (`residuals(s)`: whitened when the rows carry 1/σ) and the relative
RMS residual `√(Σ_k ‖𝔯_k‖² / data_norm2(s))`, in the same norm."""
function cost(s::CalibrationState)
    r = residuals(s)
    sse = sum(sum(abs2, rk) for rk in r)
    return sse / 2, sqrt(sse / data_norm2(s))
end
