# ---- the enumerated choices and their registries ---------------------------------------------------------------------------------
"""The start geometry of a refinement: [`BlobFrontend`](@ref) or [`SchoepsFrontend`](@ref) (tags in [`FRONTEND_TAGS`](@ref))."""
abstract type AbstractFrontend end
"""A calibration session's VisualGeometryToolkitIO records (`frontend.session`: `camera.json`, `targets_1.json`, `trajectory_1.json`,
`target_poses.json`, `textures/` — `load_session`; the board's target and its printed image are among them) and its detections
(`frontend.session`: its records and its observations record observations.json). Tag `blob`."""
struct BlobFrontend <: AbstractFrontend end
"""A Schöps `central-generic` run on a printed page (`frontend.schoeps`: intrinsics0.yaml, dataset.bin; the print
from `frontend.texture`, `frontend.texture_features`, `frontend.page_mm`). Tag `schoeps`."""
struct SchoepsFrontend <: AbstractFrontend end

"""The front-end registry: `frontend.type` tag → [`AbstractFrontend`](@ref) type."""
const FRONTEND_TAGS = TagRegistry(:frontend, "Frontend", "blob" => BlobFrontend, "schoeps" => SchoepsFrontend;
                                  paths = [r"frontend\.type"])

# The vignetting field of a refinement (`vignetting.model`) is the radial profile, the tag `radial` of VisualGeometryToolkitCore's
# `VIGNETTING_TAGS` (`RadialVignetting`, `V(𝐩) = Σ N_{i,p}(ρ(‖𝐩 − 𝐜‖)) P_i` about the camera's principal point 𝐜,
# `vignetting.node_count` nodes of degree `vignetting.degree`, the first two tied: smooth at the centre). Every other field of Core's
# registry — the tensor-product surface, the `log` field — is refused (`_vignetting_model`).
"""Core's `VIGNETTING_TAGS` with the key path the pipeline's configs carry its tags at (`vignetting.model`): Core's tags, retired
`tensor` included, and no tag of the package's own."""
const VIGNETTING_CONFIG_TAGS = merge(VIGNETTING_TAGS, TagRegistry(:vignetting, "Vignetting"; paths = [r"vignetting\.model"]))

"""The blocks `fixed` may hold (the options' top-level `fixed`): the poses, the camera (the geometric camera's parameters — a grid's
lattice or a parametric camera's lens block), the focus surface, the aperture radius, the material (`"material"`: Θ_m, the 8 nonlinear
parameters and μ — the block's free board warping and in-plane print error stay free, `_material_rides`), the vignetting, the
lighting, the in-plane print error. `"grid"` is renamed `"camera"`; `"edge"` (never a parameter) is refused."""
const FIXABLE_BLOCKS = ("poses", "camera", "focus", "aperture", "material", "vignetting", "lighting", "inplane")
"""`warping.inplane`: the in-plane print error of the board — none, or Core's `Aspect ∘ Skew` (`inplane_calibration`)."""
const INPLANE_KINDS = ("none", "aspect_skew")

# ---- the defaults: configs/default.toml, THE one statement of every option's default -------------------------------------------
# Every table field's keyword default reads the file (`_option_default`), parsed at precompile into a constant by Core's
# `parse_defaults` (its `include_dependency` makes an edit of the file recompile the package): no second set of Julia literals to
# drift from it, no sync test standing in for the design (owner D8, 2026-11-15; Estimation's config_defaults.jl, BlobBoards' before
# it). A default absent from the file is an error naming it (Core's `toml_default`), by design. The fields whose default is ABSENT
# (`io.lens_identity`, `grid.start_perturbation`: no value until a config states one)
# are `nothing`, which TOML cannot state; the file documents them in a comment.
using VisualGeometryToolkitCore: toml_default, parse_defaults
# The parsed file: the package's own defaults (`pkgdir(RadiometricCamera, "configs", "default.toml")`), read through Core's one
# `parse_defaults`/`toml_default`, as every stack package's file is (DUPLICATION.toml: the name is each package's own data).
const _TOML_DEFAULTS = parse_defaults(normpath(joinpath(@__DIR__, "..", "..", "configs", "default.toml")))
"""The default of the field `f` of the options table `T`: its configs/default.toml value, read as the TOML reader reads a config's."""
_option_default(::Type{T}, f::Symbol) where {T} = (t = _table_name(T); k = toml_key(T, f); _option_from_toml(T, Val(f), toml_default(_TOML_DEFAULTS, t, k), "$t.$k"))
"""The TOML table of the options table type `T` (its field of `PipelineOptions`)."""
_table_name(::Type{T}) where {T} = String(only(n for n in fieldnames(PipelineOptions) if fieldtype(PipelineOptions, n) === T))

# ---- the tables ------------------------------------------------------------------------------------------------------------------
"""A table of [`PipelineOptions`](@ref): compared field by field."""
abstract type AbstractOptionsTable end

"""`io.lens_identity`: the lens a capture was taken with, as the config states it (Core's `DeviceIdentity`; an empty field is
unknown, and at least one is stated). It overrides the lens the RAW files' EXIF claims (Features' `extract_capture_settings`)."""
Base.@kwdef struct LensIdentityOptions <: AbstractOptionsTable
    make::String = ""
    model::String = ""
    serial::String = ""
end
"""The `DeviceIdentity` the options' `io.lens_identity` states (VisualGeometryToolkitFeatures' `lens_identity` of a RAW states the EXIF's)."""
lens_identity(o::LensIdentityOptions) = VgtkCore.DeviceIdentity((isempty(s) ? nothing : s for s in (o.make, o.model, o.serial))...)

"""`[io]`: the photographs, the output directory, the start model, the RAW files and the diagnostic modes that write files and return."""
Base.@kwdef struct IOOptions <: AbstractOptionsTable
    images::String = _option_default(IOOptions, :images)                               # the photographs (16-bit linear PNG unless `linearize`); required
    out::String = _option_default(IOOptions, :out)                                  # the output directory; required
    linearize::Bool = _option_default(IOOptions, :linearize)                           # decode sRGB photographs
    channel::String = _option_default(IOOptions, :channel)                           # one colour channel ("R" | "G" | "B") copied to all three; "RGB" = as photographed. A RAW
                                                     #   photograph (by its file: `load_photo`) is read at its green photosites: "G"
    views::Vector{String} = _option_default(IOOptions, :views)                  # image names (a subset; empty = all)
    init::String = _option_default(IOOptions, :init)                                 # grid:<json> | lens:<json> replaces the front-end's start model
    checkpoint_each::Bool = _option_default(IOOptions, :checkpoint_each)                     # per-iteration lens / poses / state files
    render_init::Bool = _option_default(IOOptions, :render_init)                         # render the start state and return
    render_state::String = _option_default(IOOptions, :render_state)                         # render the masked pixels with this state file (state.json) and return
    raw::String = _option_default(IOOptions, :raw)                                  # the photographs' RAW files (blob front-end): the capture's camera settings from their EXIF
    lens_identity::Union{Nothing, LensIdentityOptions} = nothing   # {make, model, serial}: the lens, where EXIF misnames it (an adapted lens)
end

"""`[frontend]`: the start geometry (`type` a tag of [`FRONTEND_TAGS`](@ref)) and its inputs."""
Base.@kwdef struct FrontendOptions <: AbstractOptionsTable
    type::Type{<:AbstractFrontend} = _option_default(FrontendOptions, :type)    # blob | schoeps
    session::String = _option_default(FrontendOptions, :session)                              # blob: the session's records {camera.json, targets_1.json, trajectory_1.json,
                                                     #   target_poses.json, textures/} and its observations record observations.json (vgio/observations)
    schoeps::String = _option_default(FrontendOptions, :schoeps)                              # schoeps: the central-generic run {intrinsics0.yaml, dataset.bin}
    texture::String = _option_default(FrontendOptions, :texture)                              # schoeps: the printed design (PNG, gamma 1)
    texture_features::String = _option_default(FrontendOptions, :texture_features)                     # schoeps: dataset.bin of the detector run on the texture
    texture_features_size::Tuple{Int, Int} = _option_default(FrontendOptions, :texture_features_size)    # the texture size the feature file was detected on
    texture_downsample::Int = _option_default(FrontendOptions, :texture_downsample)                       # box-average of the texture
    page_mm::Tuple{Float64, Float64} = _option_default(FrontendOptions, :page_mm)        # schoeps: page size [mm]
    print_scale::Float64 = _option_default(FrontendOptions, :print_scale)                        # the print's scale relative to the design
    pose_distances::Vector{Float64} = _option_default(FrontendOptions, :pose_distances)    # schoeps: the multi-start pose fit's distances [m]
end

"""`[camera]`: the nominal camera, the aperture bounds, the lens coverage and the sensor characterisation."""
Base.@kwdef struct CameraOptions <: AbstractOptionsTable
    sensor_mm::Float64 = _option_default(CameraOptions, :sensor_mm)                         # sensor width [mm] (a length quantity in the pipeline: `_sensor_width`)
    f_mm::Float64 = _option_default(CameraOptions, :f_mm)                              # nominal focal length [mm] (the nominal pinhole of a grid start; the aperture start)
    fnumber::Float64 = _option_default(CameraOptions, :fnumber)                            # r₀ = f / (2N)
    lens_coverage::String = _option_default(CameraOptions, :lens_coverage)              # where light falls: "sensor" (the whole sensor, nothing
                                                     #   measured) or the path of the capture set's measured image region (an
                                                     #   rc/image_region/v1.0 record: `run_lens_coverage`)
    sensor_characterisation::String = _option_default(CameraOptions, :sensor_characterisation)              # the body's sensor characterisation (an IO `vgio/sensor_characterisation` record, Core's
                                                     #   `SensorCharacterisation`), each RAW photograph's `SensorModel` resolved from it by its
                                                     #   EXIF capture settings (`photo_set`); empty = none named (a path that does not exist is
                                                     #   refused; "none" is retired: `sensor_model = "off"`)
    sensor_model::String = _option_default(CameraOptions, :sensor_model)              # whether the run uses the sensor model (SENSOR_MODEL_TAGS, parsed
                                                     #   once by `sensor_model_mode`): "auto" (a record present is used, none: the file's black),
                                                     #   "on" (a record required), "off" (a record present is NOT used: a loud warning)
end
"""The RETIRED value of `camera.sensor_characterisation` that stated there is none (RC ≤ 0.11.1): refused since 0.11.2, naming
`camera.sensor_model = "off"` (a record present, not used) — a run that names no record leaves the key empty."""
const NO_SENSOR_CHARACTERISATION = "none"
"""The `camera.lens_coverage` that states a lens lighting the whole sensor: the coverage is the sensor's rectangle (Core's default
lens coverage), nothing is measured."""
const SENSOR_LENS_COVERAGE = "sensor"


"""`grid.start_perturbation`: the start grid's direction controls rotated by `sigma_px / f px` radians RMS per tangent axis, the
draws from `StableRNG(seed)` (`perturb_start`, grid_smoothness.jl). Both keys are required: there is no default perturbation."""
Base.@kwdef struct StartPerturbationOptions <: AbstractOptionsTable
    sigma_px::Float64                                # σ [px]: the rotation per tangent axis is σ / f px radians RMS
    seed::Int                                        # the seed of the draws (StableRNG)
end

"""`[grid]`: the grid model's control directions, the smoothness of its interior nodes and the start's perturbation (grid_smoothness.jl)."""
Base.@kwdef struct GridOptions <: AbstractOptionsTable
    lattice::LatticeRule = _option_default(GridOptions, :lattice)               # "derived" (the source's own lattice, m = 1: `exact_start`) | "node_count"
                                                     #   (over the image region: `region_start`), `LATTICE_TAGS`
    node_count::Union{Nothing, Int} = nothing        # lattice = "node_count" only: the nodes the image region supports (`region_start`); absent = none
    smoothness::AbstractSmoothness = _option_default(GridOptions, :smoothness)    # [grid.smoothness]: pspline (its Δ³) | bending | none, its weight β relative to
                                                     #   the data energy Σϕ² per node, or "gcv": chosen by GCV on the linearization at the start
                                                     #   (`select_grid_smoothness_weight`, 2026-11-04)
    nodes::Symbol = _option_default(GridOptions, :nodes)                         # the nodes the lens holds, its parameters: interior (those the imaged region reaches) |
                                                     #   all (`GRID_NODE_RULES`, `grid_present`)
    start_perturbation::Union{Nothing, StartPerturbationOptions} = nothing   # {sigma_px, seed}: the start perturbed; absent = none
end

"""`[focus]`: EXPERIMENT (focus-ab, focus_ab.jl) — the parameterisation of the focus surface."""
Base.@kwdef struct FocusOptions <: AbstractOptionsTable
    variant::String = _option_default(FocusOptions, :variant)                             # A | B | C | D (focus_ab.jl's header)
    node_count::Int = _option_default(FocusOptions, :node_count)                          # C / D: node count of the cap (0 = A's node count: the support nodes of the 5 × 5 grid, 25 without an image circle)
    lens_coverage_scale::Float64 = _option_default(FocusOptions, :lens_coverage_scale)    # A / B: the grid's nodes kept inside the lens coverage scaled by this (`_scaled_coverage`, as [masks]; `focus_start_nodes`)
    smoothness::AbstractSmoothness = _option_default(FocusOptions, :smoothness)    # [focus.smoothness]: bending ½ β (Σϕ²/N) t̂ᵀQ̃t̂ (weight 0 = none) | none
    support_min::Float64 = _option_default(FocusOptions, :support_min)                    # A / B: a node with less than this share of the mean data weight Σ b_i(𝐩)² is dependent (`focus_node_support`; 0 = every node with a pixel)
    perturb::Vector{Float64} = _option_default(FocusOptions, :perturb)              # [σ, seed]: the start's free nodal values × exp(σ z), z ~ N(0, 1); empty = none
end

"""`[masks]`: the samples of the run (the LM's; the closed-form start runs on the same pixels): `samples_per_view` of each view's usable observations."""
Base.@kwdef struct MaskOptions <: AbstractOptionsTable
    samples_per_view::Int = _option_default(MaskOptions, :samples_per_view)              # the samples per view, drawn from its usable observations (`draw_samples`, 2026-11-15)
    gradient_bias::Float64 = _option_default(MaskOptions, :gradient_bias)                # α: each usable observation drawn ∝ u^α, u the ECDF rank of its edge
                                                     #   strength (`draw_samples`); 0 = uniform, > 0 favours edges, < 0 flat regions
    edge_detector::AbstractEdgeDetector = _option_default(MaskOptions, :edge_detector)   # [masks.edge_detector]: the edge strength (`edge_detector_from_toml`):
                                                     #   gaussian_derivative {sigma}
    edge_fraction::Float64 = _option_default(MaskOptions, :edge_fraction)                # q ∈ [0, 1): > 0 the rule of 2026-10-31 — the strongest-|∇ϕ|² share q of a
                                                     #   uniform pool (`draw_samples`; with gradient_bias = 0); 0 = the weighted draw
    addon_samples::Int = _option_default(MaskOptions, :addon_samples)                    # M ≥ 0: M samples per view more, drawn ∝ u^β from the other usable
                                                     #   observations (`draw_samples`' add-on); 0 = none
    addon_gradient_bias::Float64 = _option_default(MaskOptions, :addon_gradient_bias)    # β of the add-on (only with addon_samples > 0)
    seed::Int = _option_default(MaskOptions, :seed)                                     # the masks' random seed
    lens_coverage_scale::Float64 = _option_default(MaskOptions, :lens_coverage_scale)   # the photometric pixels are sampled inside the lens coverage
                                                     #   scaled about its centre by this factor (an image circle's semi-axes; the sensor's
                                                     #   rectangle is not scaled): sampling only — the camera's coverage, its nodes, the
                                                     #   vignetting chart are the measured ones (`_sampling_region`)
end

"""`[rendering]`: the quadrature of the image formation (`RenderSettings`)."""
Base.@kwdef struct RenderingOptions <: AbstractOptionsTable
    N::Int = _option_default(RenderingOptions, :N)                                        # radiance samples
    n_rho::Int = _option_default(RenderingOptions, :n_rho)                                    # aperture quadrature, radial
    n_phi::Int = _option_default(RenderingOptions, :n_phi)                                    # aperture quadrature, angular
    n_p::Int = _option_default(RenderingOptions, :n_p)                                      # sensor samples per pixel side
    hammersley::Bool = _option_default(RenderingOptions, :hammersley)                          # the N radiance samples from the Hammersley set, shifted per key (`RadianceSampler`), instead of
                                                     #   pseudo-random draws: the same N, about half the per-pixel noise (2026-10-31)
end

"""`[lighting]`: the lighting model L^SVSH."""
Base.@kwdef struct LightingOptions <: AbstractOptionsTable
    sh_order::Int = _option_default(LightingOptions, :sh_order)                                 # l_max of L^SVSH
end

"""`[material]`: the start of Θ_m and its handling."""
Base.@kwdef struct MaterialOptions <: AbstractOptionsTable
    alpha_ink::Float64 = _option_default(MaterialOptions, :alpha_ink)                         # α_ink start
    alpha_noink::Float64 = _option_default(MaterialOptions, :alpha_noink)                        # α_¬ink start
    eta_ink::Float64 = _option_default(MaterialOptions, :eta_ink)                            # η_ink start
    eta_noink::Float64 = _option_default(MaterialOptions, :eta_noink)                         # η_¬ink start
    mu::Float64 = _option_default(MaterialOptions, :mu)                                 # μ start
    mu_bounds::Tuple{Float64, Float64} = _option_default(MaterialOptions, :mu_bounds)   # the interval (lo, hi) of μ: the LM steps on the chart μ = lo + (hi − lo) σ(w) (2026-10-31; before a box)
end

"""`[vignetting]`: the vignetting field, the radial profile about the principal point (`model`: Core's tag `radial`)."""
Base.@kwdef struct VignettingOptions <: AbstractOptionsTable
    model::Type{<:RadialVignetting} = _option_default(VignettingOptions, :model)    # "radial", the one field a refinement fits (the tensor-product surface is no option since
                                                     #   2026-11-14); the key is accepted so the configs of record, which all state it, still load
    node_count::Int = _option_default(VignettingOptions, :node_count)                                # the profile's nodes (the first two tied: one parameter fewer)
    degree::Int = _option_default(VignettingOptions, :degree)                                   # spline degree
    support_min::Float64 = _option_default(VignettingOptions, :support_min)                        # a node with less than this share of the mean data weight Σ N_i(ρ)² is dependent
                                                     #   (continues the supported ones with `boundary_order`; 0 = any masked pixel supports; 2026-10-30, the order 2026-11-07)
    smoothness::AbstractSmoothness = _option_default(VignettingOptions, :smoothness)     # [vignetting.smoothness]: pspline, the closed-form solve's Δ³ penalty over the free
                                                     #   nodes (`solve_Θ_v!`; its weight normalized as the grid's; 0 = none; "gcv" = chosen
                                                     #   by GCV per solve, 2026-11-04) | none
    boundary_order::Int = _option_default(VignettingOptions, :boundary_order)                           # the dependent nodes' extrapolation past the supported nodes, 2 Δ²c = 0 | 3 Δ³c = 0 (`BOUNDARY_ORDERS`)
    chart::Symbol = _option_default(VignettingOptions, :chart)                           # the chart of the normalized radius (Core's `RADIAL_CHARTS`, 2026-11-10): linear ρ = d / r_max, uniform
                                                     #   knots over the imaged region | projective ρ = d / (r_max + d) (Core's `RadialVignetting(…; chart = :projective)`: the linear chart's pixel
                                                     #   spacing up to r_max and p − 1 knots past it at that spacing, then the span to 1 (Core 0.10.11): no pixel is ever clamped)
end

"""`[warping]`: the board warping h (thin plate spline through nodal heights) and the in-plane print error (Core's `Aspect ∘ Skew`)."""
Base.@kwdef struct WarpingOptions <: AbstractOptionsTable
    degree::Int = _option_default(WarpingOptions, :degree)                                   # > 0: the board warping h free (grid route)
    heights_mm::Vector{Float64} = _option_default(WarpingOptions, :heights_mm)           # start / fixed nodal heights [mm]
    inplane::String = _option_default(WarpingOptions, :inplane)                          # "aspect_skew": the in-plane print error (a, u) free (INPLANE_KINDS; "inplane" in the top-level `fixed` holds it)
    inplane_start::Vector{Float64} = _option_default(WarpingOptions, :inplane_start)        # start (a, u) of the in-plane print error: the log-stretch of x against y and the shear; [] = as designed
end

"""`[edge]`: the printed-edge albedo."""
Base.@kwdef struct EdgeOptions <: AbstractOptionsTable
    sdf::String = _option_default(EdgeOptions, :sdf)                                  # the design's distance map: a vgio/raster record (distance_map.json); empty = no edge model
    ink_spread_mm::Float64 = _option_default(EdgeOptions, :ink_spread_mm)                      # the ink spread δ [mm], held
    print_sigma_mm::Float64 = _option_default(EdgeOptions, :print_sigma_mm)                   # the print's blur σ_print [mm], the edge's width, held
end

"""`[lm]`: the Levenberg–Marquardt over Θ̄."""
Base.@kwdef struct LMOptions <: AbstractOptionsTable
    max_iter::Int = _option_default(LMOptions, :max_iter)                                # iterations at most
    max_time::Float64 = _option_default(LMOptions, :max_time)                       # wall-clock budget [s]
    ftol::Float64 = _option_default(LMOptions, :ftol)                               # Core's LmStops ftol: a step's relative cost decrease below it stops (CostDecrease)
    xtol_rel::Float64 = _option_default(LMOptions, :xtol_rel)                       # Core's LmStops xtol_rel: ‖Δx‖ ≤ xtol_rel·(‖x‖ + xtol_rel) stops (IncrementSize; Ceres' parameter tolerance)
    log::String = _option_default(LMOptions, :log)                                  # EXPERIMENT (nlls5): a per-iteration line appended to this file (`_lm_log`); empty = none
end

"""`[sweep]`: the closed-form (lighting, vignetting, μ) sweep between LM chunks."""
Base.@kwdef struct SweepOptions <: AbstractOptionsTable
    lm_chunk::Int = _option_default(SweepOptions, :lm_chunk)                                 # LM iterations between sweeps (0 = no sweeps)
    iterations::Int = _option_default(SweepOptions, :iterations)                               # block-solve iterations of one sweep
    joint::Bool = _option_default(SweepOptions, :joint)                               # the closed-form blocks jointly (solve_radiometry_gn!)
end

"""`[resume]`: continuing a finished run."""
Base.@kwdef struct ResumeOptions <: AbstractOptionsTable
    state::String = _option_default(ResumeOptions, :state)                                # continue from this state file (state.json)
    iteration_offset::Int = _option_default(ResumeOptions, :iteration_offset)                         # iteration numbering of the resumed run
    sweep::Bool = _option_default(ResumeOptions, :sweep)                               # closed-form sweep on the resumed state
end

"""`[selfcheck]`: the bearing self-check of the detections through the start model."""
Base.@kwdef struct SelfcheckOptions <: AbstractOptionsTable
    only::Bool = _option_default(SelfcheckOptions, :only)                                # write selfcheck.json / selfcheck_vectors.json and return
    poses::String = _option_default(SelfcheckOptions, :poses)                                # score under this state's poses instead of the refit
end

"""`[compute]`: the device and the GPU Jacobian's batch."""
Base.@kwdef struct ComputeOptions <: AbstractOptionsTable
    backend::Type{<:AbstractComputeBackend} = _option_default(ComputeOptions, :backend)     # gpu | cpu (gpu falls back to the CPU without a functional GPU)
    jacobian_chunk::Int = _option_default(ComputeOptions, :jacobian_chunk)                           # dual partials per GPU kernel launch
end

"""
    PipelineOptions(; io, frontend, camera, grid, masks, rendering, lighting, material, vignetting, warping, edge, lm, sweep, resume,
                    selfcheck, compute)

The options of a refinement, one table each (see the table structs for every key and its default; `configs/default.toml` lists
them). Read from TOML with [`pipeline_options`](@ref), written for the run record with [`config_dict`](@ref). Required: `io.images`,
`io.out` and the front-end's inputs (`frontend.type = "blob"`: `session`; `"schoeps"`: `schoeps`, `texture`, `texture_features`).
A `selfcheck.only` run reads no photograph: it needs no `io.images`.
"""
Base.@kwdef struct PipelineOptions
    fixed::Vector{String} = String.(toml_default(_TOML_DEFAULTS, "", "fixed"))   # the blocks held at their start (`FIXABLE_BLOCKS`); none by default
    io::IOOptions = IOOptions()
    frontend::FrontendOptions = FrontendOptions()
    camera::CameraOptions = CameraOptions()
    grid::GridOptions = GridOptions()
    focus::FocusOptions = FocusOptions()
    masks::MaskOptions = MaskOptions()
    rendering::RenderingOptions = RenderingOptions()
    lighting::LightingOptions = LightingOptions()
    material::MaterialOptions = MaterialOptions()
    vignetting::VignettingOptions = VignettingOptions()
    warping::WarpingOptions = WarpingOptions()
    edge::EdgeOptions = EdgeOptions()
    lm::LMOptions = LMOptions()
    sweep::SweepOptions = SweepOptions()
    resume::ResumeOptions = ResumeOptions()
    selfcheck::SelfcheckOptions = SelfcheckOptions()
    compute::ComputeOptions = ComputeOptions()
end

const _OptionsValue = Union{AbstractOptionsTable, PipelineOptions}
Base.:(==)(a::T, b::T) where {T <: _OptionsValue} = all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
Base.isequal(a::T, b::T) where {T <: _OptionsValue} = all(isequal(getfield(a, f), getfield(b, f)) for f in fieldnames(T))
Base.hash(a::T, h::UInt) where {T <: _OptionsValue} = foldl((h, f) -> hash(getfield(a, f), h), fieldnames(T); init = hash(T, h))
