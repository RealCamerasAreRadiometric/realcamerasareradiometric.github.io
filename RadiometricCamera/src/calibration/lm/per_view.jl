# ---------------------------------------------------------------------------
# Nonlinear least squares (Levenberg–Marquardt, NLLSsolver.jl) over ALL parameters
# Θ̄ = (Θ_e, Θ_d, Θ_f, r, Θ_m, Θ_v, Θ_l) of the per-view parametric route: the blocks that enter Φ
# linearly (Θ_v, Θ_l, μ) are ordinary LM variables too, so every step uses the coupled Gauss–Newton
# curvature JᵀJ instead of alternating between the closed-form solves and LM
# (the closed-form sweep of closed_form.jl remains available as initialiser).
#
# Variable blocks (NLLSsolver duck-typed interface: `nvars`, `update`):
#   the pose (6, per view: Core's LeftRigidManifold6)   the lens (`lens_manifold`: Core's ParameterManifold of the pixel camera, boxed, scaled, restricted)   focus_block (N, Core's restricted ParameterManifold)   ApertureVar (1)
#   MaterialVar (9 = 8 nonlinear + μ; in the grid route also the board warping h and the printed-edge parameters of the board's
#   TargetState)   Θ_v: Core's restrict(ParameterManifold(vig), Z) (the V(0)=1 gauge held exactly: `gauge_basis`)   LightingVar (3J)
# Cost block: one `PerViewResidual` per view with the per-pixel residual
#   𝔯 = Φ_c(𝐮; Θ̄) − ϕ_c(𝐮)  over the mask (`computeresidual`), and a custom
# `computeresjac` building the Jacobian with ForwardDiff.jl dual numbers seeded on
# (Θ_e, Θ_d, Θ_f, Θ_m) and on r through the fixed unit-disc rule r² w̃_l L_c(·, r â_l) (paper: no boundary
# term — the r-dependence of the aperture domain sits in the integrand). The lighting, vignetting and μ columns are
# analytic (Φ is linear in Θ_l and Θ_v and affine in μ); the printed-edge parameters by finite differences in the grid
# route (per_tile.jl). NLLSsolver then forms g = Jᵀ𝔯 and H = JᵀJ;
# the extrinsics are block diagonal by construction (each cost block links only its own pose variable).
# ---------------------------------------------------------------------------

# The pose block of a view: VisualGeometryToolkitCore's `LeftRigidManifold6` on its camera → world map, an NLLSsolver variable through
# Core's manifold adapter (6 coordinates (ω, ρ); `update` re-anchors with Core's `recenter`); its pose is `model(block)`.

# The lens block of a parametric model: VisualGeometryToolkitCore's `ParameterManifold` of its pixel camera's model with the LM's search
# box (`lm_box`) and the image-size coordinate scale (`lm_scale`), `restrict`ed to the free parameters (`lens_free`) — an NLLSsolver
# variable through Core's manifold adapter (`update` = Core's `recenter`, which clamps the step into the box).
const _LensBlock = VgtkCore.SubsetManifold{<:VgtkCore.ParameterManifold{<:CameraModel}}

"""
    lens_free(camera; fixed = ()) -> (names, held)

The free parameters of a parametric model's LM block, by Core's parameter names of its pixel camera, in the camera's order: every
parameter of `parameter_names(model(camera))` (`(:f, :aspect, :skew, :cx, :cy, …the lens's)`) but the skew (the gauge, held always)
and those `fixed` names — the stack's one parameter-holding keyword, expanded by Core's `free_parameter_names` (parameter names or
groups such as `:k`, `:p`; an entry that names nothing is an `ArgumentError`). Holding `:aspect` (f_y/f_x) moves the two focal
lengths together at their ratio. Holding `cx, cy` removes the principal-point ↔ rotation gauge of narrow-FOV lenses.
`held`: no parameter is free (`fixed = parameter_names(model(camera))`, a Core lens to be sampled into a grid) — the block then keeps
the default names, which NLLSsolver requires to be non-empty, and `_lm_views!` holds it fixed. The grid lenses have no such block:
they are refined per tile (`_lm_tiles!`).
"""
function lens_free(c::_PixelCamera; fixed = ())
    m = VgtkCore.model(c)
    fx = Tuple(Symbol.(fixed))
    names = VgtkCore.free_parameter_names(m, fx)
    held = isempty(names)
    held && (names = VgtkCore.free_parameter_names(m, ()))
    return names, held
end
lens_free(lens; kwargs...) =
    throw(ArgumentError("$(nameof(typeof(lens))): a grid lens has no per-view lens block; use the per-tile grid route (build_problem_grid / _lm_tiles!)"))

"""
    lens_manifold(camera; fixed = ()) -> SubsetManifold

The per-view lens block of a parametric model (Core's pixel camera): VisualGeometryToolkitCore's `ParameterManifold` of its model —
each parameter moved by the chart that keeps it inside its `parameter_bounds` (f and the aspect through a log, the double sphere's ξ
through a log and α through a logit, an unbounded coefficient additively) — with the LM's search box `lm_box` (Core clamps every step
into it) and the image-size coordinate scale `lm_scale`, `restrict`ed to `lens_free`'s names. Its model is `VgtkCore.model(block)`.
"""
function lens_manifold(c::_PixelCamera; fixed = ())
    m = VgtkCore.model(c); W, H = VgtkCore.extent(c)
    return VgtkCore.restrict(VgtkCore.ParameterManifold(m; box = lm_box(m, W, H), scale = lm_scale(W, H)), first(lens_free(c; fixed)))
end
lens_manifold(lens; kwargs...) = lens_free(lens; kwargs...)
"""
    _seeded_lens(block, bound, cst, seed) -> camera

The camera of a Jacobian evaluation: Core's `ParameterManifold` of the block's model (its coordinate scale) with its parameters as
constants of the evaluation's dual type (`cst`; K converted entry by entry), retracted at the free coordinates seeded by `seed(j)` (`j` = 1 …
the free count, the block's coordinates in the camera's parameter order) and the fixed ones at zero — the coordinates
`NLLSsolver.update` steps in — bounded like `bound` (the view's camera).
"""
function _seeded_lens(v::_LensBlock, bound::_PixelCamera, cst, seed)
    m = VgtkCore.model(v)
    K = SMatrix{3, 3}(m.intrinsics.K)
    θl = VgtkCore.parameter_vector(m.lens)
    mc = CameraModel(LogicalIntrinsics(CameraCalibrationMat(map(cst, K))), VgtkCore.with_parameters(m.lens, map(cst, θl)))
    names = VgtkCore.parameter_names(m)
    sc = VgtkCore.coordinate_scale(v.manifold)
    z = cst(0.0)
    full = SVector(ntuple(i -> (j = findfirst(==(i), v.idx); j === nothing ? z : seed(j)), Val(length(names))))
    V = typeof(ForwardDiff.value(z))                     # the scale in the evaluation's value type: a Float32 kernel stays Float32
    ch = sc === nothing ? VgtkCore.ParameterManifold(mc) : VgtkCore.ParameterManifold(mc; scale = NamedTuple{names}(Tuple(V.(sc))))
    return VgtkCore.with_model(bound, VgtkCore.retract(ch, full))
end

# (Core's parameter protocol of a focus surface — its nodal values t̂ᵢ, rebuilt through its host-side spline system — is
# VisualRadiometryToolkitCore's, with the type)
"""
    focus_block(f) -> VgtkCore.SubsetManifold

The focus block of the LM: Core's `restrict(ParameterManifold(f; box), free)` of the focus surface `f` — its nodal values stepped
additively and clamped at 0 (the box, `t̂ᵢ ≥ 0`), every node free. Its model is the focus (`VgtkCore.model`).
"""
function focus_block(f::Focus{T, N}) where {T, N}
    box = NamedTuple{VgtkCore.parameter_names(f)}(ntuple(_ -> (0.0, Inf), N))
    return VgtkCore.restrict(VgtkCore.ParameterManifold(f; box), SVector{N, Bool}(ntuple(_ -> true, N)))
end
"""The focus block's type (`focus_block`): Core's restricted `ParameterManifold` of a `Focus`."""
const _FocusBlock = VgtkCore.SubsetManifold{<:VgtkCore.ParameterManifold{<:Focus}}

"""
    FocusSmoothnessResidual(i_f, L)

The bending-energy smoothness of the focus surface as residual rows `L t̂` (`L` already carries √weight). Depends on
the focus block only; its Jacobian is `L`.
"""
struct FocusSmoothnessResidual{N} <: NLLSsolver.AbstractResidual
    idx::SVector{1, Int}
    L::Matrix{Float64}
end
NLLSsolver.ndeps(::FocusSmoothnessResidual) = static(1)
NLLSsolver.nres(r::FocusSmoothnessResidual) = size(r.L, 1)
NLLSsolver.varindices(r::FocusSmoothnessResidual) = r.idx
NLLSsolver.getvars(r::FocusSmoothnessResidual{N}, vars::Vector) where {N} = (vars[r.idx[1]]::_FocusBlock,)
NLLSsolver.computeresidual(r::FocusSmoothnessResidual, f::_FocusBlock) = r.L * collect(VgtkCore.model(f).t̂_i)
NLLSsolver.computeresjac(varflags, r::FocusSmoothnessResidual, f::_FocusBlock) = (NLLSsolver.computeresidual(r, f), r.L[:, collect(f.idx)])
"""Add the focus smoothness `L` (the state's `focus_smoothness`, `√w L` with `‖L t̂‖² = t̂ᵀQ̃t̂`; empty = none) to a
problem's residual blocks `blocks`."""
function _add_focus_smoothness!(blocks, i_f, N, L::AbstractMatrix)
    isempty(L) && return blocks
    size(L, 2) == N || error("focus smoothness: $(size(L, 2)) columns for $N nodes")
    push!(blocks, FocusSmoothnessResidual{N}(SVector(Int(i_f)), Matrix{Float64}(L)))
    return blocks
end

"""
    ApertureVar(r)

Θ_r block: the aperture radius `r` (metres), one variable on its log chart (the step is d ln r, `NLLSsolver.update` below). It has
no bounds: the chart keeps r > 0 (the inert `lo` / `hi`, once `camera.r_min` / `camera.r_max`, are gone; a held aperture is
`fixed = ["aperture"]`, the block removed from the solve).
"""
struct ApertureVar{T}
    r::T
end
NLLSsolver.nvars(::ApertureVar) = static(1)
"""Log-parametrised radius: the step u of an `ApertureVar` is d ln r (r ← r eᵘ), so its Jacobian column is r ∂Φ/∂r
(`_log_r_column!`). In metres ∂Φ/∂r ≈ 2Φ/r dwarfs every other column, and NLLSsolver's uniform damping λ·I (λ₀ = 10⁻⁶ max diag JᵀJ)
then froze the material, vignetting and lighting blocks whenever r was free (decision 2026-10-26). The step is not boxed (decision
2026-10-31): the chart keeps r > 0, a fixed aperture is a removed block (`fixed = ["aperture"]`), and the box had pinned the
f/4 runs at 3 mm and once clamped a start above it into a worse state."""
NLLSsolver.update(v::ApertureVar, u, start = 1) = ApertureVar(v.r * exp(u[start]))
"""Scale the aperture column of `Jm` from ∂/∂r to ∂/∂ln r: the scalar-r column of an `ApertureVar`."""
@inline _log_r_column!(Jm, lay, ap::ApertureVar) = (lay.col_r > 0 && (Jm[:, lay.col_r] .*= ap.r); Jm)
"""
    MaterialVar(mat, surface, board = nothing; height = 0, inplane = false, fixed = (), mu_bounds = (0.0, 1.0))

Θ_m block: the 8 nonlinear parameters (`roughness_ink, roughness_paper, refractive_index_ink, refractive_index_paper`; boxed by `bounds`), μ ∈ [0, 1] and the board's
`TargetState` (its printed edge, δ and σ_print, is never a parameter): with `height > 0` (`NH = height_nparams(height)`) the board warping h through its
`HEIGHT_NFREE` free parameters (h += B y with `B = height_basis(board)`, the board's page: Core's TargetManifold of that
warping's height field — the piston and the two tilts, the poses', are never parameters; dual-number columns), with `inplane = true`
(`NI = INPLANE_NC`) the in-plane print error `(a, u)` of `TargetState.inplane` (Core's `Aspect ∘ Skew`, `inplane_calibration`:
additive, unbounded, no gauge — a rigid member and a scale are the poses'; dual-number columns). The surface shares the
material block because NLLSsolver caps a cost block at ten variable dependencies, which the per-tile blocks already use.

`fixed = (:material,)` holds Θ_m (the 8 nonlinear parameters and μ) at its value although the block stays variable (`HM = true`, the
type's last parameter: `material_fixed`): its Jacobian columns are zero (no AD scatter, no analytic μ column) and `update`
leaves it unchanged, while the block's warping and in-plane parameters remain free — a held material while a surface
parameter rides the block (the problem builders' `fixed`, `_hold_material`). The block size and every other column position are
unchanged; the LM's λ·I damping gives the zero columns a zero step. `mu_bounds` `(lo, hi)`: μ's chart `μ = lo + (hi − lo) σ(w)`
(`material.mu_bounds`, the state's `mu_bounds`).

`valid` is `false` for a block whose update is not finite (a NaN of a singular solve). Every residual of the block checks it
(`is_valid`) and returns NaN, so NLLSsolver rejects the step; an accepted state is always valid.
"""
struct MaterialVar{T, NH, NI, HB <: Union{Nothing, SMatrix{HEIGHT_NC, HEIGHT_NFREE, Float64}}, CH, HM}
    mat::Material{T}
    surface::TargetState{T}
    basis::HB                          # height_basis of the board's warping (NH > 0), else nothing
    dense_basis::Union{Nothing, Matrix{Float64}}   # `basis` as a dense matrix, built once (the device Jacobian's height columns, `_height_row!`)
    mu_bounds::NTuple{2, Float64}      # μ's chart interval (lo, hi)
    valid::Bool
end
"""Whether the material block's trial values are a state the residuals can render (`MaterialVar`'s `valid`; the one check of a step
the block could not represent, read by every residual of the block, CPU and device)."""
@inline is_valid(m::MaterialVar) = m.valid
"""The block's held parameters (the layouts' `fixed`): `(:material,)` while Θ_m is held inside the variable block (`HM`), else none."""
material_fixed(::MaterialVar{T, NH, NI, HB, CH, HM}) where {T, NH, NI, HB, CH, HM} = HM ? (:material,) : ()
function MaterialVar(mat::Material{T}, surface::TargetState{T}, board::Union{Nothing, TexturedTarget} = nothing;
                     height::Integer = 0, inplane::Bool = false, channel::Integer = 0, fixed = (),
                     mu_bounds = (0.0, 1.0)) where {T}
    all(k -> Symbol(k) === :material, fixed) || throw(ArgumentError("MaterialVar: fixed = $(repr(fixed)) (the block holds :material alone)"))
    height > 0 && board === nothing &&
        throw(ArgumentError("MaterialVar: a free board warping (height = $height) moves the nodal heights along its page's pure-bending basis: give the board"))
    basis = height > 0 ? height_basis(board) : nothing
    channel > 0 && (mat = tie_channels(mat, channel))
    return MaterialVar{T, height, inplane ? INPLANE_NC : 0, typeof(basis), Int(channel), !isempty(fixed)}(
        mat, surface, basis, basis === nothing ? nothing : Matrix(basis), Float64.(Tuple(mu_bounds)), _finite_material(mat, surface))
end
"""The material block's nonlinear coordinates: 8 (`α_ink, α_¬ink, η_ink[1:3], η_¬ink[1:3]`), or 4 for a single-channel state
(`channel = c > 0`: `α_ink, α_¬ink, η_ink, η_¬ink` — the measured channel's η, which the other channels follow, `tie_channels`)."""
n_material_nonlinear(::MaterialVar{T, NH, NI, HB, CH}) where {T, NH, NI, HB, CH} = length(material_sources(CH))
"""A channel-`c` material step `u` (the `material_sources(c)` coordinates, μ, the rest) as the full block's (the 8 nonlinear
through the tie basis `material_tie_basis(c)`: the measured channel's η step every channel's), in place of `u[start:end]`."""
function _tied_material_coordinates(u, start::Integer, n::Integer, c::Integer)
    s = start - 1; nk = length(material_sources(c))
    return vcat(material_tie_basis(c) * [u[s + i] for i in 1:nk], [u[s + i] for i in (nk + 1):n])
end
# The material and the surface finite.
_finite_material(m::Material, s::TargetState) =
    all(isfinite, nonlinear_parameters(m)) && isfinite(m.diffuse_weight) && all(isfinite, s.h) && all(isfinite, s.edge) && all(isfinite, s.inplane)
"""Number of free height parameters of the block: `HEIGHT_NFREE` when the warping is free (dual columns), else 0."""
n_height_free(::MaterialVar{T, NH}) where {T, NH} = NH > 0 ? HEIGHT_NFREE : 0
NLLSsolver.nvars(v::MaterialVar) = static(1 + n_material_nonlinear(v) + n_height_free(v) + n_inplane(v))
n_height(::MaterialVar{T, NH}) where {T, NH} = NH
"""Number of free in-plane print-error parameters of the block: `INPLANE_NC` when they are free (dual columns), else 0."""
n_inplane(::MaterialVar{T, NH, NI}) where {T, NH, NI} = NI
"""Θ_m after the step `u` (`update`'s): on its charts (decision 2026-10-31; before: additive steps clamped to `bounds`) — α = eᵘ
(α > 0), η = 1 + eᵘ (η > 1: a dielectric in air; the floor keeps a value rounded to η = 1 movable), μ = lo + (hi − lo) σ(w) on the
block's `mu_bounds`; held (`HM`, `fixed = (:material,)`): unchanged."""
function _material_update(v::MaterialVar, u, start)
    θ = nonlinear_parameters(v.mat)
    θ′ = SVector(ntuple(i -> i <= 2 ? θ[i] * exp(u[start + i - 1]) : 1 + max(θ[i] - 1, 1e-9) * exp(u[start + i - 1]), Val(8)))
    lo, hi = v.mu_bounds
    return with_diffuse_weight(with_nonlinear(v.mat, θ′), _μ_chart(v.mat.diffuse_weight, u[start + 8], lo, hi))
end
_material_update(v::MaterialVar{T, NH, NI, HB, CH, true}, u, start) where {T, NH, NI, HB, CH} = v.mat
function NLLSsolver.update(v::MaterialVar{T, NH, NI, HB, CH, HM}, u, start = 1) where {T, NH, NI, HB, CH, HM}
    if CH > 0                                                                    # a single-channel block: its step as the full block's
        u = _tied_material_coordinates(u, start, Int(NLLSsolver.nvars(v)), CH); start = 1
    end
    m = _material_update(v, u, start)
    surf = v.surface
    if NH > 0                                                                    # board warping: h += B y (the board's warping's pure-bending basis)
        b = start + 8
        y = SVector{HEIGHT_NFREE, T}(ntuple(i -> T(u[b + i]), Val(HEIGHT_NFREE)))
        surf = with_h(surf, surf.h + SMatrix{HEIGHT_NC, HEIGHT_NFREE, T}(v.basis) * y)
    end
    if NI > 0                                                                    # the in-plane print error (a, u): additive, unbounded
        b = start + 8 + (NH > 0 ? HEIGHT_NFREE : 0)
        surf = with_inplane(surf, surf.inplane + SVector{INPLANE_NC, T}(ntuple(i -> T(u[b + i]), Val(INPLANE_NC))))
    end
    valid = v.valid && _finite_material(m, surf)                                # a non-finite step: every residual of the block returns NaN (`is_valid`)
    return MaterialVar{T, NH, NI, HB, CH, HM}(m, surf, v.basis, v.dense_basis, v.mu_bounds, valid)
end

"""Scale the material block's 9 columns of `Jm` from the parameters to their charts' coordinates (the steps of `MaterialVar`):
∂/∂α → α ∂/∂α, ∂/∂η → (η − 1) ∂/∂η, ∂/∂μ → (μ − lo)(hi − μ)/(hi − lo) ∂/∂μ. The columns are `μ_col − 8 … μ_col − 1` (α_ink, α_¬ink,
η_ink[1:3], η_¬ink[1:3]) and `μ_col`, in both routes' layouts; nothing when Θ_m is held (`μ_col = 0`). Called at every
return of a Jacobian, beside `_log_r_column!`."""
@inline function _chart_material_columns!(Jm, lay, m::MaterialVar)
    lay.μ_col > 0 || return Jm
    θ = nonlinear_parameters(m.mat)
    @inbounds for i in 1:8
        Jm[:, lay.μ_col - 9 + i] .*= (i <= 2 ? θ[i] : θ[i] - 1)
    end
    lo, hi = m.mu_bounds
    Jm[:, lay.μ_col] .*= _μ_chart_scale(m.mat.diffuse_weight, lo, hi)
    return Jm
end

"""
    _μ_chart(μ, w, lo, hi) -> μ′

THE bounded chart of μ (decision 2026-10-31): `μ′ = lo + (hi − lo) σ(w₀ + w)`, `w₀` the logit of μ on `(lo, hi)` (clamped 1e-9 inside,
so a μ at a bound stays movable): the calibration LM's material block (`MaterialVar`) and the closed-form joint solve
(`_lm_radiometry!`) step μ on it. Device-safe.
"""
@inline function _μ_chart(μ, w, lo, hi)
    t = clamp((μ - lo) / (hi - lo), 1e-9, 1 - 1e-9)
    return lo + (hi - lo) / (1 + exp(-(log(t / (1 - t)) + w)))
end
"""∂μ/∂w of the chart `_μ_chart` at μ: `(μ − lo)(hi − μ)/(hi − lo)`."""
@inline _μ_chart_scale(μ, lo, hi) = (μ - lo) * (hi - μ) / (hi - lo)

# The Θ_v block is VisualGeometryToolkitCore's `restrict(ParameterManifold(vig), Z)` (a `LinearSubspaceManifold`, Core's
# geometry/manifolds/subset.jl: `vec(P) = vec(P₀) + Z y` over the free directions `Z`, `vignetting_basis`: one column per interior
# node of a tensor surface under the run's imaged region, a dependent node moving with its interior terms; a radial
# profile's supported nodes, the pole pair `P_0 = P_1` in one column, its dependent nodes continuing them — `RadialSupportRule`),
# an NLLSsolver variable through Core's adapter. The gauge `V(𝐜) = 1` is held exactly, always: the centre's direction is absent from
# `Z` (`gauge_basis`), and the start has V(𝐜) = 1 (`gauge_center_value`).
"""The free directions `Z` of the state's vignetting: `vignetting_basis` under its rule without the centre's direction (`gauge_basis`)."""
vignetting_basis(f::CalibrationData, θ::CalibrationParameters) = gauge_basis(θ.camera.vignetting, vignetting_basis(θ.camera.vignetting, f.vig_rule))

"""
    LightingVar(light; channel = 0)

Θ_l block: the `3 × J` SH coefficient matrix in `vec` (column-major, `a[c, j]`) order, parametrised as
`a = scale · â` with `scale = max|a|` at construction so that the optimiser sees O(1) parameters — NLLSsolver's
Levenberg–Marquardt damps with λ·I, which freezes a block whose natural scale is 10⁵ (photographs in linear
units against the unnormalised aperture integral). A single-channel state's block (`channel = c > 0`) is the measured channel's
`J` coefficients alone: every channel's row follows it (`tie_channels`), as no residual reads the others.
"""
struct LightingVar{TL <: EnvironmentLight, NL, CH}
    light::TL
    scale::Float64
end
function LightingVar(lt::EnvironmentLight; channel::Integer = 0)
    channel > 0 && (lt = tie_channels(lt, channel))
    a = coefficients(lt)
    return LightingVar{typeof(lt), channel == 0 ? length(a) : size(a, 2), Int(channel)}(lt, max(maximum(abs, a), 1e-12))
end
NLLSsolver.nvars(::LightingVar{TL, NL}) where {TL, NL} = static(NL)
function NLLSsolver.update(v::LightingVar{TL, NL, CH}, u, start = 1) where {TL, NL, CH}
    a = Matrix(coefficients(v.light))
    # the block's step through its tie basis (`light_tie_basis`: the identity for the three channels; the measured channel's step
    # every row of a single-channel block)
    a .+= v.scale .* reshape(light_tie_basis(CH, CH == 0 ? NL ÷ 3 : NL) * [u[start + i - 1] for i in 1:NL], size(a))
    return LightingVar{TL, NL, CH}(with_coefficients(v.light, a), v.scale)
end

"""
    PerViewResidual{N}

Cost block of view `k`: residual `Φ_c(𝐮; Θ̄) − ϕ_c(𝐮)` over the masked pixels. Depends on
`(pose_k, distortion, focus, aperture, material, vignetting, lighting)` (variable indices `idx`).
"""
struct PerViewResidual{N} <: NLLSsolver.AbstractResidual
    k::Int
    idx::SVector{7, Int}
    f::CalibrationData
    θ::CalibrationParameters{N}
end
Base.eltype(::PerViewResidual) = Float64
NLLSsolver.ndeps(::PerViewResidual) = static(7)
NLLSsolver.nres(r::PerViewResidual) = n_channels(r.f) * length(r.f.px[r.k])   # one residual per measured channel and pixel
NLLSsolver.varindices(r::PerViewResidual) = r.idx
NLLSsolver.getvars(r::PerViewResidual{N}, vars::Vector) where {N} =
    (vars[r.idx[1]]::VgtkCore.LeftRigidManifold6{Float64}, vars[r.idx[2]]::_LensBlock, vars[r.idx[3]]::_FocusBlock,
     vars[r.idx[4]]::ApertureVar{Float64}, vars[r.idx[5]]::MaterialVar{Float64}, vars[r.idx[6]]::VgtkCore.LinearSubspaceManifold, vars[r.idx[7]]::LightingVar)

"""View of the residual's variables (element type generic)."""
@inline function _view_of(r::PerViewResidual, pose::VgtkCore.LeftRigidManifold6{Float64}, dist::_LensBlock, foc::_FocusBlock, ap::ApertureVar)
    c = r.θ.camera
    return View(VgtkCore.model(pose), VgtkCore.with_model(c.geometric, VgtkCore.model(dist)), VgtkCore.model(foc), ap.r)
end

"""The per-view residual on every backend (NLLSsolver's `computeresidual`, per_tile.jl): the flux kernel (device/jacobian_kernels.jl, `_flux_device`)
minus the observed vector."""
function _block_residual(r::PerViewResidual, pose::VgtkCore.LeftRigidManifold6{Float64}, dist::_LensBlock, foc::_FocusBlock,
                        ap::ApertureVar, m::MaterialVar, vg::VgtkCore.LinearSubspaceManifold, lt::LightingVar)
    is_valid(m) || return fill(NaN, NLLSsolver.nres(r))      # a step the material block could not represent: rejected
    f, θ = r.f, r.θ
    view = _view_of(r, pose, dist, foc, ap)
    # a non-finite trial state must yield a NaN cost (the device kernels would silently render zero flux
    # for NaN rays, which leaves NLLSsolver's damping loop without an exit condition); the material block's finiteness is its
    # `is_valid`, checked above
    finite = all(isfinite, view.pose.t) && all(isfinite, view.pose.R) && all(isfinite, VgtkCore.parameter_vector(view.geometric.camera)) &&
             all(isfinite, view.focus.t̂_i) && isfinite(view.r) && all(isfinite, control_points(VgtkCore.model(vg))) && all(isfinite, coefficients(lt.light))
    finite || return fill(NaN, NLLSsolver.nres(r))
    fluxv = _flux_device(f, θ, r.k, view, VgtkCore.model(vg), m.mat, lt.light, m.surface)
    return weighted_rows(_view_weights(f.weights, r.k), _channel_rows(fluxv, f.channel) .- f.ϕ[r.k])   # the rows' weights (weighting.jl)
end

@inline _bit(flags, i) = (Int(flags) >> (i - 1)) & 1 == 1

struct JacTag end

"""
Column bookkeeping shared by the CPU and GPU Jacobians. AD slots cover only the genuinely nonlinear
blocks (pose 6, distortion ND, focus NS sample slots, material ∖ {μ} 8, with a free in-plane
print error its NI = INPLANE_NC parameters (`inplane_cols`, plain seeds in the scatter), the aperture radius 1;
`P = 14 + ND + NS + NI (+1)`). A free board warping's NH = HEIGHT_NC nodal heights take no slot (`height_cols`: HEIGHT_NFREE
columns, the height kernel's raw partials folded through the material block's basis by the row kernel): a slot no chunk seeds
widened every dual of the CPU's one chunk by 25 partials and split it in two. The
vignetting, lighting and μ columns are **analytic** — Φ is linear in Θ_v and Θ_l and affine in μ, so their columns are the
per-pixel `pixel_parts` responses (same quadrature nodes and CRN draws as the render: exact per realisation). The aperture
column is the AD partial of r seeded through the unit-disc rule `r² w̃_l L_c(·, r â_l)` (the derivative of the fixed-domain
integral). Returns the `(col, slot, ch)` AD scatter plus the column positions of the analytic blocks (`col_r`, `μ_col`,
`vig_cols`, `light_cols`). `fixed` (the material block's, `material_fixed`): with `:material` Θ_m is held inside the variable
block — its 9 columns stay zero (no scatter, `μ_col = 0`).
"""
function _jacobian_layout(varflags, N::Integer, NS::Integer, NF::Integer, NV::Integer, J::Integer, ND::Integer = 6; NH::Integer = 0, NI::Integer = 0,
                          fixed = ())
    off_e, off_d, off_f, off_m = 0, 6, 6 + ND, 6 + ND + NS
    ncol = 0
    scatter = Tuple{Int, Int, Int}[]; col_r = 0; μ_col = 0; vig_cols = 1:0; light_cols = 1:0; focus_cols = 1:0; height_cols = 1:0; inplane_cols = 1:0
    mat_col0 = -1                                          # the column before the material block's (−1: the block is fixed)
    if _bit(varflags, 1); for i in 1:6; push!(scatter, (ncol + i, off_e + i, 0)); end; ncol += 6; end
    if _bit(varflags, 2); for i in 1:ND; push!(scatter, (ncol + i, off_d + i, 0)); end; ncol += ND; end
    if _bit(varflags, 3); focus_cols = (ncol + 1):(ncol + N); ncol += N; end     # chain rule over NS t̂-sample slots
    off_h = off_m + 8                                      # where the dual slots continue after the 8 material parameters. The nodal heights take none:
    off_i = off_h                                          # their columns are the height kernel's (`_heights_device`); the in-plane print error's seeds (NI of them)
    off_r = off_i + NI                                     # the scalar aperture radius: one slot after the material (the heights, the in-plane error)
    if _bit(varflags, 4)
        col_r = ncol + 1; ncol += 1
        push!(scatter, (col_r, off_r + 1, 0))              # AD through r² w̃_l and r â_l (the fixed unit-disc rule)
    end
    if _bit(varflags, 5)
        mat_col0 = ncol
        if !(:material in fixed)                           # (Θ_m held: the 9 columns stay zero)
            for i in 1:8; push!(scatter, (ncol + i, off_m + i, 0)); end
            μ_col = ncol + 9
        end
        nhf = NH > 0 ? HEIGHT_NFREE : 0
        NH > 0 && (height_cols = (ncol + 10):(ncol + 9 + nhf))   # the free height parameters (AD, projected by the material block's basis; as the grid route)
        if NI > 0                                                # the in-plane print error: plain AD columns after the heights
            inplane_cols = (ncol + 10 + nhf):(ncol + 9 + nhf + NI)
            for i in 1:NI; push!(scatter, (ncol + 9 + nhf + i, off_i + i, 0)); end
        end
        ncol += 9 + nhf + NI
    end
    if _bit(varflags, 6); vig_cols = (ncol + 1):(ncol + NF); ncol += NF; end
    if _bit(varflags, 7); light_cols = (ncol + 1):(ncol + 3J); ncol += 3J; end
    return (; ncol, scatter, col_r, μ_col, mat_col0, vig_cols, light_cols, focus_cols, height_cols, inplane_cols, off_e, off_d, off_f, off_m, off_h, off_i, off_r, ND, NH, NI,
            P = off_r + (_bit(varflags, 4) ? 1 : 0))
end

"""The focus's nodal weights at every sensor sample of pixel `(u, v)` (closure-free for type stability): Core's `nodal_weights` of
the focus's thin plate spline system at the sample's sensor position."""
@inline _nodal_weights_at(tps, sensor, u, v, ::Val{NS}) where {NS} =
    ntuple(ks -> VgtkCore.nodal_weights(tps.sys, sensor_sample(sensor, ks, u, v)), Val(NS))

"""Chain rule for the focus columns of one pixel: `J[row, focus_cols[i]] = Σ_ks ∂Φ/∂inverse_focus_distance(𝐩_ks) · gᵢ(𝐩_ks)`."""
@inline function _focus_chain!(Jm, row0::Integer, dt̂::SMatrix{3}, g, focus_cols)
    fc0 = first(focus_cols) - 1
    @inbounds for c in 1:3, i in 1:length(focus_cols)
        acc = 0.0
        for ks in 1:size(dt̂, 2)
            acc += dt̂[c, ks] * g[ks][i]
        end
        Jm[row0 + c, fc0 + i] = acc
    end
    return nothing
end

"""The dual material and surface of a Jacobian evaluation: the 8 nonlinear parameters seeded at `off_m + 1:8`, μ and the printed-edge
parameters constant, the nodal heights seeded at `off_h + 1:HEIGHT_NC` when `off_h > 0` (free warping), the in-plane print error's
parameters at `off_i + 1:INPLANE_NC` when `off_i > 0`; `sd(x, slot)` = the caller's seeding function. Returns `(mat_d, surf_d)`."""
function _material_dual(m::Material, surf::TargetState, sd, off_m::Integer, ::Type{D}; off_h::Integer = 0, off_i::Integer = 0) where {D}
    θm = nonlinear_parameters(m)
    mat_d = Material(sd(θm[1], off_m + 1), sd(θm[2], off_m + 2),
                     SVector{3, D}(ntuple(i -> sd(θm[2 + i], off_m + 2 + i), Val(3))),
                     SVector{3, D}(ntuple(i -> sd(θm[5 + i], off_m + 5 + i), Val(3))), sd(m.diffuse_weight, 0))
    surf_d = TargetState{D}(SVector{HEIGHT_NC, D}(ntuple(i -> sd(surf.h[i], off_h > 0 ? off_h + i : 0), Val(HEIGHT_NC))),
                             SVector{EDGE_NC, D}(ntuple(i -> sd(surf.edge[i], 0), Val(EDGE_NC))),
                             SVector{INPLANE_NC, D}(ntuple(i -> sd(surf.inplane[i], off_i > 0 ? off_i + i : 0), Val(INPLANE_NC))))
    return mat_d, surf_d
end

"""The LM's Gauss–Newton blocks `(cost, Jᵀr, JᵀJ)` of a residual block, a view or a tile (NLLSsolver's `computecostgradhess`,
per_tile.jl), formed on the state's backend (`_block_jacobian`, `_gauss_newton_blocks`): the Jacobian never reaches the host. The
cost is `computecost`'s, ½‖r‖² of the block's cost rows (`_cost_rows`)."""
function _block_normal_equations(varflags, r, vars)
    b = _block_jacobian(varflags, r, vars)
    g, H = _gauss_newton_blocks(b.J, b.res, b.T, b.lay, r.f.run)
    return 0.5 * Float64(NLLSsolver.sqnorm(_cost_rows(b, r))), g, H
end
"""A view's Jacobian block (`_view_jacobian`) and its cost rows: the same residual, read back (`nres` values)."""
_block_jacobian(varflags, r::PerViewResidual, vars) = _view_jacobian(varflags, r, vars...)
_cost_rows(v, ::PerViewResidual) = Array(v.res)

# ---- the blocks concurrently ------------------------------------------------------------------------------------------------
# One view's kernels cover a few thousand pixels: a few percent of a device's resident threads (dgx01's B200: 6000 px per launch), so
# the views run one after the other left the device idle most of the iteration (2026-11-13: ≈ 230 ms per view, 57 s of a 70 s
# iteration); a grid problem's tiles (a few hundred pixels each, their kernels the view batch's) the same per tile. NLLSsolver visits
# the residual blocks in order, so a request for block i (a view, a tile) also starts blocks i + 1 … i + W at the same trial on pooled
# device workers (`_on_worker`: a device queue each), and their requests collect them: the same function on the same inputs, so the
# same values bit for bit, W + 1 blocks' kernels on the device at once (`_windowed`).

"""
    _window_width(f, θ, list) -> W

The blocks a request starts ahead (`_windowed`), derived at run time from the device and the problem (owner ruling 2026-11-13: no
constant, no config key — an execution setting that changes no value):
- occupancy: `W_occ = ⌈resident threads / pixels per block⌉`, the resident threads of the LM's heaviest kernel, the radiance record
  (`_note_capacity!`: its register-aware occupancy, Core's `device_resident_threads`), the pixels per block the mean over `list` (a view's masked
  pixels, a tile's);
- memory: `W_mem = ⌊free device bytes / (WINDOW_MEMORY_FOOTPRINT · bytes per block)⌋`, the bytes of the problem's first block's job
  (`_note_block_bytes!`, kept in the registered problem's window: its kernels' outputs, slot array, Jacobian and residual; a tile also
  its share of its view's batch) and Core's `device_free_bytes`;
- the CPU backend: the thread count;
- clamped to `1 … length(list)`.
Every value is the same at any W (`_windowed`); the W of a run is recorded in its provenance (`window_width`).
"""
function _window_width(f::CalibrationData, θ::CalibrationParameters, list)
    n = length(list)
    n == 0 && return 1
    be = f.backend
    KA.isgpu(be) || return clamp(VgtkCore.device_resident_threads(be), 1, n)   # the host: its threads (Core's)
    cap = lock(() -> get(_CAPACITY, typeof(be), nothing), _DEVICE_CACHE_LOCK)
    if cap === nothing
        @warn "no resident-thread query for $(typeof(be)): the LM's blocks run one at a time" maxlog = 1
        return 1
    end
    px = sum(_block_pixels, list) / n
    occ = ceil(Int, cap / max(px, 1.0))
    w = _device_entry(f, :window)
    bytes = w === nothing ? 0 : w.block_bytes[]
    mem = floor(Int, VgtkCore.device_free_bytes(be) / (WINDOW_MEMORY_FOOTPRINT * max(bytes, 1)))
    return clamp(min(occ, mem), 1, n)
end
_block_pixels(r::PerViewResidual) = length(r.f.px[r.k])
_block_pixels(r) = length(r.pix)
"""Note the device bytes `b` of a block's job of the state's registered problem (its window's `block_bytes`, atomic: the blocks run on
concurrent workers), the window's memory bound (`_window_width`); without a registered problem, nothing."""
function _note_block_bytes!(f::CalibrationData, b::Integer)
    w = _device_entry(f, :window)
    w === nothing || (w.block_bytes[] = Int(b))
    return nothing
end
"""The device memory the blocks in flight hold at their peak, per estimated job byte (`_window_width`: `W_mem = ⌊free bytes /
(WINDOW_MEMORY_FOOTPRINT · bytes per block)⌋`). 0.4: the measured high-water mark of the device's used memory during a linearization
sweep over W × the first block's job bytes (`_note_block_bytes!`) was at most 0.386 — ag10fe_bc5 0.14, 0.31, 0.24 at W = 7, 16, 64;
ag08ta_spline 0.02, 0.39 at W = 37, 64 (dgx01, one B200, jobs 14792 and 14793): a job's arrays are freed and reused while the
others run, so the estimate is an upper bound."""
const WINDOW_MEMORY_FOOTPRINT = 0.4

"""
    _register_window!(f, θ, bundle, select; width = nothing, vectors = bundle)

Enable the concurrent evaluation (`_windowed`) of the residual blocks of `bundle` (a [`CalibrationBundle`](@ref)) whose type `select`
accepts (in NLLSsolver's visit order, `_visit_order`): `width` blocks ahead of each request, or (`nothing`)
`_window_width`'s, derived after the first block. `vectors` is where the requests' variables live (`_trial_vectors`): the bundle's own
variables, or the solver's while it solves (`bundle.solver`, which VisualGeometryToolkitCore's `prepare_evaluation!` fills).
`_lm_views!` (the views), `_lm_tiles!` (the tiles) and the stall postcondition register their problem and unregister it on exit
(`_unregister_window!`).
"""
function _register_window!(f::CalibrationData, θ::CalibrationParameters, bundle, select; width::Union{Nothing, Integer} = nothing, vectors = bundle)
    _unregister_window!(f)
    _ensure_device!(f, θ); _height_board(f, θ); _view_phi(f, θ, 1)        # the device cache's lazy entries, built before workers read it
    problem = vectors
    list = [c for c in _visit_order(bundle) if select(typeof(c))]
    pos = IdDict{Any, Int}(c => i for (i, c) in enumerate(list))
    f.device_cache = merge(f.device_cache, (; window = (; problem, list, pos, width = Ref(width === nothing ? 0 : Int(width)),
                                                         pending = Vector{Any}(nothing, length(list)), block_bytes = Threads.Atomic{Int}(0),
                                                         buffers = IdDict{Task, Any}(), buffers_lock = ReentrantLock())))
    return nothing
end
"""
    _with_registered(work, f, θ, bundle, idx = nothing; concurrent = true, vectors = bundle) -> work()

`work()` with the calibration problem `bundle` (a [`CalibrationBundle`](@ref), its variables in `vectors`: `_register_window!`)
registered on the data `f` (at the parameters `θ`), and unregistered on exit whatever `work` does — the one teardown of `_lm_views!`,
`_lm_tiles!` and the grid smoothness's weight selection. A grid problem (`idx`, `build_problem_grid`'s indices) its view batches
(`_register_grid_problem!`); with `concurrent` its blocks in the window (`_register_window!`: the views of a per-view problem, the tiles
of a grid problem).
"""
function _with_registered(work, f::CalibrationData, θ::CalibrationParameters, bundle, idx = nothing; concurrent::Bool = true, vectors = bundle)
    grid = idx !== nothing && hasproperty(idx, :patches)
    grid && _register_grid_problem!(f, θ, bundle, idx; vectors)
    try
        concurrent && _register_window!(f, θ, bundle, grid ? (T -> T <: PerTileResidual) : (T -> T <: PerViewResidual); vectors)
        return work()
    finally
        concurrent && _unregister_window!(f)            # (its jobs read the grid batch: first)
        grid && _unregister_grid_problem!(f)
    end
end
"""Disable the concurrent blocks (`_register_window!`): every job in flight collected and dropped. The width the window ran at stays
in the device cache (`window_width`, the run's provenance)."""
function _unregister_window!(f::CalibrationData)
    w = _device_entry(f, :window)
    w === nothing && return nothing
    for i in eachindex(w.pending)
        job = w.pending[i]
        job === nothing || try _await(job) catch end
        w.pending[i] = nothing
    end
    lock(() -> empty!(w.buffers), w.buffers_lock)            # the solve's Jacobian buffers (`_jacobian_buffers`): released with it
    f.device_cache = merge(Base.structdiff(f.device_cache, NamedTuple{(:window,)}), (; window_width = w.width[]))
    return nothing
end
"""The entry `key` of the state's device cache (a registered problem's: `:grid`, `:window`), `nothing` without one."""
_device_entry(f::CalibrationData, key::Symbol) = (dc = f.device_cache; dc === nothing ? nothing : get(dc, key, nothing))

"""
    _windowed(f, r, vars, what) -> f(r, vars)

`f(r, vars)` of the residual block `r` (a view, a tile; `what` names the request: `f` must be the same function of `(r, vars)` for
every block with that `what`): from the job started ahead for it when its key — the variables' objects and `what` — matches, else
evaluated here; and, when the state has a registered problem (`_register_window!`) that holds `r` and whose variables hold `vars`
(`_locate_vars`), the `width[]` blocks after it started at the same trial vector (`_window_width`'s, derived after the first
block). A job that failed or whose key differs is dropped (the request evaluates, and reports a failure itself).
"""
function _windowed(f, r, vars, what)
    w = _device_entry(r.f, :window)
    w === nothing && return f(r, vars)
    i = get(w.pos, r, 0)
    i == 0 && return f(r, vars)
    vec = _locate_vars(w.problem, r, vars)
    vec === nothing && return f(r, vars)
    key = (map(objectid, vars), what)
    job = w.pending[i]; w.pending[i] = nothing
    out = job === nothing ? nothing : (x = try _await(job) catch; nothing end; x !== nothing && x[1] == key ? Some(x[2]) : nothing)
    if w.width[] == 0                                     # the width, after the first block (its kernels' occupancy and bytes)
        out === nothing && (out = Some(f(r, vars)))
        w.width[] = _window_width(r.f, r.θ, w.list)
    end
    for j in (i + 1):min(i + w.width[], length(w.list))
        w.pending[j] === nothing || continue
        rn = w.list[j]
        varsn = NLLSsolver.getvars(rn, vec)
        keyn = (map(objectid, varsn), what)
        w.pending[j] = _on_worker(() -> (keyn, f(rn, varsn)))
    end
    return out === nothing ? f(r, vars) : something(out)
end

"""
    _channel_jacobian(Jm, lay, c) -> J

A Jacobian assembled three channels per pixel (`lay`: `_jacobian_layout` or `_tile_layout`, rows pixel-major) as the residual's of
a single-channel state (`channel = c`): channel `c`'s rows (`_channel_rows`), the lighting and material blocks' columns through
their tie bases (`light_tie_basis`, `material_tie_basis`: the other channels' coefficients and refractive indices follow channel
`c`, and on its rows their columns are zero), every other column as it is — the single-channel blocks' coordinates in order.
`Jm` itself for a colour state (`c = 0`): the same matrix, bit for bit.
"""
function _channel_jacobian(Jm::AbstractMatrix, lay, c::Integer)
    c == 0 && return Jm
    return _tie_columns(Jm[c:3:size(Jm, 1), :], lay, c)
end
"""The columns of `R` (a Jacobian's, or the identity's: `_column_map`) through a channel-`c` state's tie bases (`material_tie_basis`,
`light_tie_basis`): the single-channel blocks' coordinates in order; `R` itself for a colour state (`c = 0`)."""
function _tie_columns(R::AbstractMatrix, lay, c::Integer)
    c == 0 && return R
    # each block's columns through its tie basis (`material_tie_basis`, `light_tie_basis`), the other columns as they are
    blocks = Tuple{UnitRange{Int}, Matrix{Float64}}[]
    lay.mat_col0 >= 0 && push!(blocks, ((lay.mat_col0 + 1):(lay.mat_col0 + 8), material_tie_basis(c)))
    isempty(lay.light_cols) || push!(blocks, (lay.light_cols, light_tie_basis(c, length(lay.light_cols) ÷ 3)))
    sort!(blocks; by = b -> first(b[1]))
    parts = Matrix{Float64}[]; col = 1
    for (cols, B) in blocks
        push!(parts, R[:, col:(first(cols) - 1)], R[:, cols] * B); col = last(cols) + 1
    end
    push!(parts, R[:, col:end])
    return reduce(hcat, parts)
end

"""
    _scrub_entries!(J, dims, block, run) -> J

Replace non-finite Jacobian entries by zero (an isolated derivative singularity must not poison the step), and count them in the
state's run record (`run.scrubbed`, `_LmRun`) under `block` (a Symbol, or a layout — `_jacobian_layout`, `_tile_layout` — counting each
column under its block, `_column_block`); `J`'s columns are along dimension `3 - dims` (`dims = 1`: rows × columns; `dims = 2`:
columns × rows, the device Jacobian, `_scrub!`). A non-finite derivative is a defect to find, not a case to absorb: a masked or off-board pixel has a finite zero
row, so every count here is unexpected, and `_lm_views!` / `_lm_tiles!` warn with the counts of the fit (`_report_scrubbed`). (A NaN
derivative in the light sphere's exit point, zeroed here silently, made the Jacobian rows of every wide-angle pixel zero and stalled
a fit started at the truth.) Array-generic: the count and the replacement run on `J`'s backend (the LM's device Jacobian, `_scrub!`).
"""
function _scrub_entries!(J::AbstractMatrix, dims::Integer, block, run::_LmRun)
    bad = vec(Array(sum(x -> !isfinite(x), J; dims)))
    any(>(0), bad) || return J
    J .= ifelse.(isfinite.(J), J, zero(eltype(J)))
    counts = Dict{Symbol, Int}()
    for j in eachindex(bad)
        bad[j] > 0 || continue
        b = block isa Symbol ? block : _column_block(block, j)
        counts[b] = get(counts, b, 0) + bad[j]
    end
    lock(() -> mergewith!(+, run.scrubbed, counts), run.lock)
    return J
end
"""The parameter block of column `j` of a Jacobian laid out by `lay` (`_jacobian_layout` or `_tile_layout`)."""
function _column_block(lay, j::Integer)
    j in lay.focus_cols && return :focus
    j == lay.col_r && return :aperture
    j in lay.vig_cols && return :vignetting
    j in lay.light_cols && return :lighting
    (j == lay.μ_col || j in lay.height_cols || j in lay.inplane_cols) && return :material
    for (col, slot, _) in lay.scatter
        col == j || continue
        slot <= lay.off_e + 6 && return :pose
        lay.off_m < slot <= lay.off_m + 8 && return :material
    end
    return hasproperty(lay, :patch_cols) ? :grid : :lens
end
"""Forget the scrubbed counts of the state's run record `run` (a fit resets them at its start)."""
_reset_scrubbed!(run::_LmRun) = lock(() -> empty!(run.scrubbed), run.lock)
"""Warn once with the fit's scrubbed counts (`run.scrubbed`; none: silent); returns them."""
function _report_scrubbed(run::_LmRun, fit::AbstractString)
    counts = lock(() -> copy(run.scrubbed), run.lock)
    isempty(counts) || @warn "$fit: non-finite Jacobian entries were replaced by zero (a derivative defect; these rows do not move the step)" counts
    return counts
end

# The per-view problem's blocks, by name: what `fixed` holds besides the lens model's parameters (`_split_fixed`).
const _VIEW_BLOCKS = (:poses, :distortion, :focus, :aperture, :material, :vignetting, :lighting)
"""The surface parameters the material block may carry — the board warping `:height`, the in-plane print error `:inplane` — each
held at its start by naming it in `fixed` (it then has no columns), on both routes."""
const SURFACE_HOLDS = (:height, :inplane)
"""`fixed` (the stack's one holding keyword) split into the per-view problem's blocks (`_VIEW_BLOCKS`) and the lens model's
parameter names or groups (Core's `free_parameter_names`: an entry that names nothing there is an `ArgumentError`); the surface
holds (`SURFACE_HOLDS`) are neither (the material block's own: `_surface_held`)."""
_split_fixed(fixed) = (Tuple(Symbol(k) for k in fixed if Symbol(k) in _VIEW_BLOCKS),
                       Tuple(Symbol(k) for k in fixed if !(Symbol(k) in _VIEW_BLOCKS) && !(Symbol(k) in SURFACE_HOLDS)))
"""Whether `fixed` holds the surface parameter `k` (`SURFACE_HOLDS`): held, the material block carries no columns for it."""
_surface_held(fixed, k::Symbol) = any(f -> Symbol(f) === k, fixed)
"""
    _hold_material(blocks_held, rides) -> (material_fixed, blocks_held)

`:material` in the held blocks `blocks_held` holds Θ_m. While a free surface parameter rides the material block (`rides`: the board
warping, the in-plane print error) Θ_m alone is held — the `MaterialVar`'s `fixed = (:material,)`, the block stays
variable — else the whole block (the problem's free mask). Both problem builders' one rule.
"""
_hold_material(blocks_held, rides::Bool) =
    :material in blocks_held && rides ? ((:material,), Tuple(k for k in blocks_held if k !== :material)) : ((), Tuple(blocks_held))

"""
    build_problem(f::CalibrationData, θ::CalibrationParameters; fixed = (), height_degree = 0, inplane = false) -> (problem::CalibrationBundle, idx)

The per-view LM problem at the state `s`: a [`CalibrationBundle`](@ref) (VisualGeometryToolkitCore's block contract, solved by its
`LevenbergMarquardt`) with one pose
variable (`VgtkCore.LeftRigidManifold6{Float64}`) per view, the shared distortion / focus / aperture / material / vignetting / lighting variables and one
`PerViewResidual` per view. `fixed` is the one holding keyword: the blocks it names (of `(:poses, :distortion, :focus, :aperture,
:material, :vignetting, :lighting)`) are held — the problem's `fixed` variables, `idx.free` the free mask; `:material` holds Θ_m alone
while a free surface parameter rides the block (`_hold_material`) —, the surface parameters it
names (`SURFACE_HOLDS`: `:height`, `:inplane`) are
held at their start, and its other entries hold the lens model's parameters by
Core's names in the lens block (`lens_free`; `(:cx, :cy, :k3)`, `:aspect` the aspect f_y/f_x); an entry that is none of these is an
`ArgumentError`. A lens with no free parameter is held. The model's structure (as the grid route's `build_problem_grid`): with
`height_degree > 0` the material block carries the board warping's `HEIGHT_NC` nodal heights (`MaterialVar{Float64, HEIGHT_NC}`:
HEIGHT_NFREE free parameters, h = h₀ + B y with `B = height_basis(f.board)`); with `inplane` the in-plane print error `(a, u)` of
the surface (Core's `Aspect ∘ Skew`, two dual-number columns) — each unless `fixed` holds it.
"""
function build_problem(f::CalibrationData, θ::CalibrationParameters{N}; fixed = (), height_degree::Integer = 0,
                       inplane::Bool = false) where {N}
    K = n_views(θ)
    vv = VgtkCore.restrict(VgtkCore.ParameterManifold(θ.camera.vignetting), vignetting_basis(f, θ))   # Θ_v: Core's LinearSubspaceManifold
    lv = LightingVar(θ.light; f.channel)
    blocks_held, lens_held = _split_fixed(fixed)           # the blocks `fixed` names, and the lens model's parameters
    lv_d = lens_manifold(geometric(θ); fixed = lens_held)
    NH = _surface_held(fixed, :height) ? 0 : height_nparams(height_degree)   # a free board warping: the nodal heights ride in the material block (as in the grid route)
    inplane = inplane && !_surface_held(fixed, :inplane)
    material_held, blocks_held = _hold_material(blocks_held, NH > 0 || inplane)
    mv = MaterialVar(θ.mat, θ.surface, _board(f, θ); height = NH, inplane, f.channel, fixed = material_held, f.mu_bounds)
    variables = Any[VgtkCore.LeftRigidManifold6(θ.poses[k]) for k in 1:K]
    push!(variables, lv_d, focus_block(focus(θ)),
          ApertureVar(aperture_radius(θ)), mv, vv, lv)
    i_d, i_f, i_r, i_m, i_v, i_l = (K + 1):(K + 6)
    blocks = Any[PerViewResidual{N}(k, SVector(k, i_d, i_f, i_r, i_m, i_v, i_l), f, θ) for k in 1:K]
    _add_focus_smoothness!(blocks, i_f, N, f.focus_smoothness)
    idx = (; poses = 1:K, distortion = i_d, focus = i_f, aperture = i_r, material = i_m, vignetting = i_v, lighting = i_l)
    free = trues(length(variables))
    last(lens_free(geometric(θ); fixed = lens_held)) && (free[i_d] = false)      # a fixed model: no lens columns
    _free_mask!(free, idx, blocks_held; lens_key = :distortion, lens_idx = i_d)
    return CalibrationBundle(variables, blocks, free, vars -> _extract_variables(f, θ, vars, idx)), (; idx..., free)
end

"""
    CalibrationBundle(variables, blocks, free, decode)

A calibration LM's problem — `build_problem`'s (per view) and `build_problem_grid`'s (per tile): its NLLSsolver variables (the poses,
the lens or the grid's patches, focus, aperture, material, vignetting, lighting), its residual blocks (a view's or a tile's, the
smoothness rows), the free mask `free` over the variables (a held block: NLLSsolver's unfixed mask) and `decode`, the route's reading
of a variable vector into the parameters it encodes (`vars -> (camera, poses, mat, surface, light)`). A bundle of
VisualGeometryToolkitCore's block contract (`block_structure`, `fixed_variables`, plain least squares, no residual summary; its
`block_result` assembles the estimate: `u` the NamedTuple `(; camera, poses, mat, surface, light)` the solved variables encode, the
LM's termination record, stats and metadata as they are), solved by `init(RefineProblem(bundle, bundle.variables), LevenbergMarquardt(; damping);
stops, callback)` and `solve!` (`_solve_bundle!`). `solver` holds the two vectors the solve evaluates the blocks at (NLLSsolver's
`variables` and `varnext`, as VisualGeometryToolkitCore's `prepare_evaluation!` names them; empty before a solve), where its batched
requests find their variables (`_with_registered`).
"""
struct CalibrationBundle{D} <: VgtkCore.AbstractBlockProblem
    variables::Vector{Any}
    blocks::Vector{Any}
    free::BitVector
    decode::D
    solver::Base.RefValue{Any}
end
CalibrationBundle(variables, blocks, free, decode) = CalibrationBundle(variables, blocks, free, decode, Ref{Any}((; variables = Any[], varnext = Any[])))
VgtkCore.prepare_evaluation!(b::CalibrationBundle, variables, varnext) = (b.solver[] = (; variables, varnext); nothing)
VgtkCore.block_structure(b::CalibrationBundle, _u0) = (b.variables, b.blocks, nothing)
VgtkCore.fixed_variables(b::CalibrationBundle) = findall(!, b.free)
VgtkCore.default_optimization_strategy(::CalibrationBundle) = nothing    # plain least squares: the rows are whitened at the data
VgtkCore.residual_summary(::CalibrationBundle) = nothing                 # the result reads no row: nothing swept at the solution
# the estimate the solved variables encode (`decode`: copied out of the cache's vector, which the next solve overwrites)
function VgtkCore.block_result(b::CalibrationBundle, _u0, r::VgtkCore.FitResult, _summary)
    camera, poses, mat, surface, light = b.decode(r.u)
    return VgtkCore.FitResult((; camera, poses, mat, surface, light), r.retcode, r.stop_rule, r.stats, r.rms, r.n, r.metadata;
                              r.resid, r.original)
end

"""
    _solve_bundle!(f, θ, bundle, idx; stops, damping, callback) -> FitResult

The calibration LM of `bundle` on the data `f` at the parameters `θ` (`idx` its indices): VisualGeometryToolkitCore's
LevenbergMarquardt at the optimizer's `damping` under `stops` (`_lm_stops`: their residual resolution the data's), the solve with
the bundle registered (`_with_registered`, the solver's variables `bundle.solver`), its estimate (`u`, the bundle's `block_result`)
written into `θ` (`_read_back!`). The result is the LM's: its termination record, `stats.nsteps` the accepted iterations,
`metadata.cost` ‖r‖² at the solution.
"""
function _solve_bundle!(f::CalibrationData, θ::CalibrationParameters, bundle::CalibrationBundle, idx; stops::VgtkCore.LmStops,
                        damping::VgtkCore.LmDamping, callback = nothing)
    c = VgtkCore.init(VgtkCore.RefineProblem(bundle, bundle.variables), VgtkCore.LevenbergMarquardt(; damping); stops = _lm_stops(f, stops), callback)
    r = _with_registered(() -> VgtkCore.solve!(c), f, θ, bundle, idx; vectors = bundle.solver)
    _read_back!(θ, r.u)
    return r
end
"""The residual blocks of `b` in the order the LM visits them — NLLSsolver's: by type, in the order of the blocks' `Union` (the problem
VisualGeometryToolkitCore builds holds them in a repository of that union), then as added."""
function _visit_order(b::CalibrationBundle)
    U = Union{unique(typeof.(b.blocks))...}
    return Any[c for T in _union_types(U) for c in b.blocks if typeof(c) === T]
end
_union_types(U::Union) = (_union_types(U.a)..., _union_types(U.b)...)
_union_types(T::Type) = (T,)
"""The variable vectors a request's variables are located in (`_locate_vars`): a bundle's own, or the solve's variables and trial
(NLLSsolver's `variables`, `varnext`: `bundle.solver`, as `prepare_evaluation!` last named them)."""
_trial_vectors(b::CalibrationBundle) = (b.variables,)
_trial_vectors(r::Base.RefValue{Any}) = _trial_vectors(r[])
_trial_vectors(p) = (p.variables, p.varnext)

"""
    _extract_variables(f, θ, vars, idx) -> (camera, poses, mat, surface, light)

The camera, the poses, the material, the surface and the lighting encoded in a per-view variable vector (non-mutating): the bundle's
`decode` (its `block_result`) and the per-iteration callback's.
"""
function _extract_variables(f::CalibrationData, θ::CalibrationParameters, vars::Vector, idx)
    lens_new = VgtkCore.with_model(geometric(θ), VgtkCore.model(vars[idx.distortion]::_LensBlock))
    mv = vars[idx.material]::MaterialVar
    camera = RadiometricCameraModel(lens_new, VgtkCore.model(vars[idx.focus]::_FocusBlock), (vars[idx.aperture]::ApertureVar).r,
                                    VgtkCore.model(vars[idx.vignetting]::VgtkCore.LinearSubspaceManifold))
    poses = [VgtkCore.model(vars[k]::VgtkCore.LeftRigidManifold6{Float64}) for k in idx.poses]
    return camera, poses, mv.mat, mv.surface, (vars[idx.lighting]::LightingVar).light
end

"""`_read_back!(θ, u)`: write an LM's estimate `u` (its result's, `(; camera, poses, mat, surface, light)`: the bundle's
`block_result`) into the parameters: the one read-back of both LMs."""
function _read_back!(θ::CalibrationParameters, u::NamedTuple)
    θ.camera = u.camera; θ.poses .= u.poses
    θ.mat = u.mat; θ.light = u.light
    set_surface!(θ, u.surface)
    return θ
end

"""
    Θ̄_snapshot(f, θ; camera = θ.camera, poses = θ.poses, mat = θ.mat, surface = board_surface(θ), light = θ.light) -> NamedTuple

Everything needed to re-render the state later (`camera, poses, mat, surface, light`).
"""
Θ̄_snapshot(f::CalibrationData, θ::CalibrationParameters; camera = θ.camera, poses = θ.poses, mat = θ.mat, surface = θ.surface, light = θ.light) =
    (; camera, poses = copy(poses), mat, surface, light)


"""
    _free_mask!(free, idx, fixed; lens_key, lens_idx) -> free

Clears in the free mask `free` every block `fixed` names (`build_problem`'s and `build_problem_grid`'s one mask): `:poses`, the lens
blocks `lens_idx` under `lens_key` (`:distortion` per view, `:grid` per tile), `:focus`, `:aperture`, `:material`, `:vignetting` and
`:lighting`; another name is an `ArgumentError`.
"""
function _free_mask!(free::AbstractVector{Bool}, idx, fixed; lens_key::Symbol, lens_idx)
    names = (:poses, lens_key, :focus, :aperture, :material, :vignetting, :lighting)
    for k in fixed
        k in names || throw(ArgumentError("fixed: $(repr(k)) is not a block of this problem (it has $(join(repr.(names), ", ")))"))
    end
    :poses in fixed && (free[idx.poses] .= false)
    lens_key in fixed && fill!(view(free, lens_idx), false)    # a view's per-view lens block (an index) or a tile's patches
    :focus in fixed && (free[idx.focus] = false)
    :aperture in fixed && (free[idx.aperture] = false)
    :material in fixed && (free[idx.material] = false)
    :vignetting in fixed && (free[idx.vignetting] = false)
    :lighting in fixed && (free[idx.lighting] = false)
    return free
end

"""
    _lm_callback(f, θ, idx, extract; trace, trace_meta, iter_callback, verbose, lm_log) -> callback

The per-iteration callback of a calibration LM (`_lm_views!`'s and `_lm_tiles!`'s one; VisualGeometryToolkitCore's `callback = state ->
Bool` of its LevenbergMarquardt, `state` an `LmIterationState`): with a `trace` or an `iter_callback`, after every iteration the
iterate's variables (`state.u`) read through `extract(s, vars, idx)` (the route's `_extract_variables` / `_extract_variables_grid`)
into `(camera, poses, mat, surface, light)`, a snapshot pushed to `trace` (`trace_meta` merged, `phase = :lm`) and
`iter_callback(it, cost, camera, poses, mat, surface, light)` called; else a print-out line per iteration (`verbose`, `_lm_print`)
or none (`nothing`). Either way wrapped by `_lm_log` (the file `lm_log`, the algorithm's `log`; empty: none). It never stops the
solve.
"""
function _lm_callback(f::CalibrationData, θ::CalibrationParameters, idx, extract; trace = nothing, trace_meta = (;), iter_callback = nothing, verbose::Bool = false,
                      lm_log::AbstractString = "")
    cb = if trace !== nothing || iter_callback !== nothing
        it = Ref(0)
        function (state)
            camera, poses, mat, surface, light = extract(f, θ, state.u, idx)
            it[] += 1
            trace !== nothing && push!(trace, (; trace_meta..., phase = :lm, cost = state.cost, time = time(),
                                               snap = Θ̄_snapshot(f, θ; camera, poses, mat, surface, light)))
            iter_callback !== nothing && iter_callback(it[], state.cost, camera, poses, mat, surface, light)
            return false
        end
    else
        verbose ? _lm_print : nothing
    end
    return _lm_log(cb, get(trace_meta, :ϕ², NaN), lm_log)
end

"""
    _lm_views!(f, θ; stops, damping, fixed = (), height_degree = 0, inplane = false, kwargs...) -> FitResult

Levenberg–Marquardt over all parameter blocks but those `fixed` names (of `(:poses, :distortion, :focus, :aperture, :material,
:vignetting, :lighting)`), the surface parameters it names (`:height`, `:inplane`) and the lens block's parameters `fixed` names by
Core's parameter names (`lens_free`; default none: every parameter but the skew is free; `:aspect` ties the focal lengths): one
holding keyword (`build_problem`, with the model's structure `height_degree` and `inplane`). `iter_callback(it, cost, camera, poses,
mat, surface, light)` sees the current best variables after every iteration and `trace` collects the per-iteration snapshots
(`_lm_callback`, as `_lm_tiles!`); with a file `lm_log` (the algorithm's `log`) a line per iteration is appended to it (`_lm_log`). The solve is VisualGeometryToolkitCore's one LM, its `LevenbergMarquardt` on `build_problem`'s
`CalibrationBundle` (`_solve_bundle!`) at the optimizer's `damping` under the solve's `stops` (Core's `LmStops`, their residual resolution the state's: `_lm_stops`) — the RELATIVE cost decrease `ftol`, the
relative step test `xtol_rel` and the absolute step max-norm `xtol`; no absolute cost decrease test — the cost is in the photographs' squared units, dimensionless when the rows
are whitened (a two-view 32 × 32 fixture starts at ≈ 2e-4), so an absolute threshold of 1e-12 stopped the solver with the aperture, μ and the lighting still drifting together
along a flat direction. Returns the LM's `FitResult` (`retcode`, a `ReturnCode`: `Success` at a tolerance stop — converged by definition
— or at a stationary point after the rejection cap; `stop_rule`, a `StopRule`; `stats.nsteps`; `metadata.cost`, ‖r‖² at the solution).
"""
function _lm_views!(f::CalibrationData, θ::CalibrationParameters; stops::VgtkCore.LmStops, damping::VgtkCore.LmDamping, fixed = (), verbose::Bool = false,
                    trace = nothing, trace_meta = (;), iter_callback = nothing, height_degree::Integer = 0, inplane::Bool = false,
                    lm_log::AbstractString = "")
    bundle, idx = build_problem(f, θ; fixed, height_degree, inplane)
    callback = _lm_callback(f, θ, idx, _extract_variables; trace, trace_meta, iter_callback, verbose, lm_log)
    _reset_scrubbed!(f.run)
    try                                                    # (Core's stall postcondition runs inside, the views still concurrent)
        return _solve_bundle!(f, θ, bundle, idx; stops, damping, callback)
    finally
        _report_scrubbed(f.run, "_lm_views!")
    end
end

"""The absolute step tolerance of every LM of RC: Core's of record (`LmStops()`'s `xtol`, NLLSsolver's dstep); no `[optimizer]` key."""
const CORE_LM_XTOL = VgtkCore.LmStops().xtol
"""The solve's stop settings `stops` (VisualGeometryToolkitCore's `LmStops`) of the calibration LM on the data `f`: as given, their
residual resolution the norm the data resolve — the photographs' RAW quantisation in the cost's norm, never the noise level
(`lm_abstol`), a fact of the data."""
_lm_stops(f::CalibrationData, stops::VgtkCore.LmStops) =
    VgtkCore.LmStops(; stops.max_iter, stops.ftol, stops.xtol, stops.xtol_rel, stops.stationarity_tol, stops.max_time,
                     residual_resolution = lm_abstol(f))
"""The stops of one LM chunk of a refinement: `stops` with its iteration cap `n` and its wall-clock cap `max_time` (s), the others kept."""
_chunk_stops(stops::VgtkCore.LmStops, n::Integer, max_time::Real) =
    VgtkCore.LmStops(; max_iter = n, stops.ftol, stops.xtol, stops.xtol_rel, stops.stationarity_tol, max_time = Float64(max_time),
                     stops.residual_resolution)
"""
    _lm_log(cb, ϕ², path) -> callback

With a `path` (the algorithm's `log`, the LMs' `lm_log` keyword; empty: `cb` itself) `cb` (`nothing`: none) followed by a per-iteration
line (tab-separated) appended to that file, from the iteration's `LmIterationState`: iteration, cost, rel-RMS (√(2 cost / ϕ²)), the
best cost before the iteration, λ after its update, the linear solves of the iteration (1 = accepted at once; each further solve is a
rejected step that raised λ), and the DᵀD scaling's min / max / number of entries at its floor (1e-8 × max).
"""
function _lm_log(cb, ϕ², path::AbstractString)
    isempty(path) && return cb
    prev = Ref(0)
    return function (state)
        out = cb === nothing ? false : cb(state)
        nsolves = state.linear_solves
        sc = state.scaling
        open(path, "a") do io
            println(io, join((state.iteration, state.cost, sqrt(2 * state.cost / ϕ²), state.best_cost, state.damping, nsolves - prev[],
                              sc === nothing ? "" : minimum(sc), sc === nothing ? "" : maximum(sc),
                              sc === nothing ? "" : count(<=(1e-8 * maximum(sc) * (1 + 1e-12)), sc), time()), '\t'))
        end
        prev[] = nsolves
        return out
    end
end
"""A calibration LM's print-out line per iteration (`_lm_callback` with `verbose`): the iteration, the cost, the best cost before it, λ
and the time since the solve started (its `LmIterationState`)."""
function _lm_print(state)
    @printf("iter %4d: cost %.6e (best %.6e), λ %.3e, %.1f s\n", state.iteration, state.cost, state.best_cost, state.damping, state.elapsed)
    return false
end
