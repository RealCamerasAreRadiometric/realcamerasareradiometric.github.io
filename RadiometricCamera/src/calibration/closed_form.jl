# ---------------------------------------------------------------------------
# The linear/affine parameters: closed-form least squares and the per-pixel responses.
#
# Φ is linear in Θ_v (vignetting nodes), linear in Θ_l (SH coefficients)
# and affine in μ ∈ Θ_m, so each of these blocks is the solution of a linear least
# squares problem  Θ̂_q = argmin ‖𝛷 Θ_q − ϕ‖²  (q ∈ {v, l, u}). The design rows come from the
# per-pixel `pixel_parts`, which share every quadrature node and every random draw
# with the value render (exact linearity per realisation). The same parts give the
# analytic Jacobian columns of these blocks in the joint LM (per_view.jl, the parts kernel `_PartsAt`).
#
# Every per-pixel step runs on the state's backend (KernelAbstractions; the CPU backend is the same kernels, the test oracle): the
# parts (`all_parts`, Core's `pointwise!` of `_PixelPartsAt`), the design rows of the QR solves (`_design_row!`) and the reductions
# of the cost and the normal equations (`_radiometry_cost_row!`, `_radiometry_normal_row!`, launched by Core's reductions, the precedent
# of VisualGeometryToolkitEstimation's camera-regression refine). Only the solves' small systems are the host's.
# ---------------------------------------------------------------------------

# The closed-form solves of Θ_l and Θ_v are least squares by LinearAlgebra's column-pivoted (rank-revealing) QR on the design rows
# assembled on the device, `qr(A, ColumnNorm()) \ b`: the basic solution, which sets the coefficients of (numerically) dependent
# columns to zero. Vignetting nodes without support in the masked pixels (the frame corners outside a fisheye's image
# circle) make the plain QR fail (zero pivots → a non-finite solution) — with pivoting they stay at zero.

"""
    PixelParts{J, L3, NS, F}

Per-pixel quantities for the linear solves: for every sensor sample k the aperture-integrated
light responses `Bd[k], Bs[k] = Σ_l w_l (Bd, Bs)` (3 × J each) and the sensor weights `w[k]`
and positions `𝐩[k]`, in the float type `F` of the render settings. Everything the solves need follows without re-rendering:
`I_k(a, μ) = (μ Bd[k] + (1−μ) Bs[k]) a`, `Md = Σ_k w_k V(𝐩_k) Bd[k]`, `Ms` likewise (`_linear_rows`).
"""
struct PixelParts{J, L3, NS, F}
    𝐩::SVector{NS, SVector{2, F}}
    w::SVector{NS, F}
    Bd::SVector{NS, SMatrix{3, J, F, L3}}
    Bs::SVector{NS, SMatrix{3, J, F, L3}}
end

"""
    _sample_parts(view, mat, light, board, st, rays, k, pid, ks, 𝐩, scale, Z) -> (Bd, Bs)

THE aperture quadrature of the light responses at sensor sample `ks` (position `𝐩`) of pixel id `pid` of view `k`:
`Σ_l w_l (Bd, Bs)` of `radiance_parts` over the disc rule (`Z` the zero `3 × J` matrix of the accumulators' type; `scale` the global
aperture norm). [`pixel_parts`](@ref) calls it, and through it the parts kernels `_PixelPartsAt` and `_PartsAt`.
"""
@inline function _sample_parts(view::View, mat::Material, light::EnvironmentLight, board::TexturedTarget, st::RenderSettings, rays,
                               k::Integer, pid, ks::Integer, 𝐩, scale, Z, lrays = nothing)
    bd = board
    Bd = Z; Bs = Z
    rs = view.r
    for la in 1:n_disc(st.aperture)
        𝐚 = 𝐚_sample(st.aperture, la, rs)
        o, d = world_ray(view, 𝐩, 𝐚)
        sray, 𝐱 = _parts_hit(lrays, (ks - 1) * n_disc(st.aperture) + la, rays, o, d)
        (isfinite(sray) && sray > 0) || continue
        𝐛 = chart(bd, 𝐱)
        parts = radiance_parts(bd, mat, light, st.sampler, SampleKey(k, pid, ks, la), 𝐛, 𝐱, d)
        w = 𝐚_weight(st.aperture, la, rs) * scale
        Bd += w * parts.Bd
        Bs += w * parts.Bs
    end
    return Bd, Bs
end

"""The hit of a parts ray (`_sample_parts`): `intersect_board`, or — beside a radiance record of the same parameters (`lrays`, the LM's
parts kernel) — the record's hit of that ray (`_ray_hit`), so the height field's root is not found again; a ray the record did not
reach (zeros: it misses the board, or meets it from behind, where its parts are zero) is skipped."""
@inline _parts_hit(::Nothing, ri, rays, o, d) = intersect_board(rays, o, d)
@inline function _parts_hit(lr::RadianceRays, ri, rays, o::StaticVector{3}, d::StaticVector{3})
    𝐱, _, reached = _recorded_hit(lr, ri)
    return ifelse(reached, dot(𝐱 - o, d), oftype(d[1], NaN)), 𝐱
end

"""
    pixel_parts(u, v, k, camera, pose, mat::Material, light::EnvironmentLight, board::TexturedTarget, settings::RenderSettings) -> PixelParts

THE [`PixelParts`](@ref) of Julia's 1-based pixel `(u, v)` of photograph `k`, `camera` (a `RadiometricCameraModel`) at
`pose` (camera → world), in the settings' float type: the sensor and aperture quadratures evaluated once, so the closed-form solves
of `Θ_l`, `Θ_v` and μ ([`RadiometryBlocks`](@ref)) need no re-rendering. The sample keys are `k`'s, so the Monte Carlo radiance
matches [`Φ_pixel`](@ref)'s. The vignetting is not applied: it is one of the parts' linear factors.
"""
pixel_parts(u::Integer, v::Integer, k::Integer, camera::RadiometricCameraModel, pose::EuclideanMap, mat::Material, light::EnvironmentLight,
            board::TexturedTarget, st::RenderSettings) = _pixel_parts(u, v, k, _image_height(camera), View(pose, camera), mat, light, board, st)

"""The kernels' form of [`pixel_parts`](@ref) on a `View`. Device-safe: the parts kernels run it per pixel (`_PixelPartsAt` for the
closed-form solves, `_PartsAt` for the LM's analytic columns, which gives the pixel's radiance record `lrays` for its rays' hits,
`_parts_hit`)."""
@inline function _pixel_parts(u::Integer, v::Integer, k::Integer, H::Integer, view::View, mat::Material,
                             light::EnvironmentLight{<:SphericalHarmonicMap{L, C, J}}, board::TexturedTarget, st::RenderSettings{TS, NS},
                             lrays = nothing) where {L, C, J, TS, NS}
    F = _float_of(st)
    bd = board
    pid = pixel_id(u, v, H)
    scale = 1 / (π * view.r^2)
    𝐩s = ntuple(ks -> SVector{2, F}(𝐩_sample(st.sensor, ks, u, v)), Val(NS))
    rays = BoardRays(board)                                  # the warping's relief, once for the pixel's rays (bit for bit the per-ray form)
    BdBs = ntuple(ks -> _sample_parts(view, mat, light, board, st, rays, k, pid, ks, 𝐩s[ks], scale, zero(SMatrix{3, J, F, 3J}), lrays), Val(NS))
    return PixelParts{J, 3J, NS, F}(SVector{NS}(𝐩s), SVector{NS, F}(ntuple(ks -> F(st.sensor.weights[ks]), Val(NS))),
                                    SVector{NS}(ntuple(ks -> SMatrix{3, J, F, 3J}(BdBs[ks][1]), Val(NS))),
                                    SVector{NS}(ntuple(ks -> SMatrix{3, J, F, 3J}(BdBs[ks][2]), Val(NS))))
end

"""
    _linear_rows(p::PixelParts, vig, a, μ, dP = nothing) -> (L, G, JP)

THE analytic rows of the linear blocks at a pixel, from its parts at the vignetting `vig` (moved by `dP`, a step of its control
points, `vec` order: V is linear in them, `V(𝐩) + coefficient_jacobian(vig, 𝐩) ⋅ dP`), the SH coefficients `a` and μ:
- `L = μ M_d + (1−μ) M_s` (3 × J), the lighting columns ∂Φ_c/∂a[c, j];
- `G = Σ_j a[c, j] (M_d − M_s)[c, j]` (3), the μ column ∂Φ_c/∂μ;
- `JP` (3 × NV), the raw vignetting rows ∂Φ_c/∂P_q = Σ_s w_s B_q(𝐩_s) I_{s,c}, `I_{s,c} = Σ_j a[c, j] (μ B_d + (1−μ) B_s)[c, j]`.
The flux is `Φ_c = Σ_j a[c, j] L[c, j]` (`_rowdot(a, L)`). Device-safe; the parts kernel `_PartsAt` returns these, and every
closed-form solve reads them.
"""
@inline function _linear_rows(p::PixelParts{J, L3, NS}, vig, a::SMatrix{3, J}, μ, dP = nothing) where {J, L3, NS}
    F = eltype(p.w)
    NV = n_control_points(vig)
    light_acc = zero(SMatrix{3, J, F, 3 * J})            # μ B_d + (1−μ) B_s, V-weighted
    dif_acc = zero(SMatrix{3, J, F, 3 * J})              # B_d − B_s, V-weighted (μ column)
    vig_acc = zero(SMatrix{3, NV, F, 3 * NV})            # Σ_s w_s B_q(𝐩_s) I_{s,c}
    @inbounds for ks in 1:NS
        𝐩 = p.𝐩[ks]
        B = coefficient_jacobian(vig, 𝐩)
        V𝐩 = _moved_vignetting(vig, B, 𝐩, dP)
        Bd = p.Bd[ks]; Bs = p.Bs[ks]
        mix = μ * Bd + (1 - μ) * Bs
        ws = p.w[ks]
        light_acc += (ws * V𝐩) * mix
        dif_acc += (ws * V𝐩) * (Bd - Bs)
        Ic = _rowdot(a, mix)                                   # I_{s,c} = Σ_j a[c,j] mix[c,j]
        vig_acc += (ws * Ic) * B'
    end
    # the parts' float as the rows' type: the accumulators' element type is the promotion of the inputs', which inference does not fix
    return (SMatrix{3, J, F}(light_acc), SVector{3, F}(_rowdot(a, dif_acc)), SMatrix{3, NV, F}(vig_acc))
end
@inline _moved_vignetting(vig, B, 𝐩, ::Nothing) = vig(𝐩)
@inline _moved_vignetting(vig, B, 𝐩, dP) = vig(𝐩) + dot(B, dP)

"""
    RadiometryParts

The [`PixelParts`](@ref) of every masked pixel of every view (view order, then pixel order), the measured values `ϕ` (the
measured channels per pixel, the same order) and their rows' weights `w` (weighting.jl: `Unweighted()`, or 1/σ per row), on the
state's backend: what [`all_parts`](@ref) renders once and the closed-form solves read.
"""
struct RadiometryParts{P, R, W}
    parts::P
    ϕ::R
    w::W
end
Base.length(p::RadiometryParts) = length(p.parts)

"""
    all_parts(f::CalibrationData, θ::CalibrationParameters) -> RadiometryParts

The parts of every masked pixel, rendered once on the state's backend: Core's `pointwise!` of `_PixelPartsAt` (device/jacobian_kernels.jl) per view,
in Float64 on every backend, as the nodal heights' path (`_height_float`): the closed-form solves are exact least squares on them.
"""
function all_parts(f::CalibrationData, θ::CalibrationParameters)
    be = f.backend
    dc = _ensure_device!(f, θ)
    board = _height_board(f, θ)                                                  # the Float64 board on the backend
    G = Float64
    mat = VgtkCore.convert_eltype(G, θ.mat); light = VgtkCore.convert_eltype(G, θ.light); st = VgtkCore.convert_eltype(G, f.settings)
    R = _parts_type(light, st)
    bufs = [_launch_rows(_PixelPartsAt(Int32(k), Int32(_image_height(θ)), adapt(be, VgtkCore.convert_eltype(G, view_of(θ, k))), mat, light, board, st), be, dc.px[k])
            for k in 1:n_views(θ) if !isempty(f.px[k])]               # a view without masked pixels: no launch (ndrange 0 fails)
    KA.synchronize(be)
    flat = isempty(bufs) ? KA.allocate(be, G, 0) : reduce(vcat, bufs)
    return RadiometryParts(reinterpret(R, flat), adapt(be, reduce(vcat, f.ϕ; init = Float64[])), adapt(be, _stacked_weights(f.weights)))
end

# ---- the reductions: the cost and the normal equations of the linear blocks, one kernel each on the state's backend ----------------

"""
    _RadiometryLm

The linearization state of the closed-form blocks for Core's chart LM (`lmsolve_chart!`, NLLSsolver): the parts and the measured
values on the device, the base vignetting `vig`, SH coefficients `a` and μ, the vignetting's free directions `Z` (an `NV × NF`
device matrix: `vec(P) = vec(P₀) + Z y`, the gauge's centre direction absent), the measured channels `chs`, the layout `lay` of the
chart vector `x = [y; δa; w_μ]` and the work items' partial sums. Its two reductions (`residual_cost!`, `residual_normal_equations!`)
keep every row on the device; only the `N × N` system reaches the host.
"""
struct _RadiometryLm{P, R, W, G, A, Z, C, L, B}
    parts::P
    ϕ::R
    w::W
    vig::G
    a::A
    μ::Float64
    Z::Z
    chs::C
    lay::L
    partials::B
end

"""
    _radiometry_layout(nv, J, nc; light, mu, chart, mu_bounds) -> NamedTuple

The layout of the chart vector over the free blocks: `nv` vignetting directions at `1 … nv`, the measured channels' `nc × J` SH
coefficients from `ol + 1` (coefficient `(j, jc)` at `ol + (j − 1) nc + jc`) when `light`, μ at `om + 1` when `mu` — on the LM's
bounded chart `μ = lo + (hi − lo) σ(w)` (`_μ_chart`, `mu_bounds = (lo, hi)`: the state's, `material.mu_bounds`) when `chart`, else additive.
"""
function _radiometry_layout(nv::Integer, J::Integer, nc::Integer; light::Bool, mu::Bool, chart::Bool, mu_bounds)
    lo, hi = mu_bounds
    nl = light ? nc * J : 0
    return (; nv = Int(nv), ol = Int(nv), om = Int(nv + nl), nc = Int(nc), light, mu, chart, lo = Float64(lo), hi = Float64(hi), N = Int(nv + nl + (mu ? 1 : 0)))
end

"""The SH coefficients at the chart vector `x` (the measured channels moved, `lay`); the others as `a`."""
@inline function _trial_coefficients(lay, chs::NTuple{NC, Int}, a::SMatrix{3, J, T}, x) where {NC, J, T}
    lay.light || return a
    return SMatrix{3, J}(ntuple(Val(3 * J)) do q
        c = (q - 1) % 3 + 1; j = (q - 1) ÷ 3 + 1
        jc = 0
        for t in 1:NC
            chs[t] == c && (jc = t)
        end
        jc == 0 ? a[c, j] : a[c, j] + x[lay.ol + (j - 1) * NC + jc]
    end)
end
"""μ at the chart vector `x` (`lay`) and ∂μ/∂x_μ there (1 on the additive chart)."""
@inline function _trial_μ(lay, μ, x)
    lay.mu || return μ, zero(μ)
    lay.chart || return μ + x[lay.om + 1], one(μ)
    μ′ = _μ_chart(μ, x[lay.om + 1], lay.lo, lay.hi)
    return μ′, _μ_chart_scale(μ′, lay.lo, lay.hi)
end
"""The step of the nodes `Z y` (`y = x[1 … nv]`) as an `SVector{NV}`."""
@inline function _trial_points(lay, Z, x, ::Val{NV}) where {NV}
    return SVector{NV, Float64}(ntuple(Val(NV)) do l
        s = 0.0
        for q in 1:lay.nv
            s += Z[l, q] * x[q]
        end
        s
    end)
end

"""The residual of pixel `i`'s measured channel `jc` (channel `c`) and its chart-Jacobian row at the trial rows `(L, G, JP)`."""
@inline function _radiometry_row(lay, Z, rows, a′, dμ, ϕ, i, jc, c, ::Val{N}, ::Val{NV}) where {N, NV}
    L, G, JP = rows
    r = _rowdot(a′, L)[c] - ϕ[lay.nc * (i - 1) + jc]
    Jrow = SVector{N, Float64}(ntuple(Val(N)) do q
        if q <= lay.nv                                     # ∂/∂y_q = Σ_l Z[l, q] JP[c, l]
            s = 0.0
            for l in 1:NV
                s += Z[l, q] * JP[c, l]
            end
            s
        elseif q <= lay.om                                 # the SH coefficient (j, jc′) of the measured channel jc′
            t = q - lay.ol - 1
            (t % lay.nc + 1) == jc ? Float64(L[c, t ÷ lay.nc + 1]) : 0.0
        else                                               # μ
            Float64(G[c]) * dμ
        end
    end)
    return r, Jrow
end

# Work item `w`'s share of the cost: Σ r² over its pixels `w, w + W, …` and their measured channels, into `out[1, w]` (each row under
# its weight `ω`, `_whiten`).
Base.@propagate_inbounds function _radiometry_cost_row!(out, w, parts, ϕ, ω, chs::NTuple{NC, Int}, vig, a, μ, Z, lay, x) where {NC}
    NV = n_control_points(vig)
    dP = _trial_points(lay, Z, x, Val(NV)); a′ = _trial_coefficients(lay, chs, a, x); μ′, _ = _trial_μ(lay, μ, x)
    s = 0.0
    for i in w:size(out, 2):length(parts)
        L, _, _ = _linear_rows(parts[i], vig, a′, μ′, dP)
        Φ = _rowdot(a′, L)
        for jc in 1:NC
            e = NC * (i - 1) + jc
            r = _whiten(ω, e, Φ[chs[jc]] - ϕ[e])
            s += r * r
        end
    end
    out[1, w] = s
    return nothing
end
# Work item `w`'s share of the normal equations: Σ r², Σ Jᵀr and Σ JᵀJ over its pixels into column `w` (the cost, then the N gradient
# entries, then the N × N Hessian column-major), each row and its Jacobian row under the row's weight `ω` (`_whiten`).
Base.@propagate_inbounds function _radiometry_normal_row!(out, w, parts, ϕ, ω, chs::NTuple{NC, Int}, vig, a, μ, Z, lay, x::SVector{N, Float64}) where {NC, N}
    NV = n_control_points(vig)
    dP = _trial_points(lay, Z, x, Val(NV)); a′ = _trial_coefficients(lay, chs, a, x); μ′, dμ = _trial_μ(lay, μ, x)
    s = 0.0
    g = zero(SVector{N, Float64})
    H = zero(SMatrix{N, N, Float64})
    for i in w:size(out, 2):length(parts)
        rows = _linear_rows(parts[i], vig, a′, μ′, dP)
        for jc in 1:NC
            r0, Jr0 = _radiometry_row(lay, Z, rows, a′, dμ, ϕ, i, jc, chs[jc], Val(N), Val(NV))
            e = NC * (i - 1) + jc
            r = _whiten(ω, e, r0); Jr = _whiten(ω, e, Jr0)
            s += r * r
            g += Jr * r
            H += Jr * Jr'
        end
    end
    out[1, w] = s
    for q in 1:N
        out[1 + q, w] = g[q]
    end
    for b in 1:N, q in 1:N
        out[1 + N + (b - 1) * N + q, w] = H[q, b]
    end
    return nothing
end

# The work items of the reductions: Estimation's camera-regression refine's (enough to fill an accelerator, few enough that their columns sum
# in one reduction).
_reduction_items(npx::Integer) = max(1, min(Int(npx), VgtkEstimation._REDUCTION_ITEMS))

"""The chart LM's state of the closed-form blocks of `s` on its parts (`lay`: `_radiometry_layout`)."""
function _radiometry_lm(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts, lay)
    be = f.backend
    Z = adapt(be, Matrix{Float64}(vignetting_basis(f, θ)))
    return _RadiometryLm(parts.parts, parts.ϕ, parts.w, VgtkCore.convert_eltype(Float64, θ.camera.vignetting),
                         SMatrix{3, sh_count(max_degree(θ.light)), Float64}(coefficients(θ.light)), Float64(θ.mat.diffuse_weight), Z, residual_channels(f), lay,
                         VgtkCore.device_zeros(Float64, be, 1 + lay.N + lay.N^2, _reduction_items(length(parts))))
end

# The launch and the reduction are Core's (reduce_cost, reduce_normal_equations with the row functor: one kernel over the work
# items on the partials' backend, then the 1 + N + N² sums per work item reduced).
VgtkCore.residual_cost!(_, lm::_RadiometryLm, x) =
    VgtkCore.reduce_cost(_radiometry_cost_row!, lm.parts, lm.ϕ, lm.w, lm.chs, lm.vig, lm.a, lm.μ, lm.Z, lm.lay, x; partials = lm.partials)
VgtkCore.residual_normal_equations!(_, _J, lm::_RadiometryLm, x::SVector{N, Float64}) where {N} =
    VgtkCore.reduce_normal_equations(_radiometry_normal_row!, Val(N), lm.parts, lm.ϕ, lm.w, lm.chs, lm.vig, lm.a, lm.μ, lm.Z, lm.lay, x; partials = lm.partials)

"""
    radiometry_normal_equations(f, θ, parts; light = true, vignetting = true, mu = true, chart = true) -> (½‖r‖², Jᵀr, JᵀJ)

The normal equations of the closed-form blocks at the state's parameters (the chart origin): one reduction kernel on the state's
backend (`_radiometry_normal_row!`). The unknowns are laid out as [`_lm_radiometry!`](@ref)'s (`_radiometry_layout`).
"""
function radiometry_normal_equations(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts; light::Bool = true, vignetting::Bool = true,
                                     mu::Bool = true, chart::Bool = true)
    lay = _radiometry_layout(vignetting ? size(vignetting_basis(f, θ), 2) : 0, sh_count(max_degree(θ.light)), n_channels(f); light, mu, chart, f.mu_bounds)
    return _normal_equations_at_origin(_radiometry_lm(f, θ, parts, lay), Val(lay.N))
end
_normal_equations_at_origin(lm, ::Val{N}) where {N} = VgtkCore.residual_normal_equations!(nothing, nothing, lm, zero(SVector{N, Float64}))

"""The objective `½ Σ ‖Φ − ϕ‖²` evaluated from the parts at the current (V, a, μ) — the same quadrature as the render, so this is the
exact cost without re-rendering (used to follow the block sweeps): the cost reduction at the chart origin, nothing free (the chart
vector one unread placeholder coordinate: a device launch's argument conversion rebuilds an empty `SVector` as `SVector{0, Union{}}`)."""
parts_cost(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts) =
    VgtkCore.residual_cost!(nothing, _radiometry_lm(f, θ, parts, _radiometry_layout(0, 0, n_channels(f); light = false, mu = false, chart = false, f.mu_bounds)), zero(SVector{1, Float64}))

# ---- the design rows of the QR solves, assembled on the device --------------------------------------------------------------------

# Pixel `i`'s design rows of the block `blk` (`:light`, the J lighting columns; `:vig`, the NV raw vignetting columns), one row per
# measured channel at `A[NC (i − 1) + jc, :]`.
Base.@propagate_inbounds function _design_row!(A, i, parts, chs::NTuple{NC, Int}, vig, a, μ, ::Val{blk}) where {NC, blk}
    L, _, JP = _linear_rows(parts[i], vig, a, μ)
    for jc in 1:NC
        c = chs[jc]; row = NC * (i - 1) + jc
        if blk === :light
            for j in 1:size(L, 2)
                A[row, j] = L[c, j]
            end
        else
            for q in 1:size(JP, 2)
                A[row, q] = JP[c, q]
            end
        end
    end
    return nothing
end
"""The design matrix of the block `blk` (`Val(:light)`, `Val(:vig)`) at the state's parameters: one row per pixel and measured
channel (`parts`' order), assembled on the backend by `_design_row!` and returned to the host for the QR."""
function _design_matrix(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts, blk::Val)
    chs = residual_channels(f)
    vig = VgtkCore.convert_eltype(Float64, θ.camera.vignetting)
    a = SMatrix{3, sh_count(max_degree(θ.light)), Float64}(coefficients(θ.light))
    n = blk === Val(:light) ? size(a, 2) : n_control_points(vig)
    A = VgtkCore.device_zeros(Float64, f.backend, length(chs) * length(parts), n)
    _launch_row_job!(_design_row!, A, length(parts), parts.parts, chs, vig, a, Float64(θ.mat.diffuse_weight), blk)
    return Array(A)
end

"""The least-squares system `(A, b)` of the block `blk` on the host: the design matrix (`_design_matrix`) and the measured values,
every row under its weight (`_whiten`: whitened rows, or as measured)."""
function _weighted_system(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts, blk::Val)
    ω = adapt(Array, parts.w)                                  # the host's copy (`Unweighted()` as it is)
    return _whiten(ω, _design_matrix(f, θ, parts, blk)), _whiten(ω, Array(parts.ϕ))
end

"""
    _update_lighting!(f, θ, parts = all_parts(f, θ)) -> s

Linear least squares for the SH coefficients (per measured channel, `residual_channels`), all views stacked; a single-channel
state's other channels follow the measured one (`tie_channels`).
"""
function _update_lighting!(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts = all_parts(f, θ))
    a = Matrix{Float64}(coefficients(θ.light))
    chs = residual_channels(f); nc = length(chs)
    A, b = _weighted_system(f, θ, parts, Val(:light))
    for (jc, c) in enumerate(chs)
        a[c, :] = qr(A[jc:nc:end, :], ColumnNorm()) \ b[jc:nc:end]
    end
    θ.light = f.channel == 0 ? with_coefficients(θ.light, a) : tie_channels(with_coefficients(θ.light, a), f.channel)
    return θ
end

# ---- the penalty weight: the stack's one convention (Core's `penalty_scale` and `mean_positive_eigenvalue`, Estimation's GCV) ----

"""
    vignetting_smoothness(f, θ) -> (D, m̄)

The vignetting solve's smoothness operator: the third differences Δ³ of the nodes along the lattice axes
(Core's `difference_rows(smoothness_nodes(vig, rule), 3)`: the windows of 4 consecutive free nodes) as a dense matrix over
`vec(control_points(vig))`, and Core's `mean_positive_eigenvalue` `m̄` of it (β's normalization; `NaN` without a row).
"""
function vignetting_smoothness(f::CalibrationData, θ::CalibrationParameters)
    D = VgtkCore.difference_rows(smoothness_nodes(θ.camera.vignetting, f.vig_rule), 3)
    size(D, 1) == 0 && return Matrix(D), NaN
    return Matrix(D), VgtkCore.mean_positive_eigenvalue(D)
end

"""
    _update_vignetting!(f, θ, parts = all_parts(f, θ); smoothness_weight = 0.0, gcv_record = nothing) -> s

Linear least squares for the vignetting nodes (channels stacked, achromatic V),
with the gauge `V(𝐜) = 1` held exactly (the centre's direction absent from `vignetting_basis(f, θ)`, `gauge_basis`). The unknowns are the free directions of `vignetting_basis(f, θ)`: a
tensor surface's nodes outside the run's imaged region are dependent (Core's `BoundaryRule`), not unknowns — a radial
profile's below `support_min` of the data weight likewise (`RadialSupportRule`) — and a radial profile's first two nodes move
together (the pole tie, V′(0) = 0).

`smoothness_weight` β > 0 adds the P-spline penalty `λ ‖D vec(P)‖²` over the free nodes (`vignetting_smoothness`:
third differences along the lattice axes, so a field quadratic along each axis costs nothing) with `λ = β Σϕ² / N / m̄` (`N` the
free directions, `m̄` the mean positive eigenvalue of `DᵀD`: Core's `penalty_scale`), so β means what the grid smoothness's does. `:gcv`
selects β by Estimation's generalized cross-validation (`gcv_curve`, `gcv_select`) on this solve's own linear system, and
pushes the selection to `gcv_record` (a vector; `nothing`: not recorded). `0` is the unpenalized solve, unchanged; without a window
of 4 free nodes along either axis there is no row and the solve is the unpenalized one.
"""
function _update_vignetting!(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts = all_parts(f, θ); smoothness_weight::Union{Real, Symbol} = 0.0, gcv_record = nothing)
    VgtkEstimation.check_penalty_weight(smoothness_weight, "_update_vignetting!: smoothness_weight")
    n_cp = n_control_points(θ.camera.vignetting)
    A, b = _weighted_system(f, θ, parts, Val(:vig))
    # over the free directions Z (`vignetting_basis`: the interior / supported nodes, the dependent ones following them; a
    # radial profile's pole tie): vec(P) = P₀ + Z y
    Z = vignetting_basis(f, θ); P0 = vec(Matrix{Float64}(control_points(θ.camera.vignetting)))
    D, m̄ = smoothness_weight == 0 ? (zeros(0, n_cp), NaN) : vignetting_smoothness(f, θ)
    if size(D, 1) == 0                                    # unpenalized: the solve as before the penalty existed
        y = qr!(A * Z, ColumnNorm()) \ (b - A * P0)
    else                                                  # min ‖AZ y − r‖² + λ ‖DZ y − c‖², r = b − A P₀, c = −D P₀
        AZ = A * Z; r = b - A * P0
        DZ = D * Z; c = -D * P0
        scale = VgtkCore.penalty_scale(data_norm2(f), size(Z, 2), m̄)       # λ = β Σϕ² / N / m̄
        sel = if smoothness_weight === :gcv
            VgtkEstimation.gcv_select(VgtkEstimation.gcv_curve(AZ' * AZ, AZ' * r, dot(r, r), DZ' * DZ, DZ' * c, size(AZ, 1)), scale)
        else
            (; β = Float64(smoothness_weight), λ = smoothness_weight * scale)
        end
        sλ = sqrt(sel.λ)
        y = qr!(vcat(AZ, sλ .* DZ), ColumnNorm()) \ vcat(r, sλ .* c)
        smoothness_weight === :gcv && gcv_record !== nothing && push!(gcv_record, (; sel..., m̄, rows = size(D, 1), n_free = size(Z, 2)))
    end
    set_vignetting!(θ, with_control_points(θ.camera.vignetting, P0 + Z * y))
    return θ
end

"""
    _update_diffuse_weight!(f, θ, parts = all_parts(f, θ)) -> s

Affine one-parameter least squares for μ: `Φ = S + μ (D − S)`, clamped to the state's `mu_bounds`: one Gauss–Newton step (exact: Φ is affine
in μ) of the μ-only normal equations, the reduction kernel's (`radiometry_normal_equations`, the additive chart).
"""
function _update_diffuse_weight!(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts = all_parts(f, θ))
    _, g, H = radiometry_normal_equations(f, θ, parts; light = false, vignetting = false, mu = true, chart = false)
    H[1, 1] > 0 && (θ.mat = with_μ(θ.mat, clamp(θ.mat.diffuse_weight - g[1] / H[1, 1], f.mu_bounds...)))
    return θ
end

"""
    _lm_radiometry!(f, θ, parts; damping, stops = LmStops(; max_iter = 40, ftol = 1e-12), light = true, vignetting = true, mu = true,
                         verbose = false) -> (ReturnCode.T, StopRule)

JOINT least squares over the closed-form blocks — lighting `Θ_l`, vignetting `Θ_v` (on the null space of the gauge V(𝐜) = 1, always
held) and μ — by the stack's Levenberg–Marquardt, Core's chart LM (`lmsolve_chart!`: NLLSsolver, at the optimizer's `damping`), on the
precomputed `parts` (no re-rendering). Its normal equations are one reduction kernel on the state's backend
(`_radiometry_normal_row!`; the precedent of Estimation's camera-regression refine): only the `N × N` step is the host's.
The flux is BILINEAR in (`Θ_v`, `Θ_l`) and affine in μ, so block coordinate descent ([`RadiometryBlockDescent`](@ref)) creeps
along the coupled direction: on synthetic data generated by the model itself it stopped 8 % away from the true V after 10 sweeps
(FX3 fisheye, 2026-10-13); the joint step converges quadratically near the minimum. Exact Jacobian rows from the parts (`_linear_rows`,
the parts kernel's): `∂Φ_c/∂P_q = Σ_s w_s B_q(𝐩_s) I_{s,c}`, `∂Φ_c/∂a[c,j] = (μ M_d + (1−μ) M_s)[c,j]`,
`∂Φ_c/∂μ = Σ_j a[c,j] (M_d − M_s)[c,j]`. μ moves on the LM's bounded chart `μ = lo + (hi − lo) σ(w)` (`material.mu_bounds`, `_μ_chart`).
`stops` are Core's LM stop settings (`LmStops`: 40 accepted iterations, a relative cost decrease of 1e-12, the stall
postcondition's tolerance Core's of record), their residual resolution the state's (`_lm_stops`). Returns the stack's termination record (Core's `ReturnCode` and `StopRule`);
with nothing free, `(ReturnCode.Success, ClosedForm)`.
"""
function _lm_radiometry!(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts; damping::VgtkCore.LmDamping,
                              stops::VgtkCore.LmStops = VgtkCore.LmStops(; max_iter = 40, ftol = 1e-12), light::Bool = true,
                              vignetting::Bool = true, mu::Bool = true, verbose::Bool = false)
    Z = vignetting_basis(f, θ)                                             # the free directions (interior nodes / the pole tie; the gauge's centre direction absent)
    lay = _radiometry_layout(vignetting ? size(Z, 2) : 0, sh_count(max_degree(θ.light)), n_channels(f); light, mu, chart = true, f.mu_bounds)
    lay.N == 0 && return (VgtkCore.ReturnCode.Success, VgtkCore.ClosedForm)
    lm = _radiometry_lm(f, θ, parts, lay)
    x, iterations, record = _radiometry_chart_lm(lm, Val(lay.N), _lm_stops(f, stops), damping)
    # the solution onto the state, as the kernels' trial: V's nodes P₀ + Z y, the measured channels' SH coefficients (the
    # others follow them), μ on its chart
    vignetting && set_vignetting!(θ, with_control_points(θ.camera.vignetting, vec(Matrix{Float64}(control_points(θ.camera.vignetting))) + Z * x[1:lay.nv]))
    if light
        a1 = Matrix(_trial_coefficients(lay, lm.chs, lm.a, x))
        θ.light = f.channel == 0 ? with_coefficients(θ.light, a1) : tie_channels(with_coefficients(θ.light, a1), f.channel)
    end
    mu && (θ.mat = with_μ(θ.mat, first(_trial_μ(lay, lm.μ, x))))
    verbose && @printf("      joint LS: %d unknowns (V %d, light %d, μ %d), %d iterations, %s / %s\n", lay.N, lay.nv, lay.om - lay.ol,
                       Int(mu), iterations, record[1], record[2])
    return record
end
# The chart LM at N unknowns — a function barrier static in N, at the stops (their residual resolution the state's, `_lm_stops`)
# and the damping. The workspace holds no rows: the state's reductions keep them on its device (one placeholder row, NLLSsolver's
# problem needs one), as Estimation's camera-regression refine.
function _radiometry_chart_lm(lm::_RadiometryLm, ::Val{N}, stops::VgtkCore.LmStops, damping::VgtkCore.LmDamping) where {N}
    ews = VgtkCore.ManifoldLmWorkspace(1, Val(N))
    x, _, _, iterations, record = VgtkCore.lmsolve_chart!(ews, lm, stops, damping)
    return x, iterations, record
end

"""
    RadiometryBlocks

The closed-form blocks of a calibration's [`CalibrationParameters`](@ref), the estimate of a radiometry solve: the lighting `light` (Θ_l,
VisualRadiometryToolkitCore's `EnvironmentLight`), the vignetting `vignetting` (Θ_v) and the diffuse weight `diffuse_weight`
(μ). Solved on `EstimationProblem(RadiometryBlocks, (f, θ, parts), (; light, vignetting, mu))` — the data `f`, the parameters `θ`,
their masked pixels' parts rendered once ([`all_parts`](@ref): the geometry, the material's other parameters and the quadrature
fixed, so every solve is exact for them), and which of the three blocks are free (each `true` by default) — by
[`RadiometryBlockDescent`](@ref) or [`JointRadiometry`](@ref). The solve writes the estimate into `θ` (as a
[`RadiometricRefinement`](@ref) refines its `u0` in place) and returns the `FitResult`: `u` the blocks, `rms` the parts' relative RMS
(`√(2 c / Σϕ²)`, the cost `c` in `metadata.cost`), `stats.nsteps` the sweeps run, `metadata.gcv` the vignetting smoothness's
weight selections (`[vignetting.smoothness] weight = "gcv"`).
"""
struct RadiometryBlocks{L, V}
    light::L
    vignetting::V
    diffuse_weight::Float64
end
RadiometryBlocks(θ::CalibrationParameters) = RadiometryBlocks(θ.light, θ.camera.vignetting, Float64(θ.mat.diffuse_weight))

"""
    RadiometryBlockDescent(; sweeps = 1, reltol = 1e-4, vignetting_smoothness = 0.0, verbose = false)

Block coordinate descent over the closed-form blocks ([`RadiometryBlocks`](@ref)), a fit: per sweep the lighting `Θ_l` (linear least
squares per measured channel, all views stacked; a single-channel state's other channels following the measured one), the
vignetting `Θ_v` (linear least squares over the free directions of `vignetting_basis`, the gauge `V(𝐜) = 1` held exactly; with a
`vignetting_smoothness` β > 0 the P-spline penalty of its third differences, λ = β Σϕ²/N/m̄, Core's `penalty_scale`; `:gcv`: β by
Estimation's generalized cross-validation on the solve's own system) and μ (one exact Gauss–Newton step of the affine μ, clamped to the
state's `mu_bounds`), each block when free. The blocks are coupled (the flux is bilinear in V and Θ_l), hence `sweeps` > 1 until the
parts cost falls by less than `reltol` of its value (0 sweeps: the blocks as they are). `verbose` prints the parts cost per sweep.
"""
struct RadiometryBlockDescent{W <: Union{Real, Symbol}} <: VgtkCore.AbstractEstimationAlgorithm
    sweeps::Int
    reltol::Float64
    vignetting_smoothness::W
    verbose::Bool
end
function RadiometryBlockDescent(; sweeps::Integer = 1, reltol::Real = 1e-4, vignetting_smoothness::Union{Real, Symbol} = 0.0, verbose::Bool = false)
    sweeps >= 0 || throw(ArgumentError("RadiometryBlockDescent: sweeps ≥ 0; got $sweeps"))
    VgtkEstimation.check_penalty_weight(vignetting_smoothness, "RadiometryBlockDescent: vignetting_smoothness")
    return RadiometryBlockDescent(Int(sweeps), Float64(reltol), vignetting_smoothness, verbose)
end
VgtkCore.estimator_kind(::RadiometryBlockDescent) = VgtkCore.FitKind()

"""
    JointRadiometry(; damping, sweeps = 1, reltol = 1e-4, verbose = false)

The closed-form blocks ([`RadiometryBlocks`](@ref)) JOINTLY, a fit: per sweep one Levenberg–Marquardt over the lighting, the
vignetting (on the null space of its gauge) and μ (on its bounded chart) at the `damping` (a [`RadiometricCalibration`](@ref)'s
optimizer's), on the parts — block coordinate descent creeps along the coupled V × lighting direction (on synthetic data generated by
the model itself it stopped 8 % away from the true V after 10 sweeps, FX3 fisheye, 2026-10-13), the joint step converges quadratically
near the minimum. `sweeps`, `reltol` and `verbose` as [`RadiometryBlockDescent`](@ref)'s. It has no smoothness penalty.
"""
struct JointRadiometry{D <: VgtkCore.LmDamping} <: VgtkCore.AbstractEstimationAlgorithm
    damping::D
    sweeps::Int
    reltol::Float64
    verbose::Bool
end
function JointRadiometry(; damping::VgtkCore.LmDamping, sweeps::Integer = 1, reltol::Real = 1e-4, verbose::Bool = false)
    sweeps >= 0 || throw(ArgumentError("JointRadiometry: sweeps ≥ 0; got $sweeps"))
    return JointRadiometry(damping, Int(sweeps), Float64(reltol), verbose)
end
VgtkCore.estimator_kind(::JointRadiometry) = VgtkCore.FitKind()

"""The radiometry algorithms of a calibration ([`RadiometricCalibration`](@ref)'s `radiometry`)."""
const RadiometrySolve = Union{RadiometryBlockDescent, JointRadiometry}

# the sweep schedule and the printing of a radiometry algorithm, set (its setters, beside the structs: a calibration runs its
# configured algorithm with the sweeps and the smoothness of each of its solves)
_with_schedule(a::RadiometryBlockDescent; sweeps = a.sweeps, vignetting_smoothness = a.vignetting_smoothness, verbose = a.verbose) =
    RadiometryBlockDescent(; sweeps, a.reltol, vignetting_smoothness, verbose)
_with_schedule(a::JointRadiometry; sweeps = a.sweeps, vignetting_smoothness = 0.0, verbose = a.verbose) =
    (_check_joint_smoothness(vignetting_smoothness); JointRadiometry(; a.damping, sweeps, a.reltol, verbose))
_check_joint_smoothness(w) = w == 0 ||
    throw(ArgumentError("JointRadiometry has no vignetting smoothness penalty; `vignetting.smoothness` weight = $(repr(w)) (`sweep.joint` with a smoothness)"))

const _BLOCKS_ALL = (; light = true, vignetting = true, mu = true)
_radiometry_blocks(::VgtkCore.NullParameters) = _BLOCKS_ALL
_radiometry_blocks(p::NamedTuple) = (; _BLOCKS_ALL..., p...)

function VgtkCore.__solve(prob::VgtkCore.EstimationProblem{RadiometryBlocks, <:Tuple{CalibrationData, CalibrationParameters, RadiometryParts}}, alg::RadiometrySolve)
    f, θ, parts = prob.f
    blocks = _radiometry_blocks(prob.p)
    gcv = NamedTuple[]
    n, record = _radiometry_sweeps!(f, θ, parts, alg, blocks, gcv)
    c = parts_cost(f, θ, parts)
    return VgtkCore.FitResult(RadiometryBlocks(θ), record..., VgtkCore.SolveStats(; nsteps = n), sqrt(2c / data_norm2(f)),
                              length(parts.ϕ), (; cost = c, gcv))
end

# The sweeps of `alg` over the free `blocks` (`(; light, vignetting, mu)`) on `parts`: `(n, (retcode, stop_rule))`, `n` the sweeps
# run — a block descent's `(Success, ClosedForm)` after one sweep, `CostDecrease` at the relative tolerance, `IterationCap` at
# `sweeps`; the joint LM's last record. The vignetting smoothness's GCV selections are pushed to `gcv`.
function _radiometry_sweeps!(f::CalibrationData, θ::CalibrationParameters, parts::RadiometryParts, alg::RadiometrySolve, blocks, gcv)
    sweeps, verbose = alg.sweeps, alg.verbose
    c_prev = sweeps > 1 || verbose ? parts_cost(f, θ, parts) : NaN
    verbose && @printf("      sweep: parts cost %.6e at the start\n", c_prev)
    record = (VgtkCore.ReturnCode.Success, sweeps == 1 ? VgtkCore.ClosedForm : VgtkCore.IterationCap)
    n = 0
    for sweep in 1:sweeps
        r = _radiometry_sweep!(f, θ, parts, alg, blocks, gcv)
        alg isa JointRadiometry && (record = r)
        n = sweep
        (sweeps > 1 || verbose) || break
        c_now = parts_cost(f, θ, parts)
        verbose && @printf("      sweep %d: parts cost %.6e (%.2e relative change)\n", sweep, c_now, (c_prev - c_now) / max(c_prev, eps()))
        if (c_prev - c_now) < alg.reltol * c_prev
            alg isa RadiometryBlockDescent && sweep < sweeps && (record = (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease))
            break
        end
        c_prev = c_now
    end
    return n, record
end
# one sweep: the free blocks in turn, or jointly (lighting, V and μ by Levenberg–Marquardt on the parts: no creeping along the
# V × lighting direction)
function _radiometry_sweep!(f::CalibrationData, θ::CalibrationParameters, parts, alg::RadiometryBlockDescent, blocks, gcv)
    blocks.light && _update_lighting!(f, θ, parts)
    blocks.vignetting && _update_vignetting!(f, θ, parts; smoothness_weight = alg.vignetting_smoothness, gcv_record = gcv)
    blocks.mu && _update_diffuse_weight!(f, θ, parts)
    return nothing
end
_radiometry_sweep!(f::CalibrationData, θ::CalibrationParameters, parts, alg::JointRadiometry, blocks, _gcv) =
    _lm_radiometry!(f, θ, parts; alg.damping, blocks.light, blocks.vignetting, blocks.mu, alg.verbose)

"""
    _update_radiometry!(f, θ, alg, blocks; parts = all_parts(f, θ), gcv_record = nothing) -> θ

THE radiometry solve of a calibration: `solve(EstimationProblem(RadiometryBlocks, (f, θ, parts), blocks), alg)` (θ in place), its GCV selections
appended to `gcv_record` (a vector; `nothing`: not kept).
"""
function _update_radiometry!(f::CalibrationData, θ::CalibrationParameters, alg::RadiometrySolve, blocks::NamedTuple; parts::RadiometryParts = all_parts(f, θ), gcv_record = nothing)
    r = VgtkCore.solve(VgtkCore.EstimationProblem(RadiometryBlocks, (f, θ, parts), blocks), alg)
    gcv_record === nothing || append!(gcv_record, r.metadata.gcv)
    return θ
end
