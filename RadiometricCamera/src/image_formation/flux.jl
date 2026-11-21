# ---------------------------------------------------------------------------
# Radiant flux of pixel 𝐮 (paper eq. rad-flux-with-vignetting: the aperture integral divided by |𝒜| = πr²), with the aperture
# integral pulled back to the fixed unit disc 𝒟 (𝐚 = r â, d𝐚 = r² dâ):
#
#   Φ_c(𝐮; Θ̄) ≈ 1/(πr²) ∫_{ℐ_𝐮} V(𝐩; Θ_v) ∫_𝒟 r² L_c(π̂⁻¹(𝐩, r â; Θ_c), π̂⁻¹(𝐩, r â; Θ_c) − r â; Θ_r) dâ d𝐩
#              =  1/π    ∫_{ℐ_𝐮} V(𝐩; Θ_v) ∫_𝒟    L_c(π̂⁻¹(𝐩, r â; Θ_c), π̂⁻¹(𝐩, r â; Θ_c) − r â; Θ_r) dâ d𝐩
#
# The r² of the substitution cancels against the normalisation: the rule's weights are r² w̃ (Σ w̃ = π) and Φ_pixel divides its sum
# by πr², so r is no scale of the flux and enters through the nodes r â alone (the defocus).
#
# The whole cos⁴α(𝐚 − 𝐩)/s′² obliquity (cos-fourth law) is absorbed into the vignetting spline V — a
# pure sensor-position factor: the small intra-aperture variation of cos α (order sin θ · r/d_s) is
# dropped, the same approximation the cos³α/s′² cancellation already makes. This removes the image
# distance from the model (it entered only through cos α) and with it the flat d_s ↔ V direction.
#
# evaluated with the deterministic sensor/aperture quadrature rules of quadrature.jl
# (sensor rule = mean over ℐ_𝐮, see SensorQuadrature) and the Monte-Carlo radiance
# estimator of radiance.jl. Only masked pixels are rendered: a pixel is kept when its
# whole circle of confusion (every footprint corner × every aperture border sample)
# lands on the board with padding δ, so no boundary term appears: the r-dependence of the domain
# sits in the integrand (r² w̃_l, r â_l), and ∂Φ/∂r is an interior derivative like every other
# (a dual seed on r).
# ---------------------------------------------------------------------------

"""
    RenderSettings

Quadrature rules and radiance sampler shared by all renders. Φ is the aperture integral divided by |𝒜| = πr²: the aperture's
radius is then no scale of the flux, so it cannot trade against the lighting's.

    RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 8, seed = 42, hammersley = false)
"""
struct RenderSettings{T, NS, NA, NB}
    sensor::SensorQuadrature{T, NS}
    aperture::ApertureQuadrature{T, NA, NB}
    sampler::RadianceSampler
end
function RenderSettings(; n_𝐩::Integer = 2, n_ρ::Integer = 3, n_φ::Integer = 8, N::Integer = 8, seed::Integer = 42,
                        hammersley::Bool = false, T = Float64)
    return RenderSettings(SensorQuadrature{T}(n_𝐩), ApertureQuadrature{T}(n_ρ, n_φ), RadianceSampler(N; seed, hammersley))
end

"""Linear pixel id used in the sample keys (column-major over the `H × W` image)."""
@inline pixel_id(u::Integer, v::Integer, H::Integer) = UInt64((u - 1) * H + v)

"""Per-sensor-sample seed of the hoisted `t̂` value: slot `fbase + ks` of a dual (the focus chain rule —
`∂Φ/∂t̂_i = Σ_ks ∂Φ/∂t̂(𝐩_ks) · gᵢ(𝐩_ks)` with the TPS nodal weights `g` applied on the host). `nothing`: no seed."""
@inline _t̂_seeded(x, ::Nothing, ks::Integer) = x
"""The unit partial of slot `slot` as a dual of `P` partials (zero when the slot is outside `1:P`: a sentinel or another chunk's)."""
@inline _unit_partial(::Type{Tag}, ::Type{V}, slot::Integer, ::Val{P}) where {Tag, V, P} =
    ForwardDiff.Dual{Tag}(zero(V), ForwardDiff.Partials(ntuple(i -> i == slot ? one(V) : zero(V), Val(P))))
"""
    _seed_tangents(x) -> (e₁, e₂)

An orthonormal basis of the plane across the direction `x` (any length; Frisvad's: smooth in `x` everywhere but at −ẑ, the backward
axis no lens images, so the device's Float32 and the host's Float64 direction of a sample give the same basis up to rounding, with
no branch to fall on different sides of).
"""
@inline function _seed_tangents(x::StaticVector{3})
    n = normalize(x)
    a = 1 / (1 + n[3])
    b = -n[1] * n[2] * a
    return SVector(1 - n[1] * n[1] * a, b, -n[1]), SVector(b, 1 - n[2] * n[2] * a, -n[2])
end
"""Chief-direction seed (grid-lens chain rule): the raw direction `x = chief_ray(𝐩_ks)` moved across itself by duals with unit partials
in the 2 slots `base + 2(ks−1) + c`, `x + ε₁e₁ + ε₂e₂` (`_seed_tangents`); the dual type is taken from the (already dual) hoisted
`t̂` value. The flux sees a chief direction only through its normalisation, so its gradient in `x` has no part along `x` and the two
slots carry all of it: `∂Φ/∂x = (∂Φ/∂ε₁)e₁ + (∂Φ/∂ε₂)e₂` (`_dir_gradient`) — the chain rule, exact; the patch columns' parameters
are untouched (the host fans this gradient out to them as before)."""
@inline _dir_seeded(x, ::Nothing, ks::Integer, _) = x
@inline function _dir_seeded(x::StaticVector{3}, base::Integer, ks::Integer, ::ForwardDiff.Dual{Tag, V, P}) where {Tag, V, P}
    # add-a-seed (like _t̂_seeded): keeps existing partials when the chief ray is itself dual (per-view
    # lens AD); out-of-chunk/sentinel slots contribute all-zero partials — branch-free for the device
    b = base + 2 * (ks - 1)
    e1, e2 = _seed_tangents(ForwardDiff.value.(x))
    ε1 = _unit_partial(Tag, V, b + 1, Val(P)); ε2 = _unit_partial(Tag, V, b + 2, Val(P))
    return SVector(ntuple(c -> x[c] + (ε1 * e1[c] + ε2 * e2[c]), Val(3)))
end
"""The raw chief direction's gradient `∂Φ/∂x` from its two seed partials `g₁, g₂` (`_dir_seeded`) at the direction `x`."""
@inline function _dir_gradient(g1, g2, x::StaticVector{3})
    e1, e2 = _seed_tangents(x)
    return g1 * e1 + g2 * e2
end
"""The seeded `t̂` (the focus surface's thin plate spline at 𝐩) and the unit chief direction of a sensor sample."""
@inline function _focus_and_dir(view::View, 𝐩, fseed, dirseed, ks::Integer)
    t̂s = _t̂_seeded(t̂(view.focus, 𝐩), fseed, ks)
    return t̂s, normalize(_dir_seeded(chief_ray(view.geometric, 𝐩), dirseed, ks, t̂s))
end
@inline function _t̂_seeded(x::ForwardDiff.Dual{Tag, V, P}, fbase::Integer, ks::Integer) where {Tag, V, P}
    slot = fbase + ks
    (1 <= slot <= P) || return x
    return x + ForwardDiff.Dual{Tag}(zero(V), ForwardDiff.Partials(ntuple(i -> i == slot ? one(V) : zero(V), Val(P))))
end

"""
    HeightRays(h)

The height mode of [`Φ_pixel`](@ref) (the device Jacobian's nodal-height columns): `h` are the board warping's nodal heights as
duals seeded with the unit partials (one slot per height), and only the relief of the pixel's rays sees them. The heights reach the
flux only through each ray's parameter `s` on the warped board (𝐱 = 𝐨 + s 𝐝: the board frame, the footprint and the ray do not
depend on h), so a ray's sensitivities are the outer product `∂L/∂s ⊗ ∂s/∂h` — `∂s/∂h` from the dual relief (Core's implicit-function-theorem root),
`∂L/∂s` from ONE dual slot carried through the radiance estimator — instead of `HEIGHT_NC` slots carried through it. `Φ_pixel`
then returns the `3 × HEIGHT_NC` matrix `∂Φ/∂h` (the chain rule of the evaluated objective; equal to the seeded-height partials up
to rounding). The other arguments of the call are plain (no seeds).
"""
struct HeightRays{V <: StaticVector{HEIGHT_NC}}
    h::V
end
struct HeightRayTag end
@inline _pixel_rays(::Nothing, board::TexturedTarget) = BoardRays(board)
@inline _pixel_rays(hr::HeightRays, board::TexturedTarget) = BoardRays(board, hr.h)   # Core: the height-field Newton's hit, its height partials by one IFT step
@inline _ray_zero(::Nothing, ::Type{T}) where {T} = zero(SVector{3, T})
@inline _ray_zero(::HeightRays, ::Type{T}) where {T} = zero(SMatrix{3, HEIGHT_NC, T, 3 * HEIGHT_NC})
"""The ray's board point as the radiance estimator sees it: as intersected, or (height mode) on the ray at a one-slot dual `s`."""
@inline _ray_point(::Nothing, s, 𝐱, o, d) = 𝐱
@inline _ray_point(::HeightRays, s, 𝐱, o, d) =
    o + ForwardDiff.Dual{HeightRayTag}(ForwardDiff.value(s), ForwardDiff.Partials((one(ForwardDiff.value(s)),))) * d
"""A ray's weighted radiance `wL` added to the disc sum: as it is, or (height mode) `∂(wL)/∂s ⊗ ∂s/∂h`."""
@inline _ray_add(::Nothing, inner, wL, s) = inner + wL
@inline _ray_add(::HeightRays, inner::SMatrix{3, NH}, wL, s) where {NH} =
    inner + SMatrix{3, NH}(ntuple(q -> ForwardDiff.partials(wL[(q - 1) % 3 + 1], 1) * ForwardDiff.partials(s, (q - 1) ÷ 3 + 1), Val(3 * NH)))

"""
    RadianceRecord(out, base) / RadianceRays(J, base)

The radiance-Jacobian modes of [`Φ_pixel`](@ref) (the device Jacobian): `L_c` depends on the parameters only through its local
inputs ℓ — the albedo ρ, the roughness α, the refractive indices η, the outgoing direction ω_l and the board point 𝐱 — and channel c
reads ρ and η only through ρ_c and η_c, so `RAD_NL` = 8 local slots carry its whole derivative: one shared by ρ_c, α, one shared by
η_c, ω_l's two tangent components (ω_l is a unit vector: its partials are tangent, `_seed_tangents`) and 𝐱's three.
`RadianceRecord`: at the plain parameters, each ray's `[L | ∂L/∂ℓ]` and its hit (`RadianceJac`, `_ray_hit`) are written to `out[base + ray]` (ray
`(ks − 1) NA + la` of the pixel) and its value used. `RadianceRays`: the ray's ℓ, dual in the chunk's (or the height mode's) slots,
replaces the estimator by its linearisation at the record, `L⁰ + J (ℓ − value(ℓ))` — the chain rule (∂L/∂θ = ∂L/∂ℓ · ∂ℓ/∂θ), so a
chunk's partials skip the estimator. A ray the record did not reach reads as zero radiance with zero partials (the buffer is zeroed),
as `L_c` returns for a ray the board does not face.
"""
struct RadianceRecord{A}
    out::A
    base::Int
end
struct RadianceRays{A}
    J::A
    base::Int
end
struct RadianceTag end
const RAD_NL = 8
"""A ray's `[L | ∂L/∂ℓ | 𝐱 | 𝐧]` in the kernels' float type `T` (`_kernel_float`: Float32 on a device, Float64 on the CPU backend): the
radiance and its `RAD_NL` local partials, then the ray's hit: its board point 𝐱 and the board surface's normal 𝐧 there (`_ray_hit`;
any length, 𝐧 ⋅ 𝐝 = 1)."""
const RadianceJac{T} = SMatrix{3, RAD_NL + 3, T, 3 * (RAD_NL + 3)}
struct HitTag end
"""
    _ray_hit(hrays, lrays, ri, rays, o, d) -> (s, 𝐱, 𝐧)

The hit of the ray `(o, d)` (`d` unit) of a pixel (`Φ_pixel`'s ray `ri`) with the board `rays` (`BoardRays`: the plane, or the
warped page): the ray's parameter `s`, the board point 𝐱, and — where a mode records it — the surface's normal 𝐧 there.

- The flux: `intersect_board` (Core's `intersection`: the plane hit, or the height field's Newton), no normal.
- The nodal-height mode (`HeightRays`) beside a radiance record of the same parameters (`RadianceRays`, the heights kernel's): Core's
  `intersection_at` of the dual relief at the recorded hit — the implicit-function step alone, no Newton (the root is the record's);
  without a record, `intersect_board` as the flux. A ray the record did not reach (𝐧 = 0) is a miss.
- `RadianceRecord` (the plain parameters): the same hit with the ray's origin seeded (three partials), so Core's implicit-function
  step gives `∂𝐱/∂𝐨 = I − 𝐝 𝐧ᵀ / (𝐧 ⋅ 𝐝)` beside the hit, and with it the normal: `𝐧ᵀ = 𝐝ᵀ (I − ∂𝐱/∂𝐨)` (so 𝐧 ⋅ 𝐝 = 1). The
  record keeps 𝐱 and 𝐧 with the ray's radiance.
- `RadianceRays` (a Jacobian chunk: the ray dual in the chunk's slots, the surface constant): the hit of the dual ray with the
  surface's TANGENT PLANE at the recorded hit, `s = 𝐧 ⋅ (𝐱₀ − 𝐨) / (𝐧 ⋅ 𝐝)`, `𝐱 = 𝐨 + s 𝐝` — the value the recorded hit, the partials
  the implicit-function theorem's (a surface and its tangent plane have the same first-order hit). No Newton: the height field's
  root is found once per ray, by the record, not again by every chunk (a warped board's chunk kernel 303 → 60 µs per pixel on the
  CPU, 2026-11-20). A ray the record did not reach (zeros) has 𝐧 = 0: `s` is not finite and the ray is skipped, as before.
"""
@inline _ray_hit(hrays, lrays, ri, rays, o, d) = (intersect_board(rays, o, d)..., nothing)
@inline function _ray_hit(::Nothing, ::RadianceRecord, ri, rays, o::StaticVector{3, T}, d::StaticVector{3, T}) where {T}
    D = ForwardDiff.Dual{HitTag, T, 3}
    od = SVector{3, D}(ntuple(c -> D(o[c], ForwardDiff.Partials(ntuple(i -> i == c ? one(T) : zero(T), Val(3)))), Val(3)))
    sd, 𝐱d = intersect_board(rays, od, SVector{3, D}(d))
    # 𝐧ⱼ = Σ_c 𝐝_c (δ_cj − ∂𝐱_c/∂𝐨_j)
    𝐧 = SVector{3, T}(ntuple(j -> (d[j] - (d[1] * ForwardDiff.partials(𝐱d[1], j) + d[2] * ForwardDiff.partials(𝐱d[2], j) + d[3] * ForwardDiff.partials(𝐱d[3], j))), Val(3)))
    return ForwardDiff.value(sd), SVector{3, T}(ntuple(c -> ForwardDiff.value(𝐱d[c]), Val(3))), 𝐧
end
"""A record ray's hit `(𝐱, 𝐧, reached)` (`RadianceJac`'s last two columns): `reached` false where the record did not reach the ray
(its zeros, 𝐧 = 0)."""
@inline function _recorded_hit(lr::RadianceRays, ri)
    # the six numbers alone, from the record's flat storage (a `RadianceJac` is column-major: 𝐱 at offset 3(RAD_NL + 1), 𝐧 after it)
    flat = reinterpret(eltype(eltype(lr.J)), lr.J)
    q = (lr.base + ri - 1) * (3 * (RAD_NL + 3)) + 3 * (RAD_NL + 1)
    @inbounds 𝐱, 𝐧 = SVector(flat[q + 1], flat[q + 2], flat[q + 3]), SVector(flat[q + 4], flat[q + 5], flat[q + 6])
    return 𝐱, 𝐧, !iszero(𝐧)
end
@inline function _ray_hit(::Nothing, lr::RadianceRays, ri, rays, o::StaticVector{3}, d::StaticVector{3})
    𝐱₀, 𝐧, _ = _recorded_hit(lr, ri)
    s = dot(𝐧, 𝐱₀ - o) / dot(𝐧, d)
    return s, o + s * d, nothing
end
@inline function _ray_hit(::HeightRays, lr::RadianceRays, ri, rays::BoardRays, o::StaticVector{3}, d::StaticVector{3})
    rays.warped || return (intersect_board(rays, o, d)..., nothing)
    𝐱₀, _, reached = _recorded_hit(lr, ri)
    nan = oftype(𝐱₀[1], NaN)
    𝐱₀ = ifelse(reached, 𝐱₀, SVector(nan, nan, nan))                  # a ray the record did not reach: Core's miss
    return (VgtkCore.intersection_at(rays.relief, o, d, 𝐱₀)..., nothing)
end
"""The radiance of a ray (`Φ_pixel`'s ray `ri`): `L_c`, or one of its radiance-Jacobian modes (`RadianceRecord`, `RadianceRays`)."""
@inline _ray_radiance(::Nothing, ri, bd, m, lt, sm, key, 𝐛, 𝐱, ω, 𝐧) = L_c(bd, m, lt, sm, key, 𝐛, 𝐱, ω)
@inline function _ray_radiance(lr::RadianceRecord, ri, bd, m, lt, sm, key, 𝐛, 𝐱, ω, 𝐧)
    ω_l, T, front, (ρ, α, η) = _ray_material(bd, m, lt, 𝐛, 𝐱, ω)
    front || return zero(SVector{3, T})
    D = ForwardDiff.Dual{RadianceTag, T, RAD_NL}
    ε(j) = ForwardDiff.Dual{RadianceTag}(zero(T), ForwardDiff.Partials(ntuple(i -> i == j ? one(T) : zero(T), Val(RAD_NL))))
    e1, e2 = _seed_tangents(ω_l)
    L = _radiance_mc(bd, m, lt, sm, key, ρ .+ ε(1), α + ε(2), η .+ ε(3), ω_l + ε(4) * e1 + ε(5) * e2,
                     SVector(𝐱[1] + ε(6), 𝐱[2] + ε(7), 𝐱[3] + ε(8)), D)
    @inbounds lr.out[lr.base + ri] = eltype(lr.out)(ntuple(q -> (c = (q - 1) % 3 + 1; j = (q - 1) ÷ 3;
                                                              j == 0 ? ForwardDiff.value(L[c]) : j <= RAD_NL ? ForwardDiff.partials(L[c], j) :
                                                              j == RAD_NL + 1 ? 𝐱[c] : 𝐧[c]), Val(3 * (RAD_NL + 3))))   # [L | ∂L/∂ℓ | 𝐱 | 𝐧]
    return SVector(ntuple(c -> ForwardDiff.value(L[c]), Val(3)))
end
@inline _δ(x::ForwardDiff.Dual) = x - ForwardDiff.value(x)
@inline _δ(x::Real) = zero(x)
@inline function _ray_radiance(lr::RadianceRays, ri, bd, m, lt, sm, key, 𝐛, 𝐱, ω, 𝐧)
    ω_l, Tz, front, (ρ, α, η) = _ray_material(bd, m, lt, 𝐛, 𝐱, ω)
    front || return zero(SVector{3, Tz})
    e1, e2 = _seed_tangents(ForwardDiff.value.(ω_l))
    q4 = _δ(dot(e1, ω_l)); q5 = _δ(dot(e2, ω_l)); qα = _δ(α)
    q6 = _δ(𝐱[1]); q7 = _δ(𝐱[2]); q8 = _δ(𝐱[3])
    G = @inbounds lr.J[lr.base + ri]
    return SVector{3, Tz}(ntuple(c -> G[c, 1] + G[c, 2] * _δ(ρ[c]) + G[c, 3] * qα + G[c, 4] * _δ(η[c]) + G[c, 5] * q4 + G[c, 6] * q5 +
                                      G[c, 7] * q6 + G[c, 8] * q7 + G[c, 9] * q8, Val(3)))
end

"""
    Φ_pixel(u, v, k, camera, pose, mat, light, board, settings) -> SVector{3}

Radiant flux of pixel `(u, v)` (column `u`, row `v` of the `H × W` image) of photograph `k`: `camera` (a
`RadiometricCameraModel`) at `pose` (camera → world). `k` is the photograph's index: it keys the common random numbers of
the radiance (`SampleKey`), so the flux of a state's view `k` is `Φ_pixel(u, v, k, camera, poses[k], …)`.
"""
Φ_pixel(u::Integer, v::Integer, k::Integer, camera::RadiometricCameraModel, pose::EuclideanMap, mat::Material, light::EnvironmentLight,
        board::TexturedTarget, st::RenderSettings) =
    _Φ_pixel(u, v, k, _image_height(camera), View(pose, camera), camera.vignetting, mat, light, board, st)

"""The number type a vignetting brings into the flux: its nodes' element type; the absent vignetting (VisualRadiometryToolkitCore's
`NoVignetting`: V ≡ 1, `vig(𝐩) === true`) brings none — `Bool`, the neutral element of the promotion — so a Float32 kernel stays
Float32, and its factor `w * true` is `w`: the kernel holds no multiplication by V and V has no Jacobian columns (no Θ_v)."""
@inline _vignetting_number_type(vig::AbstractVignetting) = eltype(control_points(vig))
@inline _vignetting_number_type(::NoVignetting) = Bool

"""
    _Φ_pixel(u, v, k, H, view, vig, mat, light, board, settings[, fseed, dirseed, hrays, lrays]) -> SVector{3}

The kernels' flux of pixel `(u, v)` of view `k` ([`Φ_pixel`](@ref) of the camera at its pose, the vignetting apart; `H` the image's
height, the camera's: the stride of the pixel ids, which a kernel holding a bare model of a tile cannot read from it). `@inline`: in
the device kernels a Julia-level inline lets the compiler optimize the dual arithmetic across the call (the backend's
`always_inline` only inlines at the LLVM level) — the GPU Jacobian is 12% faster. `hrays`: a `HeightRays` for the `3 × HEIGHT_NC`
nodal-height partials instead of the flux. `lrays`: a radiance-Jacobian mode (`RadianceRecord`, `RadianceRays`) for the rays'
radiance.
"""
@inline function _Φ_pixel(u::Integer, v::Integer, k::Integer, H::Integer, view::View{T}, vig::AbstractVignetting, mat::Material,
                 light::EnvironmentLight, board::TexturedTarget, st::RenderSettings, fseed = nothing, dirseed = nothing,
                 hrays = nothing, lrays = nothing) where {T}
    bd = board
    Tz = promote_type(T, eltype(mat.refractive_index_ink), eltype(coefficients(light)), _vignetting_number_type(vig))
    acc = _ray_zero(hrays, Tz)
    pid = pixel_id(u, v, H)
    rays = _pixel_rays(hrays, board)                                    # the board warping h's relief, formed once for the pixel's rays
    @inbounds for ks in 1:length(st.sensor)
        𝐩 = 𝐩_sample(st.sensor, ks, u, v)
        V𝐩 = vig(𝐩)
        t̂s, dch = _focus_and_dir(view, 𝐩, fseed, dirseed, ks)   # t̂ and the UNIT chief direction, hoisted once per sensor sample (seeded, then normalised)
        inner = _ray_zero(hrays, Tz)
        for la in 1:n_disc(st.aperture)
            𝐚 = 𝐚_sample(st.aperture, la, view.r)
            o, d = world_ray_dir(view, 𝐚, dch, t̂s)
            ri = (ks - 1) * n_disc(st.aperture) + la
            s, 𝐱, 𝐧 = _ray_hit(hrays, lrays, ri, rays, o, d)                # the board warping h (the plane when zero); a chunk: the recorded hit's tangent plane
            (isfinite(ForwardDiff.value(s)) && ForwardDiff.value(s) > 0) || continue
            𝐱ᵣ = _ray_point(hrays, s, 𝐱, o, d)
            𝐛 = chart(bd, 𝐱ᵣ)
            L = _ray_radiance(lrays, ri, bd, mat, light, st.sampler, SampleKey(k, pid, ks, la), 𝐛, 𝐱ᵣ, d, 𝐧)
            inner = _ray_add(hrays, inner, 𝐚_weight(st.aperture, la, view.r) * L, s)
        end
        acc += (st.sensor.weights[ks] * V𝐩) * inner
    end
    return acc / (π * view.r^2)
end

"""
    compute_mask(camera, pose, board, settings; δ = 0.02, external = nothing, backend) -> BitMatrix

Pixel mask of `camera` (a `RadiometricCameraModel`) at `pose` (camera → world): `true` where the chief ray and every
(footprint corner × aperture border sample) ray hits the board inside `[δ, 1−δ]²` in front of the camera; intersected with
`external`. `board_valid` over every pixel of `external` (all of them without one): one kernel on `backend`, required. A caller
that needs only some of the pixels (the start's draw, `draw_samples`) tests those alone through `BoardValidity`.
"""
compute_mask(camera::RadiometricCameraModel, pose::EuclideanMap, board::TexturedTarget, st::RenderSettings; kwargs...) =
    _compute_mask(View(pose, camera), board, st; kwargs...)
"""The kernels' form of [`compute_mask`](@ref): the camera at its pose as a `View`."""
function _compute_mask(view::View, board::TexturedTarget, st::RenderSettings; δ::Real = 0.02, external = nothing, backend)
    W, H = _image_size(view)                                                   # the image is the camera's
    ids = external === nothing ? collect(1:(H * W)) : findall(vec(external))   # column-major: (u − 1) H + v
    mask = falses(H, W)
    mask[ids[board_valid(view, board, st, ids; δ, backend)]] .= true
    return mask
end

"""
    board_valid(view, board, settings, ids; δ, backend) -> BitVector

The mask test (`compute_mask`'s) of the pixels `ids` (column-major linear indices `(u − 1) H + v`) alone, in `ids`' order: one
kernel over them (`_MaskAt` on Core's `pointwise!`) on `backend` (the caller's device: a state's or a run's `rc_backend`;
KernelAbstractions' CPU() threads it over the pixels). THE mask test: `compute_mask` over a region, `BoardValidity` over a draw's
candidates.
"""
function board_valid(view::View, board::TexturedTarget, st::RenderSettings, ids::AbstractVector{<:Integer}; δ::Real, backend)
    isempty(ids) && return falses(0)
    out = KernelAbstractions.zeros(backend, Bool, length(ids))
    VgtkCore.pointwise!(_MaskAt(view, board, st.aperture, Int32(_image_height(view)), Float64(δ)), out, adapt(backend, UInt32.(ids));
                        exec = VgtkCore.OnBackend(backend))
    KernelAbstractions.synchronize(backend)
    return BitVector(Array(out))
end

"""
    BoardValidity(view, board, settings; δ, backend)

The mask test of one view as a function of pixel ids: `t(ids)` is `board_valid(view, board, settings, ids; δ, backend)`.
The start's draw (`draw_samples`) calls it on the candidates it needs, in its keys' order, instead of the whole region.
"""
struct BoardValidity{V <: View, S <: TexturedTarget, R <: RenderSettings, B}
    view::V
    board::S
    settings::R
    δ::Float64
    backend::B
end
BoardValidity(view::View, board::TexturedTarget, st::RenderSettings; δ::Real, backend) = BoardValidity(view, board, st, Float64(δ), backend)
(t::BoardValidity)(ids::AbstractVector{<:Integer}) = board_valid(t.view, t.board, t.settings, ids; t.δ, t.backend)

"""The mask test of a pixel id (`compute_mask`): its chief ray and every footprint corner × aperture border ray on the board inside
`[δ, 1−δ]²`, in front of the camera."""
struct _MaskAt{V, B, A}
    view::V
    board::B
    aperture::A
    H::Int32
    δ::Float64
end
Adapt.@adapt_structure _MaskAt
const _MASK_CORNERS = (SVector(-0.5, -0.5), SVector(0.5, -0.5), SVector(-0.5, 0.5), SVector(0.5, 0.5), SVector(0.0, 0.0))
@inline function (f::_MaskAt)(id::UInt32)
    u, v = _decode_px(id, f.H)
    view, bd, ap = f.view, f.board, f.aperture
    for c in _MASK_CORNERS
        𝐩 = 𝐩_of_pixel(u + c[1], v + c[2])
        in_domain(view.geometric, 𝐩) || return false
        o, d = world_ray(view, 𝐩, 𝔭)
        s, 𝐱 = intersect_board(bd, o, d)
        (isfinite(s) && s > 0 && inside(chart(bd, 𝐱), f.δ)) || return false
        for j in 1:n_border(ap)
            𝐚, _, _, _ = 𝐚_border(ap, j, view.r)
            o, d = world_ray(view, 𝐩, 𝐚)
            s, 𝐱 = intersect_board(bd, o, d)
            (isfinite(s) && s > 0 && inside(chart(bd, 𝐱), f.δ)) || return false
        end
    end
    return true
end

"""Ordered list of masked pixels `(u, v)` (column-major over the image)."""
masked_pixels(mask::AbstractMatrix{Bool}) = [(I[2], I[1]) for I in findall(mask)]

"""
    render_view(k, camera, pose, mat, light, board, settings, mask; backend = CPU()) -> Array{Float64, 3}

`H × W × 3` image of photograph `k`, `camera` (a `RadiometricCameraModel`) at `pose` (camera → world), zeros outside the
mask: the flux kernel (`_FluxAt`, device/jacobian_kernels.jl) on `backend` (KernelAbstractions' `CPU()` by default; a device:
`rc_backend("gpu")`), in the backend's kernel float as a state's kernels (`_kernel_float`: Float64 on the CPU backend, Float32 on a
device, where the image agrees with the CPU's to Float32 rounding). `k` is the photograph's index: it keys the common random numbers
of the radiance, so photograph `k` of a state renders with its index.
"""
render_view(k::Integer, camera::RadiometricCameraModel, pose::EuclideanMap, mat::Material, light::EnvironmentLight, board::TexturedTarget,
            st::RenderSettings, mask::AbstractMatrix{Bool}; backend = KernelAbstractions.CPU()) =
    _render_view(k, View(pose, camera), camera.vignetting, mat, light, board, st, mask; backend)
"""The kernels' form of [`render_view`](@ref): the camera at its pose as a `View`, the vignetting apart."""
function _render_view(k::Integer, view::View, vig::AbstractVignetting, mat::Material, light::EnvironmentLight,
                      board::TexturedTarget, st::RenderSettings, mask::AbstractMatrix{Bool}; backend = KernelAbstractions.CPU())
    W, H = _image_size(view)                                                   # the image is the camera's; the mask is of it
    size(mask) == (H, W) || throw(ArgumentError("render_view: the mask is $(size(mask, 1)) × $(size(mask, 2)) (rows × columns), the camera's image $H × $W"))
    px = masked_pixels(mask)
    img = zeros(Float64, H, W, 3)
    isempty(px) && return img
    F = _kernel_float(backend)
    ids = adapt(backend, [UInt32((u - 1) * H + v) for (u, v) in px])
    as(x) = F === _float_of(st) ? x : VgtkCore.convert_eltype(F, x)          # (arguments already in the kernels' float: as they are, no copy)
    flux = _FluxAt(Int32(k), Int32(H), adapt(backend, as(view)), as(vig), as(mat), as(light), adapt(backend, as(board)), as(st))
    Φ = _device_rows(flux, backend, ids)
    for (i, (u, v)) in enumerate(px), c in 1:3
        img[v, u, c] = Φ[3 * (i - 1) + c]
    end
    return img
end

"""
    masked_vector(img, px; channel = 0) -> Vector{Float64}

The observed vector `ϕ` of the masked pixels `px`, in the residual's layout (`𝔯 = Φ − ϕ`): for `channel = 0` the three channels of
`img` (H × W × 3) per pixel, channel fastest (index `3(i−1)+c`); for `channel = c` that channel alone, ONE entry per pixel
(`fit_channels`). A single plane `img` (H × W: one measurement per pixel, the green photosites) is one entry per pixel and
needs a single-channel state (`channel` 1–3).
"""
function masked_vector(plane::AbstractMatrix{<:Real}, px::AbstractVector; channel::Integer = 0)
    channel > 0 || throw(ArgumentError("masked_vector: a single plane is one measurement per pixel: a single-channel state (channel 1–3)"))
    return Float64[plane[v, u] for (u, v) in px]
end
function masked_vector(img::AbstractArray{<:Real, 3}, px::AbstractVector; channel::Integer = 0)
    channel > 0 && return Float64[img[v, u, channel] for (u, v) in px]
    out = Vector{Float64}(undef, 3 * length(px))
    for (i, (u, v)) in enumerate(px), c in 1:3
        out[3 * (i - 1) + c] = img[v, u, c]
    end
    return out
end

"""
    tie_channels(light::EnvironmentLight, c) -> EnvironmentLight
    tie_channels(mat::Material, c) -> Material

A single-channel state's parameters (`CalibrationState`'s `channel = c`): every channel's lighting coefficients, and every
channel's refractive indices `η_ink`, `η_¬ink`, are channel `c`'s. No residual reads the other channels, so they are no parameters
of the fit (`LightingVar`, `MaterialVar`): they follow channel `c`, so a render or a record of the state shows the measured
channel's values in all three.
"""
function tie_channels(light::EnvironmentLight, c::Integer)
    a = Matrix(coefficients(light))
    return with_coefficients(light, a[[light_tie(c, ch) for ch in 1:3], :])
end
function tie_channels(m::Material, c::Integer)
    θ = Θ_m_nonlinear(m)
    return with_nonlinear(m, SVector(ntuple(i -> θ[material_tie(c)[i]], Val(8))))
end

# ---- the single-channel layout: THE statement of which coordinates a channel-`c` fit keeps and what the others follow ---------------
"""
    material_tie(c) -> NTuple{8, Int}

The single-channel layout of the material's 8 nonlinear coordinates (`Θ_m_nonlinear`'s order: `α_ink, α_¬ink, η_ink[1:3],
η_¬ink[1:3]`): each coordinate's source in a channel-`c` fit — the α's their own, every channel's η_ink channel `c`'s (`2 + c`),
every η_¬ink channel `c`'s (`5 + c`); `c = 0` (the three channels) each its own. `material_sources(c)` are the sources, the fit's
coordinates in order. `tie_channels`, `MaterialVar`'s update, the Jacobian's columns (`_channel_jacobian`) all read it.
"""
material_tie(c::Integer) = c == 0 ? ntuple(identity, 8) : (1, 2, 2 + c, 2 + c, 2 + c, 5 + c, 5 + c, 5 + c)
"The material coordinates a channel-`c` fit estimates (the distinct sources of `material_tie(c)`, in order)."
material_sources(c::Integer) = Tuple(unique(material_tie(c)))
"""The source channel of lighting channel `ch` in a channel-`c` fit (`c` itself; `ch` for the three channels): every channel's SH
coefficients follow the measured one's. `tie_channels`, `LightingVar`'s update, the Jacobian's columns and the joint radiometric solve
read it."""
light_tie(c::Integer, ch::Integer) = c == 0 ? ch : c
"""
    material_tie_basis(c) -> Matrix{Float64}

The material's channel ties as the block's basis (8 × `length(material_sources(c))`): column `j` is the full block's coordinates that
follow the fit's coordinate `j` (`material_tie`: ones where it is their source). A channel-`c` block's step is the basis times its
coordinates (`MaterialVar`'s update); its Jacobian columns are the full ones times the basis (`_channel_jacobian`), the other
channels' columns being zero on channel `c`'s rows.
"""
material_tie_basis(c::Integer) = (t = material_tie(c); s = material_sources(c); [Float64(t[i] == s[j]) for i in 1:8, j in eachindex(s)])
"""
    light_tie_basis(c, nsh) -> Matrix{Float64}

The lighting's channel ties as the block's basis (`3 nsh` × `nsh` for a channel-`c` fit, the identity for `c = 0`): every channel's
coefficient `j` follows channel `c`'s (`light_tie`). `LightingVar`'s update and `_channel_jacobian` read it, as `material_tie_basis`.
"""
light_tie_basis(c::Integer, nsh::Integer) = c == 0 ? Matrix{Float64}(I, 3nsh, 3nsh) :
    [Float64((i - 1) ÷ 3 + 1 == j) for i in 1:3nsh, j in 1:nsh]      # coefficient (ch, j) at 3(j − 1) + ch: every channel follows channel c's j
