# ---------------------------------------------------------------------------
# EXPERIMENT (focus-ab branch only): alternative charts of the focus surface t̂.
#
# The sensor-grid TPS (`Focus` kind 0, focus.jl) puts its 5 × 5 nodes on the sensor (pixels); on a fisheye most of them see no
# light, and the TPS being global they still shape the observed surface (a near-null space that the LM damping decides). The two
# charts below put the nodes on the DIRECTIONS the lens images instead, inside the cone θ ≤ θ_max about the optical axis, and evaluate
# the surface at each sensor sample's chief-ray direction d̂(𝐩) = chief_dir(lens, 𝐩) (from the CURRENT lens: the kernel moves with it):
#
#   kind 1  DIRECTION CHART  a planar thin-plate spline (VisualGeometryToolkitCore's ThinPlateSplineSystem) in the azimuthal-
#           equidistant chart ξ(d̂) = θ (cos φ, sin φ) of the direction (θ the angle to the optical axis +z, φ its azimuth);
#   kind 2  SPHERE           Wahba's spherical spline (Wahba 1981, "Spline interpolation and smoothing on the sphere", SIAM J. Sci.
#           Stat. Comput. 2(1):5–16, the m = 2 reproducing kernel) on S², constant-only null space:
#
#             t̂(d̂) = a₁ + Σᵢ wᵢ R(d̂ · d̂ᵢ),   R(z) = Σ_{l≥1} (2l+1)/(l(l+1))² P_l(z) = 1 − π²/6 + Li₂((1 + z)/2)
#
#           (the closed form follows from Δ_S² P_l = −l(l+1) P_l: R″ integrates to −ln((1−z)/2)/(1+z), whose primitive is Li₂((1+z)/2);
#           the constant from R(1) = Σ (1/l² − 1/(l+1)²) = 1). The nodes are stored as their azimuthal-equidistant chart points
#           (same storage as kind 1) and mapped to S² on evaluation.
#
# The kernel R is VisualGeometryToolkitCore's `spherical_spline_kernel` (from the chord, Li₂ = Core's `dilog`), the system Core's
# `SphericalSplineSystem` of the nodes' directions, the planar system Core's `ThinPlateSplineSystem`; the azimuthal-equidistant chart
# is Core's equidistant lens (`project` / `backproject` of `Lens(Equidistant())`). `FocusSpline` (focus.jl) carries either system with
# the coefficient map in `Focus`'s (N+3) × N layout (the sphere's two affine rows zero), so `Focus`, `focus_block` and the chain rule of
# the LM are one code for every kind.
# ---------------------------------------------------------------------------

const FOCUS_SENSOR = 0
const FOCUS_DIRECTION_CHART = 1
const FOCUS_SPHERE = 2

const _AE = VgtkCore.Lens(VgtkCore.Equidistant())
"""Azimuthal-equidistant chart point ξ = θ (cos φ, sin φ) of a unit direction `d` (θ = angle to +z): Core's equidistant `project`
(dual-generic, regular on the axis)."""
@inline ae_chart(d::StaticVector{3}) = SVector{2}(VgtkCore.project(_AE, d))
"""Unit direction of the azimuthal-equidistant chart point ξ (the inverse of `ae_chart`): Core's equidistant `backproject`."""
@inline ae_dir(ξ::StaticVector{2}) = SVector{3}(VgtkCore.backproject(_AE, ξ))

"""The value of a direction-chart or sphere focus at the unit chief direction `d` (kinds 1, 2): Core's `spherical_spline_evaluate`
of the nodes' directions (a sphere's nodes are stored as chart points, mapped here) or `tps_evaluate` in the chart."""
@inline function t̂_dir(f, d::StaticVector{3})
    f.kind == FOCUS_SPHERE && return VgtkCore.spherical_spline_evaluate(map(ae_dir, f.nodes), f.w, f.a[1], d)
    return VgtkCore.tps_evaluate(f.nodes, f.w, f.a, ae_chart(d))
end

"""
    focus_weights(tps, lens, 𝐩) -> SVector{N}

`∂t̂(𝐩)/∂t̂ᵢ` for any focus kind: Core's `nodal_weights` of the spline system — at 𝐩 for the sensor TPS (kind 0), else at the chief
direction of 𝐩 through the CURRENT `lens` (host side, Float64; its chart point for kind 1) — recomputed whenever the lens moves.
"""
function focus_weights(tps::FocusSpline, lens, 𝐩::StaticVector{2})
    tps.kind == FOCUS_SENSOR && return VgtkCore.nodal_weights(tps.sys, 𝐩)
    d = SVector{3, Float64}(chief_dir(lens, 𝐩))
    return VgtkCore.nodal_weights(tps.sys, tps.kind == FOCUS_SPHERE ? d : ae_chart(d))
end

"""
    polar_cap_nodes(N, θ_max) -> Vector{SVector{2}}

`N` nodes on the cap θ ≤ θ_max as azimuthal-equidistant chart points: the axis, then `R` rings at θ_r = r θ_max / R (equal Δθ), the
ring counts ∝ sin θ_r (equal area on S²; largest remainder), each ring rotated by half its node spacing against the previous one.
`R` is the ring count whose radial step Δθ is closest to the mean node spacing √(2π(1 − cos θ_max)/N).
"""
function polar_cap_nodes(N::Integer, θ_max::Real)
    a = sqrt(2π * (1 - cos(θ_max)) / N)
    R = max(1, round(Int, θ_max / a))
    θr = [r * θ_max / R for r in 1:R]
    w = sin.(θr); share = (N - 1) .* w ./ sum(w)
    n = floor.(Int, share); rem = N - 1 - sum(n)
    for i in sortperm(share .- n; rev = true)[1:rem]; n[i] += 1; end
    nodes = [SVector(0.0, 0.0)]
    phase = 0.0
    for r in 1:R
        for j in 0:(n[r] - 1)
            φ = phase + 2π * j / n[r]
            push!(nodes, SVector(θr[r] * cos(φ), θr[r] * sin(φ)))
        end
        phase += π / max(n[r], 1)
    end
    return nodes
end

