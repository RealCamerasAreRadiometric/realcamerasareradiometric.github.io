# ---------------------------------------------------------------------------
# The target's radiometry: what a refinement learns about the PRINT, as one record (the camera's share is the
# `RadiometricCameraModel`). It is the value the IO phase persists for a calibration target (keyed by the target's content id);
# the calibration's working containers are the state's `mat` (Θ_m) and the scene's `TargetState.edge` (δ, σ_print in mm), which
# `TargetRadiometry(mat, surface)` reads and `material` / `edge_parameters` give back.
# ---------------------------------------------------------------------------

"""
    TargetRadiometry(material, ink_spread, print_blur)
    TargetRadiometry(material, surface::TargetState)

The radiometry of a printed calibration target: its SVBSDF (`material`, a `Material`: Θ_m = the roughnesses and refractive
indices of ink and paper and the diffuse weight μ) and, for a design given as a distance map, its printed edge — the ink spread δ
(`ink_spread`, a length: a FlexUnits quantity in metres) and the print's own blur (`print_blur`, Core's `GaussianBlur` on the print,
round, its moment in m²); a plain ink spread or a plain (px²) print blur is an `ArgumentError`.
From a state's working form, `material` and the surface's `edge = (δ, σ_print)` in mm, read decimal-exactly
(VisualGeometryToolkitIO's `read_length`): `edge_parameters(r)` gives the mm values back bit for bit for values of up to 15
significant digits.
"""
struct TargetRadiometry{T, Q, B <: VgtkCore.GaussianBlur}
    material::Material{T}
    ink_spread::Q
    print_blur::B
    function TargetRadiometry(material::Material{T}, ink_spread::Q, print_blur::B) where {T, Q, B <: VgtkCore.GaussianBlur}
        (ink_spread isa VgtkCore.Quantity && VgtkCore.dimension(ink_spread) == VgtkCore.dimension(VgtkCore.m)) ||
            throw(ArgumentError("TargetRadiometry: the ink spread is a length, a FlexUnits quantity (`δ * m`, or VisualGeometryToolkitIO's " *
                                "`read_length`), got $ink_spread"))
        (VgtkCore.has_units(print_blur) && VgtkCore.dimension(cov(print_blur)[1, 1]) == VgtkCore.dimension(VgtkCore.m^2)) ||
            throw(ArgumentError("TargetRadiometry: the print's blur holds lengths² (m² quantities), not px²: a plain blur in metres is " *
                                "`with_unit(blur, m)`"))
        return new{T, Q, B}(material, ink_spread, print_blur)
    end
end
TargetRadiometry(m::Material, surface::TargetState) =
    TargetRadiometry(m, VgtkIO.read_length(Float64(surface.edge[1]), VgtkCore.mm), VgtkCore.GaussianBlur(VgtkIO.read_length(Float64(surface.edge[2]), VgtkCore.mm)))

"""The target's SVBSDF Θ_m (`Material`)."""
material(r::TargetRadiometry) = r.material
"""
    RadiometricTarget(target, warping, aspect, radiometry)

A calibration target as a refinement knows it, parallel to the geometric camera ↔ the `RadiometricCameraModel`: the
geometric target (`target`, Core's `AbstractTarget` as the session's target record holds it, so its content identity is the
session's; `nothing` for a state converted from a RadiometricCamera 0.5 `.jls` file, which did not record it), the board's warping as
the refinement learned it (`warping`, the `HEIGHT_NC` nodal heights of `TargetState.h`, plain metres along the board normal —
a calibration of the target that its content identity does not cover, so not the target's own `HeightField`), the aspect (height /
width) of the page the warping was learned on (`aspect`, `warping_aspect` of its board: the same nodal heights are another surface on
a page of another aspect), the in-plane print error as the refinement learned it (`inplane`, the `(a, u)` of Core's `Aspect ∘ Skew` in
the warping's placed chart — `inplane_calibration`; zero = the design as printed) and the print's radiometry ([`TargetRadiometry`](@ref)).
A flat warping (every nodal height zero) is the plane on any page: its aspect may be `nothing`; a warped one, or an in-plane error
(whose chart depends on the aspect), without an aspect is an `ArgumentError`. [`board_surface`](@ref) is the board's
`TargetState` of it on a board.
"""
struct RadiometricTarget{G, T, R <: TargetRadiometry}
    target::G
    warping::SVector{HEIGHT_NC, T}
    aspect::Union{Nothing, Float64}
    inplane::SVector{INPLANE_NC, Float64}
    radiometry::R
    function RadiometricTarget(target::G, warping::SVector{HEIGHT_NC, T}, aspect::Union{Nothing, Real}, radiometry::R;
                               inplane = zero(SVector{INPLANE_NC, Float64})) where {G, T, R <: TargetRadiometry}
        ip = SVector{INPLANE_NC, Float64}(inplane)
        all(isfinite, ip) || throw(ArgumentError("RadiometricTarget: the in-plane print error (a, u) must be finite, got $ip"))
        aspect === nothing && !iszero(warping) &&
            throw(ArgumentError("RadiometricTarget: a warped target needs the aspect (height / width) of the page its warping was learned on"))
        aspect === nothing && !iszero(ip) &&
            throw(ArgumentError("RadiometricTarget: an in-plane print error needs the aspect (height / width) of the page it was learned on"))
        aspect === nothing || (isfinite(aspect) && aspect > 0) ||
            throw(ArgumentError("RadiometricTarget: the page's aspect (height / width) must be positive and finite, got $aspect"))
        return new{G, T, R}(target, warping, aspect === nothing ? nothing : Float64(aspect), ip, radiometry)
    end
end
"""
    board_surface(b::TexturedTarget, t::RadiometricTarget) -> TargetState
    board_surface(aspect, t::RadiometricTarget) -> TargetState

The `TargetState` of a radiometric target on the board `b` (or a page of `aspect` = height / width): its warping, its in-plane
print error and its print's edge parameters. The target's warping and in-plane error must have been learned on a page of the board's
aspect (`t.aspect == warping_aspect(b)` to 1e-6), or be flat and as designed: an `ArgumentError` otherwise, since the nodal heights (and the
placed chart of the in-plane error) on a page of another aspect are another surface.
"""
function board_surface(aspect::Real, t::RadiometricTarget)
    (iszero(t.warping) && iszero(t.inplane)) || isapprox(t.aspect, aspect; rtol = 1e-6) ||
        throw(ArgumentError("board_surface: the target's warping was learned on a page of aspect (height / width) $(t.aspect), the board's page has $aspect: its nodal heights are another surface there"))
    return TargetState(; h = t.warping, edge = edge_parameters(t.radiometry), inplane = t.inplane)
end
board_surface(b::TexturedTarget, t::RadiometricTarget) = board_surface(warping_aspect(b), t)
"""The print's edge parameters `(δ, σ_print)` in mm, the `TargetState.edge` of a state ([`TargetRadiometry`](@ref))."""
edge_parameters(r::TargetRadiometry) =
    SVector(VgtkIO.write_length(r.ink_spread, VgtkCore.mm), VgtkIO.write_length(std(r.print_blur)[1], VgtkCore.mm))   # σ_print: Core's std of the round blur
