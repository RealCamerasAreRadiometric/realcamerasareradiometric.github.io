# ---------------------------------------------------------------------------
# Vignetting V: ℐ → ℝ — VisualGeometryToolkitCore's model (`TensorProductVignetting`: a tensor B-spline surface over the sensor;
# `RadialVignetting`: a B-spline profile in the pixel radius about a fixed centre), linear in its nodes Θ_v, so V is
# solved by linear least squares (closed_form.jl): `VrtkCore.coefficient_jacobian(v, 𝐩)` is the row of the design matrix in the
# `vec(P)` (column-major) ordering. What is the calibration's own lives here: the run's rule for the nodes the data
# constrain (`vignetting_rule`, `RadialSupportRule`), their free directions (`vignetting_basis`), the gauge V(𝐜) = 1
# (`center_constraint`, `gauge_basis`, `gauge_center_value`) and the smoothness penalty's nodes (`smoothness_nodes`).
#
# The radial profile's first two nodes are tied, P_0 = P_1 (Core's `pole_regular_basis`): V is the surface of
# revolution of the generatrix (ρ, V(ρ)) about the centre (Piegl & Tiller §8.5), smooth at the pole only when V′(0) = 0 —
# otherwise the field is the cone V(𝐜) + V′(0) ‖𝐩 − 𝐜‖ near the centre, where few pixels constrain the slope.
# ---------------------------------------------------------------------------

# The radial profile's chart (`:linear` or `:projective`) is VisualRadiometryToolkitCore's `RadialVignetting(…; chart)` (hoisted from this package
# bit-identically, 2026-11-11; its `radial_chart`, `projective_knots`, `with_center`). Its centre is the camera's principal point,
# fixed (owner ruling 2026-11-11: no free centre; `vignetting_center`).
"""The extrapolation orders of a lattice's dependent nodes that Core's `BoundaryRule(m, region; order)` takes, checked when the
options are read (`grid.boundary_order`, `vignetting.boundary_order`): 2 the natural Δ²c = 0 (`c₀ = 2c₁ − c₂`, slope carried
across the border), 3 the curvature-preserving Δ³c = 0 (`c₀ = 3c₁ − 3c₂ + c₃`). The rule governs the tensor surface outside the
imaged region and, since 2026-11-07, the radial profile past its supported nodes (`RadialSupportRule`). The vignetting keeps
its dependent nodes where the grid lens holds only its present nodes (`grid_present`): V is a field over the whole sensor — its outputs (the flat field, the
panorama's gains) are read outside the run's imaged region — so the continuation there is part of its value, not a value no pixel
reads."""
const BOUNDARY_ORDERS = (2, 3)

"""
    vignetting_rule(vig, region; order = 2) -> Union{VisualGeometryToolkitCore.BoundaryRule, Nothing}

The nodes of a tensor surface under the run's imaged `region`: Core's `BoundaryRule(vig, region)` — the points whose
knot span meets the region are the parameters, the others DEPENDENT (the extrapolation of the interior's correction with a zero
`order`-th difference, Core's `BoundaryRule(vig, region; order)`: linear for 2, quadratic for 3; so V stays defined over the whole sensor and
nothing outside the region is frozen or unconstrained). `nothing` for a radial profile (its domain is the region's radius already;
its rule is the data's, `vignetting_rule(vig, region, px)`) or no region (every node a parameter).
"""
vignetting_rule(v::VrtkCore.TensorProductVignetting, region; order::Integer = 2) = VgtkCore.BoundaryRule(v, region; order)
vignetting_rule(::VrtkCore.TensorProductVignetting, ::Nothing; order::Integer = 2) = nothing

"""
    vignetting_basis(vig, rule) -> Matrix{Float64}

The basis `Z` (`n_control_points × n_free`) of the nodes' free directions, `vec(P) = vec(P₀) + Z y`: Core's
`boundary_rule_jacobian(rule)` — one column per interior node, a dependent node moving with its interior terms — the
identity for a tensor surface without a rule, and for a radial profile the pole tie `P_0 = P_1` (Core's `pole_regular_basis`): its
first two nodes move together (one column), the others one each (`n − 1` columns), so V′(0) = 0 and the field is C² at
the centre like at every knot.
"""
vignetting_basis(vig::AbstractVignetting, ::Nothing) = Matrix{Float64}(I, VrtkCore.n_control_points(vig), VrtkCore.n_control_points(vig))
vignetting_basis(::AbstractVignetting, rule::VgtkCore.BoundaryRule) = VgtkCore.boundary_rule_jacobian(rule)
vignetting_basis(v::RadialVignetting, ::Nothing) = VrtkCore.pole_regular_basis(v)

"""
    center_constraint(v) -> (C::Matrix, d::Vector)

The gauge `V(𝐜) = 1` at the field's centre (`VgtkCore.center`: a tensor surface's sensor centre, a radial profile's centre of
symmetry) as one linear row `C ⋅ vec(P) = d`.
"""
function center_constraint(v::AbstractVignetting)
    row = VrtkCore.coefficient_jacobian(v, VgtkCore.center(v))
    return reshape(collect(Float64, row), 1, :), [1.0]
end

# a radial profile's domain is the region's radius (Core's `farthest_radius`, `_vignetting_spec`): no node lies outside it
vignetting_rule(::RadialVignetting, region; order::Integer = 2) = nothing

"""
    tie_pole(vig) -> vig

A radial profile with the pole tie applied to its values, `P_1 := P_0` (`V(𝐜)` kept): a state written before the tie (2026-10-29)
resumes on the tied profile, as `vignetting_basis` ties the increments only. A tensor surface as it is.
"""
tie_pole(v::RadialVignetting) = (P = collect(vec(VrtkCore.control_points(v))); P[2] = P[1]; VrtkCore.with_control_points(v, P))
tie_pole(v::AbstractVignetting) = v

"""
    RadialSupportRule

The support of a radial profile's nodes under a run's masked pixels, and the rule of the rest: `supported[i]` when control
point `i` carries enough DATA WEIGHT, `weight[i] = Σ_pixels N_i(ρ)²` relative to the mean over the nodes, at least `support_min`
of it (2026-10-30; before, one masked pixel in the knot span sufficed — but the last nodes of a profile over the whole sensor
live in the frame's corner triangles and their basis functions rise cubically from the last knot, so even well-covered corners gave
them 0.7–5 % of the mean weight and they ran away). `P_0` and `P_1` are always supported: the gauge's parameter, tied `P_0 = P_1`
throughout. The others are DEPENDENT under Core's `BoundaryRule` of the profile's `N × 1` lattice (`rule`; the tensor surface's rule
outside the imaged region, `BOUNDARY_ORDERS`): each continues the supported nodes inward with a zero `order`-th difference — 2 the
natural Δ²c = 0, `P_k = 2 P_{k−1} − P_{k−2}`, the slope of the last fitted span carried past the data; 3 the curvature-preserving
Δ³c = 0 — so V continues past the data as the surface continues past the region (2026-11-07). From 2026-10-30 until then a dependent
point replicated the last supported one, a constant continuation, which the f/4 diagonals of the paper's vignetting figure showed as
plateaus inside the frame (the 14 mm's last two of 16 points, the Tamron's last three).
"""
struct RadialSupportRule
    supported::BitVector
    weight::Vector{Float64}       # the data weight per node relative to the mean (empty for a rule built without it)
    rule::VgtkCore.BoundaryRule   # Core's rule of the N × 1 lattice: the supported nodes interior, the others extrapolated with its `order`
end
RadialSupportRule(supported::BitVector, weight::Vector{Float64} = Float64[]; order::Integer = 2) =
    RadialSupportRule(supported, weight, VgtkCore.BoundaryRule(reshape(supported, :, 1); order))

"""
    vignetting_rule(vig, region, px; support_min = 0, order = 2) -> Union{BoundaryRule, RadialSupportRule, Nothing}

The run's rule from its imaged `region` and its masked pixel lists `px` (per view, Julia pixel positions `(u, v)`): a tensor surface's
`vignetting_rule(vig, region; order)`; a radial profile's [`RadialSupportRule`](@ref) — node `i` is supported when its data weight
`Σ_pixels N_i(ρ)²` is at least `support_min` × the mean weight over the nodes (`support_min = 0`: one masked pixel with
`N_i(ρ) > 0` suffices, the rule before 2026-10-30), `P_0` and `P_1` always (the pole tie's parameter); the dependent nodes continue the
supported ones with a zero `order`-th difference (Core's `BoundaryRule(mask; order)`, an `ArgumentError` for an order not 2 or 3).
"""
vignetting_rule(v::AbstractVignetting, region, px; support_min::Real = 0.0, order::Integer = 2) = vignetting_rule(v, region; order)
function vignetting_rule(v::RadialVignetting, region, px; support_min::Real = 0.0, order::Integer = 2)
    w = data_weight(𝐩 -> VrtkCore.coefficient_jacobian(v, 𝐩), px, VrtkCore.n_control_points(v))   # Σ N_i(ρ)² over the masked pixels
    (; weight, supported) = node_support(w, support_min)          # the camera's one support rule (node_support.jl)
    supported[1] = supported[2] = true                            # the pole pair: the gauge's parameter, never extrapolated inward
    return RadialSupportRule(supported, weight; order)
end

"""
    vignetting_basis(v::RadialVignetting, rule::RadialSupportRule) -> Matrix{Float64}

The free directions under the support rule on top of the pole tie: Core's `boundary_rule_jacobian` of the rule — one column per
supported node, a dependent node's row the coefficients of its continuation over the supported nodes inward — with the
first two columns (`P_0`, `P_1`) summed into one, the pole tie. With every point supported this is the pole tie's basis.
"""
function vignetting_basis(v::RadialVignetting, rule::RadialSupportRule)
    rule.supported[1] && rule.supported[2] || throw(ArgumentError("vignetting_basis: the pole pair P_0, P_1 of a radial support rule must be supported"))
    J = VgtkCore.boundary_rule_jacobian(rule.rule)                 # N × n_supported, the columns in index order: P_0, P_1, then the rest
    return hcat(J[:, 1] .+ J[:, 2], J[:, 3:end])
end

"""
    tie_pole(vig, rule) -> vig

[`tie_pole`](@ref) and, under a [`RadialSupportRule`](@ref), the dependent nodes set to the rule's continuation of the
supported ones (`vignetting_basis` ties the increments only), so a resumed profile obeys the rule from its first step.
"""
function tie_pole(v::RadialVignetting, rule::RadialSupportRule)
    P = collect(vec(VrtkCore.control_points(tie_pole(v))))
    return VrtkCore.with_control_points(v, VgtkCore.boundary_rule_jacobian(rule.rule) * P[rule.supported])
end
tie_pole(v::AbstractVignetting, rule) = tie_pole(v)

"""
    smoothness_nodes(vig, rule) -> BitMatrix

The nodes the vignetting solve's smoothness penalty reads, laid like `control_points(vig)` (`NX × NY`; a radial profile
`N × 1`): the parameters under the run's rule — a tensor surface's interior nodes (`rule.interior`), a radial profile's supported
ones — or every point without a rule. Its rows are Core's `difference_rows(mask, 3)`.
"""
smoothness_nodes(v::AbstractVignetting, ::Nothing) = trues(size(VrtkCore.control_points(v)))
smoothness_nodes(::VrtkCore.TensorProductVignetting, rule::VgtkCore.BoundaryRule) = BitMatrix(rule.interior)
smoothness_nodes(::RadialVignetting, rule::RadialSupportRule) = BitMatrix(reshape(rule.supported, :, 1))

"""
    gauge_basis(vig, Z) -> Matrix{Float64}

The free directions `Z` ([`vignetting_basis`](@ref)) restricted to the null space of the centre row `c = coefficient_jacobian(vig, 𝐜)`:
with the nodes at V(𝐜) = 1 ([`gauge_center_value`](@ref)), every step `vec(P) = vec(P₀) + Z y` keeps V(𝐜) = 1 EXACTLY — the
gauge fixed by the parametrization (decision 2026-10-30). One free direction is eliminated by substitution:
the column `k` with the largest |c ⋅ Z_k| is expressed through the others. For a radial profile `c = e₁` (clamped knots: V(0) = P_0), so
the pole column (P_0 = P_1) simply drops out and every remaining column is still one node; for a tensor surface the control
point whose basis function is largest at the centre absorbs the others' centre contributions. `Z` itself when the centre depends on no
free direction (nothing to eliminate).
"""
function gauge_basis(vig::AbstractVignetting, Z::AbstractMatrix)
    Zm = Matrix{Float64}(Z)
    r = vec(collect(Float64, VrtkCore.coefficient_jacobian(vig, VgtkCore.center(vig)))' * Zm)      # the centre value's dependence on the free directions
    k = argmax(abs.(r)); abs(r[k]) > 0 || return Zm
    n = size(Zm, 2); E = zeros(n, n - 1); j = 0
    for i in 1:n
        i == k && continue
        j += 1; E[i, j] = 1.0; E[k, j] = -r[i] / r[k]
    end
    return Zm * E
end

"""
    gauge_center_value(vig) -> vig

The nodes scaled so that V(𝐜) = 1 exactly: the start of the gauge held by parametrisation (the overall scale is the lighting's). `vig` itself when
V(𝐜) is not positive.
"""
function gauge_center_value(v::AbstractVignetting)
    c = v(VgtkCore.center(v))
    c > 0 || return v
    return with_control_points(v, collect(Float64, vec(control_points(v))) ./ c)
end



"""
    vignetting_center(camera) -> SVector{2, Float64}

The centre of the radial vignetting profile of `camera`: its PRINCIPAL POINT, fixed (owner ruling 2026-11-11: the centre is no
parameter of the fit, and V(𝐜) = 1 is held there exactly, `gauge_basis`), COLMAP pixels — the one source of truth: `K`'s
principal point. A parametric camera's `K` is physical; a SplineLens start is on its optical axis (`GridStart`: Estimation's
regression (`CameraRegression`) turns its source onto the axis, Core's `axis_gauged`, K's principal point the axis's pixel), so its `K`'s principal
point is the axis too — no separate `OpticalAxis` fit here (a start whose axis fit did not converge, `GridStart`'s, is
not axis-gauged: its `K`'s principal point, not the axis, for that run).
"""
function vignetting_center(camera::VgtkCore.BoundedCamera{<:CameraModel})
    c = VgtkCore.principal_point(VgtkCore.model(camera).intrinsics)
    return SVector{2, Float64}(c[1], c[2])
end
vignetting_center(camera::CameraModel) = throw(ArgumentError(
    "vignetting_center: a bare $(nameof(typeof(camera))) — the profile's centre is a bounded camera's principal point; " *
    "bound it (`BoundedCamera`)"))
vignetting_center(camera) = throw(ArgumentError(
    "vignetting_center: a $(nameof(typeof(camera))) has no principal point the stack fits (a bounded parametric or SplineLens camera has)"))
