# ---------------------------------------------------------------------------
# The grid lens's nodes under the run's imaged region, and its smoothness (the pipeline options' `[grid.smoothness]`, smoothness.jl).
#
#   nodes        the lens HOLDS the nodes whose B-spline support meets the imaged region (Core's `node_mask(model,
#                imaged_region(camera))`, the region the camera's lens coverage and its lens's valid domain give: the run's measured
#                image circle ∩ the sensor, or the sensor), and they are the PARAMETERS. The others are absent (Core's `present` mask,
#                `present_nodes`): no value, NaN in their slots, never read — every imaged pixel's 4 × 4 window holds present nodes
#                only (2026-11-04; before, they were dependent, extrapolated from the interior by Core's `BoundaryRule`, which no
#                imaged pixel read either: the results are the same bits). The tile blocks give them no column (`_patch_fanout!`),
#                a patch block with no present node is no variable, and the lens files store the present nodes alone
#                (`vgio/cameras/v6.0`). `grid.nodes = "all"` holds every node of the rectangle.
#   smoothness   rows √(β Σϕ²/N / m̄) · Σ cₙ (d̂ₙ − d̂₀ₙ) per component, over the unit nodal directions d̂ of the INTERIOR (the start d̂₀,
#                N interior nodes, m̄ Core's `mean_positive_eigenvalue` of the operator D, λ = β Core's `penalty_scale`, so β means the
#                same for every operator, lattice and region; the rows are Core's `row_stencils` of D):
#                `bending` — the analytic bending energy ∫∫ f_xx² + 2 f_xy² + f_yy² of the CORRECTION d̂ − d̂₀ per component (Core's
#                `bending_stencils` of the lattice in pixels, BSplineKit's Galerkin form: one factor per cell, the cells whose 16 nodes
#                are interior; the start is a smooth parametric fit, and the bending of d̂ itself is that of the sphere at the node
#                spacing, which would pull the field of view in);
#                `pspline` — the `order`-th differences (Δ³ of record: order 3) of the correction along both lattice axes over the windows of order + 1 consecutive
#                interior nodes (Eilers & Marx's P-spline penalty; measured 2026-11-03: it holds the nodes whose support merely
#                grazes the region — nearly undetermined by the data, 1.8 px to 1e7 px without it — without biasing the
#                well-determined ones; the default);
#                `none`.
#   The grid A/B on the synthetic Siemens case (experiments/grid_prior/) measured them; the defaults are its winner.
#   weight       `[grid.smoothness] weight` β, or "gcv": β minimizing generalized cross-validation on the LINEARIZATION of the grid problem at
#                the start (the data's Gauss–Newton system over the grid's interior nodes, the other blocks held at the start; the
#                start's smoothness residual is zero; Estimation's `gcv_curve` and `GeneralizedCrossValidation`) — `select_grid_smoothness_weight`, after the
#                closed-form start of the radiometry.
# ---------------------------------------------------------------------------

"""The rules of the grid's nodes (the pipeline options' `grid.nodes`): those the imaged region reaches, or every node."""
const GRID_NODE_RULES = (:interior, :all)

"""
    grid_present(lens; nodes = :interior) -> BitMatrix

The nodes a grid lens holds under the region its model images (laid `[gy, gx]` like the model's controls): Core's footprint rule
`node_mask(model, imaged_region(camera))`, the nodes whose B-spline support meets the region (never a node the lens does not hold).
`nodes = :all` is every node the lens holds — every node of the lattice for a lens that holds them all (the grid A/B's other arm).
"""
function grid_present(lens; nodes::Symbol = :interior)
    nodes in GRID_NODE_RULES || throw(ArgumentError("grid_present: nodes = :$nodes — expected one of $(GRID_NODE_RULES)"))
    m = VgtkCore.model(lens)
    nodes === :all && return VgtkCore.present_nodes(m)
    return BitMatrix(VgtkCore.node_mask(m, VgtkCore.imaged_region(lens)))
end

"""The present nodes of a `[gy, gx]` mask laid `[g_x, g_y]` like the patch blocks."""
interior_nodes(present::AbstractMatrix{Bool}) = BitMatrix(permutedims(present))

"""
    GridSmoothness

The smoothness of a run's grid: per row the terms `(node, coefficient)` (node = the linear index of the `[g_x, g_y]` lattice), the start
unit directions `d0` (`[g_x, g_y]`), the row weight `w` (√(β Σϕ²/N / m̄)), and for the report the smoothness `type`, β and m̄.
"""
struct GridSmoothness
    type::Symbol
    rows::Vector{Vector{Tuple{Int, Float64}}}
    d0::Matrix{SVector{3, Float64}}
    w::Float64
    β::Float64
    m̄::Float64
end

"""
    grid_smoothness(sm, lens, ϕ², interior; weight = smoothness_weight(sm)) -> Union{GridSmoothness, Nothing}

The smoothness `sm` (`[grid.smoothness]`: a pspline of its `order`-th differences, a bending, or none) of the grid lens `lens` (its
controls the reference d̂₀) over its `interior` nodes (`[g_x, g_y]`) of weight `weight` β relative to the data energy `ϕ² = Σϕ²`;
`nothing` for none or `β = 0`.
"""
function grid_smoothness(sm::AbstractSmoothness, lens, ϕ²::Real, interior::AbstractMatrix{Bool}; weight::Real = smoothness_weight(sm))
    weight >= 0 || throw(ArgumentError("grid_smoothness: weight β = $weight < 0"))
    (sm isa NoSmoothness || weight == 0) && return nothing
    G_x, G_y = grid_dims(lens)
    size(interior) == (G_x, G_y) || throw(DimensionMismatch("grid_smoothness: interior $(size(interior)) for a $G_x × $G_y grid"))
    N = count(interior)
    d0 = [normalize(SVector{3, Float64}(_direction_node(lens, gx - 1, gy - 1))) for gx in 1:G_x, gy in 1:G_y]
    D = _smoothness_rows(sm, lens, interior)
    rows = [collect(zip(nodes, coefficients)) for (nodes, coefficients) in VgtkCore.row_stencils(D)]
    m̄ = VgtkCore.mean_positive_eigenvalue(D)
    return GridSmoothness(smoothness_type(sm), rows, d0, sqrt(weight * VgtkCore.penalty_scale(ϕ², N, m̄)), Float64(weight), m̄)
end
# the operator over the interior nodes: Core's masked `bending_rows` (the lattice in pixels) or its `order`-th differences
_smoothness_rows(::BendingSmoothness, lens, interior) = VgtkCore.bending_rows(interior; spacing = VgtkCore.lattice_spacing(VgtkCore.model(lens)))
_smoothness_rows(sm::PSplineSmoothness, lens, interior) = VgtkCore.difference_rows(interior, sm.order)

"""The smoothness cost ½‖rows‖² of the grid lens `lens` (0 without a smoothness)."""
function grid_smoothness_cost(p::GridSmoothness, lens)
    G_x = size(p.d0, 1)
    c = 0.0
    for row in p.rows
        acc = zero(SVector{3, Float64})
        for (n, cf) in row
            gx, gy = mod1(n, G_x), fld1(n, G_x)
            acc += cf * (normalize(SVector{3, Float64}(_direction_node(lens, gx - 1, gy - 1))) - p.d0[n])
        end
        c += sum(abs2, p.w * acc)
    end
    return 0.5 * c
end
grid_smoothness_cost(::Nothing, lens) = 0.0

"""
    GridSmoothnessResidual{NP, GP}

The smoothness rows whose nodes lie in the `NP` (1 to 4) patch blocks `idx` (of type `GP`): per row the 3 components
`w (Σ c d̂ₙ − bᵣ)`, `bᵣ = Σ c d̂₀ₙ`, over its terms `(patch slot, local node, coefficient, columns)`. The column of node n's tangent
coordinate k is `w c tₖ / ‖dₙ‖` (`(t₁, t₂)` its patch chart's tangents), present when bit k of `columns` is set: an absent node has
none, a coordinate the rotation gauge holds (`rotation_gauge_coordinates`) none of its own.
"""
struct GridSmoothnessResidual{NP, GP} <: NLLSsolver.AbstractResidual
    idx::SVector{NP, Int}
    terms::Vector{Vector{Tuple{Int, Int, Float64, UInt8}}}
    b::Vector{SVector{3, Float64}}
    w::Float64
end
NLLSsolver.ndeps(::GridSmoothnessResidual{NP}) where {NP} = static(NP)
NLLSsolver.nres(r::GridSmoothnessResidual) = 3 * length(r.terms)
NLLSsolver.varindices(r::GridSmoothnessResidual) = r.idx
NLLSsolver.getvars(r::GridSmoothnessResidual{NP, GP}, vars::Vector) where {NP, GP} = ntuple(i -> vars[r.idx[i]]::GP, Val(NP))
function NLLSsolver.computeresidual(r::GridSmoothnessResidual, ps::_Patch...)
    out = Vector{Float64}(undef, 3 * length(r.terms))
    for (si, tm) in enumerate(r.terms)
        acc = -r.b[si]
        for (q, n, c, _) in tm
            acc += c * normalize(SVector{3, Float64}(_patch_dirs(ps[q])[n]))
        end
        out[3si - 2] = r.w * acc[1]; out[3si - 1] = r.w * acc[2]; out[3si] = r.w * acc[3]
    end
    return out
end
function NLLSsolver.computeresjac(varflags, r::GridSmoothnessResidual{NP}, ps::_Patch...) where {NP}
    res = NLLSsolver.computeresidual(r, ps...)
    col0 = zeros(Int, NP); ncol = 0
    for q in 1:NP
        _bit(varflags, q) || continue
        col0[q] = ncol; ncol += Int(NLLSsolver.nvars(ps[q]))
    end
    J = zeros(Float64, length(res), ncol)
    for (si, tm) in enumerate(r.terms), (q, n, c, cols) in tm
        (cols != 0 && _bit(varflags, q)) || continue
        d = SVector{3, Float64}(_patch_dirs(ps[q])[n]); t1, t2 = _control_tangents(d); s = r.w * c / norm(d)
        col = col0[q] + 2 * (n - 1)
        for comp in 1:3
            (cols & 0x01) != 0 && (J[3si - 3 + comp, col + 1] += s * t1[comp])
            (cols & 0x02) != 0 && (J[3si - 3 + comp, col + 2] += s * t2[comp])
        end
    end
    return res, J
end

"""Add the smoothness's rows to a grid problem's residual blocks `blocks`, of `p = (px, py)` node patches with block indices `ip` (patch type `GP`), grouped by the
patch blocks their nodes lie in (a bending row's 4 × 4 window: up to 2 × 2 blocks); `free` the free-node mask (`[g_x, g_y]`), `held_rotation_coordinates`
the direction coordinates the rotation gauge holds (no column of their own)."""
function _add_grid_smoothness!(blocks, smooth::GridSmoothness, ip::AbstractMatrix{Int}, p::Tuple{Int, Int}, free::AbstractMatrix{Bool}, ::Type{GP};
                          held_rotation_coordinates = _NO_ROTATION_GAUGE) where {GP}
    px_n, py_n = p
    G_x = size(smooth.d0, 1)
    patch(gx, gy) = (fld(gx - 1, px_n) + 1, fld(gy - 1, py_n) + 1)
    local_node(gx, gy) = (gx - (fld(gx - 1, px_n)) * px_n) + (gy - (fld(gy - 1, py_n)) * py_n - 1) * px_n
    groups = Dict{Vector{Tuple{Int, Int}}, Tuple{Vector{Vector{Tuple{Int, Int, Float64, UInt8}}}, Vector{SVector{3, Float64}}}}()
    columns(g) = free[g...] ? UInt8(!_rotation_held(held_rotation_coordinates, g..., 1)) | (UInt8(!_rotation_held(held_rotation_coordinates, g..., 2)) << 1) : 0x00
    for row in smooth.rows
        nodes = [(mod1(n, G_x), fld1(n, G_x)) for (n, _) in row]
        key = sort(unique([patch(g...) for g in nodes]))
        length(key) <= 4 || throw(ArgumentError("grid smoothness: a row spans $(length(key)) patch blocks (at most 4)"))
        tm = [(findfirst(==(patch(g...)), key), local_node(g...), c, columns(g)) for (g, (_, c)) in zip(nodes, row)]
        b = sum(c * smooth.d0[n] for (n, c) in row)
        g = get!(() -> (Vector{Tuple{Int, Int, Float64, UInt8}}[], SVector{3, Float64}[]), groups, key)
        push!(g[1], tm); push!(g[2], b)
    end
    for key in sort!(collect(keys(groups)))
        tms, bs = groups[key]
        NP = length(key)
        push!(blocks, GridSmoothnessResidual{NP, GP}(SVector{NP, Int}([ip[k...] for k in key]), tms, bs, smooth.w))
    end
    return blocks
end

"""
    select_grid_smoothness_weight(f, θ, sm, interior; patch = GRID_PATCH, build_kwargs...) -> (smoothness, selection)

The grid smoothness `sm` (its type and order; its weight the one chosen here) over the `interior` nodes (`[g_x, g_y]`) with its weight β chosen by generalized cross-validation
(Estimation's `gcv_curve` and `GeneralizedCrossValidation`, `sm`'s weight) on the LINEARIZATION of the grid problem at the state's current values (the run's start): the Gauss–Newton
system `JᵀJ`, `Jᵀr` of the per-tile data residuals over the columns of the grid's free patch blocks — the other blocks (poses,
focus, aperture, material, vignetting, lighting) held where they are, so it is the criterion of the grid's own update — and the
unit-weight smoothness's `DᵀD` over the same columns (its residual is zero at the start, its reference d̂₀ the start). `λ = w² =
β Σϕ² / N / m̄` (the smoothness's row weight squared), so β is the same normalized weight `[grid.smoothness] weight` states. The problem is
`build_problem_grid(f, θ; patch, smoothness, build_kwargs...)`, the LM's own; one Jacobian evaluation of every tile and one
dense generalized eigendecomposition of the free columns (O(n³) in their number: 2 per present node, up to 2·7875 for a 105 × 75
grid). `(nothing, nothing)` for none. `selection` is `_gcv_selection`'s record plus the counts of the linearization.
"""
function select_grid_smoothness_weight(f::CalibrationData, θ::CalibrationParameters, sm::AbstractSmoothness, interior::AbstractMatrix{Bool}; patch::Tuple{Int, Int} = GRID_PATCH,
                                       build_kwargs...)
    lens = θ.camera.geometric
    ϕ² = data_norm2(f)
    unit = grid_smoothness(sm, lens, ϕ², interior; weight = 1.0)
    unit === nothing && return nothing, nothing
    p1 = GridSmoothness(unit.type, unit.rows, unit.d0, 1.0, unit.β, unit.m̄)       # row weight 1: its Gauss–Newton block is DᵀD
    problem, idx = build_problem_grid(f, θ; patch, smoothness = p1, build_kwargs...)
    free_blocks = sort(setdiff(vec(idx.patches), idx.dependent_patches))
    # the Gauss–Newton systems of the per-tile data and of the unit-weight smoothness over the free patch blocks, every
    # other block held: Core's linearization of the bundle (NLLSsolver's visit order; the tiles of a view evaluated in one
    # batch, one at a time, at the vector its `prepare_evaluation!` names)
    data, smooth_rows = _with_registered(f, θ, problem, idx; concurrent = false, vectors = problem.solver) do
        VgtkCore.linearize(problem, problem.variables; free = free_blocks, classes = (PerTileResidual, GridSmoothnessResidual))
    end
    G, h, rr, n = data.G, data.h, data.S, data.nrows      # min ‖J δ − (−res)‖²: the data's linearization
    Q, q = smooth_rows.G, smooth_rows.h                    # ‖J_p δ − (−res_p)‖²: the unit-weight smoothness's
    ncol = size(G, 1)
    sel = _gcv_selection(VgtkEstimation.gcv_curve(G, h, rr, Q, q, n), VgtkCore.penalty_scale(ϕ², count(interior), unit.m̄), smoothness_weight(sm))
    smooth = grid_smoothness(sm, lens, ϕ², interior; weight = sel.β)
    return smooth, (; sel..., kind = smoothness_type(sm), n_observations = n, n_columns = ncol,
                   linearization = "the grid's free patch blocks at the start (Gauss-Newton of the per-tile data; other blocks held)")
end

# ---- the start perturbation (the pipeline options' `grid.start_perturbation`) --------------------------------------------------
# The robustness start of the grid A/B (grid-prior-ab's RC_GRID_PERTURB): every direction control of the start grid rotated on its
# sphere by σ/f radians per tangent axis. Applied after the masks (they are the unperturbed start's, as on the branch) and before
# the smoothness is formed, so the smoothness's reference d̂₀ is the PERTURBED start, and the run's start grid (grid_init.json) is that grid.

"""
    perturb_directions(rng, lens, sigma_px, f_px) -> lens

The grid lens `lens` with every direction control `d` (the full lattice, every node of it drawn for) moved on its own sphere by Core's
control chart (`ControlPatchManifold` of the one control, `‖d‖ · normalize(d + δ₁t₁ + δ₂t₂)`, its norm kept) with
`(δ₁, δ₂) = ‖d‖ · (sigma_px / f_px) · (z₁, z₂)`, `z ~ N(0, I₂)` drawn from `rng` node by node in lattice order (`g_x` fastest):
a rotation by `atan((sigma_px / f_px) ‖z‖)`, `sigma_px / f_px` radians RMS per tangent axis. Its K, bound and lens coverage are kept.
"""
function perturb_directions(rng::AbstractRNG, lens::GridLike, sigma_px::Real, f_px::Real)
    (isfinite(sigma_px) && sigma_px >= 0) || throw(ArgumentError("perturb_directions: sigma_px = $sigma_px is not a finite σ ≥ 0"))
    (isfinite(f_px) && f_px > 0) || throw(ArgumentError("perturb_directions: f_px = $f_px is not a finite focal length > 0"))
    D = directions(VgtkCore.model(lens))
    s = sigma_px / f_px
    G_y, G_x = size(D)
    out = similar(D)
    for gy in 1:G_y, gx in 1:G_x
        d = SVector{3, Float64}(D[gy, gx])
        z = SVector(randn(rng), randn(rng))
        out[gy, gx] = VgtkCore.retract(VgtkCore.ControlPatchManifold(SMatrix{1, 1}(d)), norm(d) * s * z).directions[1]
    end
    return VgtkCore.with_directions(lens, out)
end

"""
    perturb_start(lens, perturbation, f_px) -> lens

The run's start grid under the options' `grid.start_perturbation` (`nothing`: `lens` itself): [`perturb_directions`](@ref) with
the version-stable `StableRNG(seed)` (as the masks' draws), deterministic from the seed; `perturbation` is the options'
`StartPerturbationOptions` (`sigma_px`, `seed`). A parametric lens has no grid to perturb (`ArgumentError`).
"""
function perturb_start(lens, perturbation, f_px::Real)
    perturbation === nothing && return lens
    is_grid_lens(lens) ||
        throw(ArgumentError("grid.start_perturbation: the start lens is $(lens_type_name(lens)), not a grid lens (the option perturbs a grid's controls)"))
    return perturb_directions(StableRNG(perturbation.seed), lens, perturbation.sigma_px, f_px)
end
