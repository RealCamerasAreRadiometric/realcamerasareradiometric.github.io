# ---------------------------------------------------------------------------
# Per-tile cost blocks of the grid lens (Core's SplineLens camera). A tile evaluates Core's
# `window` of the camera: its 2 × 2 patches' controls in the full camera's lattice, a camera of its own.
#
# NLLSsolver caps a cost block at MAX_ARGS = 10 variable dependencies and stores one dense
# Jacobian per cost block, so the grid's sparsity is exposed by granularity: the node grid is
# partitioned into fixed p × p patches (Core's `ControlPatchManifold`, 2 tangent parameters per node), and the
# image into tiles such that the bicubic support of every pixel in a tile (±1 cell footprint
# margin) lies inside a 2 × 2 patch window — a tile of cells [I·p + 1, (I+1)·p] has support
# nodes [I·p, (I+1)·p + 4] ⊂ patches {I, I+1} whenever p ≥ 5. Each tile then depends on
# exactly 10 blocks: pose, the 4 window patches, focus, aperture, material, vignetting,
# lighting. The Jacobian reuses the per-view machinery: AD slots for the pose (6), the
# interpolated chief directions (3 per sensor sample — the same chain-rule trick as the focus
# surface: the node → direction map is the constant bicubic synthesis), the focus samples, the
# material (8) and the scalar aperture radius r (one slot through the unit-disc rule r² w̃_l L_c(·, r â_l));
# vignetting, lighting and μ analytic; the board warping h (nodal heights) as dual seeds; the printed-edge parameters by finite differences.
# ---------------------------------------------------------------------------

# The patch blocks: VisualGeometryToolkitCore's `ControlPatchManifold` of a p × p block of the grid's controls (laid `[g_x, g_y]`; the
# block's coordinates per node in that column-major order): the direction controls, 2 coordinates per node on Core's
# direction chart (`direction_tangents`, the control's norm kept). NLLSsolver variables through Core's manifold adapter.
const _Patch{PX, PY} = VgtkCore.ControlPatchManifold{PX, PY, <:Any, Float64, Nothing}
"""
    GRID_PATCH = (5, 5)

The grid route's patch size in nodes `(p_x, p_y)`: the one constant [`build_problem_grid`](@ref) tiles the lattice with (its
static-array patch blocks; a larger block explodes codegen, so the size is no option) and the exact start pads its lattice to
(`pad_lattice`): both node counts of a grid run's lattice are multiples of it.
"""
const GRID_PATCH = (5, 5)

"""The patch block of the direction controls `d` (a `[g_x, g_y]` matrix)."""
_patch_block(d::AbstractMatrix) = VgtkCore.ControlPatchManifold(SMatrix{size(d, 1), size(d, 2), SVector{3, Float64}, length(d)}(d))
# The grid's one rotation gauge: turning every node direction and every view's camera frame by one rotation leaves the rays in the
# world, so the board's image, all but unchanged — only the aperture disc and the focus surface, fixed in the camera frame, do not turn
# with it (5e-7 of the cost on a test lens), a direction the data hold by nothing else. It is
# held EXACTLY by three direction coordinates whose columns stay zero in every tile and smoothness row (the λ·I-damped step of a zero
# column is zero): both tangent coordinates of the chief node (the axis) and the one tangent coordinate of a second node most
# transversal to the roll about it — a slice of the gauge orbit (`rotation_gauge_coordinates`), no pin, prior or penalty.
const _NO_ROTATION_GAUGE = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
@inline _rotation_held(held::NTuple{3, NTuple{3, Int}}, gx::Integer, gy::Integer, c::Integer) =
    (held[1] == (gx, gy, c)) | (held[2] == (gx, gy, c)) | (held[3] == (gx, gy, c))
"""
    rotation_gauge_coordinates(lens, free = nothing) -> NTuple{3, NTuple{3, Int}}

The three direction coordinates `(g_x, g_y, tangent)` that hold the grid lens's common rotation (with the poses) exactly: both
tangent coordinates of the chief node (the present node whose direction is closest to the optical axis) and, of the node farthest from it in angle, the tangent coordinate (Core's `direction_tangents`, the patch chart's) most
aligned with the roll about the chief direction, `|t_k · (ĉ × d̂)|` largest. The three are a nonsingular map of the rotation ω
(`t · (ω × d) = ω · (d × t)`), so holding them removes exactly the gauge and no degree of freedom of the lens.
"""
function rotation_gauge_coordinates(lens, free = nothing)
    D = directions(VgtkCore.model(lens))                                 # [gy, gx]
    ok(gx, gy) = (free === nothing || free[gx, gy]) && all(isfinite, D[gy, gx])
    chief = (0, 0); zbest = -Inf
    for gx in axes(D, 2), gy in axes(D, 1)
        ok(gx, gy) || continue
        z = normalize(SVector{3, Float64}(D[gy, gx]))[3]
        z > zbest && (zbest = z; chief = (gx, gy))
    end
    chief == (0, 0) && return _NO_ROTATION_GAUGE
    ĉ = normalize(SVector{3, Float64}(D[chief[2], chief[1]]))
    roll = (0, 0, 0); best = 0.0
    for gx in axes(D, 2), gy in axes(D, 1)
        ok(gx, gy) && (gx, gy) != chief || continue
        d = SVector{3, Float64}(D[gy, gx])
        a = cross(ĉ, normalize(d))
        for (k, t) in enumerate(_control_tangents(d))
            v = abs(dot(t, a))
            v > best && (best = v; roll = (gx, gy, k))
        end
    end
    roll == (0, 0, 0) && return _NO_ROTATION_GAUGE
    return ((chief[1], chief[2], 1), (chief[1], chief[2], 2), roll)
end
"""The direction controls of a patch block (`[g_x, g_y]`)."""
_patch_dirs(p) = VgtkCore.model(p).directions
"""Whether a patch block's present nodes (the block `(PI, PJ)` of `free`, `[g_x, g_y]`; `nothing`: every node) are finite: an absent
node is NaN by design (`with_present`)."""
function _patch_finite(p, free, PI::Integer, PJ::Integer)
    D = _patch_dirs(p)
    px, py = size(D)
    for b in 1:py, a in 1:px
        (free === nothing || free[PI * px + a, PJ * py + b]) || continue
        all(isfinite, D[a, b]) || return false
    end
    return true
end
"""The tangent basis a control `d` moves in: Core's `direction_tangents` of d̂ (the patch blocks' chart)."""
@inline _control_tangents(d) = VgtkCore.direction_tangents(normalize(SVector{3, Float64}(d)))


"""
    PerTileResidual{N}

Cost block of one image tile of view `k`: `pix` indexes into `state.px[k]`, `ϕt` the matching
target rows and `ωt` their weights (weighting.jl: `Unweighted()`, or the view's 1/σ of those rows), `P0` the 0-based patch coordinates of the window origin. `idx` order:
pose, patch(0,0), patch(1,0), patch(0,1), patch(1,1), focus, aperture, material, vignetting, lighting.
`GP` is the patch block type (Core's `ControlPatchManifold`) and `AV` the aperture block type (`ApertureVar`).
"""
struct PerTileResidual{N, S <: CalibrationState, GP <: VgtkCore.Manifold, AV <: ApertureVar} <: NLLSsolver.AbstractResidual
    k::Int
    pix::Vector{Int}
    ϕt::Vector{Float64}
    ωt::Union{Unweighted, Vector{Float64}}
    P0::Tuple{Int, Int}
    idx::SVector{10, Int}
    state::S
    dev::Base.RefValue{Any}                              # lazy device data (pixel-id list); nothing on CPU
    free::Union{Nothing, BitMatrix}                      # the present nodes (`[g_x, g_y]`, per_tile_smoothness.jl); an absent node has no column
    held_rotation_coordinates::NTuple{3, NTuple{3, Int}}              # the three direction coordinates (g_x, g_y, tangent) held (the rotation gauge); zeros: none
end
NLLSsolver.ndeps(::PerTileResidual) = static(10)
NLLSsolver.nres(r::PerTileResidual) = n_channels(r.state) * length(r.pix)   # one residual per measured channel and pixel
NLLSsolver.varindices(r::PerTileResidual) = r.idx
# Type-narrowed like PerViewResidual's getvars: without the asserts the downstream @unroll'd
# accumulation is inferred over the whole variable-type union across all 10 slots — a compiler
# memory explosion (the per-view path never hits this because its getvars asserts every slot).
NLLSsolver.getvars(r::PerTileResidual{N, S, GP, AV}, vars::Vector) where {N, S, GP, AV} =
    (vars[r.idx[1]]::_PoseBlock, vars[r.idx[2]]::GP, vars[r.idx[3]]::GP,
     vars[r.idx[4]]::GP, vars[r.idx[5]]::GP, vars[r.idx[6]]::_FocusBlock,
     vars[r.idx[7]]::AV, vars[r.idx[8]]::MaterialVar{Float64},
     vars[r.idx[9]]::VgtkCore.LinearSubspaceManifold, vars[r.idx[10]]::LightingVar)

"""The window's direction controls `[gy, gx]` (Core's layout) of a tile's 2 × 2 patches, each patch's laid `[g_x, g_y]`."""
function _window_directions(p00::_Patch{PX, PY}, p10, p01, p11) where {PX, PY}
    W = Matrix{Vec3{Float64}}(undef, 2PY, 2PX)
    for (q, (ox, oy)) in zip((p00, p10, p01, p11), ((0, 0), (PX, 0), (0, PY), (PX, PY)))
        d = _patch_dirs(q)
        for b in 1:PY, a in 1:PX
            v = d[a, b]; W[oy + b, ox + a] = Vec3{Float64}(v[1], v[2], v[3])
        end
    end
    return W
end
"""The full-grid rows (y) and columns (x) of the window at patch coordinates `P0` (0-based) of `PX × PY` patches."""
_window_block(P0, PX, PY) = ((P0[2] * PY + 1):((P0[2] + 2) * PY), (P0[1] * PX + 1):((P0[1] + 2) * PX))

"""Materialize the 2 × 2 patch window of a tile as Core's `window` of the camera (node origin `P0 .* p`: a bare SplineLens camera of
the block)."""
function _window_lens(base::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}, P0::Tuple{Int, Int}, p00::_Patch{PX, PY},
                      p10::_Patch{PX, PY}, p01::_Patch{PX, PY}, p11::_Patch{PX, PY}) where {PX, PY}
    rows, cols = _window_block(P0, PX, PY)
    return VgtkCore.window(VgtkCore.model(base), rows, cols; directions = _window_directions(p00, p10, p01, p11))
end

"""`(i0, j0)`: the 0-based full-grid node (x, y) of the first control of the window at patch coordinates `P0` of `PX × PY` patches."""
_window_origin(P0, PX::Integer, PY::Integer) = (P0[1] * PX, P0[2] * PY)
"""Tangent bases of the `2PX × 2PY` nodes of a tile window, laid `[a, b]` = window node (x, y) — the same bases as the patch
blocks' chart (`_control_tangents`)."""
_window_bases(lens, PX::Integer, PY::Integer) =
    (D = directions(VgtkCore.model(lens)); [_control_tangents(D[b, a]) for a in 1:2PX, b in 1:2PY])
"""Cell (0-based, x and y, in the lens's own lattice: a window's, or the full camera's) and the four B-spline weights per axis of a
sensor position on a grid lens's lattice (Core's `spline_window` at K⁻¹𝐩; the same window its π⁻¹ reads)."""
@inline function _grid_support(g, 𝐩::StaticVector{2})
    m = VgtkCore.model(g)
    b = _backward(m)
    ix, iy, wx, wy = spline_window(b.area, size(b.controls), VgtkCore.normalized_point(m.intrinsics, 𝐩))
    return ix - 3, wx, iy - 3, wy
end


"""The kernels' copy of a host grid lens: in their float type (`_kernel_float`; `F` another), its arrays adapted to the backend."""
_lens32(be, g, F = _kernel_float(be)) = adapt(be, VgtkCore.convert_eltype(F, VgtkCore.model(g)))   # a bounded grid's bare model: a kernel reads π⁻¹ only

"""
Bicubic fan-out of one pixel's per-sensor-sample chief-direction sensitivities `dget(ks, comp, c)` (∂Φ_c/∂π⁻¹_comp) onto the patch
columns: per node (tangent 1, tangent 2); the columns of the three held direction coordinates of the rotation gauge
(`lay.held_rotation_coordinates`, `rotation_gauge_coordinates`) stay zero, so the damped step leaves them where they are.
Shared by the CPU, device and batched paths (identical arithmetic).
"""
@inline function _patch_fanout!(Jm, row0::Integer, lens64, bases, lay, P0::Tuple{Int, Int}, ::Val{PX}, ::Val{PY}, NS::Integer,
                                𝐩of, dget) where {PX, PY}
    o0 = _window_origin(P0, PX, PY)
    @inbounds for ks in 1:NS
        cx, wx, cy, wy = _grid_support(lens64, 𝐩of(ks))     # in the window's lattice
        for b in 1:4, a in 1:4
            g1 = o0[1] + cx + a - 1; g2 = o0[2] + cy + b - 1   # 0-based global node
            qx = fld(g1, PX) - P0[1]; qy = fld(g2, PY) - P0[2]
            q = 1 + qx + 2 * qy
            pc = lay.patch_cols[q]
            pc > 0 || continue                              # this patch block is fixed
            (lay.free === nothing || lay.free[g1 + 1, g2 + 1]) || continue   # an absent node (`grid_present`): its column has no node
            li = g1 - (P0[1] + qx) * PX + 1; lj = g2 - (P0[2] + qy) * PY + 1
            node = li + (lj - 1) * PX
            b1, b2 = bases[g1 - o0[1] + 1, g2 - o0[2] + 1]
            w = wx[a] * wy[b]
            col = pc - 1 + 2 * (node - 1)
            t1 = !_rotation_held(lay.held_rotation_coordinates, g1 + 1, g2 + 1, 1); t2 = !_rotation_held(lay.held_rotation_coordinates, g1 + 1, g2 + 1, 2)
            for c in 1:3
                acc1 = 0.0; acc2 = 0.0
                for comp in 1:3
                    ∂ = dget(ks, comp, c)
                    acc1 += ∂ * b1[comp]; acc2 += ∂ * b2[comp]
                end
                t1 && (Jm[row0 + c, col + 1] += w * acc1)
                t2 && (Jm[row0 + c, col + 2] += w * acc2)
            end
        end
    end
    return nothing
end

"""The residual of a tile (`computeresidual`): its view batch's flux (`_view_batch!`) or its own batch's, minus the observed values."""
function _block_residual(r::PerTileResidual, pose::_PoseBlock, p00::_Patch, p10::_Patch,
                           p01::_Patch, p11::_Patch, foc::_FocusBlock, ap::ApertureVar,
                           m::MaterialVar, vg::VgtkCore.LinearSubspaceManifold, lt::LightingVar)
    is_valid(m) || return fill(NaN, NLLSsolver.nres(r))      # a step the material block could not represent: rejected
    s = r.state
    # (the material block's finiteness is its `is_valid`, checked above)
    finite = all(isfinite, VgtkCore.model(pose).t) && all(isfinite, VgtkCore.model(pose).R) && all(isfinite, VgtkCore.model(foc).t̂_i) &&
             isfinite(ap.r) && all(isfinite, control_points(VgtkCore.model(vg))) && all(isfinite, coefficients(lt.light)) &&
             all(((p, q),) -> _patch_finite(p, r.free, r.P0[1] + q[1], r.P0[2] + q[2]), zip((p00, p10, p01, p11), ((0, 0), (1, 0), (0, 1), (1, 1))))
    finite || return fill(NaN, NLLSsolver.nres(r))
    vars = (pose, p00, p10, p01, p11, foc, ap, m, vg, lt)
    ent = _view_batch!(s, r, vars, false)
    ivs = ent === nothing ? (1:length(r.pix)) : r.pix
    ent === nothing && (ent = _tile_batch(r, vars, false))
    return _tile_residual(ent.flux, r, ivs)
end

"""
Column layout of a tile block (varflags bits 1..10; patch columns 2 per node, column-major nodes).

The aperture's one column is the scalar r, seeded in ONE slot `off_r + 1`; `NH` = nodal heights of a free board warping (HEIGHT_NFREE columns through the basis);
`NI` = the in-plane print error's parameters (plain AD columns); `held_rotation_coordinates` the three direction coordinates `(g_x, g_y, tangent)` whose columns stay zero (the common
rotation's gauge, `rotation_gauge_coordinates`; zeros: none); `fixed` the material block's (`material_fixed`: with `:material` Θ_m's
9 columns stay zero, as `_jacobian_layout`'s).
AD slots: `off_e = 0 | off_dir = 6 (2NS: a chief direction's two tangent seeds per sensor sample, `_dir_seeded`) | off_f = 6 + 2NS (NS) |
off_m = 6 + 3NS (8) | off_i = off_h (NI) | off_r | P` (the nodal heights take no slot: the height kernel's columns).
"""
function _tile_layout(varflags, N::Integer, NS::Integer, PXY::Integer, NF::Integer, NV::Integer, J::Integer,
                      NH::Integer = 0, NI::Integer = 0; free = nothing,
                      held_rotation_coordinates = _NO_ROTATION_GAUGE, fixed = ())
    off_e = 0; off_dir = 6; off_f = 6 + 2 * NS; off_m = 6 + 3 * NS
    ncol = 0
    scatter = Tuple{Int, Int, Int}[]; col_r = 0; μ_col = 0
    vig_cols = 1:0; light_cols = 1:0; focus_cols = 1:0; height_cols = 1:0; inplane_cols = 1:0
    patch_cols = MVector{4, Int}(0, 0, 0, 0)
    off_h = off_m + 8                                       # where the dual slots continue after the 8 material parameters: the nodal heights take none
    off_i = off_h                                           # (the height kernel's columns); the in-plane print error's seeds (NI of them)
    mat_col0 = -1                                           # the column before the material block's (−1: the block is fixed)
    if _bit(varflags, 1)
        for i in 1:6; push!(scatter, (ncol + i, off_e + i, 0)); end
        ncol += 6
    end
    for q in 1:4                                        # patch_cols: first column of the block (0 = fixed)
        if _bit(varflags, 1 + q); patch_cols[q] = ncol + 1; ncol += 2 * PXY; end
    end
    if _bit(varflags, 6); focus_cols = (ncol + 1):(ncol + N); ncol += N; end
    if _bit(varflags, 7); col_r = ncol + 1; ncol += 1; end
    if _bit(varflags, 8)
        mat_col0 = ncol
        if !(:material in fixed)                            # (Θ_m held: its 9 columns stay zero; the height columns as usual)
            for i in 1:8; push!(scatter, (ncol + i, off_m + i, 0)); end
            μ_col = ncol + 9
        end
        nhf = NH > 0 ? HEIGHT_NFREE : 0
        NH > 0 && (height_cols = (ncol + 10):(ncol + 9 + nhf))      # the free height parameters (AD, projected by the material block's basis)
        if NI > 0                                                            # the in-plane print error: plain AD columns after the heights
            inplane_cols = (ncol + 10 + nhf):(ncol + 9 + nhf + NI)
            for i in 1:NI; push!(scatter, (ncol + 9 + nhf + i, off_i + i, 0)); end
        end
        ncol += 9 + nhf + NI
    end
    if _bit(varflags, 9); vig_cols = (ncol + 1):(ncol + NF); ncol += NF; end
    if _bit(varflags, 10); light_cols = (ncol + 1):(ncol + 3J); ncol += 3J; end
    off_r = off_i + NI                                      # (= 14 + 3 NS without heights and in-plane error: 6 pose + 3 NS (2 chief-direction + 1 t̂ per sensor sample) + 8 material)
    rad = col_r > 0                                         # the scalar r: ONE seed at off_r + 1 through r² w̃_l and r â_l
    rad && push!(scatter, (col_r, off_r + 1, 0))
    return (; ncol, scatter, col_r, μ_col, mat_col0, vig_cols, light_cols, focus_cols, patch_cols, height_cols, inplane_cols,
            off_e, off_dir, off_f, off_m, off_h, off_i, off_r, NH, NI, P = off_r + (rad ? 1 : 0), free, held_rotation_coordinates)
end

# ---------------------------------------------------------------------------
# The blocks of `build_problem_grid`.
# ---------------------------------------------------------------------------
"""`(PatchT, ApertureT)` of a grid problem over `lens0` with `px × py` patches."""
_grid_block_types(::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}, px::Integer, py::Integer) =
    (typeof(_patch_block(zeros(SVector{3, Float64}, px, py))), ApertureVar{Float64})
"""The patch variable of the nodes `rows × cols` of `lens0`."""
_patch_var(lens0::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}, rows, cols) = _patch_block(_patch_nodes(lens0, rows, cols))
"""The direction controls of the nodes `rows × cols` (1-based x, y) of a grid lens, laid `[g_x, g_y]`."""
_patch_nodes(lens0, rows, cols) = [SVector{3, Float64}(_direction_node(lens0, gx - 1, gy - 1)) for gx in rows, gy in cols]
"""The aperture variable: the scalar radius of the state."""
_aperture_var(::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}, s::CalibrationState) = ApertureVar(aperture_radius(s))

"""
    build_problem_grid(s; patch = GRID_PATCH, fixed = (), height_degree = 0,
                       smoothness = nothing, inplane = false) -> (problem::LmProblem, idx)

The per-tile LM problem over the grid-lens state, VisualGeometryToolkitCore's `LmProblem` (the stack's one LM, `lmsolve_chart!`):
pose variables per view, one patch block per grid patch, the shared focus/aperture/material/vignetting/lighting blocks,
and one `PerTileResidual` per non-empty image tile (ndeps = 10 = NLLSsolver's MAX_ARGS). Held (the problem's `fixed` variables,
`idx.free` the free mask): the dependent patches and the blocks `fixed` names (of `(:poses, :grid, :focus, :aperture, :material,
:vignetting, :lighting)`, `:grid` the patch blocks; `:material` holds Θ_m alone while a free surface parameter rides the block,
`_hold_material`); another name is an `ArgumentError`. The
patch size must satisfy `p ≥ 5` (the 2 × 2-window guarantee) and divide both grid dimensions —
and it must stay SMALL: NLLSsolver's Hessian accumulation (`updatesymA!`) slices the per-cost
Hessian with `nvars`-sized static ranges and static broadcasts, so a patch of 8 × 8 nodes
(128 parameters) makes LLVM materialize fully unrolled 128 × 128 static-array code for every
dep pair — a codegen memory explosion (hundreds of GB). 5 × 5 nodes (50 parameters) compiles
fine; pick the grid size as multiples of 5 (default 65 × 50).

* `height_degree > 0`: the material block also carries the board warping h (`board_height`; the `HEIGHT_NFREE` free nodal-height
  parameters, dual-number columns). A FIXED warping (nodal heights set, degree 0) is rendered by every route; the free columns need the
  grid route.
* `inplane`: the material block also carries the in-plane print error `(a, u)` of the surface (Core's `Aspect ∘ Skew`,
  `inplane_calibration`; two dual-number columns, additive, no gauge). A fixed error (parameters set, `inplane = false`) is rendered
  by every route.
* `:height`, `:inplane` in `fixed` (`SURFACE_HOLDS`): that surface parameter held at its start (the block carries no columns
  for it), as on the per-view route.
The parameters are the nodes the lens holds (Core's `present_nodes`; `with_present`, `grid_present`): a pixel whose window holds
an absent node is an `ArgumentError` (no residual reads an absent node), and a patch block with no present node is no variable.
* `smoothness` (a `GridSmoothness`, `grid_smoothness`; `nothing` = none): its rows, one cost block per patch block or pair of them.
"""
function build_problem_grid(s::CalibrationState{N}; patch::Tuple{Int, Int} = GRID_PATCH, fixed = (),
                            height_degree::Integer = 0,
                            smoothness = nothing, rotation_gauge::Bool = !(:poses in Symbol.(fixed)),
                            inplane::Bool = false) where {N}
    fixed = Tuple(Symbol(k) for k in fixed)                 # the holds by name, once (a config's strings): every test below reads these
    lens0 = s.camera.geometric
    is_grid_lens(lens0) || throw(ArgumentError("build_problem_grid needs grid-lens views (Core's SplineLens camera; loader lens = :grid)"))
    G_x, G_y = grid_dims(lens0)
    C_x, C_y = G_x - 3, G_y - 3
    px_n, py_n = patch
    (px_n >= 5 && py_n >= 5) || throw(ArgumentError("patch size must be ≥ 5 nodes (2×2-window coverage)"))
    (G_x % px_n == 0 && G_y % py_n == 0) || throw(ArgumentError("patch size must divide the grid ($G_x × $G_y)"))
    nPx = G_x ÷ px_n; nPy = G_y ÷ py_n
    K = n_views(s)
    vv = VgtkCore.restrict(VgtkCore.ParameterManifold(s.camera.vignetting), vignetting_basis(s))   # Θ_v: Core's LinearSubspaceManifold
    lv = LightingVar(s.light; s.channel)
    PatchT, AVT = _grid_block_types(lens0, px_n, py_n)
    NH = _surface_held(fixed, :height) ? 0 : height_nparams(height_degree)   # the surface's parameters: the model's, unless `fixed` holds them
    inplane = inplane && !_surface_held(fixed, :inplane)
    material_held, blocks_held = _hold_material(fixed, NH > 0 || inplane)   # Θ_m alone while a surface parameter rides
    mv = MaterialVar(s.mat, s.board.state, s.board; height = NH, inplane, s.channel, fixed = material_held, s.mu_bounds)
    TileT = PerTileResidual{N, typeof(s), PatchT, AVT}
    variables = Any[]; blocks = Any[]
    held = VgtkCore.present_nodes(VgtkCore.model(lens0))
    free = all(held) ? nothing : interior_nodes(held)
    # the common rotation of the directions and the poses, held once by three direction coordinates (while the poses can turn with it)
    held_rotation_coordinates = rotation_gauge ? rotation_gauge_coordinates(lens0, free) : _NO_ROTATION_GAUGE
    for k in 1:K
        push!(variables, VgtkCore.LeftRigidManifold6(s.poses[k]))
    end
    ip = Matrix{Int}(undef, nPx, nPy)
    L = px_n * py_n
    for PJ in 0:(nPy - 1), PI in 0:(nPx - 1)
        rows = (PI * px_n + 1):((PI + 1) * px_n); cols = (PJ * py_n + 1):((PJ + 1) * py_n)
        push!(variables, _patch_var(lens0, rows, cols)); ip[PI + 1, PJ + 1] = length(variables)
    end
    smoothness === nothing || _add_grid_smoothness!(blocks, smoothness, ip, (px_n, py_n), free === nothing ? trues(G_x, G_y) : free, PatchT; held_rotation_coordinates)
    push!(variables, focus_block(focus(s)), _aperture_var(lens0, s), mv, vv, lv)
    i_f, i_r, i_m, i_v, i_l = (length(variables) - 4):length(variables)
    _add_focus_smoothness!(blocks, i_f, N, s.focus_smoothness)
    n_tiles = 0
    for k in 1:K
        groups = Dict{Tuple{Int, Int}, Vector{Int}}()
        for (i, (u, v)) in enumerate(s.px[k])
            𝐩 = 𝐩_of_pixel(u, v)
            cx, _, cy, _ = _grid_support(lens0, 𝐩)
            I = clamp(fld(cx - 1, px_n), 0, nPx - 2)
            Jt = clamp(fld(cy - 1, py_n), 0, nPy - 2)
            push!(get!(() -> Int[], groups, (I, Jt)), i)
        end
        for ((I, Jt), pix) in groups
            # the 2×2-window guarantee: every pixel's support (± 1 cell margin) inside patches {I, I+1} × {J, J+1}
            for i in pix
                u, v = s.px[k][i]
                free === nothing || _check_reads_present(lens0, s.settings.sensor, u, v, k)
                𝐩 = 𝐩_of_pixel(u, v)
                cx, _, cy, _ = _grid_support(lens0, 𝐩)
                ((max(cx - 1, 0) >= I * px_n) && (min(cx + 1, C_x - 1) + 3 <= (I + 2) * px_n - 1) &&
                 (max(cy - 1, 0) >= Jt * py_n) && (min(cy + 1, C_y - 1) + 3 <= (Jt + 2) * py_n - 1)) ||
                    throw(AssertionError("tile support escapes its 2×2 patch window — decrease the tile shift or increase patch"))
            end
            nc = n_channels(s)                                  # the measured channels' entries of the tile's pixels
            rows = [nc * (i - 1) + c for i in pix for c in 1:nc]   # the tile's rows of its view: its ϕt and their weights, one gather
            ϕt = s.ϕ[k][rows]
            idx = SVector(k, ip[I + 1, Jt + 1], ip[I + 2, Jt + 1], ip[I + 1, Jt + 2], ip[I + 2, Jt + 2],
                          i_f, i_r, i_m, i_v, i_l)
            push!(blocks, TileT(k, pix, ϕt, _select_rows(_view_weights(s.weights, k), rows), (I, Jt), idx, s, Ref{Any}(nothing), free, held_rotation_coordinates))
            n_tiles += 1
        end
    end
    dependent_patches = free === nothing ? Int[] :
                     [ip[PI, PJ] for PI in 1:nPx, PJ in 1:nPy if !any(free[((PI - 1) * px_n + 1):(PI * px_n), ((PJ - 1) * py_n + 1):(PJ * py_n)])]
    idx = (; poses = 1:K, patches = ip, focus = i_f, aperture = i_r, material = i_m,
           vignetting = i_v, lighting = i_l, n_tiles, patch = (px_n, py_n), dependent_patches)
    free_vars = trues(length(variables))
    free_vars[dependent_patches] .= false                # a patch block without a present node is no variable of the solve
    _free_mask!(free_vars, idx, Tuple(k for k in blocks_held if !(k in SURFACE_HOLDS)); lens_key = :grid, lens_idx = vec(ip))
    return VgtkCore.LmProblem(variables, blocks; fixed = findall(!, free_vars), damping = LM_DAMPING), (; idx..., free = free_vars)
end

"""No residual reads an absent node: every sensor sample of pixel `(u, v)` (view `k`) has its whole 4 × 4 window of present nodes
(Core's SplineLens backprojection is finite there), else an `ArgumentError` naming the pixel — a residual there would read controls the lens does not hold.
Under `grid_present`'s footprint rule it holds for every pixel of the imaged region."""
function _check_reads_present(lens, q::SensorQuadrature, u::Integer, v::Integer, k::Integer)
    m = lens
    for (𝐩, _) in sensor_samples(q, u, v)
        all(isfinite, backproject(m, 𝐩)) || throw(ArgumentError(
            "pixel ($u, $v) of view $k: its sensor sample $(𝐩) reads an absent node of the grid lens (outside the nodes it holds, " *
            "`grid_present`); the run's pixels lie in its imaged region, which every held node's window covers"))
    end
    return nothing
end

"""Full grid lens of a trial variable vector: the patch blocks' directions."""
function _assemble_lens(lens0::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}, vec, patches)
    return VgtkCore.with_directions(lens0, _assembled(directions(VgtkCore.model(lens0)), vec, patches))
end
"""The `[gy, gx]` controls `G0` with the patch blocks' direction controls (`[g_x, g_y]` per patch) written in."""
function _assembled(G0, vec, patches)
    G = copy(G0)
    for PJ in axes(patches, 2), PI in axes(patches, 1)
        d = _patch_dirs(vec[patches[PI, PJ]]::_Patch)
        px_n, py_n = size(d)
        for b in 1:py_n, a in 1:px_n
            v = d[a, b]; G[(PJ - 1) * py_n + b, (PI - 1) * px_n + a] = eltype(G)(v[1], v[2], v[3])
        end
    end
    return G
end

"""The camera (with the re-assembled grid lens), the poses, the material, the surface and the lighting from a grid-problem variable
vector (as `extract_variables`)."""
function extract_variables_grid(s::CalibrationState, vars::Vector, idx)
    lens0 = s.camera.geometric::GridLike
    ap = vars[idx.aperture]::ApertureVar
    lens_new = _assemble_lens(lens0, vars, idx.patches)
    mv = vars[idx.material]::MaterialVar
    camera = RadiometricCameraModel(lens_new, VgtkCore.model(vars[idx.focus]::_FocusBlock), ap.r, VgtkCore.model(vars[idx.vignetting]::VgtkCore.LinearSubspaceManifold))
    poses = [VgtkCore.model(vars[k]::_PoseBlock) for k in idx.poses]
    return camera, poses, mv.mat, mv.surface, (vars[idx.lighting]::LightingVar).light
end

"""
    run_lm_grid!(s; patch = GRID_PATCH, maxiters = 10, kwargs...) -> (; code, rule, iterations, cost)

Joint LM over the per-tile grid problem (all blocks; `fixed` as in `run_lm!` with `:grid` for the patch blocks in place of
`:distortion`). `height_degree` and `smoothness` as in `build_problem_grid`, the problem and its held blocks
its; `iter_callback(it, cost, camera, poses, mat, surface, light)` sees the current best variables after every iteration, and `lm_log` as `run_lm!`'s. The solve
and its stopping tests and termination record as `run_lm!`'s (VisualGeometryToolkitCore's `lmsolve_chart!` under `_lm_stops`).
"""
function run_lm_grid!(s::CalibrationState; patch::Tuple{Int, Int} = GRID_PATCH, maxiters::Integer = 10,
                      maxtime::Real = 7200.0, fixed = (),
                      reldcost = LMOptions().ftol, xtol_rel = LMOptions().xtol_rel, xtol = CORE_LM_XTOL, verbose::Bool = false,
                      trace = nothing, trace_meta = (;), iter_callback = nothing,
                      height_degree::Integer = 0, smoothness = nothing, inplane::Bool = false, lm_log::AbstractString = "")
    problem, idx = build_problem_grid(s; patch, fixed, height_degree, smoothness, inplane)
    callback = _final_iteration_callback(_lm_callback(s, idx, extract_variables_grid; trace, trace_meta, iter_callback, verbose, lm_log), s, maxiters)
    _reset_scrubbed!(s.run)
    result = try                                                 # (Core's stall postcondition runs inside, the tiles still batched)
        _with_registered(() -> VgtkCore.lmsolve_chart!(problem, _lm_stops(s, maxiters, maxtime, reldcost, xtol_rel, xtol); callback), s, problem, idx)
    finally
        s.run.final_iteration = false
        _report_scrubbed(s.run, "run_lm_grid!")
    end
    read_back!(s, problem, idx, extract_variables_grid)
    return result
end

# ---------------------------------------------------------------------------
# The per-tile blocks on every backend: the kernels of device/jacobian_kernels.jl (`_batch_launch`) over the view's pixels (the view's batch, below) or,
# without one, over the tile's own pixels under its window lens (`_tile_batch`); the chief directions are seeded in-kernel (`dbase`,
# 2·NS slots) and fanned out to the patch tangent columns on the host with the bicubic synthesis weights — the same split as the focus
# chain rule; residuals come from the plain flux kernel (bitwise identical to computeresidual).
# ---------------------------------------------------------------------------

"""Lazy per-tile device data: the pixel-id list `px` and its index list `idx` (the tile's own batch, `_tile_batch`), the tile's view
pixels `pix` (its rows among a view's batch) and its measured values `ϕ`."""
function _tile_dev!(r::PerTileResidual)
    d = r.dev[]
    d === nothing || return d
    s = r.state
    ids = Vector{UInt32}(undef, length(r.pix))
    for (j, i) in enumerate(r.pix)
        u, v = s.px[r.k][i]
        ids[j] = UInt32((u - 1) * _image_height(s) + v)
    end
    sh(a) = _shared_read_only!(s.backend, a)                  # read on the window's queues too, written only here
    d = (; px = sh(adapt(s.backend, ids)), idx = sh(_index_list(s.backend, length(ids))), pix = sh(adapt(s.backend, Int32.(r.pix))),
          ϕ = sh(adapt(s.backend, r.ϕt)), ω = _device_weights(s.backend, r.ωt), layouts = Dict{Any, Any}())
    r.dev[] = d
    return d
end


"""The tile's own batch (`_batch_launch` over its pixels under its window lens): a tile whose view has no registered batch (a grid
problem not registered, `_register_grid_problem!`, or variables it cannot locate) — the same kernels as the view's batch, so the
same values bit for bit."""
_tile_batch(r::PerTileResidual, vars, need_jac::Bool) =
    _await(_batch_launch(r.state, r.k, vars, _window_lens(r.state.camera.geometric, r.P0, vars[2], vars[3], vars[4], vars[5]), vars[2],
                         _tile_dev!(r).px, _tile_dev!(r).idx, nothing, need_jac))

"""
    _tile_jacobian_at(varflags, r, vars...) -> (; J, res, T, flux, ivs)

A tile's device Jacobian at its variables (`_tile_jacobian`) from its view's batch (`_view_batch!`), or from the tile's own batch
(`_tile_batch`) when the view has none; `flux` the batch's host flux and `ivs` the tile's pixels in it (`_tile_residual`).
"""
function _tile_jacobian_at(varflags, r::PerTileResidual{N}, pose::_PoseBlock, p00::_Patch{PX, PY},
                           p10::_Patch, p01::_Patch, p11::_Patch, foc::_FocusBlock,
                           ap::ApertureVar, m::MaterialVar, vg::VgtkCore.LinearSubspaceManifold, lt::LightingVar) where {N, PX, PY}
    s = r.state
    _count_device_call!(s, :jac)
    NV = Int(n_control_points(VgtkCore.model(vg))); J = Int(sh_count(max_degree(lt.light)))
    NS = length(s.settings.sensor)
    lay = _tile_layout(varflags, N, NS, PX * PY, VgtkCore.tangent_size(vg), NV, J, n_height(m), n_inplane(m); r.free, r.held_rotation_coordinates,
                       fixed = material_fixed(m))
    # every varflags goes through the batched path (2026-10-24): `lay` already omits the fixed blocks' columns and the per-view
    # buffers hold every slot — formerly only the all-free case 1023 was batched, so a run with a fixed block (the paper's fixed
    # aperture) dropped every tile onto the far slower per-tile path
    vars = (pose, p00, p10, p01, p11, foc, ap, m, vg, lt)
    ent = _view_batch!(s, r, vars, true)
    ivs = ent === nothing ? (1:length(r.pix)) : r.pix
    ent === nothing && (ent = _tile_batch(r, vars, true))
    return (; _tile_jacobian(r, ent, lay, vars...; ivs)..., ent.flux, ivs)
end

"""The per-tile `computeresjac` as host arrays: ONE implementation on every backend (the pixel functors of device/jacobian_kernels.jl on the state's
backend, Float64 on the CPU; the view's batch when the grid problem is registered, `run_lm_grid!`) — the residual of the flux kernel
(bitwise `computeresidual`'s) and the tile's Jacobian in the problem's coordinates (`_tile_jacobian_at`, `_host_jacobian`), for the
tests, the stall postcondition and the grid smoothness's weight selection; the LM's iterations take the Gauss–Newton blocks on the device
(`computecostgradhess`)."""
function _block_jacobian(varflags, r::PerTileResidual, vars...)
    t = _tile_jacobian_at(varflags, r, vars...)
    return _tile_residual(t.flux, r, t.ivs), _host_jacobian(t.J, t.T, t.lay, r.state.run)
end
# ---------------------------------------------------------------------------
# Per-view batched device evaluation (memoized). A tile only receives its own 2 × 2 patch window,
# but all tiles of a view within one damped LM solve share the same variable OBJECTS — so the first tile
# call locates the trial vector (problem.variables or .varnext, by identity), assembles the full
# trial grid, runs the batched per-view kernels ONCE, and caches the raw buffers keyed by the
# objectids of every variable the view depends on; the other tiles slice their rows. Kernels are
# per-pixel independent, so sliced results are bitwise identical to per-tile launches (the parity
# tests assert ==, except Julia 1.12 under --check-bounds=yes, which rounds the two launches'
# Float32 Jacobians differently: `device_parity` in test/unit/rc_testsetups.jl). Without a batch (unregistered, unlocatable vars) a
# tile runs the same kernels over its own pixels (`_tile_batch`); partial varflags (fixed blocks) are batched too (2026-10-24).
# ---------------------------------------------------------------------------

"""Enable batched per-view evaluation for a grid problem (call before optimizing; see run_lm_grid!). `next[k]`: a tile of the view
whose tiles follow view `k`'s in the cost order (`nothing` after the last), the one the next view's batch is prefetched for;
`pending[k]`: view `k`'s batch in flight, `(; key, need_jac, fut)` (`_batch_future!`); `blocks[k]`: its tiles' Gauss–Newton batches
by varflags, `(; key, fut)` (`_tile_blocks_future!`), over `view_tiles[k]` with the problem's free variables `free_vars` (`idx.free`)
and the tiles' device layouts `layouts`; `lock`: held while they change."""
_register_grid_problem!(s::CalibrationState, p::VgtkCore.LmProblem, idx) = _register_grid_problem!(s, p.problem, idx)
function _register_grid_problem!(s::CalibrationState, problem, idx)
    dc = _ensure_device!(s)
    _unregister_grid_problem!(s)
    K = n_views(s)
    next = Vector{Any}(nothing, K)
    tiles = tile_costs(problem)
    for i in 2:length(tiles)
        k = tiles[i - 1].k
        tiles[i].k != k && next[k] === nothing && (next[k] = tiles[i])
    end
    s.device_cache = merge(s.device_cache, (; grid = (; problem, patches = idx.patches, lock = ReentrantLock(),
                      cache = [Ref{Any}(nothing) for _ in 1:K], pending = Vector{Any}(nothing, K), next,
                      # the tiles' Gauss–Newton blocks a view at a time (`_tile_blocks!`): the problem's free variables (`nothing`: not
                      # known, no batch), each view's tiles in the cost order, per view its batches in flight or done by varflags
                      free_vars = hasproperty(idx, :free) ? BitVector(idx.free) : nothing, view_tiles = [[t for t in tiles if t.k == k] for k in 1:K],
                      blocks = [Dict{Int, Any}() for _ in 1:K], layouts = Dict{Any, Any}())))
    return nothing
end
function _unregister_grid_problem!(s::CalibrationState)
    dc = s.device_cache
    dc === nothing && return nothing
    gb = get(dc, :grid, nothing)
    if gb !== nothing                                     # a batch still in flight: waited for (its worker back to the pool), dropped
        for d in gb.blocks                                # (the tiles' batches first: each waits for its view's batch)
            for e in values(d)
                try wait(e.fut) catch end
            end
            empty!(d)
        end
        for k in eachindex(gb.pending)
            p = gb.pending[k]
            p === nothing || try wait(p.fut) catch end
            gb.pending[k] = nothing
        end
    end
    s.device_cache = Base.structdiff(s.device_cache, NamedTuple{(:grid,)})   # the scene, the pixel ids (and the heights' scene) kept
    return nothing
end
_grid_batch(s::CalibrationState) = _device_entry(s, :grid)

"""The variable vector (variables or varnext) that holds exactly the objects given to the cost `r` (a tile's or a view's)."""
function _locate_vars(problem, r, given)
    for vec in (problem.variables, problem.varnext)
        length(vec) >= maximum(r.idx) || continue
        ok = true
        for i in eachindex(r.idx)
            @inbounds vec[r.idx[i]] === given[i] || (ok = false; break)
        end
        ok && return vec
    end
    return nothing
end

"""The cache key of a view's batch at the variables `vars` (a tile's) of the trial vector `vec`: the objectids of every variable the
view depends on."""
function _view_key(gb, vec, vars)
    nP = length(gb.patches)
    key = Vector{UInt64}(undef, 6 + nP)
    key[1] = objectid(vars[1]); key[2] = objectid(vars[6]); key[3] = objectid(vars[7])
    key[4] = objectid(vars[8]); key[5] = objectid(vars[9]); key[6] = objectid(vars[10])
    for n in 1:nP
        key[6 + n] = objectid(vec[gb.patches[n]])
    end
    return key
end

"""
The memoized view-level device buffers for the trial the tile's variables belong to (nothing: unbatched path).

A view's batch is one job on a device worker (`_view_batch_launch`): every kernel of it queued on that worker's device queue at once,
one wait, one copy each. Its result is a shared future (`_batch_future!`): every tile of the view at that trial — the requests and the
window's jobs (`_windowed`) — waits for the one batch, outside the grid's lock, so the batches of several views run at once. A
Jacobian batch of view `k` also starts the next view's (`gb.next[k]`, at the same trial vector), so the device runs it while view
`k`'s tiles are formed — the same kernels on the same inputs, so the same buffers bit for bit. A batch in flight whose key or need is
not the request's is dropped (its future still completes); a failed batch's exception reaches every tile that waits for it.
"""
function _view_batch!(s::CalibrationState, r::PerTileResidual{N}, vars, need_jac::Bool) where {N}
    gb = _grid_batch(s)
    gb === nothing && return nothing
    vec = _locate_vars(gb.problem, r, vars)
    vec === nothing && return nothing
    key = _view_key(gb, vec, vars)
    fut = lock(() -> _batch_future!(s, gb, r.k, vars, vec, key, need_jac), gb.lock)
    ent = fut isa Task ? _fetch_batch(fut) : fut
    lock(gb.lock) do
        p = gb.pending[r.k]
        if p !== nothing && p.fut === fut                  # the batch done: the view's entry, read by its other tiles
            gb.pending[r.k] = nothing
            gb.cache[r.k][] = ent
        end
        need_jac && _prefetch_next!(s, gb, r.k, vec)
    end
    return ent
end
"""The batch of a shared future (`_batch_future!`), its job's exception rethrown as the job threw it."""
_fetch_batch(fut::Task) = try fetch(fut) catch e; e isa TaskFailedException ? throw(e.task.exception) : rethrow() end
"""
    _batch_future!(s, gb, k, vars, vec, key, need_jac) -> entry | Task

Under the grid's lock: view `k`'s cached entry for `key` (`need_jac`: one with the Jacobian's buffers), else the future of its batch
in flight for that key, else a new batch (`_view_batch_launch`) and its future, recorded as in flight (`gb.pending`).
"""
function _batch_future!(s::CalibrationState, gb, k::Integer, vars, vec, key, need_jac::Bool)
    ent = gb.cache[k][]
    ent !== nothing && ent.key == key && (!need_jac || ent.S !== nothing) && return ent
    p = gb.pending[k]
    p !== nothing && p.key == key && (!need_jac || p.need_jac) && return p.fut
    job = _view_batch_launch(s, k, vars, vec, key, need_jac)
    fut = Threads.@spawn _await(job)                      # (a stale batch in flight is replaced; its future still completes)
    gb.pending[k] = (; key, need_jac, fut)
    return fut
end

"""Under the grid's lock: start the Jacobian batch of the view after `k` (`gb.next[k]`) at the trial vector `vec`, unless it is cached
or in flight already (a residual sweep has no host assembly to overlap: none there)."""
function _prefetch_next!(s::CalibrationState, gb, k::Integer, vec)
    rn = gb.next[k]
    rn === nothing && return nothing
    length(vec) >= maximum(rn.idx) || return nothing
    varsn = ntuple(i -> vec[rn.idx[i]], 10)
    _batch_future!(s, gb, rn.k, varsn, vec, _view_key(gb, vec, varsn), true)
    return nothing
end

"""View `k`'s batched device evaluation at the variables `vars` (a tile's, of the trial vector `vec`) as one worker job (`_await` it):
the flux and, for `need_jac`, the radiance record, the Jacobian chunks, the parts and the height partials,
all queued on the worker's device queue before it waits once — `(; key, flux, flux_d, px, C, S, parts, heights)` (on the device but the host flux), the view
cache's entry. One queue: the kernels share the scene's buffers, and a buffer used on another queue waits for the queue that used it
last, so separate queues bought little concurrency and made every launch wait for a kernel to end."""
function _view_batch_launch(s::CalibrationState, k::Integer, vars, vec, key, need_jac::Bool)
    gb = _grid_batch(s)
    lensf = _assemble_lens(s.camera.geometric::GridLike, vec, gb.patches)
    return _batch_launch(s, k, vars, lensf, vec[first(gb.patches)], s.device_cache.px[k], s.device_cache.idx[k], key, need_jac)
end
"""One chunk of a view batch's Jacobian (`_batch_launch`) in the float type `F`: the chunk's slots `c0 + 1 … c0 + C` seeded on the
lens `lens` and the board `board` (its state replaced by the seeded one), both in `F` (`_kernel_float`, or
`HIGH_FLOAT` for the pose's chunks: `hi`). Returns the `_JacAt` inputs `(view, mat, board, fbase, dbase, hi)`."""
function _tile_chunk(::Val{F}, c0, C, lay, pose, foc, ap, m, lens, board, hi::Bool) where {F}
    N = length(VgtkCore.model(foc).t̂_i)
    D, cd, pose_d, focus_d, mat_d, board_d = _seed_chunk(Val(F), c0, C, lay, pose, foc, m, board, lay.NI)
    view_d = View{D, N, typeof(lens)}(pose_d, lens, focus_d, cd(ap.r, lay.off_r + 1))
    return (view_d, mat_d, board_d, Int32(lay.off_f - c0), Int32(lay.off_dir - c0), hi)
end

"""
    _batch_launch(s, k, vars, lensf, p1, px, idx, key, need_jac) -> job

THE evaluation of the per-tile blocks' kernels: the flux and, for `need_jac`, the radiance record, the Jacobian chunks, the parts, the
height partials and the printed-edge differences of the pixel ids `px` (`idx` their index list, `_index_list`) of view `k` (all of the view's: `_view_batch_launch`; one
tile's: `_tile_batch`) under the lens `lensf` (the view's assembled grid, or the tile's window) at the variables `vars`; `p1` one of
its patch blocks (the slot layout's patch size).
"""
function _batch_launch(s::CalibrationState{N}, k::Integer, vars, lensf, p1, px, idx, key, need_jac::Bool) where {N}
    pose = vars[1]; foc = vars[6]::_FocusBlock; ap = vars[7]::ApertureVar
    m = vars[8]::MaterialVar; vg = vars[9]::VgtkCore.LinearSubspaceManifold; lt = vars[10]::LightingVar
    dc = _ensure_device!(s)
    be = s.backend
    F = _kernel_float(be)
    lens32 = _lens32(be, lensf)
    v32 = View{F, N, typeof(lens32)}(_pose_eltype(F, VgtkCore.model(pose)), lens32, VgtkCore.convert_eltype(F, VgtkCore.model(foc)),
                                     F(ap.r))
    H = _image_height(s)
    fluxf = _flux_functor(s, k, v32, VgtkCore.model(vg), m.mat, lt.light, m.surface)
    recordf = nothing; hrecordf = nothing; jacf = Tuple{Int, Any}[]; heightf = nothing; partsf = nothing
    hk = nothing
    C = s.jacobian_chunk
    P = 0
    if need_jac
        Tag = JacTag
        NV = Int(n_control_points(VgtkCore.model(vg))); J = Int(sh_count(max_degree(lt.light)))
        NS = length(s.settings.sensor)
        # the same slot arithmetic as the per-tile Jacobian (all blocks free)
        lay = _tile_layout(1023, N, NS, length(_patch_dirs(p1)), VgtkCore.tangent_size(vg), NV, J, n_height(m), n_inplane(m);
                           fixed = material_fixed(m))
        P = lay.P
        st32 = VgtkCore.convert_eltype(F, s.settings)
        C = _jac_chunk(s, P)
        # vignetting and lighting are not differentiated here (their columns are the parts kernel's): plain, not duals with
        # zero partials — the same values and partials, fewer registers
        vig32j = VgtkCore.convert_eltype(F, VgtkCore.model(vg)); l32j = VgtkCore.convert_eltype(F, lt.light)
        m32 = VgtkCore.convert_eltype(F, m.mat)
        board32 = with_state(dc.board, VgtkCore.convert_eltype(F, m.surface))
        recordf = (be, px, idx, k, H, v32, vig32j, m32, l32j, board32, st32)   # the estimator once per ray (`_launch_record`)
        if _needs_high(F, lay.NH > 0)                      # the nodal heights' kernel and its own record, in Float64 (`HIGH_FLOAT`)
            hk = _height_inputs(s, View(VgtkCore.model(pose), lensf, VgtkCore.model(foc), ap.r), VgtkCore.model(vg), m, lt.light, lensf)
            HIGH_FLOAT === F || (hrecordf = (be, px, idx, k, H, hk.view, hk.vig, hk.mat, hk.light, hk.board, hk.st))   # (CPU: the chunks' record)
            lay.NH > 0 && (heightf = (k, H, hk.view, hk.vig, hk.mat, hk.light, hk.board, hk.st, m.surface.h))
        end
        plan = _chunk_plan(_chunk_starts(P, C), F, (lens32, dc.board),
                           hk === nothing ? nothing : (hk.view.geometric, hk.board))
        for (c0, Fc, inputs, high) in plan
            push!(jacf, (c0, _tile_chunk(Val(Fc), c0, C, lay, pose, foc, ap, m, inputs..., high)))
        end
        partsf = (Int32(k), Int32(H), v32, vig32j, m32, l32j, board32, st32, (Val(J), Val(NV)))   # (`_PartsAt`, with the batch's record)
    end
    _count_device_call!(s, :flux)
    st32 = VgtkCore.convert_eltype(F, s.settings)
    vig32j = VgtkCore.convert_eltype(F, VgtkCore.model(vg)); l32j = VgtkCore.convert_eltype(F, lt.light)
    return _on_worker() do
        fo = _launch_rows(fluxf, be, px)
        # the estimator once per ray: in `HIGH_FLOAT` where a chunk needs it, that record rounded for the others (`_record_as`)
        hrays = hrecordf === nothing ? nothing : _launch_record(hrecordf...)
        rays = recordf === nothing ? nothing : hrays === nothing ? _launch_record(recordf...) : _record_as(F, hrays)
        hrays === nothing && (hrays = rays)
        ho = heightf === nothing ? nothing : _launch_rows(_height_functor(heightf..., px, hrays), be, idx; workgroupsize = 128)
        co = [(c0, hi ? _launch_rows(_JacAt(Int32(k), Int32(H), vd, hk.vig, md, hk.light, sd, hk.st, Val(C), fb, db, px, hrays), be, idx; workgroupsize = 128) :
                        _launch_rows(_JacAt(Int32(k), Int32(H), vd, vig32j, md, l32j, sd, st32, Val(C), fb, db, px, rays), be, idx; workgroupsize = 128))
              for (c0, (vd, md, sd, fb, db, hi)) in jacf]
        po = partsf === nothing ? nothing : _launch_rows(_PartsAt(partsf..., px, rays), be, idx)
        KA.synchronize(be)
        flux = Float64.(Array(fo))                         # the residual's values (a channel triple per pixel); the Jacobian's stay on the device
        S = need_jac ? _slot_rows(be, [(c0, o, C) for (c0, o) in co], P, length(px)) : nothing
        KA.synchronize(be)
        # the batch's outputs are read by its tiles' row kernels on other queues (the main task's, the window's workers) and never
        # written again: out of the per-buffer queue tracking (`_shared_read_only!`), as the view's pixel lists
        foreach(a -> a === nothing || _shared_read_only!(be, a), (fo, S, po, ho))
        bytes = _device_bytes(fo, rays, hrays === rays ? nothing : hrays, ho, po, S) + sum(x -> sizeof(x[2]), co; init = 0)
        (; key, flux, flux_d = fo, px, C, S, parts = po, heights = ho, bytes)
    end
end


"""
    _tile_jacobian(r, ent, lay, vars...; ivs = r.pix) -> (; J, res, T, lay)

One tile's Jacobian on the state's backend from the view-level device buffers `ent` (`_batch_launch`; the tile's pixels at the
buffer's pixels `ivs`): the row kernel (`_jacobian_row!`) over the tile's pixels — the AD scatter, the focus chain, the patch fan-out
on the tile's window lens, the parts' and the heights' columns — `J` (the layout's full
columns × the measured rows) and the residual `res` on the backend, the column map `T` to the problem's coordinates
(`_column_map`) and the layout (its device form cached, keyed by the layout itself, in the tile's device data). The non-finite
entries are scrubbed where the blocks are formed (`_gauss_newton_blocks`, `_host_jacobian`). The same kernels' values and the same
per-entry arithmetic as the host assembly it replaced.
"""
function _tile_jacobian(r::PerTileResidual{N}, ent, lay, pose::_PoseBlock, p00::_Patch{PX, PY},
                        p10::_Patch, p01::_Patch, p11::_Patch, foc::_FocusBlock,
                        ap::ApertureVar, m::MaterialVar, vg::VgtkCore.LinearSubspaceManifold,
                        lt::LightingVar; ivs = r.pix) where {N, PX, PY}
    s = r.state
    be = s.backend
    NV = Int(n_control_points(VgtkCore.model(vg))); J = Int(sh_count(max_degree(lt.light)))
    npx = length(r.pix)
    need_v = !isempty(lay.vig_cols)
    need_lin = need_v || !isempty(lay.light_cols) || lay.μ_col > 0
    need_f = !isempty(lay.focus_cols)
    need_g = any(q -> lay.patch_cols[q] > 0, 1:4)
    need_h = !isempty(lay.height_cols)
    lens64 = _window_lens(s.camera.geometric, r.P0, p00, p10, p01, p11)
    td = _tile_dev!(r)
    bufidx = ivs === r.pix ? td.pix : td.idx                # the tile's pixels among the buffer's: the view's (batched) or its own
    Gw, gidx = need_f ? _focus_weights(s, r.k, FocusSpline(VgtkCore.model(foc)), td.pix) : (nothing, nothing)
    chs = fit_channels(s)
    Jd, rd = _jacobian_buffers(s, lay.ncol, length(chs) * npx)
    VgtkCore._launch_rows_on!(_pixel_rows_exec(be), _jacobian_row!, Jd, npx, rd, ent.px, bufidx, gidx, Int32(_image_height(s)), ent.S, ent.flux_d, td.ϕ, td.ω, chs,
                           _cached_layout(td.layouts, be, lay), need_lin ? ent.parts : nothing, (J, NV), need_h ? ent.heights : nothing,
                           need_h ? _device_copy(s, m.dense_basis) : nothing, Gw, Val(N),
                           s.settings.sensor, need_g ? _lens32(be, lens64, Float64) : nothing,
                           need_g ? adapt(be, _window_bases(lens64, PX, PY)) : nothing, r.P0, need_g ? (Val(PX), Val(PY)) : nothing,
                           lt.scale,
                           need_v ? _device_copy(s, vg.basis) : nothing)
    _note_block_bytes!(s, _device_bytes(Jd, rd) + ent.bytes * npx ÷ max(length(ent.px), 1))   # its share of the view batch
    return (; J = Jd, res = rd, T = _column_map(lay, s.channel, ap, m), lay)
end

"""The residual of tile `r` from a view's flux `flux` (three channels per pixel, the kernels'; the tile's pixels at the view pixels
`ivs`): the measured channels' entries (`fit_channels`) minus the tile's observed `ϕt`, one per measured channel and pixel, under
the rows' weights `ωt` (`_whiten`; the row kernel's product)."""
function _tile_residual(flux::AbstractVector, r::PerTileResidual, ivs)
    chs = fit_channels(r.state); nc = length(chs)
    out = Vector{Float64}(undef, nc * length(r.pix))
    @inbounds for (i, iv) in enumerate(ivs), (j, c) in enumerate(chs)
        e = nc * (i - 1) + j
        out[e] = _whiten(r.ωt, e, flux[3 * (iv - 1) + c] - r.ϕt[e])
    end
    return out
end

# ---------------------------------------------------------------------------
# A view's tiles' Gauss–Newton blocks in one batch (decision 6). NLLSsolver asks for the tiles' blocks one at a time, and each
# request launched its row kernel over a few hundred pixels, uploaded its window lens and tangent bases and formed `J r`, `J Jᵀ` with a
# synchronisation each: per-request latency, a view's 108 tiles 0.267 s on a B200 (65 × 50 lattice). The first request of a view's
# tiles at a trial forms all of them: ONE row-kernel launch over the view's tiles' pixels (`_tiles_row!`: per row its tile's window
# lens, bases and patch origin), the products as one batch (`_batched_products`), the problem's coordinates on the host — memoized
# like the view's batch (`_view_batch!`), keyed by the trial's variable objects and the varflags, and each tile's blocks handed out
# once. The rows and the residual are the per-tile path's bit for bit (the same row kernel at the same inputs); the products are the
# backend's over the zero-padded rows (on the CPU backend the per-tile products themselves).
# ---------------------------------------------------------------------------

"""The varflags NLLSsolver gives the tile `t` when the problem's free variables are `free` (`computevarflags`: bit `j − 1` set for a
free `j`-th variable of the tile's ten)."""
_tile_flags(free::AbstractVector{Bool}, t::PerTileResidual) = sum(j -> free[t.idx[j]] ? 1 << (j - 1) : 0, 1:10)

"""
    _tile_blocks!(s, r, vars, vf) -> (Jᵀr, JᵀJ, flux) | nothing

The tile `r`'s Gauss–Newton blocks at the varflags `vf` from its view's tiles' batch (`_tile_blocks_future!`), or `nothing` (the
per-tile path): no registered grid problem or free variables known, `vf` not the flags the problem gives the tile, variables it
cannot locate. A tile asked again at a trial whose blocks it took already starts its view's batch again. A view's tiles of
other varflags (a tile beside a dependent patch) are a batch of their own. Also starts the next view's batch at the trial (`gb.next`), so the device forms it while this view's are read.
"""
function _tile_blocks!(s::CalibrationState, r::PerTileResidual, vars, vf::Integer)
    gb = _grid_batch(s)
    (gb === nothing || gb.free_vars === nothing) && return nothing
    _tile_flags(gb.free_vars, r) == vf || return nothing
    vec = _locate_vars(gb.problem, r, vars)
    vec === nothing && return nothing
    key = _view_key(gb, vec, vars)
    stale = nothing
    for _ in 1:2                                             # (a second pass: the tile's blocks were handed out at this trial already)
        fut = lock(gb.lock) do
            f = _tile_blocks_future!(s, gb, r, vars, vec, key, Int(vf); stale)
            rn = gb.next[r.k]
            if rn !== nothing && length(vec) >= maximum(rn.idx)
                varsn = ntuple(i -> vec[rn.idx[i]], 10)
                _tile_blocks_future!(s, gb, rn, varsn, vec, _view_key(gb, vec, varsn), _tile_flags(gb.free_vars, rn))
            end
            f
        end
        blocks = _fetch_batch(fut)
        blocks === nothing && return nothing                 # (no view batch at these variables)
        b = lock(() -> pop!(blocks, r, nothing), gb.lock)
        b === nothing || return b
        stale = fut
    end
    return nothing
end
"""
    _tile_blocks_future!(s, gb, r, vars, vec, key, vf) -> Task

Under the grid's lock: the future of the batch of view `r.k`'s tiles of the varflags `vf` at the trial `key` (`_view_key`), a new one
(`_tiles_normal_equations` on a device worker, after the view's batch) when none is kept for that key or the kept one is `stale`
(its tile's blocks handed out already). Its value: an `IdDict` tile → `(Jᵀr, JᵀJ, flux)`, from which each tile's blocks are taken
once (`_tile_blocks!`), or `nothing` when the view has no batch at the variables (`_view_batch!`).
"""
function _tile_blocks_future!(s::CalibrationState, gb, r::PerTileResidual, vars, vec, key, vf::Int; stale = nothing)
    e = get(gb.blocks[r.k], vf, nothing)
    e !== nothing && e.key == key && e.fut !== stale && return e.fut
    tiles = [t for t in gb.view_tiles[r.k] if _tile_flags(gb.free_vars, t) == vf]
    tvars = [NLLSsolver.getvars(t, vec) for t in tiles]
    fut = Threads.@spawn begin
        ent = _view_batch!(s, r, vars, true)                 # the view's batch (memoized; it also starts the next view's)
        ent === nothing ? nothing : _await(_on_worker(() -> _tiles_normal_equations(s, gb, tiles, tvars, ent, vf)))
    end
    gb.blocks[r.k][vf] = (; key, fut)                        # (a batch at another trial is replaced; its future still completes)
    return fut
end

"""
    _tiles_normal_equations(s, gb, tiles, tvars, ent, vf) -> IdDict tile → (Jᵀr, JᵀJ, flux)

The Gauss–Newton blocks of a view's `tiles` (their variables `tvars`, all of the varflags `vf`) from the view's batch `ent`: the
rows of every tile in one launch of `_tiles_row!` (each tile's rows padded to the largest tile's), the products `J r`, `J Jᵀ` of every
tile at once (`_batched_products`), a tile with a non-finite entry scrubbed and formed again (`_finite_products`), and each tile's in
the problem's coordinates (`_in_problem_coordinates`; one column map: every tile of the batch has the same layout). `_tile_jacobian`'s
inputs, tile by tile.
"""
function _tiles_normal_equations(s::CalibrationState{N}, gb, tiles, tvars, ent, vf::Int) where {N}
    be = s.backend
    r1 = first(tiles)
    _, p1, _, _, _, foc, ap, m, vg, lt = first(tvars)
    PX, PY = size(_patch_dirs(p1))
    NV = Int(n_control_points(VgtkCore.model(vg))); J = Int(sh_count(max_degree(lt.light)))
    lay = _tile_layout(vf, N, length(s.settings.sensor), PX * PY, VgtkCore.tangent_size(vg), NV, J, n_height(m), n_inplane(m);
                       r1.free, r1.held_rotation_coordinates, fixed = material_fixed(m))
    need_v = !isempty(lay.vig_cols)
    need_lin = need_v || !isempty(lay.light_cols) || lay.μ_col > 0
    need_g = any(q -> lay.patch_cols[q] > 0, 1:4)
    need_h = !isempty(lay.height_cols)
    chs = fit_channels(s); NC = length(chs)
    nt = length(tiles); npx = [length(t.pix) for t in tiles]; Mx = maximum(npx)
    pix = ones(Int32, Mx * nt); ϕ = zeros(NC * Mx * nt)                     # per row, padded to Mx pixels a tile
    weighted = !(r1.ωt isa Unweighted); ω = weighted ? ones(NC * Mx * nt) : nothing
    for (q, t) in enumerate(tiles)
        o = (q - 1) * Mx
        pix[(o + 1):(o + npx[q])] .= t.pix
        ϕ[(NC * o + 1):(NC * (o + npx[q]))] .= t.ϕt
        weighted && (ω[(NC * o + 1):(NC * (o + npx[q]))] .= t.ωt)
    end
    pix_d = adapt(be, pix)
    Gw, gidx = isempty(lay.focus_cols) ? (nothing, nothing) : _focus_weights(s, r1.k, FocusSpline(VgtkCore.model(foc)), pix_d)
    windows = need_g ? [_window_lens(s.camera.geometric, t.P0, v[2], v[3], v[4], v[5]) for (t, v) in zip(tiles, tvars)] : nothing
    lenses = need_g ? [_lens32(be, w, Float64) for w in windows] : nothing
    bases = need_g ? adapt(be, stack(w -> _window_bases(w, PX, PY), windows)) : nothing
    Jd = KA.zeros(be, Float64, lay.ncol, NC * Mx * nt); rd = KA.zeros(be, Float64, NC * Mx * nt)
    GC.@preserve lenses begin
        VgtkCore._launch_rows_on!(_pixel_rows_exec(be), _tiles_row!, Jd, Mx * nt, Mx, adapt(be, Int32.(npx)),
                                  need_g ? _device_elements(be, lenses) : nothing, bases, adapt(be, [t.P0 for t in tiles]),
                                  rd, ent.px, pix_d, gidx, Int32(_image_height(s)), ent.S, ent.flux_d, adapt(be, ϕ),
                                  weighted ? _device_weights(be, ω) : Unweighted(), chs, _cached_layout(gb.layouts, be, lay),
                                  need_lin ? ent.parts : nothing, (J, NV), need_h ? ent.heights : nothing,
                                  need_h ? _device_copy(s, m.dense_basis) : nothing, Gw, Val(N), s.settings.sensor,
                                  need_g ? (Val(PX), Val(PY)) : nothing, lt.scale, need_v ? _device_copy(s, vg.basis) : nothing)
    end
    _note_block_bytes!(s, (_device_bytes(Jd, rd) + ent.bytes) ÷ nt)       # a tile's share of the batch (the window's memory bound)
    J3 = reshape(Jd, lay.ncol, NC * Mx, nt); r2 = reshape(rd, NC * Mx, nt)
    ms = NC .* npx
    products = _batched_products(J3, r2, ms)
    Tsp = SparseArrays.sparse(_column_map(lay, s.channel, ap, m))
    out = IdDict{Any, Any}()
    for (q, t) in enumerate(tiles)
        g, H = _finite_products(view(J3, :, 1:ms[q], q), view(r2, 1:ms[q], q), products[q], lay, s.run)
        out[t] = (_in_problem_coordinates(Tsp, g, H)..., ent.flux)
    end
    return out
end

# The final LM iteration's speculative linearization. NLLSsolver's LM computes the Jacobian at each iteration's first trial point
# (it becomes the next linearization when the step is accepted), so a run of n iterations evaluates n + 1 Jacobians and the last
# one is never used. `run_lm_grid!` marks its state (`s.run.final_iteration`, the state's run record) once the iteration cap's last
# iteration begins (`_final_iteration_callback`);
# its tiles then return their cost alone (zero blocks: the loop stops after that iteration whatever its outcome, and nothing reads
# that linear system). The mark is set only when the iterator already holds the next linearization (`trialready`: the previous
# trial was accepted), so the linearization the last iteration starts from is never one of them. A stop by another criterion
# (cost or step tolerance, time) cannot be foreseen and keeps the extra Jacobian.
"""`cb`, and after iteration `maxiters - 1` the state marked (`s.run.final_iteration`, unmarked by `run_lm_grid!` on exit)."""
function _final_iteration_callback(cb, s, maxiters::Integer)
    return function (cost, problem, data, iterdata)
        out = cb(cost, problem, data, iterdata)
        data.iternum + 1 >= maxiters && hasproperty(iterdata, :trialready) && iterdata.trialready && (s.run.final_iteration = true)
        return out
    end
end

# ---- the LM's data blocks, a view's (per_view.jl) and a tile's: NLLSsolver's three calls, each through the window (`_windowed`) ----------
"""The calibration LM's data blocks: a view's (`PerViewResidual`) and a tile's (`PerTileResidual`)."""
const _DataBlock = Union{PerViewResidual, PerTileResidual}

"""Residual of a view or a tile (`_block_residual`)."""
NLLSsolver.computeresidual(r::_DataBlock, vars...) = _windowed((b, v) -> _block_residual(b, v...), r, vars, :residual)

"""
    computeresjac(varflags, r, vars...) -> (𝔯, J)

Residual and Jacobian of a view or a tile on the host (`_block_jacobian`), for the tests, the stall postcondition and the diagnostics.
"""
NLLSsolver.computeresjac(varflags, r::_DataBlock, vars...) =
    _windowed((b, v) -> _block_jacobian(varflags, b, v...), r, vars, (:jacobian, Int(varflags)))

"""
    computecostgradhess(varflags, r, vars...) -> (cost, Jᵀr, JᵀJ)

The LM's Gauss–Newton blocks of a view or a tile, formed on the state's backend (`_block_normal_equations`: `_gauss_newton_blocks`):
the Jacobian never reaches the host. In a grid fit's final iteration (`s.run.final_iteration`) the cost alone. A robust kernel takes
NLLSsolver's default.
"""
function NLLSsolver.computecostgradhess(varflags, r::_DataBlock, vars...)
    if r.state.run.final_iteration                       # the final iteration's trial linearization is never used: the cost only
        n = sum(i -> (Int(varflags) >> (i - 1)) & 1 == 1 ? Int(NLLSsolver.nvars(vars[i])) : 0, eachindex(vars))
        return NLLSsolver.computecost(r, vars...), zeros(n), zeros(n, n)
    end
    NLLSsolver.robustkernel(r) isa NLLSsolver.NoRobust ||
        return invoke(NLLSsolver.computecostgradhess, Tuple{Any, NLLSsolver.AbstractResidual, Vararg{Any}}, varflags, r, vars...)
    return _windowed((b, v) -> _block_normal_equations(varflags, b, v), r, vars, (:blocks, Int(varflags)))
end
"""A tile's `(cost, Jᵀr, JᵀJ)` (`computecostgradhess`): the cost of its batch's flux (`_block_residual`'s, the batch memoized); the
blocks from its view's tiles' batch (`_tile_blocks!`), else its own (`_tile_jacobian_at`, `_gauss_newton_blocks`)."""
function _block_normal_equations(varflags, r::PerTileResidual, vars)
    b = _tile_blocks!(r.state, r, vars, Int(varflags))
    b === nothing || return 0.5 * Float64(NLLSsolver.sqnorm(_tile_residual(b[3], r, r.pix))), b[1], b[2]
    t = _tile_jacobian_at(varflags, r, vars...)
    g, H = _gauss_newton_blocks(t.J, t.res, t.T, t.lay, r.state.run)
    return 0.5 * Float64(NLLSsolver.sqnorm(_tile_residual(t.flux, r, t.ivs))), g, H
end

"""All per-tile cost blocks of a grid problem (whatever their patch / aperture block types)."""
tile_costs(problem::VgtkCore.LmProblem) = tile_costs(problem.problem)
function tile_costs(problem)
    ts = [T for T in keys(problem.costs) if T <: PerTileResidual]
    isempty(ts) && return PerTileResidual[]
    return get(problem.costs, only(ts))
end
