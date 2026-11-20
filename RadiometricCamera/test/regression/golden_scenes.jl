# Golden scenes: the same small problems built with the package BEFORE the second migration (branch board-space, the "old
# stack") and with the current package. `record_golden.jl` writes their outputs from the old stack to golden_board_space.json;
# test/regression/golden_tests.jl recomputes them with the current package and compares. Only the API differs between the two (the board warping
# and the printed edge were Material fields before, the scene's TargetState now; the blob layout / tone / warp / gains / board space are
# gone) — `OLD_STACK` selects the construction, the quantities are the same.
#
# Recorded per scene: Φ_pixel on fixed pixels; the residual and the Jacobian (per column: Σ, Σ|·|, Σ², and the entries of 4 fixed rows) of
# a per-view problem (Kannala–Brandt) and of the largest tile of per-tile problems (grid model with the warping free; a distance-map
# board with the printed-edge block); the closed-form joint radiometry; three LM iterations with explicit
# stopping tests (so the defaults' change does not enter).
using RadiometricCamera, StaticArrays, LinearAlgebra
using RadiometricCamera: all_parts, aperture_radius, TargetState, build_problem, build_problem_grid, compute_mask, cost, focus,
                         Focus, FocusSpline, height_basis, HEIGHT_NFREE, geometric, material, Material, PerViewResidual,
                         TexturedTarget, render_all, RenderSettings, residuals, run_lm!, solve_radiometry_gn!, sphere_radius, t̂,
                         View, with_sdf, with_μ, Φ_pixel, 𝔟
const RCg = RadiometricCamera
import VisualGeometryToolkitCore as VgtkCore
using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, control_points, EnvironmentLight, SphericalHarmonicMap,
                                   coefficients, with_coefficients, sh_count, max_degree
include(joinpath(@__DIR__, "..", "unit", "fixture_camera.jl"))   # cam: Core's pixel camera of a test image
const GOLDEN_SCHEMA = "rc/golden_outputs/v1.0"   # the golden files' record schema (IO 0.10's `write_record` states every file's)
if OLD_STACK
    const exp_so3 = RCg.expSO3                          # the old stack's own Rodrigues map (the package's, before Core's `exp_so3` replaced it)
else
    using VisualGeometryToolkitCore: exp_so3
end

const GOLD_TEX = RCg.VgtkFeatures.load_linear_image(joinpath(@__DIR__, "..", "fixtures", "blobboards_256", "blob_pattern_061_ce03e.png"))
gold_h() = SVector{RCg.HEIGHT_NC}([saddle(𝐛) for 𝐛 in RCg.HEIGHT_NODES])
gold_mat() = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6)

"""Board, scene (with the warping h and the edge parameters as the stack wants them) and the material."""
function gold_board(W, H; h = nothing, edge = nothing, sdf = nothing)
    board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], sdf === nothing ? GOLD_TEX : ones(16, 16, 3); facing = [0.0, 0.0, 1.5])
    sdf === nothing || (board = with_sdf(board, sdf...))
    m = gold_mat()
    if OLD_STACK
        h === nothing || (m = RCg.with_height(m, h))
        edge === nothing || (m = RCg.with_edge(m, edge))
        return board, m
    end
    return with_state(board, TargetState(; h, edge)), m
end
gold_poses() = [RCg.VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.2 * cos(k), 0.2 * sin(k), 1.2 + 0.1 * k)); check = false) for k in 1:2]
const GOLD_W = 32                                       # the scenes' image: GOLD_W × GOLD_W pixels
gold_focus() = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, GOLD_W, GOLD_W))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
gold_vig() = (v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, GOLD_W, GOLD_W); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.3 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]))
function gold_light(board)
    b = 𝔟(board); a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2); a[:, 4] .= (0.1, 0.15, 0.2)
    return EnvironmentLight(SphericalHarmonicMap(a, 2), b, sphere_radius(b, [p.t for p in gold_poses()]))
end
const GOLD_ST = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4, seed = 42)   # Φ over πr² (re-recorded 2026-11-11: the un-normalized integral is retired)
const GOLD_PX = ((16, 16), (8, 24), (25, 9), (12, 20), (20, 12), (5, 5), (28, 28))

"""Column summaries of a Jacobian: Σ, Σ|·|, Σ² per column and the entries of 4 fixed rows."""
function jac_summary(J)
    rows = unique(clamp.([1, size(J, 1) ÷ 3, (2 * size(J, 1)) ÷ 3, size(J, 1)], 1, size(J, 1)))
    return (; size = collect(size(J)), sum = vec(sum(J; dims = 1)), sumabs = vec(sum(abs, J; dims = 1)), sumsq = vec(sum(abs2, J; dims = 1)),
            rows, at_rows = [collect(J[r, :]) for r in rows])
end
"""The scenes' radiometric camera: `lens` with the golden focus surface, aperture radius and vignetting."""
gold_camera(lens) = RCg.RadiometricCameraModel(lens, gold_focus(), 0.0386, gold_vig())
"""
A state of the camera of `lens` at the golden poses whose radiometry is away from the truth (non-zero residuals), and its targets
rendered at the truth. The old stack (`OLD_STACK`, the recording of golden_board_space.json) has no camera-first API: its views
and vignetting apart.
"""
function gold_state(board, m, lens)
    light = gold_light(board); poses = gold_poses(); dev = RadiometricCamera.KernelAbstractions.CPU()
    vig_start = with_control_points(gold_vig(), 1.05 .* Matrix(control_points(gold_vig())))
    light_start = with_coefficients(light, 0.9 .* Matrix(coefficients(light)))
    Wg, Hg = RadiometricCamera._image_size(lens)                       # the image is the camera's
    zero_targets = [zeros(Hg, Wg, 3) for _ in poses]
    if OLD_STACK
        views = [View(p, lens, gold_focus(), 0.0386) for p in poses]
        masks = [compute_mask(v, board, GOLD_ST; δ = 0.03, backend = dev) for v in views]
        tg = render_all(CalibrationState(board, GOLD_ST, copy(views), gold_vig(), m, light, zero_targets; masks))
        return CalibrationState(board, GOLD_ST, copy(views), vig_start, with_μ(m, 0.55), light_start, tg; masks)
    end
    camera = gold_camera(lens)
    masks = [compute_mask(camera, p, board, GOLD_ST; δ = 0.03, backend = dev) for p in poses]
    tg = render_all(CalibrationState(board, GOLD_ST, camera, poses, m, light, zero_targets; masks))
    return CalibrationState(board, GOLD_ST, RCg.with_vignetting(camera, vig_start), poses, with_μ(m, 0.55), light_start, tg; masks)
end
"""
The height columns of a tile Jacobian in a form that does not depend on the warping's basis. The columns are `J_h = HS · B`
(`B` the material block's `height_basis` of the board's warping), where `HS` holds the partials by the raw nodal heights. The render reads the heights only through their
thin-plate weights, which drop the affine fields `A = [1 b₁ b₂]` over the nodes (the poses' piston and tilts), so `HS A = 0`, and
`B` is one complement of `A`. Core 0.4's `B` was the orthogonal complement; Core 0.5's pure-bending heights are another (oblique)
one, so the raw columns moved while `HS` did not. `HS = J_h L` with `L` the first `HEIGHT_NFREE` rows of `[B A]⁻¹`.
Returned: `height_cols` and the rows of `HS` that `jac_summary` records (the column sums, then the fixed rows). golden_tests.jl
compares their Gram matrix with the one the recorded summary gives, since `(HS B₀)(HS B₀)ᵀ = HS HSᵀ` for the recorded orthonormal
complement `B₀`.
"""
function height_projection(J, t, vars, s)
    lay = RCg._tile_layout(RCg.static(1023), length(VgtkCore.model(vars[6]).nodes), length(s.settings.sensor), 25, VgtkCore.tangent_size(vars[9]),
                           Int(RCg.n_control_points(VgtkCore.model(vars[9]))), Int(sh_count(max_degree(vars[10].light))),
                           RCg.n_height(vars[8]); t.free)
    hc = collect(lay.height_cols)
    B = Matrix(vars[8].basis)                                     # the material block's: the (square) board's warping's basis
    A = hcat(ones(RCg.HEIGHT_NC), [n[1] for n in RCg.HEIGHT_NODES], [n[2] for n in RCg.HEIGHT_NODES])
    L = inv(hcat(B, A))[1:size(B, 2), :]
    HS = J[:, hc] * L
    rows = jac_summary(J).rows
    return (; height_cols = hc, height_proj = vcat(sum(HS; dims = 1), HS[rows, :]))
end
largest_tile(prob) =(t = RCg.tile_costs(prob); t[argmax([length(x.pix) for x in t])])

function gold_outputs()
    out = Dict{String, Any}()
    W = H = GOLD_W
    kb = cam(RCg.VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0), 41.216, (16.32, 15.36), W, H)
    # the recorded scene's grid lens: Core's quasi-interpolation of the double sphere at the 10 × 10 nodes, as grid_model built it
    # when the goldens were recorded (grid_model fits by Estimation's pspline_grid since 2026-11-04 — a different start, not what
    # these goldens of the flux and Jacobian pin)
    ds = cam(RCg.VgtkCore.Lens(RCg.VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.32, 15.36), W, H)
    # (Core 0.9: the SplineLens camera of that function, re-gauged to the double sphere's K — a pixel's lattice coordinate, so its
    # ray, does not depend on K)
    gcam = RCg.VgtkCore.CameraModel(RCg.VgtkCore.SplineLens, p -> (q = SVector{2, Float64}(p[1], p[2]); RCg.in_domain(ds, q) ? RCg.chief_ray(ds, q) : nothing),
                                    RCg.VgtkCore.Size2(W, H); grid_size = (10, 10), init = :quasi_interpolate)
    glens = RCg.VgtkCore.BoundedCamera(RCg.VgtkCore.CameraModel(gcam, RCg.VgtkCore.model(ds).intrinsics.K), RCg.VgtkCore.Size2(W, H))
    # 1. flux values: plane / warped board, KB / grid lens
    for (lab, lens, h) in (("kb_plane", kb, nothing), ("kb_warped", kb, gold_h()), ("grid_warped", glens, gold_h()))
        board, m = gold_board(W, H; h)
        light = gold_light(board)
        out["flux_" * lab] = OLD_STACK ?
            (view = View(gold_poses()[1], lens, gold_focus(), 0.0386);
             [collect(Φ_pixel(u, v, 1, view, gold_vig(), m, light, board, GOLD_ST)) for (u, v) in GOLD_PX]) :
            [collect(Φ_pixel(u, v, 1, gold_camera(lens), gold_poses()[1], m, light, board, GOLD_ST)) for (u, v) in GOLD_PX]
    end
    # 2. per-view problem (KB on the warped board): residual + Jacobian, closed-form joint radiometry, 3 LM iterations
    board, m = gold_board(W, H; h = gold_h())
    s = gold_state(board, m, kb)
    prob, _ = build_problem(s)
    r = get(prob.problem.costs, PerViewResidual{9})[1]
    res, J = RCg.NLLSsolver.computeresjac(RCg.static(127), r, RCg.NLLSsolver.getvars(r, prob.variables)...)
    out["perview_residual"] = res; out["perview_jacobian"] = jac_summary(J)
    s2 = deepcopy(s); parts = all_parts(s2); solve_radiometry_gn!(s2, parts; max_iter = 20); c = RCg.parts_cost(s2, parts)
    out["radiometry_gn"] = (; cost = c, P = vec(Matrix(control_points(s2.camera.vignetting))), a = vec(Matrix(coefficients(s2.light))), μ = s2.mat.diffuse_weight)
    # cost0: the start cost of the three iterations (golden_tests.jl bounds the final cost's rounding by eps·cost0)
    s3 = deepcopy(s); cost0 = cost(s3)[1]; run_lm!(s3; maxiters = 3, reldcost = 1e-12, xtol = 1e-14)
    out["lm3"] = (; cost = cost(s3)[1], cost0, Θ_d = collect(RCg.VgtkCore.parameter_vector(VgtkCore.model(geometric(s3)))), t̂ = collect(focus(s3).t̂_i), r = aperture_radius(s3),
                  poses = [vcat(vec(p.R), collect(p.t)) for p in s3.poses], mat = collect(RCg.Θ_m_nonlinear(s3.mat)), μ = s3.mat.diffuse_weight)
    # 3. per-tile problem (grid model, warping free): the largest tile
    s = gold_state(gold_board(W, H; h = 0.5 .* gold_h())..., glens)
    prob, _ = build_problem_grid(s; patch = (5, 5), height_degree = 1)
    t = largest_tile(prob)
    vars = RCg.NLLSsolver.getvars(t, prob.variables)
    res, J = RCg.NLLSsolver.computeresjac(RCg.static(1023), t, vars...)
    out["tile_warping_residual"] = res
    out["tile_warping_jacobian"] = OLD_STACK ? jac_summary(J) : merge(jac_summary(J), height_projection(J, t, vars, s))
    # 4. per-tile problem on a distance-map board (the printed edge: held, as always)
    sdf = let Hs = 160, Ws = 160, page = (200.0, 200.0), o = zeros(160, 160)
        rects = [(12.5 + 50i, 12.5 + 50j, 37.5 + 50i, 37.5 + 50j) for i in 0:3 for j in 0:3]
        for i in 1:Hs, j in 1:Ws
            x = (j - 0.5) / Ws * page[1]; y = (i - 0.5) / Hs * page[2]; din = Inf; dout = Inf
            for (x0, y0, x1, y1) in rects
                if x0 <= x <= x1 && y0 <= y <= y1; din = min(din, x - x0, x1 - x, y - y0, y1 - y)
                else dout = min(dout, hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))) end
            end
            o[i, j] = isfinite(din) ? -din : dout
        end
        (o, page)
    end
    # the print's edge (δ, σ_print) on the surface (σ_print 3 mm, about one pixel on the board: before 2026-11-14 the edge's width was the
    # camera's in-focus blur, σ_cam 0.9 px, with σ_print 0.3 mm)
    s = gold_state(gold_board(W, H; edge = SVector(0.3, 3.0), sdf)..., glens)
    prob, _ = build_problem_grid(s; patch = (5, 5))
    t = largest_tile(prob)
    res, J = RCg.NLLSsolver.computeresjac(RCg.static(1023), t, RCg.NLLSsolver.getvars(t, prob.variables)...)
    out["tile_edge_residual"] = res; out["tile_edge_jacobian"] = jac_summary(J)
    return out
end
