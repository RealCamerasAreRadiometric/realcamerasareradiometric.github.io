# Θ_m held while the board warping rides the material block (`fixed = (:material,)` with a free warping: a staged refinement's geometric
# stage, h free with the poses and intrinsics, the material held): the material block stays in the problem (`MaterialVar{…, true}`,
# `_hold_material`), its 9 columns (8 nonlinear by AD + the analytic μ column) are zero, every other column and the residual are
# unchanged, and `update` leaves Θ_m while moving the warping.
@testitem "frozen material (staged refinement)" setup = [RCFixtures] begin
    using RadiometricCamera: build_problem, build_problem_grid, _image_start, PerViewResidual
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    module FreezeMaterialTest
    using Test, RadiometricCamera, StaticArrays, LinearAlgebra
    using RadiometricCamera: build_problem_grid, _image_start, View
    const OLD_STACK = false
    include(joinpath(@__DIR__, "..", "regression", "golden_scenes.jl"))
    const NL = RCg.NLLSsolver

    function frozen_columns_check((cost0, vars0), (cost1, vars1), flags)
        r0, J0 = NL.computeresjac(flags, cost0, vars0...)
        r1, J1 = NL.computeresjac(flags, cost1, vars1...)
        @test r1 == r0
        @test size(J1) == size(J0)
        gone = findall(c -> all(iszero, J1[:, c]) && !all(iszero, J0[:, c]), axes(J0, 2))
        @test 1 <= length(gone) <= 9 && maximum(gone) - minimum(gone) < 9           # only the material block's 9 columns are zeroed
        keep = setdiff(axes(J0, 2), gone)
        @test J1[:, keep] == J0[:, keep]
    end

    W = H = 32
    @testset "per-tile block (grid lens, warping free)" begin
        glens = _image_start(cam(RCg.VgtkCore.Lens(RCg.VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.32, 15.36), W, H), (10, 10)).camera
        s = gold_state(gold_board(W, H; h = 0.5 .* gold_h())..., glens)
        prob, _ = build_problem_grid(s.f, s.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5), height_degree = 1)
        held, idx = build_problem_grid(s.f, s.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5), height_degree = 1, fixed = (:material,))
        @test idx.free[idx.material]                                                         # the block stays variable
        t = largest_tile(prob); vars = NL.getvars(t, prob.variables)
        th = largest_tile(held); varsh = NL.getvars(th, held.variables)
        frozen_columns_check((t, vars), (th, varsh), RCg.static(1023))
        mv = vars[findfirst(v -> v isa RCg.MaterialVar, vars)]
        mvh = varsh[findfirst(v -> v isa RCg.MaterialVar, varsh)]
        @test RCg.material_fixed(mvh) == (:material,) && RCg.material_fixed(mv) == ()
        u = fill(0.01, Int(NL.nvars(mv)))
        m2 = NL.update(mvh, u)
        @test RCg.Θ_m_nonlinear(m2.mat) == RCg.Θ_m_nonlinear(mv.mat) && m2.mat.diffuse_weight == mv.mat.diffuse_weight   # Θ_m held
        @test m2.surface.h != mv.surface.h                                                   # the warping moves
        m3 = NL.update(mv, u)
        @test RCg.Θ_m_nonlinear(m3.mat) != RCg.Θ_m_nonlinear(mv.mat)                         # (unfrozen: the material moves)
    end
    @testset "per-view block (Kannala–Brandt)" begin
        kb = cam(RCg.VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0), 41.216, (16.32, 15.36), W, H)
        board, m = gold_board(W, H; h = gold_h())
        s = gold_state(board, m, kb)
        prob, _ = build_problem(s.f, s.u0; damping = VgtkCore.LevenbergDamping(), height_degree = 1)
        held, idx = build_problem(s.f, s.u0; damping = VgtkCore.LevenbergDamping(), height_degree = 1, fixed = (:material,))
        @test idx.free[idx.material]
        r = get(prob.problem.costs, PerViewResidual{9})[1]; rh = get(held.problem.costs, PerViewResidual{9})[1]
        frozen_columns_check((r, NL.getvars(r, prob.variables)), (rh, NL.getvars(rh, held.variables)), RCg.static(127))
        # nothing rides the block: "material" holds the whole block (no variable Θ_m to hold inside it)
        whole, idxw = build_problem(s.f, s.u0; damping = VgtkCore.LevenbergDamping(), fixed = (:material,))
        @test !idxw.free[idxw.material] && RCg.material_fixed(whole.variables[idxw.material]) == ()
    end
    end
end
