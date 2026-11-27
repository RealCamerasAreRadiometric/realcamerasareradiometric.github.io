# Every gauge held exactly (owner rule, 2026-11-11: no pin, prior, penalty or tolerance holds one): the grid lens's common rotation
# (the node directions and the poses), the vignetting scale V(𝐜) = 1 and the aperture's radius against the lighting's scale. Each
# item shows the gauge direction and that the parametrization has no coordinate along it.

@testitem "gauges: the grid's common rotation is held by three coordinates of two nodes" setup = [RCFixtures, GridTilesScene] begin
    using RadiometricCamera: build_problem_grid, cost, _lm_tiles!
    using StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    # the near-gauge: one infinitesimal rotation ω of every node direction (its tangent coordinates t_k · (ω × d)) and of every
    # pose (left: the camera frame turns, ξ = (ω, 0)) leaves the tile residuals all but unchanged (only the aperture disc and the
    # focus, fixed in the camera frame, do not turn with it)
    sp = calibration_problem(board, st, camera_p, copy(poses_p), mat_t, light_t, targets; masks)
    prob0, idx0 = build_problem_grid(sp.f, sp.u0; patch = (5, 5), rotation_gauge = false)
    pin = RC.rotation_gauge_coordinates(glens_p, nothing)
    @test length(pin) == 3 && pin[1][1:2] == pin[2][1:2] && pin[1][3] == 1 && pin[2][3] == 2 && pin[3][1:2] != pin[1][1:2]
    # transversal: the three held coordinates of the rotation's field are a nonsingular map of ω (a slice of the gauge orbit)
    D = VgtkCore.directions(VgtkCore.model(glens_p))
    held_coord(ω, (gx, gy, c)) = (d = SVector{3, Float64}(D[gy, gx]); RC._control_tangents(d)[c] ⋅ (ω × d))
    M = [held_coord(SVector{3}(I[1:3, j]), p) for p in pin, j in 1:3]
    @test cond(M) < 1e3
    # with the gauge held: every tile's and the smoothness's columns of the held coordinates are exactly zero
    prob1, idx1 = build_problem_grid(sp.f, sp.u0; patch = (5, 5), smoothness = RC.grid_smoothness(RC.PSplineSmoothness(3, 0.1), glens_p, 1.0, trues(10, 10)))
    G_x = size(D, 2)
    px_n = 5
    for c in prob1.blocks
        T = typeof(c)
        (T <: RC.PerTileResidual || T <: RC.GridSmoothnessResidual) || continue
        vi = RC.NLLSsolver.varindices(c); vars = RC.NLLSsolver.getvars(c, prob1.variables)
        flags = 0; for k in eachindex(vi); flags |= 1 << (k - 1); end
        _, J = RC.NLLSsolver.computeresjac(flags, c, vars...)
        # the patch columns of the held coordinates in this block's layout
        col = 0
        for (k, v) in enumerate(vi)
            n = Int(RC.NLLSsolver.nvars(vars[k]))
            if v in idx1.patches
                P = findfirst(==(v), idx1.patches); PI, PJ = Tuple(P)
                for (gx, gy, cc) in pin
                    lx = gx - (PI - 1) * px_n; ly = gy - (PJ - 1) * px_n
                    (1 <= lx <= px_n && 1 <= ly <= px_n) || continue
                    j = col + 2 * ((lx - 1) + (ly - 1) * px_n) + cc
                    @test all(iszero, J[:, j])
                end
            end
            col += n
        end
    end
    # and the joint LM moves none of them (a zero column's step is zero), the poses and the grid both free: every accepted step leaves
    # the chief node where it is and moves the roll node along its free tangent only, in that step's chart (the chart turns with the
    # node, so the held coordinate is zero per step, not in the start's chart once the node has turned)
    sl = calibration_problem(board, st, camera_p, copy(poses_p), mat_t, light_t, targets; masks)
    (gx1, gy1, _) = pin[1]; (gx3, gy3, c3) = pin[3]
    node(lens, gx, gy) = normalize(SVector{3, Float64}(VgtkCore.directions(VgtkCore.model(lens))[gy, gx]))
    roll = [node(sl.u0.camera.geometric, gx3, gy3)]; chief = [node(sl.u0.camera.geometric, gx1, gy1)]
    _lm_tiles!(sl.f, sl.u0; stops = RC._calibration_stops(; max_iter = 3), damping = VgtkCore.LevenbergDamping(), patch = (5, 5), fixed = (:focus, :aperture, :vignetting, :lighting), callback = state -> (push!(roll, node(state.u.camera.geometric, gx3, gy3)); push!(chief, node(state.u.camera.geometric, gx1, gy1)); false))
    @test length(roll) == 4 && all(d -> norm(d - chief[1]) < 1e-14, chief)                       # the chief node: both coordinates held
    moves = [roll[k + 1] - roll[k] for k in 1:3]
    @test norm(roll[end] - roll[1]) > 0
    # the roll node: its held coordinate, per step — to the rounding of the move, a difference of two unit vectors, whose floor is
    # ABSOLUTE (a few eps), not relative to the move: measured 1.0e-16 at |move| = 2.4e-5 under λ·I's shorter steps (the default optimizer's)
    @test all(k -> abs(RC._control_tangents(roll[k])[c3] ⋅ moves[k]) <= 1e-12 * norm(moves[k]) + 4eps(), 1:3)
end

@testitem "gauges: V(𝐜) = 1 is held by the parametrization; the soft pseudo-observation is gone" setup = [RCFixtures] begin
    @test occursin("\"pin\"", sprint(showerror, try RC.pipeline_options(Dict("vignetting" => Dict("pin" => "hard"))); nothing catch e; e end))
    @test !isdefined(RC, :VignettingPinResidual) && !isdefined(RC, :PIN_WEIGHT)
end

@testitem "gauges: the flux is the aperture integral over πr²; the un-normalized integral is gone" setup = [RCFixtures] begin
    using RadiometricCamera: RenderSettings
    @test !hasfield(RenderSettings, :normalize_aperture)
    @test_throws MethodError RenderSettings(; normalize_aperture = false)
end
