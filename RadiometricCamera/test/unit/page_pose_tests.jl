@testitem "the page pose: one labelling, the camera on the printed side (cheirality)" begin
    using LinearAlgebra, StaticArrays, StableRNGs
    const RC = RadiometricCamera
    import VisualGeometryToolkitCore as VgtkCore
    rng = StableRNG(5)
    # an A4-sized page, x along the columns, rows along +y, the printed side's normal −z (the Schöps front end's frame)
    𝐨 = SVector(0.0, 0.0, 0.0); 𝐮 = SVector(0.21, 0.0, 0.0); 𝐯 = SVector(0.0, 0.297, 0.0)
    𝐧 = -normalize(cross(𝐮, 𝐯)); bc = 𝐨 + 0.5𝐮 + 0.5𝐯
    @test 𝐧 ≈ SVector(0.0, 0.0, -1.0)
    Xs = [𝐨 + rand(rng) * 𝐮 + rand(rng) * 𝐯 for _ in 1:60]          # generic page points: no mirror symmetry
    rays(R, t) = [normalize(R * X + t) for X in Xs]
    centre(R, t) = -R' * t
    # a camera 0.4 m in front of the printed side, tilted
    R = VgtkCore.exp_so3(SVector(0.15, -0.2, 0.05)); t = SVector(0.0, 0.0, 0.4) - R * bc
    @test centre(R, t)[3] < 0

    e, R̂, t̂ = RC._fit_page_pose(Xs, rays(R, t), 𝐨, 𝐧, bc, (0.3, 0.6))
    @test e < 1.0e-9
    @test R̂ ≈ R atol = 1.0e-8
    @test t̂ ≈ t atol = 1.0e-8
    @test VgtkCore.signed_distance(VgtkCore.Plane3D(VgtkCore.Point3(𝐨...), VgtkCore.Vec3(𝐧...)), VgtkCore.Point3(centre(R̂, t̂)...)) > 0

    # The old ±y tie: rows along −y with the camera behind (+z) is the same scene in a frame rotated 180° about x — identical rays.
    # In ITS frame (𝐧 = +z) the fit returns that frame's pose, on its printed side; each labelling has exactly one solution.
    F = SMatrix{3, 3, Float64}(Diagonal(SVector(1.0, -1.0, -1.0)))
    Xs_f = [F * X for X in Xs]
    @test [normalize((R * F) * X + t) for X in Xs_f] ≈ rays(R, t)   # F is its own inverse: R·F applied to F·X is R·X
    e_f, R̂_f, t̂_f = RC._fit_page_pose(Xs_f, rays(R, t), 𝐨, -𝐧, F * bc, (0.3, 0.6))
    @test e_f < 1.0e-9
    @test R̂_f ≈ R * F atol = 1.0e-8
    @test centre(R̂_f, t̂_f)[3] > 0

    # A mirrored texture — the print seen from behind: the rays of a camera at +z looking at the page's back. No pose on the printed
    # side reproduces a mirror image of generic points, so the fit is refused (Inf) or far off; it never lands behind the page.
    Rb = F * VgtkCore.exp_so3(SVector(0.1, 0.1, 0.0)); tb = SVector(0.0, 0.0, 0.4) - Rb * bc
    @test centre(Rb, tb)[3] > 0
    e_m, R̂_m, t̂_m = RC._fit_page_pose(Xs, rays(Rb, tb), 𝐨, 𝐧, bc, (0.3, 0.6))
    @test !isfinite(e_m) || (e_m > 1.0e-3 && centre(R̂_m, t̂_m)[3] < 0)
end
