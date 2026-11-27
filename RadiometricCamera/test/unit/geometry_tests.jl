@testitem "pose / board / back-projection" setup = [RCFixtures] begin
    using RadiometricCamera: board_normal, chart, Focus, inside, intersect_board, TexturedTarget, world_point, world_ray, backproject,
                             chief_ray, focus_point, center_of_projection
    using VisualRadiometryToolkitCore: albedo
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    # A view's pose is Core's camera → world EuclideanMap: R = R_cw, t = the camera centre
    p = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(0.1, 0.2, -0.3))), SVector{3}(SVector(1.0, 2.0, 3.0)); check = false)
    @test p isa VgtkCore.PlainEuclideanMap{3, Float64} && isbits(p)
    @test inv(p)(p.t) ≈ zeros(3) atol = 1e-12                            # the camera centre maps to the camera origin
    # The Jacobian evaluations' pose: Core's LeftRigidManifold6 at the pose (the LM's pose block) retracted at seeded (ω, ρ),
    # against the matrix exponential of [ω]ₓ written out here (LinearAlgebra's `exp`, not Rodrigues), and its derivative of a world
    # point's camera → world image against the chart's action Jacobian [−[R x + t]ₓ  I₃]
    ω = SVector(0.3, -0.2, 0.25); ρ = SVector(0.05, -0.1, 0.2)
    Eω = exp([0.0 -ω[3] ω[2]; ω[3] 0.0 -ω[1]; -ω[2] ω[1] 0.0])
    q = RC._seeded_pose(p, identity, ω, ρ)
    @test q.R ≈ Eω * p.R atol = 1e-14
    @test q.t ≈ Eω * p.t + ρ atol = 1e-14
    x = SVector(0.4, -0.3, 2.0); y = p(x)
    J = ForwardDiff.jacobian(v -> RC._seeded_pose(p, z -> z + zero(eltype(v)), v[SVector(1, 2, 3)], v[SVector(4, 5, 6)])(x), zeros(SVector{6}))
    @test J ≈ hcat([0.0 y[3] -y[2]; -y[3] 0.0 y[1]; y[2] -y[1] 0.0], Matrix(1.0I, 3, 3)) atol = 1e-12
    # a view's pose from its rays (the front-ends' refit): RC._refined_pose (Core's LevenbergMarquardt on Estimation's BearingPoseProblem) on the front-ends' bearing
    # correspondences, world → camera (X_cam = R X + t), from a perturbed start,
    # to the exact pose
    let R = exp_so3(SVector(0.1, -0.2, 0.05)), t = SVector(0.1, 0.0, 2.0)
        Xs = [SVector(x, y, 0.3 * sin(x + y)) for x in -0.5:0.25:0.5 for y in -0.5:0.25:0.5]
        guess = VgtkCore.EuclideanMap(exp_so3(SVector(0.02, 0.0, -0.01)) * R, t + SVector(0.01, -0.02, 0.05); check = false)
        r1 = RC._refined_pose(guess, RC._ray_correspondences(Xs, [R * X + t for X in Xs])); p1 = r1.u
        @test VgtkCore.successful_retcode(r1) && maximum(abs, p1.R - R) < 1e-9 && maximum(abs, p1.t - t) < 1e-9
    end
    # board
    tex = rand(64, 96, 3)
    bd = RadiometricCamera.with_facing(TexturedTarget([0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], tex), [0.0, 0.0, 0.0])
    @test board_normal(bd)[3] < 0
    s, 𝐱 = intersect_board(bd, SVector(0.0, 0.0, 0.0), SVector(0.25, 0.5, 1.0))
    @test 𝐱[3] ≈ 2 && chart(bd, 𝐱) ≈ SVector(0.5, 1.0)
    @test world_point(bd, SVector(0.5, 1.0)) ≈ 𝐱
    @test VgtkCore.centroid(bd) ≈ SVector(0.5, 0.5, 2.0)
    @test inside(SVector(0.5, 0.5), 0.1) && !inside(SVector(0.05, 0.5), 0.1)
    𝐛 = SVector((10 - 0.5) / 96, (7 - 0.5) / 64)
    @test albedo(bd, 𝐛) ≈ SVector(tex[7, 10, :]...)
    @test albedo(bd, ForwardDiff.Dual.(𝐛, 1.0)) isa SVector{3, <:ForwardDiff.Dual}
    @test all(isfinite, albedo(bd, SVector(NaN, 0.5)))
    # Core reference ray–plane intersection
    X = Base.intersect(bd.quad.plane, VgtkCore.Point3(0.0, 0.0, 0.0), VgtkCore.Vec3(0.25, 0.5, 1.0))
    @test SVector(X...) ≈ 𝐱
    # back-projection
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.0, 12.0), 24, 24)
    camera = RadiometricCameraModel(lens, Focus(5, 1.5, 24, 24), 0.03, VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 24, 24); degree = 2))
    𝐩 = SVector(16.8, 9.6); 𝐚 = SVector(0.02, -0.01, 0.0)
    o, d = world_ray(camera, p, 𝐩, 𝐚)
    @test o ≈ p.t + p.R * 𝐚 && norm(d) ≈ 1
end
