@testitem "flux" setup = [RCFixtures] begin
    using RadiometricCamera: TargetState, chart, compute_mask, Focus, intersect_board, L_c, masked_pixels, Material, pixel_id,
                             TexturedTarget, render_view, RenderSettings, SampleKey, sphere_radius, with_vignetting, with_μ,
                             world_ray, Φ_pixel, 𝐩_of_pixel, 𝔟, 𝔭
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points
    import VisualGeometryToolkitCore as VgtkCore

    tex = fixture_board()
    board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.1, 0.05, 0.0))), SVector{3}(SVector(0.05, -0.05, 1.5)); check = false)
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.0, 12.0), 24, 24)
    vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 24, 24); degree = 2); mat = Material(); light = EnvironmentLight(SphericalHarmonicMap(1.0, 2), 𝔟(board), sphere_radius(𝔟(board), [pose.t]))
    camera = RadiometricCameraModel(lens, Focus(3, 1.5, 24, 24), 0.0386, vig)
    # the camera with another aperture radius (a number or a dual)
    with_r(r) = RadiometricCameraModel(camera.geometric, camera.focus, r, camera.vignetting)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)                    # Φ = the aperture integral over πr²
    mask = compute_mask(camera, pose, board, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU())
    @test 0 < count(mask) <= length(mask)
    img = render_view(1, camera, pose, mat, light, board, st, mask)
    @test all(img[:, :, 1][mask] .> 0) && all(img[:, :, 1][.!mask] .== 0)
    u, v = masked_pixels(mask)[end ÷ 2]
    Φ = Φ_pixel(u, v, 1, camera, pose, mat, light, board, st)
    @test Φ ≈ SVector(img[v, u, :]...)
    # exact linearity in Θ_v and Θ_l, affinity in μ
    P1 = rand(4, 4); P2 = rand(4, 4); σ = 0.6
    ΦV(vg) = Φ_pixel(u, v, 1, with_vignetting(camera, vg), pose, mat, light, board, st)
    @test ΦV(with_control_points(vig, σ * P1 + P2)) ≈ σ * ΦV(with_control_points(vig, P1)) + ΦV(with_control_points(vig, P2)) atol = 1e-12
    a1 = rand(3, 9); a2 = rand(3, 9)
    Φa(a) = Φ_pixel(u, v, 1, camera, pose, mat, with_coefficients(light, a), board, st)
    @test Φa(σ * a1 + a2) ≈ σ * Φa(a1) + Φa(a2) atol = 1e-12
    Φμ(μ) = Φ_pixel(u, v, 1, camera, pose, with_μ(mat, μ), light, board, st)
    @test Φμ(0.3) ≈ 0.3 * Φμ(1.0) + 0.7 * Φμ(0.0) atol = 1e-12
    # ∂Φ/∂r by AD through the r-scaled unit-disc rule (the fixed-domain form of the aperture integral) vs finite differences
    st_f = RenderSettings(; n_𝐩 = 2, n_ρ = 8, n_φ = 32, N = 32, hammersley = true)
    Φr(r) = Φ_pixel(u, v, 1, with_r(r), pose, mat, light, board, st_f)
    h_r = 1e-4
    r0 = camera.aperture_radius
    fd = (Φr(r0 + h_r) - Φr(r0 - h_r)) / (2h_r)
    ad = ForwardDiff.partials.(Φ_pixel(u, v, 1, with_r(ForwardDiff.Dual(r0, 1.0)), pose, mat, light, board, st_f), 1)
    @test norm(ad - fd) / norm(fd) < 1e-5
    # pose derivative through ForwardDiff vs FD
    Φω(e) = Φ_pixel(u, v, 1, camera, VgtkCore.retract(VgtkCore.LeftRigidManifold6(pose), vcat(SVector(e, 0.0, 0.0), zeros(SVector{3}))), mat, light, board, st)
    ωd = SVector(ForwardDiff.Dual(0.0, 1.0), ForwardDiff.Dual(0.0, 0.0), ForwardDiff.Dual(0.0, 0.0))
    pd = VgtkCore.retract(VgtkCore.LeftRigidManifold6(VgtkCore.EuclideanMap(SMatrix{3, 3}(ForwardDiff.Dual.(pose.R, 0.0)), SVector{3}(ForwardDiff.Dual.(pose.t, 0.0)); check = false)), vcat(ωd, ForwardDiff.Dual.(zeros(SVector{3}), 0.0)))
    Φd = Φ_pixel(u, v, 1, camera, pd, mat, light, board, st)
    @test ForwardDiff.partials.(Φd, 1) ≈ (Φω(1e-6) - Φω(-1e-6)) / 2e-6 rtol = 1e-5
    # thin-lens sanity: r → 0 gives the pinhole (chief-ray) image; a large aperture blurs
    st_p = RenderSettings(; n_𝐩 = 1, n_ρ = 1, n_φ = 1, N = 4)
    Φ_small = Φ_pixel(u, v, 1, with_r(1e-6), pose, mat, light, board, st_p)
    o, d = world_ray(camera, pose, 𝐩_of_pixel(u, v), 𝔭); _, 𝐱 = intersect_board(board, o, d)
    Lchief = L_c(board, mat, light, st_p.sampler, SampleKey(1, pixel_id(u, v, 24), 1, 1), chart(board, 𝐱), 𝐱, d)
    @test Φ_small ≈ Lchief rtol = 1e-3
end
