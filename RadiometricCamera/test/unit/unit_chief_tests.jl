# The focus point and every ray use the UNIT chief direction (decision 2026-10-23): 1/t̂(𝐩) is the distance from the centre of projection to
# 𝔣(𝐩) for every lens class — including the grid lens, whose bicubic blend of unit node directions is not unit. The grid here has its
# nodes deliberately scaled (±20 %) so that the raw blend is far from unit length.
@testitem "unit chief direction (focus point)" setup = [RCFixtures] begin
    using Test, StaticArrays, LinearAlgebra, ForwardDiff
    using RadiometricCamera
    using VisualGeometryToolkitCore: exp_so3
    const RCU = RadiometricCamera

    kb_u = cam(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.0, 0.0), 57.6, (32.0, 32.0), 64, 64)
    g0_u = RCU.grid_model(kb_u, 12, 12)
    D0_u = RCU.VgtkCore.directions(RCU.VgtkCore.model(g0_u))
    G_u = [D0_u[j, i] * (1 + 0.2 * sin(0.7i) * cos(0.9j)) for j in axes(D0_u, 1), i in axes(D0_u, 2)]   # node (x = i, y = j)
    g_u = RCU.VgtkCore.with_directions(g0_u, G_u)
    focus_u = RCU.Focus(RCU.FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, 64, 64))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    camera_u = RCU.RadiometricCameraModel(g_u, focus_u, 0.004, VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 64, 64); degree = 2))
    𝐩s_u = [SVector(x, y) for x in 6.4:12.8:57.6 for y in 12.8:19.2:51.2]           # pixels of the 64 × 64 image

    @testset "unit chief direction" begin
        @test maximum(abs(norm(RCU.chief_ray(g_u, 𝐩)) - 1) for 𝐩 in 𝐩s_u) > 0.05          # the raw blend is deliberately far from unit
        for 𝐩 in 𝐩s_u
            d̂ = RCU.chief_dir(g_u, 𝐩)
            @test norm(d̂) ≈ 1 atol = 1e-14
            @test d̂ ≈ normalize(RCU.chief_ray(g_u, 𝐩)) atol = 1e-14
            @test norm(RCU.𝔣(camera_u, 𝐩)) * RCU.t̂(camera_u.focus, 𝐩) ≈ 1 atol = 1e-12  # 𝔭 = 0: the focus distance is exactly 1/t̂
            for 𝐚 in (SVector(0.0, 0.0, 0.0), SVector(0.003, -0.002, 0.0), SVector(-0.001, 0.0035, 0.0))
                @test RCU.π̂⁻¹(camera_u, 𝐩, 𝐚) ≈ normalize(RCU.𝔣(camera_u, 𝐩) - 𝐚) atol = 1e-12
                @test norm(RCU.π̂⁻¹(camera_u, 𝐩, 𝐚)) ≈ 1 atol = 1e-14
            end
            @test RCU.π̂⁻¹(camera_u, 𝐩, SVector(0.0, 0.0, 0.0)) ≈ d̂ atol = 1e-14              # π̂⁻¹(𝐩, 𝔭) = d̂(𝐩)
            # the sensor Jacobian of the UNIT direction is tangent to the unit sphere
            Jg = ForwardDiff.jacobian(p -> RCU.chief_dir(g_u, p), 𝐩)
            @test all(abs.(Jg' * d̂) .< 1e-10)
        end
        # parametric lenses are unit already: chief_dir changes nothing there
        for 𝐩 in 𝐩s_u
            @test RCU.chief_dir(kb_u, 𝐩) ≈ RCU.chief_ray(kb_u, 𝐩) atol = 1e-14
        end
    end
end
