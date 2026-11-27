# The light sphere's radius (the pipeline's rule). The light itself is VisualGeometryToolkitCore's EnvironmentLight: its spherical
# harmonics, exit direction and linearity are Core's tests (test/lighting_tests.jl there).
@testitem "light sphere radius" setup = [RCFixtures] begin
    using RadiometricCamera: TexturedTarget, sphere_radius
    @test sphere_radius([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]) ≈ 2
    # a 2 × 2 page about the origin: its corners √2 from the centre
    bd2 = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], ones(4, 4, 3)), [0.0, 0.0, 1.0])
    @test sphere_radius(bd2, [[0.0, 0.0, 1.0]]) ≈ sqrt(2)                   # a close camera: the page's corners set the pipeline's radius (2026-11-04)
    @test sphere_radius(bd2, [[0.0, 0.0, 3.0]]) ≈ 3                         # a far camera: the camera does
end
