# (the quadrature rules and the sensor domain are VisualRadiometryToolkitCore's, tested there; here: the render settings hold them)
@testitem "quadrature: the render settings' rules" setup = [RCFixtures] begin
    using RadiometricCamera: RenderSettings, SensorQuadrature
    @test length(RenderSettings(; n_𝐩 = 4).sensor) == 16
    @test RenderSettings().sensor.weights == SensorQuadrature{Float64}(2).weights
    u1, u2 = VgtkCore.hammersley(3, 8); @test 0 < u1 < 1 && 0 <= u2 < 1
end
