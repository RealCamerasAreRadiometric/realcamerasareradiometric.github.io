@testitem "quadrature" setup = [RCFixtures] begin
    using RadiometricCamera: aperture_border, aperture_samples, ApertureQuadrature, pixel_of_𝐩, RenderSettings, SensorQuadrature,
                             ℐ_𝐮, 𝐩_of_pixel
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    import VisualGeometryToolkitCore as VgtkCore

    sq = SensorQuadrature(3)
    @test sum(sq.weights) ≈ 1
    @test all(all(-0.5 .< o .< 0.5) for o in sq.offsets)
    aq = ApertureQuadrature(4, 12)
    @test sum(aq.w̃) ≈ π
    @test sum(w * a[1]^2 for (a, w) in zip(aq.â, aq.w̃)) ≈ π / 4 atol = 1e-12
    @test sum(w * (a[1]^2 + a[2]^2) for (a, w) in zip(aq.â, aq.w̃)) ≈ π / 2 atol = 1e-12
    # aperture samples at radius r integrate the area πr²
    r = 0.03
    @test sum(w for (a, w) in aperture_samples(aq, r)) ≈ π * r^2
    @test sum(ds for (a, v, n, ds) in aperture_border(aq, r)) ≈ 2π * r
    @test all(abs(dot(v, n) - 1) < 1e-14 for (a, v, n, ds) in aperture_border(aq, r))
    u1, u2 = VgtkCore.hammersley(3, 8); @test 0 < u1 < 1 && 0 <= u2 < 1
    @test 𝐩_of_pixel(3, 7) == SVector(2.5, 6.5) && pixel_of_𝐩(𝐩_of_pixel(3, 7)) ≈ SVector(3.0, 7.0)     # COLMAP pixels: pixel (u, v)'s centre
    lo, hi = ℐ_𝐮(1, 1); @test lo ≈ SVector(0.0, 0.0) && hi ≈ SVector(1.0, 1.0)
    # the sensor rule is the pixel box: the tensor Gauss–Legendre rule, its weights summing to one
    @test fieldnames(SensorQuadrature) == (:offsets, :weights) && sum(SensorQuadrature{Float64}(3).weights) ≈ 1
    @test all(o -> all(abs.(o) .< 0.5), SensorQuadrature{Float64}(3).offsets) && length(RenderSettings(; n_𝐩 = 4).sensor) == 16
    @test RenderSettings().sensor.weights == SensorQuadrature{Float64}(2).weights
end
