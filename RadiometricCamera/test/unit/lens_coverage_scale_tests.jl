# masks.lens_coverage_scale (owner, 2026-11-15): the photometric pixels are sampled inside the lens coverage scaled about its centre —
# sampling only. 1.0 is the imaged region itself (bit for bit); 0.9 samples no pixel outside the scaled ellipse; the camera's own
# coverage is untouched; the sensor's rectangle (no image circle) is not scaled.

@testitem "masks.lens_coverage_scale: sampling inside the scaled coverage only" setup = [RCFixtures] begin
    using Test
    W, H = 640, 480
    c0 = cam(VgtkCore.Division(-0.15, 0.03), 540.0, (326.4, 230.4), W, H)
    e = VgtkCore.Ellipse(VgtkCore.Point2(320.0, 240.0), 250.0, 200.0, 0.3)
    c = VgtkCore.with_lens_coverage(c0, e)
    @test VgtkCore.pixel_mask(RC._sampling_region(c, 1.0), W, H) == VgtkCore.pixel_mask(VgtkCore.imaged_region(c), W, H)
    m9 = VgtkCore.pixel_mask(RC._sampling_region(c, 0.9), W, H)
    e9 = VgtkCore.Ellipse(e.center, 0.9 * e.a, 0.9 * e.b, e.θ)
    @test count(m9) < count(VgtkCore.pixel_mask(VgtkCore.imaged_region(c), W, H))
    @test all(VgtkCore.Point2{Float64}(u - 0.5, v - 0.5) ∈ e9 for v in 1:H, u in 1:W if m9[v, u])     # no pixel centre outside it
    @test VgtkCore.lens_coverage(c) == e                                                              # the camera's coverage untouched
    @test VgtkCore.pixel_mask(RC._sampling_region(c0, 0.9), W, H) == VgtkCore.pixel_mask(VgtkCore.imaged_region(c0), W, H)   # the sensor's: not scaled
    @test RC.CalibrationSettings().masks.lens_coverage_scale == 1.0                                       # the default (configs/default.toml)
    @test_throws ArgumentError RC._sampling_region(c, 1.2)
end
