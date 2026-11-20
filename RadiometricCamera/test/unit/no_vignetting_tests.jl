# A camera without vignetting (VisualRadiometryToolkitCore's `NoVignetting`, the composed camera's default): the kernels hold no
# multiplication by V and V has no parameters. The render equals, bit for bit, the render that multiplies by a field whose value is
# exactly 1 (`UnitField`, below), on the host and on a device; it equals the flat B-spline surface's to rounding (the flat
# surface's value is the sum of its basis functions, 1 only to an ulp). The calibration, which estimates V, refuses the camera by
# name with what to give instead.

@testsetup module NoVignettingScene
    using RadiometricCamera, StaticArrays
    using RadiometricCamera: compute_mask, flat_vignetting, Focus, Material, PrintedTarget, RenderSettings, Scene, sphere_radius, 𝔟
    using VisualGeometryToolkitCore: exp_so3
    import VisualRadiometryToolkitCore
    using VisualRadiometryToolkitCore: EnvironmentLight, SphericalHarmonicMap, NoVignetting
    import VisualGeometryToolkitCore as VgtkCore
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    """A vignetting whose value is exactly `one(T)` at every point, MULTIPLIED by the kernels (unlike `NoVignetting`): the reference
    of the bitwise comparison."""
    struct UnitField{T} <: VisualRadiometryToolkitCore.AbstractVignetting end
    @inline (::UnitField{T})(::StaticVector{2}) where {T} = one(T)
    VisualRadiometryToolkitCore.control_points(::UnitField{T}) where {T} = SMatrix{1, 1, T, 1}(one(T))
    VgtkCore.convert_eltype(::Type{T}, ::UnitField) where {T} = UnitField{T}()
    function no_vignetting_scene()
        board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], fixture_board(); facing = [0.0, 0.0, 1.5])
        scene = Scene(board, 24, 24)
        pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.1, 0.05, 0.0))), SVector(0.05, -0.05, 1.5); check = false)
        lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.0, 12.0), 24, 24)
        none = RadiometricCameraModel(lens; focus = 1.5, aperture_radius = 0.0386)                    # the default: NoVignetting
        flat = RadiometricCameraModel(lens; focus = 1.5, aperture_radius = 0.0386, vignetting = flat_vignetting(lens))
        light = EnvironmentLight(SphericalHarmonicMap(1.0, 2), 𝔟(board), sphere_radius(𝔟(board), [pose.t]))
        st = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
        mask = compute_mask(flat, pose, scene, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU())
        unit = RadiometricCamera.with_vignetting(none, UnitField{Float64}())
        return (; scene, pose, lens, none, flat, unit, light, st, mask, mat = Material())
    end
    export no_vignetting_scene, NoVignetting, UnitField
end

@testitem "NoVignetting: the render is the unit field's, bit for bit; the calibration refuses it" setup = [RCFixtures, NoVignettingScene] begin
    using RadiometricCamera: compute_mask, flat_vignetting, masked_pixels, pixel_parts, render_view, Φ_pixel
    using StaticArrays
    a = no_vignetting_scene()
    CPU = RadiometricCamera.KernelAbstractions.CPU()
    @test a.none.vignetting === NoVignetting() && VrtkCore.n_control_points(a.flat.vignetting) == 16
    @test count(a.mask) > 0 && compute_mask(a.none, a.pose, a.scene, a.st; δ = 0.02, backend = CPU) == a.mask   # the mask reads no vignetting
    img_none = render_view(1, a.none, a.pose, a.mat, a.light, a.scene, a.st, a.mask)
    img_unit = render_view(1, a.unit, a.pose, a.mat, a.light, a.scene, a.st, a.mask)
    img_flat = render_view(1, a.flat, a.pose, a.mat, a.light, a.scene, a.st, a.mask)
    @test img_none == img_unit && all(img_none[:, :, 1][a.mask] .> 0)                   # bit for bit: dropping the factor 1 changes no bit
    @test img_none ≈ img_flat rtol = 1e-13                                              # the flat spline is 1 to an ulp
    u, v = masked_pixels(a.mask)[end ÷ 2]
    Φ = Φ_pixel(u, v, 1, a.none, a.pose, a.mat, a.light, a.scene, a.st)
    @test Φ === Φ_pixel(u, v, 1, a.unit, a.pose, a.mat, a.light, a.scene, a.st) && Φ == SVector(img_none[v, u, :]...)
    @test Φ ≈ Φ_pixel(u, v, 1, a.flat, a.pose, a.mat, a.light, a.scene, a.st) rtol = 1e-13
    # no number type of its own in the kernel: a Float32 view with NoVignetting is a Float32 flux
    @test RC._vignetting_number_type(NoVignetting()) === Bool && RC._vignetting_number_type(a.flat.vignetting) === Float64
    c32(x) = VgtkCore.convert_eltype(Float32, x)
    v32 = RC.View(c32(a.pose), c32(a.lens), c32(a.none.focus), 0.0386f0)
    Φ32 = RC._Φ_pixel(u, v, 1, v32, NoVignetting(), c32(a.mat), c32(a.light), c32(a.scene), c32(a.st))
    @test Φ32 isa SVector{3, Float32}
    @test Φ32 ≈ Φ rtol = 1e-4
    @test Φ32 === RC._Φ_pixel(u, v, 1, v32, UnitField{Float32}(), c32(a.mat), c32(a.light), c32(a.scene), c32(a.st))
    # no Θ_v: the linear blocks' analytic rows of a pixel have no vignetting columns
    p = pixel_parts(u, v, 1, a.none, a.pose, a.mat, a.light, a.scene, a.st)
    L, G, JP = RC._linear_rows(p, NoVignetting(), VrtkCore.coefficients(a.light), a.mat.diffuse_weight)
    Lf, Gf, JPf = RC._linear_rows(p, a.flat.vignetting, VrtkCore.coefficients(a.light), a.mat.diffuse_weight)
    @test size(JP) == (3, 0) && size(JPf) == (3, 16)
    @test L ≈ Lf rtol = 1e-13                                                         # (the flat spline is 1 to an ulp)
    @test G ≈ Gf rtol = 1e-13
    # the calibration estimates V (Θ_v under the gauge V(𝐜) = 1): a camera without vignetting is refused by name, with the start to give
    targets = [img_flat]
    @test_throws "the camera has no vignetting (`NoVignetting`" CalibrationState(a.scene, a.st, a.none, [a.pose], a.mat, a.light, targets; masks = [a.mask])
    @test_throws "flat_vignetting(geometric(camera))" CalibrationState(a.scene, a.st, a.none, [a.pose], a.mat, a.light, targets; masks = [a.mask])
    s = CalibrationState(a.scene, a.st, RC.with_vignetting(a.none, flat_vignetting(RC.geometric(a.none))), [a.pose], a.mat, a.light, targets; masks = [a.mask])
    @test s.camera.vignetting isa VrtkCore.TensorProductVignetting
end

@testitem "NoVignetting on the device: the render is the unit field's, bit for bit" tags = [:gpu] setup = [RCFixtures, RCGPU, NoVignettingScene] begin
    using RadiometricCamera: render_view
    using CUDA
    require_gpu()                                  # errors with the reason when the GPU the runner chose is not usable here
    a = no_vignetting_scene()
    dev_none = render_view(1, a.none, a.pose, a.mat, a.light, a.scene, a.st, a.mask; backend = CUDABackend())
    dev_unit = render_view(1, a.unit, a.pose, a.mat, a.light, a.scene, a.st, a.mask; backend = CUDABackend())
    dev_flat = render_view(1, a.flat, a.pose, a.mat, a.light, a.scene, a.st, a.mask; backend = CUDABackend())
    @test dev_none == dev_unit                                                        # bit for bit, in the device's Float32
    @test maximum(abs, dev_none - dev_flat) <= 1e-5 * maximum(abs, dev_none)          # the flat spline is 1 to Float32 rounding
    host = render_view(1, a.none, a.pose, a.mat, a.light, a.scene, a.st, a.mask)
    @test maximum(abs, dev_none - host) <= 1e-4 * maximum(abs, host)                  # the CPU's image to Float32 rounding
end
