# The radiometric camera as one value (RadiometricCameraModel: lens, focus, aperture, vignetting), its photographs (View) and the
# target's radiometry record (TargetRadiometry).
@testitem "radiometric camera model and target radiometry" setup = [RCFixtures] begin
    using RadiometricCamera: BoardSurface, edge_parameters, Focus, Material, material, TargetRadiometry, View, with_vignetting
    using StaticArrays, LinearAlgebra
    using Statistics: var
    using VisualGeometryToolkitCore: TensorProductVignetting
    # the camera and its photographs: the kernels' (internal) view is the camera at a pose, its parts flattened
    W = H = 24
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    focus = Focus(3, 1.5, W, H)
    vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)
    c = RadiometricCameraModel(lens, focus, 0.0386, vig)
    pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(1.0I), SVector(0.0, 0.0, -1.5); check = false)
    v = View(pose, c)
    @test v.lens === lens && v.focus == focus && v.r == 0.0386 && v.pose == pose
    @test with_vignetting(c, vig).vignetting === vig
    # the camera composed from its geometric camera: the focus surface (a distance: focused there everywhere), the aperture radius and
    # the vignetting (V ≡ 1 by default) added; the image size is the geometric camera's, stated once
    @test RadiometricCameraModel(lens; focus = 1.5, aperture_radius = 0.0386) === c
    @test RadiometricCameraModel(lens; focus, aperture_radius = 0.0386, vignetting = vig) === c
    @test Focus(lens, 1.5) == focus && RC.n_nodes(Focus(lens, 2.0; nodes = 4)) == 16 && RC.𝔱(Focus(lens, 2.0; nodes = 4), SVector(5.0, 7.0)) ≈ 2.0
    @test_throws UndefKeywordError RadiometricCameraModel(lens; focus = 1.5)          # no default aperture: r is a physical statement
    @test isbits(c) && isbits(v)                                                        # the camera and the kernels' view travel into a device kernel as they are
    # its parts read and replaced one at a time: the others are kept
    @test RC.lens(c) === lens && RC.focus(c) === focus && RC.aperture_radius(c) == 0.0386
    f2 = Focus(lens, 2.0); c2 = RC.with_aperture_radius(RC.with_focus(c, f2), 0.01)
    @test c2.focus === f2 && c2.aperture_radius == 0.01 && c2.lens === lens && c2.vignetting === vig
    lens2 = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.3, 0.4)), 30.912, (12.24, 11.52), W, H)
    @test RC.with_lens(c, lens2).lens === lens2 && RC.with_lens(c, lens2).focus === focus
    # the camera's in-focus blur is removed: no field, no constructor argument, no name of it
    @test fieldnames(RadiometricCameraModel) == (:lens, :focus, :aperture_radius, :vignetting) && fieldnames(View) == (:pose, :lens, :focus, :r)
    @test !any(n -> isdefined(RC, n), (:InFocusBlur, :NO_BLUR, :in_focus_blur, :blur_sigmas, :with_blur, :with_in_focus_blur, :set_camera!))
    @test !isdefined(RC, :_shared_camera) && !Base.ispublic(RC, :View) && !Base.ispublic(RC, :view_of)   # one camera, never parts
    @test RC.𝔣(c, SVector(5.0, 7.0)) == RC.𝔣(v, SVector(5.0, 7.0)) && RC.t̂(c, SVector(5.0, 7.0)) == RC.t̂(v, SVector(5.0, 7.0))
    @test RC.world_ray(c, pose, SVector(5.0, 7.0), SVector(0.01, 0.0, 0.0)) == RC.world_ray(v, SVector(5.0, 7.0), SVector(0.01, 0.0, 0.0))
    @test VgtkCore.project(c, SVector(0.1, 0.0, 1.0)) == VgtkCore.project(lens, SVector(0.1, 0.0, 1.0))
    # the target's radiometry: the material and the print's edge, lengths as quantities, the state's mm values back bit for bit
    m = Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.6)
    surf = BoardSurface(edge = SVector(0.6, 0.028))
    r = TargetRadiometry(m, surf)
    @test material(r) == m && edge_parameters(r) == surf.edge
    @test VgtkCore.ustrip(VgtkCore.m, r.ink_spread) ≈ 6e-4 && var(r.print_blur)[1] ≈ (2.8e-5 * VgtkCore.m)^2
    # its lengths are quantities: a plain ink spread, a plain (px²) print blur or a pixel quantity for a length is refused
    q = 2.8e-5 * VgtkCore.m
    @test_throws "the ink spread is a length" TargetRadiometry(m, 6e-4, VgtkCore.GaussianBlur(q))
    @test_throws "the ink spread is a length" TargetRadiometry(m, 6e-4 * VgtkCore.px, VgtkCore.GaussianBlur(q))
    @test_throws "with_unit(blur, m)" TargetRadiometry(m, 6e-4 * VgtkCore.m, VgtkCore.GaussianBlur(2.8e-5))
end
