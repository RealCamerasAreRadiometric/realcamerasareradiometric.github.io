# The radiometric camera as one value (RadiometricCameraModel: lens, focus, aperture, vignetting), its photographs (View) and the
# target's radiometry record (TargetRadiometry).
@testitem "radiometric camera model and target radiometry" setup = [RCFixtures] begin
    using RadiometricCamera: BoardSurface, edge_parameters, Focus, Material, material, TargetRadiometry, View, with_vignetting
    using StaticArrays, LinearAlgebra
    using Statistics: var
    using VisualGeometryToolkitCore: TensorProductVignetting
    # the camera and its photographs: a view is the pose and the camera's parts; the views of one camera rebuild it
    W = H = 24
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    focus = Focus(3, 1.5, W, H)
    vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)
    c = RadiometricCameraModel(lens, focus, 0.0386, vig)
    pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(1.0I), SVector(0.0, 0.0, -1.5); check = false)
    v = View(pose, c)
    @test v.lens === lens && v.focus == focus && v.r == 0.0386 && v.pose == pose
    @test with_vignetting(c, vig).vignetting === vig
    # the camera's in-focus blur is removed: no field, no constructor argument, no name of it
    @test fieldnames(RadiometricCameraModel) == (:lens, :focus, :aperture_radius, :vignetting) && fieldnames(View) == (:pose, :lens, :focus, :r)
    @test !any(n -> isdefined(RC, n), (:InFocusBlur, :NO_BLUR, :in_focus_blur, :blur_sigmas, :with_blur, :with_in_focus_blur, :set_camera!))
    @test RC._shared_camera([v, View(inv(pose), c)], vig) == c
    @test_throws ArgumentError RC._shared_camera([v, View(pose, lens, focus, 0.03)], vig)             # another aperture: not one camera
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
