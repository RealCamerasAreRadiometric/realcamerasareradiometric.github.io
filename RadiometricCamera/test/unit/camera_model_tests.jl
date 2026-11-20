# The radiometric camera as one value (RadiometricCameraModel: geometric camera, focus, aperture, vignetting), its photographs (View) and the
# target's radiometry record (TargetRadiometry).
@testitem "radiometric camera model and target radiometry" setup = [RCFixtures] begin
    using RadiometricCamera: TargetState, edge_parameters, flat_vignetting, Focus, Material, material, TargetRadiometry, View, with_vignetting
    using StaticArrays, LinearAlgebra
    using Statistics: var
    using VisualRadiometryToolkitCore: TensorProductVignetting, NoVignetting
    # the camera and its photographs: the kernels' (internal) view is the camera at a pose, its parts flattened
    W = H = 24
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    focus = Focus(3, 1.5, W, H)
    vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)
    c = RadiometricCameraModel(lens, focus, 0.0386, vig)
    pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(1.0I), SVector(0.0, 0.0, -1.5); check = false)
    v = View(pose, c)
    @test v.geometric === lens && v.focus == focus && v.r == 0.0386 && v.pose == pose
    @test with_vignetting(c, vig).vignetting === vig
    # (the camera — VisualRadiometryToolkitCore's RadiometricCameraModel, its composed constructor, its parts and their replacers — is
    # tested there.) The camera composed from its geometric camera has NO vignetting unless one is stated; the calibration's start of
    # the vignetting, the flat estimable surface, is asked for by name
    @test RadiometricCameraModel === VrtkCore.RadiometricCameraModel
    @test RadiometricCameraModel(lens; focus = 1.5, aperture_radius = 0.0386).vignetting === NoVignetting()
    @test flat_vignetting(lens) === vig
    @test RadiometricCameraModel(lens; focus = 1.5, aperture_radius = 0.0386, vignetting = flat_vignetting(lens)) === c
    @test isbits(c) && isbits(v)                                                        # the camera and the kernels' view travel into a device kernel as they are
    lens2 = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.3, 0.4)), 30.912, (12.24, 11.52), W, H)
    # `lens` is Core's lens model everywhere: RC's old spelling of the replacer is refused by name with the replacement
    @test_throws "with_lens is retired" RC.with_lens(c, lens2)
    @test !isdefined(RC, :lens) && !Base.ispublic(RC, :with_lens)
    # what moved to VisualRadiometryToolkitCore is not a public name here; a moved name this package does not use is refused by name
    @test !Base.ispublic(RC, :geometric) && !Base.ispublic(RC, :Focus) && Base.ispublic(RC, :flat_vignetting) && Base.isexported(RC, :RadiometricCameraModel)
    @test RC.geometric === VrtkCore.geometric && RC.Focus === VrtkCore.Focus && RC.Material === VrtkCore.Material
    for n in (:with_geometric, :with_focus, :with_aperture_radius, :pixel_of_𝐩, :ℐ_𝐮, :aperture_samples, :aperture_border, :Λ_ggx, :G1_ggx, :D_ggx, :G2_ggx, :F_dielectric, :D_v, :luma, :n_nodes, :𝔱,
              :chief_dir, :sample_vndf_caps, :PDF_reflect, :f_CT_parts, :f_CT, :t̂_dir, :ae_chart, :ae_dir)
        @test_throws "moved to VisualRadiometryToolkitCore" getfield(RC, n)(1)
    end
    # VisualRadiometryToolkitCore's public names are descriptive; the refusal names the one to call, and the paper's symbols this
    # package's source keeps are its aliases of the same functions (π̂⁻¹ is VisualGeometryToolkitCore's `backproject`)
    @test_throws "`chief_direction`" RC.chief_dir(1)
    @test_throws "`cook_torrance`" RC.f_CT(1)
    @test_throws "`backproject(Lens(Equidistant()), ξ)`" RC.ae_dir(1)
    @test RC.𝔣 === VrtkCore.focus_point && RC.t̂ === VrtkCore.inverse_focus_distance && RC.π̂⁻¹ === VgtkCore.backproject && RC.𝔭 === VrtkCore.center_of_projection
    @test RC.π̂⁻¹(v, SVector(5.0, 7.0), SVector(0.001, 0.0, 0.0)) === VrtkCore.aperture_direction(v.geometric, v.focus, SVector(5.0, 7.0), SVector(0.001, 0.0, 0.0))
    # the camera's in-focus blur is removed: no field, no constructor argument, no name of it
    @test fieldnames(RadiometricCameraModel) == (:geometric, :focus, :aperture_radius, :vignetting) && fieldnames(View) == (:pose, :geometric, :focus, :r)
    @test !any(n -> isdefined(RC, n), (:InFocusBlur, :NO_BLUR, :in_focus_blur, :blur_sigmas, :with_blur, :with_in_focus_blur, :set_camera!))
    @test !isdefined(RC, :_shared_camera) && !Base.ispublic(RC, :View) && !Base.ispublic(RC, :view_of)   # one camera, never parts
    @test RC.𝔣(c, SVector(5.0, 7.0)) == RC.𝔣(v, SVector(5.0, 7.0)) && RC.t̂(c, SVector(5.0, 7.0)) == RC.t̂(v, SVector(5.0, 7.0))
    @test RC.world_ray(c, pose, SVector(5.0, 7.0), SVector(0.01, 0.0, 0.0)) == RC.world_ray(v, SVector(5.0, 7.0), SVector(0.01, 0.0, 0.0))
    # the view's maps are VisualRadiometryToolkitCore's implementation on the view's parts (one-line forwards), bit for bit
    let 𝐩 = SVector(5.0, 7.0), 𝐚 = SVector(0.01, 0.0, 0.0)
        @test RC.chief_direction(v, 𝐩) === RC.chief_direction(lens, 𝐩) && RC.π̂⁻¹(v, 𝐩, 𝐚) === RC.π̂⁻¹(c, 𝐩, 𝐚) === RC.π̂⁻¹(v, 𝐩, 𝐚, RC.t̂(v, 𝐩))
        @test RC.world_ray(v, 𝐩, 𝐚, RC.t̂(v, 𝐩)) == RC.world_ray(v, 𝐩, 𝐚) == RC.world_ray_dir(v, 𝐚, RC.chief_direction(v, 𝐩), RC.t̂(v, 𝐩))
    end
    # the target's radiometry: the material and the print's edge, lengths as quantities, the state's mm values back bit for bit
    m = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6)
    surf = TargetState(edge = SVector(0.6, 0.028))
    r = TargetRadiometry(m, surf)
    @test material(r) == m && edge_parameters(r) == surf.edge
    @test VgtkCore.ustrip(VgtkCore.m, r.ink_spread) ≈ 6e-4 && var(r.print_blur)[1] ≈ (2.8e-5 * VgtkCore.m)^2
    # its lengths are quantities: a plain ink spread, a plain (px²) print blur or a pixel quantity for a length is refused
    q = 2.8e-5 * VgtkCore.m
    @test_throws "the ink spread is a length" TargetRadiometry(m, 6e-4, VgtkCore.GaussianBlur(q))
    @test_throws "the ink spread is a length" TargetRadiometry(m, 6e-4 * VgtkCore.px, VgtkCore.GaussianBlur(q))
    @test_throws "with_unit(blur, m)" TargetRadiometry(m, 6e-4 * VgtkCore.m, VgtkCore.GaussianBlur(2.8e-5))
end
