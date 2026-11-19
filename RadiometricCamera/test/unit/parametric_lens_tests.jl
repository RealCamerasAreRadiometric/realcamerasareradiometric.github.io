# The parametric models are Core's pixel cameras (a BoundedCamera of a CameraModel, K in pixels) over the four named Core lens families,
# and every number they compute is Core's. Reference: the lens on the normalised point K⁻¹𝐩 written out here.
@testitem "pixel cameras over Core lenses" setup = [RCFixtures] begin
    using RadiometricCamera: lm_box
    using StaticArrays, LinearAlgebra, ForwardDiff
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    W, H = 640, 480
    cv = [0.1, -0.05, 1e-3, -2e-3, 0.01, 0.02, 0.0, 0.0, 1e-3, 0.0, -1e-3, 0.0, 0.01, -0.005]
    c0 = (326.4, 230.4)
    lenses = (cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 824.32, c0, W, H; f_y = 600.0), cam(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.002, -0.001), 576.0, c0, W, H; f_y = 456.0),
              cam(VgtkCore.Division(0.144, -0.293), 864.0, c0, W, H; f_y = 364.8), cam(VgtkIO.opencv_lens(cv), 824.32, c0, W, H; f_y = 600.0))
    types = (VgtkCore.Lens{VgtkCore.DoubleSphere{Float64}, Nothing}, VgtkCore.KannalaBrandt{4, Float64}, VgtkCore.Division{2, Float64}, typeof(VgtkIO.opencv_lens(cv)))   # the Core families
    ps = [SVector(x, y) for x in 32.0:57.6:608.0, y in 24.0:43.2:456.0]
    _θ(l) = vcat(pinhole_θ(l), VgtkCore.parameter_vector(VgtkCore.lens(l)))     # `camera_of`'s vector
    for (l, T) in zip(lenses, types)
        @test has_lens(l, T) && isbitstype(typeof(l))
        # the parameters are Core's: the camera's f, aspect, skew, cx, cy of K (pixels), then the lens's coefficients
        m = VgtkCore.model(l); θ = VgtkCore.parameter_vector(m); fx, fy, cx, cy = pinhole_θ(l)
        @test VgtkCore.parameter_names(m)[1:5] == (:f, :aspect, :skew, :cx, :cy)
        @test θ[1:5] ≈ SVector(fx, fy / fx, 0.0, cx, cy) && θ[6:end] == VgtkCore.parameter_vector(VgtkCore.lens(l))
        @test has_lens(VgtkCore.with_model(l, VgtkCore.with_parameters(m, θ)), T)
        @test maximum(abs, _θ(VgtkCore.with_model(l, VgtkCore.with_parameters(m, θ))) - _θ(l)) < 1e-12
        # π⁻¹ and project against the lens on the normalised point written out here, ((𝐩 − c)/f): the same rays, to rounding
        n = 0
        for 𝐩 in ps
            d = RC._chief_ray(l, 𝐩); all(isfinite, d) || continue
            n += 1
            mm = SVector((𝐩[1] - cx) / fx, (𝐩[2] - cy) / fy)
            @test norm(d - normalize(SVector{3}(VgtkCore.backproject(VgtkCore.lens(l), mm)))) < 1e-14
            q = VgtkCore.project(VgtkCore.lens(l), d)
            @test norm(VgtkCore.project(l, d) - SVector(fx * q[1] + cx, fy * q[2] + cy)) < 1e-10
            @test norm(VgtkCore.project(l, 3.7 * d) - 𝐩) < 1e-7
        end
        @test n > length(ps) ÷ 2
        # the LM box: the solver option by name, inside the model's domain (Core's parameter_bounds)
        lo, hi = VgtkCore.parameter_box(VgtkCore.ParameterManifold(m; box = lm_box(m, W, H)))
        @test all(lo .<= θ .<= hi) && length(lo) == length(θ)
        dlo, dhi = VgtkCore.parameter_bounds(m)
        @test all(lo .>= dlo) && all(hi .<= dhi)
        # the lens block: Core's ParameterManifold of the camera on every parameter but the skew, with the LM's box and the image-size
        # coordinate scale; a step is Core's recenter (cx by W per coordinate), and a step past the box stops on it (cx at 0.8 W)
        v = RC.lens_manifold(l)
        @test v isa VgtkCore.SubsetManifold && free_names(v) == filter(!=(:skew), VgtkCore.parameter_names(m)) && Int(RC.NLLSsolver.nvars(v)) == length(θ) - 1
        @test VgtkCore.coordinate_scale(v.manifold)[4:5] == SVector(W, H)
        u = zeros(length(θ) - 1); u[1] = 0.01; u[3] = 0.02
        v2 = RC.NLLSsolver.update(v, u)
        @test VgtkCore.model(v2) == VgtkCore.model(VgtkCore.recenter(v, SVector{length(u)}(u)))
        @test pinhole_θ(block_camera(v2, l))[3] ≈ cx + 0.02W
        u[3] = 1.0
        @test pinhole_θ(block_camera(RC.NLLSsolver.update(v, u), l))[3] == 0.8W
        @test free_names(RC.lens_manifold(l; fixed = (:aspect,))) == filter(∉((:skew, :aspect)), VgtkCore.parameter_names(m))
        @test_throws ArgumentError RC.lens_manifold(l; fixed = (:f, :nope))
        # element types: Float32 (the device) and duals (the LM Jacobian: the seeded coordinates against central differences of the update)
        l32 = VgtkCore.convert_eltype(Float32, l)
        @test has_lens(l32, VgtkCore.AbstractLensModel) && eltype(pinhole_θ(l32)) == Float32 &&
              norm(RC._chief_ray(l32, SVector{2, Float32}(384.0, 192.0)) - RC._chief_ray(l, SVector(384.0, 192.0))) < 1e-5
        M = length(free_names(v)); 𝐩 = SVector(390.4, 206.4)
        J = ForwardDiff.jacobian(z -> RC._chief_ray(RC._seeded_lens(v, l, x -> x + zero(eltype(z)), j -> z[j]), 𝐩), zeros(M))
        e(i) = (z = zeros(M); z[i] = 1e-6; z)
        Jfd = hcat([(RC._chief_ray(block_camera(RC.NLLSsolver.update(v, e(i)), l), 𝐩) - RC._chief_ray(block_camera(RC.NLLSsolver.update(v, -e(i)), l), 𝐩)) / 2e-6 for i in 1:M]...)
        @test maximum(abs, J - Jfd) < 1e-7 * max(1.0, maximum(abs, Jfd))
    end
    # OpenCV's convention is VisualGeometryToolkitIO's: cv2's vector and pixel K back out exactly (opencv_parameters), and opencv_camera agrees
    l14 = lenses[4]
    pp = VgtkIO.opencv_parameters(VgtkCore.model(l14))
    @test pp.dist ≈ cv && maximum(abs, _θ(VgtkCore.BoundedCamera(VgtkIO.opencv_camera(pp.K, pp.dist), VgtkCore.Size2(W, H))) - _θ(l14)) < 1e-12
    @test camera_of(VgtkIO.opencv_lens(zeros(14)), _θ(l14), W, H) == l14
    # a camera from (f_x, f_y, c_x, c_y, coefficients…) and its family
    @test camera_of(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), _θ(lenses[1]), W, H) == lenses[1] && camera_of(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), _θ(lenses[2]), W, H) == lenses[2] &&
          camera_of(VgtkCore.Division(0.0, 0.0), _θ(lenses[3]), W, H) == lenses[3]
    # any Core lens is the same kind of camera: an equidistant fisheye, fixed in the per-view LM
    le = cam(VgtkCore.Lens(VgtkCore.Equidistant()), 900.0, (640.0, 480.0), 1280, 960)
    @test pinhole_θ(le) == SVector(900.0, 900.0, 640.0, 480.0) && last(RC.lens_free(le; fixed = VgtkCore.parameter_names(VgtkCore.model(le))))
end
