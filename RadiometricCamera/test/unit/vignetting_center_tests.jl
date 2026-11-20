# The radial profile's centre: the camera's PRINCIPAL POINT, fixed (owner ruling 2026-11-11; the free centre of 2026-11-10 and the
# `pin_center = false` state are deleted) — `vignetting_center` of a parametric camera (K's) and of a SplineLens start (its K's, the axis: the start is
# axis-gauged), the options without a centre, the start field of the options about the principal point on either of Core's charts,
# the pin V(𝐜) = 1 always held, the records.
@testitem "vignetting centre: the principal point, the options, the start field, the gauge, the records" setup = [RCFixtures] begin
    using RadiometricCamera: chart, Focus, FocusSpline, load_radiometric_camera, save_radiometric_camera
    using StaticArrays, LinearAlgebra
    using Test                          # (loads Core's TestExt: thrown_message)
    using VisualGeometryToolkitCore: thrown_message
    using VisualGeometryToolkitCore: center
    using VisualRadiometryToolkitCore: RadialVignetting, TensorProductVignetting, with_control_points, control_points, n_control_points,
                                       field_radius, knot_vectors, radial_chart, projective_knots, with_center
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    W, H = 640, 480
    # 1. the centre of a parametric camera: K's principal point
    kb = cam(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.0, 0.0), 300.0, (331.5, 228.25), W, H)
    @test RC.vignetting_center(kb) == SVector(331.5, 228.25)
    # 2. of a SplineLens start: its K's principal point, which is the optical axis — the start is fitted by Estimation's fit_camera,
    # which turns its source onto the axis (RC 0.11.0): the lattice of a radially symmetric lens about (331.5, 228.25), and a resample
    # of that lattice onto another (a SplineLens source: turned by optical_axis first)
    g = RC.grid_model(kb, 13, 10)
    @test RC.vignetting_center(g) == SVector{2, Float64}(VgtkCore.principal_point(VgtkCore.model(g).intrinsics)...)
    @test RC.vignetting_center(g) ≈ SVector(331.5, 228.25) atol = 0.05
    r = RC.exact_start(g).camera                                                       # a SplineLens source: resampled exactly on its own lattice (m = 1)
    axfit = VgtkEstimation.optical_axis(r)
    @test VgtkCore.converged(axfit)                                                    # the axis converged here: the K tests below hold
    ax = VgtkCore.estimate(axfit)                                                      # the resample's K principal point == its optical axis
    @test RC.vignetting_center(r) ≈ SVector{2, Float64}(VgtkCore.principal_point(ax.intrinsics)...) atol = 0.05
    @test RC.vignetting_center(r) ≈ SVector(331.5, 228.25) atol = 0.05
    # a bare model has no bound; anything else is no camera with a principal point the stack fits: refused with the reason
    @test_throws ArgumentError RC.vignetting_center(VgtkCore.model(kb))
    e = try RC.vignetting_center(:not_a_camera); nothing catch err; err end
    @test e isa ArgumentError && occursin("no principal point", e.msg)

    # 3. the options: no centre keys; the retired ones are unknown keys (Core check_keys lists the allowed)
    o = RC.pipeline_options(Dict("vignetting" => Dict("model" => "radial", "chart" => "projective")))
    @test o.vignetting.chart == :projective && RC.pipeline_options(RC.config_dict(o)) == o
    @test !any(k -> haskey(RC.config_dict(o)["vignetting"], k), ("center", "center_px", "center_bound_px"))
    bad(t) = thrown_message(() -> RC.pipeline_options(Dict("vignetting" => t)))
    for k in ("center", "center_px", "center_bound_px")
        m = bad(Dict("model" => "radial", k => (k == "center" ? "fixed" : k == "center_px" ? [1.0, 2.0] : 100.0)))
        @test occursin("[vignetting]", m) && occursin("unknown key", m) && occursin("\"$k\"", m) && occursin("allowed", m)
    end
    @test occursin("vignetting.chart", bad(Dict("model" => "radial", "chart" => "polar")))
    # the tensor-product surface is no option of the pipeline: its tag is refused by name, and so is its retired size key
    @test occursin("a refinement fits \"radial\" alone", bad(Dict("model" => "tensor_product")))
    @test occursin("`vignetting.size` is retired", bad(Dict("size" => [5, 5])))
    # the non-central start is removed: its key is refused by name with the migration, whatever else the config states
    m = thrown_message(() -> RC.pipeline_options(Dict{String, Any}("frontend" => Dict("noncentral" => "/nc"))))
    @test occursin("`frontend.noncentral` is retired", m) && occursin("central", m) && occursin("delete the key", m)
    @test occursin("origins", thrown_message(() -> RC.pipeline_options(Dict{String, Any}("fixed" => ["origins"]))))   # no such block
    # the region argument is retired with the separate optical-axis fit: the centre is K's, no region of its own
    @test_throws MethodError RC.vignetting_center(kb, nothing)

    # 4. the start field of the options: about the given centre (the principal point), r_max the farthest point of the region from
    # it, on Core's chart (the projective one Core's own profile, its knots Core's `projective_knots`)
    Ws, Hs = 4256, 2848
    c = SVector(2126.0, 1554.0)
    ol = RC.pipeline_options(Dict("vignetting" => Dict("model" => "radial", "node_count" => 16, "chart" => "linear")))
    # the projective chart at 19 nodes (Core 0.10.11's knots, T. Harder): 16 functions act on the fitted pixels, the linear 16's spacing
    op = RC.pipeline_options(Dict("vignetting" => Dict("model" => "radial", "node_count" => 19, "chart" => "projective")))
    vl = RC._vignetting_spec(ol, Ws, Hs, nothing, c)[1]()
    mkp, dp = RC._vignetting_spec(op, Ws, Hs, nothing, c); vp = mkp()
    @test vl isa RadialVignetting && radial_chart(vl) == :linear && center(vl) == c
    @test field_radius(vl) ≈ maximum(hypot(x - c[1], y - c[2]) for x in (0, Ws), y in (0, Hs))     # no box: r_max is the region's alone
    @test vp == RadialVignetting(19, c, field_radius(vl); chart = :projective) && radial_chart(vp) == :projective
    @test collect(knot_vectors(vp)[1]) == collect(projective_knots(19, 3)) && occursin("principal point", dp) && occursin("projective", dp)
    # the defaults of 0.11.4 (configs/default.toml): the projective chart, 19 nodes
    @test RC.PipelineOptions().vignetting.chart === :projective && RC.PipelineOptions().vignetting.node_count == 19
    @test !occursin("FREE", dp)
    @test RC.PipelineOptions().vignetting.model === RadialVignetting && !hasfield(RC.VignettingOptions, :size)   # the one field the pipeline builds

    # 5. the pin V(𝐜) = 1 is always held: no `pin_center` keyword, the state's start normalised, the free directions without the
    # centre's, the block without centre parameters
    v16 = with_control_points(RadialVignetting(16, c, 2560.5), [1.2 - 0.05 * i for i in 0:15])
    Z = RC.gauge_basis(v16, RC.vignetting_basis(v16, nothing))
    @test size(Z, 2) == size(RC.vignetting_basis(v16, nothing), 2) - 1 && all(iszero, Z[1, :])     # V(𝐜) = P₀ never moves
    vv = VgtkCore.restrict(VgtkCore.ParameterManifold(v16), Z)        # the block: Core's LinearSubspaceManifold
    @test Int(RC.NLLSsolver.nvars(vv)) == size(Z, 2) == VgtkCore.tangent_size(vv) && !hasfield(typeof(vv), :bound)
    u = zeros(size(Z, 2)); u[3] = 0.01
    @test center(VgtkCore.model(RC.NLLSsolver.update(vv, u))) == c && VgtkCore.model(RC.NLLSsolver.update(vv, u))(c) == v16(c)
    @test !any(m -> any(k -> k === :pin_center || k === :center_free, Base.kwarg_decl(m)), methods(RC.CalibrationState))
    @test !hasfield(RC.CalibrationState, :pin_center) && !hasfield(RC.CalibrationState, :vig_center_free)
    @test RC.gauge_center_value(v16)(c) ≈ 1.0

    # 6. _same_field: the centre and the chart count
    @test RC._same_field(v16, v16) && !RC._same_field(v16, with_center(v16, c + SVector(30.0, 0.0))) &&
          !RC._same_field(v16, RadialVignetting(16, c, 2560.5; chart = :projective))

    # 7. the records: Core's projective profile in a radiometric camera round-trips through IO's body
    mktempdir() do dir
        lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), 24, 24)
        focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, 24, 24))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
        vpp = with_control_points(vp, [1.0 - 0.04 * (i - 1)^1.2 for i in 1:19])
        for v in (vpp, v16)
            camera = RadiometricCameraModel(lens, focus, 0.0386, v)
            f = joinpath(dir, "camera.json"); save_radiometric_camera(f, RC.CalibratedCamera(camera, RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB), RC.RenderSettings()))
            @test load_radiometric_camera(f).model.vignetting == v
        end
    end
end
