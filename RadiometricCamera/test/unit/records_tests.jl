# RadiometricCamera's records (src/io/records.jl): rc/radiometric_camera/v3.1, rc/radiometric_target/v2.0 and
# rc/calibration_state/v3.1, every one through VisualGeometryToolkitIO's file boundary. A value comes back as it was written: bit for bit,
# except a length (metres in memory, the file's mm on disk: ≤ 2 ulp at full precision) and a pose (IO's minimal form, a quaternion).

@testitem "records: a calibration state, its radiometric camera and target read back as written" setup = [RCFixtures] begin
    using RadiometricCamera: board_surface, BoardSurface, edge_parameters, Focus, FocusSpline, HEIGHT_NC,
                             load_radiometric_camera, load_radiometric_target, load_state, Material, posed_target, PrintedTarget,
                             RECORD_SCHEMAS, RenderSettings, save_radiometric_camera, save_radiometric_target, save_state,
                             TargetRadiometry, warping_aspect, 𝔟
    using StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, RadialVignetting, with_control_points
    tex = fixture_board()
    board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    vig = let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2), g = range(-0.5, 0.5, length = 4)
        with_control_points(v, [1.0 - 0.4 * (g[i]^2 + g[j]^2) for i in 1:4, j in 1:4])
    end
    camera = RadiometricCameraModel(lens, focus, 0.0386, vig)
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                                   SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k); check = false) for k in 1:2]
    a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2)
    light = EnvironmentLight(SphericalHarmonicMap(a, 2), 𝔟(board), 2.5)
    rad = TargetRadiometry(Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4), BoardSurface(; edge = [0.021, 0.034]))
    target = RadiometricTarget(VgtkCore.model(posed_target(board)), SVector{HEIGHT_NC}(1e-4 .* sin.(1:HEIGHT_NC)), warping_aspect(board), rad)
    settings = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4, seed = 7)
    ulps(a, b) = maximum(abs.(a .- b) ./ eps.(abs.(b)))
    raw = RC.ObservationModel(RC.SensorRecord(repeat("ab", 32)), RC.Photosites(), :G)     # a calibration on RAW photosites under a record
    mktempdir() do dir
        f = joinpath(dir, "state.json")
        save_state(f, (; camera, target, poses, names = ["V01.png", "V02.png"], light, settings, observation = raw, page_flip = -1); provenance = (; iteration = 3))
        st = load_state(f)
        @test st.observation == raw && RC.calibrated_camera(st) isa RC.CalibratedCamera
        @test VgtkIO.read_record(f).schema == RECORD_SCHEMAS.calibration_state
        @test st.names == ["V01.png", "V02.png"] && st.page_flip === -1.0
        @test all(norm(st.poses[k].R - poses[k].R) < 1e-15 && st.poses[k].t == poses[k].t for k in 1:2)
        c = st.camera
        @test c.geometric == lens && c.focus == focus && c.vignetting == vig
        @test ulps(c.aperture_radius, 0.0386) <= 2
        @test coefficients(st.light) == coefficients(light) && ulps(VgtkCore.radius(st.light), 2.5) <= 2
        @test st.settings == settings
        t = st.target
        @test t.radiometry.material == rad.material && edge_parameters(t.radiometry) == SVector(0.021, 0.034)
        @test ulps(t.warping, target.warping) <= 2
        @test VgtkCore.reference(t.target) == VgtkCore.reference(target.target)                    # IO's target record
        @test t.aspect === 1.0 && board_surface(board, t).edge == SVector(0.021, 0.034) && board_surface(board, t).h == t.warping
        # the radiometric camera and target on their own: the camera a complete calibration (its observation model and settings)
        save_radiometric_camera(joinpath(dir, "camera.json"), RC.CalibratedCamera(camera, raw, settings))
        cc = load_radiometric_camera(joinpath(dir, "camera.json"))
        @test cc.model.vignetting == vig && cc.observation == raw && cc.settings == settings
        save_radiometric_target(joinpath(dir, "target.json"), RadiometricTarget(nothing, target.warping, target.aspect, rad))
        @test load_radiometric_target(joinpath(dir, "target.json")).target === nothing
        @test load_radiometric_target(joinpath(dir, "target.json")).aspect === 1.0
        # the schema is checked
        @test_throws ArgumentError load_radiometric_camera(f)
        dec = RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(true), :RGB)
        save_state(f, (; camera, target, poses, names = ["V01.png", "V02.png"], light, settings = RenderSettings(), observation = dec, page_flip = nothing))
        @test load_state(f).page_flip === nothing && load_state(f).settings == RenderSettings() && load_state(f).observation == dec
        @test_throws TypeError save_state(f, (; camera, target, poses, names = ["V01.png", "V02.png"], light, settings, observation = nothing, page_flip = nothing))
        # the camera's in-focus blur is removed: no record holds one any more (neither the camera body nor the settings' sensor rule)
        text = read(f, String)
        @test !occursin("in_focus_blur", text) && !occursin("in_focus_blur", read(joinpath(dir, "camera.json"), String))
        edited(a, b) = (occursin(a, text) || error("no $a in the state"); p = joinpath(dir, "edited.json"); write(p, replace(text, a => b; count = 1)); p)
        # the print blur's plane and the pixel frame are checked on read: the print's blur on the print (a length squared), the models
        # in COLMAP pixels
        @test_throws "print_blur: a blur on the print is on the board's plane (\"frame\": \"target:board\"), got the pixel frame colmap" load_state(edited("\"frame\": \"target:board\"", "\"pixel_frame\": \"colmap\""))
        @test_throws "the pixel frame \"opencv\"" load_state(edited("\"pixel\": \"colmap\"", "\"pixel\": \"opencv\""))
        save_radiometric_camera(joinpath(dir, "camera.json"), RC.CalibratedCamera(camera, raw, settings))
        text = read(joinpath(dir, "camera.json"), String)
        @test_throws "the pixel frame \"opencv\"" load_radiometric_camera(edited("\"pixel\": \"colmap\"", "\"pixel\": \"opencv\""))
    end
end

@testitem "records: a target's warping knows its page's aspect; another page's board refuses it" setup = [RCFixtures] begin
    using RadiometricCamera: board_surface, BoardSurface, Focus, FocusSpline, HEIGHT_NC, load_radiometric_target, load_state,
                             Material, PrintedTarget, radiometric_target, RenderSettings, save_radiometric_target, save_state,
                             Scene, TargetRadiometry, warping_aspect, 𝔟
    using StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting
    tex = fixture_board()
    square = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    tall = PrintedTarget([-1.0, -0.7, 0.0], [2.0, 0.0, 0.0], [0.0, 1.4, 0.0], tex; facing = [0.0, 0.0, 1.5])       # aspect 0.7
    @test warping_aspect(square) == 1.0 && warping_aspect(tall) ≈ 0.7
    rad = TargetRadiometry(Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4), BoardSurface(; edge = [0.021, 0.034]))
    h = SVector{HEIGHT_NC}(1e-4 .* sin.(1:HEIGHT_NC))
    warped = RadiometricTarget(nothing, h, 1.0, rad)                    # learned on a square page
    flat = RadiometricTarget(nothing, zero(h), nothing, rad)              # a flat warping: aspect-free
    # the warped target applies to a board of its aspect, and is refused on another; the flat one applies to any board
    @test board_surface(square, warped).h == h
    @test_throws "learned on a page of aspect" board_surface(tall, warped)
    @test_throws "learned on a page of aspect" board_surface(warping_aspect(tall), warped)
    @test board_surface(tall, flat).h == zero(h) && board_surface(square, flat).h == zero(h)
    @test board_surface(tall, RadiometricTarget(nothing, zero(h), 1.0, rad)).h == zero(h)       # flat, whatever aspect it states
    # a warped target states its aspect; an aspect is positive and finite
    @test_throws "needs the aspect" RadiometricTarget(nothing, h, nothing, rad)
    @test_throws ArgumentError RadiometricTarget(nothing, h, -1.0, rad)
    mktempdir() do dir
        # the record keeps the aspect (a converted state's target is `nothing`, as here), and a warped record without one is refused
        f = joinpath(dir, "target.json")
        save_radiometric_target(f, warped)
        @test load_radiometric_target(f).aspect === 1.0
        @test_throws "learned on a page of aspect" board_surface(tall, load_radiometric_target(f))
        text = read(f, String)
        g = joinpath(dir, "no_aspect.json"); write(g, replace(text, r"\"warping_aspect\": [^,\n]+,?\n?" => ""))
        @test !occursin("warping_aspect", read(g, String))
        @test_throws "needs the aspect" load_radiometric_target(g)
        save_radiometric_target(f, flat)
        @test !occursin("warping_aspect", read(f, String)) && load_radiometric_target(f).aspect === nothing
        # load_state_into!: a state learned on the square page, loaded into a run on the tall one, is refused (its material and
        # warping untouched)
        W = H = 24
        lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
        focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
        vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)
        camera = RadiometricCameraModel(lens, focus, 0.0386, vig)
        poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                                       SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k); check = false) for k in 1:2]
        a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8)
        light = EnvironmentLight(SphericalHarmonicMap(a, 2), 𝔟(tall), 2.5)
        settings = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
        names = ["V01.png", "V02.png"]
        dec = RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB)          # a state of decoded renders, as `state_on`'s
        save_state(joinpath(dir, "state.json"), (; camera, target = warped, poses, names, light, settings, observation = dec, page_flip = nothing))
        sst = load_state(joinpath(dir, "state.json"))
        mat0 = Material(; 𝛂_ink = 0.2, 𝛂_noink = 0.3, 𝛈_ink = 1.5, 𝛈_noink = 1.4, μ = 0.5)
        state_on(board) = CalibrationState(Scene(board, W, H), settings, camera, poses, mat0, light, [zeros(H, W, 3) for _ in 1:2])
        s = state_on(tall)
        @test_throws "learned on a page of aspect" RC.load_state_into!(s, sst, names)
        @test s.mat == mat0 && iszero(board_surface(s).h)
        s1 = state_on(square)
        RC.load_state_into!(s1, sst, names)
        @test board_surface(s1).h == sst.target.warping && radiometric_target(s1).aspect === 1.0
    end
end

@testitem "records: a target's in-plane print error is IO's calibration record of Aspect ∘ Skew" setup = [RCFixtures] begin
    using RadiometricCamera: board_surface, BoardSurface, HEIGHT_NC, load_radiometric_target, Material, PrintedTarget,
                             save_radiometric_target, TargetRadiometry
    using StaticArrays, LinearAlgebra
    tex = fixture_board()
    square = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    tall = PrintedTarget([-1.0, -0.7, 0.0], [2.0, 0.0, 0.0], [0.0, 1.4, 0.0], tex; facing = [0.0, 0.0, 1.5])       # aspect 0.7
    rad = TargetRadiometry(Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4), BoardSurface(; edge = [0.021, 0.034]))
    h = SVector{HEIGHT_NC}(1e-4 .* sin.(1:HEIGHT_NC))
    θ = SVector(0.0123456789, -0.00098765)
    # the record holds (a, u); a flat, as-designed target has none; the error needs the page's aspect as the warping does; it is finite
    printed = RadiometricTarget(nothing, zero(h), 1.0, rad; inplane = θ)                          # learned on a square page
    @test printed.inplane == θ && RadiometricTarget(nothing, zero(h), nothing, rad).inplane == zero(θ)
    @test RadiometricTarget(nothing, h, 1.0, rad; inplane = [0.01, 0.0]).inplane == SVector(0.01, 0.0)
    @test_throws "needs the aspect" RadiometricTarget(nothing, zero(h), nothing, rad; inplane = θ)
    @test_throws "must be finite" RadiometricTarget(nothing, zero(h), 1.0, rad; inplane = [NaN, 0.0])
    # on a board: the surface carries it; a board of another aspect refuses it (its placed chart is another map there)
    @test board_surface(square, printed).inplane == θ && board_surface(square, printed).edge == SVector(0.021, 0.034) && iszero(board_surface(square, printed).h)
    @test_throws "learned on a page of aspect" board_surface(tall, printed)
    @test iszero(board_surface(tall, RadiometricTarget(nothing, zero(h), 1.0, rad)).inplane)   # as designed: any board
    mktempdir() do dir
        f = joinpath(dir, "target.json")
        save_radiometric_target(f, printed)
        text = read(f, String)
        @test occursin("\"inplane_print_error\"", text) && occursin("\"composition\"", text) && occursin("\"aspect\"", text) && occursin("\"skew\"", text)
        rec = VgtkIO.read_record(f)
        ipe = rec.inplane_print_error
        @test ipe.type == "composition" && [m.type for m in ipe.compose] == ["aspect", "skew"]       # IO's target-calibration record, outer first
        @test ipe.compose[1].parameters == [θ[1]] && ipe.compose[2].parameters == [θ[2]]              # the parameters bit for bit
        t = load_radiometric_target(f)
        @test t.inplane == θ && t.aspect === 1.0 && iszero(t.warping)
        save_radiometric_target(f, RadiometricTarget(nothing, zero(h), 1.0, rad))
        @test !occursin("inplane_print_error", read(f, String)) && iszero(load_radiometric_target(f).inplane)   # as designed: no key
        # another kind, order or arity of the record is refused
        bad(s) = (g = joinpath(dir, "bad.json"); write(g, replace(text, s)); g)
        @test_throws "canonical order" load_radiometric_target(bad("\"skew\"" => "\"scale\""))   # IO 0.9.5's reader: aspect ∘ scale is out of order
        @test_throws "aspect ∘ skew" load_radiometric_target(bad("\"aspect\"" => "\"scale\""))   # a valid record, another composition: RC's refusal
        st = VgtkIO.VgioStyle(; length_unit = VgtkCore.mm)
        @test_throws Exception RC._inplane_from_body(st, (type = "composition", compose = [(type = "aspect", parameters = [θ[1]]), (type = "skew", parameters = [θ[2], 0.0])]))
        @test_throws "canonical order" RC._inplane_from_body(st, (type = "composition", compose = [(type = "skew", parameters = [θ[2]]), (type = "aspect", parameters = [θ[1]])]))   # the order is the canonical one (IO's reader)
        @test RC._inplane_from_body(st, (type = "composition", compose = [(type = "aspect", parameters = [θ[1]]), (type = "skew", parameters = [θ[2]])])) == θ
    end
end

@testitem "records: a grid lens reads back as written; a non-central camera record is refused by name" setup = [RCFixtures] begin
    using RadiometricCamera: Focus, FocusSpline, load_radiometric_camera, save_radiometric_camera
    using StaticArrays, LinearAlgebra, Random
    using VisualRadiometryToolkitCore: TensorProductVignetting
    import JSON
    rng = Xoshiro(3)
    W, H, GX, GY = 120, 80, 9, 7
    G = permutedims([normalize(SVector(0.05 * (i - GX / 2), 0.05 * (j - GY / 2), 1.0)) for i in 1:GX, j in 1:GY])
    K = VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(100, 0, 0, 0, 100, 0, 60, 40, 1)))
    g = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.SplineLens, G, VgtkCore.Size2(W, H); intrinsics = K), VgtkCore.Size2(W, H))
    focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), fill(0.5, 9))
    vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)
    mktempdir() do dir
        f = joinpath(dir, "camera.json")
        save_radiometric_camera(f, RC.CalibratedCamera(RadiometricCameraModel(g, focus, 0.002, vig), RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :G), RC.RenderSettings()))
        c = load_radiometric_camera(f).model
        @test typeof(c.geometric) == typeof(g)
        ps = [SVector(W * rand(rng), H * rand(rng)) for _ in 1:50]
        @test maximum(p -> norm(RC.chief_ray(c.geometric, p) - RC.chief_ray(g, p)), ps) < 1e-15
        # the record of the removed non-central model (its aperture body holds a `radius_grid`): refused by name, never read as a central camera
        d = JSON.parsefile(f); d["aperture"]["radius_grid"] = [[0.002, 0.002, 0.002, 0.002] for _ in 1:4]
        f2 = joinpath(dir, "noncentral.json"); open(io -> JSON.print(io, d), f2, "w")
        @test_throws "non-central" load_radiometric_camera(f2)
    end
end

@testitem "records: the camera's lens coverage and sensor pitch survive (IO's bounded camera body), a parametric and a grid lens" setup = [RCFixtures] begin
    using RadiometricCamera: BoardSurface, Focus, FocusSpline, HEIGHT_NC, load_radiometric_camera, load_state, Material,
                             posed_target, PrintedTarget, RenderSettings, save_radiometric_camera, save_state, TargetRadiometry, 𝔟
    using StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: Ellipse, Point2, Sensor, Size2, μm, lens_coverage, sensor, exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting
    W = H = 24
    focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)
    e = Ellipse(Point2(12.1, 11.9), 11.5, 10.25, 0.125)                     # θ ≠ 0: the angle the old parameters.json dropped
    pitch = Sensor(Size2(W, H), 5.86μm)
    central = VgtkCore.BoundedCamera(VgtkCore.model(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)), pitch;
                                     lens_coverage = e)
    # a grid lens bounded by the image's extent, its lens coverage the ellipse
    GX, GY = 9, 7
    G = permutedims([normalize(SVector(0.05 * (i - GX / 2), 0.05 * (j - GY / 2), 1.0)) for i in 1:GX, j in 1:GY])
    K = VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(30, 0, 0, 0, 30, 0, 12, 12, 1)))
    g = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.SplineLens, G, Size2(W, H); intrinsics = K), Size2(W, H); lens_coverage = e)
    mktempdir() do dir
        for lens in (central, g)
            c = RadiometricCameraModel(lens, focus, 0.0386, vig)
            f = joinpath(dir, "camera.json"); save_radiometric_camera(f, RC.CalibratedCamera(c, RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :G), RenderSettings()))
            r = load_radiometric_camera(f).model
            @test lens_coverage(r.geometric) == e && sensor(r.geometric) == sensor(lens)                    # bit for bit: the coverage, the bound
            if lens === central
                @test r.geometric == lens
            else                                   # the grid's directions bit for bit, its K
                m1, m2 = VgtkCore.model(r.geometric), VgtkCore.model(lens)
                @test isequal(VgtkCore.directions(m1), VgtkCore.directions(m2)) && m1.intrinsics == m2.intrinsics
            end
            @test haskey(VgtkIO.read_record(f).camera, :lens_coverage)                              # IO's encoding, not a second one
        end
        # …and through the calibration state that holds the camera
        tex = fixture_board()
        board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
        rad = TargetRadiometry(Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4), BoardSurface(; edge = [0.021, 0.034]))
        target = RadiometricTarget(VgtkCore.model(posed_target(board)), zero(SVector{HEIGHT_NC, Float64}), nothing, rad)
        a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8)
        light = EnvironmentLight(SphericalHarmonicMap(a, 2), 𝔟(board), 2.5)
        poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0))), SVector(0.0, 0.0, 1.4); check = false)]
        camera = RadiometricCameraModel(central, focus, 0.0386, vig)
        f = joinpath(dir, "state.json")
        save_state(f, (; camera, target, poses, names = ["V01.png"], light, settings = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4),
                       observation = RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB), page_flip = nothing))
        @test load_state(f).camera.geometric == central
    end
end

@testitem "records: an earlier RC version is refused by name — RC ≤ 0.8 (IO 0.9's envelope) and RC 0.9–0.11.0 (no observation model); this package writes v3.1" setup = [RCFixtures] begin
    using RadiometricCamera: Focus, FocusSpline, load_radiometric_camera, load_radiometric_target, load_state, RECORD_SCHEMAS, save_radiometric_camera
    import JSON
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), 24, 24)
    focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, 24, 24))), fill(1 / 1.5, 9))
    camera = RadiometricCameraModel(lens, focus, 0.0386, VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 24, 24); degree = 2))
    cc = RC.CalibratedCamera(camera, RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :G), RC.RenderSettings())
    mktempdir() do out
        f = joinpath(out, "camera.json")
        save_radiometric_camera(f, cc)
        @test VgtkIO.read_record(f).schema == RECORD_SCHEMAS.radiometric_camera == "rc/radiometric_camera/v3.1"
        @test RECORD_SCHEMAS.calibration_state == "rc/calibration_state/v3.1"
        text = read(f, String)
        for old in ("rc/radiometric_camera/v2", "rc/radiometric_camera/v1")
            g = joinpath(out, "old.json"); write(g, replace(text, "rc/radiometric_camera/v3.1" => old))
            @test_throws "written by RC ≤ 0.8 on IO 0.9: regenerate it" load_radiometric_camera(g)
        end
        # a camera of RC 0.9–0.11.0 states no observation model: refused (never migrated), naming the refinement step that regenerates it
        d = JSON.parsefile(f; dicttype = Dict{String, Any}); delete!(d, "observation_model"); d["schema"] = "rc/radiometric_camera/v3.0"
        g = joinpath(out, "v30.json"); open(io -> JSON.print(io, d), g, "w")
        @test !occursin("observation_model", read(g, String))
        @test_throws "states no observation model" load_radiometric_camera(g)
        @test_throws "--stage=refine" load_radiometric_camera(g)
        g = joinpath(out, "other.json"); write(g, replace(text, "rc/radiometric_camera/v3.1" => "vgio/cameras/v6.0"))
        @test_throws "not a rc/radiometric_camera/v3.1 record" load_radiometric_camera(g)
    end
    # the RC ≤ 0.8 states (rc/calibration_state v1, v2: test/fixtures/records_v1, records_v2, real files) are refused too: RC 0.11.1 retired
    # their version step (ruling (A')), which reached a v3.0 state — one without an observation model (owner 2026-11-15: refuse, never migrate)
    for (f, v) in ((joinpath(@__DIR__, "..", "fixtures", "records_v1", "state_blobs_fisheye.json"), "rc/calibration_state/v1"),
                   (joinpath(@__DIR__, "..", "fixtures", "records_v1", "state_fisheye_20261027_refined_v8.json"), "rc/calibration_state/v1"),
                   (joinpath(@__DIR__, "..", "fixtures", "records_v2", "state_blob_kb.json"), "rc/calibration_state/v2"))
        @test VgtkIO.read_record(f).schema == v
        @test_throws "written by RC ≤ 0.8 on IO 0.9: regenerate it" load_state(f)
    end
    mktempdir() do out
        g = joinpath(out, "target.json"); write(g, "{\"schema\": \"rc/radiometric_target/v1\", \"length_unit\": \"mm\"}")
        @test_throws "written by RC ≤ 0.8 on IO 0.9: regenerate it" load_radiometric_target(g)
    end
end
