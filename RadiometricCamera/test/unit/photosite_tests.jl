# The refinement compared AT THE GREEN PHOTOSITES of the raw mosaic (a RAW photograph, by its observation model: `observe`; owner ruling 2026-11-12): a synthetic Bayer
# mosaic of a rendered scene — each photosite its colour's channel of the render, encoded with its colour's black level — read as
# the pipeline reads it: the linear photosite values on the raw grid (Core's `linearview`)
# and the mask by plain broadcasting over Core's `photosite_colors` (both greens; the censored photosites at the ADC floor and the
# white level dropped). One residual per green photosite (a single-channel state). The camera is recovered from a perturbed start;
# the red and blue photosites never enter the fit; the device evaluates the same residual.

@testsetup module PhotositeScene
    using RadiometricCamera: compute_mask, Focus, FocusSpline, Material, TexturedTarget, render_view, RenderSettings,
                             sphere_radius, View, 𝔟
    using RadiometricCamera, StaticArrays
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))       # cam

    export photosite_board, mosaic_of, noisy_mosaic, read_photosites

    const BLACK = (512.0, 509.0, 515.0, 507.0)                  # R, G, B, G2: the two greens' black levels differ, as a camera's do
    const WHITE = 16383.0

    """The RGGB mosaic of the render `img` (H × W × 3): each photosite its colour's channel, × κ of its colour's range above black,
    recorded as `code(black, signal)` (the signal in codes above its colour's black; default: noise-free), rounded."""
    function mosaic_of(img, κ; code = (b, sig) -> b + sig)
        H, W = size(img, 1), size(img, 2)
        pat = UInt8[0 1; 3 2]
        raw = [begin
                   c = pat[mod(i - 1, 2) + 1, mod(j - 1, 2) + 1]; ch = c == 0x00 ? 1 : c == 0x02 ? 3 : 2
                   b = BLACK[c + 1]
                   round(UInt16, code(b, κ * img[i, j, ch] * (WHITE - b)))
               end for i in 1:H, j in 1:W]
        md = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing, iso = nothing,
              model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
        return VgtkCore.RawMosaic(raw, pat, BLACK, WHITE, (0, 0), (H, W), md)
    end

    """The mosaic of the render `img` (`mosaic_of`) as a camera of gain `gain` (e⁻/code) and read noise `read_noise` (codes) records
    it: each photosite's code its colour's black + Poisson(gain · signal)/gain + N(0, read_noise²), rounded (the rounding adds 1/12
    code²), clamped below white."""
    noisy_mosaic(rng, img, κ; gain, read_noise) =
        mosaic_of(img, κ; code = (b, sig) -> clamp(b + VgtkCore.Distributions.rand(rng, VgtkCore.Distributions.Poisson(max(gain * sig, 0.0))) / gain +
                                                       read_noise * randn(rng), 0, WHITE - 1))

    "The mosaic `m` as the pipeline reads it (the entry point's photosite route, `RC._observe`): its linear photosite values on the raw grid and its valid green mask."
    read_photosites(m) = RC._observe(RC.Photosites(), m, trues(size(m)), "test", VgtkCore.SensorModel())

    "The calibration fixture's two-view board, the renders and their compute masks."
    function photosite_board(; W = 24, H = 24)
        tex = fixture_board()
        board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
        lens_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 1.288 * W, (0.51 * W, 0.48 * H), W, H)
        tps_t = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H)))
        focus_t = Focus(tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
        poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
        vig_t = RC.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); g = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (g[i]^2 + g[j]^2) for i in 1:4, j in 1:4]) end)
        camera_t = RadiometricCameraModel(lens_t, focus_t, 0.0386, vig_t)
        b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
        a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2); a_t[:, 4] .= (0.1, 0.15, 0.2)
        light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
        mat_t = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
        st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
        masks = [compute_mask(camera_t, p, board, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
        renders = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks[k]) for k in 1:2]
        return (; board, st, lens_t, tps_t, poses, camera_t, vig_t, mat_t, light_t, a_t, b, r_SH, masks, renders, W, H)
    end
end

@testitem "photosites: linear values on the raw grid, the valid green photosites the mask" setup = [RCFixtures, PhotositeScene] begin
    using Random
    import VisualGeometryToolkitFeatures as VgtkFeatures
    f = photosite_board()
    κ = 0.8 / maximum(maximum, f.renders)
    m = mosaic_of(f.renders[1], κ)
    plane, valid = read_photosites(m)
    @test size(plane) == (f.H, f.W) && size(valid) == (f.H, f.W)                         # the raw grid, one plane: nothing resampled
    col = VgtkCore.photosite_colors(m)
    @test valid == ((col .== 0x01) .| (col .== 0x03))                                    # both greens of every cell (none censored here)
    lv = VgtkCore.linearview(m)                                                           # Core's one linearisation of the photosites
    @test all(isapprox(plane[i, j], lv[i, j]; rtol = 1e-14) for i in 1:f.H, j in 1:f.W)
    @test maximum(abs(plane[i, j] - κ * f.renders[1][i, j, 2]) for i in 1:f.H, j in 1:f.W if valid[i, j]) < 1 / (m.white - maximum(m.black))
    # censored at both ends: a green photosite at the white level and one at the ADC floor are out; one below black stays (noise)
    raw = copy(m.raw); g = findall(valid); i1, j1 = g[1].I; i2, j2 = g[2].I; i3, j3 = g[3].I
    raw[i1, j1] = UInt16(m.white); raw[i2, j2] = 0x0000; raw[i3, j3] = UInt16(100)       # 100 < black 509
    p2, v2 = read_photosites(VgtkCore.RawMosaic(raw, m.pattern, m.black, m.white, m.visible_origin, m.raw_size, m.metadata))
    @test !v2[i1, j1] && !v2[i2, j2] && v2[i3, j3] && count(v2) == count(valid) - 2 && p2[i3, j3] < 0
    # the edge strength reads the green photosites alone (the Gaussian derivative normalized over the support: `edge_strength`), never
    # their red and blue neighbours: a step in the green between columns 12 and 13 under red and blue noise gives the same strength for
    # any noise, strongest beside the step
    rng = Xoshiro(3)
    edge(r) = Float64[(col[i, j] == 0x01 || col[i, j] == 0x03) ? (j <= 12 ? 0.2 : 0.8) : rand(r) for i in 1:f.H, j in 1:f.W]
    d = RC.GaussianDerivativeEdges(1.5)
    g1 = RC.edge_strength(d, edge(rng), valid; device = RC.KernelAbstractions.CPU())
    g2 = RC.edge_strength(d, edge(Xoshiro(4)), valid; device = RC.KernelAbstractions.CPU())
    @test g1 ≈ g2 rtol = 1e-5
    @test minimum(g1[i, j] for i in 4:f.H-3, j in (12, 13) if valid[i, j]) > 10 * maximum(g1[i, j] for i in 4:f.H-3, j in vcat(1:5, 20:f.W) if valid[i, j] && j <= f.W)
end

@testitem "photosites: one residual per green photosite; the refinement recovers the camera; red and blue never enter" setup = [RCFixtures, PhotositeScene] begin
    using RadiometricCamera: cost, Focus, run_lm!, View
    using StaticArrays, LinearAlgebra
    using Rotations: rotation_angle, RotMatrix
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    f = photosite_board()
    κ = 0.8 / maximum(maximum, f.renders)
    mosaics = [mosaic_of(r, κ) for r in f.renders]
    ph = [read_photosites(m) for m in mosaics]
    targets = [p[1] for p in ph]
    masks = [f.masks[k] .& ph[k][2] for k in 1:2]                                       # the compute mask AND the valid green photosites
    light_κ = EnvironmentLight(SphericalHarmonicMap(κ .* f.a_t, 2), f.b, f.r_SH)          # the encoding's scale is the lighting's
    truth = CalibrationState(f.board, f.st, f.camera_t, f.poses, f.mat_t, light_κ, targets; masks, channel = 2)
    @test all(k -> length(truth.ϕ[k]) == count(masks[k]), 1:2)                         # ONE residual per green photosite
    c_truth = cost(truth)[1]
    # R/B perturbed: the photograph differs at red and blue photosites only, the mask and every residual are the same
    col = [VgtkCore.photosite_colors(m) for m in mosaics]
    pert = map(1:2) do k
        r = copy(mosaics[k].raw); r[(col[k] .== 0x00) .| (col[k] .== 0x02)] .= UInt16(4000)
        m = mosaics[k]; VgtkCore.RawMosaic(r, m.pattern, m.black, m.white, m.visible_origin, m.raw_size, m.metadata)
    end
    ph2 = [read_photosites(m) for m in pert]
    @test all(k -> ph2[k][2] == ph[k][2] && ph2[k][1] != ph[k][1], 1:2)
    truth2 = CalibrationState(f.board, f.st, f.camera_t, f.poses, f.mat_t, light_κ, [p[1] for p in ph2]; masks, channel = 2)
    @test cost(truth2)[1] == c_truth && truth2.ϕ == truth.ϕ
    # a perturbed camera (poses, lens) recovered from the green photosites alone: the truth to the 14-bit encoding
    camera_0 = RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.15, 0.5)), 1.25 * f.W, (0.5 * f.W, 0.5 * f.H), f.W, f.H),
                                      Focus(f.tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9]), 0.0386, f.vig_t)
    poses_0 = [VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), vcat(SVector(0.01, -0.008, 0.012), SVector(0.01, -0.01, 0.015))) for p in f.poses]
    s = CalibrationState(f.board, f.st, camera_0, poses_0, f.mat_t, light_κ, targets; masks, channel = 2)
    c0 = cost(s)[1]
    run_lm!(s; maxiters = 100, reldcost = 1e-12, fixed = (:focus, :aperture, :vignetting, :lighting, :material))
    @test cost(s)[1] < 1e-3 * c0 && cost(s)[1] <= 1.01 * c_truth + 1e-12
    @test all(rad2deg(rotation_angle(RotMatrix{3}(s.poses[k].R' * f.poses[k].R))) < 1e-3 for k in 1:2)
    @test maximum(abs, pinhole_θ(RC.geometric(s))[3:4] .- pinhole_θ(f.lens_t)[3:4]) < 1e-3 * f.W
end

@testitem "photosites: the device evaluates the same residual" tags = [:gpu] setup = [RCFixtures, RCGPU, PhotositeScene] begin
    using RadiometricCamera: build_problem, PerViewResidual
    import CUDA
    using CUDA: CUDABackend
    require_gpu()
    if RC_GPU_TESTS
        f = photosite_board(; W = 32, H = 32)
        κ = 0.8 / maximum(maximum, f.renders)
        ph = [read_photosites(mosaic_of(r, κ)) for r in f.renders]
        masks = [f.masks[k] .& ph[k][2] for k in 1:2]
        light_κ = EnvironmentLight(SphericalHarmonicMap(κ .* f.a_t, 2), f.b, f.r_SH)
        s_cpu = CalibrationState(f.board, f.st, f.camera_t, f.poses, f.mat_t, light_κ, [p[1] for p in ph]; masks, channel = 2)
        s_gpu = CalibrationState(f.board, f.st, f.camera_t, f.poses, f.mat_t, light_κ, [p[1] for p in ph]; masks, channel = 2,
                                 backend = CUDABackend(; always_inline = true))
        @test s_gpu.px == s_cpu.px                                                         # the same green photosites
        prob_c, _ = build_problem(s_cpu); prob_g, _ = build_problem(s_gpu)
        rc = get(prob_c.problem.costs, PerViewResidual{9})[1]; rg = get(prob_g.problem.costs, PerViewResidual{9})[1]
        r_c = RC.NLLSsolver.computeresidual(rc, RC.NLLSsolver.getvars(rc, prob_c.variables)...)
        r_g = RC.NLLSsolver.computeresidual(rg, RC.NLLSsolver.getvars(rg, prob_g.variables)...)
        @test length(r_c) == length(r_g) == length(s_cpu.px[1]) && maximum(abs, r_c .- r_g) < 2e-3 * max(1.0, maximum(abs, r_c))
    end
end

@testitem "photographs: RAW or decoded by the FILE and the TYPE, never a flag; a decoded image never reaches the photosite route" setup = [RCFixtures, PhotositeScene] begin
    using PNGFiles, ImageCore
    mktempdir() do dir
        # photo_file: the RAW file of the name's stem when the folder holds one, else the decoded file; both or neither refused
        raw = mkpath(joinpath(dir, "raw")); dec = mkpath(joinpath(dir, "dec")); both = mkpath(joinpath(dir, "both"))
        touch(joinpath(raw, "V01.ARW")); PNGFiles.save(joinpath(dec, "V01.png"), zeros(Gray{N0f16}, 6, 8))
        touch(joinpath(both, "V01.ARW")); PNGFiles.save(joinpath(both, "V01.png"), zeros(Gray{N0f16}, 6, 8))
        @test RC.photo_file(raw, "V01.png") == joinpath(raw, "V01.ARW")
        @test RC.photo_file(dec, "V01.png") == joinpath(dec, "V01.png")
        @test_throws "both a decoded image and a RAW file" RC.photo_file(both, "V01.png")
        @test_throws "no photograph" RC.photo_file(dec, "V02.png")
        # load_photo: a decoded image is its linear array; a RAW file goes to Features' load_raw_mosaic (the one RAW path), whatever
        # it is named in the views; a RAW photograph is linear (no sRGB decode)
        @test RC.load_photo(joinpath(dec, "V01.png")) isa AbstractArray && !(RC.load_photo(joinpath(dec, "V01.png")) isa VgtkCore.RawMosaic)
        bt = try RC.load_photo(joinpath(raw, "V01.ARW")); nothing catch; catch_backtrace() end     # an empty file: the RAW decoder refuses it
        @test bt !== nothing && any(fr -> fr.func === :load_raw_mosaic, stacktrace(bt))
        @test_throws "a RAW photograph is linear" RC.load_photo(joinpath(raw, "V01.ARW"); linearize = true)
    end
    # the observation model's KIND decides (`_observe`, behind the entry point `observe`) — RAW photosites: a RawMosaic's green photosites (the
    # mask restricted, its linear view); decoded: the image itself, under the mask unchanged. Neither kind reads the other's photograph.
    m = mosaic_of(zeros(24, 24, 3) .+ 0.3, 1.0); a = VgtkCore.SensorModel()
    vals, mask = RC._observe(RC.Photosites(), m, trues(24, 24), "raw", a, :G)
    @test mask == RC.photosite_mask(m, :G) && vals == VgtkCore.linearview(m)
    img = rand(Float32, 24, 24, 3); mk = rand(Bool, 24, 24)
    vals2, mask2 = RC._observe(RC.Decoded(false), img, mk, "decoded", a, :RGB)
    @test vals2 === img && mask2 == mk
    @test_throws MethodError RC._observe(RC.Photosites(), img, mk, "decoded", a, :G)
    @test_throws "the camera predicts decoded images only" RC._observe(RC.Decoded(false), m, trues(24, 24), "raw", a, :RGB)
    @test_throws "the RAW mosaic is" RC._observe(RC.Photosites(), m, trues(12, 12), "raw", a, :G)       # the camera's sensor size, refused otherwise
end

@testitem "photosites: the photosites of any one colour filter" setup = [RCFixtures, PhotositeScene] begin
    using Test
    # a mosaic with a photosite at the ADC floor and one at the white level; its four Bayer phases as windows (an odd start shifts the phase)
    m0 = mosaic_of(fill(0.3, 12, 14, 3), 1.0)
    raw = copy(parent(m0)); raw[3, 3] = 0x0000; raw[4, 6] = UInt16(m0.white); raw[5, 5] = 0x0000; raw[6, 8] = UInt16(m0.white)
    m = VgtkCore.RawMosaic(raw, m0.pattern, m0.black, m0.white, m0.visible_origin, m0.raw_size, m0.metadata)
    for r0 in 1:2, c0 in 1:2
        w = m[r0:12, c0:14]                                             # the pattern starts at each of the four Bayer positions in turn
        c = VgtkCore.photosite_colors(w)
        @test c[1, 1] == m.pattern[r0, c0]
        usable = (w .> 0) .& .!VgtkCore.saturationview(w)
        R = RC.photosite_mask(w, :R); G = RC.photosite_mask(w, :G); B = RC.photosite_mask(w, :B)
        @test R == (c .== 0) .& usable && B == (c .== 2) .& usable
        @test G == ((c .== 1) .| (c .== 3)) .& usable                   # both greens: the mask a green fit has always read, bit for bit
        @test !any(R .& G) && !any(R .& B) && !any(G .& B) && (R .| G .| B) == usable   # every usable photosite is in exactly one filter's mask
        @test count(c .== 0) == count(c .== 2) && count((c .== 1) .| (c .== 3)) == 2 * count(c .== 0)
        @test count(usable) == length(w) - count(iszero, w) - count(VgtkCore.saturationview(w)) < length(w)   # the censored photosites are in none
        # the entry point reads the observation model's channel
        for ch in (:R, :G, :B)
            vals, mask = RC._observe(RC.Photosites(), w, trues(size(w)), "raw", VgtkCore.SensorModel(), ch)
            @test mask == RC.photosite_mask(w, ch) && vals == VgtkCore.linearview(w)
        end
    end
    @test_throws "the channel :R, :G or :B (not :RGB)" RC.photosite_mask(m, :RGB)
    @test_throws "`photosite_mask(m, :G)`" RC.green_photosite_mask(m)   # retired by name
    @test Base.ispublic(RC, :photosite_mask) && !Base.ispublic(RC, :green_photosite_mask)
    # the observation model: RAW photosites of one filter, any of the three; never the three channels at once
    for ch in (:R, :G, :B)
        @test RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), ch).channel === ch
    end
    @test_throws "ONE observation each" RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :RGB)
end
