#!/usr/bin/env julia
# MIGRATION TEST, stage D (the harness environment test/integration_blobboards/Project.toml, no BlobBoards): the contract RadiometricCamera has with VisualGeometryToolkitCore —
# a parametric lens is Core's pixel camera (π⁻¹ and project are Core's, Float32 through Core's convert_eltype), a pose is Core's
# camera → world EuclideanMap, Core's lens models as pixel cameras against Core's own pixel back-projection, the board plane hit,
# and a CPU smoke of the ray-tracing pipeline (mask → render → cost) that runs through all of them, plus hand-computed fixed values
# that a convention flip cannot get through. Run after bumping VisualGeometryToolkitCore
# in the package environment ([compat] in Project.toml, then Pkg.update / instantiate):
#   julia --project=test/integration_blobboards test/integration_blobboards/test_vgc_contract.jl
using Test, RadiometricCamera, StaticArrays, LinearAlgebra, Statistics, Printf
using RadiometricCamera: _chief_ray
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: exp_so3
using VisualGeometryToolkitCore: TensorProductVignetting, EnvironmentLight, SphericalHarmonicMap
const RC = RadiometricCamera
include(joinpath(@__DIR__, "..", "unit", "fixture_board.jl"))
include(joinpath(@__DIR__, "..", "unit", "fixture_camera.jl"))       # cam: Core's pixel camera of a test image
const WORK = get(ENV, "RC_MIG_WORK", joinpath(@__DIR__, "work", "adhoc")); mkpath(WORK)

@testset "VisualGeometryToolkitCore contract (stage D)" begin
    @info "VisualGeometryToolkitCore in the package environment" version = pkgversion(VgtkCore) path = pkgdir(VgtkCore)
    open(joinpath(WORK, "rc_stack_versions.txt"), "w") do io
        println(io, "RadiometricCamera env: VisualGeometryToolkitCore ", pkgversion(VgtkCore), " at ", pkgdir(VgtkCore))
    end

    @testset "the Core types and functions RadiometricCamera builds on" begin
        for name in (:AbstractLensModel, :LogicalIntrinsics, :CameraCalibrationMat, :CameraModel, :EuclideanMap, :LeftRigidManifold6,
                     :backproject, :project, :parameter_vector, :parameter_names, :with_parameters, :convert_eltype, :direction_tangents,
                     :KannalaBrandt, :Perspective, :Additive, :Radial, :Tangential, :Pinhole, :DoubleSphere, :Division, :Lens, :SplineLens, :PlanarTarget, :Canvas)
            @test isdefined(VgtkCore, name)
        end
        for l in (cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), 24, 24), cam(VgtkCore.Division(0.144, -0.293), 32.4, (12.24, 11.52), 24, 24; f_y = 18.24))
            @test l isa VgtkCore.BoundedCamera && VgtkCore.model(l) isa VgtkCore.CameraModel && isbits(l)   # a parametric lens IS Core's pixel camera
        end
    end

    @testset "a parametric lens is Core's pixel camera: its chief ray and project are Core's" begin
        ds = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), 24, 24)
        rl = cam(VgtkCore.Division(0.144, -0.293), 32.4, (12.24, 11.52), 24, 24; f_y = 18.24)
        vk = cam(VgtkCore.KannalaBrandt(-0.01, 0.002, 0.0, 0.0), 31.2, (12.24, 11.52), 24, 24)
        for l in (ds, rl, vk), u in (SVector(19.2, 7.2), SVector(9.6, 15.6))
            @test _chief_ray(l, u) ≈ normalize(SVector{3, Float64}(VgtkCore.backproject(VgtkCore.model(l), u))) atol = 1e-12
            @test VgtkCore.project(l, _chief_ray(l, u)) ≈ u atol = 1e-9
        end
        l32 = VgtkCore.convert_eltype(Float32, ds)                         # the device precision: Core's convert_eltype
        @test _chief_ray(l32, SVector(19.2f0, 7.2f0)) isa SVector{3, Float32}
        @test _chief_ray(l32, SVector(19.2f0, 7.2f0)) ≈ _chief_ray(ds, SVector(19.2, 7.2)) atol = 1e-5
    end

    @testset "a pose is Core's camera → world EuclideanMap" begin
        p = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(0.1, 0.2, -0.3))), SVector{3}(SVector(1.0, 2.0, 3.0)); check = false)
        @test p isa VgtkCore.EuclideanMap
        @test inv(p)(p.t) ≈ zeros(3) atol = 1e-12                               # the camera centre maps to the camera origin
    end

    @testset "board plane hit" begin
        tex = rand(64, 96, 3)
        bd = PrintedTarget([0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], tex; facing = [0.0, 0.0, 0.0])
        s, 𝐱 = intersect_board(bd, SVector(0.0, 0.0, 0.0), SVector(0.25, 0.5, 1.0))
        @test 𝐱 ≈ SVector(0.5, 1.0, 2.0) atol = 1e-12
        @test s ≈ 2 atol = 1e-12
    end

    @testset "Core lens models as pixel cameras vs Core's own pixel back-projection" begin
        W₀, H₀ = 1200, 900
        for lens in (VgtkCore.KannalaBrandt(-0.02, 0.004, -0.0007, 0.00008), VgtkCore.Lens(VgtkCore.Perspective(), VgtkCore.Additive(VgtkCore.Radial((-0.12, 0.03)), VgtkCore.Tangential(0.0005, -0.0003))), VgtkCore.Pinhole())
            K = VgtkCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0.0, 0.0, 0.0, 900.0, 0.0, W₀ / 2 + 3.0, H₀ / 2 - 2.0, 1.0))
            cm = VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(K), lens)        # K in COLMAP px
            l = VgtkCore.BoundedCamera(cm, VgtkCore.Size2(W₀, H₀))
            for (x, y) in ((300.0, 200.0), (W₀ / 2, H₀ / 2), (1000.0, 700.0))
                @test _chief_ray(l, SVector(x, y)) ≈ normalize(SVector{3, Float64}(VgtkCore.backproject(cm, SVector(x, y)))) atol = 1e-12
            end
        end
    end

    @testset "ray-tracing pipeline smoke (CPU): mask → render → cost with every lens type" begin
        tex = fixture_board()
        board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
        scene = Scene(board, 24, 24)
        sensor = 1.288 * 0.066
        focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, 24, 24))), fill(1 / 1.5, 9))
        poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
        rl = cam(VgtkCore.Division(0.144, -0.293), 32.4, (12.24, 11.52), 24, 24; f_y = 18.24)
        gl = grid_model(rl, 25, 25)
        @test RC.grid_angular_error(gl, rl) < 5e-3                          # 25×25 quasi-interpolation of the rational lens: 1.1 mrad
        vk = cam(VgtkCore.KannalaBrandt(-0.01, 0.002, 0.0, 0.0), 31.2, (12.24, 11.52), 24, 24)
        for lens in (cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), 24, 24), rl, gl, vk)
            views = [View(p, lens, focus, 0.0386) for p in poses]
            vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 24, 24); degree = 2); b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
            # the contract's values are the paper's integral ∝ r²: the render divides by πr², so the lighting carries it
            light = EnvironmentLight(SphericalHarmonicMap(2.0 * π * 0.0386^2, 2), b, r_SH); mat = Material()
            st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
            masks = [compute_mask(v, scene, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for v in views]
            @test all(count(m) > 50 for m in masks)
            targets = [render_view(k, views[k], vig, mat, light, scene, st, masks[k]) for k in 1:2]
            @test all(t -> all(isfinite, t) && maximum(t) > 0, targets)
            s = CalibrationState(scene, st, copy(views), vig, mat, light, targets; masks)
            c, rms = cost(s)
            @test isfinite(c) && rms < 1e-6                                   # re-rendering the generating state: zero residual
            moved = [View(VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), SVector(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)), lens, focus, 0.0386) for p in poses]
            s2 = CalibrationState(scene, st, moved, vig, mat, light, targets; masks)
            c2, rms2 = cost(s2)
            @info "pipeline smoke" lens = nameof(typeof(lens)) rel_rms_at_truth = rms rel_rms_moved_10mrad = rms2
            @test rms2 > 1e-3                                                # the render responds to geometry
        end
    end

    # ---- hand-computed fixed values: a convention flip (pose map direction, image axis, pixel centre) satisfies every
    # round-trip test above because both directions flip together; only absolute numbers catch it ----------------------
    @testset "fixed values: lens back-projection" begin
        # (an image of 100 × 100 px, f = 100 px, the principal point at its centre: m = (𝐩 − (50, 50)) / 100)
        pin = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), 100.0, (50.0, 50.0), 100, 100)
        @test _chief_ray(pin, SVector(75.0, 37.5)) ≈ SVector(0.25, -0.125, 1.0) / sqrt(1.078125) atol = 1e-12
        # unified camera model (DoubleSphere with α = 0, ξ = 0.5): d ∝ (ξ + √(1 + (1 − ξ²) r²))/(1 + r²) · (m, 1) − (0, 0, ξ)
        ucm = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.5, 0.0)), 100.0, (50.0, 50.0), 100, 100)
        @test _chief_ray(ucm, SVector(80.0, 70.0)) ≈ SVector(0.4108716, 0.2739144, 0.8695719) atol = 1e-6
        @test VgtkCore.project(ucm, SVector(0.4108716, 0.2739144, 0.8695719)) ≈ SVector(80.0, 70.0) atol = 1e-4
        # division model: x_u = m / (1 + k₁ r² + k₂ r⁴), m = (0.3, 0.2), r² = 0.13, k₁ = −0.1 → (0.3039514, 0.2026343, 1) normalised
        div = cam(VgtkCore.Division(-0.1, 0.0), 100.0, (50.0, 50.0), 100, 100)
        @test _chief_ray(div, SVector(80.0, 70.0)) ≈ SVector(0.2854983, 0.1903322, 0.9392893) atol = 1e-6
        @test VgtkCore.project(div, SVector(0.2854983, 0.1903322, 0.9392893)) ≈ SVector(80.0, 70.0) atol = 1e-4
        # intrinsics: f and the principal point in pixels, f_y independent of f_x
        off = cam(VgtkCore.Division(0.0, 0.0), 200.0, (60.0, 40.0), 100, 100; f_y = 50.0)
        @test _chief_ray(off, SVector(80.0, 60.0)) ≈ normalize(SVector((80.0 - 60.0) / 200.0, (60.0 - 40.0) / 50.0, 1.0))
        # a pinhole Core lens with f = 900 px and the principal point at the image centre of a 1200 × 900 image
        pv = cam(VgtkCore.Pinhole(), 900.0, (600.0, 450.0), 1200, 900)
        @test _chief_ray(pv, SVector(900.0, 450.0)) ≈ normalize(SVector(300 / 900, 0.0, 1.0)) atol = 1e-12
    end

    @testset "fixed values: pose, board hit, board chart, forward projection" begin
        tex = fill(0.5, 8, 8, 3)
        bd = PrintedTarget([-1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 0.0])
        pin = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), 100.0, (50.0, 50.0), 100, 100)
        # identity pose: the pixel (75, 37.5) (m = (0.25, −0.125)) hits z = 2 at (0.5, −0.25, 2) → board chart (0.75, 0.375)
        s, X = intersect_board(bd, SVector(0.0, 0.0, 0.0), _chief_ray(pin, SVector(75.0, 37.5)))
        @test X ≈ SVector(0.5, -0.25, 2.0) atol = 1e-12
        @test chart(bd, X) ≈ SVector(0.75, 0.375) atol = 1e-12
        @test world_point(bd, SVector(0.75, 0.375)) ≈ SVector(0.5, -0.25, 2.0) atol = 1e-12
        # camera rotated +90° about z (e₁ → e₂), centre (0.1, 0.2, 0): world direction (0.125, 0.25, 1) → hit (0.35, 0.7, 2)
        p = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(0.0, 0.0, π / 2))), SVector{3}(SVector(0.1, 0.2, 0.0)); check = false)
        @test p.R * SVector(1.0, 0.0, 0.0) ≈ SVector(0.0, 1.0, 0.0) atol = 1e-12
        dw = p.R * _chief_ray(pin, SVector(75.0, 37.5))
        s, X = intersect_board(bd, p.t, dw)
        @test X ≈ SVector(0.35, 0.7, 2.0) atol = 1e-12
        @test chart(bd, X) ≈ SVector(0.675, 0.85) atol = 1e-12
        # back to the camera: R_cwᵀ(X − 𝐜) = (0.5, −0.25, 2) — the same numbers through the pose's inverse map — and the projection
        Y = p.R' * (X - p.t)
        @test Y ≈ SVector(0.5, -0.25, 2.0) atol = 1e-12
        @test inv(p)(X) ≈ Y atol = 1e-12
        @test VgtkCore.project(pin, Y) ≈ SVector(75.0, 37.5) atol = 1e-10
    end

    @testset "fixed values: the pixel convention of the toolchain" begin
        # colmap px (pixel i spans [i, i + 1), centre i + ½): Julia's 1-based pixel (u, v) has its centre at (u − ½, v − ½)
        @test 𝐩_of_pixel(1, 1) == SVector(0.5, 0.5) && 𝐩_of_pixel(512, 512) == SVector(511.5, 511.5)
        @test pixel_of_𝐩(SVector(300.25, 70.5)) == SVector(300.75, 71.0)
    end

    @testset "fixed value through the renderer: a single blob lands on its predicted pixel" begin
        # a board with ONE blob at a chosen chart position (texture rasterised from the layout, white elsewhere), a pinhole
        # looking straight down: the rendered dark spot must sit where pose + lens + board chart + pixel convention put it
        C = SVector(0.6, 0.35); Rb = 0.05
        # the blob's ink map: ‖𝐛 − C‖/R clamped to [0, 1] at the texel centres (0 = ink at the centre, 1 = paper)
        ι1 = [clamp(hypot((j - 0.5) / 256 - C[1], (i - 0.5) / 256 - C[2]) / Rb, 0.0, 1.0) for i in 1:256, j in 1:256]
        tex1 = cat(ι1, ι1, ι1; dims = 3)
        o = [-1.0, -1.0, 0.0]; u = [2.0, 0.0, 0.0]; v = [0.0, 2.0, 0.0]; facing = [0.0, 0.0, 1.5]
        board = PrintedTarget(o, u, v, tex1; facing)
        white = PrintedTarget(o, u, v, ones(256, 256, 3); facing)
        NPX = 48
        pin = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), Float64(NPX), (NPX / 2, NPX / 2), NPX, NPX)
        pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0))), SVector{3}(SVector(0.0, 0.0, 1.5)); check = false)       # R_cw = diag(1, −1, −1): looking down at the board
        view = View(pose, pin, Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, NPX, NPX))), fill(1 / 1.5, 9)), 0.02)
        X = world_point(board, C)                                                  # (0.2, −0.3, 0)
        @test X ≈ SVector(0.2, -0.3, 0.0) atol = 1e-12
        Y = pose.R' * (X - pose.t)                                              # (0.2, 0.3, 1.5)
        @test Y ≈ SVector(0.2, 0.3, 1.5) atol = 1e-12
        pc = VgtkCore.project(pin, Y)                                               # (30.4, 33.6) COLMAP px
        col, row = pixel_of_𝐩(SVector{2, Float64}(pc))                              # the 1-based pixel: (30.9, 34.1)
        st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)   # the contract's values are the paper's integral ∝ r²: the lighting carries the πr²
        function render_board(bd)
            scene = Scene(bd, NPX, NPX); b = 𝔟(bd)
            light = EnvironmentLight(SphericalHarmonicMap(2.0 * π * 0.02^2, 2), b, sphere_radius(b, [pose.t]))
            mask = compute_mask(view, scene, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU())
            return render_view(1, view, TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, NPX, NPX); degree = 2), Material(), light, scene, st, mask)[:, :, 1], mask
        end
        lum, mask = render_board(board); lum_w, _ = render_board(white)
        ratio = lum ./ max.(lum_w, 1e-9)                                             # shading and vignetting cancel: the ink map as seen
        rpx = Rb * 2 / (1.5 / NPX)                                                   # blob radius in pixels: 3.2
        w = ceil(Int, rpx) + 3
        r0 = round(Int, row); c0 = round(Int, col)
        @test mask[r0, c0] && w + 1 <= r0 <= NPX - w && w + 1 <= c0 <= NPX - w
        dark = max.(1 .- ratio[r0-w:r0+w, c0-w:c0+w], 0.0)
        @test sum(dark) > 0.5 && ratio[r0, c0] < 0.6                                  # there IS a dark spot at the predicted pixel
        rc = sum(dark[i, j] * (r0 - w + i - 1) for i in axes(dark, 1), j in axes(dark, 2)) / sum(dark)
        cc = sum(dark[i, j] * (c0 - w + j - 1) for i in axes(dark, 1), j in axes(dark, 2)) / sum(dark)
        @info "rendered blob vs prediction (1-based row, col)" predicted = (row, col) rendered = (rc, cc) blob_radius_px = rpx
        @test hypot(rc - row, cc - col) < 0.5
    end
end
