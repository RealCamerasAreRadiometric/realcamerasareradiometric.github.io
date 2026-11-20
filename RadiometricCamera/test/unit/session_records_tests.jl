# A calibration session read from its VisualGeometryToolkitIO records (`load_session`): the fixture session of three training views of
# test/fixtures/lenssim_fisheye3 (CaliBlobs' records of the BlobBoards session on board 291). Checked against the records themselves,
# read independently here through VisualGeometryToolkitIO: the chart model is the camera record's pixel camera, the board is the target's
# canvas texture (its rectangle and its image), the scene's world is the target's frame, and every view maps the board's points to where
# the trajectory record maps them.
@testitem "session records (load_session)" setup = [RCFixtures] begin
    using RadiometricCamera: board_normal, world_point, 𝔟
    using StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    fix = joinpath(@__DIR__, "..", "fixtures", "lenssim_fisheye3")
    rec = joinpath(fix, "session_io")
    mktempdir() do imgs
        # the trajectory names the session's images train_0k.png: the fixture's training views lsc_vk_ref.png
        for k in 1:3
            symlink(abspath(joinpath(fix, "lsc_v$(k)_ref.png")), joinpath(imgs, "train_0$(k).png"))
        end
        ds = RC.load_session(rec, imgs; focus_init = :board_center, downsample = 8, texture_downsample = 8)
        @test ds.names == ["train_01.png", "train_02.png", "train_03.png"] && ds.frames == [1, 2, 3] && Tuple(ds.extent) == (1024, 1024)
        @test size(ds.targets[1]) == (128, 128, 3) && length(ds.poses) == 3 && !haskey(ds, :views)
        # the camera: the record's pixel camera (COLMAP K) on the scene's pixels (the photographs box-averaged by 8: K scaled by 1/8)
        rcam = VgtkIO.load(VgtkCore.CameraModel, joinpath(rec, "camera.json"))
        K = rcam.intrinsics.K
        @test has_lens(ds.lens₀, VgtkCore.Division{2}) && VgtkCore.lens(ds.lens₀) == rcam.lens && VgtkCore.extent(ds.lens₀) == VgtkCore.Size2(128, 128)
        @test pinhole_θ(ds.lens₀) ≈ SVector(K[1, 1], K[2, 2], K[1, 3], K[2, 3]) ./ 8
        # the board: the target's canvas texture — its rectangle (metres, the design chart) and its image box-averaged by 8
        tr = VgtkCore.Rect(VgtkCore.canvas(VgtkCore.model(ds.board.target)).texture)
        b = ds.board
        @test b.size ≈ SVector(tr.widths...) && 0.13 < b.size[1] < 0.1301
        @test VrtkCore.texture_size(b) == (6143 ÷ 8, 6143 ÷ 8)
        @test world_point(b, SVector(0.0, 0.0)) ≈ SVector(tr.origin[1], tr.origin[2] + tr.widths[2], 0.0) atol = 1e-15   # the image's top-left
        @test world_point(b, SVector(1.0, 1.0)) ≈ SVector(tr.origin[1] + tr.widths[1], tr.origin[2], 0.0) atol = 1e-15   # its bottom-right
        # the poses: the trajectory's target → camera maps, in metres; the camera centre is in front of the printed side
        traj = VgtkIO.load(VgtkCore.Trajectory, joinpath(rec, "trajectory_1.json"))
        for (k, t) in enumerate(ds.frames)
            T = VgtkCore.ustrip(VgtkCore.m, traj.poses[t])                  # board (target frame, m) → camera
            for X in (SVector(0.0, 0.0, 0.0), SVector(0.03, -0.02, 0.0), SVector(-0.05, 0.04, 0.0))
                @test inv(ds.poses[k])(X) ≈ T(X) atol = 1e-12
            end
            @test dot(ds.poses[k].t - 𝔟(b), board_normal(b)) > 0
            q = VgtkCore.project(ds.lens₀, inv(ds.poses[k])(𝔟(b)))             # the board centre images inside the picture
            @test all(0 .< q .< 128)
        end
        @test ds.d_s > 0 && all(d -> 0.05 < d < 0.2, ds.center_distances)       # the 130 mm board at ≈ 9 cm (130/2000 of the 2 m render)
    end
end

@testitem "load_session's downsample: K re-expressed by the box downsample's similarity (a rendered checkerboard)" setup = [RCFixtures] begin
    # A checkerboard in the record camera's normalized pinhole plane (n = K⁻¹ [x, y, 1], COLMAP pixels: corners on the integers),
    # each pixel the mean of an s × s grid of sub-pixel samples, written at the record's 1024 × 1024 and read back by `load_session`
    # with downsample 8 (Features' box average). The downsampled photograph is that checkerboard rendered DIRECTLY at 128 × 128 through
    # the session's K (8s × 8s samples per pixel: the same sample positions), so K is exactly Core's `reframed_calibration` by x ↦ x / 8 —
    # a half-pixel offset or a size-ratio rescale would move every edge.
    using StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    import PNGFiles
    using ImageCore: RGB, N0f16
    fix = joinpath(@__DIR__, "..", "fixtures", "lenssim_fisheye3")
    rec = joinpath(fix, "session_io")
    K0 = SMatrix{3, 3, Float64}(VgtkIO.load(VgtkCore.CameraModel, joinpath(rec, "camera.json")).intrinsics.K)
    checker(n) = isodd(floor(Int, 9 * n[1]) + floor(Int, 9 * n[2])) ? 0.8 : 0.2
    render(K, W, H, s) = (Ki = inv(K); [sum(checker(Ki * SVector(u - 1 + (a - 0.5) / s, v - 1 + (b - 0.5) / s, 1.0)) for a in 1:s, b in 1:s) / s^2
                                       for v in 1:H, u in 1:W])
    full = render(K0, 1024, 1024, 2)
    mktempdir() do imgs
        q = round.(UInt16, full .* 65535)
        for k in 1:3
            PNGFiles.save(joinpath(imgs, "train_0$(k).png"), [RGB{N0f16}(reinterpret(N0f16, x), reinterpret(N0f16, x), reinterpret(N0f16, x)) for x in q])
        end
        ds = RC.load_session(rec, imgs; focus_init = :board_center, downsample = 8, texture_downsample = 8, linearize = false)
        Kd = SMatrix{3, 3, Float64}(VgtkCore.model(ds.lens₀).intrinsics.K)
        @test Kd ≈ SMatrix{3, 3, Float64}(1 / 8, 0, 0, 0, 1 / 8, 0, 0, 0, 1) * K0 rtol = 1e-14         # x ↦ x / 8: no translation
        direct = render(Kd, 128, 128, 16)
        @test maximum(abs.(ds.targets[1][:, :, 2] .- direct)) < 1e-4                                    # the 16-bit PNG's rounding
        shifted = render(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0.5 - 1 / 16, 0.5 - 1 / 16, 1) * Kd, 128, 128, 16)   # a half-pixel offset
        @test maximum(abs.(ds.targets[1][:, :, 2] .- shifted)) > 0.05
    end
    # an odd size: the remainder rows / columns are dropped by the box average and K keeps x ↦ x / f
    S = VgtkCore.SimilarityMap(RC.LinearMap(UniformScaling(1 / 4)))
    K = VgtkCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(300, 0, 0, 0, 310, 0, 101.3, 77.9, 1))
    Kd = SMatrix{3, 3, Float64}(VgtkCore.Cameras.reframed_calibration(S, K))
    img = render(SMatrix{3, 3, Float64}(K), 203, 155, 2)
    small = RC.VgtkFeatures.box_downsample(img, 4)
    @test size(small) == (38, 50) && maximum(abs.(small .- render(Kd, 50, 38, 8))) < 1e-12
end
