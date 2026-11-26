#!/usr/bin/env julia
# MIGRATION TEST, stage C (the harness environment test/integration_blobboards/Project.toml): what RadiometricCamera does with a calibration session, read from
# the stack's records — VisualGeometryToolkitIO's camera.json (cameras/v4), targets_1.json (rig/v6: board 291 with its canvas and
# texture), target_poses.json and trajectory_1.json — of the fixture session (3 training views of test/fixtures/lenssim_fisheye3, a
# LensSim fisheye render of blob board 291 with rendered board-UV truth). The records are test/fixtures/lenssim_fisheye3/session_io,
# written once by CaliBlobs' scripts/migrate_rc_session.jl from the BlobBoards session export of this fixture (README there):
#   1 camera.json → the division camera as Core's pixel camera (Division)                                                vs golden
#   3 the grid init sampled from that camera (Core's quasi-interpolation, `GridStart`) vs the division camera
#   4 target_poses.json → Blender world (similarity on the board frame) vs the true poses                      vs golden
#   5 load_session (the photo pipeline's loader of the records) for every lens option
#   6 held-out evaluation of the division camera and the grid init on 3 test views: dense lattice of true board points, pose refit
#     per view (RC's _refined_pose on its bearing correspondences: Core's LevenbergMarquardt on Estimation's BearingPoseProblem), bearing error in px  vs golden
#   7 RC_MIG_REFINE=1: a radiometric refinement from the records (RC_MIG_REFINE_ITERATIONS iterations, default 2; the held-out error per
#     iteration from its checkpoints) (bin/refine.jl --config, blob front-end, GPU): rel-RMS not
#     increasing, small grid drift                                                                              vs golden
# Not here: the blob detections (the session's IO observations record, observations.json, not yet written by its producer), so the refinement runs without
# the self-check's bearing diagnostics; the CGM samples (not migrated: a function of the camera).
# RC_MIG_RECORD=1 (re)writes golden/session_fisheye3.json from the current stack instead of comparing with it.
#   env RC_MIG_FIX=<fixture dir> RC_MIG_WORK=<run dir> [RC_MIG_REFINE=1 [RC_MIG_REFINE_ITERATIONS=30]] julia --project=test/integration_blobboards test/integration_blobboards/test_session_outputs.jl
using Test, RadiometricCamera, StaticArrays, LinearAlgebra, Statistics, Printf
using RadiometricCamera: chief_ray
using RadiometricCamera: in_domain, _image_start, grid_dims       # public, not exported (D13)
import TOML
import FileIO                                                   # EXR truth maps via ImageIO / OpenEXR (the harness environment)
const RC = RadiometricCamera
const RC_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const HARNESS_ENV = @__DIR__                                     # test/integration_blobboards/Project.toml: this stage's environment
using VisualGeometryToolkitCore: angle_between
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore: mm, ustrip
const FIX = ENV["RC_MIG_FIX"]; const SES = joinpath(FIX, "session_io")
const VIEWS = String.(split(get(ENV, "RC_MIG_VIEWS", "v1,v2,v3"), ","))
const TEST_VIEWS = ["val1", "val2", "val3"]                    # the fixture's held-out views (rendered truth only, never detected)
const RECORD = get(ENV, "RC_MIG_RECORD", "0") == "1"
const REFINE = get(ENV, "RC_MIG_REFINE", "0") == "1"
const REFINE_ITERATIONS = parse(Int, get(ENV, "RC_MIG_REFINE_ITERATIONS", "2"))   # item 7's LM iterations (the pipeline's default is 30)
const WORK = get(ENV, "RC_MIG_WORK", joinpath(@__DIR__, "work", "adhoc")); mkpath(WORK)
const GOLDEN = get(ENV, "RC_MIG_GOLDEN", joinpath(@__DIR__, "golden", "session_fisheye3.json")); mkpath(dirname(GOLDEN))
const BOARD291 = joinpath(FIX, "board_291")
const PUPIL_Z = 0.04325            # entrance pupil of the traced fisheye in front of Blender's camera origin (m)

RECORD || isfile(GOLDEN) || error("no golden file $GOLDEN — run test/integration_blobboards/run_integration_tests.sh --record on the reference stack first")
const gold = RECORD ? Dict{String, Any}() : VgtkIO.read_record(GOLDEN)
const new = Dict{String, Any}()

"""Run `cmd` with stdout and stderr into ONE handle on `log` (two handles on the same file overwrite each other); true on success."""
function run_logged(cmd, log)
    ok = false
    open(log, "w") do io
        ok = success(pipeline(cmd; stdout = io, stderr = io))
    end
    return ok
end
"""Record `value` under `name`; outside record mode compare it with the golden value (absolute or relative tolerance)."""
function check_golden(name, value, tol; rel = false)
    new[name] = value
    RECORD && return nothing
    if !haskey(gold, name)
        @error "golden value missing (re-record the golden file on the reference stack)" name
        @test false
        return nothing
    end
    g = Float64(gold[name])
    ok = rel ? abs(value - g) <= tol * abs(g) : abs(value - g) <= tol
    ok || @error "golden mismatch" name value golden = g tol rel
    @test ok
    return nothing
end

# ---- the fixture: true poses and the board frame (Blender world, metres), the rendered truth, the board-291 chart ------------
man = VgtkIO.read_record(joinpath(FIX, "manifest.json"))
svec3(v) = SVector{3, Float64}(Float64(v[1]), Float64(v[2]), Float64(v[3]))
const 𝐨 = svec3(man.board.origin); const 𝐮 = svec3(man.board.u_axis); const 𝐯 = svec3(man.board.v_axis)
"""True pose of a manifest view as (R_cw, centre): the record's `R_cw_cv` rows and `t` = camera centre."""
function pose_true(nm)
    v = only(x for x in man.views if String(x.name) == nm)
    R = SMatrix{3, 3, Float64}(Float64(v.pose_true.R_cw_cv[i][j]) for i in 1:3, j in 1:3)
    return R, svec3(v.pose_true.t)
end
"""`H × W × 3` Float64 array of an image file (the EXR truth maps through FileIO / ImageIO)."""
function load_rgb(path)
    im = FileIO.load(path)
    H, W = size(im)
    A = Array{Float64, 3}(undef, H, W, 3)
    for r in 1:H, c in 1:W
        p = RC.ImageCore.RGB(im[r, c])
        A[r, c, 1] = Float64(RC.ImageCore.red(p)); A[r, c, 2] = Float64(RC.ImageCore.green(p)); A[r, c, 3] = Float64(RC.ImageCore.blue(p))
    end
    return A
end
load_gt(nm) = (load_rgb(joinpath(FIX, "gtrays", "board_$(nm)_uv.exr")), load_rgb(joinpath(FIX, "gtrays", "board_$(nm)_flat.exr")))
"""True board chart at a continuous image position (colmap px: pixel i spans [i, i + 1), centre i + ½) by bilinear interpolation
of the uv map ÷ the flat render; Blender's v is bottom-up → b₂ = 1 − v. `nothing` off the board."""
function gt_chart(uvm, fl, x, y)
    H, W = size(fl, 1), size(fl, 2)
    c = x + 0.5; r = y + 0.5                                       # 1-based continuous pixel-centre coordinates
    c0 = clamp(floor(Int, c), 1, W - 1); r0 = clamp(floor(Int, r), 1, H - 1); fc = c - c0; fr = r - r0
    f(rr, cc) = (t = fl[rr, cc, 1]; t > 1e-3 ? SVector(uvm[rr, cc, 1] / t, uvm[rr, cc, 2] / fl[rr, cc, 2]) : nothing)
    q = (f(r0, c0), f(r0, c0 + 1), f(r0 + 1, c0), f(r0 + 1, c0 + 1)); any(isnothing, q) && return nothing
    uv = (1 - fr) * ((1 - fc) * q[1] + fc * q[2]) + fr * ((1 - fc) * q[3] + fc * q[4])
    return SVector(uv[1], 1 - uv[2])
end
"""Detection (x, y) minus the image of the true board point `btrue` (chart), in pixels via the local chart Jacobian."""
function gt_offset(uvm, fl, x, y, btrue)
    b = gt_chart(uvm, fl, x, y); b === nothing && return nothing
    bx = gt_chart(uvm, fl, x + 1, y); by = gt_chart(uvm, fl, x, y + 1); (bx === nothing || by === nothing) && return nothing
    J = hcat(bx - b, by - b)                                       # d(chart)/d(px)
    return -(J \ (btrue - b))
end
# a board-291 point (mm in BlobBoards' :world frame of the board export: canvas-centred, y up) → chart of the printed pattern,
# from the export's geometry.json (top-left corner of the pattern, size); the same convention the blob front-end assumes
geo = VgtkIO.read_record(joinpath(BOARD291, "geometry.json"))
const X0 = Float64(geo.pattern_origin_mm[1]); const YTOP = Float64(geo.pattern_origin_mm[2])
const WP = Float64(geo.pattern_size_mm[1]); const HP = Float64(geo.pattern_size_mm[2])
chart_of_mm(b) = SVector((b[1] - X0) / WP, (YTOP - b[2]) / HP)
"""Umeyama similarity B ≈ s R A + t (proper rotation) from paired points → (s, R, t)."""
function umeyama(A::Vector{SVector{3, Float64}}, B::Vector{SVector{3, Float64}})
    μa = mean(A); μb = mean(B); H = sum((b - μb) * (a - μa)' for (a, b) in zip(A, B)) / length(A)
    F = svd(Matrix(H)); Sg = Diagonal([1.0, 1.0, sign(det(F.U * F.Vt))]); R = SMatrix{3, 3, Float64}(F.U * Sg * F.Vt)
    s = tr(Diagonal(F.S) * Sg) / mean(norm(a - μa)^2 for a in A); t = μb - s * R * μa
    return s, R, t
end
"""The records' poses (target_poses.json: X_cam = R X_board + t, board 291's frame in mm) → Blender world (R_cw, centre) per frame,
via the similarity that maps board points onto the manifest's true board."""
function record_poses_world(tp)
    A = SVector{3, Float64}[]; B = SVector{3, Float64}[]
    for x in (-60.0, 0.0, 60.0), y in (-60.0, 0.0, 60.0)
        push!(A, SVector(x, y, 0.0)); b = chart_of_mm(SVector(x, y)); push!(B, 𝐨 + b[1] * 𝐮 + b[2] * 𝐯)
    end
    s, R, t = umeyama(A, B); res = [norm(s * R * a + t - b) for (a, b) in zip(A, B)]
    out = Dict{Int, Tuple{SMatrix{3, 3, Float64}, SVector{3, Float64}}}()
    for (f, poses) in tp
        m = only(values(poses))
        Rtc = SMatrix{3, 3, Float64}(m.R); ttc = SVector{3, Float64}(ustrip.(mm, m.t))
        out[f] = (R * Rtc', s * R * (-Rtc' * ttc) + t)
    end
    return out, (; scale = s, board_rms_m = sqrt(mean(res .^ 2)))
end
angle_mrad(a, b) = 1e3 * angle_between(normalize(a), normalize(b))
rot_deg(Ra, Rb) = rad2deg(VgtkCore.Rotations.rotation_angle(VgtkCore.Rotations.RotMatrix{3}(Ra' * Rb)))
"""Dense lattice of true board points of a view (every `stride` px, 8-px board-edge erosion): (pixel positions, world points)."""
function truth_lattice(nm; stride = 6)
    uvm, fl = load_gt(nm); H, W = size(fl, 1), size(fl, 2)
    𝐩s = SVector{2, Float64}[]; Xs = SVector{3, Float64}[]
    for r in 9:stride:(H - 8), c in 9:stride:(W - 8)
        all(fl[rr, cc, 1] > 1e-3 for (rr, cc) in ((r, c), (r - 8, c), (r + 8, c), (r, c - 8), (r, c + 8))) || continue
        x = c - 0.5; y = r - 0.5
        b = gt_chart(uvm, fl, x, y); b === nothing && continue
        (0 <= b[1] <= 1 && 0 <= b[2] <= 1) || continue
        push!(𝐩s, SVector{2, Float64}(x, y)); push!(Xs, 𝐨 + b[1] * 𝐮 + b[2] * 𝐯)
    end
    return 𝐩s, Xs
end
"""Hold-out score of a lens on one view: true board points `Xs` (world) seen at pixel positions `𝐩s`; the pose is refit through
the lens (RC's _refined_pose from (R0, t0), trimmed at 5 × median), the error is the bearing angle × f_px — the metric of the
experiments' hold-out tables. Returns (errors in px, fraction of points kept)."""
function holdout_errors(lens, f_px, 𝐩s, Xs, R0, t0)
    vs = [normalize(chief_ray(lens, 𝐩)) for 𝐩 in 𝐩s]
    R, t = refine_pose(Xs, vs, R0, t0)
    err(ks) = [f_px * angle_between(vs[k], normalize(R * Xs[k] + t)) for k in ks]
    e = err(eachindex(Xs)); keep = findall(e .< 5 * median(e))
    R, t = refine_pose(Xs[keep], vs[keep], R, t)
    return err(keep), length(keep) / length(Xs)
end

"""The world → camera pose `(R, t)` refined against unit rays: RC's `_refined_pose` (Core's LevenbergMarquardt on
Estimation's BearingPoseProblem) on the package's bearing correspondences."""
function refine_pose(Xs, vs, R0, t0)
    guess = VgtkCore.EuclideanMap(SMatrix{3, 3, Float64}(R0), SVector{3, Float64}(t0); check = false)
    p = RC._refined_pose(guess, RC._ray_correspondences(Xs, vs)).u     # (the refine's FitResult: its pose)
    return SMatrix{3, 3, Float64}(p.R), SVector{3, Float64}(p.t)
end

# ---- the records ---------------------------------------------------------------------------------------------------------------
cam = VgtkIO.load(VgtkCore.CameraModel, joinpath(SES, "camera.json"))
camj = VgtkIO.read_record(joinpath(SES, "camera.json"))         # the record's declarations and provenance (item 1)
const RW, RH = (bc = VgtkIO.load(VgtkCore.BoundedCamera, joinpath(SES, "camera.json")); (VgtkCore.extent(bc).width, VgtkCore.extent(bc).height))
Kc = cam.intrinsics.K
bblens = VgtkCore.BoundedCamera(cam, VgtkCore.Size2(RW, RH))   # Core's pixel camera
fpx = Float64(Kc[1, 1])
tposes = VgtkIO.load(VgtkCore.TargetPoses, joinpath(SES, "target_poses.json"))

try
@testset "RadiometricCamera consumers of the session records (stage C)" begin
    @testset "0 fixture conventions" begin
        @test (RW, RH) == (1024, 1024) && String(camj.units.pixel) == "colmap"   # the envelope's pixel frame (IO 0.11: `units.pixel`, the record read through its migration)
        @test abs(X0 + 65.00933) < 1e-3 && abs(YTOP - 65.00933) < 1e-3 && abs(WP - 130) < 1e-3   # the board-291 export as checked in
        @test chart_of_mm(SVector(X0, YTOP)) ≈ SVector(0.0, 0.0) atol = 1e-12
        @test chart_of_mm(SVector(X0 + WP, YTOP - HP)) ≈ SVector(1.0, 1.0) atol = 1e-12
        rig = VgtkIO.load(VgtkCore.Rig, joinpath(SES, "targets_1.json"))     # board 291 with its canvas and the export's texture
        @test rig isa VgtkCore.Rig
    end

    @testset "1 camera.json → the division camera (Core's pixel camera over Division)" begin
        @test VgtkCore.lens(bblens) isa VgtkCore.Division{2}
        Kb = VgtkCore.model(bblens).intrinsics.K
        pp = (; k = Tuple(VgtkCore.parameter_vector(VgtkCore.lens(bblens))))
        @test Kb[1, 1] ≈ fpx && Kb[1, 3] ≈ Kc[1, 3] && Kb[2, 3] ≈ Kc[2, 3]
        @test in_domain(bblens, SVector(512.0, 512.0)) && norm(chief_ray(bblens, SVector(819.2, 307.2))) ≈ 1
        resid = Float64(camj.provenance.refined.residual_px.after.median)
        @test resid < 0.15                                    # the detections fit the division camera at sub-0.1 px
        check_golden("refined.f_px", fpx, 0.005; rel = true)
        check_golden("refined.cx_px", Float64(Kc[1, 3]), 3.0)
        check_golden("refined.cy_px", Float64(Kc[2, 3]), 3.0)
        check_golden("refined.k1", pp.k[1], 0.01)
        check_golden("refined.k2", pp.k[2], 0.01)
        check_golden("refined.residual_median_px", resid, 0.03)
    end

    g35 = _image_start(bblens, (35, 35)).camera
    @testset "3 the grid init sampled from the camera (GridStart) vs the division camera" begin
        angs = Float64[]
        for x in 0.05:0.05:0.95, y in 0.05:0.05:0.95
            p = SVector(x * RW, y * RH); (in_domain(bblens, p) && hypot(x - 0.5, y - 0.5) <= 0.4) || continue
            push!(angs, angle_mrad(chief_ray(g35, p), chief_ray(bblens, p)))
        end
        @info "grid init vs the division camera [mrad]" median = median(angs) p95 = quantile(angs, 0.95) max = maximum(angs) n = length(angs)
        @test median(angs) < 0.05 && quantile(angs, 0.95) < 1.0
    end

    @testset "4 the records' poses → Blender world vs the true poses" begin
        est, info = record_poses_world(tposes)
        @test info.board_rms_m < 1e-9
        for (k, nm) in enumerate(VIEWS)
            @test haskey(est, k)                               # frames are numbered in image order
            haskey(est, k) || continue
            R̂, ĉ = est[k]; R, c0 = pose_true(nm); c = c0 + R * SVector(0.0, 0.0, PUPIL_Z)
            rot = rot_deg(R, R̂); cen = 1e3 * norm(ĉ - c)
            @info "pose $nm vs truth (centre shifted to the entrance pupil)" rot_deg = rot centre_mm = cen
            @test rot < 0.5 && cen < 10
            check_golden("pose.$nm.rot_deg", rot, 0.1)
            check_golden("pose.$nm.centre_mm", cen, 3.0)
        end
    end

    # the session's images: the trajectory names them train_0k.png, the fixture's training views are lsc_vk_ref.png
    IMG = mkpath(joinpath(WORK, "session_images"))
    for (k, v) in enumerate(VIEWS)
        f = joinpath(IMG, "train_0$(k).png"); islink(f) || symlink(joinpath(FIX, "lsc_$(v)_ref.png"), f)
    end
    @testset "5 load_session (the photo pipeline's loader of the records)" begin
        for lens in (:camera, :grid)
            d = RC.load_session(SES, IMG; lens, focus_init = :board_center, downsample = 8, texture_downsample = 8)
            @test length(d.poses) == length(VIEWS) && d.names == ["train_0$(k).png" for k in eachindex(VIEWS)]
            @test size(d.targets[1]) == (RH ÷ 8, RW ÷ 8, 3)
            # the records' world is board 291's frame in metres (the 130 mm board standing in for the render's 2 m one): the loader's
            # consistency is what is tested, not the scale
            @info "load_session" lens focus_distance_m = d.𝔱₀ image_distance_m = d.d_s
            @test isfinite(d.𝔱₀) && d.𝔱₀ > 0 && d.𝔱₀ ≈ mean(d.center_distances)
            @test isfinite(d.d_s) && d.d_s > 0
            if lens === :camera
                @test d.lens₀ isa VgtkCore.BoundedCamera && VgtkCore.lens(d.lens₀) isa VgtkCore.Division{2} && VgtkCore.model(d.lens₀).intrinsics.K[1, 1] ≈ fpx / 8   # on the scene's pixels (box-averaged by 8)
            else
                @test d.lens₀ isa VgtkCore.BoundedCamera{<:VgtkCore.CameraModel{<:VgtkCore.AbstractIntrinsics, <:VgtkCore.SplineLens}} && grid_dims(d.lens₀) == (65, 50)
            end
        end
    end

    # the held-out bearing errors of a lens on the test views (true board points, pose refit per view) and the kept fraction per view
    function heldout(lens, f_px)
        errs = Float64[]; kept = Float64[]
        for nm in TEST_VIEWS
            𝐩s, Xs = truth_lattice(nm)
            ok = [in_domain(lens, 𝐩) for 𝐩 in 𝐩s]; 𝐩s = 𝐩s[ok]; Xs = Xs[ok]
            length(Xs) >= 6 || continue
            R, c = pose_true(nm); R0 = SMatrix{3, 3, Float64}(R'); t0 = -R0 * c
            e, kf = holdout_errors(lens, f_px, 𝐩s, Xs, R0, t0)
            append!(errs, e); push!(kept, kf)
        end
        return errs, kept
    end
    @testset "6 held-out: true board points of the test views, pose refit per view, bearing error [px]" begin
        models = ["division" => (bblens, fpx), "cgm35" => (g35, VgtkCore.calibration_matrix(VgtkCore.model(g35))[1, 1])]
        for (key, (lens, f_px)) in models
            errs, kept = heldout(lens, f_px)
            @test !isempty(errs)
            isempty(errs) && continue
            med = median(errs); p95 = quantile(errs, 0.95)
            @info "held-out bearing error [px] on $(length(TEST_VIEWS)) test views" model = key median = med p95 = p95 n = length(errs) kept = mean(kept)
            @test med < 0.6 && mean(kept) > 0.85
            new["eval.$key.median_px"] = med; new["eval.$key.p95_px"] = p95
            RECORD || @test med <= 1.5 * Float64(gold["eval.$key.median_px"]) + 0.02
        end
    end

    @testset "7 radiometric refinement from the records (bin/refine.jl, blob front-end, RC_MIG_REFINE_ITERATIONS LM iterations, default 2)" begin
        if REFINE && !isfile(joinpath(SES, "observations.json"))
            # the blob front end reads its detections only from the session's IO observations record (owner D4, 2026-11-15) and these
            # records carry none yet: the refinement is refused by name until the producer (CaliBlobs' session records, or an IO
            # importer of BlobBoards' export) writes observations.json — the upstream follow-up of RC 0.11.0
            @test_broken isfile(joinpath(SES, "observations.json"))
        elseif REFINE
            OUT = joinpath(WORK, "refine_smoke"); mkpath(OUT); log = joinpath(WORK, "refine_smoke.log")
            # the records' world is in board-291 metres (the 2 × 2 m Blender board rendered as the 130 mm board), so the camera is
            # scaled with it: sensor 36 → 2.34 mm, f 15 → 0.975 mm, f/2.8 as rendered; the focus plane starts through the mean board centre
            RCP = RadiometricCamera
            opts = PipelineOptions(; io = RCP.IOOptions(; images = IMG, out = OUT, linearize = true, checkpoint_each = true),
                                   frontend = RCP.FrontendOptions(; type = RCP.BlobFrontend, session = SES, texture_downsample = 4),
                                   camera = RCP.CameraOptions(; sensor_mm = 2.34, f_mm = 0.975, fnumber = 2.8), grid = RCP.GridOptions(; lattice = RCP.NodeCountLattice(), node_count = 1225),   # 35 · 35 (grid.size retired, RC 0.11.4): the goldens below were recorded on the fitted 35 × 35 lattice — re-record them when this branch first runs
                                   masks = RCP.MaskOptions(; samples_per_view = 3000, seed = 1),
                                   rendering = RCP.RenderingOptions(; N = 4, n_rho = 2, n_phi = 6, n_p = 2))
            stops = RCP._calibration_stops(; max_iter = REFINE_ITERATIONS, max_time = Float64(max(1200, 600 * REFINE_ITERATIONS)))
            config = joinpath(WORK, "refine_smoke.toml")
            open(io -> TOML.print(io, RCP.config_dict(RCP.RadiometricCalibration(RCP.RadiometricRefinement(opts)), stops)), config, "w")
            cmd = Cmd(addenv(`$(Base.julia_cmd()) --project=$HARNESS_ENV $(joinpath(RC_ROOT, "bin", "refine.jl")) --config=$config`,
                             "CUDA_VISIBLE_DEVICES" => get(ENV, "CUDA_VISIBLE_DEVICES", "1")))
            @test run_logged(cmd, log)
            gf = joinpath(OUT, "grid.json"); pf = joinpath(OUT, "parameters.json")
            @test isfile(gf) && isfile(pf)
            if isfile(gf) && isfile(pf)
                pj = VgtkIO.read_record(pf)
                r0 = Float64(pj.cost.init.rel_rms); r1 = Float64(pj.cost.after_linear_solve.rel_rms); r2 = Float64(pj.cost.after_refinement.rel_rms)
                @info "photometric fit" rel_rms_init = r0 rel_rms_linear = r1 rel_rms_final = r2 iterations = Int(pj.iterations)
                @test r0 < 0.5 && r2 <= r1 + 1e-6
                check_golden("refine.rel_rms_init", r0, 0.05)
                check_golden("refine.rel_rms_final", r2, 0.05)
                gr = RC.load_grid(gf); gi = RC.load_grid(joinpath(OUT, "grid_init.json"))
                probe = [SVector(x, y) for x in -0.4:0.05:0.4, y in -0.4:0.05:0.4]
                drift = [angle_mrad(chief_ray(gr, p), chief_ray(gi, p)) for p in probe]
                @info "grid drift after $(REFINE_ITERATIONS) iterations [mrad]" median = median(drift) max = maximum(drift)
                @test median(drift) < 5.0 && maximum(drift) < 50.0
                # what the drift did, against the truth: the held-out bearing error (test views) of the start grid and the refined one.
                # A smoke setting (3 views, N = 4, 3000 px) is not an accuracy run and does NOT converge to the truth: measured 2026-10-28
                # (RC_MIG_REFINE_ITERATIONS=30, the pipeline's default) the held-out median goes 0.135 px at the start → 0.390 after iteration 1
                # → 0.39 to 0.43 px through iteration 30 while the photometric rel-RMS falls 0.120 → 0.101 (the 35 × 35 grid fits the three
                # training views). The pre-port reference stack (rc-baseline, the run its golden was recorded from) moved the same
                # grid further, to 0.854 px after 2 iterations (its grid_init and grid.json through this evaluation), so this is the
                # setting, not a regression of the port; the bound is item 6's held-out bound, not an improvement
                ei, _ = heldout(gi, VgtkCore.calibration_matrix(VgtkCore.model(gi))[1, 1]); er, _ = heldout(gr, VgtkCore.calibration_matrix(VgtkCore.model(gr))[1, 1])
                @info "held-out bearing error [px]: start grid → refined grid" start = median(ei) refined = median(er)
                # the held-out error after every iteration (the per-iteration checkpoints grid_it<N>.json)
                curve = [median(ei)]
                for it in 1:REFINE_ITERATIONS
                    f = joinpath(OUT, "grid_it$(it).json"); isfile(f) || break
                    g = RC.load_grid(f); push!(curve, median(first(heldout(g, VgtkCore.calibration_matrix(VgtkCore.model(g))[1, 1]))))
                end
                @info "held-out bearing error [px] per iteration (0 = start)" curve = round.(curve; digits = 4)
                @test median(er) < 0.6
            end
        else
            @info "refinement smoke skipped (RC_MIG_REFINE=0)"
        end
    end
end
finally
    if RECORD
        new["views"] = VIEWS; new["test_views"] = TEST_VIEWS; new["recorded"] = Libc.strftime("%Y-%m-%d %H:%M:%S", time())
        VgtkIO.write_record(GOLDEN, new)
        @info "golden values written" GOLDEN keys = length(new)
    end
end
