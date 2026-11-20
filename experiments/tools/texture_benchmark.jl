# =============================================================================
# tools/texture_benchmark.jl — the page solve of the texture test (src/texture_recovery.jl) timed on a synthetic sheet: the rows
# (`page_rows`: the footprint kernel and the operators' assembly), the penalized operator on the device, the CGLS solves
# (`page_solves`) and the refine stage as a whole (`run_page`). Not a driver of the paper: a measurement of the code.
#
#   julia -t 8 --project=experiments experiments/tools/texture_benchmark.jl [key=value …]
#     backend=cpu|gpu   the device (`rc_backend`)                                                [cpu]
#     scale=4           the image and the page's texels per mm (image 240·scale × 180·scale)     [4]
#     views=10          the photographs                                                           [10]
#     fraction=0.5      the share of each photograph's masked pixels sampled                      [0.5]
#     out=<file.json>   the record                                                                [none]
#     label=<text>      a label for the record                                                    [""]
#
# The sheet is the tests' (60 × 84 mm, a design of smooth blobs and bars), photographed defocused from 0.30–0.42 m by a double-sphere
# camera focused at 0.45 m with a 4 mm aperture; the photographs' values are the rows' own prediction of the design. Reported per
# stage: seconds (the first call, compilation included, and a second, warm), and the numbers a change must keep: the rows, the
# nonzeros, the iterations and a checksum of the textures.
# =============================================================================
using DrWatson
@quickactivate "experiments"
using LinearAlgebra, Statistics, Printf
using StaticArrays: SVector, SMatrix
import PNGFiles, JSON
using ColorTypes: RGB, N0f16
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("texture_recovery.jl"))

bench_option(key, default) = (i = findfirst(a -> startswith(a, key * "="), ARGS); i === nothing ? default : String(split(ARGS[i], "="; limit = 2)[2]))
const BENCH_BACKEND = bench_option("backend", "cpu")
const BENCH_SCALE = parse(Int, bench_option("scale", "4"))
const BENCH_VIEWS = parse(Int, bench_option("views", "10"))
const BENCH_FRACTION = parse(Float64, bench_option("fraction", "0.5"))
const BENCH_OUT = bench_option("out", "")
const BENCH_LABEL = bench_option("label", "")
const BENCH_DEVICE = RC.rc_backend(BENCH_BACKEND; inline = false, fallback = false)

function texture_benchmark_scene(dir)
    sc = BENCH_SCALE; W, H = 240sc, 180sc
    cam0 = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(600.0sc, 0, 0, 0, 600.0sc, 0, W / 2, H / 2, 1))),
                                                       VgtkCore.Lens(VgtkCore.DoubleSphere(0.2, 0.3))), VgtkCore.Size2(W, H))
    focus0 = RC.Focus(RC.FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), fill(1 / 0.45, 9))
    camera = RC.RadiometricCameraModel(cam0, focus0, 0.004, VgtkCore.TensorProductVignetting((3, 3), VgtkCore.Rect(0, 0, W, H); degree = 2))
    sett = RC.RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8)
    TW, TH = 60sc, 84sc
    design = [0.35 + 0.3 * sin(0.21 * i / sc) * cos(0.17 * j / sc) + 0.25 * (((i - 1) ÷ 6sc + (j - 1) ÷ 9sc) % 2) for i in 1:TW, j in 1:TH]
    sheetdir = joinpath(dir, "sheet"); mkpath(sheetdir)
    PNGFiles.save(joinpath(sheetdir, "texture.png"), [RGB{N0f16}(design[i, j], design[i, j], design[i, j]) for j in 1:TH, i in 1:TW])
    save_sheet_target(sheetdir; page_mm = (60.0, 84.0), blobs_mm = Dict(1 => SVector(5.0, 5.0)))
    sheet = read_sheet(sheetdir)
    rad = RC.TargetRadiometry(RC.Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4), RC.BoardSurface(; edge = [0.021, 0.034]))
    target = RC.RadiometricTarget(nothing, zero(SVector{RC.HEIGHT_NC, Float64}), 1.0, rad)
    Rx(a) = SMatrix{3, 3}(1.0, 0, 0, 0, cos(a), sin(a), 0, -sin(a), cos(a)); Ry(a) = SMatrix{3, 3}(cos(a), 0, -sin(a), 0, 1.0, 0, sin(a), 0, cos(a))
    poses = map(1:BENCH_VIEWS) do k
        C = SVector(0.03 + 0.004 * cos(1.7k), 0.042 + 0.003 * sin(2.3k), -(0.30 + 0.12 * mod(0.37k, 1.0)))
        R = Rx(0.05 * sin(1.1k)) * Ry(0.05 * cos(0.7k))
        (@sprintf("S%02d.png", k), R, -(R * C))
    end
    imgdir = joinpath(dir, "photos"); mkpath(imgdir)
    foreach(p -> PNGFiles.save(joinpath(imgdir, p[1]), zeros(RGB{N0f16}, H, W)), poses)
    cc = RC.CalibratedCamera(camera, RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB), sett)
    ctx = sheet_scene(cc, target, sheet, joinpath(sheetdir, "texture.png"), imgdir, poses, W, H)
    return ctx, design
end

timed(f) = (t = time(); v = f(); (v, time() - t))

function texture_benchmark()
    dir = mktempdir()
    ctx, design = texture_benchmark_scene(dir)
    K = BENCH_VIEWS; TW, TH = ctx.TW, ctx.TH
    rec = Dict{String, Any}("schema" => "rc/texture_benchmark/v1.0", "label" => BENCH_LABEL, "backend" => BENCH_BACKEND, "scale" => BENCH_SCALE, "views" => K,
                            "fraction" => BENCH_FRACTION, "threads" => Threads.nthreads(), "texels" => [TW, TH], "image" => [ctx.W, ctx.H])
    say(k, v) = (rec[k] = v; println(rpad(k, 28), v); flush(stdout))
    samples, t = timed(() -> [sample_pixels(ctx, k, nothing; fraction = BENCH_FRACTION, seed = 11, backend = BENCH_DEVICE) for k in 1:K])
    say("sample_pixels_s", t)
    px = [s.px for s in samples]
    rows_px = [(k, Int(p[1]), Int(p[2])) for k in 1:K for p in px[k]]
    groups = rows_by_photograph(rows_px)
    roles = split_roles(K, 5); role = [roles[r[1]] for r in rows_px]
    blocks = (; fit = role .== :fit, val = role .== :val, test = role .== :test)
    say("rows", length(rows_px))
    for pass in ("first", "warm")
        R, t = timed(() -> page_rows(ctx, rows_px, ctx.camera, blocks, groups; device = BENCH_DEVICE))
        say("page_rows_$(pass)_s", t)
        A = R.blocks.fit.A
        pass == "first" && (say("fit_rows", size(A, 1)); say("fit_nonzeros", nnz(A)); say("fit_nonzeros_sumsq", sum(abs2, nonzeros(A))))
        y = A * vec(design)
        D = VgtkCore.difference_rows((TW, TH), 1)
        op, t = timed(() -> VgtkCore.PenalizedOperator(A, D; device = BENCH_DEVICE))
        say("operator_$(pass)_s", t)
        scale = VgtkCore.penalty_scale(sum(abs2, nonzeros(A)), TW * TH, 1.0)
        sw, t = timed(() -> page_solves(op, y, fill(0.5, TW * TH); scale, weights = [1e-3, 1e-1], rtol = 1e-6, maxit = 300))
        say("page_solves_$(pass)_s", t)
        pass == "first" && (say("iterations", [sw.main.iterations; [r.iterations for r in sw.reg]]); say("texture_checksum", sum(sw.main.x) + sum(r -> sum(r.x), sw.reg)))
    end
    # the refine stage as a whole: the photographs' values the learned rows' prediction of the design
    Rall = page_rows(ctx, rows_px, ctx.camera, (; all = trues(length(rows_px))), groups; device = BENCH_DEVICE)
    I = zeros(length(rows_px)); I[Rall.blocks.all.rows] .= Rall.blocks.all.A * vec(design)
    values = [I[groups[k]] for k in 1:K]; weights = [RC.Unweighted() for _ in 1:K]
    cells = select_cells(cell_coverage(px, 3, 5, 20.0), roles; min_fit_views = 1, min_val_views = 1, min_test_views = 1, min_pixels = 20)
    settings = Dict{String, Any}("regularization_weights" => [1e-3, 1e-1], "cgls_rtol" => 1e-6, "cgls_max_iter" => 300, "cell_mm" => 20.0, "centre_rho" => 0.4,
                                 "edge_rho" => 0.6, "pinhole_r_m" => 1e-9, "start" => 0.5)
    for pass in ("first", "warm")
        out = joinpath(dir, "refine_$pass")
        _, t = timed(() -> run_page(ctx, px, roles, cells, nothing, out; settings, device = BENCH_DEVICE, values, weights, log = s -> println("    ", s)))
        say("run_page_$(pass)_s", t)
        if pass == "first"
            tex = sum(Float64, reinterpret(Float32, read(page_files(out, "learned").texture))) + sum(Float64, reinterpret(Float32, read(page_files(out, "learned").reg)))
            say("run_page_texture_checksum", tex)
            r = VgtkIO.read_record(page_files(out, "learned").record; allownan = true)
            say("run_page_test_sse", [Float64(s.test_sse) for s in r.solutions])
            say("run_page_fit_sse", [Float64(s.fit_sse) for s in r.solutions])
        end
    end
    isempty(BENCH_OUT) || (mkpath(dirname(BENCH_OUT)); open(io -> JSON.print(io, rec, 2), BENCH_OUT, "w"))
    return rec
end

texture_benchmark()
