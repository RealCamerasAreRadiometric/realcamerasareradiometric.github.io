# The start's mask test, per view (RC 0.11.5, the 0.11.4 review's item 7a): the test of every pixel of the sampling region
# (`compute_mask(…; external = region)`, RC 0.11.4's start) against the draw's test of the candidates in its keys' order
# (`draw_samples` through `BoardValidity`, RC 0.11.5's), on a synthetic 14 mm-sized scene — pixels tested and seconds (the median of
# `reps` runs after a warm-up), on the CPU and, with `gpu`, on the GPU. The results must agree with the full test's definition.
#
#   julia --project=. test/regression/mask_draw_benchmark.jl [cpu|gpu] [W H n reps]
using RadiometricCamera
using RadiometricCamera: BoardValidity, compute_mask, draw_samples, Focus, GaussianDerivativeEdges, TexturedTarget,
                         RenderSettings, View, rc_backend
import VisualGeometryToolkitCore as VgtkCore
import VisualRadiometryToolkitCore as VrtkCore
using VisualGeometryToolkitCore: exp_so3
using StableRNGs: StableRNG
using StaticArrays: SMatrix, SVector
using Statistics: median
include(joinpath(@__DIR__, "..", "unit", "fixture_board.jl"))
include(joinpath(@__DIR__, "..", "unit", "fixture_camera.jl"))

dev = rc_backend(get(ARGS, 1, "cpu"))
W, H, n, reps = length(ARGS) >= 5 ? parse.(Int, ARGS[2:5]) : (6000, 4000, 27500, 3)
board = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], fixture_board()), [0.0, 0.0, 1.5])
pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.1, 0.05, 0.0))), SVector{3}(SVector(0.05, -0.05, 1.5)); check = false)
f = 0.9 * W                                                              # the board fills about half the frame
camera = RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), f, (W / 2, H / 2), W, H), Focus(3, 1.5, W, H), 0.0386,
                                VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2))   # (the mask reads no vignetting)
st = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
region = [(u - W / 2)^2 + (v - H / 2)^2 < (0.48W)^2 for v in 1:H, u in 1:W]   # an image circle
cand = region .& [isodd(u + v) for v in 1:H, u in 1:W]                 # a RAW photograph's green photosites in it
img = ones(H, W); d = GaussianDerivativeEdges(1.5)

full() = compute_mask(camera, pose, board, st; δ = 0.01, external = region, backend = dev)
t = BoardValidity(View(pose, camera), board, st; δ = 0.01, backend = dev)   # the draw's test: the kernels' view of the camera at its pose
tested = Ref(0)
drawn() = (tested[] = 0; draw_samples(StableRNG(1), ids -> (tested[] += length(ids); t(ids)), cand, img, n; edge_detector = d, device = dev))
mask = full(); s = drawn()                                               # warm-up (compilation)
ids = findall(vec(cand)); U = rand(StableRNG(1), length(ids)); ok = findall(mask[ids])
ref = falses(H, W); ref[ids[ok[partialsortperm(U[ok], 1:min(n, length(ok)))]]] .= true
s == ref || error("the draw through BoardValidity is not the full test's draw")
tf = median([@elapsed(full()) for _ in 1:reps]); td = median([@elapsed(drawn()) for _ in 1:reps])
println("device $(nameof(typeof(dev))); $(W) × $(H), n = $(n): the full-region test $(count(region)) pixels, $(round(tf; digits = 3)) s; ",
        "the draw's test $(tested[]) pixels ($(round(count(region) / tested[]; digits = 1))× fewer), $(round(td; digits = 3)) s ",
        "(the draw itself included: the keys of $(length(ids)) candidates); speed-up $(round(tf / td; digits = 1))×")
