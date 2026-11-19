# ocv_record.jl: an OpenCV calibration (opencv_calibrate_schoeps.py's calibration.json: K, dist, image_size; OpenCV's pixel frame) as a
# VisualGeometryToolkitIO camera record of one camera `:cam` — T. Harder's frontends_history/scripts/ocv_record.jl (tag v0.10.0) on
# the current stack: IO's `opencv_camera(K, dist)` (4 / 5 / 8 / 12 / 14 coefficients; the principal point moved to the COLMAP frame),
# bounded by the image, written by RadiometricCamera's `save_lens` (the record schema IO writes today), read back by `load_camera`
# and a pixel round-tripped. (Owner GO 2026-11-17: the fisheye 30 cm OpenCV-14 start and the zoom 48 mm Brown–Conrady start.)
#   julia --project=experiments experiments/tools/ocv_record.jl <calibration.json> <out record.json> <model label> [<note>]
using RadiometricCamera, JSON
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
src, out, label = ARGS[1], ARGS[2], ARGS[3]; note = length(ARGS) >= 4 ? ARGS[4] : ""
j = JSON.parsefile(src); W, H = Int.(j["image_size"])
K = Matrix{Float64}(reduce(hcat, [Float64.(collect(r)) for r in j["K"]])')
dist = Float64.(collect(j["dist"]))
cam = VgtkCore.BoundedCamera(VgtkIO.opencv_camera(K, dist), VgtkCore.Size2(W, H))
mkpath(dirname(out))
RadiometricCamera.save_lens(out, cam; provenance = (; model = label, source = String(src), note))
c = RadiometricCamera.load_camera("lens:" * out)
back = VgtkIO.opencv_parameters(VgtkCore.model(c))
println(label, ": ", out, "  K max |Δ| ", maximum(abs.(Matrix(back.K) .- K)), ", dist max |Δ| ", maximum(abs.(collect(back.dist) .- dist)),
        " (", length(dist), " coefficients)")
