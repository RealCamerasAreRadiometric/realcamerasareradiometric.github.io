# cvfisheye_record.jl: OpenCV's fisheye calibration (the calibration.json of opencv_calibrate_schoeps.py or opencv_calibrate_aprilgrid.py
# with --model fisheye: K, dist = k1 … k4, image_size; OpenCV's pixel frame) as a VisualGeometryToolkitIO camera record of one camera
# `:cam` — T. Harder's frontends_history/scripts/cvfisheye_record.jl (tag v0.10.0) on the current stack: cv::fisheye's model
# θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) is Core's `KannalaBrandt(k1, k2, k3, k4)`, the calibration matrix moved to the COLMAP
# frame by Core's `CameraModel(K; lens, pixel_frame = :opencv)`; bounded by the image, written by RadiometricCamera's `save_lens`
# (the record schema IO writes today), read back by `load_camera` and compared with OpenCV's formula, written out below, at the
# directions that land on the sensor. ocv_record.jl is its counterpart for cv2.calibrateCamera's models.
#   julia --project=experiments experiments/tools/cvfisheye_record.jl <calibration.json> <out record.json> <model label> [<note>]
using RadiometricCamera, JSON, StaticArrays, LinearAlgebra
import VisualGeometryToolkitCore as VgtkCore
src, out, label = ARGS[1], ARGS[2], ARGS[3]; note = length(ARGS) >= 4 ? ARGS[4] : ""
j = JSON.parsefile(src); W, H = Int.(j["image_size"])
j["model"] == "fisheye" || error("$src is a calibration of model \"$(j["model"])\", not of cv2.fisheye.calibrate: ocv_record.jl writes its record")
K = SMatrix{3, 3, Float64}(reduce(hcat, [Float64.(collect(r)) for r in j["K"]])')
k = Float64.(collect(j["dist"]))
length(k) == 4 || error("cv2.fisheye.calibrate gives four coefficients; $src has $(length(k))")
cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel(K; lens = VgtkCore.KannalaBrandt(k...), pixel_frame = :opencv), VgtkCore.Size2(W, H))
mkpath(dirname(out))
RadiometricCamera.save_lens(out, cam; provenance = (; model = label, source = String(src), note))
m = VgtkCore.model(RadiometricCamera.load_camera("lens:" * out))
Kc = VgtkCore.calibration_matrix(m)                    # the record's calibration matrix: the COLMAP frame
"cv2.fisheye.projectPoints of the direction `X`, through the calibration matrix `Kc`."
function opencv_fisheye(X)
    a, b = X[1] / X[3], X[2] / X[3]; r = hypot(a, b); θ = atan(r)
    s = r > 0 ? θ * (1 + k[1] * θ^2 + k[2] * θ^4 + k[3] * θ^6 + k[4] * θ^8) / r : 1.0
    return SVector(Kc[1, 1] * s * a + Kc[1, 3], Kc[2, 2] * s * b + Kc[2, 3])
end
errs = Float64[]
for θ in deg2rad.(0:5:85), φ in deg2rad.(0:30:330)
    X = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)); p = opencv_fisheye(X)
    (0 < p[1] < W && 0 < p[2] < H) || continue
    push!(errs, norm(SVector{2, Float64}(VgtkCore.project(m, X)) - p))
end
shift = Kc[SOneTo(2), 3] - K[SOneTo(2), 3]             # the principal point's move from OpenCV's frame
println(label, ": ", out, "  f ", Kc[1, 1], " / ", Kc[2, 2], " px, principal point ", Tuple(Kc[SOneTo(2), 3]), " (OpenCV's + ", Tuple(shift),
        "), k ", Tuple(k), "; record against OpenCV's fisheye formula at ", length(errs), " directions: max ", maximum(errs), " px")
maximum(errs) < 1e-6 || error("the record does not reproduce OpenCV's fisheye model (max $(maximum(errs)) px)")
