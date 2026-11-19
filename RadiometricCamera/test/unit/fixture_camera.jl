using RadiometricCamera: lens
# Core's pixel cameras for the test scenes (included where the fixture board is: a `@testsetup` module cannot use another): the module
# that includes this file has VisualGeometryToolkitCore as `VgtkCore`.

"""`cam(lens, f, c, W, H; f_y = f)`: Core's pixel camera of a `W × H` image — K with the focal lengths `f`, `f_y` and the principal
point `c` in COLMAP pixels over `lens` — bounded by the image's extent."""
cam(lens, f, c, W, H; f_y = f) =
    VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(VgtkCore.SMatrix{3, 3}(f, 0, 0, 0, f_y, 0, c[1], c[2], 1))), lens),
                           VgtkCore.Size2(Int(W), Int(H)))
"""`camera_of(lens, θ, W, H)`: the pixel camera over `lens`'s family with the coefficients `θ[5:end]` and the pinhole `θ[1:4] =
(f_x, f_y, c_x, c_y)` (pixels) — a camera as a function of one vector (ForwardDiff, finite differences)."""
camera_of(lens, θ, W, H) = cam(VgtkCore.with_parameters(lens, θ[5:end]), θ[1], (θ[3], θ[4]), W, H; f_y = θ[2])
"""`pinhole_θ(camera)`: `(f_x, f_y, c_x, c_y)` of its K (pixels) — `camera_of`'s first four entries."""
pinhole_θ(c) = (K = VgtkCore.model(c).intrinsics.K; VgtkCore.SVector(K[1, 1], K[2, 2], K[1, 3], K[2, 3]))
"""`saddle(𝐛)`: the test scenes' saddle-shaped board warping height at the page fraction `𝐛` (m): its nodal heights are
`[saddle(𝐛) for 𝐛 in HEIGHT_NODES]` (a pure bending, no affine part to speak of)."""
# HOIST: the saddle is Core's height-surface fixture's too (VisualGeometryToolkitCore test/fixtures/height_surface/generate.jl
# `sad`, a copy of this one); it moves into Core's synthetic module (wave item "Core synthetic: the saddle board height") and
# this test then calls it (DUPLICATION.toml names the item).
saddle(𝐛) = (q = 𝐛 .- 0.5; 0.08 * q[1]^2 - 0.04 * q[1] * q[2] + 0.06 * q[2]^2)
"""`updated_vars(vars, j, u)`: the residual's variables `vars` with block `j` stepped by `u` (NLLSsolver's `update`), the others
as they are — the central differences of the Jacobian tests."""
updated_vars(vars, j, u) = ntuple(i -> i == j ? RadiometricCamera.NLLSsolver.update(vars[j], u) : vars[i], length(vars))
"""`unit_increment(n, i, h)`: the step `h` along coordinate `i` of an `n`-vector (a block's `update` increment)."""
unit_increment(n, i, h) = (z = zeros(n); z[i] = h; z)
