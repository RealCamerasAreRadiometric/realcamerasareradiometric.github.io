# Getting Started

```@meta
CurrentModule = RadiometricCamera
```

This tutorial renders two photographs of a printed board through a radiometric camera whose lighting, vignetting and material
are known, then starts a calibration from wrong values of those three blocks and recovers them from the photographs. Every
block below runs when the documentation is built.

## The board and the camera

The board is a printed 2 m × 2 m checkerboard (a 256 × 256 texture of 8 × 8 squares, linear values in [0, 1]). The geometric camera
`geometric` is VisualGeometryToolkitCore's pixel camera: a `BoundedCamera` of a `CameraModel` whose K is in the image's COLMAP pixels,
over a double-sphere fisheye lens; any Core lens plugs in the same way. The image is `sz` = 320 × 240 pixels (the
`BoundedCamera`'s `Size2`: the sensor the camera images onto), small enough that this page runs in seconds; a real calibration
runs at the sensor's full resolution. The principal point is the image's centre and the focal length is 412 pixels.

Every kernel runs on the GPU when one is functional, once its package is loaded (`using CUDA`; [`default_backend`](@ref)), and
otherwise on the CPU, with a warning that the run will be very slow; KernelAbstractions' `backend = CPU()` chooses the CPU. On a GPU the kernels
compute in Float32, so the checks below hold to Float32's precision on either device.

```@example tutorial
using CUDA                                      # the GPU, when there is one: RadiometricCamera's default device
using RadiometricCamera, VisualGeometryToolkitCore, VisualRadiometryToolkitCore
using VisualGeometryToolkitCore: cost, centroid   # Core's public `cost` (not exported): ½ Σ‖Φ − ϕ‖² of the state, here; a shape's `centroid`: the board's centre
using VisualRadiometryToolkitCore: Focus, Material, with_vignetting, with_facing   # the generic radiometric layer: the focus surface, the material, a camera's parts, the board's centre
using RadiometricCamera: RenderSettings, compute_mask, render_view, all_parts, RadiometryBlocks, JointRadiometry, sphere_radius

texture = [Float64(isodd(fld(i - 1, 32) + fld(j - 1, 32))) for i in 1:256, j in 1:256, c in 1:3]
board = with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], texture), [0.0, 0.0, 1.5])
sz = Size2(320, 240)                              # the image, in pixels
K = CameraCalibrationMat(412.0, sz)
geometric = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), sz)
settings = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
nothing # hide
```

[`RenderSettings`](@ref) fixes the quadrature: `n_𝐩 × n_𝐩` nodes on each pixel, `n_ρ × n_φ` on the unit aperture disc, and `N`
Monte Carlo samples of the radiance (common random numbers: the same samples at every evaluation, so the flux is a smooth,
deterministic function of the parameters).

## Two poses

The camera takes two photographs, from two poses (camera → world, metres) around the board.

```@example tutorial
# lookat_pose(camera centre, the point it looks at — the board's centre, the world's up direction, which fixes the roll)
poses = [lookat_pose(Vec3(0.3cos(k), 0.3sin(k), 1.4 + 0.1k), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)) for k in 1:2]
b = centroid(board)
r_SH = sphere_radius(b, [p.t for p in poses])
nothing # hide
```

The lighting lives on the sphere around the board's centre (Core's `centroid`, the paper's 𝔟) of radius `r_SH`, which [`sphere_radius`](@ref) chooses to hold
every camera centre (the pipeline's `sphere_radius(board, centres)` also every corner of the page, so that no light ray from the
board misses the sphere).

## The truth, and its photographs

The true radiometric camera is one object, a `RadiometricCameraModel`: the geometric camera `geometric`, a focus surface at
1.5 m (`Focus`: a constant inverse focus distance on a 3 × 3 node grid of the sensor), an aperture radius of 4 cm (no image
distance: the defocus is the focus surface's) and a vignetting that falls off quadratically from the sensor's centre. The true
lighting is a coloured spherical-harmonics field of degree 2, and the true material has the diffuse weight μ = 0.4. Each
photograph is the camera at one of the poses: [`compute_mask`](@ref) and [`render_view`](@ref) take the camera and the pose (and
the photograph's index `k`, which keys its Monte Carlo samples).

```@example tutorial
xs = range(-0.5, 0.5; length = 4)
vignetting_true = VisualRadiometryToolkitCore.with_control_points(TensorProductVignetting((4, 4), Rect(extent(geometric)); degree = 2), [1 - 0.4 * (x^2 + y^2) for x in xs, y in xs])
camera_true = RadiometricCameraModel(geometric; focus = 1.5, aperture_radius = 0.04, vignetting = vignetting_true)   # focused at 1.5 m (Focus(geometric, 1.5)), aperture radius 4 cm
a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2)
light_true = EnvironmentLight(SphericalHarmonicMap(a, 2), b, r_SH)   # VisualRadiometryToolkitCore's light: degree 2 per channel on the sphere (b, r_SH)
material_true = Material(; diffuse_weight = 0.4)
masks = [compute_mask(camera_true, p, board, settings; δ = 0.02) for p in poses]
photos = [render_view(k, camera_true, poses[k], material_true, light_true, board, settings, masks[k]) for k in 1:2]
size(photos[1])
```

## Calibrating the radiometry

A calibration is a refinement problem, `RefineProblem(f, u0)` ([`calibration_problem`](@ref)): its data `f` (a
[`CalibrationData`](@ref): the board, the photographs ϕ and their masks) and its parameters `u0` (a
[`CalibrationParameters`](@ref): one camera, the poses of its photographs, the lighting and the material).
Here the geometry is known: the parameters start from the true camera with a flat vignetting (`with_vignetting`), a grey
lighting and the wrong weight μ = 0.7; Core's `cost` is ½ Σ‖Φ − ϕ‖² and the relative RMS residual.

```@example tutorial
camera_start = with_vignetting(camera_true, TensorProductVignetting((4, 4), Rect(extent(geometric)); degree = 2))   # V ≡ 1
(; f, u0) = calibration_problem(board, settings, camera_start, poses, Material(; diffuse_weight = 0.7), EnvironmentLight(SphericalHarmonicMap(0.5, 2), b, r_SH), photos;
                                masks)
cost(f, u0)
```

The flux is linear in the lighting `Θ_l` with the vignetting held, linear in the vignetting `Θ_v` with the lighting held
(bilinear in the two together) and affine in μ, so each block alone has a closed-form solve. Block by block, the solves creep
along the coupling of `Θ_l` and `Θ_v`; [`JointRadiometry`](@ref) solves the three together instead — VisualGeometryToolkitCore's
`solve` of [`RadiometryBlocks`](@ref) on the data, the parameters and their parts (the pixel quadratures run once, [`all_parts`](@ref): every
[`PixelParts`](@ref)), a Levenberg–Marquardt on those parts at the `damping` given (here Levenberg's λ·I, a
[`RadiometricRefinement`](@ref)'s default). It writes the blocks into the parameters and reproduces the photographs exactly:

```@example tutorial
r = solve(EstimationProblem(RadiometryBlocks, (f, u0, all_parts(f, u0))), JointRadiometry(; damping = VisualGeometryToolkitCore.LevenbergDamping()))
successful_retcode(r), cost(f, u0)[2] < 1e-4, isapprox(u0.mat.diffuse_weight, 0.4; rtol = 1e-4)   # Float32's precision on a GPU (the CPU's Float64 reaches ~1e-12)
```

The vignetting and the lighting come back up to their common gauge: the parameters pin V = 1 at the sensor's centre, and the
lighting absorbs the true field's value there.

## Next

The geometric blocks — the poses `Θ_e`, the lens `Θ_d`, the focus `Θ_f` and the aperture radius r — enter the flux nonlinearly.
The calibration, [`RadiometricCalibration`](@ref) (VisualGeometryToolkitCore's `solve`), refines them jointly with the radiometry
by Levenberg–Marquardt with pathwise Jacobians — per view in a parametric camera's own parameters, tile by tile for the generic grid
model. [Calibrate a camera](calibrating.md) does this on real sessions; [The radiometric camera model](model.md) states the model these blocks belong to.
