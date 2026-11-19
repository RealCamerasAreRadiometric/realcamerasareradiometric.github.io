# Getting Started

```@meta
CurrentModule = RadiometricCamera
```

This tutorial renders two photographs of a printed board through a radiometric camera whose lighting, vignetting and material
are known, then starts a calibration from wrong values of those three blocks and recovers them from the photographs. Every
block below runs when the documentation is built.

## The scene and the camera

The board is a printed 2 m × 2 m checkerboard (a 256 × 256 texture of 8 × 8 squares, linear values in [0, 1]). The camera is
VisualGeometryToolkitCore's pixel camera: a `BoundedCamera` of a `CameraModel` whose K is in the image's COLMAP pixels, over a
double-sphere fisheye lens; any Core lens plugs in the same way.

```@example tutorial
using RadiometricCamera, VisualGeometryToolkitCore
using VisualGeometryToolkitCore: cost   # Core's public `cost` (not exported): ½ Σ‖Φ − ϕ‖² of the state, here
using RadiometricCamera: Focus, Material, PrintedTarget, RenderSettings, Scene, View, compute_mask, render_view, solve_linear_blocks!, sphere_radius, 𝔟

texture = [Float64(isodd(fld(i - 1, 32) + fld(j - 1, 32))) for i in 1:256, j in 1:256, c in 1:3]
board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], texture; facing = [0.0, 0.0, 1.5])
K = CameraCalibrationMat(30.9, 0.0, 0.0, 0.0, 30.9, 0.0, 12.0, 12.0, 1.0)
camera = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), Size2(24, 24))
scene = Scene(board, 24, 24)
settings = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
nothing # hide
```

[`RenderSettings`](@ref) fixes the quadrature: `n_𝐩 × n_𝐩` nodes on each pixel, `n_ρ × n_φ` on the unit aperture disc, and `N`
Monte Carlo samples of the radiance (common random numbers: the same samples at every evaluation, so the flux is a smooth,
deterministic function of the parameters).

## Two views

Two poses (camera → world, metres) around the board, each a [`View`](@ref) of the same camera, focused at 1.5 m
([`Focus`](@ref): a constant inverse focus distance on a 3 × 3 node grid of the sensor), with an aperture radius of 4 cm (no image
distance: the defocus is the focus surface's).

```@example tutorial
poses = [inv(lookat(Vec3(0.3cos(k), 0.3sin(k), 1.4 + 0.1k), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))) for k in 1:2]
views = [View(p, camera, Focus(3, 1.5, 24, 24), 0.04) for p in poses]
b = 𝔟(board)
r_SH = sphere_radius(b, [p.t for p in poses])
nothing # hide
```

The lighting lives on the sphere around the board's centre 𝔟 of radius `r_SH`, which [`sphere_radius`](@ref) chooses to hold
every camera centre (the pipeline's `sphere_radius(board, centres)` also every corner of the page, so that no light ray from the
board misses the sphere).

## The truth, and its photographs

The true vignetting falls off quadratically from the sensor's centre, the true lighting is a coloured spherical-harmonics field
of degree 2, and the true material has the diffuse weight μ = 0.4.

```@example tutorial
xs = range(-0.5, 0.5; length = 4)
vignetting_true = VisualGeometryToolkitCore.with_control_points(TensorProductVignetting((4, 4), Rect(0, 0, 24, 24); degree = 2), [1 - 0.4 * (x^2 + y^2) for x in xs, y in xs])
a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2)
light_true = EnvironmentLight(SphericalHarmonicMap(a, 2), b, r_SH)   # Core's light: degree 2 per channel on the sphere (b, r_SH)
material_true = Material(; μ = 0.4)
masks = [compute_mask(v, scene, settings; δ = 0.02, backend = RadiometricCamera.rc_backend("cpu")) for v in views]
photos = [render_view(k, views[k], vignetting_true, material_true, light_true, scene, settings, masks[k]) for k in 1:2]
size(photos[1])
```

## Calibrating the radiometry

A [`CalibrationState`](@ref) holds the camera the views share (a [`RadiometricCameraModel`](@ref)), their poses, the radiometric
blocks and the photographs ϕ. It starts here from a flat vignetting, a grey lighting and the wrong weight μ = 0.7; Core's `cost` is
½ Σ‖Φ − ϕ‖² and the relative RMS residual.

```@example tutorial
s = CalibrationState(scene, settings, copy(views), TensorProductVignetting((4, 4), Rect(0, 0, 24, 24); degree = 2), Material(; μ = 0.7),
                     EnvironmentLight(SphericalHarmonicMap(0.5, 2), b, r_SH), photos; masks)
cost(s)
```

The flux is linear in the lighting `Θ_l` with the vignetting held, linear in the vignetting `Θ_v` with the lighting held
(bilinear in the two together) and affine in μ, so each block alone has a closed-form solve. Block by block, the solves creep
along the coupling of `Θ_l` and `Θ_v`; [`solve_linear_blocks!`](@ref) with `joint = true` solves the three together instead (the
pixel quadratures run once, into [`PixelParts`](@ref); then [`solve_radiometry_gn!`](@ref), a Levenberg–Marquardt on those
parts) and reproduces the photographs exactly:

```@example tutorial
solve_linear_blocks!(s; joint = true)
cost(s)[2] < 1e-10, s.mat.μ ≈ 0.4
```

The vignetting and the lighting come back up to their common gauge: the state pins V = 1 at the sensor's centre, and the
lighting absorbs the true field's value there.

## Next

The geometric blocks — the poses `Θ_e`, the lens `Θ_d`, the focus `Θ_f` and the aperture radius r — enter the flux nonlinearly.
[`run_lm!`](@ref) refines them jointly with the radiometry by Levenberg–Marquardt with pathwise Jacobians, and
[`run_lm_grid!`](@ref) does so for the generic grid model, tile by tile. [Calibrate a camera](calibrating.md) does this on
real sessions; [The radiometric camera model](model.md) states the model these blocks belong to.
