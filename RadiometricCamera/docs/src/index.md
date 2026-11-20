# RadiometricCamera.jl

```@meta
CurrentModule = RadiometricCamera
```

RadiometricCamera calibrates a camera radiometrically, by inverse ray tracing in sensor × aperture space. A geometric camera
model π⁻¹ — the calibration of a feature-based pipeline (BlobBoards, the `central-generic` spline model of Schöps et al.'s
pipeline, Kalibr, a Kannala–Brandt or OpenCV fit, any VisualGeometryToolkitCore lens) — is refined on photographs of a printed board by rendering
the board through the full radiometric camera (chief rays, finite aperture, focus surface, vignetting, lighting, the board's
SVBSDF and warping) and minimising the photometric residual jointly over all parameters. The code uses the paper's symbols as
identifiers.

## What it provides

- **The radiometric camera** around any chief-ray model: [`RadiometricCameraModel`](@ref) (the geometric camera, the focus surface
  [`Focus`](@ref), the aperture radius and the vignetting — VisualGeometryToolkitCore's `TensorProductVignetting` and `RadialVignetting`,
  with the calibration's gauge pin [`center_constraint`](@ref)) as one object, a photograph being the camera at a pose
  ([`render_view`](@ref)), and the radiometric back-projection [`π̂⁻¹`](@ref).
- **The scene**: the printed board [`PrintedTarget`](@ref) and its warping [`BoardSurface`](@ref), its radiometry
  [`TargetRadiometry`](@ref) (the Cook–Torrance SVBSDF [`Material`](@ref) and the printed edge), and VisualGeometryToolkitCore's
  spherical-harmonics `EnvironmentLight`.
- **Image formation**: the radiant flux of a pixel [`Φ_pixel`](@ref) and the render [`render_view`](@ref), by fixed quadrature
  rules on the pixel and the aperture and a Monte Carlo radiance with common random numbers.
- **Calibration**: the [`CalibrationState`](@ref), the closed-form solves of the blocks the flux is linear in
  ([`solve_linear_blocks!`](@ref)), and the joint Levenberg–Marquardt with pathwise Jacobians ([`run_lm!`](@ref),
  [`run_lm_grid!`](@ref)) on the CPU or a CUDA GPU, its residuals whitened when a RAW photograph's sensor model
  has a noise variance law ([`photo_set`](@ref RadiometricCamera.photo_set): VisualGeometryToolkitCore's `SensorModel` resolved by the
  photograph's EXIF).
- **The pipeline**: real sessions in, through a start ([`start_calibration`](@ref), its record [`save_calibration_start`](@ref) /
  [`load_calibration_start`](@ref)), the refinement ([`refine_calibration!`](@ref)) and its records out
  ([`save_radiometric_calibration`](@ref)) — or [`calibrate`](@ref) in one call — configured by one TOML file ([`PipelineOptions`](@ref),
  defaults `configs/default.toml`); the driver is `bin/refine.jl`; the API surface: [API surface](api/surface.md).

Its design is explained in [The radiometric camera model](guide/model.md): the model in the paper's symbols, the identifier ↔
symbol table, and the data conventions (COLMAP pixels, metres, the camera frame).

## Installation

Install the package from its repository, where it lives in the subdirectory `RadiometricCamera/` (registration in General is
planned), and the stack package the example uses:

```julia-repl
pkg> add https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io:RadiometricCamera VisualGeometryToolkitCore
```

(`<repository URL>:RadiometricCamera` is Pkg's spelling of the subdirectory: `Pkg.add(url = …, subdir = "RadiometricCamera")`.)

Julia 1.13 or later. A CUDA GPU is optional: the CPU backend runs everything, more slowly.

## A first example

A 320 × 240 px fisheye camera 1.5 m in front of a printed checkerboard: the example renders the radiant flux of every pixel the
board covers, through the camera's aperture and focus surface.

```julia
using RadiometricCamera, VisualGeometryToolkitCore
using RadiometricCamera: Focus, Material, PrintedTarget, RenderSettings, Scene, compute_mask, render_view, sphere_radius, Φ_pixel, 𝔟

# A printed 2 m × 2 m checkerboard of 8 × 8 squares, and a 320 × 240 px fisheye
# camera (Core's double-sphere lens) 1.5 m in front of it, looking at it.
texture = [Float64(isodd(fld(i - 1, 32) + fld(j - 1, 32))) for i in 1:256, j in 1:256, c in 1:3]
board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], texture; facing = [0.0, 0.0, 1.5])
W, H = 320, 240                               # the image, in pixels
K = CameraCalibrationMat(412.0, 0.0, 0.0, 0.0, 412.0, 0.0, W / 2, H / 2, 1.0)   # focal length 412 px, principal point at the centre
lens = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), Size2(W, H))   # the geometric camera

# The radiometric camera, one object: the fisheye focused at 1.5 m, aperture radius 4 cm, V ≡ 1 (a flat 4 × 4 B-spline surface, Core's).
camera = RadiometricCameraModel(lens, Focus(3, 1.5, W, H), 0.04, TensorProductVignetting((4, 4), Rect(0, 0, W, H); degree = 2))
# The photograph's pose, camera → world, metres: lookat_pose(camera centre, the point it looks at, the world's up direction, which fixes the roll)
pose = lookat_pose(Vec3(0.05, -0.05, 1.5), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
scene = Scene(board, W, H)
light = EnvironmentLight(SphericalHarmonicMap(1.0, 2), 𝔟(board), sphere_radius(𝔟(board), [pose.t]))   # Core's light, L ≡ 1
settings = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
mask = compute_mask(camera, pose, scene, settings; δ = 0.02, backend = RadiometricCamera.rc_backend("cpu"))   # the device: rc_backend of a compute tag (public, not exported)
img = render_view(1, camera, pose, Material(), light, scene, settings, mask)   # photograph 1 (its index keys its Monte Carlo samples)
count(mask)                     # 76800 = W × H: the board covers every pixel
Φ = Φ_pixel(160, 120, 1, camera, pose, Material(), light, scene, settings)   # the pixel 𝐮 = (u, v) = (160, 120)
Φ ≈ img[120, 160, :]            # true: the render is Φ_c(𝐮) of each pixel (img is H × W × 3: row v, column u)
```

## Documentation

**Start here**
- [Getting Started](guide/getting-started.md): two rendered photographs, and the lighting, vignetting and material recovered
  from them.

**How to**
- [Calibrate a camera](guide/calibrating.md): plug in a geometric model, calibrate a real session, run the driver
  `bin/refine.jl`.
- [Set up the data and the hardware](guide/setup.md): the clone, what runs where (CPU, GPU), the published data.
- [Run the tests and check a change](guide/testing.md): the suite's runs and the four checks of a change.

**Explanation**
- [The radiometric camera model](guide/model.md): the model, the identifier ↔ paper-symbol table, the code's layout and the
  data conventions.

**Reference**
- [Camera](api/camera.md), [Target and lighting](api/target.md), [Calibration](api/calibration.md),
  [Pipeline and options](api/pipeline.md).

## Role in the stack

RadiometricCamera builds on the VisualGeometryToolkit stack: VisualGeometryToolkitCore's cameras, lenses, spline cameras
(`SplineLens`) and thin-plate splines, VisualGeometryToolkitIO's records,
VisualGeometryToolkitEstimation's fits and VisualGeometryToolkitFeatures' image readers. The experiments that reproduce the
paper are `experiments/` beside the package in its repository: staged drivers with configs of record, their inputs the published data tiers.

## Citing

T. Harder, F. Seegräber, D. Nakath, H. Sauer, K. Köser, J. Pritts, *Real Cameras are Radiometric*, arXiv preprint
(identifier to follow), and the code: <https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io>.

```@raw html
<p>This manual is published on the paper's <a href="../">project page</a>, beside
<a href="../paper.pdf">the paper and its supplement (PDF)</a>.</p>
```
