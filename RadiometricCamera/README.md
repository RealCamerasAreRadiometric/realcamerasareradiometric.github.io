# RadiometricCamera.jl

[![CI](https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/actions/workflows/CI.yml/badge.svg)](https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/branch/main/graph/badge.svg)](https://codecov.io/gh/RealCamerasAreRadiometric/realcamerasareradiometric.github.io)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://realcamerasareradiometric.github.io/docs/)

RadiometricCamera calibrates a camera radiometrically, by inverse ray tracing in sensor × aperture space: it renders photographs
of a printed board through a full radiometric camera (chief rays, finite aperture, focus surface, vignetting, lighting, the
board's SVBSDF and warping) and minimises the photometric residual jointly over all of its parameters. It refines the
geometric calibration of a feature-based pipeline (BlobBoards, the generic spline model of Schöps et al.'s pipeline,
Kalibr, OpenCV, any VisualGeometryToolkitCore lens), for researchers who need a camera's lens, focus and vignetting together.
It lifts a Visual Geometry Toolkit camera into VisualRadiometryToolkitCore's `RadiometricCameraModel` and builds on the Visual
Radiometry Toolkit's VisualRadiometryToolkitCore and VisualRadiometryToolkitIO.

## Installation

RadiometricCamera is registered in Julia's General registry, as are the Visual Geometry Toolkit and Visual Radiometry
Toolkit packages it builds on. Add the package, the two stack packages and CUDA.jl that the example uses:

```julia-repl
pkg> add RadiometricCamera VisualGeometryToolkitCore VisualRadiometryToolkitCore CUDA
```

Julia 1.13 or later. A CUDA GPU is the default device once `using CUDA` has loaded its package; without a functional one
every entry point runs on the CPU, much more slowly, and warns once that it does (KernelAbstractions' `backend = CPU()` chooses the CPU without the
warning).

## Example

A 320 × 240 px fisheye camera 1.5 m in front of a printed checkerboard: the example renders the radiant flux of every pixel the
board covers, through the camera's aperture and focus surface.

```julia
using CUDA                                    # the GPU when there is one: the default device (else the CPU with a warning)
using RadiometricCamera, VisualGeometryToolkitCore, VisualRadiometryToolkitCore
using RadiometricCamera: RenderSettings, compute_mask, render_view, sphere_radius, Φ_pixel
using VisualRadiometryToolkitCore: Material, 𝔟

# A printed 2 m × 2 m checkerboard of 8 × 8 squares, and a 320 × 240 px fisheye
# camera (VisualGeometryToolkitCore's double-sphere lens) 1.5 m in front of it, looking at it.
texture = [Float64(isodd(fld(i - 1, 32) + fld(j - 1, 32))) for i in 1:256, j in 1:256, c in 1:3]
board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], texture; facing = [0.0, 0.0, 1.5])
sz = Size2(320, 240)                          # the image, in pixels
K = CameraCalibrationMat(412.0, sz)   # focal length 412 px, principal point at the centre
geometric = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), sz)

# The radiometric camera, one object, composed from the geometric camera: focused at 1.5 m, aperture radius 4 cm, and no
# vignetting (VisualRadiometryToolkitCore's NoVignetting, the default: V ≡ 1, no parameters).
camera = RadiometricCameraModel(geometric; focus = 1.5, aperture_radius = 0.04)
# The photograph's pose, camera → world, metres: lookat_pose(camera centre, the point it looks at, the world's up direction, which fixes the roll)
pose = lookat_pose(Vec3(0.05, -0.05, 1.5), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
light = EnvironmentLight(SphericalHarmonicMap(1.0, 2), 𝔟(board), sphere_radius(𝔟(board), [pose.t]))   # VisualRadiometryToolkitCore's light, L ≡ 1
settings = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
mask = compute_mask(camera, pose, board, settings; δ = 0.02)   # on the default device: the GPU, else the CPU
img = render_view(1, camera, pose, Material(), light, board, settings, mask)   # photograph 1 (its index keys its Monte Carlo samples)
count(mask)                     # 76800 = 320 × 240: the board covers every pixel
Φ = Φ_pixel(160, 120, 1, camera, pose, Material(), light, board, settings)   # the pixel 𝐮 = (u, v) = (160, 120)
isapprox(Φ, img[120, 160, :]; rtol = 1e-4)   # true: the render is Φ_c(𝐮) of each pixel, to Float32 on a GPU (img is 240 × 320 × 3: row v, column u)
```

[Getting Started](docs/src/guide/getting-started.md) continues from here: it renders two photographs and recovers the
lighting, the vignetting and the material from them.

## Documentation

The documentation is online at <https://realcamerasareradiometric.github.io/docs/>, on the project page. It is a Documenter
manual in `docs/`, also built locally: `julia +1.13 --project=docs --startup-file=no docs/make.jl` writes it to
`docs/build/index.html`. Its sources:

- [Getting Started](docs/src/guide/getting-started.md): the tutorial.
- How to: [calibrate a camera](docs/src/guide/calibrating.md) (a geometric model in, the refinement driver `bin/refine.jl`),
  [set up the data and the hardware](docs/src/guide/setup.md), [run the tests and check a change](docs/src/guide/testing.md).
- Explanation: [the radiometric camera model](docs/src/guide/model.md), in the paper's symbols.
- API Reference: [camera](docs/src/api/camera.md), [target and lighting](docs/src/api/target.md),
  [calibration](docs/src/api/calibration.md), [pipeline and options](docs/src/api/pipeline.md),
  [API surface](docs/src/api/surface.md).

The paper's project page is <https://realcamerasareradiometric.github.io>; the paper and its supplement are at
<https://realcamerasareradiometric.github.io/paper.pdf>.

RadiometricCamera builds on the VisualGeometryToolkit stack: VisualGeometryToolkitCore's cameras, lenses, generic models and
thin-plate splines, VisualGeometryToolkitIO's records, VisualGeometryToolkitEstimation's fits and VisualGeometryToolkitFeatures'
image readers. It builds on the Visual Radiometry Toolkit for the generic radiometric layer:
VisualRadiometryToolkitCore's radiometric camera, focus surface, quadrature, material, textured target, vignetting models and light,
and VisualRadiometryToolkitIO's records of the vignetting and the light. The experiments that reproduce the
paper are `experiments/` beside this package in its repository: staged drivers with configs of record
([experiments/README.md](../experiments/README.md), [experiments/REPRODUCE.md](../experiments/REPRODUCE.md)).

## Citing

T. Harder, F. Seegräber, D. Nakath, H. Sauer, K. Köser, J. Pritts, *Real Cameras are Radiometric*, arXiv preprint
(identifier to follow), and the code: <https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io>.

## License

MIT, see [LICENSE](LICENSE).
