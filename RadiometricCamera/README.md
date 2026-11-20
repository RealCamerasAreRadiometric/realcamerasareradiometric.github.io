# RadiometricCamera.jl

[![CI](https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/actions/workflows/CI.yml/badge.svg)](https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/branch/main/graph/badge.svg)](https://codecov.io/gh/RealCamerasAreRadiometric/realcamerasareradiometric.github.io)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://realcamerasareradiometric.github.io/docs/)

RadiometricCamera calibrates a camera radiometrically, by inverse ray tracing in sensor × aperture space: it renders photographs
of a printed board through a full radiometric camera (chief rays, finite aperture, focus surface, vignetting, lighting, the
board's SVBSDF and warping) and minimises the photometric residual jointly over all of its parameters. It refines the
geometric calibration of a feature-based pipeline (BlobBoards, the generic spline model of Schöps et al.'s pipeline,
Kalibr, OpenCV, any VisualGeometryToolkitCore lens), for researchers who need a camera's lens, focus and vignetting together.

## Installation

Install the package from its repository, where it lives in the subdirectory `RadiometricCamera/` (registration in General is
planned), and the stack package the example uses:

```julia-repl
pkg> add https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io:RadiometricCamera VisualGeometryToolkitCore
```

(`<repository URL>:RadiometricCamera` is Pkg's spelling of the subdirectory: `Pkg.add(url = …, subdir = "RadiometricCamera")`.)

Julia 1.13 or later. A CUDA GPU is optional: the CPU backend runs everything, more slowly.

## Example

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
  [calibration](docs/src/api/calibration.md), [pipeline and options](docs/src/api/pipeline.md).

The paper's project page is <https://realcamerasareradiometric.github.io>; the paper and its supplement are at
<https://realcamerasareradiometric.github.io/paper.pdf>.

RadiometricCamera builds on the VisualGeometryToolkit stack: VisualGeometryToolkitCore's cameras, lenses, generic models and
thin-plate splines, VisualGeometryToolkitIO's records and VisualGeometryToolkitFeatures' image readers. The experiments that reproduce the
paper are `experiments/` beside this package in its repository: staged drivers with configs of record
([experiments/README.md](../experiments/README.md), [experiments/REPRODUCE.md](../experiments/REPRODUCE.md)).

## Citing

T. Harder, F. Seegräber, D. Nakath, H. Sauer, K. Köser, J. Pritts, *Real Cameras are Radiometric*, arXiv preprint
(identifier to follow), and the code: <https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io>.

## License

MIT, see [LICENSE](LICENSE).
