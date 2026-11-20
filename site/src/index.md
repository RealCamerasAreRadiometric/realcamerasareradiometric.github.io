```@raw html
---
layout: home
hero:
  name: Real Cameras are Radiometric
  text: Lifting geometric camera models to radiometric ones
  tagline: |-
    Ties Harder, Felix Seegräber, David Nakath
    Hendrik Sauer, Kevin Köser, James Pritts
    Marine Data Science
    Kiel University
    arXiv 2026
  image:
    src: /assets/landing/logo.svg
    dark: /assets/landing/logo_dark.svg
    alt: The radiometric camera of a fisheye lens in a cross-section. The rays from a sensor point through the aperture meet in a point on a curved focus surface and spread onto the scene.
  actions:
    - theme: brand
      text: Getting started
      link: /docs/guide/getting-started.html
    - theme: alt
      text: Paper
      link: https://arxiv.org/pdf/2611.25265
    - theme: alt
      text: API Reference
      link: /docs/api/camera.html
    - theme: alt
      text: Code
      link: https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io
features:
  - icon: 📷
    title: Any central camera, lifted
    details: The radiometric camera keeps the projection of a geometric model — Brown–Conrady, OpenCV, Kannala–Brandt, Double Sphere or a generic spline — and adds a finite aperture, a free-form focus surface and a generic vignetting function.
  - icon: 🔁
    title: Calibrated by inverse rendering
    details: The camera is calibrated on the RAW photosites of the same images of standard targets that a geometric calibration uses, without a feature detector, by a maximum-likelihood fit of rendered to measured radiant flux.
  - icon: 🎯
    title: More accurate, and physically meaningful
    details: The calibrated cameras have lower reprojection errors on a held-out high-precision target, their vignetting reproduces a measured flat field, and their focus surfaces recover sharp texture from defocused images.
---
```

```@raw html
<!-- The paper's teaser (paper/fig/teaser.tex and teaser-caption.tex): the image is rendered from the paper's source by
     `make -C paper teaser` (CI: the paper job) and copied here by site/build.sh; the caption is the paper's, as text. -->
<figure class="landing-teaser">
  <img src="assets/landing/teaser.png" width="1969" height="942"
       alt="Top: a crop of a panorama stitched without vignetting correction, where the seams between the images show as dark bands, and the same crop with our vignetting correction, where they vanish. Bottom: three lines of a printed page, cut into four parts from left to right: the ground truth, a defocused training photograph, the texture learned with a pinhole camera, which stays blurred, and the texture learned with our radiometric camera, which is sharp.">
  <figcaption>Our radiometric camera model can be applied in downstream tasks like vignetting correction in panorama
  stitching or texture learning in inverse rendering.</figcaption>
</figure>
```

A camera model with an aperture, a focus surface and a vignetting function around any central geometric model, calibrated from the light the sensor measures.

## Abstract

We propose the radiometric camera, a model that lifts any central geometric camera model
to a radiometric one. The lifted model keeps the projection of the geometric model and
adds a finite aperture, a free-form focus surface and a generic vignetting function. We
calibrate it by inverse rendering directly on the relative irradiance of the sensor, from
the same images of standard calibration targets used in geometric calibration. Geometric
calibration absorbs the bias of its feature detections. The proposed radiometric
calibration uses no features, and our radiometric cameras have lower reprojection errors
on held-out test sets. The focus surface extends defocus modeling from the thin lens to
any central camera, and we use it in a first multi-view radiometric reconstruction, which
recovers the in-focus reflectance of a texture from defocused images. The vignetting is
estimated jointly and accurately reproduces a flat field measured through a diffuser.
Because every standard camera model lifts to a radiometric one, the radiometric camera
naturally and directly extends generative scene models beyond multi-view geometry to
multi-view radiometric ones, where phenomena such as lighting and attenuation can be
accurately estimated.

## Install

RadiometricCamera requires Julia 1.13 or later. It is the subdirectory `RadiometricCamera/` of the repository:

```julia-repl
pkg> add https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io:RadiometricCamera VisualGeometryToolkitCore https://github.com/VisualRadiometryToolkit/visualradiometrytoolkit.github.io:VisualRadiometryToolkitCore
```

## Getting started

```@raw html
<p>A radiometric camera is one object: a geometric camera (any central model of the VisualGeometryToolkit), a focus surface, an
aperture radius and a vignetting function. A photograph is the camera at a pose. The example below builds a 320 × 240 px fisheye
camera 1.5 m in front of a printed checkerboard and renders the radiant flux of every pixel through the camera's aperture and
focus surface. The <a href="docs/guide/getting-started.html">Getting Started</a> guide continues from here: it renders two
photographs and recovers the lighting, the vignetting and the material from them.</p>
```

```julia
using RadiometricCamera, VisualGeometryToolkitCore, VisualRadiometryToolkitCore
using RadiometricCamera: RenderSettings, compute_mask, render_view, sphere_radius, Φ_pixel, 𝔟
using VisualRadiometryToolkitCore: Material

# A printed 2 m × 2 m checkerboard of 8 × 8 squares, and a 320 × 240 px fisheye
# camera (Core's double-sphere lens) 1.5 m in front of it, looking at it.
texture = [Float64(isodd(fld(i - 1, 32) + fld(j - 1, 32))) for i in 1:256, j in 1:256, c in 1:3]
board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], texture; facing = [0.0, 0.0, 1.5])
W, H = 320, 240                               # the image, in pixels
K = CameraCalibrationMat(412.0, 0.0, 0.0, 0.0, 412.0, 0.0, W / 2, H / 2, 1.0)   # focal length 412 px, principal point at the centre
geometric = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), Size2(W, H))

# The radiometric camera, one object, composed from the geometric camera: focused at 1.5 m, aperture radius 4 cm, and no
# vignetting (VisualRadiometryToolkitCore's NoVignetting, the default: V ≡ 1, no parameters).
camera = RadiometricCameraModel(geometric; focus = 1.5, aperture_radius = 0.04)
# The photograph's pose, camera → world, metres: lookat_pose(camera centre, the point it looks at, the world's up direction, which fixes the roll)
pose = lookat_pose(Vec3(0.05, -0.05, 1.5), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
light = EnvironmentLight(SphericalHarmonicMap(1.0, 2), 𝔟(board), sphere_radius(𝔟(board), [pose.t]))   # VisualRadiometryToolkitCore's light, L ≡ 1
settings = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
mask = compute_mask(camera, pose, board, settings; δ = 0.02, backend = RadiometricCamera.rc_backend("cpu"))   # the device: rc_backend of a compute tag (public, not exported)
img = render_view(1, camera, pose, Material(), light, board, settings, mask)   # photograph 1 (its index keys its Monte Carlo samples)
count(mask)                     # 76800 = W × H: the board covers every pixel
Φ = Φ_pixel(160, 120, 1, camera, pose, Material(), light, board, settings)   # the pixel 𝐮 = (u, v) = (160, 120)
Φ ≈ img[120, 160, :]            # true: the render is Φ_c(𝐮) of each pixel (img is H × W × 3: row v, column u)
```

## Method

A geometric camera model maps a sensor point to one ray through its center of projection.
The radiometric camera keeps that mapping and replaces the pinhole by an aperture:

- **Aperture and focus surface.** The rays from a sensor point through all points of the
  aperture meet in one point of focus. These points form a free-form focus surface, which
  is learned: close to a plane for a wide-angle lens, close to a sphere for a fisheye.
  With a pinhole aperture, the model is the geometric model it started from.
- **Vignetting.** A generic radial function about the principal point replaces the
  cos⁴ falloff of the thin lens, which real lenses do not follow.
- **Calibration.** The images of a calibration target are rendered and fitted to the RAW
  photosites under the noise model of the sensor. The formulation has no boundary terms,
  so every parameter of the camera, including the aperture radius and the focus surface,
  is differentiable, and its gauge freedoms are removed by the parameterization.

## Results

- **Geometric accuracy.** On a held-out high-precision target, the radiometric refinement
  lowers the median reprojection error significantly for 82 % of 28 combinations of
  camera model, lens and calibration target, and the 95th percentile for all of them.
- **Vignetting.** The learned vignetting follows a measured flat field for three lenses,
  with an RMS deviation of 0.009 to 0.042 of the value at the principal point, against
  0.059 to 0.244 for the cos⁴ falloff.
- **Defocus.** Modeling the aperture and the focus surface lowers the median reprojection
  error by 25 % and 16 % against the same refinement with a pinhole aperture.
- **A first multi-view radiometric reconstruction.** From defocused images of a printed
  page, the radiometric camera recovers sharp text where a pinhole camera returns the
  blur of the images.

## Paper

The paper and its supplementary material are on arXiv:
[arXiv:2611.25265](https://arxiv.org/abs/2611.25265) ([PDF](https://arxiv.org/pdf/2611.25265)).
The manuscript's source is in the
[repository](https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/tree/main/paper).

## Code and documentation

```@raw html
<p>The code, the experiments that reproduce the paper and the manuscript are in one repository:
<a href="https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io">RealCamerasAreRadiometric/realcamerasareradiometric.github.io</a>.
The Julia package RadiometricCamera.jl is its subdirectory <code>RadiometricCamera/</code>;
<a href="docs/">its documentation</a> covers the installation, a tutorial, the calibration and the API.
The calibration is built on the <a href="https://github.com/VisualGeometryToolkit">VisualGeometryToolkit</a>.</p>
```

## Data

Coming soon: the RAW images and the configurations of every experiment.

## Related projects

- [BlobBoards](https://blobboards.github.io): robust markers for accurate pose.
- [VisualGeometryToolkit](https://github.com/VisualGeometryToolkit): the geometry stack
  this work is built on.

## Citation

The radiometric camera is described in *Real Cameras are Radiometric*, arXiv 2026
([PDF](https://arxiv.org/pdf/2611.25265), with supplementary material). If you use this
project for your research, please cite:

```bibtex
@article{harder2026radiometric,
  author        = {Harder, Ties and Seegr{\"a}ber, Felix and Nakath, David and
                   Sauer, Hendrik and K{\"o}ser, Kevin and Pritts, James},
  title         = {Real Cameras are Radiometric},
  journal       = {arXiv preprint arXiv:2611.25265},
  year          = {2026},
  eprint        = {2611.25265},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
}
```

## Acknowledgments

James Pritts is funded by Kiel Training for Excellence, an EU Horizon Europe programme
under the Marie Skłodowska-Curie Actions (MSCA), grant agreement No. 101081480, and by KMS
Kiel Marine Science – Centre for Interdisciplinary Marine Science at Kiel University.
