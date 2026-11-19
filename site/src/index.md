```@raw html
---
layout: home
hero:
  name: Real Cameras are Radiometric
  text: Lifting geometric camera models to radiometric ones
  tagline: |-
    Ties Harder, Felix Seegräber, David Nakath, Hendrik Sauer, Kevin Köser, James Pritts
    Preprint, 2026
  image:
    src: /assets/landing/logo.svg
    dark: /assets/landing/logo_dark.svg
    alt: The radiometric camera of a fisheye lens in a cross-section. The rays from a sensor point through the aperture meet in a point on a curved focus surface and spread onto the scene.
  actions:
    - theme: brand
      text: Paper (PDF)
      link: /paper.pdf
    - theme: alt
      text: arXiv (coming soon)
      link: "#Paper"
    - theme: alt
      text: Code
      link: https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io
    - theme: alt
      text: Docs
      link: /docs/
    - theme: alt
      text: BibTeX
      link: "#Citation"
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
<!-- ARXIV: the arXiv URL goes in the "arXiv" action above and in the Paper and Citation sections below -->
```

<!-- TEASER: the paper's figures go here later (owner, 2026-11-10) -->

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

```@raw html
<p><a href="paper.pdf">The paper and its supplement (PDF)</a>, built from the
<a href="https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io/tree/main/paper">manuscript
in the repository</a>. arXiv: coming soon.</p>
```

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

```bibtex
@misc{realcameras2026,
  author = {Harder, Ties and Seegr{\"a}ber, Felix and Nakath, David and Sauer, Hendrik and K{\"o}ser, Kevin and Pritts, James},
  title  = {Real Cameras are Radiometric},
  year   = {2026},
}
```
