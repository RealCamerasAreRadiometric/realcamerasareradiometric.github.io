```@raw html
---
layout: home
hero:
  name: Real Cameras are Radiometric
  text: A radiometric camera model for calibration by inverse rendering
  tagline: |-
    Authors to be announced
    CVPR 2027 (under review)
  image:
    src: /assets/landing/icon.svg
    dark: /assets/landing/icon_dark.svg
    alt: The image circle of a fisheye lens, its brightness falling off towards the rim, under the faint lattice of a calibration grid.
  actions:
    - theme: brand
      text: Paper
      link: "#"
    - theme: alt
      text: BibTeX
      link: "#BibTeX"
    - theme: alt
      text: VisualGeometryToolkit
      link: https://github.com/VisualGeometryToolkit
features:
  - icon: 📷
    title: One radiometric camera around a geometric model
    details: The model extends the classical pinhole model by a lens and an aperture to capture the radiometric properties of real cameras — vignetting, defocus and distortion.
  - icon: 🔁
    title: Calibration by inverse rendering
    details: Radiometric and geometric camera parameters are inferred by inverse rendering of real images, from the light energy the sensor measures rather than from features found by a detector.
  - icon: 🎯
    title: Detector biases revealed
    details: The evaluation reveals systematic biases in the AprilGrid and Siemens-star detectors used by today's calibration tools, and the method outperforms the state of the art in geometric calibration accuracy.
---
```

```@raw html
<!-- AUTHORS: owner to supply the author list and affiliation before publishing -->
<!-- PAPER: the arXiv URL goes in the "Paper" action above (link: "#") and in the BibTeX below -->
```

<!-- TEASER: the paper's detector-bias figure goes here after the CVPR decision (owner, 2026-11-10) -->

## Abstract

The pinhole camera model has been proven extremely useful for 3D computer vision based on
geometric primitives such as points, lines or conics. Real cameras however do not
directly measure the position of such primitives. Rather, the geometric pinhole paradigm
requires an intermediate step that searches the direct light energy signal collected by
the sensor (the image) for discrete primitives, e.g. a feature detector. These detectors
can be biased, fail under certain conditions, and converting the original photon/sensor
noise to some “feature uncertainty” is challenging. Inspired by forward models from
computer graphics and direct SLAM methods, we propose a novel camera model for computer
vision that extends the classical pinhole model by a lens and aperture in order to
capture radiometric properties of real cameras. We demonstrate that the model can be
employed in inverse rendering approaches to infer radiometric and geometric camera
parameters from real images of different lenses that exhibit vignetting, defocus, a
capability that current differentiable rendering frameworks are lacking, and argue that
the model could be used for many other vision and photogrammetry problems with real
lenses. Besides outperforming the state-of-the-art in geometric calibration accuracy
(and yielding vignetting & depth-of-field calibration on top), our evaluation also
reveals systematic biases in april(-grid) and siemens star detectors as commonly
employed in today's calibration tools.

## BibTeX

```bibtex
@inproceedings{realcameras2027,
  author    = {Authors to be announced},
  title     = {Real Cameras are Radiometric},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2027},
}
```
