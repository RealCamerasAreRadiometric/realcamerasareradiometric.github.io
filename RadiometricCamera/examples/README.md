# Examples

`image_circle_mask.jl` — the imaged region of a camera, measured from its photographs: RadiometricCamera's `measure_image_region`
(VisualGeometryToolkitFeatures' `detect_image_circle`, its fit through VisualGeometryToolkitEstimation) on a directory of
`.png` frames (the image circle, an `Ellipse`), VisualGeometryToolkitCore's sensor `Rect` ∩ that circle (`imaged_region`) and its
`pixel_mask`, overlaid on the frames' per-pixel maximum (the masked part red). Run:
`julia --project=experiments examples/image_circle_mask.jl <frames> [out.png]`.

`radiometric_calibration.jl` / `.ipynb` — the manual-style walkthrough (the notebook is built from the script by
`build_notebook.py` and executed, so its outputs are embedded):

1. **A geometric central model** — three ways in: a calibration session's records (`load_session`, division camera),
   a Schöps `central-generic` calibration (`GridStart` of a `DerivedLattice`), any VisualGeometryToolkitCore lens as Core's pixel camera
   (`BoundedCamera(CameraModel(K, VgtkCore.KannalaBrandt(…)), extent)`); each becomes the refinable generic grid model —
   VisualGeometryToolkitCore's SplineLens camera — with `GridStart(; lattice = ImageLattice(G_x, G_y))`.
2. **The radiometric model around it** — `Focus` (inverse depth `t̂`, TPS), the vignetting (`V`), the camera that holds them with
   the geometric camera (`geometric`) and the aperture radius as one object (`RadiometricCameraModel`; a photograph is the camera at
   a pose), the light `EnvironmentLight` (a `SphericalHarmonicMap` on `(𝔟, r_SH)`), `Material` (Θ_m = `𝛂, 𝛈, μ`), `RenderSettings` —
   and what each symbol is in the paper.
3. **Problem, masks, start prediction** — `calibration_problem` (`CalibrationData`, `CalibrationParameters`), `compute_mask`, `render_all`, `cost`.
4. **Calibration** — the closed-form linear sweep (`RadiometryBlockDescent`: Θ_l, Θ_v, μ), the joint LM over all blocks on the GPU
   (`_lm_tiles!`), the loss curve, `save_grid`.
5. **Validation** — night stars (the paper project's star evaluation), difference images.

Run: `VGTK_DATA_ROOT=<data root> julia --project=. examples/radiometric_calibration.jl` (4 views at 1/8 resolution, 3 LM iterations, a few minutes on a GPU;
output in `examples/output/`, git-ignored). Data: a BlobBoards session of the paper project's published data (`VGTK_DATA_ROOT`, the tier rc_calibration: experiments/data/README.md).

## Plugging in your own geometric model

The radiometric camera needs one thing from a geometric model: its chief ray, VisualRadiometryToolkitCore's `chief_ray(camera, 𝐩)`
(one of its `CONTRACT_GENERICS`; for any VisualGeometryToolkitCore camera its `backproject(camera, 𝐩)`) — the ray direction of the
sensor point 𝐩, in the COLMAP pixels of the image (𝓘 = [0, W] × [0, H], VisualGeometryToolkitCore's `Rect(sensor)`), plus
`in_domain` and, to be refined per view,
VisualGeometryToolkitCore's parameter protocol (`parameter_vector`, `parameter_names`, `with_parameters`, `convert_eltype`). Any
VisualGeometryToolkitCore lens gets all of this as Core's pixel camera (a `BoundedCamera` of `CameraModel(LogicalIntrinsics(K), lens)`, K
in pixels; its LM block is Core's `ParameterManifold` with the box `lm_box` and the image-size coordinate scale `lm_scale`); every other
model is sampled into the generic grid model (Core's SplineLens camera) and refined per tile.
