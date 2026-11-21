# The radiometric camera model

```@meta
CurrentModule = RadiometricCamera
```

The paper's model (§Image Formation, §Radiometric Camera Model), in the paper's symbols, which the code uses as identifiers.

## The model

A sensor position `𝐩 ∈ 𝓘 = [0, W] × [0, H]` (COLMAP pixels of the image, VisualGeometryToolkitCore's `Rect(sensor)`) has the
chief ray `π⁻¹(𝐩; Θ_d)` of the geometric model. Its **radiometric back-projection** from the aperture point 𝐚 of the disc 𝒜 of
radius r is

    π̂⁻¹(𝐩, 𝐚) = (𝔣(𝐩) − 𝐚) / ‖𝔣(𝐩) − 𝐚‖,      𝔣(𝐩) = 𝔭 + t(𝐩) π⁻¹(𝐩)/‖π⁻¹(𝐩)‖,

with the focus point 𝔣 at the distance t(𝐩) along the unit chief direction (t̂ = 1/t a thin plate spline over the sensor, `Θ_f`
its nodal values; 𝔭 the centre of projection). The
radiant flux of pixel 𝐮 is

    Φ_c(𝐮; Θ̄) = ∫_{𝓘_𝐮} V(𝐩; Θ_v) ∫_𝒜 L_c(π̂⁻¹(𝐩, 𝐚), π̂⁻¹(𝐩, 𝐚) − 𝐚; Θ_r) d𝐚 d𝐩,

evaluated with fixed quadrature rules on the pixel `𝓘_𝐮` and on the unit disc 𝒟 (`𝐚 = r â`, `d𝐚 = r² dâ`: the r-dependence of
the domain sits in the integrand, so ∂Φ/∂r is an interior derivative) and a Monte-Carlo estimate of the radiance `L_c` with
common random numbers. The radiometric model `Θ_r`: the vignetting `V(𝐩) = Σ_i N_{i,3}(‖𝐩‖/r_max) P_i` (a radial cubic
B-spline, `Θ_v = {P_i}` with `P_0 = P_1` so V is smooth at the centre, gauge V(0) = 1 held exactly: the centre's
free direction is eliminated from the parametrization), the lighting `L_c^SVSH` on
the sphere `(𝔟, r_SH)` around the board centre with spherical-harmonics coefficients `Θ_l`, the spatially varying Cook–Torrance
SVBSDF `f_c^CT` (GGX D, height-correlated G₂, dielectric Fresnel `F_c`) with `Θ_m = {α_¬ink, α_ink, η_¬ink, η_ink, μ}` blended
by the texture value, and the board warping h (a thin plate spline over the board 𝓑, Core's posed height field and its ray root).

Parameter sets: `Θ_e` poses, `Θ_d` geometric model, `Θ_f` focus, r aperture, `Θ_i = Θ_d ∪ Θ_f ∪ {r}`, `Θ_c = Θ_e ∪ Θ_i`,
`Θ_r = Θ_l ∪ Θ_m` (and `Θ_v`), Θ̄ = all. The calibration minimises

    𝐫(Θ̄) = ϕ − Φ(Θ̄)

by Levenberg–Marquardt (NLLSsolver.jl) with injected **pathwise Jacobians**: forward-mode dual numbers through the same
quadrature that evaluates Φ (the derivative of the discretised objective), the linear inner maps (the bicubic blend of the grid
model's nodes, the thin plate spline of t̂) multiplied with the dual sensitivities, analytic columns for the blocks Φ is linear
in (`Θ_v`, `Θ_l`) or affine in (μ).

## Identifier ↔ paper symbol

The module's docstring holds the table.

```@docs
RadiometricCamera
```

## Data conventions

Pixel coordinates are COLMAP convention (pixel i spans [i, i+1), centre i + 0.5): every sensor position 𝐩 of the package —
the cameras' K, the focus and vignetting splines, the masks, the detections — is in them (`𝐩_of_pixel``(u, v)` of
Julia's 1-based pixel `(u, v)`). Camera frame x right, y down, z forward; a photograph is the `RadiometricCameraModel` at a
pose, VisualGeometryToolkitCore's camera → world `EuclideanMap` (`R = R_cw`, `t` = the camera centre 𝐜; world = the board / session frame, metres). Photographs are
RAW files (VisualGeometryToolkitFeatures' `load_raw_mosaic`, linearised under each photograph's sensor model:
[`load_photo`](@ref RadiometricCamera.load_photo), [`photo_set`](@ref RadiometricCamera.photo_set)) or 16-bit linear PNGs, read by VisualGeometryToolkitFeatures'
`load_linear_image` (the option `io.linearize = "on"` decodes sRGB);
board textures are read linearly too, as the renderers see them.

## Layout

```
src/RadiometricCamera.jl            module (the identifier → paper-symbol table is its docstring)
(VisualRadiometryToolkitCore)       RadiometricCameraModel (Θ_d, Θ_f, r, V: the camera, one object), 𝔣, π̂⁻¹, the world rays; Focus;
                                    the vignetting fields; the sensor domain ℐ and the quadrature rules; EnvironmentLight; Material
                                    {Θ_m} with D, G₂, F_c, visible-normal sampling and f^CT — the generic radiometric types
src/camera/                         the geometric camera's chief-ray models and the kernels' view of the radiometric camera
  models/geometric.jl                 the chief-ray interface: chief_ray, in_domain (on the sensor's COLMAP pixels)
  models/                             pixel_camera.jl: any VisualGeometryToolkitCore lens as Core's pixel camera (a BoundedCamera of a
                                      CameraModel: Lens(DoubleSphere), KannalaBrandt, Division — BlobBoards' model —, the 14-coefficient
                                      chain, …) and its LM box and scale; grid_camera.jl: Core's spline cameras (a SplineLens camera
                                      and its per-tile windows) as chief-ray models, grid_model (any camera → the grid)
  view.jl                             the kernels' View of the camera at a pose (Θ_e): one-line forwards to VisualRadiometryToolkitCore's
                                      maps; flat_vignetting, the calibration's start of V
  vignetting.jl, pose.jl              the vignetting's rules and gauge; the pose
src/target/                         board.jl (𝓑: VisualRadiometryToolkitCore's TexturedTarget — Core's posed planar target + the print:
                                    texture, printed edge, and its own state, the warping h — and the calibration's share: the light
                                    sphere, the warping's free parameters, the distance map's reader), target_radiometry.jl
src/image_formation/                radiance.jl (L_c, the per-basis responses), flux.jl (RenderSettings, Φ_pixel, masks)
src/calibration/                    state.jl (CalibrationState), weighting.jl (the rows' weights: whitened under the sensor
                                    noise, or as measured), precision.jl, closed_form.jl (closed-form Θ_l, Θ_v, μ), smoothness.jl,
                                    problem_spec.jl (the problem from the options), lm/ (per_view.jl: variable blocks, PerViewResidual,
                                    the pathwise computeresjac; per_tile.jl, per_tile_smoothness.jl: the grid models' per-tile blocks),
                                    device/ (workers.jl, flux_kernels.jl, jacobian_kernels.jl: the KernelAbstractions functors, one
                                    kernel per backend), calibrate.jl (refine_calibration!, calibrate), resume.jl
src/start/                          the start: frontends/ (calibration_session.jl, schoeps_run.jl), start_poses.jl, image_region.jl,
                                    focus_start.jl, masks.jl, radiometric_start.jl, start_calibration.jl (StartGeometry, start_calibration, load_calibration_start), prepare.jl,
                                    start_record.jl (the start record)
src/diagnostics/                    monitor.jl, selfcheck.jl, render.jl
src/options/                        tables.jl, toml.jl (PipelineOptions from TOML: pipeline_options, config_dict), device.jl, retired.jl
src/io/                             photographs.jl (photos, BlobBoards sessions), camera_models.jl (lens / grid records, Schöps import),
                                    run_outputs.jl (save_radiometric_calibration), records.jl (RC's record kinds)
bin/refine.jl                       the refinement driver (--config=<toml>; configs/default.toml lists every key)
docs/                               this manual (docs/make.jl, built locally and by CI)
examples/                           the commented walkthrough
../experiments/                     the paper project, beside the package: the staged drivers of the experiments (geometric accuracy with the
                                    chrome board and the synthetic detector bias, vignetting), their configs of record, the figures, the published data's catalog
                                    (experiments/README.md, REPRODUCE.md); experiments/test/pipeline/ runs bin/refine.jl end to end on the
                                    published case data (the tier rc_calibration) against recorded outputs (run_pipeline_checks.sh)
test/                               runtests.jl (`Pkg.test()`: the ReTestItems items of unit/ and regression/, see the testing guide)
  unit/                             the unit tests (`*_tests.jl`; shared setups: rc_testsetups.jl), incl. the Richardson finite-difference check of the pathwise Jacobian (jacobian_richardson_tests.jl)
  regression/                       the golden scenes, their recorded values and the recorded stack (record_golden.jl, recorded_stack.txt)
  integration_blobboards/           a BlobBoards session's records and VisualGeometryToolkitCore against this package (run_integration_tests.sh, one GPU)
  fixtures/                         small fixed inputs of the tests (blob pattern, three rendered fisheye views, OpenCV reference values)
```
