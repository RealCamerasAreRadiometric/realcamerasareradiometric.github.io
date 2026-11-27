"""
    RadiometricCamera

Radiometric camera calibration by inverse ray tracing in sensor × aperture space (the paper's model, in the paper's symbols).

The generic radiometric types are VisualRadiometryToolkitCore's (the radiometric camera `RadiometricCameraModel` and its back-projection,
the focus surface, the vignetting fields, the sensor's quadrature rules, the light, the material); this package is the paper's
calibration METHOD on them. A geometric back-projection π⁻¹(𝐩) (a VisualGeometryToolkitCore camera on the sensor's pixels 𝐩 ∈ [0, W] × [0, H], src/camera/)
becomes the radiometric back-projection from the aperture disc 𝒜 of radius r, with the focus point focus_point(𝐩); the radiant flux of a
pixel is evaluated by fixed quadrature rules on the pixel 𝓘_𝐮 and on the unit disc with a Monte-Carlo radiance, and the
calibration minimises 𝐫(Θ̄) over Θ̄ by Levenberg–Marquardt with injected pathwise Jacobians (forward-mode dual numbers through
the same quadrature):

```text
backproject(𝐩, 𝐚) = (focus_point(𝐩) − 𝐚)/‖focus_point(𝐩) − 𝐚‖,   focus_point(𝐩) = 𝔭 + t(𝐩) π⁻¹(𝐩)/‖π⁻¹(𝐩)‖
Φ_c(𝐮)    = ∫_{𝓘_𝐮} V(𝐩) ∫_𝒜 L_c(backproject(𝐩, 𝐚), backproject(𝐩, 𝐚) − 𝐚) d𝐚 d𝐩
𝐫(Θ̄)      = (ϕ − Φ(Θ̄), √(100 Σϕ²/|Θ_v|) (V(0) − 1))
```

Identifier ↔ paper symbol (section):

```text
chief_ray (VRTK Core's), in_domain        π⁻¹(𝐩; Θ_d), the geometric model's chief ray  (Radiometric Camera Model)
RadiometricCameraModel (VRTK Core's)       Θ_d, Θ_f, r and Θ_v                             (Radiometric Camera Model)
View(pose, camera)                         Θ_e + the camera, as the renderer evaluates it  (Radiometric Camera Model)
chief_direction, 𝔣, π̂⁻¹                    π⁻¹/‖π⁻¹‖, focus_point(𝐩), backproject(𝐩, 𝐚)                      (Radiometric Camera Model)
Focus, t̂, 𝔱, FocusSpline (VRTK Core's)     t̂ = 1/t, the thin plate spline (Core's)         (Focus surface)
𝔭                                          the centre of projection 𝔭                     (Radiometric Camera Model)
SensorQuadrature, ApertureQuadrature       the rules on 𝓘_𝐮 and on the unit disc 𝒟        (Image Formation)
VRTK Core's RadialVignetting (P)          V(𝐩) = Σ_i N_{i,p}(‖𝐩 − 𝐜‖/r_max) P_i         (Vignetting)
VRTK Core's EnvironmentLight (𝔟, r_SH, a)  L_c^SVSH on the sphere (𝔟, r_SH), a_{c,l}^m     (Lighting)
Material, cook_torrance                    Θ_m = {α_¬ink, α_ink, η_¬ink, η_ink, μ}, f_c^CT  (SVBSDF and Monte Carlo Estimation)
VRTK Core's ggx_distribution, ggx_masking_shadowing, fresnel_dielectric, visible_normal_density   D, G₂, F_c, D^v   (SVBSDF and Monte Carlo Estimation)
VRTK Core's TexturedTarget, its state's h   𝓑 (VisualGeometryToolkitCore's posed target + the print) and its warping h  (Board Warping)
Φ_pixel, L_c                               Φ_c(𝐮), L_c(𝐱, ω)                              (Image Formation)
PerViewResidual                            ϕ − Φ(Θ̄); V(0) = 1 held by `gauge_basis`        (Calibration Pipeline)
computeresjac, _PartsAt                    the pathwise Jacobian; the Θ_v, Θ_l, μ columns  (Second Order Optimization)
_focus_chain!, _patch_fanout!              inner linear map × ForwardDiff of the outer     (Second Order Optimization)
```

Layout: `camera/` (the geometric models, the kernels' view of the camera at a pose, the vignetting's rules and gauge),
`target/` (𝓑, its warping h, the print's edge), `image_formation/` (the radiance, the flux), `calibration/` (the state,
the closed-form blocks, the problem, the LM's per-view and per-tile blocks in `lm/`, the device kernels in `device/`, the LM driver),
`start/` (the start: front ends, poses, masks, the radiometric start, the start record), `diagnostics/` (the live monitor, the
self-check, the render dumps), `options/` (the options, their TOML reader, the device tags, the retired keys), `io/` (photographs,
camera models, the run's outputs, RC's records).
"""
module RadiometricCamera

# every name imported explicitly (ExplicitImports: no implicit import, none stale)
using LinearAlgebra: LinearAlgebra, ColumnNorm, Diagonal, I, UniformScaling, cross, dot, norm, normalize, qr, qr!, rank
using StaticArrays: StaticArrays, MVector, SMatrix, SVector, StaticArray, StaticVector
using Statistics: Statistics, cov, mean, median, quantile, std
using Random: Random, AbstractRNG
using StableRNGs: StableRNG
using Printf: Printf, @printf, @sprintf
using ForwardDiff: ForwardDiff
using Static: Static, static
import JSON                                                    # JSON.lower / JSON.lift of RC's types on VisualGeometryToolkitIO's style (records.jl)
import PNGFiles
using ImageCore: ImageCore, Gray, N0f8
import KernelAbstractions
import SparseArrays                                           # the row weighting of a sparse design matrix (_whiten, weighting.jl)
import StatsBase                                              # the photosite draw: tiedrank (the edge strength's ECDF rank)
using Adapt: Adapt, adapt
import LinearSolve
import NLLSsolver
using CoordinateTransformations: ComposedTransformation, LinearMap, Translation   # (Translation: a region moved by Core's SimilarityMap action); Core's composed lens (`Tilt ∘ Additive`: OpenCV's 14 coefficients)
using PrecompileTools: @setup_workload, @compile_workload      # src/precompile.jl
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import LibRaw_jll                  # Features' RAW decoder (its LibRaw extension): a RAW photograph (`load_photo`, by its file) is read through it in bin/refine.jl's environment
# RC's methods of the stack's generic functions where RC's function is the same concept (owner ruling 2026-11-11: no new binding):
# a camera's lens (the state's), the cost and residuals of a problem's state,
# an estimate refined in place (the calibration state), an options struct's TOML table, a lens's identity (the options' stated one)
import VisualGeometryToolkitCore: cost, residuals
using VisualGeometryToolkitCore: chart                          # a board point's page chart (VisualRadiometryToolkitCore's method on its textured target)
import VisualGeometryToolkitEstimation: config_dict
import VisualGeometryToolkitFeatures: lens_identity
using VisualGeometryToolkitCore: LogicalIntrinsics, CameraCalibrationMat,
                                 CameraModel, EuclideanMap, Vec3, exp_so3
using VisualGeometryToolkitCore: backproject, project
using VisualGeometryToolkitCore: grid_nodes                      # the focus and warping node grids (Core's thin-plate spline)
using VisualGeometryToolkitCore: angle_between                   # the angle of two rays (Core's atan(‖u × v‖, u · v))
using VisualGeometryToolkitCore: hammersley                      # the radiance sampler's low-discrepancy points
# VisualRadiometryToolkitCore: the generic radiometric types this package's calibration method is written on (moved there from here and
# from VisualGeometryToolkitCore; every name listed)
import VisualRadiometryToolkitCore as VrtkCore
import VisualRadiometryToolkitIO                  # the vignetting and environment-light bodies of RC's records: its methods of JSON.lower / JSON.lift on IO's style (records.jl)
using VisualRadiometryToolkitCore: EnvironmentLight, SphericalHarmonicMap, coefficients, with_coefficients, max_degree, sh_count   # the light: real SH on a sphere
using VisualRadiometryToolkitCore: AbstractVignetting, NoVignetting, TensorProductVignetting, RadialVignetting, VIGNETTING_TAGS,
                                   coefficient_jacobian, control_points, with_control_points, n_control_points   # the vignetting fields (their centre is `VgtkCore.center`)
using VisualRadiometryToolkitCore: RadiometricCameraModel, with_vignetting, chief_ray, center_of_projection, aperture_direction   # the radiometric camera, its geometric camera's chief ray, the back-projection from the aperture on a camera's parts
using VisualRadiometryToolkitCore: Focus, FocusSpline                                           # the focus surface: the sensor-grid thin plate spline
using VisualRadiometryToolkitCore: sensor_position, luma_image, SensorQuadrature, ApertureQuadrature, sensor_samples, sensor_sample,
                                   aperture_sample, aperture_weight, aperture_border_sample, n_disc, n_border              # the sensor domain and the quadrature rules
using VisualRadiometryToolkitCore: Material, nonlinear_parameters, with_nonlinear, with_diffuse_weight, texture_weight, roughness, refractive_index,
                                   sample_visible_normal, reflection_pdf, reflect, cook_torrance_terms, cook_torrance   # the material: the Cook–Torrance BSDF
# the board: the textured target (VisualGeometryToolkitCore's posed planar target with its texture, printed edge and own state), its
# chart and albedo, the printed-edge model, the warped page and the in-plane print error (its centre is VisualGeometryToolkitCore's `centroid`)
using VisualRadiometryToolkitCore: TexturedTarget, TargetState, with_state, with_h, with_edge, with_inplane, height_on, inplane_on,
                                   EDGE_NC, HEIGHT_NC, HEIGHT_NODES, INPLANE_NC, with_sdf, with_facing, board_normal, world_point, corners,
                                   intersect_board, board_plane_hit, inside, printed_albedo, warping_nodes, warping_aspect,
                                   board_height, BoardRays, inplane_calibration, design_chart, page_points
# its camera-frame maps and part accessors, which this package's records answer by forwarding (VisualRadiometryToolkitCore's
# CONTRACT_GENERICS, and VisualGeometryToolkitCore's `backproject`, the radiometric back-projection from the aperture): the kernels'
# `View` (camera/view.jl) and the calibration state (calibration/state.jl). The paper's symbols (𝔣, t̂, π̂⁻¹, …) are its notation in
# prose; the code calls the descriptive names (VisualRadiometryToolkitCore 0.2.0 deleted the aliases).
import VisualRadiometryToolkitCore: chief_direction, inverse_focus_distance, focus_point, world_ray, geometric, focus, aperture_radius

# camera: the geometric back-projection models and the kernels' view of the radiometric camera
include("camera/node_support.jl")            # the data support of a spline's nodes: one rule (vignetting, focus)
include("camera/vignetting.jl")
include("camera/pose.jl")
include("camera/models/geometric.jl")
include("camera/view.jl")                           # the kernels' record of a photograph (the camera at a pose); the vignetting's flat start
include("camera/models/pixel_camera.jl")             # the parametric chief-ray models: Core's pixel cameras (BoundedCamera of a CameraModel)
include("camera/models/grid_camera.jl")             # Core's generic models as chief-ray models (needs the parametric lenses)
include("io/camera_models.jl")                         # lens / grid JSON, Schöps import (all model types defined above)
# target: the calibration target, its warping and its SVBSDF
include("target/board.jl")                             # the board: VisualRadiometryToolkitCore's TexturedTarget; the calibration's share (the light sphere, the warping's free parameters, the distance map's reader)
include("target/target_radiometry.jl")               # the target's radiometry as one record (the material and the print's edge)
include("image_formation/radiance.jl")
# image formation, the calibration (the state, the closed form, the LM over all blocks on the device), the loaders
include("image_formation/flux.jl")
include("camera/observation_model.jl")          # the observation model: a calibrated camera's sensor record, photograph kind and channel
include("calibration/weighting.jl")                 # the rows' weights: whitened residuals under the sensor noise (D14), or as measured
include("calibration/state.jl")
include("calibration/precision.jl")
include("calibration/closed_form.jl")
include("calibration/lm/per_view.jl")
include("calibration/device/workers.jl")
include("calibration/device/flux_kernels.jl")
include("calibration/device/jacobian_kernels.jl")
include("calibration/lm/per_tile.jl")
include("calibration/smoothness.jl")                     # one smoothness vocabulary for the spline components ([<component>.smoothness])
include("calibration/lm/per_tile_smoothness.jl")      # the grid's present nodes and smoothness (the grid A/B's result)
include("io/photographs.jl")
include("start/edge_detector.jl")            # the photosite sample's edge strength (Gaussian derivative) and its ECDF rank
include("options/device.jl")       # the device tags and rc_backend
include("options/tables.jl")
include("options/toml.jl")         # RunOptions and CalibrationSettings, read from TOML (pipeline_options / config_dict)
include("options/retired.jl")
include("start/calibration_start.jl")      # StartGeometry (the diagnostics and the start read it), start_geometry, the start record, load_calibration_start
include("start/masks.jl")
include("start/image_region.jl")
include("start/focus_start.jl")
include("start/start_poses.jl")
include("start/radiometric_start.jl")
include("calibration/problem_spec.jl")
include("diagnostics/monitor.jl")
include("diagnostics/selfcheck.jl")
include("diagnostics/render.jl")
include("calibration/resume.jl")
include("start/frontends/calibration_session.jl")
include("start/frontends/schoeps_run.jl")
include("start/prepare.jl")
include("calibration/calibrate.jl")
include("io/run_outputs.jl")
include("io/records.jl")       # RC's kinds: rc/radiometric_camera/v3.1, rc/radiometric_target/v2.0, rc/calibration_state/v3.1, …
include("start/start_record.jl")              # the refinement's start as its own step: the start record
include("precompile.jl")                         # the precompile workload: the hot paths' native code, cached in the package image

include("api.jl")                                # the API surface: export (the headline) and public

end # module
