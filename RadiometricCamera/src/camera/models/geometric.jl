# ---------------------------------------------------------------------------
# The geometric camera (paper §Radiometric Camera Model): any geometric back-projection model π⁻¹ supplies the chief ray of the
# radiometric back-projection π̂⁻¹(𝐩, 𝐚) = (𝔣(𝐩) − 𝐚)/‖𝔣(𝐩) − 𝐚‖ with the focus point 𝔣(𝐩) = 𝔭 + t(𝐩) π⁻¹(𝐩)/‖π⁻¹(𝐩)‖
# (radiometric_camera.jl). A chief-ray model is a VisualGeometryToolkitCore camera of the run's image, on its pixels:
#
#   _chief_ray(c, 𝐩)     the image formation's chief-ray direction (camera frame: x right, y down, z forward; need not be normalised) of the sensor
#                        position 𝐩 ∈ ℐ = [0, W] × [0, H] (COLMAP pixels); generic in the element type (Float32, ForwardDiff.Dual) and
#                        usable in GPU kernels (isbits, fixed iteration counts, no throws)
#   in_domain(c, 𝐩)      whether the camera images 𝐩 (mask construction)
#
# The two kinds: the parametric models — Core's pixel camera, a `BoundedCamera` of a `CameraModel` (K in pixels, any Core lens;
# models/pixel_camera.jl), whose parameters, names, bounds, rebuild and element type are Core's parameter protocol and whose LM block
# is Core's `ParameterManifold` (`lens_manifold`, per_view.jl) — and the grid lenses (models/grid_camera.jl), Core's generic models
# refined per tile. Every camera is central: its chief rays start at 𝔭 = 0.
# ---------------------------------------------------------------------------

"""
    _chief_ray(camera, 𝐩) -> SVector{3}

The image formation's chief ray of a sensor position (camera frame, not necessarily unit): a parametric pixel camera's Core
`backproject`; a SplineLens camera's backward spline VALUE at K⁻¹𝐩, Core's lattice evaluation, not normalized — linear in the
lattice's controls, which the per-tile Jacobian's fan-out onto them relies on (`_dir_seeded`, the device Jacobian). Only the image
formation reads it (`chief_dir`, the flux's direction seed, the device Jacobian); every other back-projection is Core's `backproject`
(a unit ray, NaN where the lens does not image 𝐩). Core follow-up: a public name for the spline's raw backward value — then this goes.
"""
function _chief_ray end
"""Whether the camera images the sensor position 𝐩."""
function in_domain end
