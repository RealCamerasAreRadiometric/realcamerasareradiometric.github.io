# ---------------------------------------------------------------------------
# The geometric camera (paper §Radiometric Camera Model): any geometric back-projection model π⁻¹ supplies the chief ray of the
# radiometric back-projection backproject(𝐩, 𝐚) = (focus_point(𝐩) − 𝐚)/‖focus_point(𝐩) − 𝐚‖ with the focus point focus_point(𝐩) = 𝔭 + t(𝐩) π⁻¹(𝐩)/‖π⁻¹(𝐩)‖
# (VisualRadiometryToolkitCore's `π̂⁻¹`). A chief-ray model is a VisualGeometryToolkitCore camera of the run's image, on its pixels:
#
#   chief_ray(c, 𝐩)      VisualRadiometryToolkitCore's: the image formation's chief-ray direction (camera frame; need not be normalised) of
#                        the sensor position 𝐩 ∈ ℐ = [0, W] × [0, H] (COLMAP pixels) — a pixel camera's `backproject`, a SplineLens
#                        camera's raw backward spline value; generic in the element type and usable in GPU kernels
#   in_domain(c, 𝐩)      whether the camera images 𝐩 (mask construction)
#
# The two kinds: the parametric models — Core's pixel camera, a `BoundedCamera` of a `CameraModel` (K in pixels, any Core lens;
# models/pixel_camera.jl), whose parameters, names, bounds, rebuild and element type are Core's parameter protocol and whose LM block
# is Core's `ParameterManifold` (`lens_manifold`, per_view.jl) — and the grid lenses (models/grid_camera.jl), Core's generic models
# refined per tile. Every camera is central: its chief rays start at 𝔭 = 0.
# ---------------------------------------------------------------------------

"""Whether the camera images the sensor position 𝐩."""
function in_domain end
