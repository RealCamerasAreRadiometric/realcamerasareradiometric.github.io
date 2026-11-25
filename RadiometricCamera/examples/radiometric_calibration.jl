# # Radiometric camera calibration — a walkthrough
#
# This manual takes an existing **central** camera model, wraps the paper's radiometric camera around it, renders the
# calibration board through it, and refines the lens radiometrically against real photographs. Every symbol is the paper's:
#
# * `π⁻¹(𝐩)` chief-ray back-projection of the sensor point 𝐩 (COLMAP pixels, ℐ = [0, W] × [0, H]), `Θ_d` its parameters;
# * `𝔣(𝐩) = 𝔭 + π⁻¹(𝐩)/t̂(𝐩)` the focus surface, `t̂` the inverse focus depth (thin plate spline with kernel `U` and affine part
#   `a_1, a_x, a_y`, nodal values `Θ_f`); `π̂⁻¹(𝐩, 𝐚)` the radiometric back-projection through the aperture point `𝐚 = r â`;
# * `V(𝐩; Θ_v)` vignetting (B-spline surface, V = 1 at the sensor centre), `L_SH` / `L_SVSH` the ambient light on the sphere `(𝔟, r_SH)`
#   (`Θ_l`), `cook_torrance` the Cook–Torrance SVBSDF with `𝛂, 𝛈` blended by the texture value and the diffuse weight `μ` (`Θ_m`);
# * `Φ_c(𝐮; Θ̄)` the radiant flux of pixel 𝐮: Gauss–Legendre over the pixel footprint × polar rule over the fixed unit disc
#   (`𝐚 = r â`, weights `r² w̃`), `L_c` by Monte Carlo over VNDF samples with common random numbers.
#
# The script runs as `VGTK_DATA_ROOT=<data root> julia --project=. examples/radiometric_calibration.jl` (a few minutes on a GPU)
# and is the source of the notebook next to it (`build_notebook.py`). It reads the paper project's published data (the tier
# rc_calibration, experiments/data/README.md; `VGTK_DATA_ROOT` names the data root): the synthetic blob-board set of the
# geometric-accuracy experiment (`geometric_accuracy/detector_bias/data/blobs_fisheye`: a BlobBoards session's VisualGeometryToolkitIO
# records on rendered photographs) and the Schöps calibration of the FX3 fisheye (`geometric_accuracy/night_stars/data/fx3-fisheye/schoeps`).

using RadiometricCamera, StaticArrays, LinearAlgebra, Statistics, Printf, Random
using RadiometricCamera: grid_angular_error, grid_dims, grid_model, load_session, render_all, RenderSettings, run_lm_grid!,
                         save_grid, schoeps_gridlens, update_radiometry!, sphere_radius
using VisualRadiometryToolkitCore: aperture_radius, focus, Focus, FocusSpline, geometric, Material, t̂, 𝔟   # the generic radiometric layer
using VisualGeometryToolkitCore: cost                                        # a state's ½ Σ (Φ − ϕ)² and its relative RMS
import CUDA, KernelAbstractions
import VisualRadiometryToolkitCore as VrtkCore
import VisualGeometryToolkitCore as VgtkCore
EXP = joinpath(ENV["VGTK_DATA_ROOT"], "geometric_accuracy")                       # the data root (experiments/data/README.md)
OUT = isempty(ARGS) ? joinpath(@__DIR__, "output") : ARGS[1]; mkpath(OUT)    # the output directory: the first argument, or examples/output
backend = CUDA.functional() ? CUDA.CUDABackend(; always_inline = true) : KernelAbstractions.CPU();   # inlined: the kernels' fast build

# ## 1. A geometric central model
#
# The radiometric camera is built *around* a chief-ray model: anything that maps a sensor point to a ray direction. Three ways in.
#
# **(a) A calibration session's records** (BlobBoards' own feature-based calibration, stored as VisualGeometryToolkitIO records by
# CaliBlobs: the camera — here a division camera `x_u = x_d / (1 + k₁ρ² + k₂ρ⁴)` — the board's target with its printed texture, the
# frames' poses). `load_session` builds the board, the initial poses and loads the photographs (16-bit linear PNGs here, so
# `linearize = false`); `downsample = 8` keeps this walkthrough at 648 × 432.

SESSION = joinpath(EXP, "detector_bias", "data", "blobs_fisheye", "session")                # camera.json, targets_1.json, trajectory_1.json
IMAGES = joinpath(EXP, "detector_bias", "data", "blobs_fisheye", "gt", "images_linear")   # the rendered photographs (16-bit linear PNG)
ds = load_session(SESSION, IMAGES; lens = :camera, focus_init = :board_center, downsample = 8, texture_downsample = 1,
                  sensor_width = 22.3VgtkCore.mm, r₀ = 17e-3 / (2 * 8), linearize = false)   # EOS 100D: 22.3 mm sensor, 17 mm f/8
W₀, H₀ = ds.extent; W, H = Tuple(VgtkCore.extent(ds.lens₀))                # the record's image, the run's: the camera's (box-averaged by 8)
@printf("%d views, image %d × %d (used at %d × %d); division camera f = %.1f px (at %d × %d), k = %s\n", length(ds.poses), W₀, H₀, W, H,
        VgtkCore.model(ds.lens₀).intrinsics.K[1, 1], W, H, string(VgtkCore.parameter_vector(VgtkCore.lens(ds.lens₀))))

# **(b) A Schöps `central-generic` calibration** (intrinsics0.yaml: a B-spline surface of directions over the image). `schoeps_gridlens`
# resamples it EXACTLY into the grid model on his own lattice (VisualGeometryToolkitCore's SplineLens camera, extended over the sensor, never subdivided),
# its K the Schöps model's gauge (the pinhole at the pixel imaging the optical axis).

g_schoeps = schoeps_gridlens(joinpath(EXP, "night_stars", "data", "fx3-fisheye", "schoeps", "intrinsics0.yaml"), W₀, H₀)
@printf("Schöps → grid model %d × %d; central ray %s\n", grid_dims(g_schoeps)..., string(round.(normalize(VgtkCore.backproject(g_schoeps, SVector(VgtkCore.centroid(VgtkCore.Rect(0, 0, W₀, H₀))...))); digits = 4)))

# **(c) Any VisualGeometryToolkitCore lens** (Kannala–Brandt, Brown–Conrady, equidistant, pinhole…) as VisualGeometryToolkitCore's pixel camera: a
# `BoundedCamera` of `CameraModel(LogicalIntrinsics(K), lens)` with K in pixels of the image, whose parameters are Core's parameter
# protocol. Sampled into the grid model (`grid_model`) it is refined per tile — the route below.

kb = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.intrinsics(ds.lens₀), VgtkCore.KannalaBrandt(-0.01, 0.002, 0.0, 0.0)), VgtkCore.extent(ds.lens₀))
g_kb = grid_model(kb, 35, 25)
@printf("Kannala–Brandt as a pixel camera, sampled into a %d × %d grid: %.2e rad quasi-interpolation error\n", grid_dims(g_kb)..., grid_angular_error(g_kb, kb))

# This walkthrough refines the blob board's own camera: the division model of (a) sampled into a 35 × 25 grid (the production runs use
# 105 × 75). `grid_model(camera, G_x, G_y)` works for every camera above.

lens0 = grid_model(ds.lens₀, 35, 25);

# ## 2. The radiometric model around it
#
# * `Focus`: inverse focus depth `t̂` on a 5 × 5 TPS node grid over the sensor's pixels, here a plane at the taped focus distance 0.5 m —
#   `t̂ᵢ = cos αᵢ / 0.5 m` with αᵢ the angle between the ray π⁻¹(𝐩ᵢ) and the optical axis;
# * `TensorProductVignetting((5, 5), Rect(0, 0, W, H); degree = 2)` (VisualRadiometryToolkitCore's): B-spline `V` with 5 × 5 nodes
#   over the sensor, V = 1 at its centre pinned, flat at the start;
# * `RadiometricCameraModel(geometric, focus, r, vignetting)`: the radiometric camera, one object — the geometric camera, the focus surface, the
#   aperture radius r and the vignetting V; no image distance. A photograph is the camera at a pose: its extrinsics `Θ_e`, the
#   camera → world `EuclideanMap` (`ds.poses`);
# * `EnvironmentLight(SphericalHarmonicMap(radiance, l_max), 𝔟, r_SH)` (VisualRadiometryToolkitCore's): SH light of order l_max on the
#   sphere of radius `r_SH` around the board centre `𝔟` (`sphere_radius` = the sphere through the farthest camera);
# * `Material(; roughness_ink, roughness_paper, refractive_index_ink, refractive_index_paper, diffuse_weight)`: Θ_m — GGX roughness and refractive index of ink and paper (blended by the texture
#   value) and the diffuse weight μ; the board warping h and the printed-edge parameters are the board's `TargetState` (flat here);
# * `RenderSettings(; n_𝐩, n_ρ, n_φ, N)`: n_𝐩² sensor nodes per pixel, n_ρ × n_φ polar aperture nodes, N VNDF samples per node.

f_m = VgtkCore.model(ds.lens₀).intrinsics.K[1, 1] * (22.3e-3 / W); d_f = 0.5; d_s = f_m * d_f / (d_f - f_m)   # f_px × the pixel pitch; d_s derived (thin lens)
nodes = VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, W, H))
focus0 = Focus(FocusSpline(nodes), [normalize(VgtkCore.backproject(lens0, 𝐩))[3] / d_f for 𝐩 in nodes])
r0 = 17e-3 / (2 * 8)                                                          # f/8 → r = f / (2N) = 1.06 mm
vig0 = VrtkCore.TensorProductVignetting((5, 5), VgtkCore.Rect(0, 0, W, H); degree = 2)
camera0 = RadiometricCameraModel(lens0, focus0, r0, vig0)                    # the radiometric camera: geometric camera, focus, aperture, vignetting
b = 𝔟(ds.board); r_SH = sphere_radius(b, [p.t for p in ds.poses])
light0 = VrtkCore.EnvironmentLight(VrtkCore.SphericalHarmonicMap(1.0, 2), b, r_SH)
mat0 = Material(; roughness_ink = 0.15, roughness_paper = 0.4, refractive_index_ink = 1.6, refractive_index_paper = 1.33, diffuse_weight = 0.5)
st = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 8)
@printf("f %.2f mm, d_s %.3f mm, r %.2f mm, light sphere r_SH %.2f m around 𝔟 = (%.2f, %.2f, %.2f) m\n", 1e3f_m, 1e3d_s, 1e3r0, r_SH, b...)

# ## 3. The state, masks and the start prediction
#
# `CalibrationState` holds the board, every block of `Θ̄`, the photographs ϕ and the masks (pixels whose whole circle of confusion lands
# on the board, `compute_mask`). Four views keep the walkthrough short. The one thing the start cannot know is the absolute scale of
# the light (the photographs' units): the closed-form Θ_l solve below sets it. `render_all` renders Φ for every masked pixel; `cost`
# returns `(½ Σ (Φ − ϕ)², rel-RMS = √(Σ(Φ − ϕ)² / Σ ϕ²))`.

K = 4; sel = round.(Int, range(1, length(ds.poses); length = K))
s = CalibrationState(ds.board, st, camera0, ds.poses[sel], mat0, light0, ds.targets[sel]; δ = 0.01, backend)
update_radiometry!(s; light = true, vignetting = false, mu = false); c0, rms0 = cost(s)
@printf("%d views, %d masked pixels; start (flat V, closed-form lighting): rel-RMS %.4f\n", K, sum(length, s.px), rms0)
Φ0 = render_all(s);                                                          # H × W × 3 per view, zero outside the mask

# A first look at model vs photo (view 1, green channel): the start prediction is the geometric model's rendering of the board.

let img = ds.targets[sel[1]], m = s.masks[1]
    @printf("view %s: photo mean %.4f, model mean %.4f on the %d masked pixels\n", ds.names[sel[1]], mean(img[:, :, 2][m]), mean(Φ0[1][:, :, 2][m]), count(m))
end

# ## 4. Calibration
#
# The blocks that enter Φ linearly — `Θ_v`, `Θ_l` and μ — have closed-form least-squares solutions (`update_radiometry!`: one linear
# sweep from the per-pixel light / vignetting responses). Then the joint Levenberg–Marquardt over every block: poses `Θ_e`, the grid
# directions (per-tile cost blocks, `run_lm_grid!`), focus `Θ_f`, aperture r, material `Θ_m`, vignetting `Θ_v`, lighting `Θ_l`; the
# vignetting gauge V(𝐜) = 1 is exact (the centre's free direction eliminated, `gauge_basis`), no geometric prior.

update_radiometry!(s; light = true, vignetting = true, mu = true); c1, rms1 = cost(s)
@printf("linear sweep (Θ_l, Θ_v, μ) → rel-RMS %.4f, μ %.3f, V at 0.4 W off the centre %.3f\n", rms1, s.mat.diffuse_weight, s.camera.vignetting(VgtkCore.center(s.camera.vignetting) + SVector(0.4W, 0.0)))
trace = NamedTuple[]
monitor(it, cst, camera, poses, mat, surface, light) = @printf("  it %2d  cost %.4e  r %.3f mm  μ %.3f  α %.3f / %.3f\n", it, cst,
                                                               1e3 * camera.aperture_radius, mat.diffuse_weight, mat.roughness_paper, mat.roughness_ink)
res = run_lm_grid!(s; maxiters = 3, trace, iter_callback = monitor, verbose = false)
c2, rms2 = cost(s)
@printf("joint LM: %d iterations, rel-RMS %.4f → %.4f; stopped: %s\n", length(trace), rms1, rms2, res.rule)

# The loss curve and the refined lens. `save_grid` stores the grid as VisualGeometryToolkitIO's camera record for the
# evaluation scripts (`grid:<file>` model spec).

ϕ² = sum(sum(abs2, p) for p in s.ϕ)
println("rel-RMS per LM iteration: ", join([@sprintf("%.4f", sqrt(2t.cost / ϕ²)) for t in trace], " → "))
save_grid(joinpath(OUT, "grid_refined.json"), geometric(s); provenance = (; model = "example_refined"))
save_grid(joinpath(OUT, "grid_start.json"), lens0; provenance = (; model = "example_start"))
drift = [1e3 * acos(clamp(dot(normalize(geometric(s).grid[i]), normalize(lens0.grid[i])), -1, 1)) for i in eachindex(lens0.grid)]
@printf("grid moved: median %.3f mrad, max %.3f mrad; aperture r %.3f mm; focus 1/t̂ mean %.2f m\n", median(drift), maximum(drift), 1e3 * aperture_radius(s), 1 / mean(focus(s).t̂_i))

# ## 5. Validation
#
# The photo residual alone cannot tell a better lens from a lens that absorbs print or lighting defects. Two independent checks:
#
# 1. **The chrome board** (the paper project's chrome-board evaluation, `experiments/scripts/thorlabs_distortion.jl`: the Thorlabs
#    R3L3S7N1 dot grid backlit and photographed under a rotation about one point, the dots' residuals in px under one rotation centre for
#    the set) — run on the grids written above as models of a `experiments/configs/experiments/thorlabs_distortion_fx3_*.toml` evaluation.
# 2. **Difference images**: model − photo per view, before and after. Here as numbers; `bin/refine.jl` with `[io] render_state` dumps the
#    photo / model planes of a state for the difference pages.

Φ1 = render_all(s)
for k in 1:K
    m = s.masks[k]; ϕk = ds.targets[sel[k]]
    d0 = sqrt(mean(abs2, (Φ0[k] .- ϕk)[m, :][:])) / sqrt(mean(abs2, ϕk[m, :][:])); d1 = sqrt(mean(abs2, (Φ1[k] .- ϕk)[m, :][:])) / sqrt(mean(abs2, ϕk[m, :][:]))
    @printf("  %-14s rel-RMS start %.4f → refined %.4f\n", ds.names[sel[k]], d0, d1)
end
println("written to ", OUT)
