# RadiometricCamera: design and performance report

Branch `perf/optimize-and-compose` (from `main` at 8b3ba04), 2026-11-20. Not a published docs page. Measurements: the
benchmark scripts in `RadiometricCamera/benchmark/` and `experiments/tools/texture_benchmark.jl`, run as Slurm jobs on
poseidon (one whole NVIDIA B200 per GPU job; CPU jobs 16 threads on Emerald Rapids), Julia 1.13.3. Wall times are
`time_ns` around calls that return their values to the host (device kernels awaited); no vendor profiler was used
except where a line says so. Records: `/cluster/projects/vgtk/logs/rc-perf/results/*.json`.

Contents: 1 architecture map · 2 building a camera · 3 findings · 4 the texture experiment · 5 performance · 6 decisions.

---

## 1. Architecture map

### 1.1 The objects

| Object | What it is | Owner | isbits | Element types |
|---|---|---|---|---|
| geometric camera | `BoundedCamera(CameraModel(K; lens), Size2(W, H))`: π and π⁻¹ on the image's pixels; a parametric lens, or a `SplineLens` grid model | Core | yes for a parametric lens; a grid carries an array (adapted) | Float64 host, Float32 device, duals in a Jacobian |
| `Focus{T, N}` | the inverse focus distance t̂(𝐩): `N` thin-plate nodes, their values and the cached coefficients | RC (Core's spline) | yes | `T` |
| `FocusSpline{N}` | the host solver of a `Focus` (Core's `ThinPlateSplineSystem`): values → coefficients | RC | no (host only) | Float64 |
| aperture radius | a number, metres | — | yes | `T` |
| vignetting | Core's `AbstractVignetting`: `TensorProductVignetting`, `RadialVignetting`, `LogVignetting` | Core | yes | `T` |
| `RadiometricCameraModel{L, F, T, V}` | **the camera**: `(lens, focus, aperture_radius, vignetting)`, what every photograph shares | RC | yes (parametric lens) | host Float64 |
| pose | Core's `EuclideanMap`, camera → world | Core | yes | `T` |
| `View{T, N, L, P}` | internal: the camera at a pose flattened to ONE element type, the vignetting apart (the kernels differentiate V analytically) | RC | yes | Float64 / Float32 / dual |
| `PrintedTarget` | the board: Core's posed quad, its texture (or edge map), the warping's flat surface | RC over Core | no (arrays; adapted) | `T` |
| `BoardSurface{T}` | the surface parameters: nodal heights h, printed-edge parameters, in-plane print error | RC | yes | `T` |
| `Material{T}` | α, η per ink/no-ink, μ | RC | yes | `T` |
| `EnvironmentLight` | spherical harmonics on a sphere | Core | yes | `T` |
| `Scene` | board + image size + surface | RC | adapted | `T` |
| `RenderSettings` | sensor rule, aperture rule, radiance sampler | RC | yes | `T` |
| `CalibrationState` | mutable: scene, settings, ONE camera, the poses, material, light, masks, measured vectors ϕ, weights, backend, device cache, run record | RC | no | Float64 |
| `RadianceJac{T}` | a ray's record `[L | ∂L/∂ℓ | 𝐱 | 𝐧]` (33 numbers) | RC | yes | kernel float |
| `PixelParts` | a pixel's light responses per sensor sample (the linear blocks' rows) | RC | yes | kernel float |

The kernels are **functors of a pixel id** (`_FluxAt`, `_RecordAt`, `_JacAt`, `_HeightAt`, `_PartsAt`, `_PixelPartsAt`,
`_MaskAt`) run by Core's `pointwise!` on the state's backend value, plus row functions run by Core's row kernel
(`_jacobian_row!`, `_design_row!`, `_radiometry_cost_row!`, `_radiometry_normal_row!`). One implementation on every
backend: the CPU backend runs the same functors in Float64, a device in Float32 with the pose's chunks, the heights and
the closed form in Float64 (`HIGH_FLOAT`). RC's source names no vendor outside `ext/RadiometricCameraCUDAExt.jl` and
`rc_backend`.

### 1.2 The data flow

```text
camera, pose ──View──► per pixel: sensor samples × aperture samples (96 rays at the paper's rules)
   ray ─► hit on the board (plane, or Core's height-field Newton) ─► chart 𝐛 ─► material(𝐛) ─► MC radiance L_c
   Φ(𝐮) = 1/π Σ_s w_s V(𝐩_s) Σ_a w_a L                                   _Φ_pixel: THE forward function, every mode
```

One linearization of a block (a view; in the grid route a view's batch, then its tiles):

| Kernel | Float (device) | What | Reads |
|---|---|---|---|
| flux `_FluxAt` | F32 | Φ at the trial parameters: the residual, bitwise `computeresidual`'s | — |
| record `_RecordAt` | F64, rounded to F32 | per ray the radiance, its 8 local partials (ρ, α, η, ω, 𝐱), the hit 𝐱 and the surface normal 𝐧 | — |
| chunks `_JacAt` | F64 for the pose's two, F32 for the rest | `jacobian_chunk = 4` dual slots each: pose 6, lens ND, t̂ per sensor sample, material 8, r, in-plane | the record: the radiance linearised, the hit's tangent plane |
| heights `_HeightAt` | F64 | ∂Φ/∂h (25 nodal heights) through the ray parameter | the record |
| parts `_PartsAt` | F32 | the analytic columns of V, the lighting and μ | the record's hits |
| rows `_jacobian_row!` | F64 | assembles the block's J (columns × rows) and r on the device: scatter, focus chain, patch fan-out, parts through the gauge basis, heights through their basis, whitening | all of the above |
| products | F64 | `J r`, `J Jᵀ` on the backend; `ncol` and `ncol²` values to the host; the column map T (charts, ties, held nodes) applied there | J, r |

- **Per-view route** (`run_lm!`, `build_problem`): one `PerViewResidual` per photograph, 7 variable blocks. The views
  of an iteration run ahead of NLLSsolver's requests on pooled device workers (`_windowed`; width from the kernel's
  occupancy and the free memory).
- **Per-tile route** (`run_lm_grid!`, `build_problem_grid`): the lens is a spline grid, its parameters 5 × 5 patches of
  nodes; one `PerTileResidual` per image tile (10 blocks: pose, 4 patches, focus, r, material, V, light). A view's
  kernels run once for the whole view (`_view_batch!`, memoized by the variables' identity); each tile then runs the row
  kernel on its pixels and its own products.
- **LM**: Core's `lmsolve_chart!` on NLLSsolver (Levenberg λ·I). NLLSsolver linearizes at the first trial of each
  iteration (cost, gradient and Hessian in one call), so an accepted iteration is ONE linearization per block and
  nothing else; a rejected trial is a flux kernel.
- **Closed form** (`solve_linear_blocks!`): `all_parts` renders every pixel's parts once (Float64 on every backend);
  Θ_l and Θ_v by pivoted QR of design rows assembled on the device and downloaded; μ and the joint solve by a
  reduction kernel that leaves only the `N × N` system on the host (`radiometry_normal_equations`).

### 1.3 Source layout

`sensor/` (notation, quadrature) · `camera/` (vignetting, focus, pose, geometric models, the radiometric camera) ·
`target/` · `image_formation/` (radiance, flux) · `calibration/` (state, closed form, `lm/` per view and per tile,
`device/` workers and kernels) · `start/` · `options/` · `io/` · `diagnostics/` · `experiments/` (focus-ab).
The layout follows the model's nouns and is sound; section 3 lists the two places where a file does not sit where
its content belongs. I do not recommend a reorganization beyond those.

---

## 2. Building a camera

### 2.1 Before

```julia
lens   = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), Size2(W, H))
camera = RadiometricCameraModel(lens, Focus(3, 1.5, W, H), 0.04,
                                TensorProductVignetting((4, 4), Rect(0, 0, W, H); degree = 2))
```

The image size is stated three times (the bound, the focus grid, the vignetting's rectangle), the common case needs
two constructors of other types, and only the vignetting had a replacer (`with_vignetting`).

### 2.2 Proposed (implemented on the branch, commit e37cde5; additive — the positional form is unchanged)

```julia
# 1. the geometric camera: Core's, any lens
lens = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), Size2(W, H))

# 2. the radiometric camera composed from it: what a geometric camera lacks, by name
camera = RadiometricCameraModel(lens; focus = 1.5, aperture_radius = 0.04)        # V ≡ 1 by default
camera = RadiometricCameraModel(lens; focus = Focus(lens, 1.5; n = 5), aperture_radius = 0.04,
                                vignetting = RadialVignetting(5, 𝐜, r_max))

# 3. one part replaced, the others kept (immutable updates; each returns a new isbits camera)
camera = with_focus(camera, Focus(lens, 2.0))
camera = with_aperture_radius(camera, 0.02)
camera = with_vignetting(camera, v)
camera = with_lens(camera, other_lens)

# 4. the parts read
lens(camera), focus(camera), aperture_radius(camera), camera.vignetting

# 5. a photograph is the camera at a pose
img = render_view(k, camera, pose, material, light, scene, settings, mask; backend)
```

| Kind | Names | Why this kind |
|---|---|---|
| type | `RadiometricCameraModel`, `Focus`, Core's cameras and vignettings | data with a layout the kernels read; all isbits |
| function (constructor) | `RadiometricCameraModel(lens; focus, aperture_radius, vignetting)`, `Focus(lens, distance; n)` | composition; the image size read from the geometric camera's `extent` |
| function (replacer) | `with_lens`, `with_focus`, `with_aperture_radius`, `with_vignetting` | immutable update, one part each |
| function (accessor) | `lens` (Core's generic), `focus`, `aperture_radius` | the same names read a camera and a state |
| trait-like dispatch (internal) | `_chief_ray(lens, 𝐩)`, `in_domain(lens, 𝐩)`, `_model_eltype`, `is_grid_lens` | the ONLY things RC asks of a geometric camera beyond Core's `backproject`/`project`; a new lens family needs nothing else |

- `focus`: a `Focus`, or a distance (metres): the surface focused there everywhere on a 3 × 3 grid.
- `aperture_radius` has no default: r → 0 is not a pinhole limit of the code (Φ divides by πr²), and the radius is a
  physical statement.
- `vignetting` defaults to the flat 4 × 4 quadratic B-spline surface on the image, the surface a calibration then
  estimates.

**Type stability and devices.** The camera, the pose and the `View` are isbits for a parametric lens
(`isbits(camera) == isbits(View(pose, camera)) == true`, tested), so they travel into a kernel as arguments. Element
types are converted by Core's `convert_eltype` (`View`, `Focus`, `Material`, `Scene`, `RenderSettings` have methods);
arrays (a grid lens, the board's texture) are moved by `Adapt`. `View(pose, camera)`, `Φ_pixel`, `pixel_parts`,
`world_ray`, `render_view` and the flux functor infer concretely (the benchmark's `inferred` section). The one
value-dependent type is `Focus(n, …)`: its node count `N = n²` is a type parameter computed from a value, so a
constructor call with a runtime `n` is not inferable — by design (static arrays in the kernels); harmless at
construction time.

### 2.3 What I did not change, and recommend

1. **The field `lens` holds a camera.** `RadiometricCameraModel.lens` is a `BoundedCamera`, while Core's `lens(camera)`
   returns the `Lens` of a camera; RC's `lens(state)` (and now `lens(camera)`) return the geometric camera. Two
   meanings of one word across the stack. Recommendation: rename the field and accessor to `geometric` (the docs
   already say "the geometric camera"), refusing the old spelling by name as the package does elsewhere. Public API
   break; decided: section 6, decision 1.
2. **No `convert_eltype`/`Adapt` method for `RadiometricCameraModel`.** Every kernel takes a `View`, so none is
   needed today; adding both (four lines) would let user kernels take the camera itself. Low value until someone
   needs it.
3. **`View` vs Core's posed camera.** `View` is "camera at a pose, one element type, V apart". It could be a
   `PosedCamera{RadiometricCameraModel}` if Core's posed camera carried an element-type conversion that flattens;
   today `View` is the smaller thing. Keep.
4. **A constant vignetting type.** "V ≡ 1" is expressed as a flat spline. A `NoVignetting <: AbstractVignetting` in
   Core (V = 1, no parameters, no columns) would make "no vignetting" a type and remove 16 dead columns from such
   problems. Core proposal (section 3.3).

---

## 3. Findings

Ranked by measured gain. "Done" items are commits on the branch; the rest are recommendations.

### 3.1 Done on the branch

| # | Finding | Evidence | Change | Gain | Risk / numbers |
|---|---|---|---|---|---|
| A | Every Jacobian chunk re-solved each ray's hit with the board. On a warped board that is Core's height-field Newton (4+ evaluations of a 25-node thin plate spline, two `log` per node per evaluation), which costs 3× the whole radiance estimator: flux per pixel 47 µs flat, 189 µs warped (local CPU, one thread). A device linearization found up to 12 roots per ray. | CPU profile of a chunk kernel: 820 of 857 samples in `intersection`/`_height_root`; per-pixel timings | the record keeps the hit and the surface normal; chunks meet the hit's tangent plane with their dual ray (implicit-function partials, no root); the parts take the recorded hit (09551bf) | chunk kernel per pixel 303 → 53 µs; B200 warped view: chunks 51.5 → 9.5 ms | residual bitwise the same; Jacobian columns ≤ 2.2e-14 of their norm (tangent-plane partials equal Core's implicit ones to 2e-16 per ray, tested) |
| B | A FLAT board ran the same Newton in the chunks: `is_warped` was true for any dual-typed height vector, and chunks carry the heights as dual constants | same profile (flat scene) | `is_warped` reads the numbers; plane hit returned in the height field's types (1c4dc5f) | flat chunk per pixel 304 → 59 µs; B200 flat view: chunks 162.6 → 26.4 ms | columns ≤ 1.1e-14 |
| C | A device linearization ran the radiance estimator twice (Float32 and Float64 records) | B200: F32 record 29 ms + F64 record 67 ms of a 127 ms view | the F32 record is the F64 record rounded (30ad4c2) | −29 ms (flat 76,800 px), −12.8 ms (warped 16,384 px) | **device numbers change**: the F32 chunks' columns now use the F64 estimator's partials rounded to F32 — towards the CPU's values (the F32 estimator is up to 0.4 % off at a printed edge, `HIGH_FLOAT`'s own measurement). CPU unchanged. Decision 2. |
| D | `J r` on CUDA: CUBLAS's gemv of a wide matrix | gate probe (B200 slice): gemv 73 × 230,400 F64 14.9 ms; gemm of the same 0.33 ms | one-column gemm in the CUDA extension (d63e422) | products 17.5 → ~2.6 ms per full view | device gradient 3.5e-14 relative |
| E | The column map T was uploaded and applied by three dense device products per block | gate probe: 0.84 ms of a tile's 1.43 ms; on the CPU two dense ncol³ products per tile | `J r`, `J Jᵀ` on the backend, T applied on the host from its stored entries (a6fdd8b, bfbce88) | a view's products 17.5 → 0.4 ms together with D. For a grid's tiles on a B200 the effect is small and mixed: one grid LM iteration (65 × 50 lattice) 3.13 → 2.88 s, a single view's tile sweep 0.281 → 0.31–0.33 s (same job, A/B) — the tile path stays latency-bound (recommendation 1) | colour states bitwise the same (checked on a view); single-channel ties may round |
| F | KernelAbstractions' CPU workgroups of 1024 left small views to few threads | 1,728-px view on 32 threads: flux 49 ms at the default, 5.4 ms at 16 px per group, same bits | per-pixel kernels and the row kernel in groups of 16 on the CPU backend (3a20b71) | small views up to 9× per kernel; no change for ≥ 16 k px per thread-count | none (bitwise) |
| G | The nodal heights reserved 25 dual slots no chunk seeds | layout: P = 50 with free heights, the CPU's one chunk 50 wide and split in two | heights take no slot (d1ab18f) | CPU warped view with free heights 212 → 180 ms (local) | ≤ 3.6e-15 absolute |
| H | `render_view` ran on the CPU only | B200: CPU render 0.46 s, device flux kernel 7 ms for the same view | `render_view(…; backend)` (dd1e3d1) | 65× for a device render | additive API |
| I | The texture page's operators were assembled on the host | section 4 | built on the device (36c9de9) | section 4 | operators array-for-array the host's |

### 3.2 Recommended next, in RadiometricCamera

| # | Finding | Evidence | Recommendation | Expected gain | Risk | API |
|---|---|---|---|---|---|---|
| 1 | **Grid route: tile assembly is latency-bound.** A view's 108 tiles (65 × 50 lattice, ~150 px each) each launch a row kernel, upload their window lens and bases, and form their products: 1.43 ms per tile, 154 ms per view, against ~76 ms for the view's kernels | gate probe on a B200 slice (rows 1.07 ms, products 0.84 ms per tile before finding E) | one row-kernel launch and one batched product for all tiles of a view (rows grouped by tile; `J Jᵀ` per tile by a kernel over (tile, i, j) or a strided-batched syrk on padded tiles), each tile's request reading its slice of the view's memoized entry | grid linearization ~1.5–2× on a device | medium-high: tiles have different layouts (held patches), NLLSsolver requests tiles one by one | none |
| 2 | The window lens and tangent bases of a tile are uploaded per tile per linearization (two small host→device copies, each a synchronisation) | code reading; part of the 1.07 ms | hand them over as isbits static arrays in the row kernel's arguments (a 10 × 10 window: 300 numbers) | ~0.2–0.3 ms per tile | low | none |
| 3 | The heights kernel repeats the root and carries 25 partials through Core's implicit step | B200 warped view: 19 ms of 76; CPU: 570 of 2240 ms | with a Core entry point for "implicit partials at a known hit" (3.3 b), read the hit from the record | heights kernel 3–5× | low once Core has the entry point | none |
| 4 | The material's 8 slots are linear in the recorded radiance partials: ∂Φ/∂α_ink = Σ w V G_α · ∂α/∂α_ink, likewise η | derivation; chunks 5 F32 launches of 4 slots | emit the 12 material sums from the record kernel (the same accumulation as the flux, a wider "L"); and ONE t̂ slot instead of one per sensor sample | device chunks 7 → 4 launches; small after A–B (chunks are now ~10 % of a view) | low-medium | none |
| 5 | Closed form on a device downloads the design matrices for QR (398 MB of host allocation per `solve_linear_blocks!` at 228 k px) | benchmark `linear_blocks` bytes | solve Θ_l, Θ_v from the reduction kernel's normal equations with a rank-revealing factorization of the `N × N` system (the joint solve already does) | host memory; time small | medium: the QR's handling of unsupported V nodes must be reproduced; goldens of the closed form would move | none |
| 6 | Host allocation per linearization: the record buffer is allocated and zeroed per view per call (CPU: 2.3 GB per full view) | benchmark `view_jacobian` bytes | reuse per-task buffers as the Jacobian's already do (`_jacobian_buffers`) | GC pressure; a few % | low | none |
| 7 | `CalibrationState.backend::Any`, `device_cache::Any` (a NamedTuple grown by `merge`), `start::Any` | `cost(state)` infers `Tuple{Any, Any}`, `all_parts(state)` an abstract `RadiometryParts` | a type parameter for the backend; a small mutable `DeviceCache` struct with typed optional fields | none measurable (function barriers hold); clarity, fewer locks around `merge` | low | none (internal fields) |
| 8 | `Φ_pixel` allocates 4.4 kB per call on the cluster's Julia 1.13.3 (0 B on 1.13.4 locally) | benchmark `pixel` bytes | investigate when 1.13.4 is on the cluster; likely an escape-analysis difference in `View(pose, camera)` | host-only path | — | none |
| 9 | `src/experiments/` (focus-ab) lives inside the package's hot path: `t̂(camera, 𝐩)` branches on `focus.kind` per sensor sample | code | when the experiment is decided, make the chart a type parameter of `Focus` (dispatch, no branch) or remove kinds 1–2 | small; simplicity | decided: section 6, decision 5 | internal |
| 10 | `closed_form.jl` holds `pixel_parts`/`_linear_rows` (image formation) beside the solves | layout | move the parts to `image_formation/parts.jl` | none; layout | trivial | none |

### 3.3 Proposed for VisualGeometryToolkitCore

a. **A height field's value and gradient in one sweep** (branch `perf/height-field-fused-jacobian` in the monorepo,
   commit ea3cc069, local, not pushed). `_height_root` calls `tps_evaluate` and `tps_gradient` at every Newton
   iteration: two sweeps of the nodes, two `log` per node. One sweep gives the same numbers bit for bit (tested with
   `isequal`). Measured through RC with the method defined in-process (B200, warped 16,384-px view): flux 5.58 → 4.30
   ms, heights 19.1 → 16.0 ms, view Jacobian 76.7 → 66.2 ms; CPU: flux 457 → 290 ms, heights 580 → 406 ms, view
   Jacobian 2326 → 1970 ms. No API change; RC needs no change to benefit.
b. **`intersection` with a known hit / its implicit partials alone.** `_height_implicit` finds the root and then takes
   one implicit step. A public entry point for the step at a supplied hit lets every dual consumer skip the root (RC's
   heights kernel: recommendation 3).
c. **The Newton's iteration count.** `_HEIGHT_NEWTON_ITERATIONS = 4` unconditional iterations, then at least one
   more evaluation to test convergence: ≥ 5 evaluations where a gently warped board converges in 2. Testing
   convergence from the first iteration would roughly halve the root again — but changes last bits (owner's call).
d. **`_launch_rows!(…; workgroupsize)`.** RC's `_launch_pixel_rows!` re-states Core's three-line launch to set the
   CPU workgroup; a keyword on Core's launch removes it.
e. **A device constructor and transpose for `CSRMatrix`** (`CSRMatrix` from device arrays, `transpose` on the device):
   section 4's `csr_rows`/`csr_transpose` are generic and belong beside `CSRMatrix`.
f. **`NoVignetting`** (section 2.3).
g. **CUBLAS wide gemv**: Core's device LM helpers that form `Jᵀr` by `*` on a CUDA matrix have the same 100× penalty
   (finding D); worth a grep in Estimation's `refine_camera` path.

---

## 4. The texture-recovery experiment

**Structure.** The calibration is a nonlinear least squares with a few hundred parameters and a dense, small normal
matrix per block: it forms `JᵀJ` (dense, `ncol ≤ 300`) on the device and solves on the host. The texture test is a
LINEAR least squares with ~10⁶ unknowns (the page's texels) and 10⁶–10⁷ rows (sampled pixels), each row a pixel's
blur footprint over ≤ 4 · (sensor samples × aperture samples) texels: `min ‖A x − b‖² + λ ‖∇x‖²`. Its normal matrix
`AᵀA + λ DᵀD` is sparse with a bandwidth of the blur's diameter and is never formed: Core's `cgls` applies `A`, `Aᵀ`,
`D`, `Dᵀ` as CSR operators on the device (deterministic row kernels, no atomics), one solve per penalty weight on its
own device queue. So "its normal equations on the GPU" means: **the operators `A` and `Aᵀ` built on the GPU**; the
normal equations are then applied there, not stored (forming `AᵀA` explicitly would lose digits and memory for
nothing).

**Before.** Only the footprint kernel ran on the device. Every row's slots were downloaded, pushed entry by entry
into four COO lists on the host (the fit rows and three scored subsets), assembled by `sparse()`, whitened, and
uploaded again inside `PenalizedOperator` (which also builds the transposes on the host). A note in `weighting.jl`
records that a 7 M-row page ran out of host memory in this path.

**Now (36c9de9).**

```text
page_footprints   footprint kernel per photograph: each pixel's texels left ASCENDING in its slots (insertion sort in
                  the kernel) → row-parallel gather → Core CSRMatrix of every sampled row, on the device
fit_operator      csr_rows (the fit rows: one gather) → weights in a row kernel (RC's _whiten) →
                  csr_transpose (keys (column − 1)·nz + entry, one device sort, a binary search per column)
                  → PenalizedOperator(A, Aᵀ, D, Dᵀ)
page_solves       unchanged: Core's cgls per penalty weight
errors            A x − y of the ONE all-rows operator on the device; the scored subsets are index sets
```

The arrays are the host assembly's array for array (rowptr, colval, nzval of `A` and `Aᵀ`; tests on the CPU backend
and on a B200), so the textures are the same numbers. The recorded sums of squared errors are now of CSR row sums
(the host's were CSC column sums): last bits may differ. `page_rows` remains as the host form for tests and small
pages. The experiments' suite at 30ad4c2: 4713 checks succeed, 6 marked broken before this branch, and the GPU item's
8 checks succeed.

Timings: section 5.3 (the refine stage 9.4 → 1.3 s on a B200, the same numbers).

**Next for the experiment** (not done): a Jacobi preconditioner from `diag(AᵀA)` (the squared column norms: one
reduction over `Aᵀ`'s rows) — `cgls` already accepts `diagonal` — would cut iterations on pages with uneven coverage;
it changes the iterates, so the textures of record would move within the solver's tolerance (decided: section 6, decision 4).

---

## 5. Performance

### 5.1 One view's linearization, kernel by kernel (`benchmark/linearization.jl`, ms, min of 5)

B200, flat board, 76,800 pixels, 25 slots, 73 columns:

| | baseline (main) | 115f782 (A, B, F, G) | 30ad4c2 (+C) |
|---|---:|---:|---:|
| flux | 7.0 | 7.0 | 7.1 |
| record F32 | 27.5 | 29.2 | 2.7 (rounded copy) |
| record F64 | 61.8 | 67.1 | 68.1 |
| chunk F32 (× 5) | 17.0 | 3.0 | 2.9 |
| chunk F64 (× 2) | 38.8 | 6.1 | 6.1 |
| parts | 10.0 | 10.5 | 10.7 |
| products | 17.4 | 17.6 | 17.5 (0.4 at ee6c96b: D, E) |
| **view Jacobian** | **235.5** | **127.0** | **103.4** |
| **view normal equations** | **251.7** | **146.3** | **126.2** (104.3 at ee6c96b) |

B200, warped board, free heights, 16,384 pixels (a run's sample size):

| | baseline | 115f782 | 30ad4c2 | + Core 3.3 a |
|---|---:|---:|---:|---:|
| flux | 7.7 | 5.5 | 5.6 | 4.3 |
| record F32 | 12.4 | 12.8 | 0.8 | 0.5 |
| record F64 | 32.2 | 27.7 | 26.6 | −3 |
| chunk F32 (× 5) | 5.7 | 1.1 | 1.1 | 1.0 |
| chunk F64 (× 2) | 11.5 | 1.9 | 2.0 | 1.9 |
| heights | 19.3 | 19.0 | 19.2 | 16.0 |
| parts | 7.5 | 3.8 | 3.7 | 3.6 |
| **view Jacobian** | **102.4** | **82.9** | **75.6** | **66.2** (76.7 before in that job) |

CPU (16 threads), flat 76,800 pixels / warped 16,384 pixels with free heights:

| | flat baseline | flat 115f782 | warped baseline | warped 115f782 | warped + Core 3.3 a |
|---|---:|---:|---:|---:|---:|
| flux | 457 | 451 | 405 | 393 | 290 |
| record | 2084 | 1906 | 695 | 810 | 746 |
| chunk | 3178 | 730 | 2 × 1040 | 172 | — |
| heights | — | — | 504 | 570 | 406 |
| parts | 544 | 514 | 411 | 108 | 112 |
| **view Jacobian** | **6259** | **3875** | **4352** | **2238** | **1970** |

### 5.2 Stages (`benchmark/run.jl`, seconds, best of 3; B200, one whole GPU per job)

320 × 240, 3 views, every masked pixel (228,421), flat board:

| stage | baseline (main) | ee6c96b | speed-up |
|---|---:|---:|---:|
| mask | 0.0011 | 0.0011 | |
| render | 0.48 (the CPU kernel) | 0.0076 (on the device) | 63× |
| flux (one view) | 0.0073 | 0.0075 | |
| cost (3 views) | 0.0203 | 0.0200 | |
| parts | 0.049 | 0.049 | |
| radiometry normal equations | 0.049 | 0.049 | |
| linear blocks | 0.42 | 0.41 | |
| linear joint | 0.47 | 0.47 | |
| view Jacobian | 0.239 | 0.109 | 2.2× |
| view products | 0.0176 | 0.0005 | 35× |
| view normal equations | 0.254 | 0.108 | 2.4× |
| LM, 1 iteration, per view | 1.95 | 1.12 | 1.7× |
| LM, 3 iterations, per view | 3.30 | 1.73 | 1.9× |
| a view's tiles: normal equations (6 tiles) | 0.555 | 0.235 | 2.4× |
| LM, 1 iteration, grid (20 × 15) | 2.00 | 1.21 | 1.7× |
| calibration (joint closed form + 3 LM iterations) | 2.46 | 1.68 | 1.5× |

1280 × 960, 8 views × 16,384 sampled pixels, warped board, free heights (a run's shape):

| stage | baseline | ee6c96b | speed-up |
|---|---:|---:|---:|
| render | 0.43 (CPU) | 0.017 | 25× |
| view Jacobian | 0.103 | 0.075 | 1.37× |
| view normal equations | 0.102 | 0.075 | 1.36× |
| LM, 1 iteration, per view | 1.65 | 1.23 | 1.35× |
| LM, 3 iterations, per view | 3.25 | 2.31 | 1.4× |
| a view's tiles: normal equations (6 tiles) | 0.264 | 0.152 | 1.7× |
| LM, 1 iteration, grid (20 × 15) | 1.94 | 1.26 | 1.5× |
| calibration | 4.63 | 3.30 | 1.4× |

The same with the run's 65 × 50 lattice (108 tiles a view):

| stage | baseline | d63e422 | bfbce88 |
|---|---:|---:|---:|
| a view's tiles: normal equations | 0.390 | 0.281 | 0.31–0.33 |
| the tiles' rows and products alone | 0.137 | 0.138 | 0.144–0.159 |
| LM, 1 iteration, grid | 3.60 | 3.13 | 2.88 |

The warped runs gain less than the flat ones because what remains is the Float64 record (27–33 ms of a view's 75),
the heights kernel (19 ms) and, in the grid route, the tiles' assembly (as much as the view's kernels): the Core
proposals 3.3 a–c and recommendation 1.

CPU, 16 threads (`benchmark/linearization.jl` above; `run.jl`, 320 × 240 flat): view Jacobian 6.23 → 3.75 s, view
normal equations 6.38 → 3.82 s; the forward and closed-form stages unchanged (flux 0.46, cost 1.4, parts 1.8, linear
blocks 3.8, joint 6.5 s). LM, 1 iteration per view at 115f782: 24.5 s; grid 26.5 s.

### 5.3 Texture page solve (`experiments/tools/texture_benchmark.jl`, B200; 10 photographs, 739,848 sampled rows, 443,339 fit rows, 22.0 M fit nonzeros, 240 × 336 texels; warm calls)

| | baseline | 66ddb38 | speed-up |
|---|---:|---:|---:|
| the rows (footprints → operators' rows) | 4.88 s | 0.123 s | 40× |
| the fit operator (A, Aᵀ, D, Dᵀ on the device) | 0.795 s | 0.069 s | 11.5× |
| the three solves (`page_solves`) | 0.228 s | 0.228 s | — |
| `run_page` (both blur models, records) | 9.40 s | 1.28 s | 7.3× |

Unchanged to every printed digit: the iterations (245, 195, 76), the textures' checksums (114911.70310207602;
`run_page` 114911.82477413118), the fit nonzeros' energy and the recorded test and fit sums of squares. Not timed
warm and still on the host: `sample_pixels` (15.5 s first call for 10 photographs: the photograph's read and a host
loop of chief-ray hits).

### 5.4 Tests (cluster, Julia 1.13.3)

| Suite | Commit | Result |
|---|---|---|
| RadiometricCamera, CPU tier (Pkg.test, bounds checks) | ee6c96b | 16,189 checks succeed, 0 fail, 0 error, 3 marked broken before this branch |
| RadiometricCamera, GPU tier (12 items, B200 slice) | ee6c96b | 521 succeed |
| RadiometricCamera, goldens (39 checks, rtol 1e-10) | ee6c96b | 39 succeed; no golden re-recorded |
| docs build (`docs/make.jl`) | ee6c96b | exit 0, no warning |
| experiments, CPU tier + GPU item | 30ad4c2, ee6c96b | 4,713 + 8 succeed, 6 marked broken before this branch |
| RadiometricCamera, all tiers | bfbce88 (the branch's code head) | running when this was written; see the hand-back |

The GPU items had not run since the camera-first refactor; they succeed on the branch.

### 5.5 Numbers that changed, and why

| Change | Where | Size | Why it is valid |
|---|---|---|---|
| chunk partials from the tangent plane / plane hit (A, B) | Jacobian columns, every backend | ≤ 2.2e-14 of a column's norm | the same derivative (implicit-function theorem), another rounding |
| narrower duals (G) | Jacobian with free heights | ≤ 3.6e-15 absolute | the same partials in a narrower dual |
| F32 record = F64 record rounded (C) | device only: the F32 chunks' columns and the parts' hits | up to the F32 estimator's own error (≤ 0.4 % of a column at a printed edge), towards the CPU's values | the F64 estimator is the more accurate one; accepted by the owner (decision 2) |
| one-column gemm for `J r` (D) | device gradient | 3.5e-14 relative | summation order |
| T applied from its stored entries (E) | single-channel states' ties | rounding | summation order; colour states bitwise |
| texture error sums (I) | recorded sse | none seen at 17 digits on the benchmark | CSR row sums instead of CSC column sums; the textures are unchanged |

The residuals (the cost) are the same bits everywhere. No golden was re-recorded.

---

## 6. Decisions

The owner's answers (2026-11-20) to the nine questions of this report's first version, and where each stands on the branch.

| # | Question | Decision | Status |
|---|---|---|---|
| 1 | `lens` vs `geometric` | RENAME: the field, `geometric(camera)` / `geometric(state)`, the constructor's first argument, `with_lens` → `with_geometric`; the old spellings refused by name (src/api.jl); docs, README / index / landing, examples, experiments, tests follow | **not started** (≈ 150 sites; `r.lens`, `st.lens`, `m0.lens`, `ds.lens₀` are OTHER structs' fields and stay; the records' JSON keys are a separate question to settle with the rename). The constructor's argument is already named `geometric` (ee6c96b). |
| 2 | Finding C's device numbers | ACCEPTED. Follow-up of high priority: re-centred Float32 geometry with Float64 accumulation (each pixel's chief-ray hit once in Float64 as a local origin, the per-ray hits and texture / edge lookups as Float32 offsets from it), measured against the 0.4 % gate per tile on the Siemens case, to drop the Float64 estimator and record (Float64 is 1/64 rate on the RTX Pro 6000s, 1/2 on a B200) | follow-up, not started |
| 3 | Core 3.3 a–g | DO ALL, pushed and released through the stack's release process (the owner posts the Registrator comment), then RC adopts | a: committed locally on the monorepo branch `perf/height-field-fused-jacobian` (ea3cc069), not pushed; b–g not started |
| 4 | Texture solve | (a) Krylov.jl (LSMR or CGLS, Jacobi-preconditioned by diag(AᵀA)), the penalty stacked `[A; √λ D]`, cuSPARSE CSR for `A` and an explicit `Aᵀ` in the CUDA extension; benchmark against `cgls`, verify the textures within tolerance, delete the redundant custom solver / CSR code, re-run the texture records once. (b) the error scoring stays on the device | (b) done (36c9de9); (a) not started. The device construction of this branch (`page_footprints`, `fit_operator`) produces exactly the CSR arrays a `CuSparseMatrixCSR` holds, so it carries over. |
| 5 | focus-ab | keep the parameterizations as a TYPE PARAMETER of `Focus` (no `kind` branch), moved out of `src/experiments/`; find the experiment's records and re-run it on the cluster | not started |
| 6 | Grid route | one row-kernel launch and one batched product per view for all its tiles, on this branch, after 1, 3, 4 | not started; the measurements of section 5.2 (65 × 50) are its baseline |
| 7 | Release | finish the round on the branch, merge, register 0.1.0 from that commit; no version bump | — |
| 8 | Default vignetting | `NoVignetting()` (Core, 3 f): V ≡ 1 by dispatch, no parameters, no columns | needs 3 f; the branch's default is still the flat 4 × 4 surface |
| 9 | CPU workgroup | keep 16, the measurement cited at the definition | done (`_pixel_workgroup`'s docstring) |

When the round is complete: the `simplify` and `code-review` skills on the branch's diff (and the monorepo branch), their
confirmed findings fixed, and the gates again (suite, goldens, GPU tier, experiments).
