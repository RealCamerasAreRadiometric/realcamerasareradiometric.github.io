# RadiometricCamera: design and performance report

Branch `perf/optimize-and-compose` (from `main` at 8b3ba04), 2026-11-20. Not a published docs page. Measurements: the
benchmark scripts in `RadiometricCamera/benchmark/` and `experiments/tools/texture_benchmark.jl`, run as Slurm jobs on
poseidon (one whole NVIDIA B200 per GPU job; CPU jobs 16 threads on Emerald Rapids), Julia 1.13.3. Wall times are
`time_ns` around calls that return their values to the host (device kernels awaited); no vendor profiler was used
except where a line says so. Records: `/cluster/projects/vgtk/logs/rc-perf/results/*.json`.

Contents: 1 architecture map · 2 building a camera · 3 findings · 4 the texture experiment · 5 performance · 6 open
decisions.

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
   break; needs the owner's decision (question 1).
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
| E | The column map T was uploaded and applied by three dense device products per block | gate probe: 0.84 ms of a tile's 1.43 ms | `J r`, `J Jᵀ` on the backend, T applied on the host as a sparse matrix (a6fdd8b) | measured in the next benchmark run | colour states bitwise the same (checked on a view); single-channel ties may round |
| F | KernelAbstractions' CPU workgroups of 1024 left small views to few threads | 1,728-px view on 32 threads: flux 49 ms at the default, 5.4 ms at 16 px per group, same bits | per-pixel kernels and the row kernel in groups of 16 on the CPU backend (3a20b71) | small views up to 9× per kernel; no change for ≥ 16 k px per thread-count | none (bitwise) |
| G | The nodal heights reserved 25 dual slots no chunk seeds | layout: P = 50 with free heights, the CPU's one chunk 50 wide and split in two | heights take no slot (d1ab18f) | CPU warped view with free heights 212 → 180 ms (local) | ≤ 3.6e-15 absolute |
| H | `render_view` ran on the CPU only | B200: CPU render 0.46 s, device flux kernel 7 ms for the same view | `render_view(…; backend)` (dd1e3d1) | 65× for a device render | additive API |
| I | The texture page's operators were assembled on the host | section 4 | built on the device (36c9de9) | section 4 | operators array-for-array the host's |

### 3.2 Recommended next, in RadiometricCamera

| # | Finding | Evidence | Recommendation | Expected gain | Risk | API |
|---|---|---|---|---|---|---|
| 1 | **Grid route: tile assembly is latency-bound.** A view's 108 tiles (65 × 50 lattice, ~150 px each) each launch a row kernel, upload their window lens and bases, and form their products: 1.43 ms per tile, 154 ms per view, against ~76 ms for the view's kernels | gate probe on a B200 slice (rows 1.07 ms, products 0.84 ms per tile before finding E) | one row-kernel launch and one batched product for all tiles of a view (rows grouped by tile; `J Jᵀ` per tile by a kernel over (tile, i, j) or a strided-batched syrk on padded tiles), each tile's request reading its slice of the view's memoized entry | grid linearization ~1.5–2× on a device | medium-high: tiles have different layouts (held patches), NLLSsolver requests tiles one by one | none |
| 2 | The window lens and tangent bases of a tile are uploaded per tile per linearization (two small host→device copies, each a synchronisation) | code reading; part of the 1.07 ms | pass them as isbits static arrays in the row kernel's arguments (a 10 × 10 window: 300 numbers) | ~0.2–0.3 ms per tile | low | none |
| 3 | The heights kernel repeats the root and carries 25 partials through Core's implicit step | B200 warped view: 19 ms of 76; CPU: 570 of 2240 ms | with a Core entry point for "implicit partials at a known hit" (3.3 b), read the hit from the record | heights kernel 3–5× | low once Core has the entry point | none |
| 4 | The material's 8 slots are linear in the recorded radiance partials: ∂Φ/∂α_ink = Σ w V G_α · ∂α/∂α_ink, likewise η | derivation; chunks 5 F32 launches of 4 slots | emit the 12 material sums from the record kernel (the same accumulation as the flux, a wider "L"); and ONE t̂ slot instead of one per sensor sample | device chunks 7 → 4 launches; small after A–B (chunks are now ~10 % of a view) | low-medium | none |
| 5 | Closed form on a device downloads the design matrices for QR (398 MB of host allocation per `solve_linear_blocks!` at 228 k px) | benchmark `linear_blocks` bytes | solve Θ_l, Θ_v from the reduction kernel's normal equations with a rank-revealing factorization of the `N × N` system (the joint solve already does) | host memory; time small | medium: the QR's handling of unsupported V nodes must be reproduced; goldens of the closed form would move | none |
| 6 | Host allocation per linearization: the record buffer is allocated and zeroed per view per call (CPU: 2.3 GB per full view) | benchmark `view_jacobian` bytes | reuse per-task buffers as the Jacobian's already do (`_jacobian_buffers`) | GC pressure; a few % | low | none |
| 7 | `CalibrationState.backend::Any`, `device_cache::Any` (a NamedTuple grown by `merge`), `start::Any` | `cost(state)` infers `Tuple{Any, Any}`, `all_parts(state)` an abstract `RadiometryParts` | a type parameter for the backend; a small mutable `DeviceCache` struct with typed optional fields | none measurable (function barriers hold); clarity, fewer locks around `merge` | low | none (internal fields) |
| 8 | `Φ_pixel` allocates 4.4 kB per call on the cluster's Julia 1.13.3 (0 B on 1.13.4 locally) | benchmark `pixel` bytes | investigate when 1.13.4 is on the cluster; likely an escape-analysis difference in `View(pose, camera)` | host-only path | — | none |
| 9 | `src/experiments/` (focus-ab) lives inside the package's hot path: `t̂(camera, 𝐩)` branches on `focus.kind` per sensor sample | code | when the experiment is decided, make the chart a type parameter of `Focus` (dispatch, no branch) or remove kinds 1–2 | small; simplicity | owner's call (question 5) | internal |
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

Timings (`experiments/tools/texture_benchmark.jl`, synthetic sheet, 10 photographs): the baseline and branch jobs are
running; the table is in section 5.3.

**Next for the experiment** (not done): a Jacobi preconditioner from `diag(AᵀA)` (the squared column norms: one
reduction over `Aᵀ`'s rows) — `cgls` already accepts `diagonal` — would cut iterations on pages with uneven coverage;
it changes the iterates, so the textures of record would move within the solver's tolerance (question 4).

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
| products | 17.4 | 17.6 | 17.5 (→ ~2.6 with D: pending run) |
| **view Jacobian** | **235.5** | **127.0** | **103.4** |
| **view normal equations** | **251.7** | **146.3** | **126.2** |

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

### 5.2 Stages (`benchmark/run.jl`, seconds, best of 3)

B200, 320 × 240, 3 views, every masked pixel (228,421), flat:

| stage | baseline | 115f782 | note |
|---|---:|---:|---|
| mask | 0.0011 | 0.0012 | |
| render (CPU kernel) | 0.48 | 0.46 | on the device from dd1e3d1: the flux kernel, 0.007 |
| flux (one view) | 0.0073 | 0.0072 | |
| cost (3 views) | 0.0203 | 0.0200 | |
| parts | 0.049 | 0.049 | |
| radiometry normal equations | 0.049 | 0.050 | |
| linear blocks | 0.42 | 0.39 | 398 MB host (recommendation 5) |
| linear joint | 0.47 | 0.47 | |
| view Jacobian | 0.239 | 0.132 | |
| view normal equations | 0.254 | 0.149 | |
| LM, 1 iteration, per view | 1.95 | 1.33 | |
| LM, 3 iterations, per view | 3.30 | 2.12 | |
| a view's tiles: normal equations | 0.555 | 0.292 | 6 tiles (20 × 15 lattice) |
| LM, 1 iteration, grid | 2.00 | 1.27 | |
| calibration (joint closed form + 3 LM iterations) | 2.46 | 1.83 | |

CPU, 16 threads, the same scene: view Jacobian 6.23 → 3.75 s; view normal equations 6.38 → 3.82 s; the forward and
closed-form stages unchanged (flux 0.46, cost 1.4, parts 1.8, linear blocks 3.8, joint 6.5 s). LM, 1 iteration per
view at 115f782: 24.5 s (baseline run still in the queue).

The runs with C, D, E, H together (66ddb38 and later), the realistic warped set (8 views × 16,384 px) for the baseline,
the run's 65 × 50 lattice (108 tiles a view) and the texture benchmark are in progress; this section will be updated
with them.

### 5.3 Texture page solve

Pending (jobs `base-gpu`, `head3-gpu`: `tex` set).

### 5.4 Tests

- RadiometricCamera suite on the cluster (Pkg.test, bounds checks on), at 30ad4c2: the CPU tier 16,188 checks succeed,
  0 fail, 3 marked broken before this branch, 1 error — `ExplicitImports` reported the import of `lens` stale after
  the accessor was added; fixed in 66ddb38 (run in progress). The GPU tier and the goldens run after the CPU tier in
  that job: results pending.
- Experiments suite at 30ad4c2 (CPU tier + GPU item on a B200 slice): 4,713 + 8 checks succeed, 6 marked broken before
  this branch.

### 5.5 Numbers that changed, and why

| Change | Where | Size | Why it is valid |
|---|---|---|---|
| chunk partials from the tangent plane / plane hit (A, B) | Jacobian columns, every backend | ≤ 2.2e-14 of a column's norm | the same derivative (implicit-function theorem), another rounding |
| narrower duals (G) | Jacobian with free heights | ≤ 3.6e-15 absolute | the same partials in a narrower dual |
| F32 record = F64 record rounded (C) | device only: the F32 chunks' columns and the parts' hits | up to the F32 estimator's own error (≤ 0.4 % of a column at a printed edge), towards the CPU's values | the F64 estimator is the more accurate one; `HIGH_FLOAT` exists because the F32 one is not accurate enough for the pose |
| one-column gemm for `J r` (D) | device gradient | 3.5e-14 relative | summation order |
| T applied sparsely (E) | single-channel states' ties | rounding | summation order; colour states bitwise |
| texture error sums (I) | recorded sse | last bits | CSR row sums instead of CSC column sums; the textures are unchanged |

The residuals (the cost) are the same bits everywhere. No golden was re-recorded.

---

## 6. Open decisions for the owner

1. **`lens` vs `geometric`.** Rename the camera's field and accessor so that "lens" means Core's lens everywhere?
   (Public break; I held it back.)
2. **Finding C changes device numbers** (towards the CPU's). Accept, or keep two estimator runs for bit-continuity of
   device runs of record?
3. **Core changes.** May 3.3 a (bitwise-neutral, branch ready locally) be pushed and released as a Core patch? Do you
   want 3.3 b–g as issues or as branches?
4. **Texture experiment.** Is a Jacobi-preconditioned CGLS acceptable for the page solves (iterates change within
   the tolerance), and should the scored subsets' errors stay on the device as now?
5. **focus-ab.** Is the experiment decided? Its `kind` branch sits in the hot path (recommendation 9).
6. **Grid route.** Recommendation 1 (one launch per view for all tiles) is the largest remaining device gain I see for
   the paper's spline runs and the largest change. Go ahead on this branch, or separately?
7. **README example.** The README, the docs' index and the landing page now show the composed constructor, which
   0.1.0 as registered does not have. Keep for the next release, or revert the three pages until then?
8. **Default vignetting** of the composed constructor: the flat 4 × 4 surface (as the README used), or required?
9. **CPU workgroup of 16 pixels** is a fixed number in `_pixel_workgroup` (no option, as the window's width is
   derived): fine, or derive it from the thread count?
