# Integration tests: the BlobBoards stack and VisualGeometryCore

Purpose: after upgrading the stack in the package environment (the old-stack BlobBoards front-end environment and its stage A
were deleted on 2026-11-13: they named the retired VisualGeometryCore 0.16 checkouts), run one command and learn whether the radiometric refinement still works with the new blob detector,
calibration and initialisation, whether the ray tracer still works on the new core, and whether the numbers moved.

```bash
# 1. once, BEFORE the migration, on the reference stack: record the golden values (versioned in golden/)
bash test/integration_blobboards/run_integration_tests.sh --record
# 2. after the migration
bash test/integration_blobboards/run_integration_tests.sh                     # ≈ 15 min, GPU 1; exit code 0 = everything succeeded
bash test/integration_blobboards/run_integration_tests.sh --with-pkg-tests    # + the package suite (VisualGeometryCore bump)
```

Every invocation writes into a fresh `work/run-<time>/` (git-ignored; `work/latest` points at it): the scratch files of every stage, `logs/<stage>.log`. Nothing is deleted. The golden file (`golden/session_fisheye3.json`)
and the stack it was recorded on (`golden/recorded_stack.txt`) are versioned.

Fixture: `test/fixtures/lenssim_fisheye3/` (README there) — three synthetic fisheye views of blob board 291 (`v1`–`v3`, the
session input), the rendered board-UV truth of those and of three held-out views (`val1`–`val3`), the manifest with the true
poses, and the board-291 export. Plus `test/fixtures/blobboards_256/`, the package test suite's 256² pattern.

## What is checked, stage by stage

**C `test_session_outputs.jl`** (the harness env `test/integration_blobboards/Project.toml`, a member of the repository's workspace, resolved in the one root Manifest; until 2026-10-29 `experiments/`) — RadiometricCamera on the session RECORDS, the stack's
interface: `test/fixtures/lenssim_fisheye3/session_io/` holds VisualGeometryToolkitIO's `camera.json` (cameras/v4, COLMAP pixels),
`targets_1.json` (rig/v6: board 291 with its canvas and texture), `target_poses.json` and `trajectory_1.json`, written once
(2026-10-28) by CaliBlobs' `scripts/migrate_rc_session.jl` (caliblobs-rc-sessions f178276, BlobBoards blobboards-rc-sessions
68b4854) from the BlobBoards session export of the three training views (rc-baseline work/run-20261026-135120/session_out).
A new session is written by CaliBlobs' `save_calibration` in the same records; RadiometricCamera reads them and depends on
neither BlobBoards nor CaliBlobs. Absolute checks plus the comparison with `golden/session_fisheye3.json`:

| # | consumer | absolute check | golden tolerance |
|---|---|---|---|
| 0 | fixture conventions | board-291 chart from `geometry.json` = the checked-in corners; the COLMAP pixel frame (`𝐩_of_pixel`); the rig record loads | — |
| 1 | `camera.json` → the division camera as Core's pixel camera (a `BoundedCamera` of a `Division` camera model) | the session's residual median (provenance) < 0.15 px | f 0.5 %, centre 3 px, k 0.01, residual 0.03 px |
| 3 | the 35 × 35 grid init sampled from that camera (`GridLens`) | vs the division camera inside ρ ≤ 0.4: median < 0.05 mrad, p95 < 1 mrad | — |
| 4 | `target_poses.json` → Blender world (similarity on the board frame) vs the manifest's true poses (centre at the entrance pupil) | rotation < 0.5°, centre < 10 mm, board similarity residual < 1e-9 | 0.1°, 3 mm |
| 6 | held-out evaluation of the division camera and the grid on `val1`–`val3`: dense lattice of true board points, pose refit per view (Core's `LevenbergMarquardt` on VisualGeometryToolkitEstimation's `BearingPoseProblem`), bearing error × f_px | median < 0.6 px, > 85 % of the lattice kept | median ≤ 1.5 × golden + 0.02 |

Not in stage C any more (2026-10-28): the blob detections (the session's IO observations record observations.json, not yet written by its producer; the blob front end refuses a session without it), the CGM
samples (not migrated: a function of the camera), `load_blobboards_session` and the blob front-end refinement smoke of
`bin/refine.jl` (they read the session export; the de-duplication's part 2 moves them onto these records and brings the items
back).

**D `test_vgc_contract.jl`** (the harness env, no BlobBoards) — the contract of the ray-tracing code with VisualGeometryCore,
which the package pins separately (`[compat] VisualGeometryCore = "0.6"`, the path dependency `<dev-checkout>/VisualGeometryCore.jl`):
- the imported names exist (`AbstractLensModel`, `AbstractIntrinsics`, `LogicalIntrinsics`, `CameraCalibrationMat`,
  `CameraModel`, `Camera`, `EuclideanMap`, `Plane3D`, `Point3`, `Vec3`, `FinitePupil`, `backproject`, `project`,
  `has_distortion`, `pupil_style`, `has_closed_form_inverse`, `undistort_normalized`, `lens`, `is_central`, and the lens
  types the adapter wraps: `KannalaBrandt`, OpenCV's `Lens(Perspective(), Additive(Radial(…), Tangential(…)))`, `Lens(Perspective())`, `Lens(Equidistant())`); the four lenses subtype
  `AbstractLensModel`;
- the interface methods the package adds for `DoubleSphere`, the pixel camera (Division) and `VGCLens` still dispatch: `camera(lens, ext)`
  builds a `VGC.Camera` whose `backproject` equals `π⁻¹`, the functor / `undistort_normalized` pair inverts, `project`;
- `VGCLens` on Kannala–Brandt, Brown–Conrady and pinhole lenses agrees with VGC's own `backproject(intrinsics, lens, pixel)`;
- `CameraPoseCW` ↔ `EuclideanMap` round trip (both constructor forms), `retract` on the map;
- the board plane: `Plane3D` intersection equals `intersect_board`, the plane rebuilds from its coefficients;
- `DoubleSphere` parity with VGC's own SoA kernel (its argument order is version-specific);
- ray-tracing smoke on the CPU with every lens type (`DoubleSphere`, the pixel camera (Division), `GridLens`, `VGCLens`): `compute_mask`,
  `render_view`, `CalibrationState`, `cost` at the generating state (residual 0) and after a 10 mrad pose move (residual > 0);
- **hand-computed fixed values** (round trips cannot catch a convention that flips in both directions at once): the
  pinhole, unified-camera (ξ = 0.5) and division-model (k₁ = −0.1) back-projections of the normalized point (0.3, 0.2) to 1e-6, the
  adapter's pinhole; a pixel through a pose rotated 90° about z with centre (0.1, 0.2, 0) hitting the board at (0.35, 0.7, 2)
  = chart (0.675, 0.85), the camera coordinates (0.5, −0.25, 2) through `R_cwᵀ(X − 𝐜)` and through the VGC `EuclideanMap`,
  and the projection back to the pixel; `𝐩_of_pixel` / `pixel_of_𝐩` (the COLMAP pixel frame);
  and one **rendered** check: a board with a single blob at chart (0.6, 0.35), seen by a pinhole 1.5 m above it, must show its
  dark spot at the predicted 1-based pixel (34.1, 30.9).

**E `Pkg.test()`** (`--with-pkg-tests`) — the package's full suite (19 testsets, GPU included). Run it whenever
VisualGeometryCore is bumped in the package environment; the bump itself is a manual step (edit `[compat]` and the path
dependency, re-resolve), stages C and D then tell whether the refinement and the ray tracer still work on top of it.

## Reading a failure

- **Stage C absolute checks succeed, golden checks fail**: the new detector or calibration changed the numbers beyond the
  tolerances but the pipeline is consistent. Decide whether the change is acceptable, then re-record.
- **Stage D fails**: VisualGeometryCore in the package environment changed a type, a constructor or an interface signature
  the lenses extend; the failing testset names it. Fix `src/radiometric_central_model/{double_sphere,rational_lens,vgc_lens}.jl`,
  `src/radiometric_back_projection/pose.jl` or `src/target/printed_target.jl` accordingly, then run `--with-pkg-tests`.

## Not covered

The 0.21 calibration flow (`load_blobboards_calibration`); the photo pipeline on real photographs (the same code path as
item 5 and 7, different images — the recipes under `experiments/generic-spline/scripts/` are the full runs);
Schöps's pipeline; the GPU backend beyond the two-iteration refinement (stage E covers it); OpenCV.

## History

Ported on 2026-10-04 from the old repository's `test/migration/` (branch `experiments-old`), where stage C ran the old
experiment scripts (`make_blob_features.jl`, `cgm_to_grid.jl`, `gt_board_eval`, `control_division.jl`, `ours_fisheye_arm.jl`);
here it runs the package functions and the clean refinement driver instead, with the held-out metric of the experiments
(bearing error after a pose refit) in place of the old forward-projection error. The golden values were re-recorded on the
same stack as before (BlobBoards 0.22.0, VisualGeometryCore 0.16.0 / 0.6.0).
