# Golden outputs

`golden_scenes.jl` builds small fixed problems (the 256² blob fixture board, two views, 32 × 32 pixels) and evaluates the flux of three
lens / board combinations, the residual and the Jacobian of a per-view problem (Kannala–Brandt) and of two per-tile problems (the grid
model with the board warping free, a distance-map board with the printed-edge block), the closed-form joint
radiometry and three Levenberg–Marquardt iterations with explicit stopping tests.

| file | recorded with | used by |
|---|---|---|
| `golden_migration_snd_try.json` | this branch's package (`julia --project=. test/regression/record_golden.jl …`) | `test/regression/golden_tests.jl`, at rtol 1e-10: the reference for a migration |
| `golden_board_space.json` | the package before the second migration (branch board-space, `GOLDEN_OLD_STACK=1`) | record only: the second migration reproduced every value to 1e-10 except the Jacobian columns through the printed-edge profile (≤ 4.9e-6 relative), where the derivative of the normal CDF became the derivative of the approximation that is evaluated |

Re-record `golden_migration_snd_try.json` only for an intended numerical change.

Re-recorded 2026-10-26 for the log-parametrised aperture radius (the LM step of the scalar r is d ln r, `per_view.jl` `ApertureVar`): the r
column of the per-view and per-tile Jacobians is now r ∂Φ/∂r (exactly ×0.0386 in these scenes) and the three LM iterations with r free
follow a different path (cost 1.06e-10 → 4.93e-11); every other recorded value, including the non-central r(𝐩) spline column, is unchanged at 1e-10.

Re-recorded 2026-10-28 for the pose chart: the LM step of a view's pose is VisualGeometryToolkitCore's `LeftRigidManifold6`
(`𝐜 ← exp([ω]ₓ)𝐜 + ρ`) instead of the package's own `exp([ω]ₓ)𝐜 + J_l(ω)ρ` (`pose.jl`). The two charts agree to first order (J_l(0) = I),
so every Jacobian is unchanged at 1e-10 and only the three LM iterations follow another path: lm3 moves by ≤ 6.8e-10 (Θ_d), 3.0e-10 (poses),
4.9e-10 (material), 1.8e-11 (r), cost 4.92759147e-11 → 4.92759189e-11. Run to convergence, both charts reach the scene's exact fit
(cost 4e-32 and 1e-32) with the same parameters (Θ_d within 5.7e-13, poses within 1.3e-15).

Re-recorded 2026-10-28 for the board warping's height basis: the free height directions are VisualGeometryToolkitCore's
`TargetManifold` of the height field (`HEIGHT_BASIS`, its `retract_jacobian`) instead of the package's `nullspace` basis `HEIGHT_Z` —
another orthonormal basis of the same subspace (the complement of the piston and the two tilts). Only the warping tile's 22 height
columns of its Jacobian change (their coordinates are rotated); mapped back to nodal heights they are the old ones, `J_h Bᵀ = J_h,old Zᵀ`
to 2.6e-14 relative on every recorded row, and every other recorded value is unchanged at 1e-10 (the joint radiometry's cost is the
roundoff floor, checked by its absolute bound).

Re-recorded 2026-11-03 (Julia 1.13.2, 8 threads) for three intended changes of 30–31.10 that did not re-record it (the file
failed 32 of its 43 checks from then on). Attributed by evaluating the scenes on the current code with the soft pin forced
(`hard_pin = false`): every value then matches the old file to ≤ 2e-14 except the columns of the two chart changes, which
match after their chain-rule factor, and the three LM iterations.

- the material block on charts (α = eᵘ, η = 1 + eᵘ, μ logistic on the μ bounds (0, 1)): the 9 material columns of
  every Jacobian are the old ones times dα/du = α (0.12, 0.35), dη/du = η − 1 (0.65 × 3, 0.33 × 3) and dμ/du = μ(1 − μ) = 0.2475
  at the start's μ = 0.55; measured on every summary (Σ, Σ|·|, Σ², the four rows) of all four Jacobians to ≤ 4.1e-14 relative.
- the non-central r(𝐩) spline on the log chart: the non-central tile's 25 r(𝐩) columns are the old ones times the
  node radii r_i = 0.0386 (1 + 0.05 sin(i + j)) (≤ 2.3e-15).
- the hard vignetting pin (the default `CalibrationState(…; hard_pin = true)`): one vignetting column fewer in every
  Jacobian (per-view 76 → 75, tiles 290 → 289, 272 → 271, 492 → 491). A pinned state is normalised to V(𝐜) = 1 when it is built,
  so the targets now render V/V(𝐜) (V(𝐜) = 0.9833) and the start's 1.05 × V is the same field, a pure gauge move: the residuals
  carry the μ and lighting offsets only (they move by 0.34 of their scale), and the joint radiometry returns the truth's lighting
  a instead of V(𝐜) a (0.017 = 1/V(𝐜) − 1 relative); its V, μ and cost floor are unchanged.
- lm3: the three iterations take another path under the new charts and the pin (cost 4.90e-11 → 1.41e-10; Θ_d and the poses
  move by ≤ 1.1e-4 relative, t̂ 2.9e-3, r 2.2e-2, the material 2.4e-2, μ 3.6e-2). Run to 40 iterations on the current code
  the scene reaches its exact fit (cost 3e-33) with Θ_d within 7e-13, the material 5e-14, μ 3e-14, r 1e-16, the poses 2e-16,
  V within 1e-15 of the pinned truth and the lighting 1e-13 of the truth's a.

The flux values are unchanged (0). Note that under the hard pin the scenes' 1.05 × V start no longer moves V away from
the truth (only μ and the lighting do).

Re-recorded 2026-11-05 (Julia 1.13.2, 8 threads, RadiometricCamera 0.5.0 on the registered VisualGeometryToolkitCore 0.5.6) for two
intended layout changes that the test had compared through a mapping: the non-central tile's Jacobian has 591 columns, not 491,
because its patch blocks carry (δ₁, δ₂, Δx, Δy, Δz) per node with the chief node's origin pinned, where they had (δ₁, δ₂, Δx, Δy) with
Δz held (the origins move in 3-D). The warping tile's 22 height columns are in Core 0.5's pure-bending complement of the
affine heights, where they had been in the orthogonal one. The recording now also carries the warping's `height_cols` and
`height_proj`. Every other output is the earlier recording's to 1.1e-14 relative to its vector's scale (lm3; the Jacobians 6e-19;
the fluxes, residuals and radiometry 0). Both items now compare every field at 1e-10, like the rest: the non-central one's
column mapping and its `@test_broken` are gone, and so is the warping one's basis-free Gram comparison, which assumed an orthonormal
recorded basis.

Re-recorded 2026-11-11 (Julia 1.13.2, 32 threads; every value compares at rtol 1e-10, the radiometry's cost by its absolute bound)
for two intended changes, the gauges held exactly (owner rule: no pin, prior, penalty or tolerance holds a gauge). Attributed
against the previous file, summary by summary:

- The flux is the aperture integral over πr² (`normalize_aperture = false`, under which r² traded against the lighting's scale, is
  deleted; the scenes had recorded the un-normalized integral). Every central flux, residual and Jacobian column is the old one
  times 1/(πr²) = 213.63653 (r = 0.0386) to ≤ 1.4e-14 relative, except the aperture column (it gains the derivative of 1/(πr²):
  per-view column 24, tile column 216) and the warping tile's 22 height columns, whose basis-invariant `height_proj` scales
  likewise (4.7e-14). The non-central flux, residual and Jacobian are unchanged (≤ 1e-14: that lens always averages per
  position). The joint radiometry's V, a and μ are unchanged; lm3 takes another path (cost ×621.45, the scale; r, μ and the
  material move by ≤ 5 %).
- The grid's common rotation is held by three direction coordinates (`rotation_gauge_coordinates`): the tile Jacobians' columns of
  the chief node's two tangent coordinates and of the roll node's one are zero (central tiles: columns 97, 98, 148; non-central:
  232, 233, 358); every other column as above (the non-central ones bit for bit).

Re-recorded 2026-11-11 on dgx01 (Julia 1.13.3, 8 threads; the stack at VisualGeometryToolkit's 0.9 head b4aa170b) for two intended
changes of branch rc-gpu-jacobian: the CPU Jacobians are the device's kernels run on the CPU backend in Float64 (the
radiance estimator's local Jacobian once per ray, the nodal heights through the ray parameter, one chunk of every dual slot), and the
ray parameter's height partials come from the implicit function theorem at the height field's Newton hit. Attributed
against the previous file: lm3's final cost moved by 2.5e-10 relative (its poses, lens, focus, aperture, material and μ by ≤ 2.6e-13);
every flux, residual and Jacobian summary is the previous recording's to ≤ 2.4e-14 (the warping tile's height columns through the
basis-invariant `height_proj`: 1.5e-14); the joint radiometry's V, a, μ to ≤ 5.8e-16 (its cost is the roundoff floor).

The non-central model is removed (2026-11-14): `golden_scenes.jl` loses its scene 5 and the record its three entries
(`flux_noncentral`, `tile_noncentral_residual`, `tile_noncentral_jacobian`), cut from the file as text — no other byte of the
record moves, nothing is re-recorded. The entries above that name the non-central lens are the history of those three.

The camera's in-focus blur is removed (2026-11-14): scene 4, the printed-edge tile, had the blur as its edge's width (σ_cam 0.9 px
with σ_print 0.3 mm) and three σ_cam columns. It now has σ_print 3 mm (about one pixel on the board) and no blur, so its two entries
are re-recorded: `tile_edge_residual` (3060 values) and `tile_edge_jacobian` (268 columns, 271 before: the block's finite-difference
column is the ink spread δ alone), on rtx-01-mds (Julia 1.13.3, 8 threads) and spliced into the record as text — no other byte of
it moves. Every other scene has no blur and is the previous commit's bit for bit on that machine (9000 numbers). A full recording
there differs from this record in the last digits of `lm3`, `perview_jacobian` and `tile_warping_jacobian` (the machine's
rounding; `lm3`'s cost by 1.35e-10 relative, the one check that misses rtol 1e-10 there), which is why only the two entries moved.
`lm3`'s final cost is now bounded by eps·cost0 instead (2026-11-19): its three iterations cancel the cost from cost0 ≈ 7.65 to
≈ 8.79e-8 (×8.7e7), so the final cost's absolute rounding scales with eps·cost0, not eps·cost — on the desktop (rtx-01-mds) it is
the dgx01 record's to 1.02e-10 relative, 9.0e-18 absolute against the bound 1.7e-15, while every other lm3 field agrees to ≤ 1e-13. The
record gains the start cost `cost0` (inserted as text, no other byte moves); cost0 has no cancellation and stays under rtol 1e-10.
`record_golden.jl` writes `gold_outputs()` whole, so a re-recording carries `cost0` with it.

The printed edge is never a parameter (2026-11-14): scene 4's tile is built without the edge block, so `tile_edge_jacobian` loses
its one printed-edge column, the ink spread δ's forward difference (column 226: 267 columns, 268 before). Every other column of
it, `tile_edge_residual` and every other scene are the previous recording's bit for bit on rtx-01-mds; the entry is spliced into the
record as text.
