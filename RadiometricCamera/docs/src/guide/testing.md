# Run the tests and check a change

```@meta
CurrentModule = RadiometricCamera
```

## Run the tests

The suite is ReTestItems, as in every stack package: each `test/**/*_tests.jl` file holds `@testitem`s, each item runs in a
module of its own on a pool of worker processes, and `test/runtests.jl` runs them in three runs by tag — the untagged CPU
items on the pool (`RETESTITEMS_NWORKERS` workers, default one per two logical CPUs, 2 threads each), then the `:gpu` items
(the device parts) on one worker with 8 threads when a functional GPU with ≥ 20 GiB free is visible and `RC_TEST_GPU` is not
`0` (the runner decides once and hands the verdict to the worker, whose items fail with the reason if the GPU is no longer
usable there), then the `:golden` regression on one worker with 8 threads — the closed-form radiometry sums over thread
chunks, so the thread count is part of the arithmetic, and 8 is the `JULIA_NUM_THREADS` the pipeline checks fix
(`experiments/test/pipeline/run_pipeline_checks.sh`); the goldens (rtol 1e-10) also succeed on one thread.

`julia --project=. -e 'using Pkg; Pkg.test()'` runs everything; `Pkg.test(; test_args = ["focus|grid"])` runs the items whose
name matches the pattern (in every run); `RETESTITEMS_NWORKERS=0` runs every item serially in the calling process, for
debugging. The long pole is the GPU run on its one worker.

Two items hold the documentation: `docs` (every public name has a docstring, attached to its definition, whose signature
lines match the methods) and `README` (the README's example runs verbatim in a fresh environment built from its install
line, and the landing page shows the same install line and example). `Pkg.test(; test_args = ["docs|README"])` runs them.
The manual's own gates — every public docstring in the manual, the doctests and `@example` blocks, the links — are its
build's: `julia +1.13 --project=docs --startup-file=no docs/make.jl`.

## Check a change

Four checks hold the code to the behaviour of this release; run them before and after a change (a dependency bump, a Julia
version, a refactor). Checks 1 and 2 run in the package's directory `RadiometricCamera/`, checks 3 and 4 in the repository's
root:

1. **`julia --project=. -e 'using Pkg; Pkg.test()'`** — the unit tests, `test/unit/jacobian_richardson_tests.jl` (the pathwise
   Jacobian against Richardson-extrapolated finite differences: ≤ 1e-6 relative for every dual-number family, ≤ 1e-10 for the
   analytic columns) and `test/regression/golden_tests.jl` (flux, residuals, Jacobians, the closed-form radiometry and three LM
   iterations of fixed scenes against `test/regression/golden_migration_snd_try.json`, recorded from this release's code, at 1e-10).
   The GPU items run when a GPU with ≥ 20 GiB free is visible.
2. **`bash test/integration_blobboards/run_integration_tests.sh`** — the stack's records and VisualGeometryToolkitCore: every
   RadiometricCamera consumer of a BlobBoards session's VisualGeometryToolkitIO records (three fixture views) against the
   rendered truth and against `test/integration_blobboards/golden/` (recorded from this release's code), and the
   VisualGeometryToolkitCore contract of the ray tracer. One GPU (`test/integration_blobboards/README.md`).
3. **`bash experiments/test/pipeline/run_pipeline_checks.sh`** — the refinement driver `bin/refine.jl` end to end on CPU on the published
   case data (`VGTK_DATA_ROOT`, the tier `rc_calibration`; Schöps front-end
   with the coarse grid, the board warping and the free radiometry), every numeric
   output against `experiments/test/pipeline/golden/` (recorded from this release's code) at 1e-8. About 15 minutes.
4. **`julia --project=experiments experiments/test/runtests.jl`** (a paper project is not a package: no `Pkg.test`) — the paper
   project's suite: its drivers plan, their evaluation
   and figure code reproduce the stored runs' fixtures, and, with the data root bound, the published data's checks. Its
   drivers' `evaluate` / `figures` stages regenerate the paper's figures and numbers from the runs (`experiments/REPRODUCE.md`).

The golden files of 1–3 are this release's state: a change that should not move the numbers succeeds when they are reproduced. Re-record them
(`record_golden.jl`, `--record`) only for an intended numerical change, and say why in the commit.
