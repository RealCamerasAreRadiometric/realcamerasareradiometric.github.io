# experiments

The experiments of the paper on RadiometricCamera: every figure, table and
number the manuscript (`../paper/`) reads, each
produced by a tracked driver under `scripts/` from a config of record under
`configs/`. It follows the stack's
[paper-project conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEval/stable/paper-projects/)
(a [DrWatson](https://juliadynamics.github.io/DrWatson.jl/) project, not a
package). The environment is this project's own (RadiometricCamera from
`../RadiometricCamera`, its `[sources]`): its Manifest is not tracked, and the stack's packages resolve from
the registries (the Visual Geometry Toolkit from General; the Visual Radiometry Toolkit from
[CauRegistry](https://github.com/prittjam/CauRegistry.jl), added once beside General). Until their next releases are
registered, PanoramaStitching and BlobBoards come from the branches its `[sources]` name
(each marked TEMPORARY there, with the release it waits for).

Run from the repository root:

```
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/scripts/radiometric_refinement.jl --config=experiments/configs/experiments/radiometric_refinement.toml --plan   # a driver's stage plan
julia --project=experiments experiments/scripts/radiometric_refinement.jl --config=experiments/configs/experiments/radiometric_refinement.toml          # run what is pending
julia --project=experiments experiments/test/runtests.jl                      # conventions + the project's tests
```

`--stage=NAME` runs one stage and `--force` runs stages again.
`scripts/calibrate.jl` (the calibration pipeline on one case of
`configs/calibration/`) is the skeleton every driver copies: a config of record
in, its stages, a directory keyed by the config's run key out. The experiment
drivers of the paper's experiments — calibration accuracy (`scripts/radiometric_refinement.jl` and `scripts/chrome_evaluation.jl` the
chrome board — `scripts/chrome_evaluation_paper.jl` on its paper config, `configs/experiments/chrome_evaluation_paper.toml`, writes the
chrome tables), the large-aperture refinements (`scripts/radiometric_refinement.jl` on
`configs/experiments/radiometric_refinement_vignetting.toml`; `scripts/focus_surfaces.jl` the focus surfaces) and defocus (the Tamron f/8 cases of `radiometric_refinement.toml`
scored on the chrome board; `scripts/texture_recovery.jl` the learned texture) — read
`configs/experiments/<name>[_<variant>].toml` (`--config`), run their stages (`--case=a,b` restricts the per-case ones) and
write their figures into `plots/` and their generated tables into `plots/latex/` (REPRODUCE.md).
`scripts/regenerate_paper.jl` runs the drivers of `configs/figures/numbers.toml` (the GPU refinements, the texture
sets and the sweeps' chains they read are run first: the one command, `scripts/reproduce/`), then gathers the number macros
(`scripts/numbers.jl`) and verifies and deploys the figures into `../paper/fig/` and the tables into `../paper/latex/`
(a submitted asset only with sign-off; `REPRODUCE.md`).
The RAW decoder is VisualGeometryToolkitFeatures' `load_raw` (LibRaw_jll), in Julia, for every
`[prepare.<set>]` mode: the one RAW path.

## Data

Inputs are read under `data_root()` (`data/exp_raw`, a link or directory per
machine, or the artifact view of the packed tiers declared in `Artifacts.toml` and
`configs/data_artifacts.toml`); results are written under `results_root()`
(`data/exp_pro`), in directories keyed by run key, each with `_config.toml` and
`_provenance.toml`. `data/` and `plots/` are not tracked. See `data/README.md`.
Every input is a tier of record of `configs/data_artifacts.toml` (the photographs, the external tools' outputs, the targets,
the golden set, the sensor model, the calibrator's image): `scripts/fetch_data.jl` installs them into the machine's artifact store
and links them into the artifact view, `scripts/make_artifacts.jl` packs them from an author's data root (`--dry-run` lists them).
The photographs are the author's published folder on the university's MDS share (`paper-data/` under `data_root()`; a config's
subset of a folder is stated beside it). The front ends that made the starts of record — Schöps's camera_calibration, OpenCV,
Kalibr, the parametric fits, run by hand on the cluster — are frozen as they ran in the tree `experiments/frontends_history/` of
tag `v0.10.0` (`git show v0.10.0:experiments/frontends_history/README.md`): the historical reference for the starts of record, run
by nothing. The project's own front ends replace them: `scripts/frontends.jl` (config `configs/experiments/frontends.toml`)
decodes the photographs through Features, runs the external calibrator in its pinned container and imports its run through IO.

## The layout

| Path | Holds |
|---|---|
| `configs/` | the configs of record (TOML): every value a result depends on |
| `scripts/` | the drivers: one per paper asset, stage-idempotent |
| `src/` | the shared prelude (`experiments.jl`), the drivers' machinery (`drivers.jl`), the evaluation (`evaluation.jl`), the figures and tables (`figures.jl`) and the board records (`boards.jl`), the texture recovery (`texture_recovery.jl`, its GPU operators `texture_gpu.jl`), the data tiers' rule (`data.jl`) and the paper's generated assets after the drivers (`assets.jl`: verify, deploy with sign-off, the number macros) |
| `test/` | the suite (ReTestItems: `runtests.jl` runs the `*_tests.jl` items in parallel; `paper_project_tests.jl` calls `paper_project_violations`) |
| `../paper/` | the manuscript; generated assets under `../paper/fig/` (figures) and `../paper/latex/` (tables, `numbers.tex`) |
| `data/`, `plots/` | the per-machine data bindings and local renderings (not tracked) |
| `REPRODUCE.md` | the asset index: every figure, table and macro the paper reads and its generator |
| `Artifacts.toml`, `configs/data_artifacts.toml` | the published input data and their catalog |

## Publishing and handover (the owner's steps)

1. Pack the tiers of record from your data root: `VGTK_DATA_ROOT=<data root> julia --project=experiments
   experiments/scripts/make_artifacts.jl` (`--dry-run` lists them; it refuses a tier holding an excluded photograph); the tarballs
   land in `--stage` (default `experiments/output/artifacts/tarballs`), `--check` lists each against the privacy rule again.
2. Publish each Zenodo record of the catalog with its tarballs (the catalog's `[records]`: four input records under 50 GB each, the
   tools record only if the container's base-image licence allows), after `make_artifacts.jl --check` on the staged tarballs.
3. Bind the records: `make_artifacts.jl --record=<record>=<Zenodo id>,…` rewrites `Artifacts.toml` (a URL per tarball); add each
   record's `id` to the catalog; commit both.
4. Check a fresh fetch: `julia --project=experiments experiments/scripts/fetch_data.jl` (the required entries; `--all` would also
   take the retired private tiers and the container image, which is never uploaded) on a clean machine, then the
   package's golden gate (`experiments/test/pipeline/` of the repository root).
5. Run the experiments on the GPU machine with the one command, `bash experiments/scripts/reproduce/reproduce_results_vcai.sh`
   (`--dry-run` first; REPRODUCE.md, One command): the GPU arrays, then `scripts/regenerate_paper.jl --gpu` at the arrays'
   thread count (the thread count keys every run) and the results branch.
6. Sign off the first deploy: set `replace_submitted = "on"` per figure or table config (and for `numbers.toml`), deploy, review
   the diff against the submitted assets (`REPRODUCE.md`).
