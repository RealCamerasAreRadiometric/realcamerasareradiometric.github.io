# data/

DrWatson's data directory, bound per machine (VisualGeometryToolkitEval Experiments). Nothing here but this file is tracked:
inputs are the published tiers of `configs/data_artifacts.toml`, whose trees live in the artifact store; results are never tracked.

| Tree | Resolved by | Access |
|---|---|---|
| `data_root()` — inputs | `VGTK_DATA_ROOT`, else `data/exp_raw` (a link, a directory or the artifact view), else the site profile's `[storage] data_root` (`VGTK_SITE`), once it exists | read-only: a driver never writes here (`assert_not_input`) |
| `results_root()` — results | `VGTK_RESULTS_ROOT`, else `data/exp_pro`, else the site profile's `[storage] results_root` | add-only: one directory per run key, each with `_config.toml` and `_provenance.toml` |
| the artifact store | `VGTK_ARTIFACT_STORE`, else the site profile's `[storage] artifact_store` (VisualGeometryToolkitCore `artifact_store`; never a cluster's depot) | content-addressed trees, shared by every project on the machine |
| `datadir("sims")` | — | smoke runs, never read by a paper asset |

`data/exp_raw`, `data/exp_pro` and `data/sims` are ignored by git. Configs name inputs relative to their tier's root, the data
root; no config, code, test or document names a machine path (a machine's paths are its site profile's, VisualGeometryToolkitEval
`configs/sites/`).

## The inputs: the tiers of record

`configs/data_artifacts.toml` lists one entry per tarball (`<tarball>_<yyyymmdd>`, packed from the data root of the run of record):
the photographs of every session the configs read (`paper-data/<experiment>/<lens>/<session>`, the author's published folder on
the university's MDS share), the external tools' outputs the refinements start from and the frozen starts the front ends are
checked against, the printed targets, the golden calibration set, the sensor model and its frames, and the calibrator's container
image. Each entry's `tier` is the tier name the configs' inputs use (it keys their runs); `Artifacts.toml` binds each by tree hash
and sha256 to its Zenodo record (a placeholder until the record is published).

```
julia --project=experiments experiments/scripts/fetch_data.jl --list          # the catalog
julia --project=experiments experiments/scripts/fetch_data.jl                 # the required tiers
julia --project=experiments experiments/scripts/fetch_data.jl --only=rc_golden_20261121,rc_targets_20261122   # the golden gate's inputs
```

`scripts/fetch_data.jl` installs each tree into the artifact store (sha256 and tree hash verified; a tree the store already holds is
not downloaded again) and links it file by file into `data/exp_raw`, an artifact view. A binding of the data root (the variable,
the author's own `data/exp_raw`) wins and is never written to; on a site whose profile names an existing data root, that root
is the inputs' and nothing is fetched there (a view would outrank it). The author packs the tiers with
`scripts/make_artifacts.jl` (`--dry-run` lists them; `--check` lists every staged tarball against the privacy rule).

**Privacy.** The photographs DSC06045, DSC06052, DSC06053, DSC06054 and DSC06055 of the 14 mm Siemens-star session show another
person: they are in no tier and no record, under any name variant (`src/data.jl` `private_photographs`; the packer refuses them,
the suite checks every entry). The fisheye AprilGrid set of the board-warping session holds five other photographs with the same
file names (the camera's counter came round): those five paths, and nothing else, are published (`SAME_NAME_PHOTOGRAPHS`).
