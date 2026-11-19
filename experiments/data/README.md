# data/

DrWatson's data directory, bound per machine (VisualGeometryToolkitEval Experiments):

| Tree | Resolved by | Access |
|---|---|---|
| `data_root()` — inputs | `VGTK_DATA_ROOT`, else `data/exp_raw` (a link or a directory), else the artifact view of the published data (`configs/data_artifacts.toml`) | read-only: a driver never writes here (`assert_not_input`) |
| `results_root()` — results | `VGTK_RESULTS_ROOT`, else `data/exp_pro` | add-only: one directory per run key, each with `_config.toml` and `_provenance.toml` |
| `datadir("sims")` | — | smoke runs, never read by a paper asset |

`data/exp_raw`, `data/exp_pro` and `data/sims` are ignored by git. Configs
name inputs relative to `data_root()`; no code names a machine path.

**The photographs** (tier `rc_raw_photos`) are the author's published folder on the university's MDS share,
bound under `data_root()` as `paper-data/` — a link to a mount of the share, or a copy checked with its
`MD5SUMS` (`DATA_LOCATIONS.md`). Every raw-photo input of a config is a path `paper-data/<experiment>/<lens>/<session>`.
**The derived inputs:** the baseline tier (`rc_baselines`: the front ends' outputs, the decoded chrome photographs, the flat
fields and image-circle frames) is on the share beside the photographs as `derived/rc_baselines_2026-11-10/` (`DATA_LOCATIONS.md`:
lay it over the data root; the front ends that made it are frozen in `frontends/` beside it and in the repository's history, `experiments/frontends_history/` of tag `v0.10.0` (the stack's own stage replaces them: scripts/frontends.jl)), and so are the targets the configs
read (`derived/rc_calibration_2026-11-10/`) and the synthetic set with its renders (`derived/rc_synthetic_2026-11-10/`).
`DATA_LOCATIONS.md` (this directory) says where each lives and how it is regenerated; `configs/data_artifacts.toml`
`[[unpacked]]` lists them.

## The ladder: where the inputs come from

1. **`VGTK_DATA_ROOT`** — a data root the author keeps (the source the tiers are
   packed from): `VGTK_DATA_ROOT=<data root> julia --project=experiments …`.
2. **`data/exp_raw` as a link** to such a directory — the same, without the
   variable (on a shared machine such as dgx01: `ln -s <the shared data
   directory> experiments/data/exp_raw`; the path is the machine's, never a
   config's).
3. **The artifact view** — on any other machine, `scripts/fetch_data.jl`
   downloads the published tiers (lazy Julia artifacts, `Artifacts.toml`;
   sha256 and tree hash verified) into the depot and links them file by file
   into `data/exp_raw` (marked `.artifact_view`). A binding of rung 1 or 2 wins
   and is never written to.

```
julia --project=experiments experiments/scripts/fetch_data.jl --list
julia --project=experiments experiments/scripts/fetch_data.jl --all
julia --project=experiments experiments/scripts/fetch_data.jl --only=rc_calibration_20261029      # the golden gate's cases alone
```

**The packed tiers of 2026-10-29** that `Artifacts.toml` binds are superseded by the share: the photographs as `paper-data/`
and the baseline tier as `derived/rc_baselines_2026-11-10/` (`DATA_LOCATIONS.md`). Until the final run's tiers are packed and
rebound, rung 3 fetches the old tiers only; a colleague builds a rung-1 data root from the share instead (the photographs
linked as `paper-data/`, the derived folder laid over it, both checked with their `MD5SUMS`):

```
VGTK_DATA_ROOT=<data root> julia --project=experiments …
```

## The tiers (configs/data_artifacts.toml)

| Tier | Holds | Read by |
|---|---|---|
| `rc_raw_photos_<date>` | the 248 raw photographs (ARW, CR2) and their manifest `rc_raw_photos.sha256` | the prepare stages (RAW decode) of radiometric_refinement, texture_recovery and the front ends (the panorama sweeps' frames stay in the tier, read by no config since rc-0116; the macro set's 14 CR2 stay in the tier, read by no config since 2026-11-14) |
| `rc_synthetic_<date>` | the detector-bias ground truth (`gt/`: true lens and poses, design reports, render states, linear photographs) and the render inputs | the golden calibration case (`configs/calibration/siemens_coarse_full.toml`: `fisheye_full`); the synthetic detector bias that made it left the project in rc-0116 |
| `rc_baselines_<date>` | the external detectors' and calibrations' outputs (Schöps, Kalibr, OpenCV, the parametric fits), the decoded chrome photographs, the flat fields and image-circle frames; the scripts that made them are `frontends/` beside the tier on the share = `experiments/frontends_history/` of tag `v0.10.0` (the 2026-10-29 tarball carries the retired experiments' tool scripts as `provenance/`) | the detect stages and their consumers |
| `rc_calibration_<date>` | the printed targets and every input of the three golden calibration cases | the calibration cases (`configs/calibration/`, the package's golden gate) and every driver's boards |
| `rc_thorlabs_<date>` | the Thorlabs R3L3S7N1 distortion target's record (17 × 17 dots at 5 mm, the maker's dimensions, `Rigid`) and 17 photographs of it | the experiments on the distortion target |

The layout of every tier is the layout of `data_root()` (the retired
`experiments/` tree's paths, without the `experiments/` prefix), so an
author's directory, a packed tier and a fetched view look alike. The author
packs the tiers with `scripts/make_artifacts.jl` (`--dry-run` lists them).
The data of record are published on the MDS share beside the photographs
(the baseline tier since 2026-11-10, the rest after the final run); how
`Artifacts.toml` binds the final tiers (it needs a URL per tarball;
`--record=rc=…` rebinds) is decided with the final run. The bindings it
carries today are the packed tiers of 2026-10-29 on the private dataset they
were kept on until then.

`scripts/py/fetch_raw_photos.sh` (with `scripts/py/raw_photos.sha256`) fetched
the raw photographs from that dataset; the published photographs are the
share's.
