# REPRODUCE — experiments

Every figure, table and number macro of the paper is indexed below with what makes it: a driver under `scripts/` on its config of
record under `configs/`, or — where the paper's file was made outside this project — its external generator, named as such. The
suite (`test/paper_project_tests.jl`: `paper_project_violations`, and `test/reproduce_tests.jl`: `reproduce_index_violations` with a
config of record required for every generated asset) checks the asset index (the last section) against what the paper reads.

The experiments are in three groups, each a directory of configs (owner 2026-11-18):

- **main** — `configs/experiments/`: the experiments of record whose products the main paper (or the main paper and the
  supplement) reads; their products under `results_root()/<experiment>/…`;
- **supplement** — `configs/supplement/`: the experiments of record whose assets appear in the supplement alone; products under
  `results_root()/supplement/<experiment>/…`;
- **studies** — `configs/studies/`: ideas tried that never came to fruition in a concrete result, each with its rationale and the
  decision of record it backed (`decided`); products under `results_root()/studies/<study>/…`. No launch runs them by default.

A group's location keys nothing (`src/drivers.jl` `product_experiment`, `variant_location`): moving a config between the groups
re-keys no product. The paper is `../paper` (the Blens manuscript with T. Harder's Overleaf update of 2026-11-18, its release build). Section,
figure and table numbers below are the paper's (main paper Sec. 1–6, supplement Sec. A.1–A.14).

## One command

Every result, figure, table and number of the paper, from the published inputs, on a machine of record:

```
bash experiments/scripts/reproduce/reproduce_results_dgx.sh --dry-run       # dgx01 (Slurm): the job graph, the missing inputs; nothing runs
bash experiments/scripts/reproduce/reproduce_results_dgx.sh                 # submit it: every stage an sbatch job, chained with afterok
bash experiments/scripts/reproduce/reproduce_results_dgx.sh status          # per stage: the jobs' states, the log, the first error line
bash experiments/scripts/reproduce/reproduce_results_desktop.sh --only=texture_recovery   # the desktop (one RTX 4090, no Slurm): here, in order
```

The stage graph of record is `configs/reproduce.toml` (which launcher runs which driver on which config of record, after what);
the runner is VisualGeometryToolkitEval's (Reproduce: `reproduce_main`, its docs' "One command: the runner"), and so are the
machine profiles (`dgx01`: one Slurm node, 8 × B200, partition `gpu`, one GPU per GPU task; `desktop`: one RTX 4090, no
scheduler, `CUDA_VISIBLE_DEVICES=0`). Each stage's CPUs and memory are the graph's (`cpus_per_task`, `mem_per_cpu` per stage).
Where this project's data and results are on each machine is `[sites.<name>]` of the graph — the one place the paths are written —
and `VGTK_DATA_ROOT` / `VGTK_RESULTS_ROOT` / `VGTK_SITE` or `--data-root` / `--results-root` override them.

- **Hierarchical.** No flags: every stage of record — setup, the front ends, the image regions and distance maps, the chrome
  refinements (decode first), the chrome evaluations, the texture sets, the f/4 refinements, then `regenerate_paper`
  (`scripts/regenerate_paper.jl --gpu --results-branch`) and the results branch. `--only=<stage>,…` runs those and what they run
  after; `--no-deps` those alone; the studies' stages (`sampling_strategy*`, `warping_ablation`), `chrome_index_grid`, `golden`,
  `convergence`, `detector_bias`, `timing`, the board warping across lenses (`board_warping_lenses` and the two `radiometric_refinement_board_warping*` stages it runs after), the panoramas' stages (`panorama_vignetting`, `panorama_stitching`) and `paper_pdf` run only when named.
- **Inputs first.** Every input of every selected stage's configs of record is checked before anything is submitted; the missing
  ones are printed as `path  tier  used by` and nothing runs (exit 3).
- **Safe to re-run.** Each launch writes a run manifest, `<results root>/_reproduce/<UTC stamp>-<commit>.toml`; a stage whose run
  key finished in an earlier launch is reused, one still in flight is depended on, and every driver reuses its own complete runs by
  their keys.
- **The results branch.** The last stage puts the regenerated figures, tables and `numbers.tex` on a branch
  `results/<stamp>-<commit>` of this repository, with `RUN.toml` and `COMPARISON.md`; nothing reaches `paper/` until it is accepted
  (VisualGeometryToolkitEval `accept_results`).

### The run of record

The paper's run of record runs on a registered release, in one command on dgx01 (`configs/run_of_record.toml`,
`scripts/run_of_record.jl`, `src/run_of_record.jl`): its `[launch] stages` are the MAIN and SUPPLEMENT groups' stages — never a
study's.

```
bash experiments/scripts/slurm/run_of_record.sh <version> --plan    # verify the registry, make the tree, print the job graph and its projected wall-clock
bash experiments/scripts/slurm/run_of_record.sh <version>           # the same, then every stage submitted
```

It refuses (nothing runs) unless RadiometricCamera `<version>` is in CauRegistry and resolves in a fresh environment from the
registries alone; it clones the repository at `v<version>` into `/raid/data/jpr/ror-<version>/tree`, instantiates it in a CPU job
and runs the runner on the stages of `[launch] stages` into the new results root `/raid/data/jpr/rc-results-<version>`. The
paper's results branch are those of 2026-11-17 (RC 0.11.5 with the configs of record): the reference every
product of a later run of record must reproduce.

### Adding your machine

Copy VisualGeometryToolkitEval's `configs/sites/_template.toml` to `<machine>.toml`, add `[sites.<machine>]` `data_root` and
`results_root` to `configs/reproduce.toml` (or set `VGTK_DATA_ROOT` and `VGTK_RESULTS_ROOT`), and run
`bash experiments/scripts/reproduce/reproduce.sh --site=path/to/<machine>.toml --dry-run`: the commands, the roots and the missing
inputs, with nothing run.

## The main paper

What each part of the submitted main paper shows and what makes it (`--plan` of a driver lists its stages; `--case=a,b` restricts
the per-case stages, `--stage=a,b` the stages). **Ours**: an asset of this project's drivers. **External**: the paper's file was
made by a generator outside this project (T. Harder's, in Blens or on his machine) from the products named; no driver stage of this
project reproduces it (a known limitation), and the asset index states it.

| paper location | label → asset | made by | config of record | the runs it reads |
|---|---|---|---|---|
| Sec. 1, Fig. 1 (teaser) | `fig:teaser` → `../paper/fig/teaser.tex` (+ `teaser/`) | **ours**: `scripts/teaser.jl` stage `teaser` (src/teaser.jl: the crops; drawn by src/paper_figures/teaser.jl, the port of T. Harder's teaser.py) | `configs/experiments/teaser.toml` (`[teaser]`: the part of the panorama `box`, its `gain`, the strip's `segments`) | the panoramas product of the Tamron 20 mm's sweep (`scripts/panorama_stitching.jl` on `configs/experiments/panorama_stitching_fx3_tamron.toml`: the panorama composed from the frames as they are and with the learned V divided out; its sweep's chain `scripts/panorama_vignetting.jl`, its V the refinement `f4_tamron_spline_warped`) and the tables of `fig:texture-learned` (the bands product of `scripts/texture_recovery.jl`: the strip's four bands) |
| Sec. 2, Tab. 1 | `tab:related` (in `../paper/sec/introduction.tex`) | hand-written | — | — |
| Sec. 3, Fig. 2 | `fig:model` → `../paper/latex/fig_model.tex` | hand-written TikZ | — | — |
| Sec. 4.1, Fig. 3 | `fig:boundary` → `../paper/latex/fig_boundary.tex` | hand-written TikZ | — | — |
| Sec. 5.1, Fig. 4 | `fig:detector-bias` → `../paper/fig/detector-bias-mitsuba.tex` (+ `detector-bias-mitsuba/`) | **ours**: `scripts/detector_bias.jl` stage `figures` (src/detector_bias.jl; drawn by src/paper_figures/detector_bias.jl, the port of T. Harder's detector_bias.py) | `configs/experiments/detector_bias.toml` (`[cases]`: db_ocv14_mitsuba_fov, db_spline_mitsuba_siemens_fisheye; `[evaluate]`; `[figure]`) | its own stages `start`, `refine` and `evaluate` per case — a detector-based geometric calibration, its radiometric refinement and both against the truth, on 200 of F. Seegräber's Mitsuba renders of the case's set — and, for the bias panels, the render pipeline's detections of all 10 000 renders of each set. On the evaluations of the paper's runs (RC 0.11.5, commit 16f7dcd9; 6,000 and 16,528 samples per photograph) the stage's tables are the submitted figure's byte for byte and its files the submitted ones in every line but the generator line and in every pixel (test/fixtures/paper_figures/detector-bias-mitsuba/README.md). A run of this project is another run — 8,192 and 16,384 samples per photograph (`[kinds]` of `radiometric_refinement.toml`), RadiometricCamera 0.2 — and has not been made: the inputs are not in the data root of record yet (`data/DATA_LOCATIONS.md`) |
| Sec. 5.2, Tab. 2 | `tab:refinement-chrome` → `../paper/fig/refinement-table-ci.tex` | **external**: `refinement-table/tools/refinement_table.py` (T. Harder) from the chrome evaluations of RC 0.11.5 | ours beneath it: `configs/experiments/chrome_evaluation_paper.toml` (`[tables.refinement_chrome]`, the same numbers, rendered into `plots/latex/table_refinement_chrome.tex`) | the refinements (`scripts/radiometric_refinement.jl`, `configs/experiments/radiometric_refinement.toml`; GPU) and their chrome scores (`scripts/chrome_evaluation.jl`, `configs/experiments/chrome_evaluation_fx3_*.toml`) |
| Sec. 5.2, Fig. 5 | `fig:convergence-rms` → `../paper/fig/convergence-rms.tex` (+ `convergence-rms/`) | **ours**: `scripts/convergence.jl` stage `panels` (src/convergence_panels.jl, T. Harder's rms_figure.py style) | `configs/experiments/convergence.toml` (`[panels]`: si30_spline_warped left, ag30_spline_warped right) | per panel its refinement of record (25 iterations, a checkpoint each) and its chrome set's index; the detections' true reprojection median and the held-out Thorlabs median per iteration; the sentence's numbers as macros `\convergence<Siemens|Aprilgrid><Detections|Heldout><Start|Rise|End>` (not yet cited: the prose types 0.59/0.71/0.56/0.68) |
| Sec. 5.3, Fig. 6 | `fig:vignetting` → `../paper/fig/vignetting.tex` (+ `vignetting/`) | **ours**: `scripts/vignetting_figure.jl` stage `figures` (src/vignetting_figure.jl; drawn by src/paper_figures/vignetting.jl, the port of T. Harder's vignetting.py) | `configs/experiments/vignetting_figure.toml` (`[lenses]`: the Sigma 14 mm, the fisheye, the Tamron 20 mm; `[diagonal]`) | per lens the camera record of its f/4 refinement (`configs/experiments/radiometric_refinement_vignetting.toml`: the learned vignetting, its centre, the lens coverage), the case's nominal focal length (its `camera.f_mm`: the thin lens model's) and its diffuser photograph, read at its RAW green photosites under the camera's sensor record. Its figure is not the submitted one, on purpose: the submitted files read the flat field from a decoded copy of the same photographs, end the 14 mm's and the fisheye's diagonals at an image circle of an earlier run and draw the Tamron's cos⁴α with 2484.1 px; on the refinements of 2026-11-16 the stage's flat field differs from theirs by at most 8.4e-4, those two diagonals are 78 px and 36 px longer (the lens coverage less 15 px) and the Tamron's cos⁴α (21.1975 mm, 2534.2 px) differs by at most 0.012 (test/fixtures/paper_figures/vignetting/README.md) |
| Sec. 5.4, Fig. 7 (left) | `fig:focus-surfaces` → `../paper/fig/focus-surfaces.tex` (+ `focus-surfaces/`) | **ours**: `scripts/focus_surfaces.jl` stage `panels` (src/paper_figures/focus_surfaces.jl, the port of T. Harder's focus_surfaces.py); its stage `figures` draws an earlier PNG into `plots/` (`focus-surfaces.png`; not in the paper) | `configs/experiments/focus_surfaces.toml` (`[panels]`: f4_fisheye_spline_warped left, sitla_spline_warped right, a lattice sample every 16 px) | the refined cameras (`radiometric_camera.json`) of `f4_fisheye_spline_warped` (`radiometric_refinement_vignetting.toml`) and `sitla_spline_warped` (`radiometric_refinement.toml`): every pixel's in-focus point on the lattice |
| Sec. 5.4, Fig. 7 (right) | `fig:texture-learned` → `../paper/fig/texture-learned.tex` (+ `texture-learned/`) | **ours**: `scripts/texture_recovery.jl` stage `bands` (src/paper_figures/texture_learned.jl, the port of T. Harder's texture_learned.py), but for the band "training photograph": an input of record, cut out of a RAW photograph by an author-side step (`tools/texture_learned_photograph.py`: rawpy, OpenCV; `configs/inputs/texture_learned_photograph_20261121.toml`, tier `rc_baselines_2026-11-21`). The driver's stage `figures` also draws an earlier PNG into `plots/` (`texture-learned.png`; not in the paper) | `configs/experiments/texture_recovery_14mm_small.toml` (`[bands]`: truth, photograph, pinhole, ours) | its own stages `evaluate` (the design and the two textures), `init` (the poses the training photograph is chosen by) and `refine` (the photographs' roles), and the camera of `texcal_14mm_spline_warped` |
| Sec. 5.4, Tab. 3 | `tab:defocus` → `../paper/fig/defocus-table-ci.tex` | **external**: `defocus-table/tools/defocus_table.py` (T. Harder) from the chrome evaluations of RC 0.11.5 | ours beneath it: `chrome_evaluation_paper.toml` (`[tables.defocus]`) | `sitla_spline_warped`, `sitla_spline_pinhole_warped`, `sizl_bc5_warped`, `sizl_bc5_pinhole_warped` and their chrome scores |
| the prose's numbers | `../paper/latex/numbers.tex` | **ours**: `scripts/numbers.jl` (via `scripts/regenerate_paper.jl`) | `configs/figures/numbers.toml` | the drivers of its `[runs] experiments` |

## The supplement

| paper location | label → asset | made by | config of record | the runs it reads |
|---|---|---|---|---|
| Sec. A.11, Tab. A1 | `tab:dataset-lenses` (in `X_suppl.tex`) | hand-written | — | — |
| Sec. A.11, Tab. A2 | `tab:supp-dataset` (in `X_suppl.tex`) | hand-written from the configs of record | — | the capture sets of `radiometric_refinement.toml` and the chrome sets |
| Sec. A.14 Runtime, `tab:supp-runtime` (Tab. A6; promised in Sec. 5: "together with the runtime for each experiment") | `tab:supp-runtime` → `../paper/fig/runtime-table.tex` (from `plots/timing/runtime-table.tex`); its statement (`timing_statement.tex`, the text of A.11.1) and the `\timing…` macros of `../paper/latex/numbers.tex`; of record: product `f28d6ccaa124db2b`, 703.5 ± 3.7 s over 8 runs (2026-11-18) | **ours**: `scripts/timing.jl` (stages `start`, `time`, `record`) | `configs/supplement/timing.toml` (case `si30_spline_warped`, 8 replicates one per B200, a warm-up each) | the case's start (`radiometric_refinement.toml`, untimed) and bin/refine.jl `--repeat=2` per replicate: the timed span is `refine_calibration!` + `save_radiometric_calibration` |
| Sec. A.12 | the geometric starts (text) | the front ends (`scripts/frontends.jl`, `configs/experiments/frontends.toml`) and the starts of other systems, inputs of record (tiers `rc_baselines_2026-11-10`, `rc_baselines_2026-11-17`: the OpenCV step `configs/inputs/opencv_starts_20261117.toml`) | — | — |
| Sec. A.13.1, Tab. A3, Fig. A1 | `tab:supp-thorlabs-target` (in `X_suppl.tex`), `fig:supp-thorlabs-target` → `../paper/fig/thorlabs-target.jpg` | hand-written; a photograph | — | — |
| Sec. A.13.2, Tab. A4, A5 | `tab:supp-refinement-a/b` → `../paper/fig/refinement-table-supp-a.tex`, `-b.tex` | **external**: `refinement-table/tools/refinement_table.py` (T. Harder) | ours beneath it: `chrome_evaluation_paper.toml` (`chrome_all`) | every model on every chrome set |
| — (the boards' heights from the chrome table's capture sets; not in the paper, not in `../paper`) | the boards' heights → `../paper/fig/board-warping.tex` (+ `board-warping/`: six panels and the colour key; from `plots/board-warping.tex`, `plots/board-warping/`) | **ours**: `scripts/board_warping.jl` (stage `figures`; the drawing code `src/paper_figures/board_warping.jl`, the port of T. Harder's `board-warping/tools/board_warping.py`) | `configs/supplement/board_warping.toml` (six capture sets, each with its spline case; drawn as 3-D surfaces) | the refine products of the six cases `<set>_spline_warped` of `radiometric_refinement.toml` (each one's `radiometric_target.json`: the page and the warping's nodal heights); the surface between the nodes is VisualRadiometryToolkitCore's `board_height` |
| Supp., the board-warping section (in T. Harder's manuscript after the panoramas; not yet in `../paper`) | `fig:board-warping` → `../paper/fig/board-warping-lenses.tex` (+ `board-warping-lenses/`: six surfaces, six maps with their images, the surfaces' key and the maps' key; from `plots/board-warping-lenses.tex`, `plots/board-warping-lenses/`) | **ours**: `scripts/board_warping.jl` on its second config (stage `figures`; the drawing code `src/paper_figures/board_warping_offsets.jl`, the port of T. Harder's `board-warping-lenses-combined-offsets/tools/board_warping_offsets.py`, its variant b6 with large surfaces, the surfaces first and maps of 0.40 of a surface's width; the same session's heights alone are the drawing `src/paper_figures/board_warping_lenses.jl`, which no config of record names) | `configs/supplement/board_warping_lenses.toml` (the session's six sets — two boards × three lenses, each board in place — each with its spline case; per board three surfaces, the planar board plus the board's offsets learned with each lens drawn 50 times enlarged, and three maps of the length of the difference between the offsets learned with two lenses; the stage's `meta.toml` states that difference per board and pair: its RMS, its parts in the plane and along the normal, its largest value) | the refine products of the six cases `bw_<board>_<lens>_spline_warped` (`ag`, `si`; `14mm`, `fisheye`, `tamron`) of `configs/experiments/radiometric_refinement_board_warping.toml` (each one's `radiometric_target.json`: the page, the warping's nodal heights, the in-plane aspect and skew); their starts: the front ends' Schöps runs of the three Siemens sets (`frontends.toml` `[schoeps.bw_<lens>_siemens_w<window>]`) and, for the AprilGrid sets, inputs of record of tier `rc_baselines` (not yet on the share: `data/DATA_LOCATIONS.md`). The paper's files were drawn from the RC 0.11.5 refinements of 2026-11-20 (the fisheye AprilGrid set with 186 views; this config's case has 181) |
| Supp., the panoramas (in T. Harder's manuscript; not yet in `../paper`) | the panoramas figure → `../paper/fig/panoramas-james.tex` (+ `panoramas-james/`: six panels; from `plots/panoramas-james.tex`, `plots/panoramas-james/`) | **ours**: `scripts/panorama_figure.jl` (stage `figures`; the drawing code `src/paper_figures/panoramas.jl`, the port of T. Harder's `panorama-james/tools/panorama_pgf.py`) | `configs/supplement/panorama_figure.toml` (the three lenses, each with its panorama_stitching config; the display curve, the panels' width) | per lens the panoramas product of `scripts/panorama_stitching.jl` on `configs/experiments/panorama_stitching_fx3_<lens>.toml` (14mm, fisheye, tamron): the frames of its pure-rotation sweep (`scripts/panorama_vignetting.jl` on `panorama_vignetting_fx3_<lens>.toml`: PanoramaStitching's chain) composed as they are and divided by the V of its f/4 refinement (`f4_<lens>_spline_warped` of `radiometric_refinement_vignetting.toml`) |

```
julia --project=experiments experiments/scripts/timing.jl --plan                                          # the timing's stages
julia --project=experiments experiments/scripts/timing.jl --stage=start                                   # the case's start (once)
julia --project=experiments experiments/scripts/timing.jl --stage=time --replicate=<k>                    # one replicate per GPU job
julia --project=experiments experiments/scripts/timing.jl --stage=record                                  # the record, the tables, the macros
julia --project=experiments experiments/scripts/board_warping.jl --plan                                   # the board warping's stage
julia --project=experiments experiments/scripts/board_warping.jl                                          # the heights, their table, the figure (CPU; the refinements must exist)
julia --project=experiments experiments/scripts/board_warping.jl --config=experiments/configs/supplement/board_warping_lenses.toml --plan   # the paper's board-warping figure (the three-lens session): its stage
julia --project=experiments experiments/scripts/radiometric_refinement.jl --config=experiments/configs/experiments/radiometric_refinement_board_warping.toml --plan   # its six refinements (GPU; the Siemens sets' starts: scripts/frontends.jl)
bash experiments/scripts/reproduce/reproduce_results_dgx.sh --only=board_warping_lenses --dry-run            # the whole chain as jobs: the front ends, the six refinements, the figure
julia --project=experiments experiments/scripts/panorama_vignetting.jl --config=experiments/configs/experiments/panorama_vignetting_fx3_<lens>.toml   # per lens (14mm, fisheye, tamron) its sweep's chain (one GPU)
julia --project=experiments experiments/scripts/panorama_stitching.jl --config=experiments/configs/experiments/panorama_stitching_fx3_<lens>.toml     # per lens the learned V's export and the panoramas (a GPU job; the f/4 refinement must exist)
julia --project=experiments experiments/scripts/panorama_figure.jl                                        # the three lenses' panoramas as the figure (CPU; the panoramas must exist)
```

## The studies

Each study's config states its question, hypothesis, what it decides, its evidence and the decision of record it backed
(`decided`); `test/studies_tests.jl` requires both blocks. None is run by the run of record.

| study | config(s) | driver | ran? (results) | decided |
|---|---|---|---|---|
| the LM sampling strategy | `configs/studies/sampling_strategy.toml` (+ `_smoke`) | `scripts/sampling_strategy.jl` | yes: `rc-results-0.11/sampling_strategy/record/952231f4538a4b78` | the uniform draw (α = 0) for every case of record (owner 2026-11-16) |
| its sample-count side studies | `configs/studies/sampling_strategy_kb_count.toml`, `_fe10_count.toml`, `_fe10_kb16384.toml` | `scripts/sampling_strategy.jl` | yes (rc-results-0.11.5): KB 14 mm 16,384 vs 6,000 Δ −0.0006 [−0.0015, +0.0003] px n.s.; fisheye 10 cm at 32,768 n.s. but the Brown–Conrady (the wrong model); fisheye KB 16,384 Δ +0.0004 [−0.0024, +0.0042] px n.s. | 8,192 samples per photograph for every global model, 16,384 for every spline (owner 2026-11-16 23:30; `radiometric_refinement.toml [kinds]`) |
| the lens-coverage margin | `configs/studies/radiometric_refinement_ablation_lens_coverage_scale.toml` | `scripts/radiometric_refinement.jl` | no (as a whole) | `masks.lens_coverage_scale` 1.0 (owner 2026-11-15) |
| the board warping per model family | `configs/studies/warping_ablation.toml` | `scripts/warping_ablation.jl` | no | heights only for the global models, the heights and the in-plane aspect/skew for the spline (owner 2026-11-16) |
| the fisheye 30 cm spline's convergence and smoothness | `configs/studies/radiometric_refinement_smoothness_si30fe.toml` | `scripts/radiometric_refinement.jl` (the per-iteration Thorlabs curves: branch `exp-convergence-si30fe`, root `rc-results-exp-conv`) | yes, on the branch | the smoothness weight 0.1 kept (it does not bind); the held-out RMS rise comes from ≈ 23 dots at the top rim of the image circle |

```
julia --project=experiments experiments/scripts/sampling_strategy.jl --config=experiments/configs/studies/sampling_strategy.toml --plan
julia --project=experiments experiments/scripts/radiometric_refinement.jl --config=experiments/configs/studies/radiometric_refinement_ablation_lens_coverage_scale.toml --plan
julia --project=experiments experiments/scripts/warping_ablation.jl --plan
julia --project=experiments experiments/scripts/radiometric_refinement.jl --config=experiments/configs/studies/radiometric_refinement_smoothness_si30fe.toml --plan
```

**The stochastic LM** (not moved in: it changes `src/` of the package). Branch `exp-stochastic-lm`, results root
`/raid/data/jpr/rc-results-exp-slm` on dgx01: the LM's samples redrawn every iteration (16,384 per photograph, λ carried across the
redraws) on `si30_spline_warped` against its run of record (7a66fb15e2329e87): held-out Thorlabs median Δ −0.0003 [−0.0055, +0.0039]
px, n.s.; RMS −0.0099 [−0.019, −0.003] px, from the rim; 2.3× the time. 4,096 per photograph per iteration is worse on the median.
Decided: the fixed draw of record stays.

**Experiments as overrides.** An experiment or study is a config of record plus keyword overrides (`[experiment] base`, the cases it
runs and `[overrides]` `table.key = [values…]`: DrWatson's `dict_list`, `expand_overrides` in `src/drivers.jl`); a variant whose
values are the case's own IS that case (its run of record reused, by its key), any other is `<case>__<table.key>=<value>[…]`, keyed
by its own options and located in its group's directory.

## By hand

```
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'           # from the repository root; the Manifest is the root's, untracked
julia --project=experiments experiments/scripts/regenerate_paper.jl --plan  # the regeneration's stages
julia --project=experiments experiments/scripts/regenerate_paper.jl --gpu   # the listed drivers' pending stages, the numbers, verify, deploy (make -C paper figures-gpu)
julia --project=experiments experiments/scripts/<driver>.jl --plan          # one driver: what is done, what is pending
```

`scripts/regenerate_paper.jl` (config `configs/figures/regenerate_paper.toml`)
runs, in order: the stages of every experiment driver of
`configs/figures/numbers.toml`'s `[runs] experiments` (the one list of drivers,
in dependency order), each in a module of its own (VisualGeometryToolkitEval's
`run_isolated`), reusing every complete run — the refinements those drivers read are not in
the list: the Slurm launchers run them first (Running on Slurm, below), and a driver that misses one stops and names the
command; `numbers` (`scripts/numbers.jl`: the drivers' `numbers.tex`
fragments gathered into `../paper/latex/numbers.tex`); `verify` (each render
against the submitted copy: the figures of `plots/` against `../paper/fig/` →
`plots/verify.tsv`, the generated tables of `plots/latex/` against
`../paper/latex/` → `plots/latex/verify.tsv`); `deploy` (each render into
`../paper/fig/` or `../paper/latex/` when identical or new). Verify and
deploy are VisualGeometryToolkitEval's (`verify_assets`, `deploy_assets`), one
call per kind of asset. A figure config of record (`configs/figures/<asset>.toml`;
none at present — the paper's figures and tables are external, `numbers.tex` the one generated file the
regeneration deploys; `test/fixtures/figure_configs/` holds examples)
names its kind by `[render] asset`: `"figures.<name>"` a PNG the driver's
`[figures.<name>]` table draws, `"tables.<name>"` a LaTeX table its
`[tables.<name>]` table writes (`src/assets.jl`).

**A submitted asset — a figure, a generated table or `numbers.tex` — is never
replaced silently**: a differing render is deployed only once its figure config
says `[deploy] replace_submitted = "on"`, and differing numbers only once
`configs/figures/numbers.toml` says so — the author's sign-off of the difference
the verify report states (for the numbers, the `verify.tsv` of the numbers run);
otherwise deploy refuses and names the report. Every macro `numbers.tex` defines
must be cited by a tex file the paper reads (`uncited_macros`,
test/reproduce_tests.jl); a number of a generated table is never re-typed as a
macro — the tables carry their numbers themselves.

**The GPU.** Each driver declares its GPU stages (`GPU_STAGES`: the stages
whose `[pipeline]` tables bin/refine.jl runs on `compute.backend = "gpu"`, the
package's default, and the texture page solve). Without `--gpu` they are not run: a complete one is reused,
a pending one stops the regeneration and names `--gpu`. With `--gpu` the
regeneration is refused before anything runs unless GPU 0 (nvidia-smi's index)
has `[gpu] min_free_gib` (20 GiB) free.

**The thread count keys every run** (`src/drivers.jl`): the runs of record were
made with 16 threads (`JULIA_NUM_THREADS=16`, `--cpus-per-task=16` on every launcher),
so the drivers and the regeneration are run at the same count (`julia -t 16`), or
they re-run under new keys.

**The manuscript's paths.** The paper is built from `paper/` with `src/` on
TEXINPUTS, so `main.tex` reads `src/sec/...`, `src/fig/...` and
`src/latex/...` from there and `preamble`, `sec/0_abstract_new`, `sec/X_suppl` from
`src/`; the reproduce checks resolve the same way (`search_dirs = ["paper"]`,
`PAPER_SEARCH_DIRS` of `src/assets.jl`). The manuscript text is the co-author's
and is not rewritten for the checks.

**Two opt-outs of the paper-project rules**, both on `scripts/regenerate_paper.jl`
and stated there (`# paper-project: allow`): **P1** (a producing script writes
provenance) — it is an orchestrator: it writes no result of its own; every
result is a driver's keyed run with its `_config.toml` and `_provenance.toml`,
and it only copies their renders into the paper; **P2** (a producing script
keys its outputs by a run key) — it keys no output of its own: the drivers key
their runs, and the paper reads fixed paths.


## Data

Inputs are read from `data_root()` (read-only: `VGTK_DATA_ROOT`, else
`data/exp_raw`, else the artifact view of the published data, see
`configs/data_artifacts.toml`); results are written under `results_root()`
(`VGTK_RESULTS_ROOT`, else `data/exp_pro`), one directory per run key, each
with its `_config.toml` and `_provenance.toml`.

Every input of a config of record names its catalog tier — `rc_raw_photos`
(the RAW photographs: the author's published folder on the university's MDS share, bound under `data_root()` as
`paper-data/`, see `data/DATA_LOCATIONS.md`; a config's subset of a folder is stated beside its path, `frames` or `exclude`),
`rc_synthetic` (the synthetic renders and their truth),
`rc_baselines` (the outputs of the external detectors and calibrations:
Schöps, Kalibr, OpenCV, BlobBoards, the parametric fits; the decoded chrome photographs, the flat fields, image-circle frames and
exported vignetting maps), `rc_calibration` (the calibration targets and cases) and `rc_thorlabs` (the
Thorlabs R3L3S7N1 distortion target and its photographs) — and its path under
`data_root()`; each config states how its inputs were made. The tiers are
catalogued in `configs/data_artifacts.toml`. The photographs and the baseline tier of record live on the MDS share
(`data/DATA_LOCATIONS.md`: the photographs bound as `paper-data/`, the front ends' outputs and the decoded chrome photographs as
`derived/rc_baselines_2026-11-10/`, laid over the data root); the packed tiers of 2026-10-29 that `Artifacts.toml` binds are
superseded by them and are rebound when the final run's tiers are packed. The packed tiers, on a new machine, fetch
into the artifact view `data/exp_raw` (`data/README.md`):

```
julia --project=experiments experiments/scripts/fetch_data.jl --all            # or --only=rc_calibration_20261029,…; --list
julia --project=experiments experiments/scripts/regenerate_paper.jl --gpu
```

An author's own copy is bound instead with `VGTK_DATA_ROOT=<data root>` (or a
`data/exp_raw` link); `scripts/make_artifacts.jl` packs the tiers from it.

**The derived inputs.** Most inputs of the experiments below are newer than the packed tiers of 2026-10-29: each is an
`[[unpacked.<tier>]]` entry of `configs/data_artifacts.toml` (its paths, the configs that read it, where it lies, how it was
made). The baseline tier (`rc_baselines`: the front ends' outputs, the decoded chrome photographs, the flat fields and image-circle
frames) is on the share since 2026-11-10 (`derived/rc_baselines_2026-11-10/`, `data/DATA_LOCATIONS.md`), and so are the targets the
configs read (`derived/rc_calibration_2026-11-10/`) and the synthetic set with its renders (`derived/rc_synthetic_2026-11-10/`): a
data root built from the share's folders holds every input of the configs of record. A fetch of the packed tiers alone does not.


**Inputs to re-derive after the release (the deleted rawpy decoder's).** The decoded photographs of these configs are inputs of
record made by `scripts/py/convert_raw.py --mode fullres` (rawpy's demosaic), which is deleted: radiometric work reads the green
cells of the Bayer cells (Features' `:green_cells`, `[prepare.<set>] mode = "green"`), never a demosaic. Each is re-derived from
its declared RAW input through Features after the release; until then the configs read the inputs of record as they are:
- `configs/experiments/chrome_evaluation_fx3_14mm_central30cm.toml` and `chrome_evaluation_fx3_tamron_17cm_si.toml`
  (`[inputs.photos]`, the decoded chrome photographs);


**The lens coverage.** Every capture set's lens coverage is one keyed step (`src/image_region.jl`, owner ruling 2026-11-15): each
config of record states it per set, `lens_coverage = "sensor"` (the lens lights the whole sensor; nothing measured) or
`"image_region:<name>"`, its `[image_regions.<name>]` the set's own frames (each pinned by its sha256) and the detector's settings.
`scripts/image_regions.jl` (`configs/experiments/image_regions.toml`, stage `image_regions` of the stage graph) measures them —
RadiometricCamera's `measure_image_region`, Features' `detect_image_circle` on the device — into `image_region/detect/<key>/image_region.json`
(rc/image_region/v1.0); the refinements, the front ends and the chrome evaluations only read the records.

**The distance maps.** The printed boards' edge distance maps (RadiometricCamera's `edge.sdf`) are derived files, never inputs of record (owner 2026-11-16): `scripts/distance_maps.jl`
(`configs/experiments/distance_maps.toml`, stage `distance_maps` of the stage graph) converts each board generator's `sdf.f32` + `sdf.json`
(inputs of record, pinned by their sha256) into a VisualGeometryToolkitIO raster record, `distance_maps/prepare/<key>/distance_map.json`
(+ `distance_map.tiff`), keyed by those two inputs and the code of Core and IO. Every consumer names a map as a `[runs.<id>]` of that
config (`edge.sdf = "run:<id>"`) and produces it on demand when missing; the stage graph runs it once before the case arrays.


### The records

Every record under `data_root()` the configs of record read is in VisualGeometryToolkitIO's current schemas (`vgio/cameras/v4`,
`vgio/rig/v6`, `vgio/target/v3`); the board records (`geometric_accuracy/data/target/<board>/board`: `target.json`, `texture.png`,
the design) are checked by `test/boards_tests.jl` against the package's designs. No Serialization file is an input of record.

## Running on Slurm

The launchers in `scripts/slurm/` run the paper experiments' drivers on a Slurm cluster, each on its config of record, in
parallel: one GPU per case of the chrome-board refinements and the f/4 refinements and per texture set. Their resources are those of dgx01 (one node: 224 CPUs, 1900 GB, 8 GPUs, partition
`gpu`; memory per CPU at the node's ratio, `--mem-per-cpu=8G`). The one command (above) submits them all
(`reproduce_results_dgx.sh`: `setup.sbatch` — instantiate the workspace and precompile experiments and the package, CUDA and the GPU
extensions included — and then the others with `--dependency=afterok` on what they read, printing the job ids); each is also
submitted by hand from the repository root (`sbatch experiments/scripts/slurm/<launcher>.sbatch`, an array with `--array=0-<n-1>`; `case_task.sbatch` with `EXPERIMENT`,
`STAGES` and `--job-name` as configs/reproduce.toml states them).
The environment binds the trees (`scripts/slurm/common.sh`): inputs from `VGTK_DATA_ROOT`, results into `VGTK_RESULTS_ROOT` (required:
the runner sets it from the site, a launcher submitted by hand names it; never a directory named by a date — a run is found by its
key, so every launch shares the one results root); no launcher gives `--force`, and a results root equal to `experiments/data/exp_pro` is refused. Logs go to `./logs/`.

| experiment | driver | config of record | launcher | resources | runs |
|---|---|---|---|---|---|
| setup | — | — | `scripts/slurm/setup.sbatch` | 1 GPU, 16 CPUs, 2 h | `Pkg.instantiate`, `Pkg.precompile` of `experiments` and the package; `CUDA.functional()` |
| decode (per experiment with a `prepare` stage) | the experiment's driver | its config of record | `scripts/slurm/prepare.sbatch` (`EXPERIMENT`) | CPU only, 8 CPUs (the case tasks' thread count, which keys the prepare run they read), 4 h | stages `prepare` (the raw decode, Features' `load_raw`) and `detect`, once, before the experiment's array |
| the refinements (Sec. 5.2, 5.4) | `scripts/radiometric_refinement.jl` | `configs/experiments/radiometric_refinement.toml` | `scripts/slurm/case_task.sbatch` (`EXPERIMENT=radiometric_refinement`, `STAGES=refine`), after `prepare.sbatch` (`EXPERIMENT=radiometric_refinement`) | an array, one task per case: 1 GPU, 28 CPUs; `--time` = the largest `max_time` + 3 h | stage `refine` of task i's case; the cases are `CHROME_CASES`, else every case. The chrome scores (`scripts/chrome_evaluation.jl`, detect on the GPU) run after these, as the `sets` stage of `scripts/chrome_evaluation.jl` on `configs/experiments/chrome_evaluation_paper.toml` in `scripts/regenerate_paper.jl` |
| the texture sets (Sec. 5.4) | `scripts/texture_recovery.jl` | `configs/experiments/texture_recovery_<set>.toml` | `scripts/slurm/texture_recovery.sbatch`, after the refinement array (its case `texcal_14mm_spline_warped` is the camera held fixed) | an array, one task per set (`TEXTURE_SETS`, default `14mm_small`): 1 GPU, 28 CPUs, 4 h | stages `prepare`, `detect`, `sheet`, `init`, `refine`, `folds`, `evaluate`, `fold_report`, `compare` and `report` (summary.md) of task i's set; `figures` and `bands` are `scripts/regenerate_paper.jl`'s (`bands` reads the photograph band of tier `rc_baselines_2026-11-21`, which must lie in the data root) |
| study: the sampling strategy | `scripts/sampling_strategy.jl` | `configs/studies/sampling_strategy.toml` | `scripts/slurm/sampling_strategy.sbatch`: `STAGES=configs,prepare` once, then the arms (an array of packs, `ARM_PACKS`), then `STAGES=evaluate,texture,compare,record,figure` | the prepare 1 GPU, 8 CPUs, 1 h; a pack 1 GPU, 24 CPUs (3 arms × 8 threads), 8 h; the score 1 GPU, 8 CPUs, 4 h | the variants' starts, the lattice guard, the refinements; the chrome evaluations, the texture runs and their compare; the record, the figure and the table |
| the f/4 refinements (Sec. 5.3, 5.4) | `scripts/radiometric_refinement.jl` | `configs/experiments/radiometric_refinement_vignetting.toml` | `scripts/slurm/case_task.sbatch` (`EXPERIMENT=radiometric_refinement_vignetting`, `STAGES=refine`), after `prepare.sbatch` (`EXPERIMENT=radiometric_refinement_vignetting`) | an array, one task per case: 1 GPU, 28 CPUs; `--time` = the largest `max_time` + 3 h | stage `refine` of task i's case; the cases are `VIG_CASES`, else every case. The focus surfaces read `f4_fisheye_spline_warped` |
| the regeneration | `scripts/regenerate_paper.jl --gpu` | `configs/figures/regenerate_paper.toml` | `scripts/slurm/regenerate_paper.sbatch`, once the arrays have ended | 1 GPU, 28 CPUs (the arrays' thread count), 224 GB, 24 h | the listed drivers' pending stages, numbers, verify, deploy (`REGENERATE_STAGES` names a subset; deploy refuses a differing asset without sign-off); in the one command `--results-branch` (`REGENERATE_RESULTS_BRANCH`): nothing deployed, the renders go to the results branch |
| study: the board warping ablation | `scripts/warping_ablation.jl` | `configs/studies/warping_ablation.toml` | `scripts/slurm/warping_ablation.sbatch`; only with `--only=warping_ablation` | three phases: `upstream` (1 GPU, 8 CPUs, one task), `refine` (an array, one task per mode, `WARP_MODES` else every mode: 1 GPU, 8 CPUs, its refinements in turn), `score` (CPU, 8 CPUs, one task) | stages `upstream`; `refine --mode=<mode>`; `evaluate,record,figure` |
| supplement: the timing (Sec. A.14) | `scripts/timing.jl` | `configs/supplement/timing.toml` | `sbatch` by hand (no stage of the graph yet): `--stage=start` once (1 GPU), then one job per replicate `--stage=time --replicate=<k>` (1 GPU each, 8 CPUs, `JULIA_NUM_THREADS=8,1`, all at once, one per GPU), then `--stage=record` (CPU) | 1 GPU per replicate, 8 CPUs, 2 h | the start; per replicate a warm-up and a timed refinement + write; the record, the runtime table, the macros |
| supplement: the board warping | `scripts/board_warping.jl` | `configs/supplement/board_warping.toml` | the regeneration (its config is one of `configs/figures/numbers.toml` `[runs] experiments`): `scripts/slurm/regenerate_paper.sbatch`, after the refinement array of `radiometric_refinement.toml`; by hand the commands above | CPU only: the regeneration's job (about 90 s of it) | stage `figures`: the six cases' height fields (201 × 201 samples), each page's size and surface's range (`meta.toml`), the figure into `plots/` |
| supplement: the board warping across lenses | `scripts/radiometric_refinement.jl`, then `scripts/board_warping.jl` | `configs/experiments/radiometric_refinement_board_warping.toml`, then `configs/supplement/board_warping_lenses.toml` | `scripts/slurm/prepare.sbatch` and `scripts/slurm/case_task.sbatch` (`EXPERIMENT=radiometric_refinement_board_warping`, `STAGES=start,refine`), then `scripts/slurm/board_warping.sbatch` (`BOARD_WARPING_CONFIG`); only with `--only=board_warping_lenses` | the prepare CPU only, 8 CPUs, 30 min; an array, one task per case (`BW_CASES`, else all six): 1 GPU, 8 CPUs, 4 h (not yet measured on this stack: with RC 0.11.5 a case took 20–66 min); the figure CPU only, 8 CPUs, 10 min (about 80 s) | stages `prepare`, `detect`; `start`, `refine` of task i's case; `figures`: the six height fields (201 × 201 samples), `meta.toml` (each page and surface, the in-plane aspect and skew, per board the lenses' agreement in the heights and the difference between the board's offsets learned with two lenses), the figure — surfaces, maps, keys — into `plots/` |
| the sweeps' chains (Supp., the panoramas) | `scripts/panorama_vignetting.jl` | `configs/experiments/panorama_vignetting_fx3_<lens>.toml` | `scripts/slurm/panorama_vignetting.sbatch`; only with `--only=panorama_vignetting` (or `--only=panorama_stitching`, which runs it first) | an array, one task per lens (`SETS`): 1 GPU, 8 CPUs, 1 h (a sweep's chain took about 4 min on one GPU in the run of 2026-11-14, on an earlier version of the chain) | stage `evaluate`: the decode, the rotations (LoMa on the GPU), the sweep's own vignetting, its seam scores and renders |
| the panoramas (Supp., the panoramas) | `scripts/panorama_stitching.jl` | `configs/experiments/panorama_stitching_fx3_<lens>.toml` | `scripts/slurm/panorama_stitching.sbatch`; only with `--only=panorama_stitching`, after the f/4 refinements and the sweeps' chains | an array, one task per lens (`PS_SETS`), two phases: `export` (CPU, 8 CPUs, 30 min) and `panoramas` (a GPU job, 8 CPUs, 1 h: it reads the sweep's registration through PanoramaStitching's example, which resolves LoMa's CUDA provider; the seam scores and renders run on the host, about 1 min per candidate) | stages `refine`, `evaluate`, `rooftop`; then `panoramas`: per lens `panorama_original.png`, `panorama_learned.png` and `panoramas.json` (the seam scores) |
| supplement: the panoramas figure | `scripts/panorama_figure.jl` | `configs/supplement/panorama_figure.toml` | by hand (no stage of the graph yet): one CPU job after the three lenses' panoramas | CPU only, 8 CPUs, 16 GB, 10 min | stage `figures`: the six panoramas through the display curve, at the panels' width, the figure into `plots/` |
| the teaser (Fig. 1) | `scripts/teaser.jl` | `configs/experiments/teaser.toml` | by hand (no stage of the graph yet): one CPU job after the Tamron's panoramas and the texture set's stage `bands` | CPU only, 8 CPUs, 16 GB, 10 min | stage `teaser`: the two crops of the panoramas, the strip of the texture figure's bands, the figure into `plots/` |
| the detector bias (Sec. 5.1) | `scripts/detector_bias.jl` | `configs/experiments/detector_bias.toml` | `scripts/slurm/case_task.sbatch` (`EXPERIMENT=detector_bias`, `STAGES=start,refine,evaluate`), after the image regions and the distance maps; only with `--only=detector_bias`. Then, by hand, one CPU job: `julia --project=experiments experiments/scripts/detector_bias.jl --stage=figures` | an array, one task per case (`DETECTOR_BIAS_CASES`, else both): 1 GPU, 8 CPUs, 2 h; the figure CPU only, 8 CPUs (the cases' thread count, which keys the evaluations it reads), 32 GB, 10 min | stages `start`, `refine` and `evaluate` of task i's case; stage `figures`: the tables and the figure into `plots/` |
| the distance maps (the boards' `edge.sdf`) | `scripts/distance_maps.jl` | `configs/experiments/distance_maps.toml` | `scripts/slurm/distance_maps.sbatch`, after setup, before every stage whose cases read a map | CPU only, 8 CPUs, 30 min | stage `prepare`: each map's `sdf.f32` + `sdf.json` → `distance_map.json` (+ `.tiff`), produced or loaded by its key |
| the chrome evaluations | `scripts/chrome_evaluation.jl` | each `[chrome.<key>] config` of `configs/experiments/chrome_evaluation_paper.toml` | `scripts/slurm/chrome_evaluation.sbatch` (stages `chrome_evaluation`, `chrome_evaluation_score`) | an array, one task per chrome set (`CHROME_SETS`): phases decode (CPU), detect (1 GPU), index (CPU), 8 CPUs, 1 h; then evaluate (CPU) after the refinements | the products `chrome_evaluation_paper.jl`'s `sets` stage reads in the regeneration (reused there) |
| golden calibration cases (the package's gate) | `scripts/calibrate.jl` | `configs/calibration/siemens_coarse_full.toml` | `scripts/slurm/golden.sbatch`; only with `--only=golden` | CPU, 8 CPUs, `JULIA_NUM_THREADS=8`, 3 h | stage `refine` of each case, compared with the recorded goldens (`GOLDEN_TOL`) |

The thread count keys a run (`src/drivers.jl`), so each launcher fixes `JULIA_NUM_THREADS` from its CPUs, one count for every
launcher: a driver finds another driver's run by its key at its own count. The launchers' files say 28; the runs of record were made
with `--cpus-per-task=16` (16 threads), which the regeneration then needs too. A launch makes its runs in its own
results tree. The machine's options are its site profile's (VisualGeometryToolkitEval `configs/sites/`): `cpus_per_task` and
`mem_per_cpu` for every job (the count then keys every run of the launch) and `gpu_limit` (at most that many GPU array tasks at a
time: the arrays are throttled and run one after another). The one command ends with the regeneration and the results branch: the
author reviews the branch before anything reaches the paper.

## Asset index

The rows the checks read: every file the paper reads (`\input`, `\includegraphics`), generated by a driver of this project or
static with its provenance. A file the paper reads through `\rcfigdir` (the panels of the external generators' figures) is not a
static reference; the files below name their folders, and the suite's allow-list (`test/fixtures/paper_allow.toml`) states each
such reference with its reason (the manuscript's files are the co-author's and are not rewritten for the checks).

| label | asset | kind | generator | config | data | provenance |
|---|---|---|---|---|---|---|
| `fig:convergence-rms` | `../paper/fig/convergence-rms.tex` | generated | `scripts/convergence.jl` writes (stage panels, its panels in the folder beside it), `scripts/regenerate_paper.jl` deploys | `configs/experiments/convergence.toml` | tiers `rc_raw_photos`, `rc_baselines`, `rc_calibration`, `rc_thorlabs` | Main Sec. 5.2, Fig. 5: the median per LM iteration (0..25) of the detections' true reprojection error and of the held-out Thorlabs error, px, si30_spline_warped (left) and ag30_spline_warped (right); the submitted files (T. Harder's rms_figure.py) are reproduced from their medians, every line but the generator line (test/convergence_tests.jl) |
| — | `../paper/latex/numbers.tex` | generated | `scripts/numbers.jl` | `configs/figures/numbers.toml` | the experiments' tiers: `rc_raw_photos`, `rc_baselines`, `rc_calibration`, `rc_thorlabs` | Main: the number macros the prose cites; every macro cited (test/reproduce_tests.jl) |
| `fig:teaser` | `../paper/fig/teaser.tex` | generated | `scripts/teaser.jl` writes (stage teaser, its panels in the folder beside it) | `configs/experiments/teaser.toml` | the upstream configs' inputs: tiers `rc_raw_photos`, `rc_baselines`, `rc_calibration` | Main Sec. 1, Fig. 1: a part of the Tamron 20 mm sweep's panorama (columns 500 … 960, rows 0 … 250 of 3600 × 725 px, brightened by 1.2) composed from the frames as they are and with the learned vignetting divided out, over one strip of the first three text lines of the page from the four bands of `fig:texture-learned`, panels in `../paper/fig/teaser/`; the submitted files (T. Harder's teaser.py and crop_pair.py, from the panoramas of the run of 2026-11-14 and the texture figure's tables) are reproduced from those panoramas and tables, every line but the generator line and every pixel (test/teaser_tests.jl: the files and the strip; the crops' arithmetic); the panoramas this project's stage composes are a later version's of the chain and not those pixels. Placed in the paper by hand (known limitation: regenerate_paper does not deploy it; the paper's copies state `\rcfigdir`'s default as `fig`, the stage writes `src/fig`) |
| — | `../paper/fig/teaser-caption.tex` | static | — | — | — | Main Sec. 1, Fig. 1: the teaser's caption, written by hand (the stage writes a caption of its own into `plots/`, which the paper does not read) |
| `fig:model` | `../paper/latex/fig_model.tex` | static | — | — | — | Main Sec. 3, Fig. 2: hand-written TikZ |
| `fig:boundary` | `../paper/latex/fig_boundary.tex` | static | — | — | — | Main Sec. 4.1, Fig. 3: hand-written TikZ |
| `fig:detector-bias` | `../paper/fig/detector-bias-mitsuba.tex` | generated | `scripts/detector_bias.jl` writes (stage figures, its panels and keys in the folder beside it) | `configs/experiments/detector_bias.toml` | tiers `rc_synthetic`, `rc_baselines`, `rc_calibration` | Main Sec. 5.1, Fig. 4: on Mitsuba renders with a known camera and known poses (an AprilGrid through a FOV-model camera, OpenCV's corners and its OpenCV-14 calibration; the Siemens-star board through a Double Sphere fisheye, T. Schöps's star detector and his spline), per row one render, the detector's bias over all renders of the set and the error of the calibrated camera against the truth before and after the radiometric refinement, under the calibrated and under the true poses, panels in `../paper/fig/detector-bias-mitsuba/`; the submitted files (T. Harder's detector_bias.py) are reproduced from the evaluations of the paper's runs, every line but the generator line and the default of `\rcfigdir` (the repository's copy says `fig`), every pixel (test/detector_bias_tests.jl draws the figure from its tables); the stage writes into `plots/`, and the paper's files are the submitted ones until they are replaced by hand (the driver is not in the regeneration's list, `configs/figures/numbers.toml`; its runs are the stage `detector_bias` of the graph, only when named); the caption is written by hand in the paper |
| `tab:refinement-chrome` | `../paper/fig/refinement-table-ci.tex` | static | — | — | — | Main Sec. 5.2, Tab. 2: EXTERNAL — T. Harder's `refinement-table/tools/refinement_table.py` from the chrome evaluations of RC 0.11.5 (the results of 2026-11-17); ours beneath it: `scripts/chrome_evaluation_paper.jl` `[tables.refinement_chrome]` |
| `fig:vignetting` | `../paper/fig/vignetting.tex` | generated | `scripts/vignetting_figure.jl` writes (stage figures, its panels in the folder beside it), `scripts/regenerate_paper.jl` runs it | `configs/experiments/vignetting_figure.toml` | tiers `rc_raw_photos`, `rc_calibration` | Main Sec. 5.3, Fig. 6: per lens the flat field of a diffuser photograph, the vignetting its f/4 refinement learned and the thin lens model's cos⁴α along the sensor's diagonal through the centre of the learned profile; the generator writes the submitted files (T. Harder's vignetting.py) from their tables, every line but the generator line (test/vignetting_figure_tests.jl); the stage's own tables differ from those on purpose (the flat field from the RAW photosites, the diagonals to the lens coverage less 15 px, the focal length the refinement's `camera.f_mm`), so its figure is not the submitted one; the stage writes into `plots/`, and the paper's files are the submitted ones until they are replaced by hand (the driver is not in the regeneration's list, `configs/figures/numbers.toml`); the caption is written by hand in the paper |
| `fig:focus-surfaces` | `../paper/fig/focus-surfaces.tex` | generated | `scripts/focus_surfaces.jl` writes (stage panels, its panels in the folder beside it) | `configs/experiments/focus_surfaces.toml` | the two refinements' inputs: tiers `rc_raw_photos`, `rc_calibration` | Main Sec. 5.4, Fig. 7 (left): the in-focus surfaces of the refined cameras of f4_fisheye_spline_warped (left) and sitla_spline_warped (right), every pixel's in-focus point on a 16 px lattice, panels in `../paper/fig/focus-surfaces/`; the submitted files (T. Harder's focus_surfaces.py, from the refinements of 2026-11-16) are reproduced from those two cameras, every line but the generator line and the default of `\rcfigdir` (the repository's copy says `fig`; test/focus_surfaces_tests.jl draws the figure from its tables); the stage writes into `plots/`, and the paper's files are the submitted ones until they are replaced by hand (known limitation: regenerate_paper runs the stage and does not deploy a figure of several files); the caption is written by hand in the paper |
| `fig:texture-learned` | `../paper/fig/texture-learned.tex` | generated | `scripts/texture_recovery.jl` writes (stage bands, its bands in the folder beside it), `scripts/regenerate_paper.jl` runs it | `configs/experiments/texture_recovery_14mm_small.toml` | tiers `rc_raw_photos`, `rc_baselines`, `rc_calibration` | Main Sec. 5.4, Fig. 7: the first three lines of the page's text in the design, in a training photograph (DSC03316, number 39 of 63), in the pinhole's texture and in ours, bands in `../paper/fig/texture-learned/`; the submitted files (T. Harder's texture_learned.py) are reproduced from the run's records, every line but the generator line and every pixel (test/texture_learned_figure_tests.jl). The photograph's band is an input of record (`configs/inputs/texture_learned_photograph_20261121.toml`: `tools/texture_learned_photograph.py`, rawpy and OpenCV, author-side); the stage refuses it when the run chooses another photograph. Placed in the paper by hand (known limitation: regenerate_paper does not deploy it; the paper's copies state `\rcfigdir`'s default as `fig`, the stage writes `src/fig`) |
| `tab:defocus` | `../paper/fig/defocus-table-ci.tex` | static | — | — | — | Main Sec. 5.4, Tab. 3: EXTERNAL — T. Harder's `defocus-table/tools/defocus_table.py` from the chrome evaluations of RC 0.11.5; ours beneath it: `[tables.defocus]` of `chrome_evaluation_paper.toml` |
| `tab:supp-refinement-a` | `../paper/fig/refinement-table-supp-a.tex` | static | — | — | — | Supp. Sec. A.13.2, Tab. A4: EXTERNAL — T. Harder's `refinement-table/tools/refinement_table.py` (every model on every chrome set) |
| `tab:supp-refinement-b` | `../paper/fig/refinement-table-supp-b.tex` | static | — | — | — | Supp. Sec. A.13.2, Tab. A5: EXTERNAL — as Tab. A4 |
| `tab:supp-runtime` | `../paper/fig/runtime-table.tex` | generated | `scripts/timing.jl` |  `configs/supplement/timing.toml` | the case's inputs: `rc_raw_photos`, `rc_baselines` | Supp. Sec. A.14, Tab. A6: the runtime of one calibration (si30_spline_warped, 8 B200 runs), product of record f28d6ccaa124db2b; the stage record writes the table, its statement and the timing macros; placed in the paper by hand on 2026-11-18 (known limitation: regenerate_paper does not deploy it) |
| `fig:supp-thorlabs-target` | `../paper/fig/thorlabs-target.jpg` | static | — | — | — | Supp. Sec. A.13.1, Fig. A1: a photograph of the Thorlabs R3L3S7N1 target |
| — | `../paper/preamble.tex`, `../paper/sec/0_abstract_new.tex`, `../paper/sec/introduction.tex`, `../paper/sec/image_formation.tex`, `../paper/sec/ablation.tex`, `../paper/sec/pinhole_disadvantages.tex`, `../paper/sec/related_work.tex`, `../paper/sec/approach_new.tex`, `../paper/sec/calibration_pipeline.tex`, `../paper/sec/experiments.tex`, `../paper/sec/conclusion.tex`, `../paper/sec/X_suppl.tex` | static | — | — | — | the Blens manuscript (subtree `paper/`), hand-written |
