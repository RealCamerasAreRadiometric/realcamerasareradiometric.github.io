#!/usr/bin/env julia
# reproduce: reads configs/supplement/board_warping.toml
# reproduce: writes ../paper/fig/board-warping.tex
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/board_warping.jl — the supplement's board-warping figure (fig:board-warping): the height of the printed calibration
# boards above their plane as the radiometric refinements of record estimated it, the targets in rows and the lenses in columns.
# Config of record: configs/supplement/board_warping.toml (a supplement experiment: its products under
# results_root()/supplement/board_warping/) (src/board_warping.jl; the drawing code src/paper_figures/board_warping.jl).
#
# STAGES (each a product keyed by its config record and the keys of the products it reads; <results>/supplement/board_warping/<stage>/<key>/)
#   figures   per set of [sets] the height field over the page and the warping's nodes, from the calibrated target of its case's
#             refinement (cases.tsv, dump/); each page's size and surface's range (meta.toml); the paper's figure → plotsdir();
#             numbers.tex (empty: the figure cites no number)
#
#   julia --project=experiments experiments/scripts/board_warping.jl [--config=<toml>] [--plan] [--force]
#
# The refinements are scripts/radiometric_refinement.jl's products of the [upstream] config (GPU runs): the stage reads each by its
# key, and one that is missing is produced by that driver where this process holds a GPU, or named with the command that does.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))                                  # the refinement config the sets' cases are cases of (refine_product)
include(srcdir("paper_figures", "PaperFigures.jl"))                           # the figure's PGFPlots text (its generator: board_warping.jl there)
include(srcdir("board_warping.jl"))

const DRIVER = "scripts/board_warping.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "supplement", "board_warping.toml")))   # default: the config of record
const CFG = load_board_warping_config(CFG_PATH)
const RC_PATH = projectdir(CFG["upstream"]["radiometric_refinement"])
const RC_CFG = board_warping_refinements(CFG)
const CASES = board_warping_cases(CFG)

"The figures product (its output numbers.tex): keyed by [heights], [sets], [figure] and the keys of the refine products it reads."
figures_product() = board_warping_figures_product(CFG)

"The directory of the refine product of `case`, produced on demand by the refinements' own driver (`refinement_upstream`)."
refine_dir(case) = refinement_upstream(RC_CFG, RC_PATH, case, "refine", "the refinement of case $case")

"""Stage figures: every case's height field from its refine product, the tables and the drawing (`board_warping_figure`); the
figure's files kept in the product (`assets/`) and published to plotsdir() (`produce_figures`); numbers.tex, the output, written last.
The figure's files name this stage as their generator, and whose generator it runs."""
figures(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH,
                             inputs = [product_dir(refine_product(RC_CFG, c.case)) for c in CASES]) do dir
    board_warping_figure(dir, [(; c..., product = refine_dir(c.case)) for c in CASES], CFG;
                         writer = "experiments/$DRIVER (stage figures; T. Harder's board_warping.py)")
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
    println(">> $(CFG["figure"]["asset_dir"]) → ", joinpath(dir, "assets"))
end

"The figures stage computes on no device (the calibrated targets are records); a missing refinement is its own driver's GPU run."
const GPU_STAGES = Symbol[]

const STAGES = [
    Stage(:figures, "every set's height field, its page and range (meta.toml), the board-warping figure → plotsdir(), numbers.tex";
          items = () -> [figures_product()], product = identity, run = figures),
]

run_stages("board_warping", STAGES)
