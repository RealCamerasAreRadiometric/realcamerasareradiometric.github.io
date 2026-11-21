#!/usr/bin/env julia
# reproduce: reads configs/supplement/panorama_figure.toml
# reproduce: writes ../paper/fig/panoramas-james.tex
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/panorama_figure.jl — the supplement's panoramas figure: per lens the panorama of its pure-rotation sweep composed from the
# frames as they are and from the frames with the learned vignetting divided out (src/panorama_figure.jl), drawn as the paper's
# PGFPlots files (src/paper_figures/panoramas.jl). Config of record: configs/supplement/panorama_figure.toml (a supplement
# experiment: its products under results_root()/supplement/panorama_figure/).
#
# STAGES (each a product keyed by its config record and the keys of the products it reads; <results>/supplement/panorama_figure/<stage>/<key>/)
#   figures   per lens the two panoramas of its panoramas product through the display curve, at the panels' width, the figure's
#             files → plotsdir(<asset_dir>.tex, <asset_dir>/<panel>-<column>.{tex,png}, <asset_dir>-caption.tex), numbers.tex
#             (empty: the figure cites no number)
#
#   julia --project=experiments experiments/scripts/panorama_figure.jl [--config=<toml>] [--plan] [--force]
#
# The stage reads the panoramas product of each lens's config (scripts/panorama_stitching.jl); one that is missing is produced by
# that driver where this process holds a GPU, or named with the command that does.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("panorama_example.jl"))                                        # the sweeps' configs the lenses' configs read (panorama_vignetting_product)
include(srcdir("radiometric_refinement.jl"))                                  # the refinement configs they read (refine_product)
include(srcdir("panorama_stitching.jl"))                                      # the lenses' configs and their panoramas products
include(srcdir("panorama_figure.jl"))
include(srcdir("paper_figures", "PaperFigures.jl"))                           # the figure's PGFPlots text (its generator: panoramas.jl there)
using VisualGeometryToolkitEval.Experiments: run_isolated                     # a missing panoramas product is produced by its own driver (driver_run)

const DRIVER = "scripts/panorama_figure.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "supplement", "panorama_figure.toml")))   # default: the config of record
const CFG = load_panorama_figure_config(CFG_PATH)
"Per lens its panorama_stitching config and that config's panoramas product (`panorama_figure_sources`), resolved once."
const SOURCES = panorama_figure_sources(CFG; path = CFG_PATH)

"The figure's product: keyed by the config's tables and the keys of the lenses' panoramas products (`panorama_figure_product`)."
figures_product() = panorama_figure_product(CFG, SOURCES)

"The directory of lens `k`'s panoramas product, produced on demand by its own driver (`panoramas_upstream`)."
panoramas_dir(k) = panoramas_upstream(SOURCES[k].config, "[lenses.$k]: the panoramas of $(SOURCES[k].config)")

"""Stage figures: the figure drawn by its generator (src/paper_figures/panoramas.jl, with the settings of `[figure]` and
`[lenses]`; its files name this stage as their generator) from the lenses' panoramas products, kept in the product (`assets/`)
and published to plotsdir() (`produce_figures`); numbers.tex, the output, written last."""
figures(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH,
                             inputs = [product_dir(SOURCES[k].product) for k in CFG["figure"]["order"]]) do dir
    written = PaperFigures.generate("panoramas"; panoramas = Dict{String, String}(k => panoramas_dir(k) for k in CFG["figure"]["order"]),
                                    out = joinpath(dir, "assets"), generator = "experiments/$DRIVER (stage figures; T. Harder's panorama_pgf.py)",
                                    panorama_figure_settings(CFG)...)
    foreach(f -> println(">> figures → ", f), written)
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
end

"The stage runs on no device (the panoramas are files); a missing panoramas product is its own driver's run."
const GPU_STAGES = Symbol[]

const STAGES = [
    Stage(:figures, "the supplement's panoramas figure (plotsdir(<asset_dir>.tex, <asset_dir>/<panel>-<column>.{tex,png}, <asset_dir>-caption.tex)), numbers.tex";
          items = () -> [figures_product()], product = identity, run = figures),
]

run_stages("panorama_figure", STAGES)
