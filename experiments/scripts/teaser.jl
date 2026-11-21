#!/usr/bin/env julia
# reproduce: writes ../paper/fig/teaser.tex
# reproduce: reads configs/experiments/teaser.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/teaser.jl — the paper's teaser figure (fig:teaser): a part of one lens's panorama, composed from the frames as they are
# and from the frames with the learned vignetting divided out, over one strip of the first text lines of the printed page whose
# parts come from the learned-texture figure's bands (src/teaser.jl), drawn as the paper's PGFPlots files
# (src/paper_figures/teaser.jl). Config of record: configs/experiments/teaser.toml.
#
# STAGES (each a product keyed by its config record and the keys of the products it reads; <results>/teaser/<stage>/<key>/)
#   teaser    the crops of the lens's panoramas (tables/<panel>.png: [teaser] box and gain), the figure's files from them and from
#             the texture figure's tables → plotsdir(<asset_dir>.tex, <asset_dir>/<panel>.{tex,png},
#             <asset_dir>/texture-strip.{tex,png}, <asset_dir>-caption.tex), numbers.tex (empty: the figure cites no number)
#
#   julia --project=experiments experiments/scripts/teaser.jl [--config=<toml>] [--plan] [--force]
#
# The stage reads two products by their keys: the panoramas product of [upstream] panorama_stitching
# (scripts/panorama_stitching.jl) and the bands product of [upstream] texture_recovery (scripts/texture_recovery.jl); one that is
# missing is produced by its own driver, or named with the command that does.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("panorama_example.jl"))                                        # the sweep's config the lens's config reads (panorama_vignetting_product)
include(srcdir("radiometric_refinement.jl"))                                  # the refinement config it reads (refine_product)
include(srcdir("panorama_stitching.jl"))                                      # the lens's config and its panoramas product
include(srcdir("paper_figures", "PaperFigures.jl"))                           # the figure's PGFPlots text (its generator: teaser.jl there) and its PNG files
using VisualGeometryToolkitEval.Experiments: run_isolated                     # the texture driver in a module of its own (its bands product); a missing upstream's driver (driver_run)
include(srcdir("teaser.jl"))

const DRIVER = "scripts/teaser.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "teaser.toml")))   # default: the config of record
const CFG = load_teaser_config(CFG_PATH)
"The lens's panoramas product and the learned-texture figure's bands product (`teaser_panoramas`, `teaser_bands`), resolved once."
const PANORAMAS = teaser_panoramas(CFG; path = CFG_PATH)
const BANDS = teaser_bands(CFG; path = CFG_PATH)

"The teaser's product: keyed by `[teaser]` and the keys of the two products it reads (`teaser_product`)."
figure_product() = teaser_product(CFG, PANORAMAS, BANDS)

"""Stage teaser: the crops of the lens's panoramas (`write_teaser_crops`, into the product's tables/) and the figure drawn from
them and from the texture figure's tables by its generator (src/paper_figures/teaser.jl, with the settings of `[teaser]`; its
files name this stage as their generator), kept in the product (`assets/`) and published to plotsdir() (`produce_figures`);
numbers.tex, the output, written last. Both products it reads are produced on demand."""
teaser(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH,
                            inputs = [product_dir(PANORAMAS.product), product_dir(BANDS.product)]) do dir
    pdir = panoramas_upstream(PANORAMAS.config, "[upstream] panorama_stitching: the panoramas of $(PANORAMAS.config)")
    bdir = produce_upstream(BANDS.product, "[upstream] texture_recovery: the learned-texture figure's tables (stage bands of $(BANDS.config))";
                            driver_run("scripts/texture_recovery.jl", BANDS.config, "bands")...)
    crops = write_teaser_crops(joinpath(dir, "tables"), pdir, CFG["teaser"])
    written = PaperFigures.generate("teaser"; crops, texture = joinpath(bdir, "tables"), out = joinpath(dir, "assets"),
                                    generator = "experiments/$DRIVER (stage teaser; T. Harder's teaser.py)", teaser_settings(CFG)...)
    foreach(f -> println(">> teaser → ", f), written)
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
end

"The stage runs on no device (the panoramas and the texture figure's tables are files); a missing upstream is its own driver's run."
const GPU_STAGES = Symbol[]

const STAGES = [
    Stage(:teaser, "the paper's figure fig:teaser (plotsdir(<asset_dir>.tex, <asset_dir>/<panel>.{tex,png}, <asset_dir>/texture-strip.{tex,png}, <asset_dir>-caption.tex)), its crops, numbers.tex";
          items = () -> [figure_product()], product = identity, run = teaser),
]

run_stages("teaser", STAGES)
