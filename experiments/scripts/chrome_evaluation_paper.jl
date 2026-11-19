#!/usr/bin/env julia
# reproduce: reads configs/experiments/chrome_evaluation_paper.toml
# reproduce: writes plots/color-wheels.png (plotsdir(); not in the paper)
# reproduce: writes plots/latex/table_refinement_chrome.tex (plotsdir("latex"); the paper's Tab. 2 is the external refinement-table-ci.tex)
# reproduce: writes plots/latex/table_defocus.tex (plotsdir("latex"); the paper's Tab. 3 is the external defocus-table-ci.tex)
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/chrome_evaluation_paper.jl — the chrome evaluation's paper stages: the paper's chrome tables and figures from every chrome
# set's evaluate product. Config of record: configs/experiments/chrome_evaluation_paper.toml ([chrome.<key>] the sets' configs,
# [tables.<name>], [figures.<name>], [render]; src/chrome_evaluation_paper.jl). Each set is scripts/chrome_evaluation.jl's.
#
# STAGES (each a product keyed by its config record and the keys of the products it reads; datadir("exp_pro", <experiment>, <stage>, <key>))
#   sets      every [chrome] set's stages (prepare, detect on the GPU, index, evaluate): scripts/chrome_evaluation.jl on the set's
#             config, in a module of its own; a set whose evaluate product is produced skipped (every set with --force); --chrome=a,b
#             names a subset
#   figures   from the sets' evaluate products: every [tables.<name>] — table_<name>.{tsv,md,tex} into the figures product and
#             plotsdir("latex") (tab:refinement-chrome, tab:defocus, chrome_all; scripts/regenerate_paper.jl deploys the paper's) —,
#             numbers.tex (tab:defocus's macros, the convergence counts; scripts/numbers.jl gathers it) and every [figures.<name>]
#             (the chrome residual maps → plotsdir(): fig:color-wheels)
#
#   julia --project=experiments experiments/scripts/chrome_evaluation_paper.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--chrome=a,b] [--force]
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("chrome_evaluation.jl"))
include(srcdir("figures.jl"))                                                 # the renderers (module Figures)
include(srcdir("chrome_evaluation_paper.jl"))                                 # the paper stages' functions
import VisualGeometryToolkitEval.Reproduce as Reproduce
using VisualGeometryToolkitEval.Experiments: run_isolated

const DRIVER = "scripts/chrome_evaluation_paper.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "chrome_evaluation_paper.toml")))   # default: the config of record
const PAPER = load_chrome_paper_config(CFG_PATH)
"The chrome sets stage sets scores: every [chrome] key, or those `--chrome=a,b` names (the tables and figures read every set)."
const SETS = cli_subset(ARGS, "chrome", chrome_paper_sets(PAPER); what = "chrome set")
const PRODUCTS, SET_CFGS, RUNS = chrome_products(PAPER)
"Stage sets, per set: its stages, scripts/chrome_evaluation.jl on its config in a module of its own (each product produced or loaded)."
set_stages(k) = run_isolated(projectdir("scripts", "chrome_evaluation.jl"); args = ["--config=$(chrome_set_path(PAPER, k))"; "--force" in ARGS ? ["--force"] : String[]])
"The figures product (src/chrome_evaluation_paper.jl `chrome_figures_product`; scripts/numbers.jl reads its numbers.tex)."
figures_product() = chrome_figures_product(PAPER, RUNS)
"Stage figures: the tables (plotsdir(\"latex\")), their number macros (numbers.tex, the output: written last) and the residual maps (plotsdir())."
figures(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH, inputs = [product_dir(RUNS[k]) for k in chrome_paper_sets(PAPER)]) do dir
    rows = chrome_tables_rows(PAPER, PRODUCTS)                                    # every table's rows, once
    println(">> tables → ", join(write_chrome_tables(PAPER, (joinpath(dir, "assets", "latex"),), rows), ", "))   # published to plotsdir("latex")
    foreach(f -> println(">> figure → ", f), chrome_residual_figures(PAPER, joinpath(dir, "assets"), PRODUCTS))   # published to plotsdir()
    # the number macros (tab:defocus's, the convergence counts) through Eval's number writer, each with its provenance (this driver
    # and config, the evaluate product the values are read from and its commit); scripts/numbers.jl gathers the fragment
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), chrome_table_macros(PAPER, CFG_PATH, rows, PRODUCTS.provenance; driver = DRIVER))
end
"The sets stage runs each set's stages, among them detect, on the GPU where its [detect] device is \"gpu\" (the configs of record's)."
const GPU_STAGES = any(k -> get(get(SET_CFGS[k], "detect", Dict()), "device", "cpu") == "gpu", chrome_paper_sets(PAPER)) ? [:sets] : Symbol[]
const STAGES = [
    Stage(:sets, "every [chrome] set's stages (scripts/chrome_evaluation.jl on the set's config: prepare, detect, index, evaluate)";
          items = () -> SETS, product = k -> RUNS[k], run = set_stages),
    Stage(:figures, "every [tables.<name>] → plotsdir(\"latex\"), numbers.tex; every [figures.<name>] → plotsdir()";
          items = () -> [figures_product()], product = identity, run = figures),
]

run_stages("chrome_evaluation_paper", STAGES)
