#!/usr/bin/env julia
# reproduce: writes ../paper/latex/numbers.tex
# reproduce: reads configs/figures/regenerate_paper.toml
# reproduce: reads configs/figures/numbers.toml
# provenance: every result is a keyed run of a driver with its _config.toml and _provenance.toml (VisualGeometryToolkitEval produce);
# the verify reports (plotsdir()/verify.tsv the figures, plotsdir()/latex/verify.tsv the tables) state each deployed asset against the
# submitted one.
# paper-project: allow P1 an orchestrator: it writes no result of its own; every result is a driver's keyed run with its records (each driver is loaded by run_isolated, not a static include), and it only copies their renders into the paper
# paper-project: allow P2 an orchestrator: it keys no output of its own; the drivers key their runs and the paper reads fixed paths
# =============================================================================
# scripts/regenerate_paper.jl — THE entry point that regenerates every generated paper asset (REPRODUCE.md indexes each asset:
# label → asset → generator → config → data) from the listed drivers' runs; the refinements those drivers read
# (scripts/radiometric_refinement.jl) are run before it: scripts/reproduce/reproduce_results_dgx.sh (configs/reproduce.toml). `make figures` in paper/ runs it; `make
# figures-gpu` adds the GPU stages. Config of record: configs/figures/regenerate_paper.toml.
#
#   julia --project=experiments experiments/scripts/regenerate_paper.jl [--gpu] [--plan] [--stage=a,b] [--force] [--config=<toml>]
#                                                                 [--results-branch]
#
# STAGES (in dependency order; a failing stage is fatal)
#   <config>  one per experiment config of the numbers config's [runs] experiments (the one list of drivers), named by the
#             config (chrome_evaluation_paper, texture_recovery_14mm_small, …): the driver
#             (scripts/<name>.jl) loaded in a module of its own (VisualGeometryToolkitEval run_isolated) and its stages run; a
#             stage whose run is complete is skipped (each driver's own done predicates). Without --gpu the stages the driver
#             declares in GPU_STAGES are not run: a pending one stops the regeneration and names --gpu. With --gpu, GPU 0 must
#             have [gpu] min_free_gib free, or the regeneration is refused before anything runs.
#   numbers   scripts/numbers.jl ([runs] numbers): the drivers' numbers.tex fragments → ../paper/latex/numbers.tex, replaced
#             only with the numbers config's [deploy] replace_submitted when the submitted copy differs
#   verify    every figure config of record (configs/figures/*.toml with [render]; [render] asset "figures.<name>" a PNG,
#             "tables.<name>" a generated LaTeX table): its render in plotsdir() (the tables plotsdir()/latex) against the
#             submitted ../paper/fig (../paper/latex) copy → plotsdir()/verify.tsv, plotsdir()/latex/verify.tsv
#             (VisualGeometryToolkitEval verify_assets, one report per kind)
#   deploy    each render into ../paper/fig or ../paper/latex when identical or new; a differing one only with its config's
#             [deploy] replace_submitted = "on" (refused otherwise, naming the verify report and the configs: src/assets.jl
#             deploy_figures, one deploy_assets call per kind)
#
# --results-branch (the one-command reproduction, configs/reproduce.toml: its results-branch stage carries the renders into a
# results branch of the paper for review): nothing is deployed into paper/ — numbers.tex is gathered into its run and copied to
# plotsdir()/latex/ beside the tables, and the deploy stage is left out.
# --stage selects among these; --force reruns every stage of every selected driver. --plan prints this table and runs nothing
# (each driver's own plan: scripts/<driver>.jl --plan). Data: VGTK_DATA_ROOT binds data_root() (REPRODUCE.md).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("assets.jl"))
using VisualGeometryToolkitEval: cli_option
using VisualGeometryToolkitEval.Reproduce: verify_assets, assets_deployed

const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "figures", "regenerate_paper.toml")))   # default: the config of record
const CFG = load_config_tables(CFG_PATH, REGENERATE_CONFIG_KEYS)
const GPU = "--gpu" in ARGS
const FORCE = "--force" in ARGS
const RESULTS_BRANCH = "--results-branch" in ARGS   # nothing deployed into paper/: the renders stay in plotsdir() for the results branch
const EXPERIMENTS = driver_configs(CFG)
const FIGURES = [(cfg = load_figure_config(p), path = p) for p in figure_configs()]           # every figure config of record: figures and tables (none at
                                                                                              # present: the paper's figures and tables are external or numbers)
const FIGURE_ASSETS = [figure_asset(f.cfg) for f in FIGURES if asset_kind(f.cfg)[1] === :figure]   # the PNGs of ../paper/fig
const TABLE_ASSETS = [figure_asset(f.cfg) for f in FIGURES if asset_kind(f.cfg)[1] === :table]     # the generated tables of ../paper/latex

function run_driver_stages(config::AbstractString)
    args = ["--plan", "--config=$config"]
    FORCE && push!(args, "--force")
    m = run_isolated(driver_script(config); args)
    name = TOML.parsefile(config)["experiment"]["name"]
    gpu_stages = driver_gpu_stages(m, name)
    stages = [host_stage(s; driver = name, gpu_stages, gpu = GPU) for s in Base.invokelatest(getfield, m, :STAGES)]
    run_stages(name, stages; argv = FORCE ? ["--force"] : String[])
end

# the renders of each kind against the submitted copies: one verify report per kind (plotsdir()/verify.tsv the figures,
# plotsdir()/latex/verify.tsv the tables)
function verify()
    isempty(FIGURE_ASSETS) || (verify_assets(plotsdir(), paper_figdir(), FIGURE_ASSETS); println(">> ", plotsdir("verify.tsv")))
    isempty(TABLE_ASSETS) || (verify_assets(plotsdir("latex"), paper_latexdir(), TABLE_ASSETS); println(">> ", plotsdir("latex", "verify.tsv")))
    return nothing
end

deploy() = (isempty(FIGURES) || deploy_figures(plotsdir(), paper_figdir(), FIGURES; latexdir = paper_latexdir()); nothing)
deployed() = (isempty(FIGURE_ASSETS) || assets_deployed(plotsdir(), paper_figdir(), FIGURE_ASSETS)) && (isempty(TABLE_ASSETS) || assets_deployed(plotsdir("latex"), paper_latexdir(), TABLE_ASSETS))

const STAGES = [
    # a stage per config, named by the config (its driver's name where the driver has one config: focus_surfaces; a variant's own
    # where it has several: texture_recovery_14mm_small), so --stage picks one
    [Stage(Symbol(first(splitext(basename(c)))), "the stages of $(relpath(driver_script(projectdir(c)), projectdir())) --config=$c";
           check = () -> run_driver_stages(projectdir(c))) for c in EXPERIMENTS];
    Stage(:numbers, RESULTS_BRANCH ? "scripts/numbers.jl → plotsdir()/latex/numbers.tex" : "scripts/numbers.jl → ../paper/latex/numbers.tex";
          check = () -> (run_isolated(scriptsdir("numbers.jl");
                                      args = ["--config=$(projectdir(CFG["runs"]["numbers"]))"; RESULTS_BRANCH ? ["--results-branch"] : String[]]);
                         nothing));
    Stage(:verify, "the renders against the submitted figures and tables → plotsdir()/verify.tsv, plotsdir()/latex/verify.tsv"; check = verify);
    (RESULTS_BRANCH ? Stage[] :
     [Stage(:deploy, "the renders into ../paper/fig and ../paper/latex (a differing one only with [deploy] replace_submitted)"; check = () -> deployed() || deploy())])...
]

GPU && !("--plan" in ARGS) && require_gpu_memory(CFG["gpu"]["min_free_gib"])
run_stages("regenerate_paper", STAGES)

if !("--plan" in ARGS)
    println("\nregenerate_paper: DONE", GPU ? "." : " (the GPU stages were reused, not run; add --gpu).",
            RESULTS_BRANCH ? " The renders are in plotsdir(), for the results branch." : " Now: make -C paper")
end
