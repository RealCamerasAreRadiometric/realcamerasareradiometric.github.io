#!/usr/bin/env julia
# reproduce: writes ../paper/latex/numbers.tex
# reproduce: reads configs/figures/numbers.toml
# provenance: the gathered file is a keyed run, <results_root()>/numbers/numbers-<key>/ with _config.toml and _provenance.toml
# (VisualGeometryToolkitEval produce), copied into the paper; its header names every fragment (results_root()-relative).
# =============================================================================
# scripts/numbers.jl — the paper's number macros: the numbers.tex fragment each experiment driver's figures stage writes into its
# keyed figures run (<results_root()>/<experiment>/figures-<key>/numbers.tex), for the experiment configs of [runs] experiments, gathered into
# one numbers.tex — one \newcommand per number, sorted (VisualGeometryToolkitEval Reproduce gather_numbers) — in a keyed run of its
# own, then deployed into ../paper/latex/numbers.tex under the figures' rule (Reproduce deploy_assets): a submitted numbers.tex
# that differs is replaced only with the config's [deploy] replace_submitted = "on". Each driver is loaded in a module of its own
# with --plan (it runs nothing) to find its figures run; a missing fragment names the driver stage to run.
# Config of record: configs/figures/numbers.toml.
#
#   julia --project=experiments experiments/scripts/numbers.jl [--config=<toml>] [--plan] [--results-branch]
#
# --results-branch: the gathered numbers.tex is copied to plotsdir()/latex/ (beside the generated tables, for the results branch
# of the one-command reproduction), not deployed into the paper.
#
# scripts/regenerate_paper.jl runs it after the drivers.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("assets.jl"))
using VisualGeometryToolkitEval.Reproduce: gather_numbers, deploy_assets, tex_macro_values

const DRIVER = "scripts/numbers.jl"

const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "figures", "numbers.toml")))   # default: the config of record
const CFG = load_config_tables(CFG_PATH, NUMBERS_CONFIG_KEYS)
const RESULTS_BRANCH = "--results-branch" in ARGS

# The product is keyed by this config, the figures products it gathers and the numbers they hold (a fragment rewritten in place
# keys a new product); the paper's copy is the product's numbers.tex.
numbers_product(fragments) = driver_product("numbers", Dict{String,Any}("experiment" => CFG["experiment"], "runs" => CFG["runs"],
    "upstream" => Dict(c => Dict{String,Any}("run" => basename(dirname(f)), "numbers" => Dict(tex_macro_values(f)))
                       for (c, f) in zip(CFG["runs"]["experiments"], fragments))); output = "numbers.tex")

"The numbers.tex fragment of the figures run of each experiment config of [runs] experiments; a missing one names its stage."
fragments() = map(CFG["runs"]["experiments"]) do c
    f = numbers_fragment(projectdir(c))
    require_input(f; what = "the numbers of $c",
                  provenance = "the figures stage of $(relpath(driver_script(projectdir(c)), projectdir())) --config=$c")
end

"""
    gather(fragments = fragments(); latexdir = dirname(paper_numbers_path())) -> path

The `fragments` gathered into the keyed numbers product (produced or loaded) and its numbers.tex deployed into `latexdir`: when the
submitted copy there differs, only with [deploy] replace_submitted = "on" (refused otherwise, naming the run's verify.tsv).
"""
function gather(frags = fragments(); latexdir::AbstractString = dirname(paper_numbers_path()))
    p = numbers_product(frags); key = p.key
    dir = produce(p; product_records(DRIVER, CFG_PATH)..., inputs = frags) do d
        gather_numbers(frags, joinpath(d, "numbers.tex"); generator = "scripts/regenerate_paper.jl (scripts/numbers.jl)", root = results_root())
        nothing
    end
    out = product_output(p)
    if RESULTS_BRANCH                                   # for the results branch: beside the tables in plotsdir(), not deployed
        dst = joinpath(mkpath(plotsdir("latex")), basename(out))
        cp(out, dst; force = true)
        println(">> ", dst, " (run ", key, "; for the results branch, not deployed)")
        return dst
    end
    deploy_assets(dir, latexdir, [basename(out)]; replace_submitted = replace_submitted(CFG))
    println(">> ", joinpath(latexdir, basename(out)), " (run ", key, ")")
    return joinpath(latexdir, basename(out))
end

run_stages("numbers", [Stage(:gather, "the drivers' numbers.tex fragments → ../paper/latex/numbers.tex"; check = () -> (gather(); nothing))])
