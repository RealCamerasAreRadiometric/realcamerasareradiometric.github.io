#!/usr/bin/env julia
# reproduce: reads configs/experiments/convergence.toml
# reproduce: writes ../paper/fig/convergence-rms.tex
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/convergence.jl — the convergence experiment (owner 2026-11-15): the bias of T. Schöps's method, per LM iteration of our
# radiometric refinement from his camera — the bearing divergence of his detections (deg) against the Thorlabs reprojection error of
# our camera (px), on one figure with two y axes. Config of record: configs/experiments/convergence.toml (src/convergence.jl).
#
# STAGES (each product keyed by only what it reads and the keys of the products it reads)
#   refine       per case, its refinement of record with a checkpoint per iteration: scripts/radiometric_refinement.jl on the
#                [upstream] config (prepare, start, refine of that case), in a module of its own — its own products, produced or loaded
#   chrome       per chrome set the cases name, its index: scripts/chrome_evaluation.jl on the set's config (prepare, detect on the
#                GPU, index), in a module of its own — produced or loaded
#   convergence  per case, the record convergence.json: every iteration's bearing divergence and Thorlabs scores (src/convergence.jl)
#   panels       the paper's figure (fig:convergence-rms): the [panels] cases' detections' reprojection medians and held-out Thorlabs
#                medians per iteration, px, as the paper's PGFPlots frame and panels (src/convergence_panels.jl) and their macros
#   figure       per case, the dual-axis pgfplots fragment, the style sheet and a preview (plotsdir(<asset_dir>/<case>/))
#
#   julia --project=experiments experiments/scripts/convergence.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--case=a,b] [--force]
#
# --force re-runs this driver's stages, never an upstream's (refine, chrome reuse their complete keyed runs).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("chrome_evaluation.jl"))
include(srcdir("figure_style.jl"))
include(srcdir("convergence.jl"))
using VisualGeometryToolkitEval.Experiments: run_isolated

const DRIVER = "scripts/convergence.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "convergence.toml")))   # default: the config of record
const CFG = load_convergence_config(CFG_PATH)
const CASES = cli_subset(ARGS, "case", sort(collect(keys(CFG["cases"]))); what = "case")
const RC_PATH = projectdir(CFG["upstream"]["radiometric_refinement"])
const RC_CFG = convergence_refinements(CFG)
const CHROMES = sort(unique([CFG["cases"][c]["chrome"] for c in CASES]))

"Stage refine of `case`: scripts/radiometric_refinement.jl on its config of record, that case's prepare, start and refine (in a module of its own)."
refine(case) = driver_run(REFINEMENT_DRIVER, RC_PATH, "prepare,start,refine"; case).make()
"Stage chrome of the set config `c`: scripts/chrome_evaluation.jl on it, prepare, detect and index (in a module of its own)."
chrome(c) = driver_run(CHROME_EVALUATION_DRIVER, c, "prepare,detect,index").make()
"Stage figure of `case`: the record's dual-axis fragment, the style sheet and the preview (plotsdir(<asset_dir>/<case>/))."
figure(case) = produce_figures(convergence_figure_product(CFG, case), String[]; driver = DRIVER, config_path = CFG_PATH,
                               inputs = [product_dir(convergence_product(CFG, case))]) do dir
    r = load_convergence_record(joinpath(produce_upstream(convergence_product(CFG, case), "the convergence record of $case";
                                                          make = () -> run_convergence(CFG, case; driver = DRIVER, config_path = CFG_PATH),
                                                          command = driver_run(DRIVER, CFG_PATH, "convergence"; case).command), CONVERGENCE_RECORD_FILE))
    f = write_convergence_figure(joinpath(dir, "assets", CFG["figure"]["asset_dir"], case), r, CFG["figure"])
    cp(f.fragment, joinpath(dir, "figure.tex"); force = true)                 # the output, written last
    println(">> figure → ", f.fragment)
end

"""Stage panels: the paper's figure (fig:convergence-rms): per panel case its convergence record's held-out Thorlabs medians and the
detections' reprojection medians (`detection_reprojection_series`), the PGFPlots frame and panels (src/convergence_panels.jl) under
plotsdir(<asset_dir>), the per-iteration table panels.tsv and the sentence's macros (numbers.tex, the output)."""
function panels(p)
    P = CFG["panels"]; rc = RC_CFG
    cases = (P["left"], P["right"])
    produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH,
                    inputs = [product_dir(convergence_product(CFG, c)) for c in cases]) do dir
        nums = driver_numbers(CFG_PATH, DRIVER, p.key)
        rows = String["case\tit\tdet_median_px\tchrome_median_px\tdet_rms_px\tchrome_rms_px\tdet_n"]
        for c in cases
            r = load_convergence_record(joinpath(produce_upstream(convergence_product(CFG, c), "the convergence record of $c";
                                                                  make = () -> run_convergence(CFG, c; driver = DRIVER, config_path = CFG_PATH),
                                                                  command = driver_run(DRIVER, CFG_PATH, "convergence"; case = c).command), CONVERGENCE_RECORD_FILE))
            sdir = product_dir(start_product(rc, c)); rdir = product_dir(refinement_product(rc, c))
            d = detection_reprojection_series(sdir, rdir, r.iterations)
            h = r.thorlabs_px.motion_heldout
            for narrow in (false, true)
                f = joinpath(dir, "assets", P["asset_dir"], "$c$(narrow ? "-narrow" : "").tex"); mkpath(dirname(f))
                write(f, convergence_panel_tex(c, P[c]["title"], P[c]["target"], r.iterations, d.median, h.median; narrow))
            end
            convergence_panel_numbers!(nums, P, c, r.iterations, d.median, h.median)
            for (k, i) in enumerate(r.iterations)
                push!(rows, join((c, i, g6(d.median[k]), g6(h.median[k]), g6(d.rms[k]), g6(h.rms[k]), d.n), '\t'))
            end
            println(">> panel $c: detections $(g6(first(d.median))) → $(g6(last(d.median))) px (n $(d.n)), held-out $(g6(first(h.median))) → $(g6(last(h.median))) px")
        end
        write(joinpath(dir, "assets", P["asset_dir"] * ".tex"), convergence_figure_tex(cases...))
        write(joinpath(dir, "panels.tsv"), join(rows, '\n') * '\n')
        Reproduce.write_numbers(joinpath(dir, "numbers.tex"), nums)
    end
end

"The GPU stages: the refinements (bin/refine.jl's compute.backend) and the chrome sets' detect where its [detect] device is \"gpu\"."
const GPU_STAGES = [gpu_stages(:refine => [RC_CFG["cases"][c] for c in CASES]);
                    any(c -> get(load_chrome_evaluation_config(projectdir(c))["detect"], "device", "cpu") == "gpu", CHROMES) ? [:chrome] : Symbol[]]
const STAGES = [
    Stage(:refine, "every case's refinement with its per-iteration checkpoints (scripts/radiometric_refinement.jl: prepare, start, refine)";
          items = () -> CASES, product = c -> refinement_product(RC_CFG, c), run = refine),
    Stage(:chrome, "every chrome set's index (scripts/chrome_evaluation.jl: prepare, detect, index)";
          items = () -> CHROMES, product = c -> index_product(load_chrome_evaluation_config(projectdir(c))), run = chrome),
    Stage(:convergence, "every case's record: the bearing divergence and the Thorlabs scores per LM iteration (convergence.json)";
          items = () -> CASES, product = c -> convergence_product(CFG, c), run = c -> run_convergence(CFG, c; driver = DRIVER, config_path = CFG_PATH)),
    Stage(:panels, "the paper's figure fig:convergence-rms (plotsdir(<asset_dir>.tex, <asset_dir>/<case>[-narrow].tex), panels.tsv, numbers.tex)";
          items = () -> haskey(CFG, "panels") ? [convergence_panels_product(CFG)] : Product[], product = identity, run = panels),
    Stage(:figure, "every case's dual-axis pgfplots figure (plotsdir(<asset_dir>/<case>/): convergence_<case>.tex, figure_styles.tex, preview.tex/.pdf)";
          items = () -> CASES, product = c -> convergence_figure_product(CFG, c), run = figure),
]

run_stages("convergence", STAGES)
