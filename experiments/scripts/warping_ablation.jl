#!/usr/bin/env julia
# reproduce: reads configs/studies/warping_ablation.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/warping_ablation.jl — the board warping ablation (owner 2026-11-16): which board parameterisation is right for each camera
# model, judged only by the held-out Thorlabs (chrome) error. Config of record: configs/studies/warping_ablation.toml
# (src/warping_ablation.jl): every global-model case of the paper's chrome sets and its capture set's central spline (the control),
# each refined with a rigid board (A), the heights only (B) and the config of record (C, heights + in-plane), scored on its chrome set.
#
# STAGES (each product keyed by only what it reads and the keys of the products it reads)
#   upstream  per chrome set, every upstream its refinements and its scoring read, produced or loaded (the decodes, the front ends'
#             runs — Schöps's on the GPU —, the image regions, the chrome set's detect on the GPU and its index, the start models), once
#             before the refinements so no two of them make one at the same time
#   refine    per case and mode, its refinement: the refinement driver's own start and refine products (src/drivers.jl run_refine) on
#             the mode's override of the case of record; ONE per GPU
#   evaluate  per chrome set, every start and every mode's refinement in one scoring by the chrome scorer under the set's protocol
#             of record, with each mode's paired bootstrap against the reference mode and against its start (CPU)
#   record    warping_ablation.json and warping_ablation.tsv (src/warping_ablation.jl warping_record)
#   figure    the pgfplots figure and the booktabs table, the style sheet and a preview (plotsdir(<asset_dir>/))
#
#   julia --project=experiments experiments/scripts/warping_ablation.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--chrome=a,b]
#         [--mode=a,b] [--case=a,b] [--shard=k/n] [--force]
#
# --chrome, --mode and --case (a case of record, or a variant's name) restrict the upstream, refine and evaluate stages' items; record
# and figure read every scoring. --force re-runs this driver's stages, never an upstream's.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("chrome_evaluation.jl"))
include(srcdir("figure_style.jl"))
include(srcdir("warping_ablation.jl"))

const DRIVER = "scripts/warping_ablation.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "studies", "warping_ablation.toml")))   # default: the config of record
const CFG = load_warping_ablation_config(CFG_PATH)
const PLAN = warping_plan(CFG)
const R = (; driver = DRIVER, config_path = CFG_PATH)

const CHROMES = cli_subset(ARGS, "chrome", sort(collect(keys(PLAN.sets))); what = "chrome set")
const MODES = cli_subset(ARGS, "mode", PLAN.modes; what = "mode")
"The refinements the refine stage runs: those of the selected chrome sets and modes, and of `--case` (a case of record or a variant)."
const ROWS = let rows = [r for r in PLAN.rows if r.chrome in CHROMES && r.mode in MODES], sel = cli_option(ARGS, "case", "")
    names = sort(unique([[r.case for r in PLAN.rows]; [r.variant for r in PLAN.rows]]))
    want = isempty(sel) ? names : cli_subset(ARGS, "case", names; what = "case")
    cli_shard(ARGS, [r for r in rows if r.case in want || r.variant in want])   # --shard=k/n: array task k of n (configs/reproduce.toml)
end

println(">> the ablation's $(length(unique(r.case for r in PLAN.rows))) cases × $(length(PLAN.modes)) modes ($(join(PLAN.modes, ", "))) on the chrome sets $(join(sort(collect(keys(PLAN.sets))), ", "))")
for r in PLAN.rows
    println("   ", rpad(r.label, 22), " ", rpad(r.chrome, 14), " ", r.variant)
end

record_dirs() = (; evaluate = Dict(k => product_dir(warping_evaluate_product(CFG, PLAN, k)) for k in keys(PLAN.sets)),
                 refine = Dict(r.variant => product_dir(refine_product(PLAN.rc, r.variant)) for r in PLAN.rows),
                 start = Dict(r.variant => product_dir(start_product(PLAN.rc, r.variant)) for r in PLAN.rows))
"Stage record: every scoring read (produced on demand: `warping_evaluate`), the record warping_ablation.json written last, its table before it."
record(p) = produce(p; R...) do dir
    for k in sort(collect(keys(PLAN.sets)))
        produce_upstream(warping_evaluate_product(CFG, PLAN, k), "the scoring of chrome set $k";
                         make = () -> warping_evaluate(CFG, PLAN, k; R...), command = driver_run(DRIVER, CFG_PATH, "evaluate").command)
    end
    d = record_dirs()
    rec = warping_record(CFG, PLAN, d)                                   # its schema the table's: EXPERIMENT_RECORD_SCHEMAS.warping_ablation
    VgtkIO.write_record(joinpath(dir, WARPING_RECORD_FILE * ".tmp.json"), rec)                       # read back through IO: the table is the record's
    write_tsv(joinpath(dir, WARPING_TABLE_FILE), WARPING_TABLE_COLUMNS, warping_table_rows(load_warping_record(joinpath(dir, WARPING_RECORD_FILE * ".tmp.json"))))
    mv(joinpath(dir, WARPING_RECORD_FILE * ".tmp.json"), joinpath(dir, WARPING_RECORD_FILE); force = true)   # the output, written last
    for w in rec.winners
        @printf(">> %-8s %-8s start %.4f  best %-8s %.4f px\n", w.prefix, w.family, w.start_median_px, w.best, w.best_median_px)
    end
    Produced(; inputs = [collect(values(d.evaluate)); collect(values(d.refine)); collect(values(d.start))])
end
"Stage figure: the record's figure and table, the style sheet and the preview (plotsdir(<asset_dir>/))."
figure(p) = produce_figures(p, String[]; R..., inputs = [product_dir(warping_record_product(CFG, PLAN))]) do dir
    rp = warping_record_product(CFG, PLAN)
    r = load_warping_record(joinpath(produce_upstream(rp, "the warping ablation's record"; make = () -> record(rp),
                                                      command = driver_run(DRIVER, CFG_PATH, "record").command), WARPING_RECORD_FILE))
    f = write_warping_figure(joinpath(dir, "assets", CFG["figure"]["asset_dir"]), r, CFG["figure"])
    cp(f.fragment, joinpath(dir, "figure.tex"); force = true)                 # the output, written last
    println(">> figure → ", f.fragment, ", table → ", f.table)
end

"The GPU stages: the upstreams (Schöps's runs, the chrome detect) and the refinements (bin/refine.jl's compute.backend)."
const GPU_STAGES = [:upstream; gpu_stages(:refine => [PLAN.rc["cases"][r.variant] for r in ROWS])]
const STAGES = [
    Stage(:upstream, "every upstream of each chrome set's refinements and scoring (decodes, front ends' runs, image regions, chrome detect and index, start models)";
          items = () -> CHROMES, product = k -> index_product(PLAN.sets[k]), run = k -> warping_upstreams(CFG, PLAN, k; R...)),
    Stage(:refine, "every case's refinement in every mode (the refinement driver's start and refine products; one per GPU)";
          items = () -> [r.variant for r in ROWS], product = v -> refine_product(PLAN.rc, v), run = v -> warping_refine(CFG, PLAN, v; R...)),
    Stage(:evaluate, "every chrome set's scoring of its starts and every mode's refinement (the chrome scorer; paired bootstrap against the reference and the start)";
          items = () -> CHROMES, product = k -> warping_evaluate_product(CFG, PLAN, k), run = k -> warping_evaluate(CFG, PLAN, k; R...)),
    Stage(:record, "the record warping_ablation.json and its table warping_ablation.tsv";
          items = () -> [warping_record_product(CFG, PLAN)], product = identity, run = record),
    Stage(:figure, "the pgfplots figure, the booktabs table, figure_styles.tex and preview.tex (plotsdir(<asset_dir>/))";
          items = () -> [warping_figure_product(CFG, PLAN)], product = identity, run = figure),
]

run_stages("warping_ablation", STAGES)
