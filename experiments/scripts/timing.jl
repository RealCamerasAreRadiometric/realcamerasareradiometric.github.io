#!/usr/bin/env julia
# reproduce: reads configs/supplement/timing.toml
# reproduce: writes ../paper/fig/runtime-table.tex
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/timing.jl — the timing experiment (owner 2026-11-18, the supplement's runtime): the wall time of one radiometric refinement
# of record, from its loaded start and prepared photographs to its written calibrated camera record, replicated one job per GPU.
# Config of record: configs/supplement/timing.toml (a supplement experiment: its products under results_root()/supplement/timing/) (src/timing.jl).
#
# STAGES (each product keyed by only what it reads and the keys of the products it reads; <results>/supplement/timing/<stage>/<key>/)
#   start    the timed case's start: scripts/radiometric_refinement.jl (prepare, start) on [upstream] radiometric_refinement, in a
#            module of its own — that driver's own product, produced or loaded (UNTIMED: the decode and the initial guess)
#   time     per replicate (GPU, one job per GPU), bin/refine.jl --start=<the start> --repeat=<warmup + 1> on the case's [pipeline]:
#            the warm-up solves, then the timed one; its timing.json from bin/refine.jl's TIMING / PARAMETERS lines, the GPU, the host
#   record   the replicates gathered: timing_record.json, the supplement's runtime table (runtime-table.tex, the paper's table style),
#            the per-run table and the supplement's statement (plotsdir(timing/)) and the number macros (numbers.tex, the output)
#
#   julia --project=experiments experiments/scripts/timing.jl [--config=<toml>] [--plan] [--stage=start,time,record] [--replicate=1,2] [--force]
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("timing.jl"))
import CUDA

const DRIVER = "scripts/timing.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "supplement", "timing.toml")))   # default: the config of record
const CFG = load_timing_config(CFG_PATH)
const TC = timing_case(CFG)
const REPLICATES = parse.(Int, cli_subset(ARGS, "replicate", string.(1:CFG["timing"]["replicates"]); what = "replicate"))
const R = product_records(DRIVER, CFG_PATH)

"The case's start product (untimed): the refinement driver's prepare and start of the base case, produced or loaded by its key."
start_dir() = refinement_upstream(TC.rc, TC.rc_path, TC.case, "start", "the timing of $(TC.case)")

"The device this process holds (CUDA's name, the vendor's prefix dropped): the GPU the replicate's child process runs on."
gpu_model() = replace(CUDA.name(CUDA.device()), r"^NVIDIA\s+" => "")

"""Stage time of replicate `r`: bin/refine.jl from the case's start, `warmup` untimed solves and the timed one in one process, into the
replicate's product; its record timing.json (`timing_replicate_record`) written last."""
function run_time(r::Integer)
    p = timing_product(CFG, TC, r)
    produce(p; R...) do dir
        sdir = start_dir()
        t = TC.rc["cases"][TC.case]
        require_prepared(TC.rc, t, "the timed case $(TC.case)"; driver = REFINEMENT_DRIVER, config_path = TC.rc_path)
        require_lens_coverage(TC.rc, t, "the timed case $(TC.case)")
        idir = init_start_dir(TC.rc, _start_init(TC.rc, TC.case), TC.case; driver = REFINEMENT_DRIVER, config_path = TC.rc_path)
        pipe = resolve_pipeline(TC.rc, t; init_dir = idir)                 # run_refine's resolution (an [init] start's grid included)
        log = run_pipeline(pipe, dir; source = relpath(TC.rc_path, projectdir()), label = "timing replicate $r of $(TC.case) (product $(p.key))",
                           args = ["--start=$sdir", "--repeat=$(CFG["timing"]["warmup"] + 1)"])
        rec = timing_replicate_record(log, dir; replicate = r, warmup = CFG["timing"]["warmup"], gpu = gpu_model(), host = gethostname(),
                                      threads = execution_threads())
        VgtkIO.write_record(joinpath(dir, TIMING_REPLICATE_FILE), merge((; schema = EXPERIMENT_RECORD_SCHEMAS.timing), rec))
        println(">> replicate $r on $(rec.gpu) ($(rec.host)): $(round(rec.timed.wall_s; digits = 1)) s (refinement $(round(rec.timed.refine_s; digits = 1)) s + write $(round(rec.timed.save_s; digits = 1)) s), $(rec.timed.iterations) iterations")
        Produced(; inputs = [pipeline_input_paths(pipe); sdir])
    end
end

"Every replicate's product, in replicate order."
replicate_products() = [timing_product(CFG, TC, r) for r in 1:CFG["timing"]["replicates"]]

"The figures product (its output numbers.tex): keyed by [timing], [record] and the keys of every replicate's product."
figures_product() = driver_product("figures", Dict{String, Any}("experiment" => CFG["experiment"], "timing" => CFG["timing"], "record" => CFG["record"],
                                                                "replicates" => [q.key for q in replicate_products()]);
                                   output = "numbers.tex", threads = nothing)

"Stage record: every replicate's record (each produced on its own GPU job: refused here, with its command, when missing), the summary, the table, the statement and the macros."
function run_record(p)
    ps = replicate_products()
    produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH, inputs = String[]) do dir
        records = map(enumerate(ps)) do (r, q)
            d = produce_upstream(q, "timing replicate $r"; make = () -> run_time(r), command = driver_run(DRIVER, CFG_PATH, "time").command * " --replicate=$r", gpu = true)
            load_timing_replicate(joinpath(d, TIMING_REPLICATE_FILE))
        end
        s = timing_summary(records)
        rc = CFG["record"]; digits = Int(rc["seconds_digits"]); prefix = String(rc["macro_prefix"])
        for (asset, text) in ((rc["runtime_table_asset"], runtime_table(s, String(rc["label"]), digits)), (rc["table_asset"], timing_table(records, s, digits)),
                              (rc["statement_asset"], timing_statement(s, prefix)))
            f = joinpath(dir, "assets", asset); mkpath(dirname(f)); write(f, text)
        end
        VgtkIO.write_record(joinpath(dir, TIMING_RECORD_FILE), (; schema = EXPERIMENT_RECORD_SCHEMAS.timing_summary, case = TC.case, replicates = [q.key for q in ps], summary = s))
        Reproduce.write_numbers(joinpath(dir, "numbers.tex"), timing_numbers!(driver_numbers(CFG_PATH, DRIVER, p.key), s, prefix, digits))
        println(">> $(TC.case) on $(s.n) × $(s.gpu): $(round(s.wall_s.mean; digits = 1)) ± $(round(s.wall_s.std; digits = 1)) s " *
                "(min $(round(s.wall_s.min; digits = 1)), max $(round(s.wall_s.max; digits = 1))), $(s.iterations) iterations, " *
                "$(round(s.seconds_per_iteration.mean; digits = 2)) s/it; warm-up $(round(s.warmup_wall_s.mean; digits = 1)) s")
    end
end

"The time stage runs bin/refine.jl on the case's compute backend (the GPU of record); start runs the refinement driver's start there too."
const GPU_STAGES = gpu_stages(:start => [TC.rc["cases"][TC.case]], :time => [TC.rc["cases"][TC.case]])

const STAGES = [
    Stage(:start, "the timed case's start (scripts/radiometric_refinement.jl prepare, start: untimed)";
          items = () -> [TC.case], product = c -> start_product(TC.rc, c), run = c -> start_dir()),
    Stage(:time, "per replicate, the warm-up and the timed refinement + write from the start (bin/refine.jl --repeat; one job per GPU)";
          items = () -> REPLICATES, product = r -> timing_product(CFG, TC, r), run = run_time),
    Stage(:record, "the replicates gathered: timing_record.json, the per-job table, the statement, numbers.tex";
          items = () -> [figures_product()], product = identity, run = run_record),
]

run_stages("timing", STAGES)
