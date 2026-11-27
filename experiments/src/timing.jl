# =============================================================================
# src/timing.jl — the timing experiment (configs/supplement/timing.toml; scripts/timing.jl): the wall time of one radiometric
# refinement of record, from its loaded start to its written calibrated camera record, replicated one job per GPU. Included after
# src/drivers.jl and src/radiometric_refinement.jl.
#
# The measurement is bin/refine.jl's own (`--repeat`): its `TIMING r/N key=value …` lines (the RadiometricCalibration solve and
# save_radiometric_calibration wall seconds, LM iterations and seconds) and its `PARAMETERS key=value …` line (the LM's free
# parameters per block), read from the replicate's refine.log — one code path, the run of record's program.
# =============================================================================
using Statistics: std
import Printf
import JSON

"The tables of configs/supplement/timing.toml."
const TIMING_TABLES = ("experiment", "inputs", "upstream", "timing", "overrides", "record")
"The keys of [timing]."
const TIMING_KEYS = ("case", "replicates", "warmup")
"The keys of [record]."
const TIMING_RECORD_KEYS = ("macro_prefix", "seconds_digits", "label", "runtime_table_asset", "table_asset", "statement_asset")
"A replicate's record (the time product's output) and the record of the experiment (the figures product's)."
const TIMING_REPLICATE_FILE = "timing.json"
const TIMING_RECORD_FILE = "timing_record.json"

"""
    load_timing_config(path) -> Dict

The timing config at `path`, every table and key checked: [timing] `case` (a case of [upstream] radiometric_refinement),
`replicates` ≥ 2 (a standard deviation needs two), `warmup` ≥ 1 (a timed solve after a compiled one: the first solve of a process
compiles); [overrides] `"<table>.<key>" = value` (a
single value each: refine-only keys, the start read by its key — `timing_case` refuses an override that re-keys the start).
"""
function load_timing_config(path::AbstractString)
    c = load_driver_config(path, "timing"; tables = TIMING_TABLES)
    t = c["timing"]; r = c["record"]
    VgtkCore.check_keys(t, TIMING_KEYS, "$path [timing]"); check_required_keys(t, TIMING_KEYS, "$path [timing]")
    VgtkCore.check_keys(r, TIMING_RECORD_KEYS, "$path [record]"); check_required_keys(r, TIMING_RECORD_KEYS, "$path [record]")
    t["case"] isa AbstractString || throw(ArgumentError("$path [timing] case: a case name"))
    t["replicates"] isa Integer && t["replicates"] >= 2 || throw(ArgumentError("$path [timing] replicates: an integer ≥ 2 (a standard deviation needs two)"))
    t["warmup"] isa Integer && t["warmup"] >= 1 || throw(ArgumentError("$path [timing] warmup: an integer ≥ 1 (the first solve of a process compiles)"))
    o = get(c, "overrides", Dict{String, Any}())
    o isa AbstractDict && all(v -> !(v isa AbstractVector), values(o)) ||
        throw(ArgumentError("$path [overrides]: each entry is `\"<table>.<key>\" = <one value>` (the timed case is one variant)"))
    haskey(c["upstream"], "radiometric_refinement") || throw(ArgumentError("$path [upstream]: radiometric_refinement is required"))
    return c
end

"""
    timing_case(cfg) -> (; rc, rc_path, case)

The refinement config `rc` (at `rc_path`) with the timed case: [timing] case as it is there, or with [overrides] set in its [pipeline]
(`_set_override!`, the experiments' one setter) under the SAME name — a case's name keys its start (`start_product`), so the timed case
keeps the case's own start, which the refinement driver produces. An override that re-keys the start (a start setting) is
refused: the timing times a refinement from the case's own start.
"""
function timing_case(cfg::AbstractDict)
    rc_path = projectdir(cfg["upstream"]["radiometric_refinement"])
    rc = load_radiometric_refinement_config(rc_path)
    base = String(cfg["timing"]["case"])
    haskey(rc["cases"], base) || throw(ArgumentError("[timing] case: $(repr(base)) is no case of $(cfg["upstream"]["radiometric_refinement"])"))
    rc["cases"][base]["kind"] == "grid" ||
        throw(ArgumentError("[timing] case: $base writes a $(rc["cases"][base]["kind"]) model; the timing counts the per-tile (spline) problem's parameters (bin/refine.jl PARAMETERS)"))
    k0 = start_product(rc, base).key
    ov = get(cfg, "overrides", Dict{String, Any}())
    if !isempty(ov)
        rc["cases"][base] = deepcopy(rc["cases"][base])
        foreach(kx -> _set_override!(rc["cases"][base]["pipeline"], kx...), ov)
        start_product(rc, base).key == k0 ||
            throw(ArgumentError("[overrides] re-key the start of $base: the timing times a refinement from the case's own start (refine-only keys only)"))
    end
    return (; rc, rc_path, case = base)
end

"""
    timing_product(cfg, tc, r) -> Product

Replicate `r` of the timed refinement (`tc = timing_case(cfg)`): keyed by [timing], the replicate, the case's refine-only settings
(`refinement_settings`) and its start's key; output timing.json (`timing_replicate_record`).
"""
function timing_product(cfg::AbstractDict, tc, r::Integer)
    rec = Dict{String, Any}("experiment" => cfg["experiment"], "timing" => cfg["timing"], "replicate" => Int(r),
                            "refine" => Dict(tc.case => refinement_settings(tc.rc["cases"][tc.case])),
                            "upstream" => Dict{String, Any}("start" => start_product(tc.rc, tc.case).key))
    return driver_product("time", rec; output = TIMING_REPLICATE_FILE)
end

# `key=value` fields of a log line after its tag (`TIMING 2/2 …`, `PARAMETERS …`)
_log_fields(line::AbstractString) = Dict{String, String}(String(k) => String(v) for (k, v) in (split(f, '='; limit = 2) for f in split(line) if occursin('=', f)))

"""
    parse_refine_timing(log) -> (; solves, parameters)

bin/refine.jl's timing lines of `log` (a `--repeat` run): `solves`, one NamedTuple per `TIMING r/N` line in order (`refine_s`,
`save_s`, `wall_s` = their sum, `iterations`, `lm_seconds`, `refine_compile_s`, `save_compile_s`, `refine_gc_s`), and `parameters`,
the `PARAMETERS` line's fields (`Dict`). An error when the log has no TIMING line or no PARAMETERS line.
"""
function parse_refine_timing(log::AbstractString)
    lines = readlines(log)
    t = [l for l in lines if startswith(l, "TIMING ")]
    p = [l for l in lines if startswith(l, "PARAMETERS ")]
    isempty(t) && throw(ArgumentError("$log: no TIMING line (bin/refine.jl --repeat)"))
    isempty(p) && throw(ArgumentError("$log: no PARAMETERS line (bin/refine.jl --repeat=N with N > 1)"))
    solves = map(t) do l
        f = _log_fields(l); x(k) = parse(Float64, f[k])
        (; refine_s = x("refine_s"), save_s = x("save_s"), wall_s = x("refine_s") + x("save_s"), iterations = parse(Int, f["iterations"]),
           lm_seconds = x("lm_seconds"), refine_compile_s = x("refine_compile_s"), save_compile_s = x("save_compile_s"), refine_gc_s = x("refine_gc_s"))
    end
    return (; solves, parameters = _log_fields(only(p)))
end

"""
    timing_replicate_record(log, run_dir; replicate, warmup, gpu, host, threads) -> NamedTuple

A replicate's record from its refine.log: the `warmup` untimed solves and the one timed solve (the last; `parse_refine_timing`), the
LM's free parameters, the problem's size from the run's parameters.json (its photographs, their resolution, the samples per photograph
and their total, the observations of one iteration), and where it ran (`gpu`, `host`, `threads`).
"""
function timing_replicate_record(log::AbstractString, run_dir::AbstractString; replicate::Integer, warmup::Integer, gpu::AbstractString,
                                 host::AbstractString, threads::AbstractString)
    t = parse_refine_timing(log)
    length(t.solves) == warmup + 1 || throw(ArgumentError("$log: $(length(t.solves)) solves, expected $(warmup + 1) (warmup $warmup + 1 timed)"))
    pj = JSON.parsefile(joinpath(run_dir, "parameters.json"))
    views = length(pj["views"]); per_view = Int(pj["samples"]["per_view"])
    P = t.parameters; i(k) = parse(Int, P[k])
    camera = i("lens") + i("focus") + i("aperture") + i("vignetting")
    return (; replicate = Int(replicate), warmup = t.solves[1:warmup], timed = last(t.solves),
            parameters = (; camera, lens = i("lens"), focus = i("focus"), aperture = i("aperture"), vignetting = i("vignetting"),
                          poses = i("poses"), target = i("material"), lighting = i("lighting"), total = camera + i("poses") + i("material") + i("lighting"),
                          grid_nodes_present = i("grid_nodes_present"), lens_solver_coordinates = i("lens_solver_coordinates")),
            problem = (; views, image_size = Int.(pj["image_size"]), samples_per_view = per_view, observations = i("samples")),
            gpu = String(gpu), host = String(host), threads = String(threads))
end

"""
    load_timing_replicate(path) -> NamedTuple

A replicate's record (`timing_replicate_record`'s fields) read back through VisualGeometryToolkitIO.
"""
function load_timing_replicate(path::AbstractString)
    r = VgtkIO.read_record(path)
    r["schema"] == EXPERIMENT_RECORD_SCHEMAS.timing || throw(ArgumentError("$path: schema $(r["schema"]), not $(EXPERIMENT_RECORD_SCHEMAS.timing)"))
    timing_entry(d) = (; refine_s = Float64(d["refine_s"]), save_s = Float64(d["save_s"]), wall_s = Float64(d["wall_s"]), iterations = Int(d["iterations"]),
                lm_seconds = Float64(d["lm_seconds"]), refine_compile_s = Float64(d["refine_compile_s"]), save_compile_s = Float64(d["save_compile_s"]),
                refine_gc_s = Float64(d["refine_gc_s"]))
    P = r["parameters"]; q = r["problem"]
    return (; schema = String(r["schema"]), replicate = Int(r["replicate"]), warmup = [timing_entry(w) for w in r["warmup"]], timed = timing_entry(r["timed"]),
            parameters = NamedTuple(Symbol(k) => Int(P[String(k)]) for k in (:camera, :lens, :focus, :aperture, :vignetting, :poses, :target,
                                                                              :lighting, :total, :grid_nodes_present, :lens_solver_coordinates)),
            problem = (; views = Int(q["views"]), image_size = Int.(q["image_size"]), samples_per_view = Int(q["samples_per_view"]),
                       observations = Int(q["observations"])),
            gpu = String(r["gpu"]), host = String(r["host"]), threads = String(r["threads"]))
end

"""
    timing_summary(records) -> NamedTuple

The statistics of the timed solves of the replicate `records` (`timing_replicate_record`'s): n, mean, sample std, min and max of the
wall seconds (the refinement and the write), of the refinement and of the write alone, and of the warm-up solves; the seconds per
iteration (the timed wall over its iterations); the problem, the parameters, the GPUs and the thread count (one each, else an error:
the replicates time one problem on one kind of hardware).
"""
function timing_summary(records)
    stats(x) = (; n = length(x), mean = mean(x), std = std(x), min = minimum(x), max = maximum(x))
    one(f, what) = (u = unique(f.(records)); length(u) == 1 || throw(ArgumentError("the replicates differ in $what: $u")); only(u))
    its = one(r -> r.timed.iterations, "LM iterations")
    wall = [r.timed.wall_s for r in records]
    return (; n = length(records), wall_s = stats(wall), refine_s = stats([r.timed.refine_s for r in records]),
            save_s = stats([r.timed.save_s for r in records]), warmup_wall_s = stats([w.wall_s for r in records for w in r.warmup]),
            seconds_per_iteration = stats([r.timed.lm_seconds / its for r in records]),   # the LM's own seconds per iteration (the write apart)
            iterations = its, problem = one(r -> r.problem, "problem"), parameters = one(r -> r.parameters, "parameters"),
            gpu = one(r -> r.gpu, "GPU model"), threads = one(r -> r.threads, "thread count"))
end

_tex_int(n::Integer) = Reproduce.tex_number(n; group_from = 4)
_sec(x, d) = Printf.format(Printf.Format("%.$(d)f"), x)

"""
    timing_table(records, s, digits) -> String

The per-job table (booktabs): one row per replicate — its GPU's host, the wall seconds of the timed solve (refinement + write), the
refinement and the write alone, the warm-up's wall seconds, the seconds per LM iteration — then the mean ± std row.
"""
function timing_table(records, s, digits::Integer)
    io = IOBuffer()
    println(io, "\\begin{tabular}{@{}rrrrrr@{}}"); println(io, "\\toprule")
    println(io, "Run & Refinement + write [s] & Refinement [s] & Write [s] & Warm-up [s] & Per iteration [s] \\\\"); println(io, "\\midrule")
    for r in sort(collect(records); by = r -> r.replicate)
        println(io, join((string(r.replicate), _sec(r.timed.wall_s, digits), _sec(r.timed.refine_s, digits), _sec(r.timed.save_s, digits),
                          _sec(mean(w.wall_s for w in r.warmup), digits), _sec(r.timed.lm_seconds / r.timed.iterations, 2)), " & "), " \\\\")
    end
    println(io, "\\midrule")
    pm(x, d) = "\$$(_sec(x.mean, d)) \\pm $(_sec(x.std, d))\$"
    println(io, join(("Mean \\(\\pm\\) std", pm(s.wall_s, digits), pm(s.refine_s, digits), pm(s.save_s, digits), pm(s.warmup_wall_s, digits),
                      pm(s.seconds_per_iteration, 2)), " & "), " \\\\")
    println(io, "\\bottomrule"); println(io, "\\end{tabular}")
    return String(take!(io))
end

# an integer as the paper's tables print it: thousands separated by commas (40,999)
_thousands(n::Integer) = (d = string(abs(n)); (n < 0 ? "-" : "") * join(reverse([reverse(c) for c in Iterators.partition(reverse(d), 3)]), ","))

"""
    runtime_table(s, label, digits) -> String

The supplement's runtime table in the paper's table style (booktabs, \\footnotesize, `tabular*` at column width, the data set as an
underlined italic group row, numbers typed as the paper's tables type them — a table carries its numbers, never macros): two columns,
key and value — the photographs and their resolution, the samples per photograph, the observations per iteration, the free
parameters (camera / poses / target / lighting), the LM iterations, the time per iteration, the refinement time (mean ± std over the
runs, min–max: the refinement and the calibrated camera's write, from its loaded start) and the GPU with the CPU threads.
"""
function runtime_table(s, label::AbstractString, digits::Integer)
    P = s.parameters; q = s.problem; w = s.wall_s
    rows = [("Photographs", "$(_thousands(q.views)) at \$$(q.image_size[1])\\times$(q.image_size[2])\$\\,px"),
            ("Samples per photograph", _thousands(q.samples_per_view)),
            ("Observations per iteration", _thousands(q.observations)),
            ("Free parameters (camera / poses / target / lighting)", join(_thousands.((P.camera, P.poses, P.target, P.lighting)), " / ")),
            ("LM iterations", _thousands(s.iterations)),
            ("Time per iteration [s]", _sec(s.seconds_per_iteration.mean, 1)),
            ("Refinement time [s] (mean \$\\pm\$ std, min--max)",
             "$(_sec(w.mean, digits)) \$\\pm\$ $(_sec(w.std, digits)) ($(_sec(w.min, digits))--$(_sec(w.max, digits)))"),
            ("Runs", string(s.n)),
            ("GPU, CPU threads", "NVIDIA $(s.gpu), $(first(split(s.threads, ',')))")]
    io = IOBuffer()
    println(io, "\\footnotesize"); println(io, "\\setlength{\\tabcolsep}{1.5pt}")
    println(io, "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lr@{}}"); println(io, "\\toprule")
    println(io, "\\multicolumn{2}{@{}l}{\\underline{\\textit{$label}}} \\\\")
    foreach(((k, v),) -> println(io, "$k & $v \\\\"), rows)
    println(io, "\\bottomrule"); println(io, "\\end{tabular*}")
    return String(take!(io))
end

"""
    timing_numbers!(nums, s, prefix, digits) -> nums

The number macros of the summary `s`: the mean and std of the timed wall seconds, the seconds per iteration, the warm-up, the runs,
the photographs, their resolution, the samples per photograph, the observations per iteration, the iterations, the free parameters
(camera, poses, target, lighting, total), the GPU and the threads.
"""
function timing_numbers!(nums, s, prefix::AbstractString, digits::Integer)
    m(parts...) = Reproduce.number_macro_name(prefix, parts...)
    add(n, v; kw...) = Reproduce.add_number!(nums, m(n...), v; kw...)
    add(("mean", "s"), s.wall_s.mean; digits, label = "timed refinement + write, mean wall seconds")
    add(("std", "s"), s.wall_s.std; digits, label = "timed refinement + write, sample std of the wall seconds")
    add(("min", "s"), s.wall_s.min; digits); add(("max", "s"), s.wall_s.max; digits)
    add(("per", "iteration", "s"), s.seconds_per_iteration.mean; digits = 1, label = "mean LM seconds per iteration (the LM's own clock; the write apart)")
    add(("warmup", "s"), s.warmup_wall_s.mean; digits, label = "the untimed warm-up solve (compile), mean wall seconds")
    add(("runs",), _tex_int(s.n)); add(("iterations",), _tex_int(s.iterations))
    add(("photographs",), _tex_int(s.problem.views))
    add(("width",), _tex_int(s.problem.image_size[1])); add(("height",), _tex_int(s.problem.image_size[2]))
    add(("samples",), _tex_int(s.problem.samples_per_view)); add(("observations",), _tex_int(s.problem.observations))
    for k in (:camera, :poses, :target, :lighting, :total)
        add(("parameters", String(k)), _tex_int(getfield(s.parameters, k)))
    end
    add(("gpu",), s.gpu); add(("threads",), first(split(s.threads, ',')))
    return nums
end

"""
    timing_statement(s, prefix) -> String

The supplement's one-sentence runtime statement, its numbers the macros of `timing_numbers!` (the prose cites them, never a re-typed value).
"""
function timing_statement(s, prefix::AbstractString)
    M(parts...) = "\\" * Reproduce.number_macro_name(prefix, parts...) * "{}"
    return "The radiometric refinement of the Sigma 14\\,mm Siemens-star set (" * M("photographs") * " photographs of " * M("width") *
           "\\,\$\\times\$\\," * M("height") * "\\,px, " * M("samples") * " samples each, " * M("observations") *
           " observations per iteration; " * M("iterations") * " LM iterations over " * M("parameters", "total") * " free parameters: " *
           M("parameters", "camera") * " of the camera, " * M("parameters", "poses") * " of the poses, " * M("parameters", "target") *
           " of the target and " * M("parameters", "lighting") * " of the lighting) takes " * M("mean", "s") * "\\,\$\\pm\$\\," * M("std", "s") *
           "\\,s (mean \$\\pm\$ std over " * M("runs") * " runs, the calibrated camera's write included; " * M("per", "iteration", "s") * "\\,s per LM iteration) on one NVIDIA " *
           M("gpu") * " with " * M("threads") * " CPU threads, from its loaded start; the decoding and the geometric start are excluded.\n"
end
