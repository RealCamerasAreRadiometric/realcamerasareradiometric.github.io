# =============================================================================
# src/material_grid.jl — the material start of a radiometric refinement: stage material_grid of scripts/radiometric_refinement.jl,
# between start and refine. The flux is linear in the lighting and the vignetting and affine in μ, so the start solves them in
# closed form; it is not linear in the roughnesses α and the indices of refraction η of the print (each of the ink and of the blank
# print), so their start is searched: for every candidate (α_ink, α_¬ink, η_ink, η_¬ink) of a coarse uniform grid over the physical
# range — `[material_grid]` of the config of record: per quantity its range and the number of equal cells it is cut into, a
# candidate at each cell's midpoint — the start's own closed-form solves run once (the lighting with V ≡ 1, then the lighting, the
# vignetting and μ), and the candidate of the lowest cost is the material the refinement starts from, with its lighting, its
# vignetting and its μ.
#
# The stage's product is a START RECORD (RadiometricCamera's start/start_record.jl) of the chosen material: the case's start record
# with the chosen candidate's state, costs and μ, its start options the case's with the chosen α and η — what bin/refine.jl
# --stage=start writes for that material — beside material_grid.json, the grid, every candidate's costs and the chosen one.
# The refinement reads it with --start (`run_refine`), its [pipeline] at the chosen α and η (`with_material_start`), so the
# refinement's own records state the material it started from. scripts/material_grid.jl is the stage's child process.
#
# The candidates are independent of each other, so a run may split them over several GPUs (the driver's --grid-parts=n): stage
# material_grid_part runs the same solves at part k of n of a case's candidates (every n-th from the k-th; a keyed product per case
# and part, a GPU job each with the driver's --shard) and writes their rows, material_grid_part.json. Stage material_grid then reads
# the n parts' rows instead of solving at every candidate, solves once more at the chosen candidate alone — its state is what the
# start record holds, and its row must be its part's number for number — and writes the same records under the same key.
# Included at the end of src/drivers.jl.
# =============================================================================
import VisualRadiometryToolkitCore as VrtkCore
using Printf: @printf

"""The keys of `[material_grid]`, every one required: per quantity — the roughness α and the index of refraction η, each of the ink
and of the blank print — its range `[lo, hi]` and the number of equal cells the range is cut into."""
const MATERIAL_GRID_KEYS = ("alpha_range", "alpha_cells", "eta_range", "eta_cells")
"The `[material]` options a candidate replaces: the start of α and of η, of the ink and of the blank print."
const MATERIAL_START_KEYS = ("alpha_ink", "alpha_noink", "eta_ink", "eta_noink")

"""
    check_material_grid(t, where) -> t

A `[material_grid]` table: exactly `MATERIAL_GRID_KEYS`, every one required. A range is `[lo, hi]` with `lo < hi` inside the
quantity's physical range (a roughness in [0, 1], an index of refraction ≥ 1; a candidate is a cell's midpoint, never a bound, so
α > 0 and η > 1 as the refinement's charts need); the cells a positive integer. An `ArgumentError` naming `where` otherwise.
"""
function check_material_grid(t, where::AbstractString)
    check_required_keys(t, MATERIAL_GRID_KEYS, where)
    for (q, lo, hi) in (("alpha", 0.0, 1.0), ("eta", 1.0, Inf))
        r = t["$(q)_range"]; n = t["$(q)_cells"]
        r isa AbstractVector && length(r) == 2 && all(x -> x isa Real && !(x isa Bool) && isfinite(x), r) && lo <= r[1] < r[2] <= hi ||
            throw(ArgumentError("$where $(q)_range: `[lo, hi]` with $lo ≤ lo < hi$(isfinite(hi) ? " ≤ $hi" : ""), got $(repr(r))"))
        n isa Integer && !(n isa Bool) && n >= 1 || throw(ArgumentError("$where $(q)_cells: a positive integer, got $(repr(n))"))
    end
    return t
end

"""
    material_grid_table(cfg, path) -> Dict

The `[material_grid]` of the refinement config `cfg` (at `path`), checked (`check_material_grid`): its own keys, or
`source = "<config>"` alone (project-relative) — that config's own table, so the configs of record state the grid in one place, as
`[kinds] source` does for the samples per view. A config without the table is refused: every refinement states the grid its
material start is searched on. The resolved table is returned, the values that key a case's product whichever file states them.
"""
function material_grid_table(cfg::AbstractDict, path::AbstractString)
    t = get(cfg, "material_grid", nothing)
    t isa AbstractDict || throw(ArgumentError("$path: no [material_grid] table — every refinement states the grid its material start is searched on " *
                                              "($(join(MATERIAL_GRID_KEYS, ", ")), or `source`: the config that states them)"))
    where = "$path [material_grid]"
    if haskey(t, "source")
        length(t) == 1 || throw(ArgumentError("$where: `source` names another config's [material_grid]; nothing beside it"))
        src = String(t["source"])
        t = get(TOML.parsefile(projectdir(src)), "material_grid", nothing)
        t isa AbstractDict && !haskey(t, "source") || throw(ArgumentError("$where source: $src states no [material_grid] of its own"))
        where = "$src [material_grid] (the source of $path)"
    end
    check_material_grid(t, where)
    return Dict{String, Any}(k => t[k] for k in MATERIAL_GRID_KEYS)
end

"""
    cell_midpoints(range, cells) -> Vector{Float64}

The candidates of one quantity on its uniform grid: the midpoints of the `cells` equal cells of `range = [lo, hi]`.
"""
function cell_midpoints(range, cells::Integer)
    lo = Float64(range[1]); hi = Float64(range[2])
    return [lo + (hi - lo) * (2i - 1) / (2cells) for i in 1:cells]
end

"""
    material_candidates(grid) -> Vector{NamedTuple}

Every candidate `(; alpha_ink, alpha_noink, eta_ink, eta_noink)` of the `[material_grid]` table `grid` (the names of the `[material]`
options a candidate replaces, `MATERIAL_START_KEYS`): the α midpoints for the ink times those for the blank print times the η
midpoints for each, the ink's α running fastest. The order is the record's rows'; of candidates of equal cost the first is kept.
"""
function material_candidates(grid::AbstractDict)
    αs = cell_midpoints(grid["alpha_range"], grid["alpha_cells"]); ηs = cell_midpoints(grid["eta_range"], grid["eta_cells"])
    return vec([(; alpha_ink, alpha_noink, eta_ink, eta_noink) for alpha_ink in αs, alpha_noink in αs, eta_ink in ηs, eta_noink in ηs])
end

"""
    material_sweep!(s, opts, candidate) -> (; costs, mu, vignetting_gcv)

The start's own closed-form solves at the material `candidate` (RadiometricCamera's start, start/prepare.jl: the same solves with
the same settings on the same pixels): the state's α and η set to the candidate's and μ to the options' start (`material.mu`), V
flat, the parts rendered once, the lighting solved with V ≡ 1 (`costs.init`), then one sweep of the lighting, the vignetting and μ
(`costs.linear`; μ held when the options fix the material). The lighting solve is a full linear solve, so the lighting the state
came with is not read. Each cost is the package's `(½ Σ r², relative RMS)` of the state's pixels (Core's `cost`); `mu` the solved μ;
`vignetting_gcv` the vignetting smoothness's selection when its weight is chosen by GCV. The state is left at the solved lighting,
vignetting and μ.
"""
function material_sweep!(s::RadiometricCamera.CalibrationState, opts::RadiometricCamera.PipelineOptions, candidate)
    s.mat = VrtkCore.Material(; roughness_ink = candidate.alpha_ink, roughness_paper = candidate.alpha_noink, refractive_index_ink = candidate.eta_ink,
                              refractive_index_paper = candidate.eta_noink, diffuse_weight = opts.material.mu)
    v = s.camera.vignetting
    RadiometricCamera.set_vignetting!(s, VrtkCore.with_control_points(v, ones(size(VrtkCore.control_points(v)))))
    parts = RadiometricCamera.all_parts(s)
    RadiometricCamera.solve_linear_blocks!(s; parts, light = true, vignetting = false, mu = false)
    init = VgtkCore.cost(s)
    gcv = NamedTuple[]
    RadiometricCamera.solve_linear_blocks!(s; parts, light = true, vignetting = true, mu = !("material" in opts.fixed), joint = opts.sweep.joint,
                                           vignetting_smoothness = RadiometricCamera.smoothness_weight(opts.vignetting.smoothness), gcv_record = gcv)
    parts = nothing; GC.gc()                                   # a candidate's parts are released before the next one's are rendered
    linear = VgtkCore.cost(s)
    return (; costs = (; init, linear), mu = s.mat.diffuse_weight, vignetting_gcv = gcv)
end

"""A candidate's row in the stage's records: its α and η, the solved μ, and the cost and relative RMS after the lighting solve
(`cost_init`, `rel_rms_init`) and after the sweep (`cost`, `rel_rms`: what the candidates are compared by)."""
candidate_row(candidate, mu, cost_init, rel_rms_init, cost, rel_rms) = (; candidate..., mu, cost_init, rel_rms_init, cost, rel_rms)

"The index of the row of the lowest finite `cost` of `rows`, of equal ones the first (the search's rule); `nothing` when none is finite."
function lowest_cost(rows)
    best = nothing
    for (i, r) in enumerate(rows)
        isfinite(r.cost) && (best === nothing || r.cost < rows[best].cost) && (best = i)
    end
    return best
end

"""A start whose grid smoothness weight was chosen by GCV is refused (an `ArgumentError`): that choice was made on the linearization
at the start's own material, and the stage does not make it again."""
refuse_gcv_start(s::RadiometricCamera.CalibrationState) = s.start.grid_gcv === nothing ||
    throw(ArgumentError("the material grid: this start chose its grid smoothness weight by GCV on the linearization at its own material " *
                        "(grid.smoothness weight = \"gcv\"), which another material start changes: state the weight"))

"""
    material_sweeps!(s, opts, candidates, indices; io = stdout) -> (; rows, best)

[`material_sweep!`](@ref) at the candidates `indices` of `candidates`, in that order: `rows`, a row each (`candidate_row`), and
`best`, the one of the lowest cost after its sweep (of equal ones the first) as its solves left the state — its index, its sweep,
the material, the lighting and the vignetting — or `nothing` when no cost is finite. A line per candidate, with its seconds, goes
to `io`. A start whose grid smoothness weight was chosen by GCV is refused (`refuse_gcv_start`).
"""
function material_sweeps!(s::RadiometricCamera.CalibrationState, opts::RadiometricCamera.PipelineOptions, candidates, indices; io::IO = stdout)
    refuse_gcv_start(s)
    rows = NamedTuple[]; best = nothing
    for i in indices
        c = candidates[i]; t = time()
        r = material_sweep!(s, opts, c)
        push!(rows, candidate_row(c, r.mu, r.costs.init..., r.costs.linear...))
        lowest = isfinite(r.costs.linear[1]) && (best === nothing || r.costs.linear[1] < best.sweep.costs.linear[1])
        lowest && (best = (; index = i, sweep = r, material = s.mat, light = s.light, vignetting = s.camera.vignetting))
        @printf(io, "%4d/%d  α %.4f / %.4f  η %.4f / %.4f  μ %.4f  rel-RMS %.5f (lighting alone %.5f)  cost %.6e  (%.1f s)%s\n", i, length(candidates),
                c.alpha_ink, c.alpha_noink, c.eta_ink, c.eta_noink, r.mu, r.costs.linear[2], r.costs.init[2], r.costs.linear[1], time() - t, lowest ? "  *" : ""); flush(io)
    end
    return (; rows, best)
end

"""
    material_grid_search!(s, opts, grid; io = stdout, rows = nothing) -> (; start, candidates, chosen)

The material start of the started state `s` (a start record's, `load_calibration_start`, of the start options `opts`) on the
`[material_grid]` table `grid`: [`material_sweep!`](@ref) at every candidate (`material_candidates`), and the candidate of the
lowest cost after its sweep kept — the state left at its material, lighting, vignetting and μ, and the state's record of its start
(`s.start`: the costs and μ a start record states) the chosen candidate's. Returns the stage's record: `start`, the material the
start record was computed at with its costs (the config's own `[material]`, for comparison); `candidates`, a row each in their
order; `chosen`, the kept row's index. A line per candidate, with its seconds, goes to `io`.

With `rows`, every candidate's row as its part computed it (`read_material_grid_parts`), no candidate is solved but the chosen
one (`lowest_cost`), whose solves leave the state the start record holds: its row must be the given one number for number (the
solves read nothing of the state they start from), an `ArgumentError` naming both otherwise — rows of another start, or of
another machine's arithmetic, are not this search's.

A start whose grid smoothness weight was chosen by GCV is refused (`refuse_gcv_start`).
"""
function material_grid_search!(s::RadiometricCamera.CalibrationState, opts::RadiometricCamera.PipelineOptions, grid::AbstractDict; io::IO = stdout, rows = nothing)
    refuse_gcv_start(s)
    meta = s.start
    m = opts.material
    start = (; alpha_ink = m.alpha_ink, alpha_noink = m.alpha_noink, eta_ink = m.eta_ink, eta_noink = m.eta_noink, mu = meta.μ_lin,
               cost_init = meta.costs.init[1], rel_rms_init = meta.costs.init[2], cost = meta.costs.linear[1], rel_rms = meta.costs.linear[2])
    candidates = material_candidates(grid)
    @printf(io, "material grid: %d candidates (α %s; η %s) on %d samples of %d views; the start's own material α %g / %g, η %g / %g: rel-RMS %.5f\n",
            length(candidates), join(cell_midpoints(grid["alpha_range"], grid["alpha_cells"]), ", "), join(cell_midpoints(grid["eta_range"], grid["eta_cells"]), ", "),
            sum(length, s.px), length(s.px), m.alpha_ink, m.alpha_noink, m.eta_ink, m.eta_noink, start.rel_rms); flush(io)
    if rows === nothing
        (; rows, best) = material_sweeps!(s, opts, candidates, eachindex(candidates); io)
    else
        length(rows) == length(candidates) || throw(ArgumentError("the material grid: $(length(rows)) rows for its $(length(candidates)) candidates"))
        i = lowest_cost(rows); best = nothing
        if i !== nothing
            println(io, "the candidates' rows are their parts': the solves run at the chosen candidate alone"); flush(io)
            swept = material_sweeps!(s, opts, candidates, i:i; io)
            only(swept.rows) == rows[i] ||
                throw(ArgumentError("the material grid: candidate $i's solves give $(only(swept.rows)), its part's row is $(rows[i]): the parts are not " *
                                    "this start's solves on this machine (produce the stage without --grid-parts)"))
            best = swept.best
        end
    end
    best === nothing && error("the material grid: no candidate of $(length(candidates)) has a finite cost")
    s.mat = best.material; s.light = best.light
    RadiometricCamera.set_vignetting!(s, best.vignetting)
    s.start = (; meta..., costs = best.sweep.costs, μ_lin = best.sweep.mu, vignetting_gcv = best.sweep.vignetting_gcv)
    c = rows[best.index]
    @printf(io, "material start: candidate %d, α %.4f / %.4f, η %.4f / %.4f, μ %.4f: rel-RMS %.5f (the start's own material: %.5f)\n", best.index,
            c.alpha_ink, c.alpha_noink, c.eta_ink, c.eta_noink, c.mu, c.rel_rms, start.rel_rms); flush(io)
    return (; start, candidates = rows, chosen = best.index)
end

"""
    material_start_options(opts, candidate) -> PipelineOptions

The options `opts` with the candidate's α and η as their `[material]` start (`MATERIAL_START_KEYS`): the start options of the
stage's start record, and what the refinement from it runs on (`with_material_start`). μ stays the options': the value the
closed-form solves start from.
"""
function material_start_options(opts::RadiometricCamera.PipelineOptions, candidate)
    d = RadiometricCamera.config_dict(opts)
    d["material"] = merge(d["material"], Dict{String, Any}(k => Float64(getproperty(candidate, Symbol(k))) for k in MATERIAL_START_KEYS))
    return pipeline_options(d)
end

"The stage's record in its product `dir`: the grid, the start's own material and costs, every candidate's and the chosen one."
material_grid_record(dir::AbstractString) = joinpath(dir, "material_grid.json")

"""
    write_material_grid(dir, grid, result) -> path

The record of a material grid search (`material_grid_search!`'s `result` on the `[material_grid]` table `grid`) as
`material_grid_record(dir)`: the `grid`; `start`, the material the case's start record was computed at and its costs; `candidates`,
per candidate its α and η, the solved μ, and the cost and relative RMS after the lighting solve (`cost_init`, `rel_rms_init`) and
after the sweep (`cost`, `rel_rms`: what the candidates are compared by); `chosen`, the kept candidate's row with its index (from 1).
"""
function write_material_grid(dir::AbstractString, grid::AbstractDict, result)
    path = material_grid_record(dir)
    VgtkIO.write_record(path, (; schema = EXPERIMENT_RECORD_SCHEMAS.material_grid, grid = Dict{String, Any}(k => grid[k] for k in MATERIAL_GRID_KEYS),
                               start = result.start, candidates = result.candidates,
                               chosen = (; index = result.chosen, result.candidates[result.chosen]...)))
    return path
end

# ---- the candidates in parts ---------------------------------------------------------------------------------------------------------

"""
    material_grid_parts(argv = ARGS) -> Union{Nothing, Int}

The number of parts a run splits every case's candidates into, `--grid-parts=n` (an integer ≥ 2: a GPU job per part), or `nothing`
without the option: stage material_grid then solves at every candidate itself. The split changes no result and no key.
"""
function material_grid_parts(argv = ARGS)
    s = cli_option(argv, "grid-parts", "")
    isempty(s) && return nothing
    n = tryparse(Int, s)
    n !== nothing && n >= 2 || throw(ArgumentError("--grid-parts=$s: the number of parts the candidates are split into, an integer ≥ 2"))
    return n
end

"""
    part_candidates(count, part, parts) -> the indices

The candidates of part `part` of `parts` among `count`, by their indices (`1 ≤ part ≤ parts ≤ count`, an `ArgumentError`
otherwise): every `parts`-th from the `part`-th, so the parts hold every candidate once and as many each, to within one.
"""
function part_candidates(count::Integer, part::Integer, parts::Integer)
    1 <= part <= parts <= count || throw(ArgumentError("the material grid: part $part of $parts of $count candidates (1 ≤ part ≤ parts ≤ the candidates)"))
    return part:parts:count
end

"The record of a part of the candidates in its product `dir`: the candidates' rows, each with its index."
material_grid_part_record(dir::AbstractString) = joinpath(dir, "material_grid_part.json")

"""
    write_material_grid_part(dir, grid, part, parts, indices, rows) -> path

The record of part `part` of `parts` of the candidates of the `[material_grid]` table `grid` (`material_sweeps!`'s `rows` at the
candidates `indices`) as `material_grid_part_record(dir)`: the `grid`, the `part` and the `parts`, and `candidates`, per candidate
its `index` (from 1) and its row as the stage's own record states it (`write_material_grid`).
"""
function write_material_grid_part(dir::AbstractString, grid::AbstractDict, part::Integer, parts::Integer, indices, rows)
    path = material_grid_part_record(dir)
    VgtkIO.write_record(path, (; schema = EXPERIMENT_RECORD_SCHEMAS.material_grid_part, grid = Dict{String, Any}(k => grid[k] for k in MATERIAL_GRID_KEYS),
                               part, parts, candidates = [(; index, row...) for (index, row) in zip(indices, rows)]))
    return path
end

"""
    read_material_grid_parts(dirs, grid) -> Vector{NamedTuple}

Every candidate's row of the `[material_grid]` table `grid`, in the candidates' order, from the part products `dirs` (their
`material_grid_part.json`): what `material_grid_search!` takes as `rows`. An `ArgumentError` unless the records hold exactly one
row per candidate, each at its candidate's α and η — parts of another grid, a missing or a repeated part.
"""
function read_material_grid_parts(dirs, grid::AbstractDict)
    candidates = material_candidates(grid)
    rows = Vector{Union{Nothing, NamedTuple}}(nothing, length(candidates))
    for dir in dirs, d in VgtkIO.read_record(material_grid_part_record(dir))["candidates"]
        i = d["index"]
        i isa Integer && checkbounds(Bool, rows, i) && all(k -> d[k] == getproperty(candidates[i], Symbol(k)), MATERIAL_START_KEYS) ||
            throw(ArgumentError("the material grid's part $dir: its row of index $(repr(i)) is not a candidate of this grid"))
        rows[i] === nothing || throw(ArgumentError("the material grid's part $dir: candidate $i has a row of another part already"))
        rows[i] = candidate_row(candidates[i], (Float64(d[k]) for k in ("mu", "cost_init", "rel_rms_init", "cost", "rel_rms"))...)
    end
    missing_rows = findall(isnothing, rows)
    isempty(missing_rows) || throw(ArgumentError("the material grid's parts ($(join(dirs, ", "))) hold no row of the candidates $(join(missing_rows, ", "))"))
    return NamedTuple[r for r in rows]
end

"""
    with_material_start(table, dir) -> Dict

The `[pipeline]` table `table` with the α and η that the material grid's product `dir` chose (its record's `chosen`) as its
`[material]` start: the table of a refinement from that product. Its start options are then the product's start record's
(RadiometricCamera refuses a start of other options), and the refinement's records state the material it started from. μ stays the
table's, as in the record: the value the closed-form solves start from.
"""
function with_material_start(table::AbstractDict, dir::AbstractString)
    chosen = VgtkIO.read_record(material_grid_record(dir))["chosen"]
    p = deepcopy(Dict{String, Any}(table))
    m = get!(p, "material", Dict{String, Any}())
    foreach(k -> m[k] = Float64(chosen[k]), MATERIAL_START_KEYS)
    return p
end

# ---- the stage ----------------------------------------------------------------------------------------------------------------------

"""
    material_grid_product(cfg, case) -> Product

The material start of `[cases.<case>]`: keyed by the config's `[material_grid]`, the key of the case's start (`start_product`, which
holds every setting the solves read, its inputs and its views) and the schemas of its two records (the stage's own and
RadiometricCamera's start record). Its output: the start record's start.json, written last.
"""
function material_grid_product(cfg::AbstractDict, case::AbstractString)
    rec = Dict{String, Any}("experiment" => cfg["experiment"], "material_grid" => Dict{String, Any}(case => cfg["material_grid"]),
                            "upstream" => Dict{String, Any}("start" => start_product(cfg, case).key),
                            "record" => EXPERIMENT_RECORD_SCHEMAS.material_grid, "start_record" => RadiometricCamera.RECORD_SCHEMAS.refinement_start)
    return driver_product("material_grid", rec; output = RadiometricCamera.START_RECORD_FILE, experiment = variant_location(cfg, case))
end

"""
    material_grid_part_product(cfg, case, part, parts) -> Product

Part `part` of `parts` of the candidates of `[cases.<case>]`'s material grid (`part_candidates`): keyed as the case's material start
(`material_grid_product`: the config's `[material_grid]` and the key of the case's start), by the part and by its record's schema.
Its output: the part's rows, material_grid_part.json.
"""
function material_grid_part_product(cfg::AbstractDict, case::AbstractString, part::Integer, parts::Integer)
    rec = Dict{String, Any}("experiment" => cfg["experiment"], "material_grid" => Dict{String, Any}(case => cfg["material_grid"]),
                            "upstream" => Dict{String, Any}("start" => start_product(cfg, case).key), "part" => [part, parts],
                            "record" => EXPERIMENT_RECORD_SCHEMAS.material_grid_part)
    return driver_product("material_grid_part", rec; output = basename(material_grid_part_record("")), experiment = variant_location(cfg, case))
end

"""
    run_material_grid_search(table, grid, start_dir, run_dir; source, label, args = String[]) -> String

The stage's child process, scripts/material_grid.jl, on the start record `start_dir`: the `[pipeline]` table `table` (the case's
start settings, as its start read them) with `io.out = run_dir` and the `[material_grid]` table `grid` are written to
`run_dir/material_grid_input.toml` and run in a child process (this project's environment, this process's Julia and thread count:
the closed-form solves sum over thread chunks) whose working directory is data_root(), the directory the table's relative input
paths are relative to. `args`: the child's further arguments (`--part`, a part of the candidates alone; `--parts`, the candidates'
rows read from their parts). The child's output goes to `run_dir/material_grid.log`. Returns that log.
"""
function run_material_grid_search(table::AbstractDict, grid::AbstractDict, start_dir::AbstractString, run_dir::AbstractString;
                                  source::AbstractString, label::AbstractString, args = String[])
    assert_not_input(run_dir)
    mkpath(run_dir)
    t = deepcopy(Dict{String, Any}(table))
    get!(t, "io", Dict{String, Any}())["out"] = run_dir
    input = joinpath(run_dir, "material_grid_input.toml")
    open(input, "w") do io
        println(io, "# The material grid of $label from $source as scripts/material_grid.jl read it: [pipeline], the case's start settings with")
        println(io, "# io.out set by the driver (relative input paths are relative to data_root(), the working directory of the run), and the")
        println(io, "# config's [material_grid].")
        TOML.print(io, Dict{String, Any}("pipeline" => t, "material_grid" => grid); sorted = true)
    end
    threads = execution_threads()
    cmd = `$(Base.julia_cmd()) --threads=$threads --project=$(projectdir()) $(projectdir("scripts", "material_grid.jl")) --config=$input --start=$start_dir $(collect(String, args))`
    log = joinpath(run_dir, "material_grid.log")
    println(">> scripts/material_grid.jl: $label → ", run_dir, "   (threads ", threads, "; in data_root() = ", data_root(), "; log material_grid.log)")
    flush(stdout)
    ok = open(log, "w") do io
        success(pipeline(Cmd(cmd; dir = data_root()); stdout = io, stderr = io))
    end
    ok || error("scripts/material_grid.jl failed on $label: see $log\n" * join(last(readlines(log), 15), "\n"))
    return log
end

# The child of the stage's two products of a case (its material start, a part of its candidates) into `dir`: the case's start
# produced on demand, its photographs decoded, scripts/material_grid.jl on the case's start settings and the config's
# [material_grid] with `args`. Returns the inputs it read.
function material_grid_child(cfg::AbstractDict, case::AbstractString, dir::AbstractString, label::AbstractString, args; driver::AbstractString, config_path::AbstractString)
    t = cfg["cases"][case]
    sdir = produce_upstream(start_product(cfg, case), "the material grid of case $case: its start"; make = () -> run_start(cfg, case; driver, config_path),
                            command = driver_run(driver, config_path, "prepare,start"; case).command, gpu = runs_on_gpu(t))
    require_prepared(cfg, t, "the material grid of case $case"; driver, config_path)
    p = resolve_pipeline(cfg, start_settings(t); init_dir = init_start_dir(cfg, _start_init(cfg, case), case; driver, config_path))
    run_material_grid_search(p, cfg["material_grid"], sdir, dir; source = relpath(config_path, projectdir()), label, args)
    return [run_inputs_through(cfg, t, "start"); pipeline_input_paths(p); sdir]
end

"""
    run_material_grid_part(cfg, case, part, parts; driver, config_path) -> dir

Stage material_grid_part of `[cases.<case>]`: part `part` of `parts` of its candidates into its product
(`material_grid_part_product`), produced or loaded — from the case's start (produced on demand), scripts/material_grid.jl
`--part=<part>/<parts>` on the case's start settings and the config's `[material_grid]`.
"""
function run_material_grid_part(cfg::AbstractDict, case::AbstractString, part::Integer, parts::Integer; driver::AbstractString, config_path::AbstractString)
    g = material_grid_part_product(cfg, case, part, parts)
    produce(g; product_records(driver, config_path)...) do dir
        Produced(; inputs = material_grid_child(cfg, case, dir, "case $case, part $part of $parts of its candidates (product $(g.key))", ["--part=$part/$parts"];
                                                driver, config_path))
    end
end

"""
    run_material_grid(cfg, case; driver, config_path, parts = material_grid_parts()) -> dir

Stage material_grid of `[cases.<case>]` into its product (`material_grid_product`), produced or loaded: from the case's start
(produced on demand: its stage start, `run_start`), scripts/material_grid.jl on the case's start settings and the config's
`[material_grid]` (`run_material_grid_search`). With `parts` (the run's `--grid-parts`) the child reads the candidates' rows from
the case's `parts` part products (`--parts`), each produced on demand as the start is (its stage material_grid_part), and solves
at the chosen candidate alone; the product, its records and its key are the same.
"""
function run_material_grid(cfg::AbstractDict, case::AbstractString; driver::AbstractString, config_path::AbstractString, parts = material_grid_parts())
    g = material_grid_product(cfg, case); gpu = runs_on_gpu(cfg["cases"][case])
    produce(g; product_records(driver, config_path)...) do dir
        pdirs = parts === nothing ? String[] :
                [produce_upstream(material_grid_part_product(cfg, case, k, parts), "the material grid of case $case: part $k of $parts of its candidates";
                                  make = () -> run_material_grid_part(cfg, case, k, parts; driver, config_path), gpu,
                                  command = driver_run(driver, config_path, "prepare,start,material_grid_part"; case).command * " --grid-parts=$parts") for k in 1:parts]
        inputs = material_grid_child(cfg, case, dir, "case $case (product $(g.key))", isempty(pdirs) ? String[] : ["--parts=$(join(pdirs, ','))"]; driver, config_path)
        Produced(; inputs = [inputs; pdirs])
    end
end

"""The directory of the material grid's product of `[cases.<case>]`, the start record a refinement of the case reads, produced on
demand (its stage material_grid, [`run_material_grid`](@ref); on a GPU where the case's [pipeline] runs on one)."""
material_grid_dir(cfg::AbstractDict, case::AbstractString; driver::AbstractString, config_path::AbstractString) =
    produce_upstream(material_grid_product(cfg, case), "case $case: its material start"; make = () -> run_material_grid(cfg, case; driver, config_path),
                     command = driver_run(driver, config_path, "prepare,start,material_grid"; case).command, gpu = runs_on_gpu(cfg["cases"][case]))
