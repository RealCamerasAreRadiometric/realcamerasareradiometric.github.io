# ---------------------------------------------------------------------------
# The refinement's start as its own step (the owner's rule of 2026-11-14: every step of a pipeline caches its intermediate, keyed by
# ONLY what it reads). The start — the front end (the start lens on the grid, its initialization region, the PnP / page poses), the
# image circle, the masks, the vignetting centre, the focus start, the closed-form radiometric start and the GCV weights — is computed
# by the start (a `RadiometricCalibration`'s `init`, `_start_state`) from `start_options(opts)` (the refinement-only options at their defaults: a start never reads them) and written
# as a START RECORD, a directory; every refinement starts from one (`load_refinement_start` → `load_calibration_start(dir; options)`), so
# a start loaded and a start computed are one code path. bin/refine.jl: `--stage=start` writes it, `--start=<dir>` refines from it.
#
#   start.json            rc/refinement_start/v1.0: the views, the image size, fpx, R0, d_c, f_m, the self-check bearings, the
#                         front end's frame choice, the vignetting centre, the costs of the closed-form start, μ, the GCV selections,
#                         the non-central centre, the exact resample of a SplineLens start (`resample`, `GridStart`'s exact resample: its lattice,
#                         the extrapolated nodes, the folds of its extension and its fidelity to the source)
#   pipeline_config.toml  the start options (start_options; write_pipeline_config)
#   grid.json | lens.json the start's lens as the refinement starts from it (perturbed, its present nodes; save_grid / save_lens),
#                         beside a grid its initialization_region.json: what `load_start` and a later run's `io.init` read
#   frontend/grid.json | frontend/lens.json  the front end's lens with the run's image circle (before the perturbation and the
#                         present nodes): what the refinement's state is rebuilt from
#   state.json            the state after the closed-form start (save_state)
#   masks/<k>.png         each view's final mask (8-bit, 0 / 255)
# ---------------------------------------------------------------------------

"""
    REFINE_ONLY_KEYS

The [`PipelineOptions`](@ref) keys (`table.key`) that only the refinement reads (the `RadiometricCalibration` solve, after the start): the LM's iteration and
time caps and its stop tolerances, the per-iteration checkpoints, the closed-form sweeps between LM chunks, a resumed run's iteration offset and its sweep.
A start is computed from [`start_options`](@ref), which holds them at their defaults, so a change of one never changes (or
recomputes) a start. `resume.state` is not among them: the front end reads its page frame.
"""
const REFINE_ONLY_KEYS = ("optimizer.max_iter", "optimizer.max_time", "optimizer.ftol", "optimizer.xtol_rel", "optimizer.type", "io.checkpoint_each",
                          "sweep.iterations", "sweep.lm_chunk",
                          "resume.iteration_offset", "resume.sweep")

"""
    split_refine_only(table; keys = REFINE_ONLY_KEYS) -> (start, refine)

A `[pipeline]` table (or `config_dict`) split in two: `start`, the table without `keys` (the tables they leave empty dropped), and
`refine`, those of `keys` it states. THE split of a run's options into its start's and its refinement's: `start_options` and the
paper project's drivers (the start's and the refinement's keys) read it. `keys` are current keys, looked up as the table's version
names them ([`config_key`](@ref): a version-1 table, the configs of record, names the optimizer's table `[lm]`): the table is split as
it is written, so a driver's start and refinement keys of a version-1 table are the ones it always had.
"""
function split_refine_only(table::AbstractDict; keys = REFINE_ONLY_KEYS)
    start = deepcopy(Dict{String, Any}(table)); refine = Dict{String, Any}()
    for k in keys
        t, key = split(config_key(table, k), '.'; limit = 2)           # the key as the table's version names it
        haskey(start, t) && start[t] isa AbstractDict && haskey(start[t], key) || continue
        get!(refine, t, Dict{String, Any}())[key] = pop!(start[t], key)
        isempty(start[t]) && delete!(start, t)
    end
    return start, refine
end

"""
    start_options(opts) -> PipelineOptions

The options a start is computed from and keyed by: `opts` with [`REFINE_ONLY_KEYS`](@ref) and the output keys
(`PIPELINE_OUTPUT_KEYS`: `io.out`, `optimizer.log`) at their defaults (`split_refine_only`).
"""
start_options(opts::PipelineOptions) = pipeline_options(start_pipeline_table(config_dict(opts)))

"""
    start_pipeline_table(table) -> Dict

The `[pipeline]` table `table` as its start reads it: without the refinement-only keys (`REFINE_ONLY_KEYS`) and the outputs
(`PIPELINE_OUTPUT_KEYS`: `io.out`, `optimizer.log`) — THE definition of what a start reads, `start_options`' and a driver's start key's.
"""
start_pipeline_table(table::AbstractDict) = first(split_refine_only(table; keys = (REFINE_ONLY_KEYS..., PIPELINE_OUTPUT_KEYS...)))

"The start record's files ([`save_calibration_start`](@ref))."
const START_RECORD_FILE = "start.json"
"The start record's directory of its masks, one PNG per view ([`save_calibration_start`](@ref))."
const START_MASKS_DIR = "masks"

"""The start record's camera file, by the route the record names (its `parametric` entry): `lens.json` per view, `grid.json` per tile."""
_start_lens_file(parametric::Bool) = parametric ? "lens.json" : "grid.json"
_mask_file(dir, k) = joinpath(dir, START_MASKS_DIR, "$(k).png")

"""
    save_start(dir, s, meta, opts) -> dir

The start of a refinement (the start's state `s` and `meta`, of the options `opts`, its `start_options`) as the start record
`dir` (the file header lists its files): every value a refinement takes from its start, so `load_calibration_start(dir; options)` rebuilds
the start's state from the photographs and this record without recomputing it.
"""
function save_start(dir::AbstractString, s::CalibrationState, meta, opts::PipelineOptions)
    mkpath(joinpath(dir, START_MASKS_DIR))
    names = meta.names
    save_state(joinpath(dir, "state.json"), s, names; page_flip = meta.page_flip, provenance = (; stage = "start (closed-form sweep)"))
    tag = _frontend_tag(opts)
    route = lm_route(geometric(s))                         # the state's camera's route (a SplineLens camera per tile)
    for (file, g, what) in ((joinpath(dir, "$(_camera_file(route)).json"), geometric(s), "start"),
                            (joinpath(dir, "frontend", "$(_camera_file(route)).json"), meta.lens_start, "frontend"))
        mkpath(dirname(file))
        _save_run_camera(route, file, g, meta.init_region; provenance = (; model = "$(what)_$(_model_label(route, g))_$tag", frontend = tag))
    end
    for (k, m) in enumerate(s.masks)
        PNGFiles.save(_mask_file(dir, k), Gray{N0f8}.(Array(m)))
    end
    write_pipeline_config(dir, opts; header = "# the start options of this start record (start_options: the refinement-only keys at their defaults)")
    VgtkIO.write_record(joinpath(dir, START_RECORD_FILE),
        (; schema = RECORD_SCHEMAS.refinement_start, names = String.(names), image_size = [meta.W₀, meta.H₀], parametric = route isa PerViewRoute,
           fpx = meta.fpx, R0 = meta.R0, d_c = meta.d_c, f_m = meta.f_m,
           bearing_errors_px = meta.be0, page_flip = meta.page_flip, board_y_sign = meta.board_y_sign,
           vignetting_center = meta.vig_c === nothing ? nothing : collect(Float64, meta.vig_c),
           cost_init = collect(meta.costs.init), cost_linear = collect(meta.costs.linear), mu_linear = meta.μ_lin,
           vignetting_gcv = meta.vignetting_gcv, grid_gcv = meta.grid_gcv,
           samples = get(meta, :samples, nothing),             # the masks' record: samples_per_view, the samples drawn per view, the shortfall per view
           sensor_model = get(meta, :sensor_model, nothing),   # the run's use of the sensor model (`sensor_model_record`): mode, used, the record present
           resample = meta.resample))   # the exact resample's record (`GridStart`'s): its lattice, marks, folds and fidelity; null when none
    return dir
end

# A JSON object of the record as the NamedTuple the pipeline wrote, bit for bit: its fields in the order written, numbers, flags and
# strings as read (an integer an Int, a flag a Bool), a vector of one kind typed as it, and a GCV record's `kind` (the smoothness's
# type) a Symbol again — so a refinement from the record carries the computed start's very selection (its parameters.json the same bytes).
_plain(v::AbstractVector) = all(x -> x isa Bool, v) ? Bool[x for x in v] : all(x -> x isa Integer, v) ? Int[x for x in v] :
                            all(x -> x isa Real, v) ? Float64[x for x in v] : Any[_plain(x) for x in v]
_plain(v::AbstractDict) = _plain_fields([(Symbol(k), x) for (k, x) in pairs(v)])
_plain(v) = v
_plain_fields(kv) = NamedTuple{Tuple(first.(kv))}(Tuple(k === :kind ? Symbol(x) : _plain(x) for (k, x) in kv))
_plain_object(o) = o === nothing ? nothing : _plain_fields([(Symbol(k), x) for (k, x) in pairs(o)])

"""
    load_refinement_start(dir) -> NamedTuple

The start record `dir` ([`save_calibration_start`](@ref)): `(; names, W₀, H₀, parametric, fpx, R0, d_c, f_m, be0, page_flip, board_y_sign,
vig_c, costs, μ_lin, vignetting_gcv, grid_gcv, resample, options, lens, init_region, state, masks, dir)` — the front end's lens
(`frontend/`, read as the start was), its initialization region, the state
([`load_state`](@ref)) and the masks (`BitMatrix`, one per view; `masks`, when given, the very masks the record was written from:
not read again).
"""
function load_refinement_start(dir::AbstractString; masks = nothing)
    r, _ = _read_kind(joinpath(dir, START_RECORD_FILE), :refinement_start)
    parametric = Bool(r.parametric)
    lensfile = joinpath(dir, "frontend", _start_lens_file(parametric))
    st = load_start((parametric ? "lens:" : "grid:") * lensfile)
    names = String[String(n) for n in r.names]
    masks = masks === nothing ? [BitMatrix(PNGFiles.load(_mask_file(dir, k)) .> 0.5) for k in eachindex(names)] :   # the record's masks, or the
            (length(masks) == length(names) || throw(ArgumentError("$dir: $(length(masks)) masks given for $(length(names)) views")); copy(masks))   #   start's own (`load_calibration_start`)
    vec2(v) = v === nothing ? nothing : SVector{2, Float64}(v...)
    gcv = r.vignetting_gcv
    return (; names, W₀ = Int(r.image_size[1]), H₀ = Int(r.image_size[2]), parametric, fpx = Float64(r.fpx), R0 = Float64(r.R0),
              d_c = Float64(r.d_c), f_m = Float64(r.f_m), be0 = Float64[x for x in r.bearing_errors_px], page_flip = Float64(r.page_flip),
              board_y_sign = Float64(r.board_y_sign), vig_c = vec2(r.vignetting_center),
              costs = (; init = (Float64(r.cost_init[1]), Float64(r.cost_init[2])), linear = (Float64(r.cost_linear[1]), Float64(r.cost_linear[2]))),
              μ_lin = Float64(r.mu_linear), vignetting_gcv = NamedTuple[_plain_object(g) for g in gcv], grid_gcv = _plain_object(r.grid_gcv),
              options = pipeline_options(joinpath(dir, PIPELINE_CONFIG_FILE)),
              resample = _plain_object(get(r, :resample, nothing)),   # the exact resample's record (`GridStart`'s), as written; a start without one null
              lens = st.lens, init_region = st.init_region, state = load_state(joinpath(dir, "state.json")), masks, dir = String(dir))
end
"""
    check_start_options(start, opts)

Refuses (`ArgumentError`, naming the differing keys) a start record whose start options are not `start_options(opts)`: a start of
other settings is never taken for this run's.
"""
function check_start_options(start, opts::PipelineOptions)
    want = start_options(opts)
    have = start.options
    have == want && return nothing
    a, b = config_dict(have), config_dict(want)
    delete!(a, VgtkCore.CONFIG_VERSION_KEY); delete!(b, VgtkCore.CONFIG_VERSION_KEY)   # both the current one: tables and keys compared
    diff = sort!(["$t.$k" for t in union(keys(a), keys(b)) for k in union(keys(get(a, t, Dict())), keys(get(b, t, Dict())))
                  if get(get(a, t, Dict()), k, nothing) != get(get(b, t, Dict()), k, nothing)])
    throw(ArgumentError("the start record $(start.dir) was computed from other start options than this run's: $(join(diff, ", ")) " *
                        "(a start is keyed by every option outside REFINE_ONLY_KEYS: write this run's with bin/refine.jl --stage=start)"))
end
