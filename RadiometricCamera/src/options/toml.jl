# ---------------------------------------------------------------------------
# The options of the calibration pipeline: ONE value, `PipelineOptions`, read from TOML.
#
# A run is configured by a TOML file of tables (`[io]`, `[frontend]`, `[camera]`, `[grid]`, `[focus]`, `[masks]`, `[rendering]`,
# `[lighting]`, `[material]`, `[vignetting]`, `[warping]`, `[edge]`, `[lm]`, `[sweep]`, `[resume]`, `[selfcheck]`, `[compute]`), one struct per
# table. The defaults are stated ONCE, in configs/default.toml (every field's keyword default reads it, tables.jl); a key absent from a config takes it. An unknown table or
# key is refused (`ArgumentError` naming it and the allowed keys), never ignored. An enumerated choice is a tag of a `TagRegistry`
# (VisualGeometryToolkitCore's one tag → type map): `FRONTEND_TAGS`, `VIGNETTING_TAGS`, `BACKEND_TAGS`.
#
#   pipeline_options(path_or_dict; set = ["lm.max_iter=2", …]) -> PipelineOptions     the reader (`set`: dotted-key overrides)
#   config_dict(opts) -> Dict{String, Any}                                              the writer (the run record; reads back equal)
#   PIPELINE_PATH_KEYS, PIPELINE_OUTPUT_KEYS                                            the keys that name an input / an output path
#
# configs/default.toml states every key with its default and one comment each: every table field's keyword default reads it (tables.jl).
# These options replace the environment variables the recipes used before (test/unit/pipeline_options_tests.jl holds the table of
# that retirement, former variable → key).
# ---------------------------------------------------------------------------


# ---- the keys that name paths ----------------------------------------------------------------------------------------------------
"""
The keys (`table.key`) whose values name an INPUT file or directory of a run: what a driver resolves against its data root and
records as the run's inputs. `io.init` is a model spec, `lens:<path>` or `grid:<path>`. Every `String` option of
[`PipelineOptions`](@ref) is here, in `PIPELINE_OUTPUT_KEYS` or in `_NOT_PATH_KEYS` (the suite checks the three cover
the options): a new string option has to be declared one or the other.
"""
const PIPELINE_PATH_KEYS = ("io.images", "io.init", "io.render_state", "io.raw", "frontend.session", "frontend.schoeps",
                            "frontend.texture", "frontend.texture_features", "camera.lens_coverage", "camera.sensor_characterisation", "edge.sdf",
                            "resume.state", "selfcheck.poses")
"""
    names_input_path(key, value) -> Bool

Whether the option `key` (`table.key`) with `value` names an input path of the run: a key of `PIPELINE_PATH_KEYS` with a non-empty
string, except `camera.lens_coverage = "sensor"` (`SENSOR_LENS_COVERAGE`: the whole sensor, nothing measured). THE test a driver reads a run's inputs by.
"""
names_input_path(key::AbstractString, v) =
    key in PIPELINE_PATH_KEYS && v isa AbstractString && !isempty(v) && !(key == "camera.lens_coverage" && v == SENSOR_LENS_COVERAGE)
"""The keys whose values name an OUTPUT: the run's directory `io.out`, the LM's log `lm.log`."""
const PIPELINE_OUTPUT_KEYS = ("io.out", "lm.log")
# the string options that are no path (a tag-like choice)
const _NOT_PATH_KEYS = ("io.channel", "warping.inplane", "camera.sensor_model")

# ---- derived values the pipeline reads -------------------------------------------------------------------------------------------
"""The sensor width as a length quantity (`camera.sensor_mm` mm)."""
_sensor_width(opts::PipelineOptions) = opts.camera.sensor_mm * VgtkCore.mm
"""The nominal focal length as a length quantity (`camera.f_mm` mm)."""
_focal_length(opts::PipelineOptions) = opts.camera.f_mm * VgtkCore.mm
"""The printed page `(width, height)` as length quantities (`frontend.page_mm` mm)."""
_page_size(opts::PipelineOptions) = (opts.frontend.page_mm[1] * VgtkCore.mm, opts.frontend.page_mm[2] * VgtkCore.mm)
"""The printed edge `(δ, σ_print)` of the options (the print's: `TargetState.edge`)."""
_edge_start(opts::PipelineOptions) = (e = opts.edge; (e.ink_spread_mm, e.print_sigma_mm))
"""The front-end's tag (the labels of the output files)."""
_frontend_tag(opts::PipelineOptions) = tag_of(FRONTEND_TAGS, opts.frontend.type)

# ---- the reader ------------------------------------------------------------------------------------------------------------------
# A TOML value → the field type, with the key in every message: Core's keyed `convert_field` (a table refused, a tuple's length
# checked, a vector converted element-wise, a string refused for an array).
# The per-field hook: a tag field reads through its registry, every other field converts to its type.
_option_from_toml(::Type{T}, ::Val{F}, v, key) where {T, F} = _field_from_toml(fieldtype(T, F), v, key)
# a Bool field is a switch: "on" | "off" (`read_switch`); every other type Core's `convert_field`
_field_from_toml(::Type{Bool}, v, key) = read_switch(v, key)
_field_from_toml(::Type{S}, v, key) where {S} = convert_field(S, v, key)

# ---- the switches: "on" | "off" (RC 0.11.4, owner 2026-11-15) -------------------------------------------------------------------
# A switch of a config is a string, parsed ONCE here into the Bool the code reads (a setting that picks an implementation is a tag of a
# registry instead): TOML true/false is refused naming the new spelling (src/options/retired.jl), anything else refused.
"""The spellings of a two-way switch and the Bool each reads as (`read_switch`'s default cases)."""
const SWITCH_SPELLINGS = ("on" => true, "off" => false)

"""
    read_switch(v, key, cases = SWITCH_SPELLINGS) -> value

The switch `key` of a config with TOML value `v`, read through its `cases` (`word => value`): a two-way switch `"on"` → `true`,
`"off"` → `false`; a switch whose availability is a condition states its own cases, `"auto"` among them, each a concrete value (a
type: `camera.sensor_model`'s `SENSOR_MODEL_TAGS`), so "auto" is resolved here and the core code never sees a string. TOML's
`true`/`false` is refused naming the new spelling (`key = true` → use `key = "on"`; RC 0.11.4, no alias), any other value refused
naming the cases. THE ONE reader of every switch: the pipeline's options, the sensor model and the paper project's drivers.
"""
function read_switch(v, key::AbstractString, cases = SWITCH_SPELLINGS)
    v isa Bool && throw(ArgumentError(retired_boolean_switch(key, v)))
    for (w, x) in cases
        v == w && return x
    end
    throw(ArgumentError("`$key` = $(repr(v)) is not one of $(join(repr.(first.(cases)), ", ")) (a switch)"))
end
"""The config spelling of the switch value `x` among `cases` (`read_switch`'s inverse; the writer's): `"on"` / `"off"` for a Bool."""
switch_spelling(x, cases = SWITCH_SPELLINGS) = first(cases[findfirst(p -> last(p) === x, cases)])
# a setting that picks an implementation is a tag of a TagRegistry (the stack's convention, this file's header): the same reader and
# writer, the registry's type instantiated (`[grid] lattice`'s LATTICE_TAGS, `camera.sensor_model`'s SENSOR_MODEL_TAGS)
function read_switch(v, key::AbstractString, r::VgtkCore.TagRegistry)
    v isa Bool && throw(ArgumentError(retired_boolean_switch(key, v)))
    v isa AbstractString && haskey(r, v) || throw(ArgumentError("`$key` = $(repr(v)) is not one of $(join(repr.(collect(keys(r))), ", "))"))
    return r[v]()
end
switch_spelling(x, r::VgtkCore.TagRegistry) = VgtkCore.tag_of(r, x)

"""
    boolean_switch_violations(config_dirs; data = ()) -> Vector{String}

The lint of the switch convention: every `.toml` under `config_dirs` (recursively) parsed, each TOML boolean found at a key path
(VisualGeometryToolkitCore's leaf paths, `table.key`, an array's element `key[i]`) that no regex of `data` matches whole — a boolean
that is a switch, which is spelled `"on"` | `"off"` — is a finding `"file:path = value"`. `data` declares the booleans that are data,
not switches (a statement of fact, or a key of another package's schema that package reads). A file that does not parse is a finding.
"""
function boolean_switch_violations(config_dirs; data = ())
    out = String[]
    for dir in config_dirs
        isdir(dir) || continue
        for (root, _, files) in walkdir(dir), f in sort(files)
            endswith(f, ".toml") || continue
            path = joinpath(root, f)
            doc = try
                TOML.parsefile(path)
            catch e
                push!(out, "$path: does not parse ($(sprint(showerror, e)))")
                continue
            end
            for (p, v) in VgtkCore._toml_leaves(doc, "", Pair{String, Any}[])
                v isa Bool && !any(rx -> (m = match(rx, p); m !== nothing && m.match == p), data) && push!(out, "$path:$p = $v")
            end
        end
    end
    return out
end
_option_from_toml(::Type{FrontendOptions}, ::Val{:type}, v, key) = FRONTEND_TAGS[convert_field(String, v, key)]
_option_from_toml(::Type{VignettingOptions}, ::Val{:model}, v, key) = _vignetting_model(VIGNETTING_TAGS[convert_field(String, v, key)], key)
# Core's registry names every vignetting field; a refinement fits the radial profile alone.
_vignetting_model(::Type{M}, key) where {M <: RadialVignetting} = M
_vignetting_model(::Type{M}, key) where {M} =
    throw(ArgumentError("`$key` = $(repr(tag_of(VIGNETTING_TAGS, M))): a refinement fits $(repr(tag_of(VIGNETTING_TAGS, RadialVignetting))) alone, the radial profile " *
                        "about the principal point (the tensor-product surface is no option of the pipeline since 2026-11-14)"))
_option_from_toml(::Type{ComputeOptions}, ::Val{:backend}, v, key) = BACKEND_TAGS[convert_field(String, v, key)]
# a penalty weight: a number β ≥ 0, or the string "gcv" (selected by generalized cross-validation)
function _penalty_weight_from_toml(v, key)
    v isa AbstractString && (v == "gcv" ? (return :gcv) : throw(ArgumentError("`$key` = $(repr(v)) is not a weight β ≥ 0 or \"gcv\"")))
    return VgtkEstimation.check_penalty_weight(convert_field(Float64, v, key), "`$key`")
end
# `[<component>.smoothness]`: a table dispatched by its `type` (`SMOOTHNESS_TYPES`), its keys the type's — `order` and `weight` of a
# pspline, `weight` of a bending, none of none; each required (no default inside a stated table), an unknown one refused
const _SMOOTHNESS_KEYS = (pspline = ("type", "order", "weight"), bending = ("type", "weight"), none = ("type",))
"""
    typed_table_type(v, key, keys_by_type; retired = (;), shape) -> Symbol

THE reader of a table dispatched by its `type` (`[<component>.smoothness]`, `[masks] edge_detector`): `v` a table (else an error
naming `shape`), its `type` required and one of `keys_by_type`'s (a retired one refused with its reason from `retired`), exactly the
type's keys stated, each required (no default inside a stated table). Returns the type; the caller reads its values.
"""
function typed_table_type(v, key, keys_by_type::NamedTuple; retired::NamedTuple = (;), shape::AbstractString)
    v isa AbstractDict || throw(ArgumentError("`$key` = $(repr(v)) is not a table $shape"))
    VgtkCore.required_key(v, "type", key)
    ts = convert_field(String, v["type"], "$key.type"); t = Symbol(ts)
    haskey(retired, t) && throw(ArgumentError("`$key.type` = $(repr(ts)): $(retired[t])"))
    haskey(keys_by_type, t) || throw(ArgumentError("`$key.type` = $(repr(ts)) is not one of $(String.(collect(keys(keys_by_type))))"))
    ks = keys_by_type[t]
    check_keys(v, collect(ks), key)
    foreach(k -> VgtkCore.required_key(v, k, "$key (type $(repr(ts)))"), ks)
    return t
end
function _smoothness_from_toml(v, key)
    t = typed_table_type(v, key, _SMOOTHNESS_KEYS; shape = "{type = \"pspline\" | \"bending\" | \"none\", …}")
    t === :none && return NoSmoothness()
    w = _penalty_weight_from_toml(v["weight"], "$key.weight")
    t === :bending && return BendingSmoothness(w)
    o = convert_field(Int, v["order"], "$key.order")
    o >= 1 || throw(ArgumentError("`$key.order` = $o: a difference order ≥ 1"))
    return PSplineSmoothness(o, w)
end
_option_from_toml(::Type{<:Union{GridOptions, VignettingOptions, FocusOptions}}, ::Val{:smoothness}, v, key) = _smoothness_from_toml(v, key)
_option_from_toml(::Type{MaskOptions}, ::Val{:edge_detector}, v, key) = edge_detector_from_toml(v, key)
"The `[<component>.smoothness]` table of a smoothness (the reader's inverse)."
smoothness_table(s::PSplineSmoothness) = Dict{String, Any}("type" => "pspline", "order" => s.order, "weight" => s.weight === :gcv ? "gcv" : s.weight)
smoothness_table(s::BendingSmoothness) = Dict{String, Any}("type" => "bending", "weight" => s.weight === :gcv ? "gcv" : s.weight)
smoothness_table(::NoSmoothness) = Dict{String, Any}("type" => "none")
_option_from_toml(::Type{GridOptions}, ::Val{:lattice}, v, key) = read_switch(v, key, LATTICE_TAGS)   # parsed once into its rule (dispatch)
_option_from_toml(::Type{GridOptions}, ::Val{:node_count}, v, key) = convert_field(Int, v, key)
function _option_from_toml(::Type{GridOptions}, ::Val{:nodes}, v, key)
    r = Symbol(convert_field(String, v, key))
    r in GRID_NODE_RULES || throw(ArgumentError("`$key` = $(repr(v)) is not one of $(String.(collect(GRID_NODE_RULES)))"))
    return r
end

# `grid.start_perturbation` is a table whose keys are all required (an absent table: no perturbation); unknown keys are refused
function _option_from_toml(::Type{GridOptions}, ::Val{:start_perturbation}, v, key)
    v isa AbstractDict || throw(ArgumentError("`$key` = $(repr(v)) is not a table {sigma_px = …, seed = …}"))
    check_keys(v, allowed_keys(StartPerturbationOptions), key)
    absent = [k for k in allowed_keys(StartPerturbationOptions) if !haskey(v, k)]
    isempty(absent) || throw(ArgumentError("`$key` lacks $(absent): every key of it is required (no default perturbation)"))
    return _read_table(StartPerturbationOptions, v, key)
end

# `io.lens_identity` is a table of {make, model, serial} (an absent table: EXIF's lens); unknown keys are refused, one at least stated
function _option_from_toml(::Type{IOOptions}, ::Val{:lens_identity}, v, key)
    v isa AbstractDict || throw(ArgumentError("`$key` = $(repr(v)) is not a table {make = …, model = …, serial = …}"))
    o = _read_table(LensIdentityOptions, v, key)
    all(isempty, (o.make, o.model, o.serial)) && throw(ArgumentError("`$key` states none of make, model, serial"))
    return o
end
toml_field(c::IOOptions, ::Val{:lens_identity}) =
    c.lens_identity === nothing ? nothing : filter(kv -> !isempty(kv[2]), toml_table(c.lens_identity))

# The writer's per-field hook (VisualGeometryToolkitCore's `toml_field`): a switch is written "on" | "off", a tag field as its tag.
toml_field(c::AbstractOptionsTable, ::Val{F}) where {F} = _field_toml(getfield(c, F))
_field_toml(b::Bool) = switch_spelling(b)
_field_toml(x) = VgtkCore.toml_value(x)
toml_field(c::FrontendOptions, ::Val{:type}) = tag_of(FRONTEND_TAGS, c.type)
toml_field(c::VignettingOptions, ::Val{:model}) = tag_of(VIGNETTING_TAGS, c.model)
toml_field(c::ComputeOptions, ::Val{:backend}) = tag_of(BACKEND_TAGS, c.backend)
toml_field(c::GridOptions, ::Val{:lattice}) = switch_spelling(c.lattice, LATTICE_TAGS)
toml_field(c::GridOptions, ::Val{:start_perturbation}) = c.start_perturbation === nothing ? nothing : toml_table(c.start_perturbation)
toml_field(c::Union{GridOptions, VignettingOptions, FocusOptions}, ::Val{:smoothness}) = smoothness_table(c.smoothness)
toml_field(c::MaskOptions, ::Val{:edge_detector}) = edge_detector_table(c.edge_detector)

# One table: unknown keys refused, present keys typed, absent keys the struct's default — Core's `check_keys` and `read_table` (its
# `toml_key` walk, the writer's), the per-field hook given the dotted key.
function _read_table(::Type{T}, d, section::AbstractString) where {T <: AbstractOptionsTable}
    d isa AbstractDict || throw(ArgumentError("`$section` must be a table [$section]"))
    check_keys(d, allowed_keys(T), section)
    return T(; read_table(T, d; from_toml = (S, F, v) -> _option_from_toml(S, F, v, "$section.$(toml_key(S, _field(F)))"))...)
end
_field(::Val{F}) where {F} = F

# Whether the run chooses its grid smoothness's weight by GCV: a grid route (`start_is_grid`) with a smoothness of weight "gcv".
_grid_gcv(opts::PipelineOptions) = smoothness_weight(opts.grid.smoothness) === :gcv && start_is_grid(opts)
# A component's smoothness: of a type it takes, a pspline of the difference order it implements, its weight ≥ 0 (or "gcv" where taken).
function _check_smoothness(sm::AbstractSmoothness, key; types = (PSplineSmoothness, BendingSmoothness, NoSmoothness), order = nothing, gcv = true)
    any(T -> sm isa T, types) || throw(ArgumentError("`$key.type` = $(repr(String(smoothness_type(sm)))): the component takes $(join([repr(String(smoothness_type(T))) for T in types], ", "))"))
    sm isa NoSmoothness && return nothing
    order === nothing || sm.order == order || throw(ArgumentError("`$key.order` = $(sm.order): the component's penalty is its order-$order differences"))
    w = smoothness_weight(sm)
    w === :gcv && !gcv && throw(ArgumentError("`$key.weight` = \"gcv\": the component's weight is a number β ≥ 0"))
    VgtkEstimation.check_penalty_weight(w, "`$key.weight`")
    return nothing
end
# The checks a single field's type cannot state.
function _check_options(opts::PipelineOptions)
    isempty(opts.camera.lens_coverage) && throw(ArgumentError("`camera.lens_coverage` = \"\": state it — \"sensor\" (the lens lights the whole sensor) " *
                                                              "or the path of the capture set's measured image region (rc/image_region/v1.0)"))
    opts.camera.sensor_characterisation == NO_SENSOR_CHARACTERISATION &&
        throw(ArgumentError("`camera.sensor_characterisation = \"none\"` is retired (RC 0.11.2): a run that does not use a sensor record present " *
                            "states `camera.sensor_model = \"off\"`; a run that names none leaves `camera.sensor_characterisation` empty"))
    sensor_model_mode(opts.camera.sensor_model)                 # one of SENSOR_MODEL_TAGS
    "grid" in opts.fixed && throw(ArgumentError("\"grid\" in `fixed` is renamed \"camera\" (the geometric camera's own parameters, a grid's or a parametric camera's; RC 0.11.0)"))
    "edge" in opts.fixed && throw(ArgumentError("\"edge\" in `fixed` is refused: the printed edge is never a parameter (its δ and σ_print are held at " *
                                                "`edge.ink_spread_mm` and `edge.print_sigma_mm`; RC 0.11.0) — delete it"))
    bad = setdiff(opts.fixed, FIXABLE_BLOCKS)
    isempty(bad) || throw(ArgumentError("`fixed` has unknown block(s) $(bad); allowed: $(collect(FIXABLE_BLOCKS))"))
    _check_lattice(opts)
    opts.grid.nodes in GRID_NODE_RULES || throw(ArgumentError("`grid.nodes` = :$(opts.grid.nodes) is not one of $(collect(GRID_NODE_RULES))"))
    opts.vignetting.boundary_order in BOUNDARY_ORDERS ||
        throw(ArgumentError("`vignetting.boundary_order` = $(opts.vignetting.boundary_order) is not one of $(collect(BOUNDARY_ORDERS))"))
    opts.warping.inplane in INPLANE_KINDS || throw(ArgumentError("`warping.inplane` = $(repr(opts.warping.inplane)) is not one of $(collect(INPLANE_KINDS))"))
    length(opts.warping.inplane_start) in (0, INPLANE_NC) && all(isfinite, opts.warping.inplane_start) ||
        throw(ArgumentError("`warping.inplane_start` needs $INPLANE_NC finite values [a, u] (the log-stretch and the shear) or none, got $(opts.warping.inplane_start)"))
    opts.vignetting.chart in VrtkCore.RADIAL_CHARTS || throw(ArgumentError("`vignetting.chart` = :$(opts.vignetting.chart) is not one of $(collect(VrtkCore.RADIAL_CHARTS))"))
    # the components' smoothness: the grid any type; the vignetting profile a pspline of its Δ³ (the closed-form solve's operator,
    # `vignetting_smoothness`) or none; the focus surface a bending of a number (no GCV) or none
    _check_smoothness(opts.grid.smoothness, "grid.smoothness")
    _check_smoothness(opts.vignetting.smoothness, "vignetting.smoothness"; types = (PSplineSmoothness, NoSmoothness), order = 3)
    _check_smoothness(opts.focus.smoothness, "focus.smoothness"; types = (BendingSmoothness, NoSmoothness), gcv = false)
    # the grid smoothness's weight by GCV linearizes the grid's patches: with the camera held ("camera" in fixed) it has no column to
    # choose for (a grid route only: the parametric route runs no GCV, `start_is_grid`)
    _grid_gcv(opts) && "camera" in opts.fixed &&
        throw(ArgumentError("`grid.smoothness.weight = \"gcv\"` with \"camera\" in `fixed`: the grid is held, so GCV has no grid column to " *
                            "weigh; state a number for `grid.smoothness.weight` (or free the grid)"))
    opts.sweep.joint && smoothing(opts.vignetting.smoothness) &&
        throw(ArgumentError("`vignetting.smoothness` = $(smoothness_line(opts.vignetting.smoothness)) with `sweep.joint`: the joint solve has no smoothness penalty"))
    0 < opts.focus.lens_coverage_scale <= 1 || throw(ArgumentError("`focus.lens_coverage_scale` = $(opts.focus.lens_coverage_scale) is not in (0, 1]: the scale of the lens coverage the focus nodes keep inside"))
    opts.masks.samples_per_view >= 1 || throw(ArgumentError("`masks.samples_per_view` = $(opts.masks.samples_per_view): at least one sample per view"))
    let m = opts.masks; check_draw(m.samples_per_view, m.gradient_bias, m.edge_fraction, m.addon_samples, m.addon_gradient_bias); end   # the draw's settings together (masks.jl)
    0 <= opts.focus.support_min < 1 || throw(ArgumentError("`focus.support_min` = $(opts.focus.support_min) is not a share of the mean data weight in [0, 1)"))
    opts.io.channel == "RGB" || opts.io.channel in CHANNEL_NAMES ||
        throw(ArgumentError("`io.channel` = $(repr(opts.io.channel)) is not one of RGB, $(join(CHANNEL_NAMES, ", "))"))
    sp = opts.grid.start_perturbation
    sp === nothing || (isfinite(sp.sigma_px) && sp.sigma_px >= 0) ||
        throw(ArgumentError("`grid.start_perturbation.sigma_px` = $(sp.sigma_px) is not a finite σ ≥ 0"))
    return opts
end

# The lattice's mode against the start (owner ruling 2026-11-15): "node_count" states its count and is a parametric start's — a Schöps
# start is his own lattice ("derived"); a SplineLens / parametric source the mode does not fit is refused at its start (`start_grid`).
function _check_lattice(opts::PipelineOptions)
    g = opts.grid
    g.node_count === nothing || check_node_count(g.node_count)
    (start_is_grid(opts) && g.lattice isa NodeCountLattice) || return nothing
    g.node_count === nothing && throw(ArgumentError("`grid.lattice = \"node_count\"` needs `grid.node_count`: the nodes the image region supports " *
                                                    "(e.g. 7875 = 105 · 75, the lattice of record's nodes)"))
    _frontend_tag(opts) == "schoeps" && (r = lattice_refusal(g.lattice, Val(:spline)); r === nothing || throw(ArgumentError(r * " (frontend.type = \"schoeps\")")))
    return nothing
end

# ---- a value the run's input makes unusable is an ERROR, never ignored (owner ruling 2026-11-15 19:55) ----------------------------
"""
    INAPPLICABLE_KEYS

The keys a run's other options make unusable: `(table, key) => why`, `key === nothing` for every key of the table, `why(opts)` the
reason the key cannot act (a string) or `nothing` where it applies. The reader refuses a key a config STATES where its reason holds
(`pipeline_options`: the key and the reason named), and the writer (`config_dict`) leaves such a key out, so a run's record reads back.
"""
const INAPPLICABLE_KEYS = (
    ("grid", nothing) => o -> start_is_grid(o) ? nothing :
        "a parametric record start (`io.init = \"lens:…\"`) is refined per view in its own parameters: it has no lattice",
    ("grid", "node_count") => o -> (o.grid.lattice isa DerivedLattice && _frontend_tag(o) == "schoeps") ?
        "a Schöps start (`frontend.type = \"schoeps\"`, `grid.lattice = \"derived\"`) is his own lattice extended over the lens coverage, " *
        "never subdivided (m = 1, `schoeps_start`): no node count acts on it" : nothing,
    ("focus", "smoothness") => o -> "focus" in o.fixed ? "the focus surface is held (\"focus\" in `fixed`): no smoothness acts on it" : nothing,
)
"""Why the option `table.key` cannot act in a run of `opts` (`INAPPLICABLE_KEYS`), or `nothing` where it applies."""
function inapplicable_reason(opts::PipelineOptions, table::AbstractString, key::AbstractString)
    for ((t, k), why) in INAPPLICABLE_KEYS
        t == table && (k === nothing || k == key) || continue
        r = why(opts)
        r === nothing || return r
    end
    return nothing
end
# a key the document `d` states that the run's options make unusable: refused, naming the key and the reason
function _refuse_inapplicable(d::AbstractDict, opts::PipelineOptions)
    for (t, tb) in d
        tb isa AbstractDict || continue
        for k in sort(collect(String, keys(tb)))
            r = inapplicable_reason(opts, t, k)
            r === nothing || throw(ArgumentError("`$t.$k` cannot be used: $r — delete it (a value the run cannot use is an error, never ignored)"))
        end
    end
    return opts
end

# `key.path=value` overrides into a copy of the parsed document: the value is read as a TOML value (`2`, `1e-10`, `true`, `"x"`,
# `[1, 2]`), and as a bare string when it is not one (`io.out=/some/dir`).
function _with_overrides(d::AbstractDict, set)
    isempty(set) && return d
    d = deepcopy(Dict{String, Any}(d))
    for s in set
        occursin('=', s) || throw(ArgumentError("override $(repr(s)) is not `key.path=value`"))
        k, v = split(s, '='; limit = 2)
        path = split(strip(k), '.')
        (isempty(path) || any(isempty, path)) && throw(ArgumentError("override $(repr(s)): empty key"))
        value = try
            doc = TOML.parse("v = " * v)
            keys(doc) == Set(["v"]) || throw(ArgumentError("override $(repr(s)): the value is not one TOML value"))
            doc["v"]
        catch err
            err isa TOML.ParserError || rethrow()
            String(v)
        end
        t = d
        for p in path[1:(end - 1)]
            t = get!(() -> Dict{String, Any}(), t, String(p))
            t isa AbstractDict || throw(ArgumentError("override $(repr(s)): `$p` is not a table"))
        end
        t[String(path[end])] = value
    end
    return d
end

"""
    pipeline_options(path::AbstractString; set = ()) -> PipelineOptions
    pipeline_options(table::AbstractDict; set = ()) -> PipelineOptions

The options of a refinement from a TOML file (or its parsed table). Every table and key is optional and takes its struct default when
absent; an unknown table or key, a value of the wrong type or length, an unknown or retired tag and an unknown `fixed` block are
refused with an `ArgumentError` that names them. `set` holds `key.path=value` overrides applied before reading (the `--set` of
bin/refine.jl): `pipeline_options("run.toml"; set = ["lm.max_iter=2", "io.out=/tmp/x"])`.
"""
function pipeline_options(path::AbstractString; set = ())
    try
        return pipeline_options(TOML.parsefile(path); set)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("$path$(isempty(set) ? "" : " (with the overrides $(collect(set)))"): $(err.msg)"))
    end
end
function pipeline_options(d::AbstractDict; set = ())
    d = _with_overrides(d, set)
    for ((table, key), migration) in RETIRED_KEYS
        t = get(d, table, nothing)
        t isa AbstractDict && haskey(t, key) && throw(ArgumentError("`$table.$key` is retired: $migration"))
    end
    unknown = setdiff(collect(String, keys(d)), allowed_keys(PipelineOptions))
    isempty(unknown) || throw(ArgumentError("unknown table(s) $(sort(unknown)); allowed tables: $(sort(allowed_keys(PipelineOptions)))"))
    kw = [f => (f === :fixed ? _read_fixed(get(d, "fixed", toml_default(_TOML_DEFAULTS, "", "fixed"))) :
                haskey(d, String(f)) ? _read_table(fieldtype(PipelineOptions, f), d[String(f)], String(f)) : fieldtype(PipelineOptions, f)())
          for f in fieldnames(PipelineOptions)]
    return _refuse_inapplicable(d, _check_options(PipelineOptions(; kw...)))
end

"""
    config_dict(opts::PipelineOptions) -> Dict{String, Any}

The options as a TOML document, one table per field (tags as their tag strings): what a run writes as its record. An absent
optional table (`grid.start_perturbation = nothing`) is not written, nor a key the run cannot use (`INAPPLICABLE_KEYS`).
`pipeline_options(config_dict(opts)) == opts` for options a config can state.
"""
config_dict(opts::PipelineOptions) =
    Dict{String, Any}(String(f) => (f === :fixed ? copy(opts.fixed) :
                                    filter(kv -> kv[2] !== nothing && inapplicable_reason(opts, String(f), kv[1]) === nothing, toml_table(getfield(opts, f))))
                      for f in fieldnames(PipelineOptions))
"""The top-level `fixed` of a config: a list of block names (strings)."""
_read_fixed(v) = v isa AbstractVector && all(x -> x isa AbstractString, v) ? String.(v) :
                 throw(ArgumentError("`fixed` is a list of block names ($(collect(FIXABLE_BLOCKS))), got $(repr(v))"))

"""The file a run's options are written to in its `io.out` ([`write_pipeline_config`](@ref)), and a record's regeneration hint reads."""
const PIPELINE_CONFIG_FILE = "pipeline_config.toml"

"""
    write_pipeline_config(dir, opts; header = "") -> path

The options `opts` of a run as `dir/pipeline_config.toml` (`config_dict`, with `header` as its comment), its `io.out` written as an
ABSOLUTE path (resolved against the writer's working directory, the one the run resolved it against): what bin/refine.jl and
experiments/scripts/material_grid.jl write, so a record's regeneration hint can tell its own writer's options from another run's in a shared
directory.
"""
function write_pipeline_config(dir::AbstractString, opts::PipelineOptions; header::AbstractString = "")
    d = config_dict(opts)
    isempty(opts.io.out) || (d["io"]["out"] = abspath(opts.io.out))
    path = joinpath(dir, PIPELINE_CONFIG_FILE)
    VgtkCore.write_toml(path, d; header)
    return path
end
