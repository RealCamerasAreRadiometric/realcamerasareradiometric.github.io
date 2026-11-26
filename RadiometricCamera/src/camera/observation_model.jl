# ---------------------------------------------------------------------------
# The observation model of a calibrated camera (if a sensor was estimated with a model, then it must be used to predict the
# model): how the photographs a camera was calibrated on were read into observations — the sensor statement (the
# sensor characterisation the RAW photographs were linearised under, identified by its sha256, or the statement that there is none)
# and the photographs' kind (RAW photosites, or decoded images) with the channel measured. A `CalibratedCamera` is the camera model
# with its observation model and its render settings: the complete calibration, the unit the records save and load. Every reading of
# photographs a camera predicts goes through ONE entry point (`photo_set` / `observe`, io/photographs.jl), which dispatches on the
# observation model's types — never on a flag — and refuses a characterisation, a photograph kind or a frame the camera was not
# calibrated under.
# ---------------------------------------------------------------------------

"""The sensor statement of a calibration: [`SensorRecord`](@ref) or [`NoSensorRecord`](@ref)."""
abstract type SensorStatement end

"""
    SensorRecord(sha256)

The camera was calibrated on RAW photographs linearised under the sensor characterisation (a `vgio/sensor_characterization` record)
whose file's sha256 is `sha256` (lower-case hex, Core's `sha256_hex`): its black (a dark per exposure, or the measured channel black),
its noise law and its defects. A prediction of the camera reads photographs under exactly that record; any other file is refused.
"""
struct SensorRecord <: SensorStatement
    sha256::String
    function SensorRecord(sha256::AbstractString)
        occursin(r"^[0-9a-f]{64}$", sha256) || throw(ArgumentError("SensorRecord: $(repr(sha256)) is not a sha256 (64 lower-case hex digits)"))
        return new(String(sha256))
    end
end

"""
    NoSensorRecord()

The camera was calibrated without a sensor characterisation (none named, or `camera.sensor_model = "off"`): RAW photographs under the file's
black, no noise law and no defects (Core's absent `SensorModel()`), or decoded images, which no sensor model describes.
"""
struct NoSensorRecord <: SensorStatement end

"""The kind of the photographs a camera was calibrated on: [`Photosites`](@ref) or [`Decoded`](@ref)."""
abstract type PhotoKind end
"""RAW photographs: the photosites of ONE colour filter on the raw grid — the observation model's channel, `:R`, `:G` (both greens)
or `:B` — censored at the ADC floor and the white level, a sensor record's defects masked (`photosite_mask`)."""
struct Photosites <: PhotoKind end
"""
    Decoded(srgb)

Decoded images (PNG, TIFF, the synthetic renders): their pixels, read by Features' `load_linear_image` — sRGB-decoded when `srgb`
(the calibration's `io.linearize`), the samples as stored otherwise.
"""
struct Decoded <: PhotoKind
    srgb::Bool
end

"""The channels an observation model measures: `:RGB` (the three, decoded images), `:R`, `:G`, `:B` (one; a RAW photograph's
photosites are those of one colour filter: one of the three)."""
const OBSERVATION_CHANNELS = (:RGB, :R, :G, :B)

"""
    ObservationModel(sensor::SensorStatement, kind::PhotoKind, channel::Symbol)

How a calibration's photographs were observed: the `sensor` statement, the photographs' `kind` and the `channel` measured
(`OBSERVATION_CHANNELS`). RAW photosites are those of one colour filter (`:R`, `:G` or `:B`, never `:RGB`); a sensor record describes RAW photosites only (a decoded image
under a [`SensorRecord`](@ref) is refused).
"""
struct ObservationModel{S <: SensorStatement, K <: PhotoKind}
    sensor::S
    kind::K
    channel::Symbol
    function ObservationModel(sensor::S, kind::K, channel::Symbol) where {S <: SensorStatement, K <: PhotoKind}
        channel in OBSERVATION_CHANNELS ||
            throw(ArgumentError("ObservationModel: the channel $(repr(channel)) is not one of $(join(OBSERVATION_CHANNELS, ", "))"))
        kind isa Photosites && channel === :RGB &&
            throw(ArgumentError("ObservationModel: a RAW photograph's photosites are ONE observation each, those of one colour filter: " *
                                "the channel :R, :G or :B (not $(repr(channel)))"))
        sensor isa SensorRecord && kind isa Decoded &&
            throw(ArgumentError("ObservationModel: a sensor record describes RAW photosites; a decoded image has none"))
        return new{S, K}(sensor, kind, channel)
    end
end
Base.:(==)(a::ObservationModel, b::ObservationModel) = a.sensor == b.sensor && a.kind == b.kind && a.channel === b.channel

"""The colour channels by name, in the photographs' and every per-channel quantity's order (the lighting rows, the η's,
`io.channel`)."""
const CHANNEL_NAMES = ("R", "G", "B")
"""The observation model's channel of a state's channel index (0: `:RGB`; 1–3: `:R`, `:G`, `:B`)."""
_channel_symbol(c::Integer) = c == 0 ? :RGB : Symbol(CHANNEL_NAMES[c])

# ---- the run's use of the sensor model: `camera.sensor_model` -----------------------------------
"""Whether a run uses the sensor model (`camera.sensor_model`, parsed once by [`sensor_model_mode`](@ref)): [`AutoSensorModel`](@ref),
[`OnSensorModel`](@ref) or [`OffSensorModel`](@ref). Every decision on it is a dispatch on these types."""
abstract type SensorModelMode end
"""`sensor_model = "auto"` (the default): a sensor record present is used; with none the run reads the file's black (one info line)."""
struct AutoSensorModel <: SensorModelMode end
"""`sensor_model = "on"`: the run requires a sensor model — a sensor record present is used; with none the run is refused."""
struct OnSensorModel <: SensorModelMode end
"""`sensor_model = "off"`: a sensor record present is NOT used (the file's black, no noise law, no defects), with the loud warning
[`warn_sensor_model_off`](@ref); with none the run reads the file's black."""
struct OffSensorModel <: SensorModelMode end
"""The values of `camera.sensor_model`, each the mode it parses to."""
const SENSOR_MODEL_TAGS = VgtkCore.TagRegistry(:sensor_model, "SensorModel", "auto" => AutoSensorModel, "on" => OnSensorModel, "off" => OffSensorModel;
                                      paths = [r"camera\.sensor_model"])
"""
    sensor_model_mode(value) -> SensorModelMode

The mode of `camera.sensor_model = value` (`SENSOR_MODEL_TAGS`): another value is an `ArgumentError` naming the three.
"""
sensor_model_mode(value) = read_switch(value, "camera.sensor_model", SENSOR_MODEL_TAGS)   # the one switch reader (src/options/toml.jl)
"""The TOML value of a mode (the run's records)."""
mode_name(m::SensorModelMode) = switch_spelling(m, SENSOR_MODEL_TAGS)

"""
    warn_sensor_model_off(record)

THE warning of a run that does not use a sensor record present (`camera.sensor_model = "off"`): one multi-line `@warn` banner naming
`record` (its location and sha256, or the camera's statement), that the run does not use it, that estimation and prediction may then
disagree, and the TOML line. Emitted once per run (the run's resolution, `observation_model`; a prediction, `photo_set`).
"""
function warn_sensor_model_off(record::AbstractString)
    @warn """
    ================================ SENSOR MODEL OFF ================================
    A sensor characterisation record IS present for these photographs:
        $record
    and this run does NOT use it (`[camera] sensor_model = "off"`): every RAW photograph is read under the file's
    black, with no noise law and no defect mask. Estimation and prediction may then DISAGREE with a run that uses
    the sensor model — the vignetting at the rim is biased by the dark offset, and the results are not comparable
    with sensor-model runs. Remove `sensor_model = "off"` (the default "auto" uses the record) unless this is meant.
    ==================================================================================="""
    return nothing
end
_record_line(path::AbstractString, r::SensorRecord) = "$path (sha256 $(r.sha256))"
_record_line(r::SensorRecord) = "the camera's sensor record (sha256 $(r.sha256))"

"""
    sensor_statement(spec) -> SensorRecord

The sensor record of `camera.sensor_characterization = spec`: the [`SensorRecord`](@ref) of the file `spec` (its sha256, Core's
`sha256_hex` of the file). A `spec` that names no file, an empty one, and the retired `"none"` are `ArgumentError`s.
"""
function sensor_statement(spec::AbstractString)
    spec == NO_SENSOR_CHARACTERIZATION &&
        throw(ArgumentError("`camera.sensor_characterization = \"none\"` is retired: `camera.sensor_model = \"off\"` does not use a record present"))
    isempty(spec) && throw(ArgumentError("`camera.sensor_characterization` names no record"))
    isfile(spec) || throw(ArgumentError("`camera.sensor_characterization`: no file $spec (a named record must exist: a typo never turns the sensor model off)"))
    return SensorRecord(open(VgtkCore.sha256_hex, spec))
end
"""The named record of `spec`: `nothing` (empty), else `(; path, record)` ([`sensor_statement`](@ref)'s refusals)."""
_named_record(spec::AbstractString) = isempty(spec) ? nothing : (; path = String(spec), record = sensor_statement(spec))

"""
    observation_model(files, spec; linearize = false, channel = "G", sensor_model = AutoSensorModel(), stated = NoSensorRecord()) -> ObservationModel

The observation model of a calibration reading the photographs `files` (the producer's: the calibration's start), by the files
(`is_raw_file`), never by a flag: RAW photographs are [`Photosites`](@ref) under the sensor statement `sensor_model` decides
(`sensor_model_mode`) from the record present — `spec` (`camera.sensor_characterization`: a named record must exist) or `stated`, the
sensor statement of the camera the run continues (a start or resumed state's):
- [`AutoSensorModel`](@ref): the named record's [`SensorRecord`](@ref); none named — [`NoSensorRecord`](@ref) (one info line), or, when
  the camera states a record, refused (its location is needed);
- [`OnSensorModel`](@ref): the named record's; none named — refused;
- [`OffSensorModel`](@ref): [`NoSensorRecord`](@ref), with [`warn_sensor_model_off`](@ref) when a record is present.
Decoded images are [`Decoded`](@ref)`(linearize)` under [`NoSensorRecord`](@ref): a record named for them is refused (no sensor model
describes a decoded image), so is `sensor_model = "on"`. RAW and decoded photographs mixed are refused; `channel` is `io.channel`.
"""
function observation_model(files::AbstractVector{<:AbstractString}, spec::AbstractString; linearize::Bool = false, channel::AbstractString = "G",
                           sensor_model::SensorModelMode = AutoSensorModel(), stated::SensorStatement = NoSensorRecord())
    rec = _named_record(spec)
    raw = is_raw_file.(files)
    ch = Symbol(channel)
    if !any(raw)
        rec === nothing ||
            throw(ArgumentError("`camera.sensor_characterization` = $(repr(spec)): the photographs are decoded images, which no sensor model describes"))
        sensor_model isa OnSensorModel &&
            throw(ArgumentError("`camera.sensor_model = \"on\"`: the photographs are decoded images, which no sensor model describes (\"auto\" reads them)"))
        return ObservationModel(NoSensorRecord(), Decoded(linearize), ch)
    end
    all(raw) || throw(ArgumentError("the photographs mix RAW files and decoded images ($(count(raw)) of $(length(raw)) RAW): one kind per run"))
    linearize && throw(ArgumentError("a RAW photograph is linear (Features' linearisation): `io.linearize` does not apply"))
    ch === :RGB && throw(ArgumentError("the photographs are RAW: their photosites are ONE observation each, those of one colour filter — " *
                                       "`io.channel = \"R\"`, \"G\" or \"B\" (not $(repr(channel)))"))
    return ObservationModel(_run_sensor(sensor_model, rec, stated), Photosites(), ch)
end
_run_sensor(::Union{AutoSensorModel, OnSensorModel}, rec::NamedTuple, _) = rec.record
_run_sensor(::AutoSensorModel, ::Nothing, ::NoSensorRecord) =
    (@info "sensor model: no sensor characterisation record is named — every RAW photograph is read under the file's black (no noise law, no defects)"; NoSensorRecord())
_run_sensor(::Union{AutoSensorModel, OnSensorModel}, ::Nothing, r::SensorRecord) =
    throw(ArgumentError("the camera this run continues was calibrated under the sensor record sha256 $(r.sha256): name its location " *
                        "(`camera.sensor_characterization`), or state `camera.sensor_model = \"off\"` to read the photographs without it"))
_run_sensor(::OnSensorModel, ::Nothing, ::NoSensorRecord) =
    throw(ArgumentError("`camera.sensor_model = \"on\"`: the run requires a sensor model, and no record is named — name the body's " *
                        "vgio/sensor_characterization record in `camera.sensor_characterization` (or use \"auto\" or \"off\")"))
_run_sensor(::OffSensorModel, rec::NamedTuple, _) = (warn_sensor_model_off(_record_line(rec.path, rec.record)); NoSensorRecord())
_run_sensor(::OffSensorModel, ::Nothing, r::SensorRecord) = (warn_sensor_model_off(_record_line(r)); NoSensorRecord())
_run_sensor(::OffSensorModel, ::Nothing, ::NoSensorRecord) = NoSensorRecord()

"""
    sensor_model_record(mode, spec, stated, om) -> NamedTuple

What a run's records state of its sensor model: the `mode` (`camera.sensor_model`), whether the observation model `om` uses a record
(`used`), and the record present (`present`: the named one's `path` and `sha256`, or the camera's statement's sha256; `nothing` when
none). `off` with a record present is a run that did not use it (the tables flag it).
"""
function sensor_model_record(mode::SensorModelMode, spec::AbstractString, stated::SensorStatement, om::ObservationModel)
    present = isempty(spec) ? (stated isa SensorRecord ? (; path = nothing, sha256 = stated.sha256) : nothing) :
              (; path = String(spec), sha256 = sensor_statement(spec).sha256)
    return (; mode = mode_name(mode), used = om.sensor isa SensorRecord, present)
end
"""The location `photo_set` reads the run's photographs with: the named record's path when the run's observation model uses one."""
_characterization_path(spec::AbstractString, om::ObservationModel) = om.sensor isa SensorRecord ? String(spec) : nothing

"""
    CalibratedCamera(model, observation, settings)

A COMPLETE calibration: the `RadiometricCameraModel` `model` (the kernel type: lens, focus surface, aperture, vignetting), the
[`ObservationModel`](@ref) its photographs were observed under and the [`RenderSettings`](@ref) its predictions were integrated with.
What a prediction of the camera must reproduce; the unit `rc/radiometric_camera` and `rc/calibration_state` records save and load
(`load_radiometric_camera`, `calibrated_camera(load_state(path))`). Photographs it predicts are read through [`photo_set`](@ref).
"""
struct CalibratedCamera{C <: RadiometricCameraModel, O <: ObservationModel, ST <: RenderSettings}
    model::C
    observation::O
    settings::ST
end

"""
    calibrated_camera(state) -> CalibratedCamera

The calibrated camera of a state: a [`load_state`](@ref) record's (its `camera`, `observation` and `settings`); a calibration's
is `calibrated_camera(f, θ)` (its parameters' camera, its data's `observation_model(f)`).
"""
calibrated_camera(st::NamedTuple) = CalibratedCamera(st.camera, st.observation, st.settings)
"""One line of an observation model for a message or a log: its sensor statement, the photographs' kind and the channel."""
observation_line(o::ObservationModel) = "$(_sensor_line(o.sensor)), $(_kind_line(o.kind)), channel $(o.channel)"
_sensor_line(r::SensorRecord) = "sensor record sha256 $(r.sha256)"
_sensor_line(::NoSensorRecord) = "no sensor record"
_kind_line(::Photosites) = "RAW photosites"
_kind_line(k::Decoded) = k.srgb ? "decoded images (sRGB-decoded)" : "decoded images"
