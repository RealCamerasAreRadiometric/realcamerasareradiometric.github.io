# =============================================================================
# src/sensor_characterization.jl — the config of scripts/sensor_characterization.jl and its one computation: the Sony ILME-FX3's
# sensor characterisation at ISO 100 from the campaign of 2026-11-13 (E1/E2 lens-cap dark frames, E4 photon transfer), by Core's
# estimators alone (owner ruling D12: RC's own dark statistics are deleted in favour of Core's MasterDark / defect_onsets):
#   black    Core's `Darks` (owner 2026-11-14: "we took darks at all exposures for a reason"): ONE Dark per E1_E2 burst (Core's
#            `MasterDark`: per photosite the temporal mean and variance, its exposure band the burst's own EXIF exposure), their
#            shared channel black the E1 master dark's (a measured ChannelBlack ± SE)
#   noise    a measured PoissonGaussian: the gain K from Core's multi-frame `PhotonTransfer` (groups, darks) on the E4 series (each
#            exposure's frames a group; each exposure's E1_E2 darks' ReadNoise its intercept; the E1 dark subtracted and its defects
#            left out), σ_r per colour index ± SE from the E1 master dark's ReadNoise
#   defects  ONE DefectList by Core's `defect_onsets` over every E1_E2 exposure's master-dark mask, each site at its onset (EXIF)
# into ONE Core SensorCharacterization of the body (EXIF), written through IO (vgio/sensor_characterization). Every capture fact —
# the body, the ISO, each burst's exposure — is the frames' EXIF (Features' `exposure_settings`, `body_identity`), never a folder
# name.
#
# Stack functions considered (the duplication gate): Core MasterDark, defect_onsets, PhotonTransfer (raw/sensor_estimators.jl:560,
# :617, :290) — used, the only estimators; Core SensorCharacterization, PoissonGaussian, SensorModel (raw/sensor_model.jl) — the
# record's constructors, Core Darks the black slot's (this experiment is the record's producer); Features load_raw_mosaic, exposure_settings, body_identity
# (raw_image.jl) — used; IO save(path, ::SensorCharacterization) — used. Nothing here estimates.
# =============================================================================

"The keys of the config's `[characterize]` table (every one required, a literal with its derivation in the config)."
const CHARACTERISE_KEYS = ("dark", "false_defects", "tier")
"The inputs the config of record names: the E1_E2 and E4 trees, the tier's manifest and the record of record."
const CHARACTERISATION_INPUTS = ("dark_frames", "ptc_frames", "manifest", "record_of_record")

"""
    load_sensor_characterization_config(path) -> Dict

The config of record of scripts/sensor_characterization.jl ([`load_driver_config`](@ref): tables experiment, inputs, characterize),
`[characterize]` holding exactly [`CHARACTERISE_KEYS`](@ref) (`dark` the E1 burst's folder of the E1_E2 tree, `false_defects` Core
`MasterDark`'s expected false defects over the sensor, `tier` the campaign's tier name as its MANIFEST.tsv rows state it) and `[inputs]` exactly [`CHARACTERISATION_INPUTS`](@ref), every one of stage
evaluate.
"""
function load_sensor_characterization_config(path::AbstractString)
    cfg = load_driver_config(path, "sensor_characterization"; tables = ("experiment", "inputs", "characterize"))
    t = get(cfg, "characterize", nothing)
    t isa AbstractDict || throw(ArgumentError("$path: no [characterize] table"))
    VgtkCore.check_keys(t, CHARACTERISE_KEYS, "$path [characterize]")
    foreach(k -> haskey(t, k) || throw(ArgumentError("$path [characterize] lacks `$k`")), CHARACTERISE_KEYS)
    t["dark"] isa AbstractString || throw(ArgumentError("$path [characterize] dark: the E1 burst's folder name"))
    t["false_defects"] isa Real && 0 < t["false_defects"] < 1 || throw(ArgumentError("$path [characterize] false_defects: in (0, 1)"))
    t["tier"] isa AbstractString && !isempty(t["tier"]) || throw(ArgumentError("$path [characterize] tier: the campaign's tier name"))
    sort(collect(keys(cfg["inputs"]))) == sort(collect(CHARACTERISATION_INPUTS)) ||
        throw(ArgumentError("$path [inputs]: exactly $(join(CHARACTERISATION_INPUTS, ", "))"))
    all(t -> t["stage"] == "evaluate", values(cfg["inputs"])) || throw(ArgumentError("$path [inputs]: every input is of stage evaluate"))
    return cfg
end

"The one EXIF exposure of a burst of frames (Features' `exposure_settings`): an `ArgumentError` when its frames state several."
function burst_exposure(frames)
    ts = unique(VgtkFeatures.exposure_settings(f).exposure_time for f in frames)
    length(ts) == 1 || throw(ArgumentError("a burst of $(length(frames)) frames states $(length(ts)) EXIF exposures: $ts"))
    return only(ts)
end

"""
    exposure_index(bursts, tree) -> Dict{exposure, folder}

The bursts of a tree (`folder => EXIF exposure`) indexed by their exposure: ONE burst per exposure, two at one exposure an
`ArgumentError` naming both folders (`tree` names the tree) — never one silently replacing the other.
"""
function exposure_index(bursts, tree::AbstractString)
    idx = Dict{Any, String}()
    for (d, t) in bursts
        haskey(idx, t) && throw(ArgumentError("$tree: the bursts $(idx[t]) and $d are both at the EXIF exposure $t: one burst per exposure"))
        idx[t] = String(d)
    end
    return idx
end
"The EXIF exposure of the burst in `dir` (its frames' headers alone, Features' `load_raw_metadata`; `burst_exposure`)."
burst_exposure(dir::AbstractString) = burst_exposure([VgtkFeatures.load_raw_metadata(f) for f in raw_files(dir)])

"The bursts of a tree (`root/<folder>/*.ARW`): `folder => path`, every subdirectory holding RAW files."
bursts(root::AbstractString) =
    sort([d => joinpath(root, d) for d in readdir(root) if isdir(joinpath(root, d)) && !isempty(raw_files(joinpath(root, d)))]; by = first)

"""
    characterize_sensor(dark_root, ptc_root; dark, false_defects) -> (; characterization, fit, black, read_noise, defects, exposures)

THE computation of the experiment (this file's header): the E1 burst `dark_root/<dark>`'s master dark, every E1_E2 burst's master
dark (its Dark, defect mask and read noise; the black slot a `Darks` of them all, the E1 dark's channel black shared; `black` that
channel black), the E4 series' photon transfer, into ONE `SensorCharacterization` of the frames' body at their ISO, by Core's builder
`DarksAndFlats` (its `MasterDark` at `false_defects`, the E1 burst the reference). Each burst is read once (Features'
`load_raw_mosaic`), keyed by its EXIF exposure (its headers read first, `exposure_index`: two bursts of one tree at one exposure are
refused); every burst is held at once (the builder's data). `fit` is the series' PhotonTransfer FitResult (its metadata: the
range, the Birge ratio).
"""
function characterize_sensor(dark_root::AbstractString, ptc_root::AbstractString; dark::AbstractString, false_defects::Real)
    load(dir) = [VgtkFeatures.load_raw_mosaic(f) for f in raw_files(dir)]
    dark_bursts = bursts(dark_root); ptc_bursts = bursts(ptc_root)
    exposure_index([d => burst_exposure(dir) for (d, dir) in dark_bursts], "the dark tree $dark_root")    # one burst per exposure, or refused
    exposure_index([d => burst_exposure(dir) for (d, dir) in ptc_bursts], "the photon-transfer tree $ptc_root")
    t1 = burst_exposure(joinpath(dark_root, dark))                                      # the reference: the E1 burst's exposure (its headers)
    darks = Dict(burst_exposure(fr) => fr for fr in (load(dir) for (_, dir) in dark_bursts))
    flats = Dict(burst_exposure(fr) => fr for fr in (load(dir) for (_, dir) in ptc_bursts))
    first_frame = first(darks[t1])
    body = VgtkFeatures.body_identity(first_frame); iso = VgtkFeatures.exposure_settings(first_frame).iso
    # Core's builder of a characterization from its darks and flats: every burst's MasterDark, their defect onsets, the series'
    # PhotonTransfer, the reference burst (the E1 dark) the shared black, the read noise and the series' linearisation
    r = VgtkCore.solve(VgtkCore.EstimationProblem(VgtkCore.SensorCharacterization, (darks, flats), (; body, iso, reference = t1)),
                       VgtkCore.DarksAndFlats(; dark = VgtkCore.MasterDark(; false_defects)))
    c = r.u; fit = r.metadata.series; md = r.metadata.darks[t1]
    return (; characterization = c, fit, black = md.u.black, read_noise = md.metadata.read_noise, defects = c.entries[Float64(iso)].defects,
            exposures = sort!(collect(keys(r.metadata.darks)); by = t -> VgtkCore.ustrip(VgtkCore.parse_unit("s"), t)))
end

"""The SHA-256 of a set's rows of the tier's MANIFEST.tsv (`<tier>/<set>/…` rows, in the file's order, newline-joined): the record's
provenance names each set it read by it."""
manifest_sha(manifest::AbstractString, tier::AbstractString, set::AbstractString) =
    VgtkCore.sha256_hex(join(filter(l -> startswith(l, "$tier/$set/"), readlines(manifest)), '\n'))

"The summary table's header: per colour index the black and read noise with their standard errors."
const CHARACTERISATION_COLUMNS = ("colour_index", "black", "black_se", "read_noise", "read_noise_se")
