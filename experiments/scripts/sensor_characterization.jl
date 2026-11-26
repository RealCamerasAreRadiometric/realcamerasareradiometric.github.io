#!/usr/bin/env julia
# reproduce: reads configs/experiments/sensor_characterization.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/sensor_characterization.jl — the Sony ILME-FX3's sensor characterisation at ISO 100 (owner ruling D12: the experiment renamed; its
# dark statistics replaced by Core's MasterDark / defect_onsets), from the campaign of 2026-11-13 (E1/E2 lens-cap
# darks, E4 photon transfer): src/sensor_characterization.jl's `characterize_sensor`, Core's estimators alone.
#
# STAGES (keyed by the config record and the inputs it reads; <results>/sensor_characterization/evaluate-<key>/)
#   evaluate   the characterisation: sensor_characterization.json (vgio/sensor_characterization through IO, its dark rasters beside it),
#              channels.tsv (per colour index the black and read noise ± SE), summary.toml (K ± SE, the Birge ratio, the fitted range,
#              the defects, and whether the record equals the record of record — the configs' input fx3_sensor_characterization)
#
#   julia --project=experiments experiments/scripts/sensor_characterization.jl [--config=<toml>] [--plan] [--stage=evaluate] [--force]
#
# Data: the campaign's frames and the record of record are inputs of tier "unpublished" (sha256-pinned where one file;
# configs/data_artifacts.toml's SENSOR MODELS block). CPU only (about 570 frames decoded one burst at a time).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("sensor_characterization.jl"))
using VisualGeometryToolkitEval.Render: write_tsv
using Printf: @sprintf

const DRIVER = "scripts/sensor_characterization.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "sensor_characterization.toml")))   # default: the config of record
const CFG = load_sensor_characterization_config(CFG_PATH)
const CHAR = CFG["characterize"]
const TIER = CHAR["tier"]   # the campaign's tier name (the config's literal), as its MANIFEST.tsv rows state it

evaluate_product() = driver_product("evaluate", Dict{String, Any}("experiment" => CFG["experiment"], "characterize" => CHAR,
                                                                  "inputs" => inputs_record(CFG)); output = "summary.toml")

evaluate(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    srcs = Dict(id => config_input(CFG, id) for id in CHARACTERISATION_INPUTS)
    _evaluate(dir, srcs)
    Produced(; inputs = collect(values(srcs)))
end
function _evaluate(dir, srcs)
    r = characterize_sensor(srcs["dark_frames"], srcs["ptc_frames"]; dark = CHAR["dark"], false_defects = Float64(CHAR["false_defects"]))
    path = joinpath(dir, "sensor_characterization.json")
    VgtkIO.save(path, r.characterization; provenance = (; tier = TIER, manifest_sha = (E1_E2 = manifest_sha(srcs["manifest"], TIER, "E1_E2"),
                                                                                   E4 = manifest_sha(srcs["manifest"], TIER, "E4"))))
    back = VgtkIO.load(VgtkCore.SensorCharacterization, path)
    of_record = VgtkIO.load(VgtkCore.SensorCharacterization, srcs["record_of_record"])
    K = r.fit.u; m = r.fit.metadata
    write_tsv(joinpath(dir, "channels.tsv"), CHARACTERISATION_COLUMNS,
              [(c - 1, r.black.values[c], r.black.std_error[c], r.read_noise.σ[c], r.read_noise.std_error[c]) for c in 1:4];
              comments = ["the E1 master dark's channel black and read noise per colour index (LibRaw's 0–3), codes"])
    summary = Dict{String, Any}("gain" => K.gain, "gain_std_error" => K.std_error.gain, "birge_ratio" => m.birge_ratio,
                                "level_range" => collect(m.level_range), "defects" => length(r.defects.x), "round_trip" => back == r.characterization,
                                "equals_record_of_record" => back == of_record, "exposures_s" => [VgtkCore.ustrip(VgtkCore.parse_unit("s"), t) for t in r.exposures])
    open(io -> TOML.print(io, summary), joinpath(dir, "summary.toml"), "w")
    println(">> K = ", @sprintf("%.5f ± %.5f", K.gain, K.std_error.gain), " e⁻/code (Birge ", @sprintf("%.3f", m.birge_ratio), "), ",
            length(r.defects.x), " defects; equals the record of record: ", back == of_record, " → ", dir)
    return dir
end

const STAGES = [Stage(:evaluate, "the characterisation by Core's estimators (sensor_characterization.json, channels.tsv, summary.toml)";
                      items = () -> [evaluate_product()], product = identity, run = evaluate)]

run_stages("sensor_characterization", STAGES)
