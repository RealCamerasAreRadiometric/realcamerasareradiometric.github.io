#!/usr/bin/env julia
# reproduce: reads configs/experiments/panorama_vignetting_fx3_14mm.toml
# reproduce: reads configs/experiments/panorama_vignetting_fx3_fisheye.toml
# reproduce: reads configs/experiments/panorama_vignetting_fx3_tamron.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/panorama_vignetting.jl — a rotating camera's sweep, registered: the chain of PanoramaStitching's example of record,
# examples/panorama_vignetting/run.jl, run from a config of record of this project
# (configs/experiments/panorama_vignetting_<set>.toml). The driver binds the example's inputs to the data tiers ([inputs], under
# data_root()), writes the example's config ([example], the inputs' resolved paths) into the run and calls the example's
# `run_example` — nothing of the chain is copied here (src/panorama_example.jl binds it). The example writes its outputs into its
# own keyed products, <results>/panorama_vignetting/<name>/<product>/<key>/ (Eval's results root and product keys).
# scripts/panorama_stitching.jl reads the sweep's run as its [upstream] (its decoded frames, rotations and lens).
#
# STAGES (the run keyed by its config record and its inputs; <results>/panorama_vignetting/evaluate/<key>/, the example's config)
#   evaluate   the example's chain: the RAW frames decoded (VisualGeometryToolkitFeatures' load_raw, cached in the example's set
#              folder), their rotations (register_rotations), the sweep's own vignetting (fit_vignetting, interpolation free), the
#              seam scores before and after (score_seams), the corrected and uncorrected panoramas (render_panorama); the fit's
#              record, summary.md and run.log in the example's run
#
#   julia --project=experiments experiments/scripts/panorama_vignetting.jl --config=<toml> [--plan] [--stage=evaluate] [--force]
# On a GPU ([example.compute] device = "cuda"): CUDA_VISIBLE_DEVICES=0.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("panorama_example.jl"))

const DRIVER = "scripts/panorama_vignetting.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "panorama_vignetting_fx3_14mm.toml")))
const CFG = load_panorama_vignetting_config(CFG_PATH)
"Whether the chain runs on the GPU ([example.compute] device)."
const GPU_STAGES = CFG["example"]["compute"]["device"] == "cuda" ? [:evaluate] : Symbol[]

evaluate_product() = panorama_vignetting_product(CFG)

evaluate(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    path = write_panorama_example_config(CFG, dir)
    run = panorama_example_call(:run_example, path)            # the example's own keyed run (its directory)
    # an outcome the config does not state fails the stage loudly (the run's outputs stay for the diagnosis, the run is not complete):
    # the registration the example's run produced, read back by its settings' key (the frames from its decode cache)
    check_registration(CFG, example_registration(panorama_example_call(:load_config, path)).reg; where = "$CFG_PATH evaluate")
    Produced(; inputs = filter(!=(RadiometricCamera.SENSOR_LENS_COVERAGE),
                               [config_input(CFG, "raw"), config_input(CFG, "lens"), config_input(CFG, "sensor_characterisation"),
                                lens_coverage_record(CFG, CFG["lens_coverage"]; what = "the sweep"), run]))
end

const STAGES = [
    Stage(:evaluate, "the example's chain: decode, rotations, fit_vignetting, score_seams, panoramas (summary.md)";
          items = () -> [evaluate_product()], product = identity, run = evaluate),
]

run_stages("panorama_vignetting", STAGES)
