#!/usr/bin/env julia
# reproduce: reads configs/experiments/image_regions.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/image_regions.jl — the lens coverage of the paper's capture sets: ONE keyed step per measured image region (owner ruling
# 2026-11-15; src/image_region.jl). Every config of record states each set's coverage (`lens_coverage`): "sensor" (the lens lights the
# whole sensor; no step, the detector never runs) or "image_region:<name>", the step of its [image_regions.<name>] table — the set's own
# frames, each pinned by its sha256, and the detector's settings. This driver measures every region the configs of its config of record
# name (configs/experiments/image_regions.toml [measure] configs), once per step: two configs naming the same frames and settings name
# the same step (its key). The consumers — the refinements' starts (bin/refine.jl's camera.lens_coverage), the front ends' bounded
# cameras, the chrome evaluation's radial frame, the panorama cameras — only read the record, and one whose record is missing fails.
#
# STAGES (each step keyed by its frames, its settings and the code it runs; <results>/image_region/detect/<key>/)
#   detect   RadiometricCamera's measure_image_region (VisualGeometryToolkitFeatures' detect_image_circle on the region's geometry image, raw_green: the bilinear green on the raw grid,
#            on the device) → image_region.json (rc/image_region/v1.0, through IO: the ellipse, the photographs' size, the frames
#            and the fit's diagnostics)
#
#   julia --project=experiments experiments/scripts/image_regions.jl [--config=<toml>] [--plan] [--stage=detect] [--force]
# On a GPU (detect_image_circle's resampling, device = RadiometricCamera.default_backend()): CUDA_VISIBLE_DEVICES=0.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))

const DRIVER = IMAGE_REGIONS_DRIVER
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir(IMAGE_REGIONS_CONFIG)))   # default: the config of record
const CFG = load_image_regions_config(CFG_PATH)
"The steps to measure, one per key: `key => (cfg, name, path)` — the config that names the region, its name and the config's path."
const REGIONS = image_region_products(CFG)
const GPU_STAGES = [:detect]

const STAGES = [
    Stage(:detect, "the capture sets' image regions: measure_image_region on each set's own frames (image_region.json)";
          items = () -> sort(collect(keys(REGIONS))), product = k -> image_region_product(REGIONS[k].cfg, REGIONS[k].name),
          run = k -> run_image_region(REGIONS[k].cfg, REGIONS[k].name; driver = DRIVER, config_path = REGIONS[k].path)),
]

run_stages("image_regions", STAGES)
