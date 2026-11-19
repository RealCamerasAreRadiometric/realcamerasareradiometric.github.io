#!/usr/bin/env julia
# reproduce: reads configs/experiments/distance_maps.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/distance_maps.jl — the printed boards' edge distance maps (RadiometricCamera's edge.sdf): ONE keyed step per map of the
# config of record (src/distance_maps.jl; owner 2026-11-16: a derived file is produced, never an input of record). Each step reads the
# board generator's sdf.f32 + sdf.json (inputs of record, pinned by their sha256) and writes a VisualGeometryToolkitIO raster record
# (vgio/raster/v2.0, distance_map.json + distance_map.tiff). Every consumer — the refinements' and the calibration case's edge.sdf,
# the detector-bias ground truth's distance_map — names a map as a [runs.<id>] of this config and produces it on demand when missing.
#
# STAGES (each step keyed by its two inputs and the code it runs; <results>/distance_maps/prepare/<key>/)
#   prepare   sdf.f32 + sdf.json → distance_map.json (+ distance_map.tiff), read back bit for bit
#
#   julia --project=experiments experiments/scripts/distance_maps.jl [--config=<toml>] [--plan] [--stage=prepare] [--force]
# CPU only (a conversion of two files: no device work).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))

const DRIVER = DISTANCE_MAPS_DRIVER
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir(DISTANCE_MAPS_CONFIG)))   # default: the config of record
const CFG = load_distance_maps_config(CFG_PATH)
const GPU_STAGES = Symbol[]

const STAGES = [
    Stage(:prepare, "the boards' distance maps: the generator's sdf.f32 + sdf.json as IO raster records (distance_map.json)";
          items = () -> sort(collect(keys(CFG["maps"]))), product = n -> distance_map_product(CFG, n),
          run = n -> run_distance_map(CFG, n; driver = DRIVER, config_path = CFG_PATH)),
]

run_stages("distance_maps", STAGES)
