# =============================================================================
# src/experiments.jl — the shared prelude every driver includes after
# `@quickactivate "experiments"`: `include(srcdir("experiments.jl"))`. It binds the
# project's data trees (VisualGeometryToolkitEval Experiments: data_root() for the
# read-only inputs, results_root() for the results) and brings in the stage
# runner and the keyed step (produce: DrWatson's produce_or_load). Project code shared by several
# drivers goes here, never a data constant (those are configs/).
# =============================================================================
using DrWatson
using TOML: TOML
using StableRNGs: StableRNG
using VisualGeometryToolkitEval.Experiments: Stage, run_stages, require_input, run_key, Product, product_dir, product_output, produce, produced, Produced,
    set_project_root!, data_root, results_root, write_provenance
using VisualGeometryToolkitEval.Render: HOUSE_AXIS_COMMON, HOUSE_LINE_WIDTH, HOUSE_MARK_SIZE
import VisualGeometryToolkitIO as VgtkIO

# residual_statistics: the one summary of a residual vector (src/residual_statistics.jl, shared with src/figures.jl's module)
include("residual_statistics.jl")
# the schemas of the experiments' own result records, one table (src/record_schemas.jl; IO 0.10's write_record requires each)
include("record_schemas.jl")
# the run summaries' tables: VisualGeometryToolkitEval's tab-separated tables read back as values (src/summary_table.jl)
include("summary_table.jl")

# data/exp_raw (inputs) and data/exp_pro (results) under this project, unless
# VGTK_DATA_ROOT / VGTK_RESULTS_ROOT bind them elsewhere (data/README.md).
set_project_root!(projectdir(); create_results = true)
