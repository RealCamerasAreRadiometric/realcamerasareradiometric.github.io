# Test board: a 256² BlobBoards.jl pattern (seed 0x1234, min σ 3 px, cutoff 2σ; 99 blobs), generated
# with `julia --project=blobboards scripts/blobboards_board.jl generate --size 256 --min-scale 3 --seed 0x1234`
# and checked in as PNG + bundle JSON (BlobBoards' schema).
using VisualGeometryToolkitFeatures: load_linear_image
const FIXTURE_DIR = joinpath(@__DIR__, "..", "fixtures", "blobboards_256")
const FIXTURE_BASE = "blob_pattern_061_ce03e"
fixture_paths() = (png = joinpath(FIXTURE_DIR, FIXTURE_BASE * ".png"), bundle = joinpath(FIXTURE_DIR, FIXTURE_BASE * ".json"))
"""Texture (`H × W × 3` Float64) of the BlobBoards test board."""
fixture_board() = load_linear_image(fixture_paths().png)        # VisualGeometryToolkitFeatures' linear PNG reader
