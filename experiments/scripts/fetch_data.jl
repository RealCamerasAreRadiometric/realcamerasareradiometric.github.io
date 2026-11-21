#!/usr/bin/env julia
# =============================================================================
# scripts/fetch_data.jl — the new-machine step of REPRODUCE.md: install the paper's published input data (the tiers of
# configs/data_artifacts.toml, lazy Julia artifacts of experiments/Artifacts.toml) into the artifact store and link them where the
# drivers read them, data_root() (data/README.md). VisualGeometryToolkitEval DataArtifacts `fetch_data!`, the one way the inputs
# reach a machine; a no-op where the data are already bound.
#
#   julia --project=experiments experiments/scripts/fetch_data.jl                 the required tiers
#   julia --project=experiments experiments/scripts/fetch_data.jl --all           every tier (the retired ones too: the private dataset)
#   julia --project=experiments experiments/scripts/fetch_data.jl --only=rc_targets_20261121,rc_golden_20261121
#   julia --project=experiments experiments/scripts/fetch_data.jl --list          the catalog, nothing fetched
#
# Each tree is installed once into the artifact store (VisualGeometryToolkitCore's artifact_store: VGTK_ARTIFACT_STORE, else the
# site profile VGTK_SITE names; never a cluster's depot), its sha256 and tree hash verified; a record not yet published (its
# placeholder URL) is refused unless the store already holds the tree. The trees are linked file by file into data/exp_raw, made an
# artifact view; an existing binding wins (VGTK_DATA_ROOT, the site profile's data root, or the author's own data/exp_raw) and is
# never written to.
# =============================================================================
using DrWatson
@quickactivate "experiments"
using VisualGeometryToolkitEval.DataArtifacts: fetch_data!, project_catalog

fetch_data!(project_catalog())
