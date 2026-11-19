#!/usr/bin/env julia
# =============================================================================
# scripts/fetch_data.jl — fetch the paper's published input data (the tiers of configs/data_artifacts.toml: lazy Julia artifacts
# of experiments/Artifacts.toml, on the record's host) and link them where the drivers read them, data_root() (data/README.md). The new-machine
# step of REPRODUCE.md; a no-op where the data are already bound.
#
#   julia --project=experiments experiments/scripts/fetch_data.jl --all                       every tier
#   julia --project=experiments experiments/scripts/fetch_data.jl --only=rc_calibration_20261029,rc_synthetic_20261029
#   julia --project=experiments experiments/scripts/fetch_data.jl --list                      the catalog, nothing fetched
#   julia --project=experiments experiments/scripts/fetch_data.jl                             the required tiers
#
# Each tier is downloaded once into the Julia depot (sha256 and tree hash verified; a placeholder URL — the record not yet
# published — is refused) and linked file by file into data/exp_raw, made an artifact view; an existing binding wins
# (VGTK_DATA_ROOT, or the author's own data/exp_raw) and is never written to (VisualGeometryToolkitEval DataArtifacts
# `fetch_artifacts!`).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("data.jl"))
using .DataTiers: tier_name
using VisualGeometryToolkitEval: cli_option
using VisualGeometryToolkitEval.DataArtifacts: load_catalog, required_artifacts, fetch_artifacts!

const CAT = load_catalog(projectdir("configs", "data_artifacts.toml"))

if "--list" in ARGS
    meta = isfile(CAT.artifacts_toml) ? TOML.parsefile(CAT.artifacts_toml) : Dict{String,Any}()
    for (n, a) in sort(collect(CAT.artifacts); by = first)
        println(rpad(n, 28), get(a, "required", false) ? " required " : " optional ", "tier ", rpad(tier_name(n), 16), a["what"])
        println(" "^28, "  ", haskey(meta, n) ? meta[n]["download"][1]["url"] : "(not bound)")
    end
    exit(0)
end

selected = filter(!isempty, split(cli_option(ARGS, "only", ""), ','))
wanted = !isempty(selected) ? String.(selected) : "--all" in ARGS ? sort(collect(keys(CAT.artifacts))) : required_artifacts(CAT)
for r in fetch_artifacts!(CAT, wanted)
    println(rpad(r.name, 28), " ", r.dst === nothing ? "not linked: the data root is bound ($(data_root()))" : "$(r.linked) new link(s) under $(r.dst)")
end
println("data_root() = ", data_root())
println("FETCH_DATA_DONE")
