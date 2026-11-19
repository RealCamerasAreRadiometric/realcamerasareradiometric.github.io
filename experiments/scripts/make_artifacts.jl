#!/usr/bin/env julia
# =============================================================================
# scripts/make_artifacts.jl — AUTHOR-SIDE: pack the tiers of configs/data_artifacts.toml from data_root() into lazy Julia
# artifacts (VisualGeometryToolkitEval DataArtifacts `pack_artifacts`, Eval's `tree` entries with a `filter`) and bind them
# into experiments/Artifacts.toml (tree hash, the tarball's sha256, the URL on the record's host — Hugging Face until publication,
# Zenodo then — the record's placeholder until the tarballs are uploaded). NEVER uploads (data/README.md).
#
#   VGTK_DATA_ROOT=<the author's data root> julia --project=experiments experiments/scripts/make_artifacts.jl --dry-run
#   VGTK_DATA_ROOT=… julia --project=experiments experiments/scripts/make_artifacts.jl [--only=rc_calibration_20261029,…]
#       [--stage=<dir> (default output/artifacts, git-ignored)] [--record=rc=<Hugging Face commit | Zenodo id>] [--local-toml=<path>]
#
# --dry-run lists every tier: its files, their bytes, the overlap with the other tiers, and the files of data_root() that no
# tier holds; nothing is packed. A tier's files are those of its packed version: files at or below a path of the catalog's
# [[unpacked.<tier>]] (its NEXT version's) are counted apart, and are in no tier's count and not "in no tier" either; a cache
# directory (a CACHEDIR.TAG, src/data.jl is_cache_dir: the Hugging Face CLI's .cache/huggingface/) is skipped. --local-toml writes a copy of the bindings with file:// URLs to the staged tarballs (the
# fresh-machine test: EXPERIMENTS_ARTIFACTS_TOML names it to scripts/fetch_data.jl; never committed). A staged tarball whose file
# listing is current is reused.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("data.jl"))
using .DataTiers: tier_name, unpacked_entries, is_unpacked, is_cache_dir
using VisualGeometryToolkitEval: cli_option
using VisualGeometryToolkitEval.DataArtifacts: load_catalog, artifact_files, pack_artifacts, download_url

const CAT = load_catalog(projectdir("configs", "data_artifacts.toml"))
CAT.artifacts_toml == projectdir("Artifacts.toml") ||
    error("make_artifacts binds the committed experiments/Artifacts.toml; the catalog resolves to $(CAT.artifacts_toml) (unset EXPERIMENTS_ARTIFACTS_TOML)")
const ONLY = filter(!isempty, split(cli_option(ARGS, "only", ""), ','))
const NAMES = isempty(ONLY) ? sort(collect(keys(CAT.artifacts))) : String.(ONLY)

gb(b) = round(b / 1e9; digits = 3)

if "--dry-run" in ARGS
    println("data_root() = ", data_root())
    held = Set{String}()
    unpacked = unpacked_entries(projectdir("configs", "data_artifacts.toml"))
    packed(n) = (f = artifact_files(CAT, n); later(p) = is_unpacked(unpacked, tier_name(n), p);
                 (; files = filter(!later, f.files), pieces = filter(q -> !later(q.src), f.pieces)))
    files = Dict(n => packed(n) for n in NAMES)
    for n in NAMES
        f = files[n]
        bytes = sum(p -> p.len, f.pieces; init = 0)
        union!(held, f.files)
        others = [m for m in NAMES if m != n && !isempty(intersect(Set(f.files), Set(files[m].files)))]
        println(rpad(n, 28), lpad(length(f.files), 6), " files ", lpad(gb(bytes), 9), " GB",
                isempty(others) ? "" : "   shares files with $(join(others, ", "))")
    end
    if isempty(ONLY)
        caches = String[]
        all_files = String[]
        for (r, _, fs) in walkdir(data_root(); follow_symlinks = true)
            any(c -> r == c || startswith(r, c * "/"), caches) && continue
            is_cache_dir(r) && (push!(caches, r); continue)
            append!(all_files, relpath(joinpath(r, x), data_root()) for x in fs)
        end
        foreach(c -> println("cache skipped (CACHEDIR.TAG): ", relpath(c, data_root()), "/"), caches)
        later = filter(p -> is_unpacked(unpacked, nothing, p), all_files)
        for (t, es) in sort(collect(unpacked); by = first), e in es, q in e.paths
            k = count(p -> p == q || startswith(p, q * "/"), later)
            println(lpad(k, 6), " files [unpacked.", t, "] ", q, k == 0 ? " (not in this data root)" : "")
        end
        rest = sort(setdiff(all_files, held, later))
        println(length(rest), " files of data_root() in no tier", isempty(rest) ? "" : ":")
        for d in sort(unique(dirname.(rest))); println("    ", d, "/ (", count(==(d), dirname.(rest)), ")"); end
    end
    exit(0)
end

const STAGE = abspath(cli_option(ARGS, "stage", projectdir("output", "artifacts")))
const RECORD_IDS = Dict(String(first(kv)) => String(last(kv))
                        for kv in (split(p, '=') for p in filter(!isempty, split(cli_option(ARGS, "record", ""), ','))))
built = pack_artifacts(CAT, NAMES; stage = STAGE, record_ids = RECORD_IDS, local_toml = cli_option(ARGS, "local-toml", ""))
for b in built
    println(rpad(b.name, 28), lpad(gb(b.raw), 9), " GB raw ", lpad(gb(b.bytes), 9), " GB tar.gz  tree ", bytes2hex(b.hash.bytes),
            "  sha256 ", b.sha, "  ", download_url(CAT, b.name; record_ids = RECORD_IDS))
end
println("MAKE_ARTIFACTS_DONE ", CAT.artifacts_toml, " (tarballs: ", joinpath(STAGE, "tarballs"), ")")
