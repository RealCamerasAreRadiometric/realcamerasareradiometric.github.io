#!/usr/bin/env julia
# =============================================================================
# scripts/make_artifacts.jl — AUTHOR-SIDE: pack the tiers of configs/data_artifacts.toml from data_root() into lazy Julia
# artifacts (VisualGeometryToolkitEval DataArtifacts `pack_artifacts`, Eval's `tree` entries with a `filter`) and bind them
# into experiments/Artifacts.toml (tree hash, the tarball's sha256, the URL on the record's host — the Zenodo records of the tiers
# of record — the record's placeholder until the tarballs are uploaded). NEVER uploads (data/README.md).
#
#   VGTK_DATA_ROOT=<the author's data root> julia --project=experiments experiments/scripts/make_artifacts.jl --dry-run
#   VGTK_DATA_ROOT=… julia --project=experiments experiments/scripts/make_artifacts.jl [--only=rc_targets_20261121,…]
#       [--stage=<dir> (default output/artifacts, git-ignored)] [--record=<record>=<Zenodo id>,…] [--local-toml=<path>] | --check [--stage=<dir>]
#
# --dry-run lists every entry of record (the retired ones with --only): its files, their bytes, the overlap with the other entries,
# and the files of data_root() that no entry holds (the catalog's [[unpacked]] files counted apart; a cache directory, a
# CACHEDIR.TAG, src/data.jl is_cache_dir, skipped). PRIVACY (src/data.jl PRIVATE_PHOTOGRAPHS): an entry holding one of the excluded
# photographs is refused before anything is packed, and every packed tarball's listing is checked again; --check lists every staged
# tarball under --stage against the rule (the step before any upload), exit 1 on a hit. --local-toml writes a copy of the bindings
# with file:// URLs to the staged tarballs (EXPERIMENTS_ARTIFACTS_TOML names it to scripts/fetch_data.jl, which installs the trees
# into the artifact store from them; never committed). A staged tarball whose file listing is current is reused.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("data.jl"))
using .DataTiers: tier_of, unpacked_entries, is_unpacked, is_cache_dir, private_photographs
using VisualGeometryToolkitEval: cli_option
using VisualGeometryToolkitEval.DataArtifacts: load_catalog, artifact_files, pack_artifacts, download_url

const CAT = load_catalog(projectdir("configs", "data_artifacts.toml"))
CAT.artifacts_toml == projectdir("Artifacts.toml") ||
    error("make_artifacts binds the committed experiments/Artifacts.toml; the catalog resolves to $(CAT.artifacts_toml) (unset EXPERIMENTS_ARTIFACTS_TOML)")
const ONLY = filter(!isempty, split(cli_option(ARGS, "only", ""), ','))
const NAMES = isempty(ONLY) ? sort([n for (n, a) in CAT.artifacts if get(a, "retired", false) !== true]) : String.(ONLY)
const STAGE = abspath(cli_option(ARGS, "stage", projectdir("output", "artifacts")))

"The paths a staged tarball holds (`tar -tzf`)."
tarball_paths(t) = filter(!isempty, split(read(`tar -tzf $t`, String), '\n'))

if "--check" in ARGS
    dir = joinpath(STAGE, "tarballs")
    tbs = isdir(dir) ? sort(filter(f -> endswith(f, ".tar.gz"), readdir(dir; join = true))) : String[]
    isempty(tbs) && (println("no staged tarball under $dir"); exit(1))
    bad = [(t, p) for t in tbs for p in private_photographs(tarball_paths(t))]
    foreach(((t, p),) -> println("PRIVATE ", basename(t), " holds ", p), bad)
    println(length(tbs), " tarball(s) checked, ", length(bad), " excluded photograph(s)")
    exit(isempty(bad) ? 0 : 1)
end

gb(b) = round(b / 1e9; digits = 3)

if "--dry-run" in ARGS
    println("data_root() = ", data_root())
    held = Set{String}()
    unpacked = unpacked_entries(projectdir("configs", "data_artifacts.toml"))
    packed(n) = artifact_files(CAT, n)
    files = Dict(n => packed(n) for n in NAMES)
    for n in NAMES
        f = files[n]
        bytes = sum(p -> p.len, f.pieces; init = 0)
        union!(held, f.files)
        others = [m for m in NAMES if m != n && !isempty(intersect(Set(f.files), Set(files[m].files)))]
        priv = private_photographs(f.files)
        println(rpad(n, 44), lpad(length(f.files), 6), " files ", lpad(gb(bytes), 9), " GB  tier ", tier_of(CAT, n),
                isempty(others) ? "" : "   shares files with $(join(others, ", "))", isempty(priv) ? "" : "   PRIVATE: $(join(priv, ", "))")
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

const RECORD_IDS = Dict(String(first(kv)) => String(last(kv))
                        for kv in (split(p, '=') for p in filter(!isempty, split(cli_option(ARGS, "record", ""), ','))))
for n in NAMES                                  # privacy first: nothing is packed while an entry holds an excluded photograph
    priv = private_photographs(artifact_files(CAT, n).files)
    isempty(priv) || error("make_artifacts: $n holds the excluded photograph(s) $(join(priv, ", ")) (src/data.jl PRIVATE_PHOTOGRAPHS): fix its filter")
end
built = pack_artifacts(CAT, NAMES; stage = STAGE, record_ids = RECORD_IDS, local_toml = cli_option(ARGS, "local-toml", ""))
for b in built
    priv = private_photographs(tarball_paths(b.tarball))
    isempty(priv) || (rm(b.tarball); error("make_artifacts: the tarball of $(b.name) holds $(join(priv, ", ")): removed, nothing bound may be published"))
    println(rpad(b.name, 44), lpad(gb(b.raw), 9), " GB raw ", lpad(gb(b.bytes), 9), " GB tar.gz  tree ", bytes2hex(b.hash.bytes),
            "  sha256 ", b.sha, "  ", download_url(CAT, b.name; record_ids = RECORD_IDS))
end
println("MAKE_ARTIFACTS_DONE ", CAT.artifacts_toml, " (tarballs: ", joinpath(STAGE, "tarballs"), ")")
