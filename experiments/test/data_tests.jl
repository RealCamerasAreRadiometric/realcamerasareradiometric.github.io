# =============================================================================
# The published data (configs/data_artifacts.toml, src/data.jl): the catalog loads and agrees with Artifacts.toml (A1); every entry
# of record is an Eval `tree` entry with a `filter`, a `tier`, a Zenodo record and its file count; every input of every config of
# record names a tier of record and a path inside data_root() that an entry of that tier selects; the [[unpacked]] files are read by
# no config; no driver or project source names a machine path or the retired experiments/ tree; the privacy rule (no entry, tree or
# tarball holds an excluded photograph); and, where the artifact store or data_root() holds the trees, their files.
# =============================================================================

# The catalog and the configs' inputs, shared by the items below.
@testsetup module PublishedData
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("calibration.jl"))
    include(srcdir("data.jl"))
    using .DataTiers: DataTiers, tier_of, tier_entries, unpacked_entries, PRIVATE_PHOTOGRAPHS, private_photographs
    using VisualGeometryToolkitEval.DataArtifacts: load_catalog, catalog_violations, artifact_files, required_artifacts
    import VisualGeometryToolkitEval.DataArtifacts: _glob_regex      # Eval's glob rule (the catalog's `filter`), for the static coverage

    const CATALOG = load_catalog(projectdir("configs", "data_artifacts.toml"))
    const TIERS = tier_entries(CATALOG)                                   # tier => [entry, …] (the entries of record)
    const OF_RECORD = sort(reduce(vcat, values(TIERS)))
    const RETIRED = sort([n for (n, a) in CATALOG.artifacts if get(a, "retired", false) === true])

    """Every (config, input id, tier, path) of the configs of record: the experiments' [inputs] and the calibration cases' [pipeline]
    inputs (tier rc_calibration; a case's `io.images` as the photographs of its `io.views`, and its `[experiment] dataset`, the case's
    directory, which only has to exist)."""
    function config_inputs()
        out = Tuple{String,String,String,String}[]
        for f in readdir(projectdir("configs", "experiments"); join = true)
            for (id, t) in get(TOML.parsefile(f), "inputs", Dict{String, Any}())   # (an experiment of overrides: the base's inputs)
                push!(out, (basename(f), id, t["tier"], t["path"]))
            end
        end
        for f in readdir(projectdir("configs", "calibration"); join = true)
            cfg = TOML.parsefile(f)
            views = get(get(cfg["pipeline"], "io", Dict()), "views", String[])
            for (k, p) in pipeline_inputs(cfg)
                if k == "io.images" && !isempty(views)
                    append!(out, [(basename(f), k, "rc_calibration", joinpath(p, v)) for v in views])
                else
                    push!(out, (basename(f), k, "rc_calibration", p))
                end
            end
            push!(out, (basename(f), "[experiment] dataset", "rc_calibration", cfg["experiment"]["dataset"]))
        end
        return out
    end

    "Whether entry `n`'s `filter` selects the data_root()-relative path `p` (a file) or every file below it (a directory input)."
    function selects(n, p)
        globs = String.(CATALOG.artifacts[n]["filter"])
        inc, exc = filter(g -> !startswith(g, "!"), globs), [g[2:end] for g in globs if startswith(g, "!")]
        any(g -> occursin(_glob_regex(g), p), exc) && return false
        return any(g -> occursin(_glob_regex(g), p) || g == p * "/**" ||
                        (endswith(g, "/**") && startswith(p, chopsuffix(g, "**"))), inc)
    end

    const UNPACKED = unpacked_entries(projectdir("configs", "data_artifacts.toml"))

    files_under(root, p) = isfile(joinpath(root, p)) ? [p] :
        [relpath(joinpath(r, f), root) for (r, _, fs) in walkdir(joinpath(root, p); follow_symlinks = true) for f in fs]
end

@testitem "published data (configs/data_artifacts.toml)" default_imports = false setup = [PublishedData] begin
    using Test
    using DrWatson
    using TOML: TOML
    using .PublishedData: CATALOG, TIERS, OF_RECORD, RETIRED, UNPACKED, config_inputs, selects, tier_of, catalog_violations,
        required_artifacts

    @testset "the catalog loads and agrees with Artifacts.toml (A1)" begin
        @test CATALOG.artifacts_toml == projectdir("Artifacts.toml")
        v = catalog_violations(CATALOG)
        foreach(x -> @info("catalog: " * x), v)
        @test isempty(v)
        @test all(a -> a["kind"] == "tree" && haskey(a, "dir") && !isempty(get(a, "filter", [])) && a["files"] > 0, values(CATALOG.artifacts))
        @test all(n -> occursin(r"^rc_[a-z0-9_]+_[0-9]{8}$", n), keys(CATALOG.artifacts))   # <tarball>_<yyyymmdd>
        # the entries of record: each names its tier and a Zenodo record; the retired ones are the private dataset's, never required
        @test all(n -> haskey(CATALOG.artifacts[n], "tier") && get(CATALOG.records[CATALOG.artifacts[n]["record"]], "host", "zenodo") == "zenodo", OF_RECORD)
        @test !isempty(RETIRED) && all(n -> CATALOG.artifacts[n]["record"] == "rc" && !CATALOG.artifacts[n]["required"], RETIRED)
        @test issubset(required_artifacts(CATALOG), OF_RECORD)
        @test all(r -> r["license"] == "CC-BY-4.0", values(CATALOG.records))
    end

    @testset "every input of every config names a tier of record, a path inside data_root(), and an entry of its tier selects it" begin
        ins = config_inputs()
        @test length(ins) > 50
        @test issubset(Set(t for (_, _, t, _) in ins), keys(TIERS))       # the configs' tier names (they key the runs), every one held
        for (cfg, id, tier, p) in ins
            inside = !isabspath(p) && !startswith(p, "~") && !(".." in splitpath(p))
            held = id == "[experiment] dataset" ? any(n -> any(g -> startswith(g, p * "/"), CATALOG.artifacts[n]["filter"]), get(TIERS, tier, String[])) :
                   any(n -> selects(n, p), get(TIERS, tier, String[]))
            (inside && held) || @info "input outside data_root() or in no entry of its tier" cfg id tier p
            @test inside && held
        end
    end

    @testset "[unpacked]: files no config of record reads" begin
        ins = config_inputs()
        for (tier, es) in UNPACKED, u in es
            @test isempty(u.configs) && !isempty(u.paths) && !isempty(u.where) && !isempty(u.provenance)
            for q in u.paths
                namers = [(cfg, id) for (cfg, id, _, p) in ins if p == q || startswith(p, q * "/") || startswith(q, p * "/")]
                isempty(namers) || @info "an [unpacked] path is named by a config" tier q namers
                @test isempty(namers)
            end
        end
    end

    @testset "no driver or project source names a machine path or the retired experiments/ tree" begin
        hits = String[]
        for dir in ("scripts", "src", "configs"), (root, _, files) in walkdir(projectdir(dir)), f in files
            any(e -> endswith(f, e), (".jl", ".toml", ".py", ".sh")) || continue
            for (i, line) in enumerate(eachline(joinpath(root, f)))
                code = strip(first(split(line, '#'; limit = 2)))                   # comments and provenance prose may name history
                occursin(r"\"/(home|Users|tmp|workspaces|mnt|data|raid|scratch|cluster)/", line) && push!(hits, "$(relpath(joinpath(root, f), projectdir())):$i")
                occursin(r"(?<!configs/)experiments/(geometric_accuracy|macro_defocus|vignetting|lib|grid_prior|focus_ab|fetch_raw_photos|raw_photos)", code) &&
                    !startswith(lstrip(line), "provenance") && !occursin(r"[Mm]oved from|[Pp]orted from", line) &&
                    push!(hits, "$(relpath(joinpath(root, f), projectdir())):$i")
            end
        end
        foreach(h -> @info("machine path / experiments/: " * h), hits)
        @test isempty(hits)
    end
end

# The privacy rule: the excluded photographs are in no entry, no installed tree and no staged tarball.
@testitem "published data: the excluded photographs are never published" default_imports = false setup = [PublishedData] begin
    using Test
    using DrWatson
    using TOML: TOML
    using .PublishedData: CATALOG, TIERS, OF_RECORD, PRIVATE_PHOTOGRAPHS, private_photographs, selects
    using VisualGeometryToolkitCore: artifact_store

    @testset "the rule: every prefix, directory and extension variant" begin
        @test PRIVATE_PHOTOGRAPHS == ("DSC06045", "DSC06052", "DSC06053", "DSC06054", "DSC06055")
        names = ["a/b/DSC06045.ARW", "DSC06052.png", "x/_DSC06053.ARW", "x/dsc06054.arw", "DSC06055.ARW.xmp", "y/DSC06055.JPG",
                 "paper-data/geometric-accuracy/14mm/siemens-stars/DSC06045.ARW"]
        @test private_photographs(names) == names
        @test isempty(private_photographs(["DSC06044.ARW", "DSC06046.ARW", "DSC06051.png", "DSC06056.ARW", "DSC060450/x.txt"]))
    end

    @testset "no entry selects one: every photograph entry excludes them, by name in every directory" begin
        for n in OF_RECORD
            excl = [g for g in CATALOG.artifacts[n]["filter"] if startswith(g, "!")]
            tier = CATALOG.artifacts[n]["tier"]
            tier == "rc_raw_photos" && @test all(x -> "!**/*$(x)*" in excl, PRIVATE_PHOTOGRAPHS)
            d = CATALOG.artifacts[n]["dir"]
            for x in PRIVATE_PHOTOGRAPHS, ext in (".ARW", ".png", ".JPG")
                p = (d == "." ? "" : d * "/") * x * ext
                tier == "rc_raw_photos" && @test !selects(n, p)
            end
            @test isempty(private_photographs(CATALOG.artifacts[n]["filter"]))  # no include glob names one
        end
    end

    @testset "no installed tree holds one (the artifact store, where it holds the trees)" begin
        store = try artifact_store() catch; nothing end
        meta = TOML.parsefile(CATALOG.artifacts_toml)
        trees = store === nothing ? String[] :
                filter(isdir, [joinpath(store.source === :depot ? joinpath(first(DEPOT_PATH), "artifacts") : store.path,
                                        meta[n]["git-tree-sha1"]) for n in keys(CATALOG.artifacts) if haskey(meta, n)])
        isempty(trees) && @info "privacy: the artifact store holds no tree of the catalog here; the installed-tree check is skipped"
        for t in trees
            files = [relpath(joinpath(r, f), t) for (r, _, fs) in walkdir(t) for f in fs]
            @test isempty(private_photographs(files))
        end
    end
end

# Where the artifact store holds a tree of record, its file count is the catalog's; where data_root() holds the inputs, every input
# is there and lies in its tier. A :data item, its checks skipped (recorded) where neither is bound.
@testitem "published data: the trees and data_root()" default_imports = false tags = [:data] setup = [PublishedData] begin
    using Test
    using DrWatson
    using TOML: TOML
    using .PublishedData: CATALOG, TIERS, OF_RECORD, config_inputs, files_under, data_root
    using VisualGeometryToolkitCore: artifact_store

    store = try artifact_store() catch; nothing end
    meta = TOML.parsefile(CATALOG.artifacts_toml)
    if store !== nothing && store.source !== :depot
        for n in OF_RECORD
            t = joinpath(store.path, meta[n]["git-tree-sha1"])
            isdir(t) || continue
            @test count(_ -> true, (f for (_, _, fs) in walkdir(t) for f in fs)) == CATALOG.artifacts[n]["files"]
        end
    end
    root = try data_root() catch; nothing end
    if root === nothing || !all(n -> ispath(joinpath(root, CATALOG.artifacts[n]["dir"])), filter(n -> CATALOG.artifacts[n]["required"], OF_RECORD))
        @info "published data: data_root() does not hold the required tiers (fetch them: scripts/fetch_data.jl); its checks are skipped"
    else
        for (cfg, id, tier, p) in config_inputs()
            haskey(TIERS, tier) || continue
            any(n -> !CATALOG.artifacts[n]["required"], TIERS[tier]) && !ispath(joinpath(root, p)) && continue   # an optional entry not fetched
            ispath(joinpath(root, p)) || @info "an input is not under data_root()" cfg id p
            @test ispath(joinpath(root, p)) && (id == "[experiment] dataset" || !isempty(files_under(root, p)))
        end
    end
end
