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
    using .DataTiers: tier_of, entries_of_record, tier_entries, unpacked_entries, PRIVATE_PHOTOGRAPHS, SAME_NAME_PHOTOGRAPHS, private_photographs
    using VisualGeometryToolkitEval.DataArtifacts: load_catalog, catalog_violations, artifact_files, required_artifacts
    using VisualGeometryToolkitCore: artifact_store

    const CATALOG = load_catalog(projectdir("configs", "data_artifacts.toml"))
    const TIERS = tier_entries(CATALOG)                                   # tier => [entry, …] (the entries of record)
    const OF_RECORD = entries_of_record(CATALOG)
    const RETIRED = sort(collect(setdiff(keys(CATALOG.artifacts), OF_RECORD)))

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

    """The files each entry of record selects (Eval's own `artifact_files`, its filter rule) from a PROBE data root: an empty file at
    every path of `paths` (a path without an extension, a directory input, gets one file `_probe` below it). `entry => Set(files)`."""
    function probe_selection(paths)
        root = mktempdir()
        for p in paths
            f = isempty(splitext(p)[2]) ? joinpath(root, p, "_probe") : joinpath(root, p)
            mkpath(dirname(f)); touch(f)
        end
        return withenv("VGTK_DATA_ROOT" => root) do
            Dict(n => Set(artifact_files(CATALOG, n).files) for n in OF_RECORD if isdir(joinpath(root, CATALOG.artifacts[n]["dir"])))
        end
    end
    "Whether `sel` (a [`probe_selection`](@ref) set) holds the probe of `p`: the file itself, or a file below a directory input."
    holds(sel, p) = p in sel || any(f -> startswith(f, p * "/"), sel)

    "The installed tree of entry `n` in the artifact store, or `nothing` where the store does not hold it (or none resolves)."
    function installed_tree(n)
        store = try artifact_store() catch; return nothing end
        meta = TOML.parsefile(CATALOG.artifacts_toml)
        haskey(meta, n) || return nothing
        t = joinpath(store.path, meta[n]["git-tree-sha1"])
        return isdir(t) ? t : nothing
    end

    const UNPACKED = unpacked_entries(projectdir("configs", "data_artifacts.toml"))

    files_under(root, p) = isfile(joinpath(root, p)) ? [p] :
        [relpath(joinpath(r, f), root) for (r, _, fs) in walkdir(joinpath(root, p); follow_symlinks = true) for f in fs]
end

@testitem "published data (configs/data_artifacts.toml)" default_imports = false setup = [PublishedData] begin
    using Test
    using DrWatson
    using TOML: TOML
    using .PublishedData: CATALOG, TIERS, OF_RECORD, RETIRED, UNPACKED, config_inputs, probe_selection, holds, tier_of,
        catalog_violations, required_artifacts

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
        inside(p) = !isabspath(p) && !startswith(p, "~") && !(".." in splitpath(p))
        sel = probe_selection([p for (_, _, _, p) in ins if inside(p)])
        for (cfg, id, tier, p) in ins
            held = inside(p) && any(n -> holds(get(sel, n, Set{String}()), p), get(TIERS, tier, String[]))
            held || @info "input outside data_root() or in no entry of its tier" cfg id tier p
            @test held
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

# The privacy rule: the excluded photographs are in no entry and no installed tree; and each installed tree has the catalog's file count.
@testitem "published data: the excluded photographs are never published" default_imports = false setup = [PublishedData] begin
    using Test
    using .PublishedData: CATALOG, OF_RECORD, PRIVATE_PHOTOGRAPHS, SAME_NAME_PHOTOGRAPHS, private_photographs, probe_selection, installed_tree

    @testset "the rule: every prefix, directory and extension variant" begin
        names = ["a/b/DSC06045.ARW", "DSC06052.png", "x/_DSC06053.ARW", "x/dsc06054.arw", "DSC06055.ARW.xmp", "y/DSC06055.JPG",
                 "paper-data/geometric-accuracy/14mm/siemens-stars/DSC06045.ARW", "set/DSC06053/features.png"]
        @test private_photographs(names) == names
        @test isempty(private_photographs(["DSC06044.ARW", "DSC06046.ARW", "DSC06051.png", "DSC06056.ARW", "DSC06044/x.txt"]))
    end

    @testset "the exception: five other photographs of the same names, at exactly their paths" begin
        d = "paper-data/board-warping/april-board/fisheye"       # the fisheye AprilGrid set of the board-warping session
        @test collect(SAME_NAME_PHOTOGRAPHS) == [d * "/" * x * ".ARW" for x in PRIVATE_PHOTOGRAPHS]
        @test isempty(private_photographs(collect(SAME_NAME_PHOTOGRAPHS)))
        @test isempty(private_photographs(["./" * p for p in SAME_NAME_PHOTOGRAPHS]))                # a tarball's listing
        # every other spelling in that directory, and the names in every other directory, stay excluded
        other = [d * "/_DSC06045.ARW", d * "/DSC06045.png", d * "/dsc06045.arw", d * "/DSC06045.ARW.xmp", d * "/DSC06045/features.png",
                 d * "/sub/DSC06045.ARW", "/abs/" * first(SAME_NAME_PHOTOGRAPHS), "x/" * first(SAME_NAME_PHOTOGRAPHS),
                 "paper-data/board-warping/april-board/14mm/DSC06045.ARW", "paper-data/board-warping/siemens-board/fisheye/DSC06052.ARW",
                 "paper-data/geometric-accuracy/14mm/siemens-stars/DSC06055.ARW"]
        @test private_photographs(other) == other
    end

    @testset "no entry selects one (Eval's selection of a probe tree holding them in every entry's directory)" begin
        dirs = unique(CATALOG.artifacts[n]["dir"] for n in OF_RECORD)
        sel = probe_selection([normpath(joinpath(d, f)) for d in dirs for x in PRIVATE_PHOTOGRAPHS for f in (x * ".ARW", x * ".png", "a/_" * x * ".JPG", x * "/features.png")])
        for (n, files) in sel
            @test isempty(private_photographs(files))
        end
        # every entry of record also names the exclusion itself, so a widened include glob cannot let one in. The one entry of the
        # five other photographs of the same names lists exactly their paths and no glob (src/data.jl SAME_NAME_PHOTOGRAPHS)
        includes(n) = filter(g -> !startswith(g, "!"), CATALOG.artifacts[n]["filter"])
        same = [n for n in OF_RECORD if includes(n) == collect(SAME_NAME_PHOTOGRAPHS)]
        @test length(same) == 1 && CATALOG.artifacts[only(same)]["files"] == length(SAME_NAME_PHOTOGRAPHS)
        for n in OF_RECORD
            @test all(x -> "!**/*$(x)*/**" in CATALOG.artifacts[n]["filter"], PRIVATE_PHOTOGRAPHS)
            @test n in same || all(x -> "!**/*$(x)*" in CATALOG.artifacts[n]["filter"], PRIVATE_PHOTOGRAPHS)
            @test n in same || isempty(intersect(includes(n), SAME_NAME_PHOTOGRAPHS))
        end
    end

    @testset "no installed tree holds one, and each has the catalog's file count (where the artifact store holds the trees)" begin
        trees = filter(x -> last(x) !== nothing, [n => installed_tree(n) for n in OF_RECORD])
        isempty(trees) && @info "privacy: the artifact store holds no tree of record here; the installed-tree checks are skipped"
        for (n, t) in trees
            files = [relpath(joinpath(r, f), t) for (r, _, fs) in walkdir(t) for f in fs]
            @test isempty(private_photographs(files))
            @test length(files) == CATALOG.artifacts[n]["files"]
        end
    end
end

# Where data_root() holds the inputs, every input is there. A :data item, its checks skipped (recorded) where it does not.
@testitem "published data: the inputs under data_root()" default_imports = false tags = [:data] setup = [PublishedData] begin
    using Test
    using .PublishedData: CATALOG, TIERS, OF_RECORD, config_inputs, files_under, data_root, probe_selection, holds

    root = try data_root() catch; nothing end
    if root === nothing || !all(n -> ispath(joinpath(root, CATALOG.artifacts[n]["dir"])), filter(n -> CATALOG.artifacts[n]["required"], OF_RECORD))
        @info "published data: data_root() does not hold the required tiers (fetch them: scripts/fetch_data.jl); its checks are skipped"
    else
        ins = [x for x in config_inputs() if haskey(TIERS, x[3])]
        sel = probe_selection(unique(x[4] for x in ins))
        for (cfg, id, tier, p) in ins
            # an input only optional entries hold (not fetched by default) may be absent
            all(n -> !CATALOG.artifacts[n]["required"], filter(n -> holds(get(sel, n, Set{String}()), p), TIERS[tier])) &&
                !ispath(joinpath(root, p)) && continue
            ispath(joinpath(root, p)) || @info "an input is not under data_root()" cfg id p
            @test ispath(joinpath(root, p)) && (id == "[experiment] dataset" || !isempty(files_under(root, p)))
        end
    end
end
