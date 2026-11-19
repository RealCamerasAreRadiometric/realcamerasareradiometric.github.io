# =============================================================================
# The published data (configs/data_artifacts.toml, src/data.jl): the catalog loads and agrees with Artifacts.toml (A1); every input
# of every config of record names a tier of the catalog and a path inside data_root(); no driver or project source names a machine
# path or the retired experiments/ tree; every tier is an Eval `tree` entry with a `filter`; and, where data_root() holds the tiers, scripts/make_artifacts.jl
# --dry-run lists the tiers with the file counts the catalog records (their packed versions: the catalog's [[unpacked.<tier>]] files
# counted apart), the three disjoint tiers are disjoint, every file of a
# config's input lies in the input's tier, rc_calibration holds every input of the golden cases, and the raw photographs match
# their manifest.
# =============================================================================

# The catalog and the configs' inputs, shared by the two items below.
@testsetup module PublishedData
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("calibration.jl"))
    include(srcdir("data.jl"))
    using .DataTiers: DataTiers, tier_name, tier_entries, unpacked_entries
    using VisualGeometryToolkitEval.DataArtifacts: load_catalog, catalog_violations, artifact_files, required_artifacts

    const CATALOG = load_catalog(projectdir("configs", "data_artifacts.toml"))
    const TIERS = tier_entries(CATALOG)                                   # tier => artifact

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

    "The catalog's `[[unpacked.<tier>]]` (files no packed version of the tier holds yet): `tier => [(; paths, configs, where, provenance)]`."
    const UNPACKED = unpacked_entries(projectdir("configs", "data_artifacts.toml"))
    "Whether `p` lies at or below a path of `tier`'s `[[unpacked.<tier>]]` entries."
    is_unpacked(tier, p) = DataTiers.is_unpacked(UNPACKED, tier, p)
    "A tier's packed files under data_root(): its filter's files less its `[[unpacked.<tier>]]` ones."
    packed_files(n) = filter(p -> !is_unpacked(tier_name(n), p), artifact_files(CATALOG, n).files)

    files_under(root, p) = isfile(joinpath(root, p)) ? [p] :
        [relpath(joinpath(r, f), root) for (r, _, fs) in walkdir(joinpath(root, p); follow_symlinks = true) for f in fs]
end

@testitem "published data (configs/data_artifacts.toml)" default_imports = false setup = [PublishedData] begin
    using Test
    using DrWatson
    using TOML: TOML
    using .PublishedData: CATALOG, TIERS, CATALOG_TIERS, UNPACKED, config_inputs, tier_name, catalog_violations, required_artifacts

    @testset "the catalog loads and agrees with Artifacts.toml (A1)" begin
        @test CATALOG.artifacts_toml == projectdir("Artifacts.toml")
        v = catalog_violations(CATALOG)
        foreach(x -> @info("catalog: " * x), v)
        @test isempty(v)
        @test all(a -> a["kind"] == "tree" && haskey(a, "dir") && haskey(a, "filter") && haskey(a, "files"), values(CATALOG.artifacts))
        @test sort(collect(keys(TIERS))) == sort(collect(CATALOG_TIERS))                # one version of each catalogued tier the configs name
        @test all(n -> occursin(r"^rc_[a-z0-9_]+_[0-9]{8}$", n), keys(CATALOG.artifacts))   # rc_<tier>_<yyyymmdd>; a tier may name a lens (rc_fx3_14mm)
        @test sort(required_artifacts(CATALOG)) == sort(collect(values(TIERS)))
        @test CATALOG.records["rc"]["license"] == "CC-BY-4.0"
    end

    @testset "every input of every config names a catalogued tier (or Eval's unpublished) and a path inside data_root()" begin
        ins = config_inputs()
        @test length(ins) > 50
        for (cfg, id, tier, p) in ins
            ok = (haskey(TIERS, tier) || tier == "unpublished") && !isabspath(p) && !startswith(p, "~") && !(".." in splitpath(p))   # unpublished: Eval's reserved tier (pinned by sha256)
            ok || @info "input outside the catalog or data_root()" cfg id tier p
            @test ok
        end
    end

    @testset "no driver or project source names a machine path or the retired experiments/ tree" begin
        hits = String[]
        for dir in ("scripts", "src", "configs"), (root, _, files) in walkdir(projectdir(dir)), f in files
            any(e -> endswith(f, e), (".jl", ".toml", ".py", ".sh")) || continue
            for (i, line) in enumerate(eachline(joinpath(root, f)))
                code = strip(first(split(line, '#'; limit = 2)))                   # comments and provenance prose may name history
                occursin(r"\"/(home|Users|tmp|workspaces|mnt|data)/", line) && push!(hits, "$(relpath(joinpath(root, f), projectdir())):$i")
                occursin(r"(?<!configs/)experiments/(geometric_accuracy|macro_defocus|vignetting|lib|grid_prior|focus_ab|fetch_raw_photos|raw_photos)", code) &&
                    !startswith(lstrip(line), "provenance") && !occursin(r"[Mm]oved from|[Pp]orted from", line) &&
                    push!(hits, "$(relpath(joinpath(root, f), projectdir())):$i")
            end
        end
        foreach(h -> @info("machine path / experiments/: " * h), hits)
        @test isempty(hits)
    end

    @testset "[unpacked]: each path an input of its tier in the configs it lists, or, listing none, named by no config" begin
        ins = config_inputs()
        for (tier, es) in UNPACKED, u in es
            @test haskey(TIERS, tier) && !isempty(u.paths) && !isempty(u.where) && !isempty(u.provenance)
            for q in u.paths
                if isempty(u.configs)                                   # files no config of record reads yet
                    namers = [(cfg, id) for (cfg, id, _, p) in ins if p == q || startswith(p, q * "/") || startswith(q, p * "/")]
                    isempty(namers) || @info "[unpacked] path listing no configs is named by a config" tier q namers
                    @test isempty(namers)
                else
                    # the configs' inputs at or below the path (an experiment's [inputs], a calibration case's pipeline inputs: config_inputs)
                    named = [t for c in u.configs for (cfg, _, t, p) in ins if cfg == basename(c) && (p == q || startswith(p, q * "/"))]
                    @test !isempty(named) && all(==(tier), named)
                end
            end
        end
    end

    @testset "the tiers: Eval's tree kind with a filter" begin
        @test all(a -> a["kind"] == "tree" && haskey(a, "filter") && !isempty(a["filter"]), values(CATALOG.artifacts))
        @test tier_name("rc_synthetic_20261029") == "rc_synthetic"
    end
end

# Where data_root() holds the tiers, their content: a :data item, skipped (a recorded skip) where no data root is bound.
@testitem "published data: the tiers under data_root()" default_imports = false tags = [:data] setup = [PublishedData] skip = !(
        haskey(ENV, "VGTK_DATA_ROOT") ||
        (p = joinpath(dirname(Base.active_project()), "data", "exp_raw"); islink(p) || isdir(p))) begin
    using Test
    using DrWatson
    using TOML: TOML
    using SHA: sha256
    using .PublishedData: CATALOG, TIERS, UNPACKED, config_inputs, is_unpacked, packed_files, files_under, pipeline_inputs, data_root
    const DATA_ROOT = data_root()
    # The data root holds the tiers when every catalogued tier has at least the files its entry records.
    const HAS_TIERS = isfile(joinpath(DATA_ROOT, "rc_raw_photos.sha256"))

    if !HAS_TIERS
        @info "published data: data_root() does not hold the tiers (fetch them: scripts/fetch_data.jl); the data checks are skipped"
    else
        files = Dict(n => packed_files(n) for n in keys(CATALOG.artifacts))
        @testset "make_artifacts.jl --dry-run lists the tiers with the catalogued file counts, the [unpacked] files apart" begin
            out = read(addenv(`$(Base.julia_cmd()) --project=$(projectdir()) $(projectdir("scripts", "make_artifacts.jl")) --dry-run`,
                              "VGTK_DATA_ROOT" => DATA_ROOT), String)
            for (n, a) in CATALOG.artifacts
                m = match(Regex("^$n\\s+([0-9]+) files", "m"), out)
                @test m !== nothing && parse(Int, m[1]) == a["files"] == length(files[n])
            end
            @test occursin(r"^0 files of data_root\(\) in no tier"m, out)
            for (t, es) in UNPACKED, u in es, q in u.paths                  # each [unpacked] path's files present here, counted apart
                k = ispath(joinpath(DATA_ROOT, q)) ? length(files_under(DATA_ROOT, q)) : 0
                m = match(Regex("^\\s*([0-9]+) files \\[unpacked\\.$t\\] \\Q$q\\E( |\$)", "m"), out)
                @test m !== nothing && parse(Int, m[1]) == k
            end
            @test CATALOG.artifacts[TIERS["rc_raw_photos"]]["files"] == 248 + 1                # the photographs and their manifest
        end
        @testset "the raw photographs: 248, each as its manifest states" begin
            manifest = [split(l; limit = 2) for l in eachline(joinpath(DATA_ROOT, "rc_raw_photos.sha256"))]
            @test length(manifest) == 248
            @test read(joinpath(DATA_ROOT, "rc_raw_photos.sha256")) == read(projectdir("scripts", "py", "raw_photos.sha256"))   # the tracked copy
            @test sort([String(strip(p)) for (_, p) in manifest]) == sort(filter(f -> f != "rc_raw_photos.sha256", files[TIERS["rc_raw_photos"]]))
            # the content of a sample (the whole manifest is verified by `sha256sum -c` in fetch_raw_photos.sh and on packing)
            for (h, p) in manifest[1:31:end]
                @test bytes2hex(open(sha256, joinpath(DATA_ROOT, strip(p)))) == h
            end
        end
        @testset "rc_raw_photos, rc_synthetic and rc_baselines are disjoint" begin
            a, b, c = (Set(files[TIERS[t]]) for t in ("rc_raw_photos", "rc_synthetic", "rc_baselines"))
            @test isempty(intersect(a, b)) && isempty(intersect(a, c)) && isempty(intersect(b, c))
        end
        @testset "every file of an input lies in its tier" begin
            for (cfg, id, tier, p) in config_inputs()
                id == "[experiment] dataset" && continue                                   # a directory that only has to exist
                haskey(TIERS, tier) || continue                                           # tier unpublished: no artifact packs it
                is_unpacked(tier, p) && continue                                          # the tier's next version packs it ([unpacked])
                ispath(joinpath(DATA_ROOT, p)) || continue                              # (the renders not on this machine: the report)
                stray = setdiff(files_under(DATA_ROOT, p), files[TIERS[tier]])
                isempty(stray) || @info "files of an input outside its tier" cfg id tier first(stray, 3)
                @test isempty(stray)
            end
        end
        @testset "rc_calibration holds every input of the golden cases" begin
            for f in readdir(projectdir("configs", "calibration"); join = true), (k, p) in pipeline_inputs(TOML.parsefile(f))
                @test ispath(joinpath(DATA_ROOT, p)) && !isempty(files_under(DATA_ROOT, p))
            end
        end
    end
end
