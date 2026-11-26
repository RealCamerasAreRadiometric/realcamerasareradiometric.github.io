# The config lock (RC 0.11.4, owner 2026-11-15): every config of record resolves through RadiometricCamera's `pipeline_options` to the
# 0.11.5 hyperparameters of record (test/fixtures/config_lock/hyperparameters_0115.toml), any deviation an explicit, commented override
# on the case's own line; every Schöps-started grid case takes his own lattice (grid.lattice = "derived": m = 1, never subdivided), every
# other grid start the image region's lattice at its stated node count (grid.lattice = "node_count", owner 2026-11-15 23:18).

@testsetup module ConfigLock
    using TOML
    # the leaves of a TOML table, `a.b.c => value`
    leaves(d, pre = "") = reduce(vcat, [v isa AbstractDict ? leaves(v, pre * k * ".") : [(pre * k) => v] for (k, v) in d]; init = Pair{String, Any}[])
    # Whether the config text `lines` states `key` of the table `header` on a commented line: a `#` outside a string on its line, or a
    # comment line directly above it. An inline table's key (`smoothness = { … }`) is found by its parent's line. `nothing`: not stated.
    function commented(lines, header::AbstractString, key::AbstractString)
        cur = ""
        parts = split(key, '.')
        for (i, l) in enumerate(lines)
            m = match(r"^\s*\[([^\]]+)\]", l)
            m === nothing || (cur = strip(m.captures[1]); continue)
            for j in length(parts):-1:1                        # the key on its table's line, or an inline table's parent on its own
                h = j == 1 ? header : header * "." * join(parts[1:(j - 1)], ".")
                cur == h && occursin(Regex("^\\s*" * parts[j] * "\\s*="), l) || continue
                return occursin('#', replace(l, r"\"[^\"]*\"" => "\"\"")) || (i > 1 && startswith(strip(lines[i - 1]), "#"))
            end
        end
        return nothing
    end
    export leaves, commented
end

@testitem "config lock: every config of record resolves to the 0.11.4 hyperparameters; a deviation is a commented override" default_imports = false setup = [ConfigLock] begin
    using Test, TOML, RadiometricCamera
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))                                                   # inherit_kinds!: the settings a config states once per model kind
    const RC = RadiometricCamera
    root = normpath(pkgdir(RadiometricCamera), "..")   # the repository: the package and the experiments side by side
    lock = Dict(leaves(TOML.parsefile(joinpath(root, "experiments", "test", "fixtures", "config_lock", "hyperparameters_0115.toml"))))
    files = [joinpath(r, f) for (r, _, fs) in walkdir(joinpath(root, "experiments", "configs")) for f in fs if endswith(f, ".toml")]
    ncases = 0; nschoeps = 0; uncommented = String[]
    for f in sort(files)
        raw = TOML.parsefile(f); lines = readlines(f)
        # a key set by [kinds.<kind>] is stated (and commented) there, in this file or in its [kinds] source
        kinds = get(raw, "kinds", Dict{String, Any}())
        klines = haskey(kinds, "source") ? readlines(joinpath(root, "experiments", kinds["source"])) : lines
        kdoc = haskey(kinds, "source") ? get(TOML.parsefile(joinpath(root, "experiments", kinds["source"])), "kinds", Dict{String, Any}()) : kinds
        inherited(kind, k) = (p = split(k, '.'; limit = 2); length(p) == 2 && haskey(get(get(get(kdoc, kind, Dict()), "pipeline", Dict()), p[1], Dict()), p[2]))
        doc = inherit_kinds!(raw, f)
        runs = [("$tb.$c.pipeline", t["pipeline"], get(t, "kind", "")) for tb in ("cases", "init") for (c, t) in get(doc, tb, Dict{String, Any}()) if t isa AbstractDict && haskey(t, "pipeline")]
        haskey(doc, "pipeline") && push!(runs, ("pipeline", doc["pipeline"], ""))
        for (header, p, kind) in runs
            ncases += 1
            alg, stops = RC.read_calibration(p); o = alg.options                    # resolves (a refusal is a failure of the lock)
            resolved = Dict(RC.config_key(p, k) => v for (k, v) in leaves(RC.config_dict(alg, stops)))   # keyed as the file's version names them
            for (k, v) in lock
                haskey(resolved, k) || continue                                     # a key the run cannot use (INAPPLICABLE_KEYS): not written
                # the in-plane start acts only with inplane = "aspect_skew": a heights-only case (inplane = "none", its commented override;
                # owner decision 2026-11-16) states none
                k == "warping.inplane_start" && get(resolved, "warping.inplane", "") == "none" && continue
                resolved[k] == v && continue
                inherited(kind, k) && commented(klines, "kinds.$kind.pipeline", k) === true && continue
                commented(lines, header, k) === true || push!(uncommented, "$(relpath(f, root)) [$header] $k = $(repr(resolved[k])) (lock $(repr(v)))")
            end
            # a Schöps-started grid case: his own lattice, m = 1 (the exact start never subdivides: test/unit/exact_start_tests.jl)
            if get(get(p, "frontend", Dict()), "type", "") == "schoeps" && RC.start_is_grid(o)
                nschoeps += 1
                @test o.grid.lattice === RC.DerivedLattice()
                @test commented(lines, header, "grid.lattice") === true             # stated, with its derivation
            end
            @test !haskey(get(p, "grid", Dict()), "lattice_spacing") && !haskey(get(p, "grid", Dict()), "size")
            # the configuration converter (RadiometricCamera's RC_CONFIG_VERSIONS, 0.4.0): a config of record stays version 1 on disk, its
            # [lm] read whole as the current [optimizer]; its start and refinement settings split as written, the keys of version 1 —
            # the split of record (before 0.4.0), so its products keep their keys
            @test !haskey(p, RC.VgtkCore.CONFIG_VERSION_KEY)
            let m = RC.VgtkCore.migrate_config(p, RC.RC_CONFIG_VERSIONS), old = "sensor_characteris" * "ation"   # (split: the banned spelling)
                @test get(m, "optimizer", nothing) == get(p, "lm", nothing)
                @test get(get(m, "camera", Dict()), "sensor_characterization", nothing) ==
                      get(get(p, "camera", Dict()), "sensor_characterization", get(get(p, "camera", Dict()), old, nothing))
            end
            let refine_of_record = Dict{String, Any}()
                for k in ("lm.max_iter", "lm.max_time", "lm.ftol", "lm.xtol_rel", "io.checkpoint_each", "sweep.iterations", "sweep.lm_chunk",
                          "resume.iteration_offset", "resume.sweep")
                    tb, key = split(k, '.')
                    haskey(get(p, tb, Dict()), key) && (get!(refine_of_record, tb, Dict{String, Any}())[key] = p[tb][key])
                end
                start, refine = RC.split_refine_only(p)
                @test refine == refine_of_record
                @test start == Dict{String, Any}(k => (v isa AbstractDict && haskey(refine, k) ? Dict{String, Any}(kk => vv for (kk, vv) in v if !haskey(refine[k], kk)) : v)
                                                 for (k, v) in p if !(haskey(refine, k) && length(refine[k]) == length(v)))
            end
            # every other grid start (a parametric or any non-Schöps source) lays its lattice over the image region at a stated,
            # commented node count — the case's bound on its present nodes (owner 2026-11-15, 23:18)
            if get(get(p, "frontend", Dict()), "type", "") != "schoeps" && RC.start_is_grid(o)
                @test o.grid.lattice === RC.NodeCountLattice() && o.grid.node_count !== nothing
                @test commented(lines, header, "grid.node_count") === true
            end
        end
    end
    @test ncases >= 37 && nschoeps >= 14                                          # the survey of 2026-11-15: 36 cases + the golden; 14 Schöps grid cases
    @test isempty(uncommented)
    isempty(uncommented) || foreach(println, uncommented)
end

@testitem "config lock: its reader — a commented override is accepted, a bare one refused" default_imports = false setup = [ConfigLock] begin
    using Test
    lines = ["[cases.a.pipeline.lm]", "max_iter = 50   # the fisheye: 50 iterations", "max_time = 1.0",
             "[cases.a.pipeline.grid]", "# the derivation above the line", "lattice = \"derived\"", "smoothness = { type = \"none\" }   # none"]
    @test commented(lines, "cases.a.pipeline", "lm.max_iter") === true
    @test commented(lines, "cases.a.pipeline", "lm.max_time") === false
    @test commented(lines, "cases.a.pipeline", "grid.lattice") === true
    @test commented(lines, "cases.a.pipeline", "grid.smoothness.type") === true
    @test commented(lines, "cases.a.pipeline", "masks.seed") === nothing
end
