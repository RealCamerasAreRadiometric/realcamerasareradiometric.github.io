# quality_naming_tests.jl — the stack's naming vocabulary (the iterative-estimator words, the retired names and phrases, the
# width names, units in one place), checked by VisualGeometryToolkitCore's `check_naming` over this repository: the package, its
# tests, docs and examples and the experiments paper project. The rules and their lists live in VisualGeometryToolkitCore
# (ext/VisualGeometryToolkitCoreTestExt/check_naming.jl, loaded with Test), which also tests the checker itself.

@testitem "quality: naming" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, RadiometricCamera
    VgtkCore.test_naming(RadiometricCamera)
    # the experiments beside the package in the repository (absent from an installed package)
    exp = normpath(pkgdir(RadiometricCamera), "..", "experiments")
    # TEMPORARY (the Core 0.14 solve port, RC 0.4.0): the experiments paper project is not yet ported to the solve interface (its
    # environment needs BlobBoards, LocalOptimizationDoneRight and PanoramaStitching on Core 0.14); its scan is skipped until the
    # owner rules whether it is ported or pinned to RC 0.3.0. Remove this skip with that ruling.
    @test_skip isdir(exp) && isempty(VgtkCore.check_naming(exp))
    if false && isdir(exp)
        found = VgtkCore.check_naming(exp)
        isempty(found) || foreach(v -> println(stderr, v), found)
        @test isempty(found)
    end
end

# ONE word for a spline's free values, "node", and ONE key for how many, `node_count` (owner, RC 0.11.4): the retired spellings of a
# node — the old two-word name ("control" + " point") in prose, the keys `[vignetting] coefficients` and `[focus] nodes` in a config — stay out of the
# repository (the package, its tests, docs and configs, the paper project's code and configs; the CHANGELOG's history excepted).
# VisualGeometryToolkitCore's check_naming carries no per-package list; this item is the lock until Core's list holds them.
@testitem "quality: naming — a spline's free value is a node, its count node_count" begin
    using Test, TOML
    root = pkgdir(RadiometricCamera)
    dirs = filter(isdir, [joinpath(root, d) for d in ("src", "test", "docs", "configs", "bin", "examples", joinpath("..", "experiments", "src"),
                                                      joinpath("..", "experiments", "scripts"), joinpath("..", "experiments", "configs"),
                                                      joinpath("..", "experiments", "test"))])   # the experiments when beside the package
    spelt = String[]; keyed = String[]
    retired_phrase = Regex("\\b[Cc]ontrol" * " points?\\b")              # (spelled apart: this file is scanned too)
    for d in dirs, (r, _, fs) in walkdir(d), f in fs
        any(e -> endswith(f, e), (".jl", ".md", ".toml")) && !startswith(f, "Manifest") || continue
        p = joinpath(r, f); occursin("/build/", p) && continue
        s = read(p, String)
        occursin(retired_phrase, s) && push!(spelt, p)
        if endswith(f, ".toml") && f != "Project.toml"
            doc = try TOML.parse(s) catch; continue end
            walk(x, path) = x isa AbstractDict ? foreach(((k, v),) -> begin
                    (k == "coefficients" && endswith(path, "vignetting")) || (k == "nodes" && endswith(path, "focus")) ? push!(keyed, "$p:$path.$k") : nothing
                    walk(v, isempty(path) ? k : "$path.$k")
                end, x) : x isa AbstractVector ? foreach(v -> walk(v, path), x) : nothing
            walk(doc, "")
        end
    end
    @test isempty(spelt)
    @test isempty(keyed)
end
