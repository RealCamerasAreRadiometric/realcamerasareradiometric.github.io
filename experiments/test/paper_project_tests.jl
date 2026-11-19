# =============================================================================
# The paper-project conventions (VisualGeometryToolkitEval PaperProjects): layout, data policy, the REPRODUCE.md index, configs of
# record, provenance and the data catalog; the data policy (D1 to D4); and no environment knob in a driver or the prelude.
# =============================================================================

@testitem "paper-project conventions" default_imports = false begin
    using Test
    using DrWatson
    using VisualGeometryToolkitEval.PaperProjects: paper_project_violations
    using VisualGeometryToolkitEval.Experiments: DataPolicyAllow
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("assets.jl"))   # PAPER_SEARCH_DIRS: where the manuscript's paths resolve

    # The deferred exceptions, each with its reason.
    const PAPER_PROJECT_ALLOW = [
        # L4 — the suite runs on ReTestItems: runtests.jl includes nothing, and the call is THIS item, which ReTestItems discovers
        # by its file's name (paper_project_tests.jl). L4's static walk follows `include` only. Fix (upstream, VisualGeometryToolkitEval
        # PaperProjects): let L4 also follow the `*_tests.jl` files ReTestItems discovers under test/.
        DataPolicyAllow("L4", "test/runtests.jl", "the suite does not call paper_project_violations"),
        # the manuscript's own files (Blens): test/fixtures/paper_allow.toml, each with its reason
        (DataPolicyAllow(a["rule"], a["file"], a["match"]) for a in TOML.parsefile(projectdir("test", "fixtures", "paper_allow.toml"))["allow"])...,
    ]
    # The manuscript is built from paper/ (src/ on TEXINPUTS): its paths resolve there too (PAPER_SEARCH_DIRS).
    v = paper_project_violations(projectdir(); papers = ["../paper/main.tex"], reproduce = (; search_dirs = PAPER_SEARCH_DIRS),
                                 allow = PAPER_PROJECT_ALLOW)
    foreach(x -> @info(sprint(show, x)), v)
    @test isempty(v)
end

@testitem "data policy (D1 to D4)" default_imports = false begin
    using Test
    using DrWatson
    using VisualGeometryToolkitEval.Experiments: data_policy_violations
    v = data_policy_violations(projectdir())
    foreach(x -> @info(sprint(show, x)), v)
    @test isempty(v)
end

# No environment knob: a driver or the prelude reads its configuration from TOML only.
@testitem "scripts and src read no environment variable" default_imports = false begin
    using Test
    using DrWatson
    hits = String[]
    for dir in ("scripts", "src"), (root, _, files) in walkdir(projectdir(dir)), f in files
        endswith(f, ".jl") || continue
        for (i, line) in enumerate(eachline(joinpath(root, f)))
            startswith(lstrip(line), "#") && continue
            occursin(r"\bENV\b", line) && push!(hits, "$(relpath(joinpath(root, f), projectdir())):$i: $(strip(line))")
        end
    end
    foreach(h -> @info("ENV read: " * h), hits)
    @test isempty(hits)
end
