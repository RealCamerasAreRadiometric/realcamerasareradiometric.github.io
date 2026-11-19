# The excluded capture sets (configs/excluded.toml, owner 2026-11-15): kept as data, run by no driver — a case of an excluded set is
# skipped and said so, a config of one refused by name; no config of record keeps an active table of one.

@testitem "excluded sets: skipped by every driver, refused whole, commented out of the configs of record" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    ex = excluded_sets()
    @test haskey(ex, "fisheye2710_siemens") && occursin("superseded", ex["fisheye2710_siemens"]["reason"])
    e = ex["fisheye2710_siemens"]
    # a config of the excluded set is refused by name, with the reason
    for c in e["configs"]
        @test_throws "excluded set fisheye2710_siemens" load_driver_config(projectdir(c), "chrome_evaluation"; tables = ("experiment",))
    end
    # a case of it in a shared config is skipped, the others kept
    cfg = Dict{String, Any}("cases" => Dict{String, Any}("si27fe_kb_warped" => Dict(), "si30_kb_warped" => Dict()))
    @test collect(keys(skip_excluded(cfg)["cases"])) == ["si30_kb_warped"]
    # no config of record keeps an active table of its cases or runs
    for f in readdir(projectdir("configs", "experiments"); join = true)
        endswith(f, ".toml") && !(relpath(f, projectdir()) in e["configs"]) || continue
        t = TOML.parsefile(f)
        for tb in ("cases", "schoeps", "fits", "compare"), c in e["cases"]
            @test !haskey(get(t, tb, Dict()), c)
        end
    end
end
