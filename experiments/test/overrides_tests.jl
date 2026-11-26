# An experiment = the config of record + keyword OVERRIDES of its hyperparameters (owner, 2026-11-15; `expand_overrides`, through every
# driver's `load_driver_config`): the base's cases expanded per combination of the values (DrWatson's dict_list); a variant whose values
# are the case's own resolved ones IS the case (its name, its table, its run key: the run of record reused); another is named by its keys.

@testitem "experiment overrides: the config of record + dict_list variants, the record's run reused" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    rpath = projectdir("configs", "experiments", "radiometric_refinement.toml")
    rc = load_radiometric_refinement_config(rpath)
    mktempdir() do d
        f = joinpath(d, "radiometric_refinement_ablation_test.toml")
        write(f, """
            [experiment]
            name = "radiometric_refinement"
            base = "configs/experiments/radiometric_refinement.toml"
            cases = ["si30_bc5_warped"]
            [overrides]
            "lm.max_iter" = [25, 3]
            """)
        ab = load_radiometric_refinement_config(f)
        @test sort(collect(keys(ab["cases"]))) == ["si30_bc5_warped", "si30_bc5_warped__lm.max_iter=3"]
        @test ab["cases"]["si30_bc5_warped"] == rc["cases"]["si30_bc5_warped"]                   # 25 is the case's own: the case itself
        @test ab["cases"]["si30_bc5_warped__lm.max_iter=3"]["pipeline"]["lm"]["max_iter"] == 3
        @test refinement_product(ab, "si30_bc5_warped").key == refinement_product(rc, "si30_bc5_warped").key   # the run of record reused
        @test refinement_product(ab, "si30_bc5_warped__lm.max_iter=3").key != refinement_product(rc, "si30_bc5_warped").key
        @test ab["experiment"] == rc["experiment"] && ab["inputs"] == rc["inputs"]
        write(f, "[experiment]\nname = \"radiometric_refinement\"\nbase = \"configs/experiments/radiometric_refinement.toml\"\ncases = [\"nope\"]\n[overrides]\n\"lm.max_iter\" = [3]\n")
        @test_throws "no case nope" load_radiometric_refinement_config(f)
    end
end

@testitem "experiment overrides: the lens_coverage_scale ablation of record; a top-level key (fixed) overridden" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    rc = load_radiometric_refinement_config(projectdir("configs", "experiments", "radiometric_refinement.toml"))
    ab = load_radiometric_refinement_config(projectdir("configs", "studies", "radiometric_refinement_ablation_lens_coverage_scale.toml"))
    cases = TOML.parsefile(projectdir("configs", "studies", "radiometric_refinement_ablation_lens_coverage_scale.toml"))["experiment"]["cases"]
    @test sort(collect(keys(ab["cases"]))) == sort([cases; [c * "__masks.lens_coverage_scale=0.9" for c in cases]])   # 1.0: default.toml's, the case itself
    for c in cases
        @test ab["cases"][c] == rc["cases"][c]
        @test ab["cases"][c * "__masks.lens_coverage_scale=0.9"]["pipeline"]["masks"]["lens_coverage_scale"] == 0.9
    end
    mktempdir() do d
        f = joinpath(d, "radiometric_refinement_fixed_test.toml")
        write(f, "[experiment]\nname = \"radiometric_refinement\"\nbase = \"configs/experiments/radiometric_refinement.toml\"\ncases = [\"si30_bc5_warped\"]\n[overrides]\nfixed = [[\"poses\", \"aperture\"]]\n")
        fx = load_radiometric_refinement_config(f)
        v = only(k for k in keys(fx["cases"]) if startswith(k, "si30_bc5_warped__fixed="))
        @test fx["cases"][v]["pipeline"]["fixed"] == ["poses", "aperture"]
    end
end

# One place per setting (owner 2026-11-16 23:30): a setting of every case of a model kind is stated once, in [kinds.<kind>], and
# inherited (`inherit_kinds!`); a case that states it again is refused; the vignetting refinements read the refinements' [kinds].
@testitem "kinds: one place per model kind, inherited by every case, a restatement refused" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    for f in ("radiometric_refinement.toml", "radiometric_refinement_vignetting.toml")
        rc = load_radiometric_refinement_config(projectdir("configs", "experiments", f))
        @test !haskey(rc, "kinds")
        for (c, t) in rc["cases"]                                  # owner 2026-11-16 23:30: 8,192 every global case, 16,384 every spline
            @test t["pipeline"]["masks"]["samples_per_view"] == (t["kind"] == "grid" ? 16384 : 8192)
        end
    end
    raw() = Dict{String, Any}("kinds" => Dict{String, Any}("grid" => Dict{String, Any}("pipeline" => Dict{String, Any}("masks" => Dict{String, Any}("samples_per_view" => 16384)))),
                              "cases" => Dict{String, Any}("a" => Dict{String, Any}("kind" => "grid", "pipeline" => Dict{String, Any}("masks" => Dict{String, Any}("seed" => 1))),
                                                           "b" => Dict{String, Any}("kind" => "lens", "pipeline" => Dict{String, Any}())))
    c = inherit_kinds!(raw(), "t.toml")
    @test c["cases"]["a"]["pipeline"]["masks"] == Dict("seed" => 1, "samples_per_view" => 16384) && !haskey(c["cases"]["b"]["pipeline"], "masks")
    r = raw(); r["cases"]["a"]["pipeline"]["masks"]["samples_per_view"] = 8192
    @test_throws "one place per setting" inherit_kinds!(r, "t.toml")
    r = raw(); r["kinds"]["spline"] = r["kinds"]["grid"]
    @test_throws "a kind is one of" inherit_kinds!(r, "t.toml")
end
