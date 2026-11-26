# The experiment groups beside configs/experiments/ (owner 2026-11-18): configs/supplement/ (experiments of record whose assets appear
# in the supplement alone) and configs/studies/ (ideas tried that never came to fruition). Each study states why it exists and the
# decision of record it backed (its RATIONALE block with a `decided` line); every group's products lie under results_root()/<group>/…,
# a location that keys nothing.

@testitem "groups: every study states its rationale and the decision it backed; every supplement experiment its rationale" default_imports = false begin
    using Test
    using DrWatson
    for (group, needs_decided) in (("studies", true), ("supplement", false)), f in readdir(projectdir("configs", group); join = true)
        endswith(f, ".toml") || continue
        head = String[]
        for l in eachline(f)
            startswith(l, "#") || break
            push!(head, l)
        end
        @test any(l -> occursin("RATIONALE", l), head)
        needs_decided && @test any(l -> occursin(r"^#\s+decided:?\s+\S", l), head)
    end
end

@testitem "groups: a group's products lie under results_root()/<group>/…; their keys are the configs' alone" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    @test product_experiment("sampling_strategy") == "studies/sampling_strategy"
    @test product_experiment("warping_ablation") == "studies/warping_ablation"
    @test product_experiment("timing") == "supplement/timing"
    @test product_experiment("radiometric_refinement") == "radiometric_refinement"     # the experiments of record: where they were
    @test config_group(projectdir("configs", "studies", "warping_ablation.toml")) == "studies"
    @test config_group(projectdir("configs", "experiments", "radiometric_refinement.toml")) === nothing
    rc = load_radiometric_refinement_config(projectdir("configs", "experiments", "radiometric_refinement.toml"))
    ab = load_radiometric_refinement_config(projectdir("configs", "studies", "radiometric_refinement_ablation_lens_coverage_scale.toml"))
    v = "si30_kb_warped__masks.lens_coverage_scale=0.9"
    @test haskey(ab["cases"], v)
    @test variant_location(ab, v) == "studies/radiometric_refinement_ablation_lens_coverage_scale/radiometric_refinement"
    @test variant_location(ab, "si30_kb_warped") == "radiometric_refinement"            # the 1.0 variant IS the case: its run of record
    @test refinement_product(ab, "si30_kb_warped").key == refinement_product(rc, "si30_kb_warped").key
    @test startswith(product_dir(refinement_product(ab, v)), joinpath(results_root(), "studies", "radiometric_refinement_ablation_lens_coverage_scale"))
    sm = load_radiometric_refinement_config(projectdir("configs", "studies", "radiometric_refinement_smoothness_si30fe.toml"))
    @test length(sm["cases"]) == 3 && all(t -> t["pipeline"]["lm"]["max_iter"] == 25, values(sm["cases"]))
    @test sort([t["pipeline"]["grid"]["smoothness"]["weight"] for t in values(sm["cases"])]) == [0.001, 0.01, 0.1]
end
