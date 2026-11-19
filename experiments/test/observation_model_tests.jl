# The paper's consumers of a calibrated camera read their photographs through RadiometricCamera's ONE entry point under the camera's own
# observation model (RC 0.11.1; owner 2026-11-15: "if a sensor was estimated with a model, then it must be used to predict the model"):
# the calibration (RC's start), chrome detection and texture recovery. No other decode path of a
# predicted photograph is left in the project's sources (the one-RAW-path lint).

@testitem "observation model: one RAW path — no consumer decodes a photograph another way" default_imports = false begin
    using Test
    using DrWatson
    # every call of the decode primitives in the project's sources and scripts, outside the allow-list: RC's entry point is RC's own
    # (src/io/photographs.jl, not here); the front ends' geometric decode (decode_raw, green_quantile), the sensor characterisation's
    # producer (it reads the dark frames), and the chrome detector (its frames' models from photo_set) are the exemptions
    allowed = Dict("src/drivers.jl" => ("load_raw_mosaic(", "LinearRaw(", "linearview("),
                   "src/sensor_characterisation.jl" => ("load_raw_mosaic(",),
                   "scripts/chrome_evaluation.jl" => ("load_raw_mosaic(",))
    primitives = ("photo_observations(", "photo_sensor_models(", "load_photo(", "load_raw(", "load_raw_mosaic(", "LinearRaw(", "linearview(")
    bad = String[]
    for dir in ("src", "scripts"), (root, _, files) in walkdir(projectdir(dir)), f in files
        endswith(f, ".jl") || continue
        rel = relpath(joinpath(root, f), projectdir())
        for (i, line) in enumerate(eachline(joinpath(root, f)))
            code = first(split(line, '#'; limit = 2))
            for p in primitives
                occursin(p, code) && !(p in get(allowed, rel, ())) && push!(bad, "$rel:$i: $p")
            end
        end
    end
    @test isempty(bad)
    isempty(bad) || foreach(println, bad)
    # the consumers go through the entry point
    src(f) = read(projectdir(f), String)
    @test occursin("RC.photo_set(cc,", src("src/texture_recovery.jl")) && occursin("RC.observe(ctx.photos", src("src/texture_recovery.jl"))
    @test occursin("RadiometricCamera.photo_set(om, files", src("src/drivers.jl"))                  # raw_sensor_models: the chrome detect's and the front ends' decode
    @test occursin("raw_sensor_models(files, spec;", src("src/chrome_evaluation.jl"))
    @test occursin("characterisation = case_characterisation(REFINEMENT, CASE)", src("scripts/texture_recovery.jl"))
end

@testitem "observation model: a case's characterisation is located by its input, sha256 checked" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    cfg = Dict{String, Any}("cases" => Dict{String, Any}("a" => Dict{String, Any}("pipeline" => Dict{String, Any}("camera" => Dict{String, Any}("sensor_characterisation" => ""))),
                                                         "b" => Dict{String, Any}("pipeline" => Dict{String, Any}()),
                                                         "c" => Dict{String, Any}("pipeline" => Dict{String, Any}("camera" => Dict{String, Any}("sensor_characterisation" => "x/y.json")))),
                            "inputs" => Dict{String, Any}())
    @test case_characterisation(cfg, "a") === nothing && case_characterisation(cfg, "b") === nothing
    @test_throws "is no input of the config" case_characterisation(cfg, "c")
end
