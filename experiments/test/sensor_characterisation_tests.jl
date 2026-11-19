# The sensor characterisation (scripts/sensor_characterisation.jl, configs/experiments/sensor_characterisation.toml,
# src/sensor_characterisation.jl): the config of record loads and a wrong one is refused, and it names the refinements' record of record.
# The computation is Core's estimators alone (tested in Core); the real run needs the campaign's frames (REPRODUCE.md) and is not this
# suite: its summary.toml states whether it reproduces the record of record.

@testitem "sensor_characterisation: the config of record loads, a wrong one is refused" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("sensor_characterisation.jl"))
    path = projectdir("configs", "experiments", "sensor_characterisation.toml")
    c = load_sensor_characterisation_config(path)
    @test c["characterise"]["dark"] == "1_8000" && c["characterise"]["false_defects"] == 0.01
    @test c["inputs"]["record_of_record"]["sha256"] == "add8fc120b83806d1e70ecf4252a3b0bcf64d9fe007f42886789212f8c155737"
    # the record of record is the refinements' sensor model input, the same file
    rr = TOML.parsefile(projectdir("configs", "experiments", "radiometric_refinement.toml"))["inputs"]["fx3_sensor_characterisation"]
    @test rr["path"] == c["inputs"]["record_of_record"]["path"] && rr["sha256"] == c["inputs"]["record_of_record"]["sha256"]
    good = TOML.parsefile(path); dir = mktempdir()
    function refused(edit!)
        g = deepcopy(good); edit!(g)
        p = joinpath(dir, "sensor_characterisation.toml")
        open(io -> TOML.print(io, g), p, "w")
        return try load_sensor_characterisation_config(p); nothing catch err; err end
    end
    @test refused(identity) === nothing
    for k in CHARACTERISE_KEYS
        e = refused(g -> delete!(g["characterise"], k)); @test e isa ArgumentError && occursin(k, e.msg)
    end
    @test refused(g -> g["characterise"]["false_defects"] = 1.5) isa ArgumentError
    @test refused(g -> g["characterise"]["hot_sigma"] = 6.0) isa ArgumentError                  # the retired dark statistics' keys
    @test refused(g -> delete!(g["inputs"], "manifest")) isa ArgumentError
    @test !isdefined(@__MODULE__, :dark_statistics) && !isdefined(@__MODULE__, :defect_masks)   # D12: deleted, Core's estimators instead
    @test c["characterise"]["tier"] == "rc_sensor_characterisation_20261113"                    # the manifest rows' tier: a config literal
    @test refused(g -> delete!(g["characterise"], "tier")) isa ArgumentError
    # a burst per exposure: two bursts at one EXIF exposure are refused by name, never one silently replacing the other
    s = VgtkCore.parse_unit("s")
    idx = exposure_index(["1_60" => (1 / 60) * s, "1_8000" => (1 / 8000) * s], "E1_E2")
    @test idx[(1 / 60) * s] == "1_60" && length(idx) == 2
    e = try exposure_index(["1_60" => (1 / 60) * s, "1_60_again" => (1 / 60) * s], "E1_E2"); nothing catch err; err end
    @test e isa ArgumentError && occursin("1_60", e.msg) && occursin("1_60_again", e.msg)
end
