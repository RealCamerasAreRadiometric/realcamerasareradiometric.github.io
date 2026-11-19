# The blob front end reads a calibration session's detections only through VisualGeometryToolkitIO (owner D4, 2026-11-15: no hand
# parser): the session's observations record `observations.json`; a session without one is refused by name, with the record it needs,
# before any record is read — a BlobBoards `sightings.json` beside the records is never read.

@testitem "blob front end: observations only from the session's IO record; sightings.json never read" begin
    using Test, RadiometricCamera
    const RC = RadiometricCamera
    d = mktempdir(); write(joinpath(d, "sightings.json"), "{}")
    o = RC.pipeline_options(Dict("frontend" => Dict("type" => "blob", "session" => d), "io" => Dict("images" => d, "out" => mktempdir())))
    e = try RC._frontend(RC.BlobFrontend, o, (; R0 = 1e-3, height0 = Float64[], page_flip = 1.0, inplane0 = nothing, stored = nothing)); nothing catch err; err end
    @test e isa ArgumentError && occursin("observations.json", e.msg) && occursin("vgio/observations", e.msg) && occursin("sightings.json is not read", e.msg)
    src = join((read(f, String) for (r, _, fs) in walkdir(joinpath(pkgdir(RadiometricCamera), "src")) for n in fs if endswith(n, ".jl") for f in (joinpath(r, n),)), "\n")
    @test !occursin("sightings", replace(src, "a BlobBoards `sightings.json` is not read" => "", "a BlobBoards sightings.json is not read" => ""))
end

@testitem "blob front end: the session's observations record read back per frame, with the target's points (planar z = 0)" setup = [RCFixtures] begin
    using StaticArrays
    struct StubTarget; P::Vector{SVector{2, Float64}}; end              # a planar target: its points by index (the target's frame, metres)
    VgtkCore.points(t::StubTarget) = t.P
    target = StubTarget([SVector(0.0, 0.0), SVector(0.01, 0.0), SVector(0.0, 0.02), SVector(0.03, 0.04)])
    F = VgtkCore.PixelFrame{:colmap}
    frames = Dict(3 => Dict(:board => VgtkCore.correspondences([1, 4], [SVector(10.5, 20.25), SVector(30.0, 40.75)])),
                  7 => Dict(:board => VgtkCore.correspondences([2], [SVector(1.0, 2.0)])))
    obs = VgtkCore.TargetObservations{F}(frames; images = Dict(3 => "V03.png", 7 => "V07.png"), sensors = "cam", targets = [repeat("ab", 32)])
    mktempdir() do session
        VgtkIO.save(joinpath(session, "observations.json"), obs)
        f = RC._session_observations_file(session)
        @test f == joinpath(session, "observations.json")
        out = RC._session_observations(f, target)
        @test sort(collect(keys(out))) == [3, 7]                              # by frame index (the session's frames)
        @test sort(out[3]) == [(10.5, 20.25, 0.0, 0.0, 0.0), (30.0, 40.75, 0.03, 0.04, 0.0)]   # (x, y, X, Y, Z): point i of the target
        @test out[7] == [(1.0, 2.0, 0.01, 0.0, 0.0)]
        two = Dict(1 => Dict(:board => VgtkCore.correspondences([1], [SVector(1.0, 1.0)]), :other => VgtkCore.correspondences([2], [SVector(2.0, 2.0)])))
        p2 = joinpath(session, "two", "observations.json"); mkpath(dirname(p2))
        VgtkIO.save(p2, VgtkCore.TargetObservations{F}(two; images = Dict(1 => "V01.png"), sensors = "cam", targets = [repeat("ab", 32), repeat("cd", 32)]))
        @test_throws "the observations of 2 targets (board, other)" RC._session_observations(p2, target)
    end
end
