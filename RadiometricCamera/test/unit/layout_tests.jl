# The source layout (RC 0.11.0 reorganization): the include list of src/RadiometricCamera.jl reaches every file under src/, once, and
# names only files that exist; the retired directories (calibration_pipeline/, pipeline/, board/) hold nothing.

@testitem "layout: the include list reaches every src file once; the retired directories are gone" begin
    using Test, RadiometricCamera
    src = joinpath(pkgdir(RadiometricCamera), "src")
    included = [m[1] for m in eachmatch(r"^\s*include\(\"([^\"]+)\"\)"m, read(joinpath(src, "RadiometricCamera.jl"), String))]
    files = sort([relpath(joinpath(d, f), src) for (d, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl") && f != "RadiometricCamera.jl"])
    @test allunique(included)
    @test sort(included) == files
    @test all(d -> !isdir(joinpath(src, d)), ("calibration_pipeline", "pipeline", "board"))
end
