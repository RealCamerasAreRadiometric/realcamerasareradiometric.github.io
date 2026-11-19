# The API surface (owner D13, 2026-11-15): RadiometricCamera exports its headline API only — the list docs/src/api/surface.md states,
# bullet for bullet — and every other user-facing name is `public`; the paper's symbols are public, never exported (D14).

@testitem "API surface: the exports are the documented headline; the rest public" begin
    using Test, RadiometricCamera
    const RC = RadiometricCamera
    doc = read(joinpath(pkgdir(RadiometricCamera), "docs", "src", "api", "surface.md"), String)
    sec = doc[findfirst("## Exported", doc).stop:findfirst("## Public", doc).start]
    documented = Set(Symbol(m[1]) for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", sec))
    exported = Set(n for n in names(RC) if Base.isexported(RC, n) && n !== :RadiometricCamera)   # (names() lists the public names too)
    @test exported == documented
    @test length(exported) <= 20                                                   # a headline, not the package
    for n in (:𝔣, :𝔭, :ρ_c, :Φ_pixel, :View, :build_problem, :run_lm!, :save_state, :load_distance_map, :selfcheck)
        @test Base.ispublic(RC, n) && !Base.isexported(RC, n)
    end
    # the retired names (RC 0.11.0) are refused by name with Core's replacement, never public
    @test_throws "use VisualGeometryToolkitCore's `backproject(camera, 𝐩)`" RC.π⁻¹(nothing, nothing)
    @test_throws "use VisualGeometryToolkitCore's `project(camera, X)`" RC.project_pixel(nothing, nothing, nothing)
    @test !Base.ispublic(RC, :π⁻¹) && !Base.ispublic(RC, :project_pixel)
end
