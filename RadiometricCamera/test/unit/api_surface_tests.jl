# The API surface (owner D13, 2026-11-15): RadiometricCamera exports its headline API only — the list docs/src/api/surface.md states,
# bullet for bullet — and every other user-facing name is `public`; the paper's symbols are public, never exported (D14).

@testitem "API surface: the exports are the documented headline; the rest public" begin
    using Test, RadiometricCamera
    const RC = RadiometricCamera
    doc = read(joinpath(pkgdir(RadiometricCamera), "docs", "src", "api", "surface.md"), String)
    sec = doc[findfirst("## Exported", doc).stop:findfirst("## Public", doc).start]
    documented = Set(Symbol(m[1]) for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", sec))
    reexported = Set(Symbol(m[1]) for m in eachmatch(r"`([^`]+)` \(re-exported", sec))   # an upstream name in the headline: no docstring of this package
    @test reexported == Set([:RadiometricCameraModel]) && RC.RadiometricCameraModel === RC.VrtkCore.RadiometricCameraModel
    documented = union(documented, reexported)
    exported = Set(n for n in names(RC) if Base.isexported(RC, n) && n !== :RadiometricCamera)   # (names() lists the public names too)
    @test exported == documented
    @test length(exported) <= 20                                                   # a headline, not the package
    @test !Base.ispublic(RC, :𝔣) && !Base.ispublic(RC, :𝔭) && RC.𝔣 === RC.VrtkCore.𝔣     # the paper's camera symbols are VisualRadiometryToolkitCore's
    for n in (:Φ_pixel, :render_view, :build_problem, :run_lm!, :save_state, :load_distance_map, :selfcheck)
        @test Base.ispublic(RC, n) && !Base.isexported(RC, n)
    end
    # the board is VisualRadiometryToolkitCore's textured target with its own state, and there is no scene (owner 2026-11-20): its names
    # are that package's (imported here where used, never public here), and the retired ones are refused by name with the replacement
    @test RC.TexturedTarget === RC.VrtkCore.TexturedTarget && RC.TargetState === RC.VrtkCore.TargetState && RC.𝔟 === RC.VrtkCore.target_center
    @test !any(n -> Base.ispublic(RC, n), (:TexturedTarget, :TargetState, :PrintedTarget, :BoardSurface, :Scene, :with_surface, :with_state, :ρ_c, :𝔟,
                                           :intersect_board, :HEIGHT_NC, :edge_albedo))
    @test_throws "the board is VisualRadiometryToolkitCore's `TexturedTarget`" RC.PrintedTarget(nothing, nothing)
    @test_throws "VisualRadiometryToolkitCore's `TargetState(; h, edge, inplane)`" RC.BoardSurface()
    @test_throws "there is no scene type. The image size is the camera's" RC.Scene(nothing, 4, 4)
    @test_throws "VisualRadiometryToolkitCore's `with_state(board, state)`" RC.with_surface(nothing, nothing)
    @test_throws "`albedo` (the paper's symbol stays its alias, `ρ_c`)" RC.ρ_c(nothing, nothing)
    @test_throws "`edge_albedo(target, 𝐛)`" RC.edge_albedo(nothing, nothing, nothing)
    @test_throws "`posed_target(target)`" RC.posed_target(nothing)
    @test_throws "`physical_chart`" RC.physical_chart(nothing, nothing, nothing)
    for n in (:sphere_radius, :HEIGHT_NFREE, :height_basis, :height_manifold, :height_nparams, :board_surface, :set_surface!, :load_distance_map)
        @test Base.ispublic(RC, n)                                                 # the calibration's share of the board stays here
    end
    # the camera-first API (owner 2026-11-20): the kernels' View record and view_of are internal
    @test !Base.ispublic(RC, :View) && !Base.ispublic(RC, :view_of)
    # the retired names (RC 0.11.0) are refused by name with Core's replacement, never public
    @test_throws "use VisualGeometryToolkitCore's `backproject(camera, 𝐩)`" RC.π⁻¹(nothing, nothing)
    @test_throws "use VisualGeometryToolkitCore's `project(camera, X)`" RC.project_pixel(nothing, nothing, nothing)
    @test !Base.ispublic(RC, :π⁻¹) && !Base.ispublic(RC, :project_pixel)
end
