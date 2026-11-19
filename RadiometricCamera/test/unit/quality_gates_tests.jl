# The package's tooling gates (package-cleanup Phase 6; RC 0.11.0): the calling conventions (argorder_check, Core's vendored copy,
# against .argorder_allow — the allow-list only shrinks), Aqua's checks one at a time, ExplicitImports (no implicit import, no stale
# explicit one) and the JET ratchet (the report count of the package alone may only fall).

@testitem "quality: calling conventions (argorder_check)" begin
    using Test, RadiometricCamera
    import VisualGeometryToolkitCore
    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "argorder_check.jl"))
    allow = read_allowlist(joinpath(pkgdir(RadiometricCamera), ".argorder_allow"))
    found = check(RadiometricCamera)
    bad = [f for f in found if (f[1], f[2]) ∉ allow]
    @test isempty(bad)
    isempty(bad) || foreach(f -> @warn("argument-order finding", f), bad)
    @test all(a -> any(f -> (f[1], f[2]) == a, found), allow)                   # an allow-list entry that matches nothing is stale
    # the radiance draws take their type after the sampler (Julia's `rand(rng, T, dims)`); the old order is a MethodError
    s = RadiometricCamera.RadianceSampler(8)
    @test_throws MethodError RadiometricCamera.draws(s, UInt64(7), 1, Float32)
    @test RadiometricCamera.draws(s, Float32, UInt64(7), 1) isa NTuple{2, Float32}
end

@testitem "quality: Aqua" begin
    using Test, RadiometricCamera, Aqua
    Aqua.test_unbound_args(RadiometricCamera)
    Aqua.test_undefined_exports(RadiometricCamera)
    Aqua.test_project_extras(RadiometricCamera)
    Aqua.test_deps_compat(RadiometricCamera)
    Aqua.test_piracies(RadiometricCamera)
    Aqua.test_persistent_tasks(RadiometricCamera)
    Aqua.test_undocumented_names(RadiometricCamera)
    # not gated (recorded at the RC 0.11.0 baseline): test_ambiguities (2 between RC's methods and the stack's), test_stale_deps
    # (CUDA: a hard dependency only the extension reads — its move to [weakdeps] needs the device resolver's vendor load moved to the
    # driver; a follow-up)
end

@testitem "quality: ExplicitImports" begin
    using Test, RadiometricCamera, ExplicitImports
    @test check_no_implicit_imports(RadiometricCamera) === nothing
    @test check_no_stale_explicit_imports(RadiometricCamera) === nothing
end

@testitem "quality: JET ratchet" begin
    using Test, RadiometricCamera, JET
    # the count of JET's reports on the package alone, at its RC 0.11.0 baseline: it may only fall — a new report fails here. 36: the
    # T0 count 34 plus the new public entry points start_geometry (two methods) and _start_state re-reporting the existing leaf sites
    # (load_session, load_state, board_features) that prepare_refinement reached (RC 0.11.0 ledger) — no new leaf
    BASELINE = 36
    n = length(JET.get_reports(JET.report_package(RadiometricCamera; target_modules = (RadiometricCamera,), toplevel_logger = nothing)))
    @test n <= BASELINE
    n < BASELINE && @info "JET: $n reports, below the baseline $BASELINE — lower BASELINE to $n"
end
