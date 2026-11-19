# The package against the golden outputs recorded from THIS branch's code (test/regression/golden_migration_snd_try.json, recorded by
# test/regression/record_golden.jl; scenes in test/regression/golden_scenes.jl): flux of three lens / board combinations, the residual and
# Jacobian of a per-view problem and of two per-tile problems (warping, printed edge), the closed-form joint
# radiometry and three Levenberg–Marquardt iterations. A migration of the code or of its dependencies must reproduce every value to
# roundoff (rtol 1e-10 relative to each vector's scale); a deliberate numerical change re-records the file (record_golden.jl).
# Tag :golden: test/runtests.jl runs this item last, on one worker with 8 threads (the closed-form radiometry sums over thread
# chunks, so the thread count is part of the arithmetic; 8 is the JULIA_NUM_THREADS the pipeline runner fixes).
@testitem "golden outputs of the package before the migration" tags = [:golden] begin
    import VisualGeometryToolkitIO as VgtkIO
    # The scenes are evaluated in a module of their own, so that they see the package's functions and nothing else (when every test
    # file shared Main, their globals, e.g. `lens = DoubleSphere(…)`, shadowed the package functions the scenes call).
    module GoldenScenes
    const OLD_STACK = false
    include(joinpath(@__DIR__, "golden_scenes.jl"))
    end
    record = VgtkIO.read_record(joinpath(@__DIR__, "golden_migration_snd_try.json"))
    @test record.schema == GoldenScenes.GOLDEN_SCHEMA                  # record_golden.jl states it (IO 0.10's `write_record`)
    gold = record.outputs
    now_out = GoldenScenes.gold_outputs()
    relclose(a, b; rtol = 1e-10) = (a = Float64.(collect(Iterators.flatten(a))); b = Float64.(collect(Iterators.flatten(b)));
                                    length(a) == length(b) && maximum(abs, a .- b; init = 0.0) <= rtol * max(maximum(abs, b; init = 0.0), 1e-300))
    @testset "golden: $k" for k in sort(collect(keys(now_out)))
        g = gold[Symbol(k)]; n = now_out[k]
        if n isa NamedTuple
            for f in keys(n)
                f === :rows && (@test collect(n[f]) == collect(g[f]); continue)
                f === :size && (@test collect(n[f]) == collect(g[f]); continue)
                # the joint radiometry fits its targets exactly (they are rendered with the same Θ_r and V): its cost is the roundoff floor
                # (≈ 1e-33), whose value depends on the summation order (Pkg.test compiles with --check-bounds=yes) → an absolute bound
                k == "radiometry_gn" && f === :cost && (@test abs(n[f]) < 1e-24 && abs(g[f]) < 1e-24; continue)
                # the three LM iterations cancel the cost from cost0 ≈ 7.65 to ≈ 8.79e-8 (×8.7e7), so the final cost carries an absolute
                # rounding ∝ eps·cost0, not eps·cost (desktop vs dgx01: 9.0e-18 apart, 1.02e-10 relative) → bounded by eps·cost0 (1.7e-15);
                # cost0 itself (no cancellation) is under the relclose check below
                k == "lm3" && f === :cost && (@test abs(n[f] - g[f]) <= eps(Float64) * g.cost0; continue)
                # The tile Jacobian's free-height columns are in the warping's free-height basis B, the null space of Core's
                # TargetManifold, which is unique only up to rotation, and its rounding follows the BLAS thread count (1 vs 16 threads:
                # B moves by up to 0.35 while BBᵀ agrees to 1.1e-15). Those columns are compared through the stored basis-invariant
                # `height_proj`; the per-column statistics compare every other column, as before.
                if k == "tile_warping_jacobian" && f in (:sum, :sumabs, :sumsq, :at_rows)
                    hc = collect(Int, g.height_cols)
                    other(x) = first(x) isa AbstractVector ? [deleteat!(collect(r), hc) for r in x] : deleteat!(collect(x), hc)
                    ok = relclose(other(n[f]), other(g[f])); ok || @info "golden mismatch" k f
                    @test ok
                    continue
                end
                ok = relclose(n[f], g[f]); ok || @info "golden mismatch" k f
                @test ok
            end
        else
            ok = relclose(n, g); ok || @info "golden mismatch" k
            @test ok
        end
    end
end
