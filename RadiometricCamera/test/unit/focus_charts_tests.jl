# EXPERIMENT (focus-ab): the direction-chart and spherical focus surfaces (src/camera/focus_charts.jl).
@testitem "focus charts (focus-ab experiment)" setup = [RCFixtures] begin
    using RadiometricCamera: Focus, FocusSpline, t̂, View
    using StaticArrays, LinearAlgebra, Statistics
    import RadiometricCamera: ForwardDiff
    import VisualGeometryToolkitCore as VgtkCore

    # (Li₂, Wahba's kernel and the azimuthal-equidistant chart are VisualGeometryToolkitCore's: dilog, spherical_spline_kernel, the
    # equidistant lens — their accuracy is Core's tests)
    @testset "systems interpolate and weights are exact" begin
        nodes = RC.polar_cap_nodes(25, 1.4)
        @test length(nodes) == 25
        lens = cam(VgtkCore.KannalaBrandt(-0.04, 0.003, -0.004, 0.002), 72.0, (200.0, 150.0), 400, 300; f_y = 81.0)   # f_x, f_y, c (pixels), k₁…k₄
        for kind in (1, 2)
            tps = FocusSpline(nodes, kind)
            t = [0.5 + 0.3 * sin(3 * n[1]) * cos(2 * n[2]) for n in nodes]
            f = Focus(tps, t)
            @test f.kind == kind
            for (i, ξ) in enumerate(nodes)
                @test RC.t̂_dir(f, RC.ae_dir(ξ)) ≈ t[i] atol = 1e-10
            end
            𝐩 = SVector(252.0, 126.0)
            g = RC.focus_weights(tps, lens, 𝐩)
            d = RC.chief_dir(lens, 𝐩)
            @test dot(g, t) ≈ RC.t̂_dir(f, d) atol = 1e-12
            @test sum(g) ≈ 1 atol = 1e-10                                              # constants reproduced
            fd = ForwardDiff.gradient(tt -> RC.t̂_dir(Focus(tps, tt), d), t)
            @test fd ≈ g atol = 1e-10
            @test FocusSpline(f).kind == kind
            # the focus smoothness's rows: Core's bending rows of the chart's system over Core's m̄ — ‖L t‖² is Core's bending energy over
            # the mean positive eigenvalue of Q
            L = RC.VgtkCore.bending_rows(tps.sys) ./ sqrt(RC.VgtkCore.mean_positive_eigenvalue(RC.VgtkCore.bending_rows(tps.sys)))
            λ = eigvals(Symmetric(tps.sys.Q)); λp = λ[λ .> 1e-10 * maximum(λ)]
            @test sum(abs2, L * t) ≈ RC.VgtkCore.bending_energy(tps.sys, t) / mean(λp) rtol = 1e-8
            @test norm(L * ones(25)) < 1e-6 * norm(L)                                  # constants carry no bending
        end
    end
    @testset "kind 0 unchanged" begin
        tps = FocusSpline(VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, 400, 300)))
        f = Focus(tps, collect(range(0.1, 2.0; length = 25)))
        @test f.kind == 0 && tps.kind == 0
        lens = cam(VgtkCore.KannalaBrandt(-0.04, 0.003, -0.004, 0.002), 72.0, (200.0, 150.0), 400, 300; f_y = 81.0)
        @test RC.focus_weights(tps, lens, SVector(240.0, 210.0)) == RC.VgtkCore.nodal_weights(tps.sys, SVector(240.0, 210.0))
        v = View(VgtkCore.EuclideanMap(SMatrix{3, 3}(SMatrix{3, 3}(1.0I)), SVector{3}(zero(SVector{3})); check = false), lens, f, 1e-3)
        @test t̂(v, SVector(240.0, 210.0)) === t̂(f, SVector(240.0, 210.0))
    end
end
