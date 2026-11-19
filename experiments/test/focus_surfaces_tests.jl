# The paper's focus-surface figure (scripts/focus_surfaces.jl, src/figures.jl focus_surface_figure). The driver's --plan is the
# drivers item's (drivers_tests.jl); the chrome tables moved with their stage (chrome_evaluation_paper_tests.jl).
@testitem "focus surfaces figure: a synthetic sphere and plane" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("figures.jl"))
    using .Figures: focus_surface_figure
    # a sphere of radius 0.7 m about the camera over a 120° cone and a plane at z = 0.5 m over a 40° one, NaN outside: both render
    sphere = let n = 25, X = fill(NaN, n, n), Y = copy(X), Z = copy(X), D = copy(X)
        for j in 1:n, i in 1:n
            θx = (i - (n + 1) / 2) / n * 2π / 3; θy = (j - (n + 1) / 2) / n * 2π / 3
            d = [tan(θx), tan(θy), 1.0]; d ./= sqrt(sum(abs2, d)); hypot(θx, θy) <= π / 3 || continue
            X[i, j], Y[i, j], Z[i, j], D[i, j] = (0.7 .* d)..., 0.7
        end
        ("fisheye · sphere", X, Y, Z, D)
    end
    plane = let n = 25, X = fill(NaN, n, n), Y = copy(X), Z = copy(X), D = copy(X)
        for j in 1:n, i in 1:n
            x = (i - (n + 1) / 2) / n * 0.4; y = (j - (n + 1) / 2) / n * 0.3
            X[i, j], Y[i, j], Z[i, j], D[i, j] = x, y, 0.5, hypot(x, y, 0.5)
        end
        ("Tamron · plane", X, Y, Z, D)
    end
    # the gridded look (2026-11-15): no colour bar, a line along every grid_every-th row and column, a view elevation per panel
    fig = focus_surface_figure([sphere, plane]; size_in = (3.3, 1.75), dpi = 60, grid_every = 6, elevations_deg = [14.4, 36.0])
    out = Figures._save(fig, "focus.png", mktempdir())
    @test isfile(out) && filesize(out) > 1000
    @test (img = Figures.PNGFiles.load(out); (size(img, 2), size(img, 1))) == (198, 105)   # 3.3 × 1.75 in at 60 dpi
    @test !any(x -> x isa Figures.Makie.Colorbar, fig.content) && count(x -> x isa Figures.Makie.Axis3, fig.content) == 2
    @test_throws ArgumentError focus_surface_figure([]; dpi = 60)
    @test_throws ArgumentError focus_surface_figure([("empty", fill(NaN, 2, 2), fill(NaN, 2, 2), fill(NaN, 2, 2), fill(NaN, 2, 2))]; dpi = 60)
    @test_throws ArgumentError focus_surface_figure([sphere, plane]; dpi = 60, elevations_deg = [14.4])   # one elevation per panel
    @test_throws ArgumentError focus_surface_figure([sphere]; dpi = 60, grid_every = 1)
end
