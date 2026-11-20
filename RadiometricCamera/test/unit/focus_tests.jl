@testitem "TPS / focus" setup = [RCFixtures] begin
    using RadiometricCamera: Focus, FocusSpline, in_domain, n_nodes, t̂, 𝔱
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    # the focus surface is Core's thin-plate spline (its interpolation and bending energy are Core's tests): the Focus of affine nodal
    # values is that affine function, and of any values interpolates them
    W, H = 400, 300                                                  # the sensor [0, W] × [0, H] (pixels)
    tps = FocusSpline(VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, W, H)))
    @test tps.nodes[1] == SVector(0.0, 0.0) && tps.nodes[25] == SVector(400.0, 300.0)
    vals = [0.3 + 1e-3 * n[1] - 2e-3 * n[2] for n in tps.nodes]
    fa = Focus(tps, vals)
    @test maximum(abs, fa.w) < 1e-9
    @test t̂(fa, SVector(244.0, 30.0)) ≈ 0.3 + 0.244 - 0.06 atol = 1e-10
    vals2 = rand(25)
    @test maximum(abs(t̂(Focus(tps, vals2), tps.nodes[i]) - vals2[i]) for i in 1:25) < 1e-9
    f = Focus(5, 1.5, W, H)
    @test 𝔱(f, SVector(280.0, 240.0)) ≈ 1.5 && t̂(f, SVector(40.0, 180.0)) ≈ 1 / 1.5
    fd = Focus(tps, ForwardDiff.Dual.(vals2, 1.0))
    @test t̂(fd, SVector(240.0, 180.0)) isa ForwardDiff.Dual
    @test n_nodes(f) == 25

    # The focus start (focus_start): exactly cos αᵢ / 𝔱₀ when every node is usable with cos α ≥ 0; every start value floored at 0 (the
    # plane start has no meaning past 90°, and the LM keeps t̂ ≥ 0); a node outside the lens's domain or outside the given support is
    # filled in t̂ space by the TPS through the usable nodes, clamped to their (non-negative) range; fewer than 3 non-collinear usable
    # nodes fill with their mean; none is an ArgumentError naming the lens
    let Wd = 100, nd = VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, Wd, Wd))
        ds = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 128.8, (51.0, 48.0), Wd, Wd)
        @test RC.focus_start(ds, nd, 1.5) == [normalize(VgtkCore.backproject(ds, 𝐩))[3] / 1.5 for 𝐩 in nd]
    end
    let Wk = 5208, Hk = 3476, nodes = VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, Wk, Hk))
        kb = cam(VgtkCore.KannalaBrandt(-0.0416, 0.000407, 0.000185, -7.7e-5), 932.2, (2604.0, 1738.0), Wk, Hk; f_y = 931.6)   # the blob_kb lens: 16 nodes past θ = π
        imaged = [in_domain(kb, 𝐩) for 𝐩 in nodes]
        t = RC.focus_start(kb, nodes, 0.232)
        @test 3 <= count(imaged) < length(nodes) && all(isfinite, t) && all(>=(0), t)
        @test t[imaged] == [max(normalize(VgtkCore.backproject(kb, 𝐩))[3] / 0.232, 0.0) for 𝐩 in nodes[imaged]]
        @test any(i -> imaged[i] && normalize(VgtkCore.backproject(kb, nodes[i]))[3] < 0, eachindex(nodes)) && count(iszero, t[imaged]) >= 1   # back rays floored
        lo, hi = extrema(t[imaged])
        f = Focus(FocusSpline(nodes[imaged]), t[imaged])
        @test all(t[i] == clamp(RC.t̂(f, nodes[i]), lo, hi) for i in findall(.!imaged))
        # a support (the blob_kb image circle: the centre cross of 5 nodes) restricts the usable nodes further
        sup = [hypot(𝐩[1] - 2604, 𝐩[2] - 1738) < 1318 for 𝐩 in nodes]
        ts = RC.focus_start(kb, nodes, 0.232; support = sup)
        @test count(sup .& imaged) == 5 && ts[sup] == t[sup] && all(isfinite, ts) && all(>=(0), ts)
        # collinear usable nodes (one row): their mean; none usable: ArgumentError
        v = fill(NaN, 25); v[1:5] .= 1.0:5.0
        @test RC.fill_unimaged_nodes(v, nodes) == [1.0:5.0; fill(3.0, 20)]
        @test_throws ArgumentError RC.fill_unimaged_nodes(fill(NaN, 25), nodes; lens = kb)
    end
end

@testitem "focus nodes over the imaged region" setup = [RCFixtures] begin
    using StaticArrays
    W, H = 400, 300
    pin = cam(VgtkCore.Pinhole(), 300.0, (200.0, 150.0), W, H)       # a W × H pixel camera: the run's region is Core's imaged_region of it
    fo = RC.PipelineOptions().focus
    # no image circle: the sensor's 5 × 5 grid, every node kept (the sensor's edges are exact)
    @test RC.focus_grid_nodes(VgtkCore.imaged_region(pin)) == VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, W, H))
    @test RC.focus_start_nodes(pin, fo).nodes == VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, W, H))
    # an image circle: the grid over its bounding box; inside ρ ≤ focus.lens_coverage_scale the inner 3 × 3 (the 4 corners and the 8 (±1, ±½) nodes
    # lie outside the circle, the 4 side midpoints ON its rim)
    circle = VgtkCore.Ellipse(VgtkCore.Point2{Float64}(200.0, 150.0), 120.0, 120.0, 0.0)
    r = VgtkCore.imaged_region(VgtkCore.with_lens_coverage(pin, circle))
    nb = RC.focus_grid_nodes(r)
    @test nb[1] == SVector(80.0, 30.0) && nb[13] == SVector(200.0, 150.0) && nb[25] == SVector(320.0, 270.0)
    @test count(p -> RC._in_region(r, p...), nb) == 13                                  # membership alone keeps the rim's four
    @test length(RC.focus_start_nodes(VgtkCore.with_lens_coverage(pin, circle), fo).nodes) == 9
    # a circle reaching past the sensor: the box is clipped to the sensor
    big = VgtkCore.Ellipse(VgtkCore.Point2{Float64}(200.0, 150.0), 260.0, 260.0, 0.0)
    @test RC.focus_grid_nodes(VgtkCore.imaged_region(VgtkCore.with_lens_coverage(pin, big))) == VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, W, H))
    # a rotated ellipse: its axis-aligned extent
    ell = VgtkCore.Ellipse(VgtkCore.Point2{Float64}(200.0, 150.0), 120.0, 80.0, 0.5)
    hx = hypot(120cos(0.5), 80sin(0.5)); hy = hypot(120sin(0.5), 80cos(0.5))
    ne = RC.focus_grid_nodes(VgtkCore.imaged_region(VgtkCore.with_lens_coverage(pin, ell)))
    @test ne[1] ≈ SVector(200 - hx, 150 - hy) && ne[25] ≈ SVector(200 + hx, 150 + hy)
    # the TPS needs at least 4 non-collinear nodes
    @test_throws ArgumentError RC._check_focus_nodes(nb[1:3], nothing)
    @test_throws ArgumentError RC._check_focus_nodes(nb[1:5], nothing)          # 5 collinear nodes (the first row)
    @test RC._check_focus_nodes(nb[[1, 5, 21, 25]], nothing) == nb[[1, 5, 21, 25]]
end

@testitem "focus_ab node counts: a masked pixel without a ray" setup = [RCFixtures] begin
    using RadiometricCamera: Focus
    using StaticArrays
    # a lens that gives no ray (NaN) right of u = 300: the pixel counts skip it and count it, the largest chief-ray angle stays
    # finite (2026-11-08: two refined parametric lenses had masked corner pixels past their invertible range, focus_ab.json failed)
    struct _HalfLens end
    RadiometricCamera._chief_ray(::_HalfLens, 𝐩::StaticVector{2}) = 𝐩[1] > 300 ? SVector(NaN, NaN, NaN) : SVector(𝐩[1] - 200, 𝐩[2] - 150, 400.0)
    W, H = 400, 300
    f = Focus(5, 1.5, W, H)
    px = [[(10, 10), (390, 20), (100, 200)], [(250, 150), (399, 299)]]
    cnt, θpx, noray = RadiometricCamera.focus_node_pixel_counts(f, _HalfLens(), px, W, H)
    @test noray == 2 && sum(cnt) == 3
    @test isfinite(θpx) && 0 < θpx < π / 2
end
