# The samples of a run (owner, 2026-11-15): a FIXED COUNT per view, `masks.samples_per_view`, drawn without replacement from the
# view's USABLE observations (the board-valid pixels of the sampling region that the observation model keeps: a RAW photograph's
# green photosites, no defect, uncensored) — so a small image circle and a lens filling the sensor get the same count; a view with
# fewer uses them all, its shortfall recorded. Each observation is drawn ∝ u^α (`masks.gradient_bias`), u the ECDF rank of its edge
# strength (`masks.edge_detector`, the Gaussian derivative): α = 0 the uniform draw itself (RC 0.11.5's default; 0.11.4's was α = 1). Two more
# modes of the one draw (the sampling-strategy experiment, RC 0.11.5): `masks.edge_fraction` (the rule of 2026-10-31, restored
# verbatim) and `masks.addon_samples` (M more ∝ u^β on the same stream).

@testitem "samples: a fixed count of the usable observations per view" setup = [RCFixtures] begin
    using RadiometricCamera: draw_samples, sample_record, GaussianDerivativeEdges
    using StableRNGs: StableRNG
    W, H = 400, 300
    green = [isodd(u + v) for v in 1:H, u in 1:W]                          # a Bayer mosaic's green photosites (the observed channel)
    small = VgtkCore.pixel_mask(VgtkCore.Rect2{Float64}(0.0, 0.0, W, H) ∩ VgtkCore.Ellipse(VgtkCore.Point2{Float64}(200.0, 150.0), 60.0, 55.0, 0.3), W, H)
    full = trues(H, W)                                                     # a lens filling the sensor
    img = [1.5 + sin(u / 7) * cos(v / 5) for v in 1:H, u in 1:W]            # a photograph with edges
    N = 2000; d = GaussianDerivativeEdges(1.5); cpu = RadiometricCamera.KernelAbstractions.CPU()
    for region in (small, full), α in (0.0, 2.0)
        usable = region .& green
        s = draw_samples(StableRNG(1), usable, img, N; gradient_bias = α, edge_detector = d, device = cpu)
        @test count(s) == N                                                # exact, whatever the region's size
        @test all(s .<= usable)                                            # every sample in the region and the channel
        @test s == draw_samples(StableRNG(1), usable, img, N; gradient_bias = α, edge_detector = d, device = cpu)   # seeded: reproducible
        @test s != draw_samples(StableRNG(2), usable, img, N; gradient_bias = α, edge_detector = d, device = cpu)   # another seed
    end
    # a view with fewer usable observations than N uses them all, and its shortfall is recorded; never padded
    tiny = falses(H, W); tiny[150, 100:2:180] .= true
    mt = draw_samples(StableRNG(1), tiny, img, N; gradient_bias = 2.0, edge_detector = d)
    @test mt == tiny
    r = sample_record(N, [draw_samples(StableRNG(1), full .& green, img, N; edge_detector = d), mt])
    @test r.per_view == N && r.drawn == [N, count(tiny)] && r.shortfall == [0, N - count(tiny)]
    @test_throws ArgumentError draw_samples(StableRNG(1), tiny, img, 0; edge_detector = d)
    # the start draws from the observe()'d photosites of the sampling region, the start view's mask test (`BoardValidity`) the
    # usable ones among them, tested in the draw's key order (RC 0.11.5); the Bernoulli share and the cap are retired by name
    src = read(joinpath(pkgdir(RadiometricCamera), "src", "start", "prepare.jl"), String)
    @test occursin("masks = [inreg for _ in 1:K]", src) && !occursin("compute_mask(", src)
    @test occursin("img, masks[k] = observe(pset::PhotoSet, k, masks[k]; name = names[k])", src)
    @test occursin("draw_samples(rng, BoardValidity(mask_views[k], board, st; δ = 0.01, backend), masks[k], img, N;", src)
    @test !occursin("lm_fraction", src) && !occursin("max_pixels_per_view", src)
    for k in ("lm_fraction", "max_pixels_per_view")
        e = try RC.pipeline_options(Dict("masks" => Dict(k => 1))); nothing catch err; err end
        @test e isa ArgumentError && occursin("samples_per_view", sprint(showerror, e))
    end
    @test RC.CalibrationSettings().masks.samples_per_view == 27500
end

@testitem "samples: α = 0 is the uniform draw, bit for bit" setup = [RCFixtures] begin
    using RadiometricCamera: draw_samples, GaussianDerivativeEdges, _subsample
    using StableRNGs: StableRNG
    H, W = 120, 160
    usable = [isodd(u + v) && (u - 80)^2 + (v - 60)^2 < 55^2 for v in 1:H, u in 1:W]
    img = [1.0 + (u > 80) for v in 1:H, u in 1:W]
    idx = findall(usable); n = 700
    s = draw_samples(StableRNG(7), usable, img, n; gradient_bias = 0.0, edge_detector = GaussianDerivativeEdges(1.5))
    ref = falses(H, W); ref[idx[_subsample(StableRNG(7), length(idx), n)]] .= true          # rc-focus-fix's uniform draw (RC 0.11.4)
    @test s == ref
    # the weighted draw with equal weights IS the uniform draw: the exponential keys order the items as the uniforms do
    for c in (1.0, 0.37, 5.0)
        @test _subsample(StableRNG(3), length(idx), n, fill(c, length(idx))) == _subsample(StableRNG(3), length(idx), n)
    end
end

@testitem "samples: α > 0 raises the edge share monotonically, α < 0 lowers it" setup = [RCFixtures] begin
    using RadiometricCamera: draw_samples, GaussianDerivativeEdges
    using StableRNGs: StableRNG
    H, W = 120, 200
    usable = trues(H, W)
    img = [0.2 + 0.6 * (u > 100) for v in 1:H, u in 1:W]                    # one vertical step at u = 100 | 101
    near(m) = count(I -> abs(I[2] - 100.5) <= 4, findall(m)) / count(m)     # the edge share: within 4 px of the step
    d = GaussianDerivativeEdges(1.5)
    shares = [near(draw_samples(StableRNG(1), usable, img, 1500; gradient_bias = α, edge_detector = d)) for α in (-2.0, 0.0, 1.0, 2.0, 4.0)]
    @test issorted(shares) && shares[1] < shares[2] < shares[3] < shares[5]
    @test shares[2] ≈ 8 * H / (H * W) atol = 0.02                          # uniform: the band's share of the image
end

@testitem "samples: the ECDF rank is invariant to an affine change of intensity" setup = [RCFixtures] begin
    using RadiometricCamera: edge_strength, edge_rank, GaussianDerivativeEdges, draw_samples
    using StableRNGs: StableRNG
    using Random: Xoshiro
    H, W = 90, 110
    usable = [isodd(u + v) for v in 1:H, u in 1:W]
    r = Xoshiro(1); img = [sin(u / 6) + 0.5 * cos(v / 9) + 0.05 * randn(r) + 2.0 for v in 1:H, u in 1:W]
    idx = findall(usable); d = GaussianDerivativeEdges(1.5)
    u1 = edge_rank(edge_strength(d, img, usable), idx)
    u2 = edge_rank(edge_strength(d, 3.7 .* img .+ 0.25, usable), idx)                  # another exposure and black level
    # equal up to the filter's float rounding, which can swap two near-equal strengths' adjacent ranks (one rank: 1/n)
    @test maximum(abs, u1 .- u2) <= 2 / length(idx) && all(x -> 0 < x <= 1, u1) && maximum(u1) == 1
    @test draw_samples(StableRNG(2), usable, img, 900; gradient_bias = 2.0, edge_detector = d) ==
          draw_samples(StableRNG(2), usable, 3.7 .* img .+ 0.25, 900; gradient_bias = 2.0, edge_detector = d)
end

@testitem "samples: edge_fraction is the rule of 2026-10-31, verbatim (RC 0.11.4)" setup = [RCFixtures] begin
    using RadiometricCamera: draw_samples, GaussianDerivativeEdges, _subsample, _edge_select, masked_pixels
    using StableRNGs: StableRNG
    H, W = 120, 160
    green = [isodd(u + v) for v in 1:H, u in 1:W]
    usable = green .& [(u - 80)^2 + (v - 60)^2 < 55^2 for v in 1:H, u in 1:W]
    img = [sin(u / 7) * cos(v / 5) for v in 1:H, u in 1:W]
    idx = findall(usable); n = 500; q = 0.37; d = GaussianDerivativeEdges(1.5)
    s = draw_samples(StableRNG(4), usable, img, n; edge_fraction = q, edge_detector = d, diagonal = true)
    # RC 0.11.4's draw: a uniform pool of ⌈n / q⌉ (`_subsample`), then its n strongest |∇ϕ|² (`_edge_select`), on the same stream
    pool = falses(H, W); pool[idx[_subsample(StableRNG(4), length(idx), ceil(Int, n / q))]] .= true
    @test s == _edge_select(pool, img, n; diagonal = true) && count(s) == n && all(s .<= pool)
    # its rule: every kept pixel at least as strong as every dropped interior one of the pool (the diagonal differences of the photosites)
    g(u, v) = ((img[v + 1, u + 1] - img[v - 1, u - 1])^2 + (img[v + 1, u - 1] - img[v - 1, u + 1])^2) / 8
    inner = [(u, v) for (u, v) in masked_pixels(pool) if 2 <= u <= W - 1 && 2 <= v <= H - 1]
    @test minimum(g(u, v) for (u, v) in masked_pixels(s)) >= maximum((g(u, v) for (u, v) in inner if !s[v, u]); init = -Inf)
    # a jump in the green between columns 12 and 13 under red and blue noise: the diagonal rule reads the green alone
    Hs, Ws = 20, 26; gs = [isodd(i + j) for i in 1:Hs, j in 1:Ws]
    jump = [gs[i, j] ? (j <= 12 ? 0.2 : 0.8) : rand(StableRNG(i * Ws + j)) for i in 1:Hs, j in 1:Ws]
    beside = falses(Hs, Ws); for i in 2:Hs-1, j in (12, 13); beside[i, j] = gs[i, j]; end
    @test _edge_select(gs, jump, count(beside); diagonal = true) == beside
    @test _edge_select(gs, jump, count(beside)) != beside                 # the edge-neighbour rule reads the R/B noise
end

@testitem "samples: the add-on is M more ∝ u^β on the same stream, the n untouched" setup = [RCFixtures] begin
    using RadiometricCamera: draw_samples, GaussianDerivativeEdges
    using StableRNGs: StableRNG
    H, W = 120, 200
    usable = trues(H, W)
    img = [0.2 + 0.6 * (u > 100) for v in 1:H, u in 1:W]                    # one vertical step at u = 100 | 101
    near(m) = count(I -> abs(I[2] - 100.5) <= 4, findall(m))
    d = GaussianDerivativeEdges(1.5); n = 1500; M = 750
    for α in (0.0, 1.0)
        base = draw_samples(StableRNG(5), usable, img, n; gradient_bias = α, edge_detector = d)
        both = draw_samples(StableRNG(5), usable, img, n; gradient_bias = α, edge_detector = d, addon_samples = M, addon_gradient_bias = 2.0)
        @test count(both) == n + M && all(base .<= both)                  # the n are the draw without the add-on, bit for bit
        extra = both .& .!base
        @test near(extra) / M > 2 * 8 / W                                  # the add-on is edge-biased: twice the band's share of the image
        flat = draw_samples(StableRNG(5), usable, img, n; gradient_bias = α, edge_detector = d, addon_samples = M)   # β = 0: uniform among the rest
        @test count(flat) == n + M && all(base .<= flat) && near(flat .& .!base) / M < near(extra) / M
    end
    # the next view's stream is the draw's without the add-on: no extra random number
    r1 = StableRNG(9); r2 = StableRNG(9)
    draw_samples(r1, usable, img, n; edge_detector = d); draw_samples(r2, usable, img, n; edge_detector = d, addon_samples = M, addon_gradient_bias = 2.0)
    @test rand(r1) == rand(r2)
    # a view with fewer usable observations than n + M uses them all
    tiny = falses(H, W); tiny[60, 1:2:200] .= true
    @test draw_samples(StableRNG(1), tiny, img, 80; edge_detector = d, addon_samples = 40, addon_gradient_bias = 2.0) == tiny
    # n < N ≤ n + M: every usable observation, and the stream consumed as by the draw without the add-on (code review 0.11.5)
    q1 = StableRNG(3); q2 = StableRNG(3)
    draw_samples(q1, tiny, img, 80; edge_detector = d); @test draw_samples(q2, tiny, img, 80; edge_detector = d, addon_samples = 40) == tiny
    @test rand(q1) == rand(q2)
end

@testitem "samples: the options — the detector table, α = 0 (uniform) the default, the modes together, L0 and importance weights retired" setup = [RCFixtures] begin
    using RadiometricCamera: GaussianDerivativeEdges
    o = RC.CalibrationSettings().masks
    @test o.gradient_bias == 0.0 && o.edge_detector == GaussianDerivativeEdges(1.5) && o.edge_fraction == 0.0 && o.addon_samples == 0 && o.addon_gradient_bias == 0.0
    @test !hasfield(typeof(o), :importance_weights)
    p = last(RC.pipeline_options(Dict("masks" => Dict("gradient_bias" => 2.0, "edge_detector" => Dict("type" => "gaussian_derivative", "sigma" => 2.5))))).masks
    @test p.gradient_bias == 2.0 && p.edge_detector == GaussianDerivativeEdges(2.5)
    @test last(RC.pipeline_options(RC.config_dict(RC.pipeline_options(Dict("masks" => Dict("edge_detector" => Dict("type" => "gaussian_derivative", "sigma" => 2.5))))...))).masks.edge_detector ==
          GaussianDerivativeEdges(2.5)                                                       # the writer round-trips
    @test last(RC.pipeline_options(Dict("masks" => Dict("gradient_bias" => 0.0, "edge_fraction" => 0.37)))).masks.edge_fraction == 0.37
    a = RC.pipeline_options(Dict("masks" => Dict("gradient_bias" => 0.0, "addon_samples" => 100, "addon_gradient_bias" => 2.0)))
    @test last(RC.pipeline_options(RC.config_dict(a...))).masks == last(a).masks                 # the writer round-trips (every key, valid together)
    e = try RC.pipeline_options(Dict("masks" => Dict("edge_fraction" => 0.37, "gradient_bias" => 1.0))); nothing catch err; err end   # the old rule needs α = 0
    @test e isa ArgumentError && occursin("gradient_bias", sprint(showerror, e))
    @test last(RC.pipeline_options(Dict("masks" => Dict("edge_fraction" => 0.37)))).masks.edge_fraction == 0.37    # the default α = 0 admits it
    refused(m) = (try RC.pipeline_options(Dict("masks" => m)); nothing catch err; err end) isa ArgumentError
    @test refused(Dict("edge_fraction" => 1.0, "gradient_bias" => 0.0))
    @test refused(Dict("edge_fraction" => 0.37, "gradient_bias" => 0.0, "addon_samples" => 10))   # one mode at a time
    @test refused(Dict("addon_gradient_bias" => 2.0))                               # inapplicable without an add-on
    @test refused(Dict("addon_samples" => -1))
    @test refused(Dict("edge_detector" => Dict("type" => "l0", "lambda" => 0.02)))   # retired, by name
    @test refused(Dict("edge_detector" => Dict("type" => "canny")))
    @test refused(Dict("edge_detector" => Dict("type" => "gaussian_derivative")))          # sigma required
    @test refused(Dict("edge_detector" => Dict("type" => "gaussian_derivative", "sigma" => 1.5, "lambda" => 0.1)))   # another key
    e = try RC.pipeline_options(Dict("masks" => Dict("importance_weights" => "on"))); nothing catch err; err end
    @test e isa ArgumentError && occursin("retired", sprint(showerror, e))
end

# The mask test in the draw's key order (RC 0.11.5, the 0.11.4 review's item 7a): every candidate gets its uniform key, and where no
# weight reads the edge ranks the draw is the m smallest keys among the usable candidates — so the candidates are tested in key
# order until m are valid, the very draw of the test of every candidate, at a few m pixels tested instead of the whole region.
@testitem "samples: the mask test in key order is the draw of the full test, a few n pixels tested" setup = [RCFixtures] begin
    using RadiometricCamera: draw_samples, GaussianDerivativeEdges, _edge_select, edge_rank, edge_strength, edge_plane
    using StableRNGs: StableRNG
    H, W = 300, 400
    cand = [isodd(u + v) && (u - 200)^2 + (v - 150)^2 < 140^2 for v in 1:H, u in 1:W]   # the region's green photosites
    board = [u + 0.6v < 330 for v in 1:H, u in 1:W]                                    # the board-valid pixels (the mask test)
    tested = Ref(0); valid = ids -> (tested[] += length(ids); board[ids])               # the mask test, counting the pixels it tests
    img = [1.5 + sin(u / 7) * cos(v / 5) for v in 1:H, u in 1:W]
    d = GaussianDerivativeEdges(1.5); n = 1200; ids = findall(vec(cand))
    # the definition: every candidate's key U (rand(rng, #candidates)), the m smallest among the usable ones (board-valid)
    function ref(seed, m)
        U = rand(StableRNG(seed), length(ids)); ok = findall(board[ids])
        r = falses(H, W); r[ids[ok[partialsortperm(U[ok], 1:min(m, length(ok)))]]] .= true
        return r
    end
    tested[] = 0; s = draw_samples(StableRNG(1), valid, cand, img, n; edge_detector = d)
    @test s == ref(1, n) && count(s) == n && all(s .<= cand .& board)
    p = count(cand .& board) / count(cand)                                             # the valid share: n / p tested, expected
    @test tested[] <= 2n / p && length(ids) > 20n                                      # a few n tested, not the region's candidates
    # the uniform add-on is the next M keys, the n the draw without it; the rule of 2026-10-31's pool the ⌈n / q⌉ smallest
    M = 300; tested[] = 0
    a = draw_samples(StableRNG(1), valid, cand, img, n; edge_detector = d, addon_samples = M)
    @test a == ref(1, n + M) && all(s .<= a) && tested[] <= 2(n + M) / p
    q = 0.37; e = draw_samples(StableRNG(1), valid, cand, img, n; edge_detector = d, edge_fraction = q, diagonal = true)
    @test e == _edge_select(ref(1, ceil(Int, n / q)), img, n; diagonal = true)
    # a weight that reads the edge ranks tests every candidate: the weighted draw of the usable ones under the same keys
    tested[] = 0; w = draw_samples(StableRNG(1), valid, cand, img, n; gradient_bias = 2.0, edge_detector = d)
    @test tested[] == length(ids)
    U = rand(StableRNG(1), length(ids)); ok = findall(board[ids]); idx = ids[ok]
    usable = falses(H, W); usable[idx] .= true
    u = edge_rank(edge_strength(d, edge_plane(img, (1,)), usable), idx)
    wr = falses(H, W); wr[idx[partialsortperm(-log1p.(-U[ok]) ./ u .^ 2.0, 1:n)]] .= true
    @test w == wr
    # a view with fewer usable candidates than n: all of them, every candidate tested; each view consumes #candidates uniforms
    few = [u + 0.6v < 60 for v in 1:H, u in 1:W]; tf = Ref(0)
    @test draw_samples(StableRNG(1), ids -> (tf[] += length(ids); few[ids]), cand, img, n; edge_detector = d) == cand .& few && tf[] == length(ids)
    r1 = StableRNG(5); r2 = StableRNG(5); draw_samples(r1, valid, cand, img, n; edge_detector = d); rand(r2, length(ids))
    @test rand(r1) == rand(r2)
end

@testitem "samples: BoardValidity is compute_mask's test, the draw tests a few n of the region's pixels" setup = [RCFixtures] begin
    using RadiometricCamera: BoardValidity, board_valid, compute_mask, draw_samples, Focus, GaussianDerivativeEdges,
                             TexturedTarget, RenderSettings, View
    using StableRNGs: StableRNG
    using StaticArrays: SMatrix, SVector
    using VisualGeometryToolkitCore: exp_so3
    cpu = RadiometricCamera.KernelAbstractions.CPU()
    W = H = 160
    board = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], fixture_board()), [0.0, 0.0, 1.5])
    pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.1, 0.05, 0.0))), SVector{3}(SVector(0.05, -0.05, 1.5)); check = false)
    camera = RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 140.0, (80.0, 80.0), W, H), Focus(3, 1.5, W, H), 0.0386,
                                    VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2))
    view = View(pose, camera)        # the mask test's internal record (BoardValidity, board_valid take the kernels' view)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
    region = [(u - 80)^2 + (v - 80)^2 < 75^2 for v in 1:H, u in 1:W]
    mask = compute_mask(camera, pose, board, st; δ = 0.01, external = region, backend = cpu)
    @test 0 < count(mask) < count(region)                                                # the board does not fill the region
    ids = findall(vec(region)); t = BoardValidity(view, board, st; δ = 0.01, backend = cpu)
    @test t(ids) == mask[ids] && board_valid(view, board, st, ids[1:2:end]; δ = 0.01, backend = cpu) == mask[ids[1:2:end]]
    # the draw through the view's test: the n smallest keys among compute_mask's pixels; its work, the pixels tested, against the
    # full-region test's count(region) (the timing check of the 0.11.4 review's item 7a; seconds: test/regression/mask_draw_benchmark.jl)
    n = 400; tested = Ref(0)
    s = draw_samples(StableRNG(3), ids -> (tested[] += length(ids); t(ids)), region, ones(H, W), n; edge_detector = GaussianDerivativeEdges(1.5), device = cpu)
    U = rand(StableRNG(3), length(ids)); ok = findall(mask[ids]); ref = falses(H, W); ref[ids[ok[partialsortperm(U[ok], 1:n)]]] .= true
    @test s == ref && tested[] <= 2n * count(region) / count(mask) && count(region) > 30n
end
