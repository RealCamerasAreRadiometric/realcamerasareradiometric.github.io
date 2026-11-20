# The run's support is Core's imaged region of a camera whose lens coverage is the capture set's image region: MEASURED on the set's
# own photographs by its keyed step (`measure_image_region`: detect_image_circle; its rc/image_region/v1.0 record, read by
# `run_lens_coverage` through `camera.lens_coverage`) or the sensor, never predicted from a lens (a lens's fold says where the MODEL is
# valid, not where light falls): the grid's lens coverage, the masks, the focus nodes' support and the vignetting's radius all read it.

@testitem "imaged region: the measured image region, stored by the grid" setup = [RCFixtures] begin
    using RadiometricCamera: grid_model, in_domain
    using StaticArrays, Random, Statistics, PNGFiles, ImageCore
    using VisualGeometryToolkitCore: thrown_message
    using VisualRadiometryToolkitCore: TensorProductVignetting, RadialVignetting, with_control_points
    W, H = 520, 348
    # a fisheye whose fold lies inside the sensor: the blob board's Kannala–Brandt fit (its fold at 133°), f = 93.2 px
    kb = VgtkCore.KannalaBrandt(-0.041634590670699925, 0.00040687025248368753, 0.00018535670662213487, -7.735440193838873e-5)
    fish = cam(kb, 93.2, (260.0, 174.0), W, H)
    circle = VgtkCore.Ellipse(VgtkCore.Point2{Float64}(260.0, 174.0), 150.0, 150.0, 0.0)     # the image circle a run measured
    region = VgtkCore.imaged_region(VgtkCore.with_lens_coverage(fish, circle))             # the sensor ∩ the circle (Core)
    raster = [(x, y) for x in 0.5:4:W, y in 0.5:4:H]
    inside(r, (x, y)) = VgtkCore.Point2{Float64}(x, y) ∈ r
    # a grid built from the lens does not inherit its fold: its lens coverage is the one it is given (the source's, the sensor, by
    # default)
    @test VgtkCore.lens_coverage(grid_model(fish, 35, 25)) == VgtkCore.Rect2{Float64}(0, 0, W, H)
    g = VgtkCore.with_lens_coverage(grid_model(fish, 35, 25), circle)
    @test VgtkCore.lens_coverage(g) == circle
    @test all(inside(VgtkCore.imaged_region(g), p) == inside(region, p) for p in raster)
    # the record stores it: a grid read back keeps its lens coverage
    path = tempname() * ".json"; RC.save_grid(path, g)
    @test VgtkCore.lens_coverage(RC.load_grid(path)) == circle
    # in_domain reads the lens coverage: no pixel outside it
    @test all(p -> !in_domain(g, SVector(p...)) || inside(region, p), raster)
    # its present nodes: those whose support meets the region (Core's node_mask) — the corners' are absent
    held = RC.grid_present(g)
    free = RC.interior_nodes(held)
    @test !free[1, 1] && !free[end, end] && free[18, 13] && count(free) < length(free)
    @test held == VgtkCore.node_mask(VgtkCore.model(g), VgtkCore.imaged_region(g))
    # the lens holding them evaluates every imaged pixel bit for bit, and its record stores the present nodes alone
    gm = RC.VgtkCore.with_present(g, held)
    @test all(p -> !inside(region, p) || RC.chief_ray(gm, SVector(p...)) == RC.chief_ray(g, SVector(p...)), raster)
    path_m = tempname() * ".json"; RC.save_grid(path_m, gm)
    back = RC.load_grid(path_m)
    D(c) = VgtkCore.directions(VgtkCore.model(c))
    @test VgtkCore.present_nodes(VgtkCore.model(back)) == held && D(back)[held] == D(g)[held] && VgtkCore.lens_coverage(back) == circle
    @test RC.grid_present(back) == held && RC.grid_present(back; nodes = :all) == held     # a held lens: never an absent node back
    # the masks: the pixels whose centres lie in the region
    m = VgtkCore.pixel_mask(region, W, H)
    @test m[H ÷ 2, W ÷ 2] && !m[1, 1] && isapprox(count(m), π * circle.a * circle.b; rtol = 0.02)
    # the focus nodes' support: imaged by the start lens AND inside the region
    sup = RC._focus_support(fish, region)
    @test sup(SVector(260.0, 174.0)) && !sup(SVector(520.0, 348.0)) && !sup(SVector(0.0, 348.0))
    # the focus TPS nodes: the 5 × 5 grid over the region's bounding box — the circle's box [110, 410] × [24, 324] here — kept where the
    # start lens images them inside the region and within ρ ≤ focus.lens_coverage_scale of the circle (2026-11-15): the inner 3 × 3 (membership alone
    # keeps 13 of 25, the rim's four side midpoints included); the sensor's own grid without a circle
    nb = RC.focus_grid_nodes(region)
    @test length(nb) == 25 && nb[1] == SVector(110.0, 24.0) && nb[25] == SVector(410.0, 324.0)
    @test count(sup, nb) == 13 && all(p -> sup(p) == inside(region, (p[1], p[2])), nb)
    fs = RC.focus_start_nodes(VgtkCore.with_lens_coverage(fish, circle), RC.PipelineOptions().focus)
    @test fs.nodes == nb[[7, 8, 9, 12, 13, 14, 17, 18, 19]] && fs.nodes_box == nb && all(sup, fs.nodes)
    @test RC.focus_grid_nodes(VgtkCore.imaged_region(fish)) == VgtkCore.grid_nodes(5, VgtkCore.Rect2(VgtkCore.imaged_region(fish)))
    @test_throws ArgumentError RC._check_focus_nodes(nb[1:3], fish)
    @test RC._check_focus_nodes(nb[sup.(nb)], fish) == nb[sup.(nb)]
    # the radial vignetting's radius: the region's radius seen from the sensor centre (the circle's, centred there)
    radius(cov) = VgtkCore.farthest_radius(cov isa VgtkCore.Ellipse ? VgtkCore.Rect(0, 0, W, H) ∩ cov : VgtkCore.Rect(0, 0, W, H), SVector(W / 2, H / 2))
    @test radius(circle) ≈ circle.a rtol = 1e-6
    # off centre: the farthest imaged point (on the arc inside the sensor), never extrapolated
    off = VgtkCore.Ellipse(VgtkCore.Point2(300.0, 174.0), 150.0, 150.0, 0.0)
    @test radius(off) ≈ 190.0 rtol = 1e-6
    # the tensor surface: its nodes whose knot span meets the region are the parameters, the others dependent (Core's
    # BoundaryRule through the surface's support trait); V on every imaged pixel reads interior nodes only
    vig = TensorProductVignetting((7, 5), VgtkCore.Rect(0, 0, W, H); degree = 2)
    vrule = RC.vignetting_rule(vig, region)
    @test vrule.interior == VgtkCore.node_mask(vig, region) && !vrule.interior[1, 1] && vrule.interior[4, 3]
    @test RC.vignetting_rule(vig, nothing) === nothing && RC.vignetting_rule(RadialVignetting(8, SVector(260.0, 174.0), 150.0), region) === nothing
    P = [1.0 - 0.3 * ((i - 4)^2 + (j - 3)^2) / 10 for i in 1:7, j in 1:5]
    full = with_control_points(vig, VgtkCore.extend_controls(vrule, P, ones(7, 5)))
    junk = with_control_points(vig, [vrule.interior[i, j] ? P[i, j] : 7.0 for i in 1:7, j in 1:5])
    @test all(p -> !inside(region, p) || full(SVector(p...)) ≈ junk(SVector(p...)), raster)

    # no image circle (the lens lights the whole sensor): the region is the sensor's rect, every grid node interior
    pin = cam(VgtkCore.Pinhole(), 416.0, (260.0, 174.0), W, H; f_y = 417.6)
    rp = VgtkCore.imaged_region(pin)
    @test !(VgtkCore.lens_coverage(pin) isa VgtkCore.Ellipse) && all(inside(rp, p) for p in raster)
    gp = grid_model(pin, 10, 10)
    @test VgtkCore.lens_coverage(gp) == VgtkCore.Rect2{Float64}(0, 0, W, H) && all(RC.grid_present(gp))
    @test all(VgtkCore.pixel_mask(rp, W, H)) && radius(VgtkCore.lens_coverage(pin)) ≈ hypot(W, H) / 2
    @test all(RC._focus_support(pin, rp), VgtkCore.grid_nodes(5, VgtkCore.Rect(0, 0, W, H)))

    # where the run's coverage comes from (owner ruling 2026-11-15): the capture set's image-region step measures it on the set's own
    # photographs and writes its record; the run reads the record its options name, or states the sensor. Photographs of a bright disc
    # on black, each darkened over a different part of the rim, written as 16-bit PNGs.
    cover(x, y, ρ) = mean((x - 1 + (i - 0.5) / 4 - 260.0)^2 + (y - 1 + (j - 0.5) / 4 - 174.0)^2 <= ρ^2 for i in 1:4, j in 1:4)
    disc = [cover(u, v, 150.0) for v in 1:H, u in 1:W]
    rng = Xoshiro(3); dir = mktempdir()
    for k in 1:4
        φ = π / 2 * (k - 1); notch = [hypot(u - 0.5 - 260 - 150cos(φ), v - 0.5 - 174 - 150sin(φ)) < 60 for v in 1:H, u in 1:W]
        PNGFiles.save(joinpath(dir, "V$k.png"), Gray{N0f16}.(clamp.(0.7 .* disc .* .!notch .+ 0.005 .* randn(rng, H, W), 0, 1)))
    end
    frames = RC.photo_files(dir; join = true)
    m = RC.measure_image_region(frames; image = "luma", outlier_halfwidth_px = 0.1 * hypot(W, H), seed = 1)
    e = m.lens_coverage
    @test hypot(e.center[1] - 260, e.center[2] - 174) < 0.2 && 0 < e.a - 150 < 1.5   # ≈ 0.6 px outward: the lit disc's OUTER edge (Features 0.9.2), within its radial sampling (≤ 1 px)
    @test m.extent == (W, H) && m.frames == ["V1.png", "V2.png", "V3.png", "V4.png"] && m.inliers <= m.candidates
    @test RC.measure_image_region(frames; image = "luma", outlier_halfwidth_px = 0.1 * hypot(W, H), seed = 1).lens_coverage == e   # seeded: the same region
    # the record (IO's region body) round-trips the region bit for bit; a run's options name it, and the run reads it
    rec = joinpath(mktempdir(), "image_region.json"); RC.save_image_region(rec, m; provenance = (; source = "the test's frames"))
    r = RC.load_image_region(rec)
    @test r.lens_coverage == e && r.extent == (W, H) && r.frames == m.frames
    @test RC.run_lens_coverage(PipelineOptions(; camera = RC.CameraOptions(; lens_coverage = rec)), W, H) == e
    # no fallback: a record of other photographs, a missing record, a record of another kind — each refused by name
    @test occursin("photographs, the run's are", thrown_message(() -> RC.run_lens_coverage(PipelineOptions(; camera = RC.CameraOptions(; lens_coverage = rec)), W + 2, H)))
    @test occursin("no image-region record", thrown_message(() -> RC.run_lens_coverage(PipelineOptions(; camera = RC.CameraOptions(; lens_coverage = rec * ".missing")), W, H)))
    RC.save_initialization_region(dirname(rec), circle)
    @test occursin("rc/image_region/v1.0", thrown_message(() -> RC.load_image_region(joinpath(dirname(rec), RC.INITIALIZATION_REGION_FILE))))
    # opt-in: a set stating the sensor is the sensor's rectangle, nothing measured — never a grid start's stored region (no fallback)
    @test RC.run_lens_coverage(PipelineOptions(), W, H) === nothing && RC.PipelineOptions().camera.lens_coverage == "sensor"
    @test VgtkCore.lens_coverage(VgtkCore.with_lens_coverage(g, RC.run_lens_coverage(PipelineOptions(), W, H))) == VgtkCore.Rect2{Float64}(0, 0, W, H)
    # the detector is the step's alone: measured on the device (GPU first), never inside a run
    calls = Ref(0)
    stub(rng, fs; outlier_halfwidth, device) = (calls[] += 1; nothing)
    @test occursin("no image region", thrown_message(() -> RC.measure_image_region(frames; image = "luma", outlier_halfwidth_px = 1.0, seed = 1, detect = stub))) && calls[] == 1
    # the geometry image is the config's statement (owner 2026-11-16), required, of the one vocabulary; decoded frames are "luma",
    # RAW frames "raw_green" (the bilinear green on the raw grid), never the other; the green cells are retired for geometry
    @test occursin("image: required", thrown_message(() -> RC.check_geometry_image(Dict{String, Any}(), "[t]")))
    @test occursin("retired", thrown_message(() -> RC.check_geometry_image(Dict("image" => "green_cells"), "[t]")))
    @test occursin("one of", thrown_message(() -> RC.check_geometry_image(Dict("image" => "bin2"), "[t]")))
    @test occursin("reads RAW frames", thrown_message(() -> RC.measure_image_region(frames; image = "raw_green", outlier_halfwidth_px = 1.0, seed = 1, detect = stub)))
    @test_throws UndefKeywordError RC.measure_image_region(frames; outlier_halfwidth_px = 1.0, seed = 1, detect = stub)
    @test !isdefined(RC, :run_image_circle) && !occursin("detect_image_circle(", read(joinpath(pkgdir(RC), "src", "start", "prepare.jl"), String))
    # no circle is typed in, and none predicted from the lens
    @test !hasfield(RC.CameraOptions, :image_circle) && !isdefined(RC, :start_region) && !isdefined(RC, :_source_domain)
end
