# The exact start of a SplineLens source (RC 0.11.0: Estimation's fit_camera(SplineLens, source, sensor ∩ coverage); never subdivided, RC 0.11.4):
# the start covers the sensor ∩ its lens coverage (the set-level coverage assertion), the source's present nodes are kept bit for bit
# (through the axis turn), the subdivision reaches the target cell, the record of the resample round-trips through the start record's
# reader; and the fold policy on output — a calibrated camera that still folds is flagged non-converged.
# A synthetic Schöps central-generic model (a 640 × 480 camera) calibrated on a box inside the sensor (his calibrated area), 7 × 6
# nodes: the source of the exact start's items (this file's and lattice_padding_tests.jl's).
@testsetup module SchoepsSource
    using StaticArrays, LinearAlgebra
    function schoeps_yaml(path; W = 640, H = 480, GW = 7, GH = 6, cmin = (100, 80), cmax = (539, 399))
        dirs = [normalize(SVector(0.12 * (i - (GW + 1) / 2), 0.12 * (j - (GH + 1) / 2), 1.0)) for i in 1:GW, j in 1:GH]
        flat(v) = join((string(x) for d in vec(v) for x in d), ", ")
        write(path, "type: CentralGenericModel\nwidth: $W\nheight: $H\ngrid_width: $GW\ngrid_height: $GH\n" *
                    "calibration_min_x: $(cmin[1])\ncalibration_min_y: $(cmin[2])\ncalibration_max_x: $(cmax[1])\ncalibration_max_y: $(cmax[2])\n" *
                    "grid: [$(flat(dirs))]\n")
    end
    # a source on a finer lattice: his cell split m × m by Estimation's exact resample (its `lattice_spacing`, over the sensor ∩ his
    # coverage) — the exact start itself never subdivides (m = 1, RC 0.11.4)
    import RadiometricCamera
    const Core_ = RadiometricCamera.VgtkCore
    subdivided(src, m; region = RadiometricCamera.coverage_region(src)) = Core_.with_model(src, Core_.estimate(RadiometricCamera.VgtkEstimation.fit_camera(Core_.SplineLens, src, region;
        lattice_spacing = Float64(Core_.lattice_spacing(Core_.model(src))[1]) / m * Core_.px, tolerance = Inf, forward = false)))
    export schoeps_yaml, subdivided
end

@testitem "exact start: coverage, bit-exact nodes, no subdivision (m = 1), fold policy" setup = [RCFixtures, SchoepsSource] begin
    using RadiometricCamera: calibrated_folds, check_coverage, coverage_region, exact_start, fold_termination, grid_dims, start_grid
    using StaticArrays, LinearAlgebra
    using KernelAbstractions: CPU

    W, H = 640, 480
    path = tempname() * ".yaml"; schoeps_yaml(path)
    src = VgtkIO.load(VgtkCore.BoundedCamera, path)
    sensor = VgtkCore.Rect(0.0, 0.0, Float64(W), Float64(H))
    h = Tuple(Float64.(VgtkCore.lattice_spacing(VgtkCore.model(src))))         # his own cell (px)
    @test !(sensor ⊆ VgtkCore.valid_domain(VgtkCore.model(src)))               # he does not cover the sensor

    # 1. the start covers the sensor ∩ the coverage (set-level), every pixel imaged — with the sensor and with an image circle
    st = exact_start(src)
    @test coverage_region(st.camera) == sensor ∩ VgtkCore.lens_coverage(src)
    @test sensor ⊆ VgtkCore.valid_domain(VgtkCore.model(st.camera))
    @test check_coverage(st.camera, sensor; who = "test") == sensor
    px = VgtkCore.pixel_centres(sensor, 1:W, 1:H)
    @test all(r -> all(isfinite, r), VgtkCore.backproject(VgtkCore.model(st.camera), px; device = CPU()))
    circle = VgtkCore.Ellipse(VgtkCore.Point2(320.0, 240.0), 300.0, 260.0, 0.0)
    sc = exact_start(src; coverage = circle)
    @test VgtkCore.lens_coverage(sc.camera) == circle && (sensor ∩ circle) ⊆ VgtkCore.valid_domain(VgtkCore.model(sc.camera))
    # the assertion refuses a camera that does not cover: the source itself over the sensor
    @test_throws VgtkCore.RegionBeyondSupport check_coverage(src, sensor; who = "test")

    # 2. his present nodes bit for bit: at subdivision 1 × 1 the start's control at his node (offset by the extension) is the axis
    # turn R of his — Core's exact axis_gauged, c ↦ R·c — with the R the record holds
    r = st.resample
    @test r.subdivision == [1, 1] && r.lattice_spacing ≈ collect(h)
    R = SMatrix{3, 3, Float64}(permutedims(reduce(hcat, r.axis_rotation)))
    D0 = VgtkCore.directions(VgtkCore.model(src)); D = VgtkCore.directions(VgtkCore.model(st.camera)); o = r.extension_offset
    @test all(D[o[1] + i, o[2] + j] == R * D0[i, j] for i in axes(D0, 1), j in axes(D0, 2))
    @test !any(r.extrapolated[o[1] + i][o[2] + j] for i in axes(D0, 1), j in axes(D0, 2))               # his nodes: not marked
    @test count(Iterators.flatten(r.extrapolated)) == prod(r.padding.from) - length(D0)                     # every new node of the resample: marked
    @test r.extrapolated_fraction > 0 && r.max_px < 1e-6                                                  # his rays to rounding where he images

    # 3. never subdivided (owner ruling 2026-11-15, grid.lattice = "derived"): his cell, m = 1 — through start_grid by the source's TYPE
    # and the lattice's mode; a finer source keeps ITS cell (the start is the source's own lattice)
    gs = start_grid(src, RC.GridOptions())
    @test gs.resample.subdivision == [1, 1] && gs.resample.lattice_spacing ≈ collect(h) && grid_dims(gs.camera) == Tuple(gs.resample.grid_size)
    s2 = exact_start(subdivided(src, 2)).resample
    @test s2.subdivision == [1, 1] && s2.lattice_spacing ≈ [h[1] / 2, h[2] / 2]
    # a target of present nodes (owner 2026-11-15, 23:25): at his own count m = 1; at ≈ 4× it m = 2, exact (his rays to rounding); the
    # target bounds the present nodes, and a target below his own count is refused
    p1 = RC.exact_subdivision(VgtkCore.model(src), RC.coverage_region(src), nothing)
    @test p1.m == 1 && start_grid(src, RC.GridOptions(; node_count = p1.present)).resample.subdivision == [1, 1]
    g4 = start_grid(src, RC.GridOptions(; node_count = 4 * p1.present))
    @test g4.resample.subdivision == [2, 2] && g4.resample.max_px < 1e-6
    @test count(VgtkCore.node_mask(VgtkCore.model(g4.camera), RC.coverage_region(g4.camera))) <= 4 * p1.present
    m0 = try start_grid(src, RC.GridOptions(; node_count = p1.present - 1)); "" catch e; e.msg end
    @test occursin("grid.node_count", m0) && occursin("present nodes", m0)
    # a SplineLens source under lattice = "node_count" is refused (its own lattice; never a second one fitted over it)
    m1 = try start_grid(src, RC.GridOptions(; lattice = RC.NodeCountLattice(), node_count = 150)); "" catch e; e.msg end
    @test occursin("grid.lattice = \"derived\"", m1)
    # a parametric source's lattice is fitted over the image region at grid.node_count (no resample: region_lattice_tests.jl)
    kb = cam(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.0, 0.0), 300.0, (320.0, 240.0), W, H)
    gp = start_grid(kb, RC.GridOptions(; lattice = RC.NodeCountLattice(), node_count = 150))
    @test gp.resample === nothing && all(grid_dims(gp.camera) .% RC.GRID_PATCH .== 0)

    # 4. the fold policy on output: an unfolded calibrated camera keeps the LM's record; one that folds is no calibration
    f0 = calibrated_folds(st.camera)
    @test isempty(f0.folded) && f0.samples > 0
    @test fold_termination(VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease, f0) == (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    m = VgtkCore.model(st.camera); C = copy(VgtkCore.directions(m))
    gy, gx = size(C) .÷ 2
    C[gy, gx] = C[gy, gx] + 3.0 * (C[gy, gx + 1] - C[gy, gx - 1])                                          # one node pushed past its neighbour
    folded = VgtkCore.with_model(st.camera, VgtkCore.with_directions(m, C))
    f1 = calibrated_folds(folded)
    @test !isempty(f1.folded)
    code, rule = fold_termination(VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease, f1)
    @test (code, rule) == (VgtkCore.ReturnCode.Failure, VgtkCore.ShapeGate) && !VgtkCore.successful_retcode(code)
    @test calibrated_folds(kb) === nothing                                                                # a parametric camera: no lattice
end

# RC 0.11.3: a housing-like image circle clipped by the sensor — the sensor reaches past the start's lattice on one axis, the circle on
# the other, while their intersection lies inside it (the 14 mm housing cases: RegionBeyondSupport at the exact start). Core 0.10.8's
# `RegionIntersection ⊆ Rect` held only when one part fits alone (RC 0.11.3 tested the intersection's box); Core 0.10.9 tests the
# intersection itself, and RC 0.11.4 hands it the region.
@testitem "exact start: a coverage circle clipped by the sensor, inside the lattice only as their intersection" setup = [RCFixtures, SchoepsSource] begin
    using RadiometricCamera: check_coverage, coverage_region, exact_start
    using KernelAbstractions: CPU
    path = tempname() * ".yaml"; schoeps_yaml(path)
    src = VgtkIO.load(VgtkCore.BoundedCamera, path)
    h = Float64(VgtkCore.lattice_spacing(VgtkCore.model(src))[1])
    housing = VgtkCore.Ellipse(VgtkCore.Point2(320.0, 240.0), 170.0, 300.0, 0.0)      # inside the sensor in x, past it in y
    for m in (1, 2, 4)
        sub = m == 1 ? src : subdivided(src, m; region = RC.coverage_region(VgtkCore.with_lens_coverage(src, housing)))   # his lattice split over that region alone
        st = exact_start(sub; coverage = housing)
        vd = VgtkCore.valid_domain(VgtkCore.model(st.camera))
        sensor = VgtkCore.Rect(0.0, 0.0, 640.0, 480.0)
        @test !(sensor ⊆ vd) && !(housing ⊆ vd)                                       # neither part lies in the lattice alone
        region = coverage_region(st.camera)
        @test check_coverage(st.camera, region; who = "test", device = CPU()) === region
    end
    # a region that does reach past the lattice is still refused
    @test_throws VgtkCore.RegionBeyondSupport check_coverage(src, VgtkCore.Rect(0.0, 0.0, 640.0, 480.0) ∩ housing; who = "test")
end

@testitem "exact start: a Schöps (central-generic) start computed on the device is the host's" tags = [:gpu] setup = [RCFixtures, RCGPU, SchoepsSource] begin
    # The start of a Schöps run's central-generic model (`schoeps_start` → `exact_start`: Estimation's exact resample of his lattice over
    # the sensor ∩ his coverage) computed on the device, against the host's: the same lattice, region and record of the resample, the
    # resample's own bound (the items above: < 1e-6 px from his rays where he images) and the two starts' rays over the sensor.
    using RadiometricCamera: coverage_region, grid_dims, schoeps_start
    using StaticArrays, LinearAlgebra
    using KernelAbstractions: CPU
    using CUDA: CUDABackend
    require_gpu()
    W, H = 640, 480
    path = tempname() * ".yaml"; schoeps_yaml(path)
    host = schoeps_start(path, W, H; device = CPU())
    dev = schoeps_start(path, W, H; device = CUDABackend())
    @test grid_dims(dev.lens) == grid_dims(host.lens) && dev.init_region == host.init_region
    @test coverage_region(dev.lens) == coverage_region(host.lens)
    @test dev.resample.subdivision == host.resample.subdivision
    @test host.resample.max_px < 1e-6
    @test dev.resample.max_px < 1e-6                                           # the device's resample, by the host tests' bound
    sensor = VgtkCore.Rect(0.0, 0.0, Float64(W), Float64(H))
    px = VgtkCore.pixel_centres(sensor, 1:4:W, 1:4:H)
    rh = VgtkCore.backproject(VgtkCore.model(host.lens), px; device = CPU())
    rd = VgtkCore.backproject(VgtkCore.model(dev.lens), px; device = CPU())   # the device's start, its rays evaluated on the host
    @test all(r -> all(isfinite, r), rd)
    fpx = RC.fpx_of(host.lens, W, H)                                            # px per radian: the rays' difference as pixels
    @test fpx * maximum(norm(normalize(a) - normalize(b)) for (a, b) in zip(rh, rd)) < 1e-6
    rdev = Array(VgtkCore.backproject(VgtkCore.model(dev.lens), px; device = CUDABackend()))   # and evaluated on the device
    @test fpx * maximum(norm(normalize(SVector{3, Float64}(a)) - normalize(b)) for (a, b) in zip(rdev, rh)) < 1e-3
end
