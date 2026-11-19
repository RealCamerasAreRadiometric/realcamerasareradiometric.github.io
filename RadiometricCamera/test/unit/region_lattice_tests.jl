# A PARAMETRIC start's lattice sized by the image region and a node count (RC 0.11.4, owner 2026-11-15): square cells over the bounding
# box of the region (sensor ∩ lens coverage), the cell chosen so the nodes the region supports (Core's node_mask) come to ≈
# grid.node_count, the nodes it does not support absent, then padded to the grid route's patch multiple (`pad_lattice`). The retired
# grid.size is refused by name; a Schöps start (a SplineLens source) keeps his lattice, bit for bit.

@testitem "region lattice: a parametric start over an ellipse ≈ N supported nodes, square cells, patch multiples" setup = [RCFixtures] begin
    using RadiometricCamera: GRID_PATCH, coverage_region, grid_dims, region_layout, region_spacing, region_start, start_grid
    using StaticArrays, LinearAlgebra
    using KernelAbstractions: CPU

    W, H = 640, 480
    kb = cam(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.0, 0.0), 300.0, (320.0, 240.0), W, H)
    circle = VgtkCore.Ellipse(VgtkCore.Point2(320.0, 240.0), 200.0, 180.0, 0.0)
    rays(c, px) = Array(VgtkCore.backproject(VgtkCore.model(c), px; device = CPU()))
    for N in (150, 400, 1200)
        st = region_start(kb, N; coverage = circle)
        m = VgtkCore.model(st.camera)
        region = coverage_region(st.camera)
        @test region == VgtkCore.Rect(0.0, 0.0, Float64(W), Float64(H)) ∩ circle
        # ≈ N from below: the cell is the integer whose supported count is the largest at most N
        n(h) = RC._supported_nodes(VgtkCore.model(kb), region, h)
        # node_count is the case's BOUND on its present nodes (owner 2026-11-15): the largest count at most N, the next finer cell's above it
        @test st.supported == n(st.spacing) && st.supported <= N && n(st.spacing - 1) > N
        @test (N - st.supported) / N < 0.15                                    # from below: one integer cell step of a small lattice is ≈ 13 %
        # square cells: one spacing for both axes, the lattice's first node one cell before the box's origin
        @test all(VgtkCore.lattice_spacing(m) .≈ st.spacing)                                  # (to the rounding of K⁻¹ of the box)
        @test collect(VgtkCore.node_position(m, CartesianIndex(1, 1))) ≈ st.origin .- st.spacing
        # over the region's bounding box: the layout's area holds the box and overhangs it by less than one cell per side
        L = region_layout(region, st.spacing)
        A = (L.grid_size .- 3) .* st.spacing
        @test collect(L.origin) == st.origin && all(L.origin .<= (120, 60)) && all(L.origin .+ A .>= (520, 420)) && all(A .- (400, 360) .< 2st.spacing)
        # node counts: multiples of the patch (padded); the present nodes of the unpadded lattice are the supported ones (the padding
        # adds present nodes only where the region's edge is a knot: `pad_lattice`)
        @test all(grid_dims(st.camera) .% GRID_PATCH .== 0) && all(grid_dims(st.camera) .- L.grid_size .< GRID_PATCH)
        P = VgtkCore.present_nodes(m)
        @test count(P[1:L.grid_size[2], 1:L.grid_size[1]]) == st.supported && count(P) >= st.supported
        # the camera images every region pixel, the source's rays to the fit's accuracy
        px = VgtkCore.pixel_centres(region, 1:W, 1:H)
        r1 = rays(st.camera, px); r0 = rays(kb, px)
        @test all(r -> all(isfinite, r), r1) && maximum(acos(clamp(dot(normalize(a), normalize(b)), -1, 1)) for (a, b) in zip(r1, r0)) < 1e-4
    end
    # the cell falls as N grows
    @test region_spacing(VgtkCore.model(kb), coverage_region(VgtkCore.with_lens_coverage(kb, circle)), 1200) <
          region_spacing(VgtkCore.model(kb), coverage_region(VgtkCore.with_lens_coverage(kb, circle)), 150)
    # start_grid of a parametric source: region_start at grid.node_count over the run's coverage, no resample record
    gs = start_grid(kb, RC.GridOptions(; lattice = RC.NodeCountLattice(), node_count = 400); coverage = circle)
    @test gs.resample === nothing && grid_dims(gs.camera) == grid_dims(region_start(kb, 400; coverage = circle).camera)
end

@testitem "region lattice: a sensor region gives a lattice covering the sensor" setup = [RCFixtures] begin
    using RadiometricCamera: GRID_PATCH, coverage_region, grid_dims, region_start
    using KernelAbstractions: CPU

    W, H = 640, 480
    ds = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 300.0, (322.0, 238.0), W, H)
    sensor = VgtkCore.Rect(0.0, 0.0, Float64(W), Float64(H))
    for N in (200, 875)
        st = region_start(ds, N)
        m = VgtkCore.model(st.camera)
        G = grid_dims(st.camera)
        @test coverage_region(st.camera) == sensor ∩ sensor
        @test all(G .% GRID_PATCH .== 0)
        @test all(VgtkCore.lattice_spacing(m) .≈ st.spacing)
        # the lattice's area (its first node + one cell to its last − two cells before the padding) holds the sensor
        lo = VgtkCore.node_position(m, CartesianIndex(1, 1)) .+ st.spacing
        @test all(lo .<= 0) && all(lo .> -st.spacing)
        # every node of the unpadded lattice is supported (a rectangle the lattice covers): the count is its node count
        @test st.supported == prod(RC.region_layout(sensor, st.spacing).grid_size) && abs(st.supported - N) / N < 0.15
        px = VgtkCore.pixel_centres(sensor, 1:W, 1:H)
        @test all(r -> all(isfinite, r), Array(VgtkCore.backproject(m, px; device = CPU())))
    end
    # N = 7875 on the FX3 sensor (configs/default.toml's derivation of the example count): the 41 px cell, 107 × 73 = 7811 nodes, padded to 110 × 75
    fx3 = cam(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), 3000.0, (2128.0, 1424.0), 4256, 2848)
    @test RC.PipelineOptions().grid.node_count === nothing                       # no default: a parametric start states it (lattice = "node_count")
    s = region_start(fx3, 7875)
    @test s.spacing == 41 && s.supported == 107 * 73 && grid_dims(s.camera) == (110, 75)
end

@testitem "region lattice: grid.size is refused, naming grid.lattice and grid.node_count" setup = [RCFixtures] begin
    m = try
        RC.pipeline_options(Dict("grid" => Dict("size" => [105, 75]))); ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("`grid.size` is retired", m) && occursin("grid.node_count", m) && occursin("7875", m)
    @test occursin("grid.lattice", m) && occursin("\"derived\"", m) && occursin("silently", m)   # a Schöps start: never ignored again
    @test !hasfield(RC.GridOptions, :size) && !hasfield(RC.GridOptions, :lattice_spacing)
    @test occursin("grid.node_count", try RC.pipeline_options(Dict("grid" => Dict("lattice" => "node_count", "node_count" => 9))); "" catch e; sprint(showerror, e) end)
end

@testitem "region lattice: a Schöps start is his own lattice (bit for bit), whatever the coverage" setup = [RCFixtures, SchoepsSource] begin
    using RadiometricCamera: exact_start, start_grid
    W, H = 640, 480
    path = tempname() * ".yaml"; schoeps_yaml(path)
    src = VgtkIO.load(VgtkCore.BoundedCamera, path)
    circle = VgtkCore.Ellipse(VgtkCore.Point2(320.0, 240.0), 300.0, 260.0, 0.0)
    for coverage in (nothing, circle)
        a = exact_start(src; coverage)
        b = start_grid(src, RC.GridOptions(); coverage)                          # grid.lattice = "derived": the exact start, m = 1
        mb = VgtkCore.model(b.camera); ma = VgtkCore.model(a.camera)
        @test isequal(VgtkCore.directions(mb), VgtkCore.directions(ma)) && VgtkCore.present_nodes(mb) == VgtkCore.present_nodes(ma)
        @test VgtkCore.calibration_matrix(mb) == VgtkCore.calibration_matrix(ma) && VgtkCore.lattice_spacing(mb) == VgtkCore.lattice_spacing(ma)
        @test isequal(b.resample, a.resample) && b.resample.subdivision == [1, 1]
    end
end

# Core 0.10.10: a lens that is not radially symmetric (Brown–Conrady's tangential terms) has a STAR POLYGON valid domain through each
# azimuth's fold (`PolygonRegion`); a decentred parametric start's lattice over the image region moves it into the lattice's pixel
# frame (`_in_frame`: Core's SimilarityMap action, the one for every region) — release session job 19974: a MethodError there.
@testitem "region lattice: a decentred Brown–Conrady start (a polygon valid domain, Core 0.10.10) through the session's start; its initialization region saved and reloaded" setup = [RCFixtures] begin
    using RadiometricCamera: GRID_PATCH, grid_dims
    W, H = 1280, 960
    bc = cam(VgtkCore.Lens(VgtkCore.Perspective(), VgtkCore.Additive(VgtkCore.Radial((-0.3, 0.04)), VgtkCore.Tangential(1.0e-3, -5.0e-4))), 800.0, (700.0, 430.0), W, H)
    @test VgtkCore.valid_domain(VgtkCore.model(bc)) isa VgtkCore.PolygonRegion        # the polygon, about a decentred principal point
    opts = RC.pipeline_options(Dict("grid" => Dict("lattice" => "node_count", "node_count" => 300)))   # the sensor as the coverage (default)
    st = RC._session_start(opts, bc)
    @test st.resample === nothing && all(grid_dims(st.lens) .% GRID_PATCH .== 0)
    @test count(VgtkCore.present_nodes(VgtkCore.model(st.lens))) <= 300                 # the node count's bound
    mktempdir() do dir
        RC.save_grid(joinpath(dir, "grid.json"), st.lens); RC.save_initialization_region(dir, st.init_region)
        @test RC.stored_initialization_region(joinpath(dir, "grid.json")) == st.init_region   # the region through IO (its polygon kind)
    end
end

@testitem "region lattice: grid.node_count is refused on a Schöps start (derived: his own lattice, never subdivided)" setup = [RCFixtures] begin
    # owner ruling 2026-11-16 (the inapplicable-keys rule): the Schöps route (`schoeps_start` → `exact_start`) reads no node count, so a
    # stated `grid.node_count` under `grid.lattice = "derived"` is an error naming the key and why, never silently ignored
    d = Dict("frontend" => Dict("type" => "schoeps"), "grid" => Dict("lattice" => "derived", "node_count" => 5000))
    m = try RC.pipeline_options(d); "" catch e; sprint(showerror, e) end
    @test occursin("grid.node_count", m) && occursin("Schöps", m)
    @test RC.pipeline_options(Dict("frontend" => Dict("type" => "schoeps"), "grid" => Dict("lattice" => "derived"))).grid.node_count === nothing
    @test RC.inapplicable_reason(RC.pipeline_options(Dict("frontend" => Dict("type" => "schoeps"))), "grid", "node_count") !== nothing
end

