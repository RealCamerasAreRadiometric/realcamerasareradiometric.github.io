# The exact start's lattice padded to the grid route's patch multiple (RC 0.11.2): the exact resample's node counts m·cells + 3 are
# set by the data, the grid route tiles with fixed `GRID_PATCH` blocks, so the start extends its lattice at the right and bottom edges
# (Core's `extend_lattice`) to the next multiple per axis. The padded nodes lie outside the region: the camera on every region pixel is
# the unpadded one bit for bit, no region pixel's window reads a padded node, the padded nodes are absent (never a parameter) but the
# ring a region whose edge is a knot touches, the free nodes of an exact start are the resample's, the region's folds are unchanged,
# and a start whose counts are not multiples runs through build_problem_grid and the LM.

@testitem "lattice padding: the exact start's node counts reach the patch multiple, the region unchanged" setup = [RCFixtures, SchoepsSource] begin
    using RadiometricCamera: GRID_PATCH, coverage_region, exact_start, grid_dims, grid_present, pad_lattice, start_grid
    using StaticArrays, LinearAlgebra
    using KernelAbstractions: CPU

    W, H = 640, 480
    path = tempname() * ".yaml"; schoeps_yaml(path)
    src = VgtkIO.load(VgtkCore.BoundedCamera, path)
    h = Tuple(Float64.(VgtkCore.lattice_spacing(VgtkCore.model(src))))
    sensor = VgtkCore.Rect(0.0, 0.0, Float64(W), Float64(H))
    circle = VgtkCore.Ellipse(VgtkCore.Point2(320.0, 240.0), 300.0, 260.0, 0.0)

    for (m, coverage) in ((1, nothing), (2, nothing), (4, nothing), (4, circle))
        st = exact_start(subdivided(src, m); coverage)
        r = st.resample
        G = grid_dims(st.camera)
        # the node counts: multiples of the patch, the resample's own counts before, padded at the right and bottom by fewer than a patch
        @test all(G .% GRID_PATCH .== 0)
        G0 = Tuple(r.padding.from)
        @test collect(G) == r.grid_size && collect(G .- G0) == r.padding.nodes && all(0 .<= r.padding.nodes .< GRID_PATCH)
        region = coverage_region(st.camera)
        # the marks: the resample's kept; the padded nodes ABSENT (NaN controls, unmarked): this region lies inside the lattice
        X = reduce(vcat, permutedims.(r.extrapolated))
        P = VgtkCore.present_nodes(VgtkCore.model(st.camera))
        @test size(X) == size(P) == (G[2], G[1]) && !any(X[(G0[2] + 1):end, :]) && !any(X[:, (G0[1] + 1):end])
        @test !any(P[(G0[2] + 1):end, :]) && !any(P[:, (G0[1] + 1):end]) && all(P[1:G0[2], 1:G0[1]])
        # no padded node is free, under either rule of grid.nodes: the free set is the resample's own
        @test !any(grid_present(st.camera; nodes = :interior) .& .!P) && grid_present(st.camera; nodes = :all) == P
        # no region pixel's window reads a padded node: every region pixel is imaged (finite) through absent padded nodes
        px = VgtkCore.pixel_centres(region, 1:W, 1:H)
        @test all(r -> all(isfinite, r), VgtkCore.backproject(VgtkCore.model(st.camera), px; device = CPU()))
        # the region's folds unchanged by the padding (the resample's report), and the padded call's own report recorded
        f = VgtkCore.folded_pixels(st.camera, region)
        @test length(f.folded) == length(r.folds.folded) && f.unimaged == r.folds.unimaged
        r.padding.nodes == [0, 0] ? (@test r.padding.folds === nothing) :
                                    (@test r.padding.folds.samples > 0 && haskey(r.padding.folds, :folded) && haskey(r.padding.folds, :unimaged))
    end
    # the padded camera is the unpadded one bit for bit on every region pixel (backproject and project): `pad_lattice` of a 27 × 23
    # lattice (a parametric start's, over the whole sensor) and of Schöps's own 7 × 6 (over his domain). Both regions' edges are the
    # lattice's (a knot): the first padded ring touches the region (Core's closed footprint) and is present, its basis zero there
    kb = cam(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.0, 0.0), 300.0, (320.0, 240.0), W, H)
    for (c0, region) in ((RC.grid_model(kb, 27, 23), sensor), (src, sensor ∩ VgtkCore.valid_domain(VgtkCore.model(src))))
        p = pad_lattice(c0, region)
        G0 = grid_dims(c0); G = grid_dims(p.camera)
        @test all(G .% GRID_PATCH .== 0) && collect(G .- G0) == p.nodes && any(>(0), p.nodes)
        @test VgtkCore.directions(VgtkCore.model(p.camera))[1:G0[2], 1:G0[1]] == VgtkCore.directions(VgtkCore.model(c0))
        P = VgtkCore.present_nodes(VgtkCore.model(p.camera))
        ring = falses(size(P))                                                              # the first padded column and row
        p.nodes[1] > 0 && (ring[1:G0[2], G0[1] + 1] .= true)
        p.nodes[2] > 0 && (ring[G0[2] + 1, 1:G0[1]] .= true)
        all(>(0), p.nodes) && (ring[G0[2] + 1, G0[1] + 1] = true)
        @test P[1:G0[2], 1:G0[1]] == VgtkCore.present_nodes(VgtkCore.model(c0))
        @test P .& .!((axes(P, 1) .<= G0[2]) .& (axes(P, 2) .<= G0[1])') == ring                      # the ring alone of the padded nodes
        px = VgtkCore.pixel_centres(region, 1:W, 1:H)
        r0 = Array(VgtkCore.backproject(VgtkCore.model(c0), px; device = CPU()))
        @test all(r -> all(isfinite, r), r0) && Array(VgtkCore.backproject(VgtkCore.model(p.camera), px; device = CPU())) == r0
        # no region pixel's window reads a padded node: the ring made absent too, every pixel the same bits
        cut = VgtkCore.with_present(p.camera, P .& .!ring)
        @test Array(VgtkCore.backproject(VgtkCore.model(cut), px; device = CPU())) == r0
        q0 = VgtkCore.project(VgtkCore.model(c0), r0, px; device = CPU()); q1 = VgtkCore.project(VgtkCore.model(p.camera), r0, px; device = CPU())
        @test all(q0.ok) && q1.ok == q0.ok && q1.px == q0.px
        # the free nodes: the unpadded camera's, and of the padded ones the ring alone (zero basis on the region: no data column)
        for rule in (:interior, :all)
            held = grid_present(p.camera; nodes = rule)
            @test held[1:G0[2], 1:G0[1]] == grid_present(c0; nodes = rule) && held .& .!((axes(P, 1) .<= G0[2]) .& (axes(P, 2) .<= G0[1])') == ring
        end
        # the region's folds unchanged by the padding
        f0 = VgtkCore.folded_pixels(c0, region); f1 = VgtkCore.folded_pixels(p.camera, region)
        @test f1.folded == f0.folded && f1.samples == f0.samples && f1.unimaged == f0.unimaged
    end
    @test pad_lattice(RC.grid_model(kb, 15, 10), sensor).nodes == [0, 0]                                     # multiples already: nothing added

    # the resample at his own cell (9 × 8 nodes) is padded to 10 × 10
    @test grid_dims(exact_start(src).camera) == (10, 10)
    # a parametric start's lattice (sized by grid.node_count over the image region, RC 0.11.4): padded to the patch multiple too
    @test all(grid_dims(start_grid(kb, RC.GridOptions(; lattice = RC.NodeCountLattice(), node_count = 150)).camera) .% GRID_PATCH .== 0)
end

@testitem "lattice padding: a start whose node counts are not patch multiples runs through build_problem_grid and the LM" setup = [RCFixtures, SchoepsSource] begin
    using RadiometricCamera: build_problem_grid, compute_mask, exact_start, Focus, FocusSpline, GRID_PATCH, grid_dims, grid_model,
                             grid_present, Material, PrintedTarget, render_view, RenderSettings, run_lm_grid!, Scene, sphere_radius, 𝔟
    using RadiometricCamera, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap

    tex = fixture_board()
    board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    scene = Scene(board, W, H)
    src_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    spline = subdivided(grid_model(src_t, 10, 10), 2)                      # a SplineLens source of 7 × 7 cells split 2 × 2: 17 nodes per axis
    st = exact_start(spline)                                               # his lattice, m = 1: 17 nodes per axis, padded to 20
    @test st.resample.padding.from == [17, 17] && grid_dims(st.camera) == (20, 20)
    glens = VgtkCore.with_present(st.camera, grid_present(st.camera; nodes = :interior))   # the run's free nodes (grid.nodes = "interior")
    focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    vig = RC.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end)
    camera = RadiometricCameraModel(glens, focus, 0.0386, vig)
    b = 𝔟(board)
    a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2)
    light = EnvironmentLight(SphericalHarmonicMap(a, 2), b, sphere_radius(b, [p.t for p in poses]))
    mat = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
    rs = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera, p, scene, rs; δ = 0.02, backend = RC.KernelAbstractions.CPU()) for p in poses]
    targets = [render_view(k, camera, poses[k], mat, light, scene, rs, masks[k]) for k in 1:2]
    s = CalibrationState(scene, rs, camera, copy(poses), mat, light, targets; masks)
    prob, idx = build_problem_grid(s)                                      # the patch default: GRID_PATCH, which divides the padded lattice
    @test GRID_PATCH == (5, 5) && size(idx.patches) == (4, 4)
    out = run_lm_grid!(s; maxiters = 1, fixed = (:focus, :aperture, :vignetting, :lighting))
    @test out.iterations >= 1 && isfinite(out.cost)
end
