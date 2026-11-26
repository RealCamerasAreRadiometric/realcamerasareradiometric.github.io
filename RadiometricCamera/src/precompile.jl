# The precompile workload (PrecompileTools): tiny but real instances of the package's hot paths, run once while the package image is
# built, so that their native code is cached in it and a fresh process (a test worker, a pipeline run) does not compile it again.
# The suite's items spent 89 % of their time compiling (a release gate), most of it in these paths. CPU only, no data files, no
# output: a synthetic texture and two views of the 24 × 24 test scene. The types are the ones the tests and the pipeline use
# (Float64, a 3 × 3 focus spline, a 4 × 4 vignetting surface, lighting of order 2), since compiled code is cached per argument type.
#
# No NLLSsolver solve runs here: NLLSsolver's `optimize!` evaluates into its own module (`getiterator`), which a precompiling
# package may not do. The workload builds each route's problem and evaluates its cost blocks' residuals and Jacobians instead (the
# LM's flags as an `Int`, the tests' as a static number), which is where the compilation is; the LM loop around them is compiled
# on first use.

@setup_workload begin
    W = H = 24
    tex = [0.5 + 0.4 * sin(0.7i + 1.3c) * cos(0.5j) for i in 1:32, j in 1:32, c in 1:3]
    poses = [EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                          SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k); check = false) for k in 1:2]
    pixel_camera(lens) = VgtkCore.BoundedCamera(CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3}(30.912, 0, 0, 0, 30.912, 0, 12.24, 11.52, 1))), lens),
                                                VgtkCore.Size2(W, H))
    vignetting() = VrtkCore.with_control_points(VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2),
                                                [1.0 - 0.1 * (i + j) / 8 for i in 1:4, j in 1:4])
    # the residual and the Jacobian of a problem's first cost block, with every variable free
    function first_block(prob, blocks)
        r = first(blocks)
        vars = NLLSsolver.getvars(r, prob.variables)
        flags = 2^length(vars) - 1
        NLLSsolver.computeresidual(r, vars...)
        NLLSsolver.computeresjac(flags, r, vars...)                 # the LM's flags
        NLLSsolver.computeresjac(static(flags), r, vars...)         # the tests'
        return nothing
    end
    @compile_workload begin
        board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
        focus_t = Focus(FocusSpline(grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
        b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
        a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2)
        light = EnvironmentLight(SphericalHarmonicMap(a, 2), b, r_SH)
        mat = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
        st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
        # the refinement problem of the scene seen through `lens`, its targets rendered at the truth
        function problem(lens, board_ = board, st_ = st; kw...)
            camera = RadiometricCameraModel(lens, focus_t, 0.0386, vignetting())
            cpu = KernelAbstractions.CPU()                                       # the workload's CPU data, chosen (no default's warning)
            masks = [compute_mask(camera, p, board_, st_; δ = 0.02, backend = cpu) for p in poses]
            targets = [render_view(k, camera, poses[k], mat, light, board_, st_, masks[k]; backend = cpu) for k in 1:2]
            return calibration_problem(board_, st_, camera, poses, with_μ(mat, 0.5), light, targets; masks, backend = cpu, kw...)
        end
        damping = RadiometricRefinement(PipelineOptions()).optimizer.damping   # the algorithm's default
        per_view(p; kw...) = ((prob, _) = build_problem(p.f, p.u0; damping, kw...); first_block(prob, get(prob.problem.costs, PerViewResidual{9})))
        per_tile(p; kw...) = ((prob, _) = build_problem_grid(p.f, p.u0; damping, patch = GRID_PATCH, kw...); first_block(prob, tile_costs(prob)))

        # the closed-form radiometry and the parametric (per-view) blocks on Core's lenses: the double sphere (a free board
        # warping), Kannala–Brandt, OpenCV's 14 coefficients, the division lens and a fixed lens
        ds = pixel_camera(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)))
        p = problem(ds)
        parts = all_parts(p.f, p.u0)
        VgtkCore.solve(VgtkCore.EstimationProblem(RadiometryBlocks, (p.f, p.u0, parts)), RadiometryBlockDescent())
        VgtkCore.solve(VgtkCore.EstimationProblem(RadiometryBlocks, (p.f, p.u0, parts)), JointRadiometry(; damping))
        cost(p.f, p.u0); render_all(p.f, p.u0)
        per_view(p)
        per_view(problem(ds); height_degree = 1)
        per_view(problem(pixel_camera(VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0))))
        per_view(problem(pixel_camera(VgtkIO.opencv_lens(zeros(14)))))
        per_view(problem(pixel_camera(VgtkIO.opencv_lens(zeros(14)))); fixed = (:cx, :cy, :tx, :ty, :k2, :k3, :d, :p, :sx, :sy))
        per_view(problem(pixel_camera(VgtkCore.Division(0.144, -0.293))))
        per_view(problem(pixel_camera(VgtkCore.KannalaBrandt(-0.01, 0.002, 0.0, 0.0))); fixed = (:f, :aspect, :cx, :cy, :k))

        # the grid lens (Core's central generic model): evaluate, project, back-project, and its per-tile blocks (with the warping)
        g = VgtkCore.solve(VgtkCore.EstimationProblem(SplineLens, ds), GridStart(; lattice = ImageLattice(10, 10))).u
        𝐩 = SVector(10.0, 13.0)
        d = chief_ray(g, 𝐩); project(g, d); backproject(g, 𝐩); in_domain(g, 𝐩)
        grid_angular_error(g, ds)
        pg = problem(g)
        per_tile(pg)
        per_tile(pg; height_degree = 2)

        # the pipeline's options: the defaults, an override, the record's round trip through TOML
        alg, stops = read_calibration(Dict{String, Any}("config_version" => 2); set = ["optimizer.max_iter=2"])
        TOML.print(IOBuffer(), config_dict(alg, stops))
        read_calibration(config_dict(alg, stops))
        pipeline_options(config_dict(alg.refinement.options))
        _drain_device_pool!()          # the evaluations' pooled workers: a package image cannot hold a task
    end
end
