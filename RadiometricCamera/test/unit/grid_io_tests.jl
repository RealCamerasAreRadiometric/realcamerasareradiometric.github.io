# Model I/O and the Schöps imports: the pixel convention, the records' round trips through VisualGeometryToolkitIO (a parametric
# model ↔ a camera record, a grid camera ↔ a camera record), an old grid file read
# through IO's migration, and synthetic Schöps intrinsics0.yaml files (VisualGeometryToolkitIO's readers) resampled into our grid lenses.
@testitem "grid I/O (Schöps import)" setup = [RCFixtures] begin
    using RadiometricCamera: grid_dims, grid_model, initialization_region, load_camera, load_grid, load_lens, load_model,
                             load_parametric_lens, load_start, pixel_of_𝐩, save_grid, save_initialization_region, save_lens,
                             save_run_parameters, schoeps_gridlens, schoeps_start, stored_initialization_region, 𝐩_of_pixel
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    using Random

    @testset "Julia pixel ↔ COLMAP" begin
        @test 𝐩_of_pixel(1, 1) == SVector(0.5, 0.5)                                # colmap centre 0.5 = 1-based pixel 1
        @test pixel_of_𝐩(SVector(320.0, 240.0)) == SVector(320.5, 240.5)
    end

    @testset "the records: camera and grid camera round trips" begin
        W, H = 640, 480; rl = cam(VgtkCore.Division(-0.15, 0.03), 540.0, (326.4, 230.4), W, H; f_y = 530.0)
        ps = [SVector(W * rand(), H * rand()) for _ in 1:30]
        # a parametric model ↔ vgio/cameras/v5: the pixel camera (K in COLMAP px), the lens's own record
        lpath = tempname() * ".json"
        save_lens(lpath, rl; provenance = (; model = "test"))
        l2 = load_parametric_lens(lpath)
        @test VgtkCore.extent(l2) == VgtkCore.Size2(W, H) && has_lens(l2, VgtkCore.Division{2})
        @test maximum(abs, pinhole_θ(l2) - pinhole_θ(rl)) < 1e-12 && VgtkCore.lens(l2) == VgtkCore.lens(rl)
        @test maximum(norm(RC.chief_ray(l2, p) - RC.chief_ray(rl, p)) for p in ps) < 1e-12
        unp, label = load_model("lens:" * lpath, W, H)
        @test unp(SVector(448.0, 288.0)) ≈ normalize(RC.chief_ray(rl, SVector(448.0, 288.0))) && startswith(label, "lens:")
        # a grid camera ↔ vgio/cameras/v5 (its K the initializing camera's own: the gauge, ruling 1; its lens coverage kept)
        circle = VgtkCore.Ellipse(VgtkCore.Point2{Float64}(320.0, 240.0), 300.0, 230.0, 0.1)
        g = grid_model(rl, 35, 25; coverage = circle); path = tempname() * ".json"
        @test g isa VgtkCore.BoundedCamera{<:VgtkCore.CameraModel{<:VgtkCore.AbstractIntrinsics, <:VgtkCore.SplineLens}}
        @test VgtkCore.calibration_matrix(VgtkCore.model(g)) == VgtkCore.calibration_matrix(VgtkCore.model(rl))
        save_grid(path, g; provenance = (; model = "test"))
        g2 = load_grid(path)
        @test VgtkCore.directions(VgtkCore.model(g2)) == VgtkCore.directions(VgtkCore.model(g)) && grid_dims(g2) == (35, 25)
        @test VgtkCore.calibration_matrix(VgtkCore.model(g2)) ≈ VgtkCore.calibration_matrix(VgtkCore.model(g)) && VgtkCore.lens_coverage(g2) == circle
        @test maximum(norm(normalize(RC.chief_ray(g2, p)) - normalize(RC.chief_ray(g, p))) for p in ps) < 1e-12
        unp2, _ = load_model("grid:" * path, W, H)
        @test unp2(SVector(448.0, 288.0)) ≈ normalize(RC.chief_ray(g, SVector(448.0, 288.0)))
        @test VgtkCore.directions(VgtkCore.model(load_lens("grid:" * path))) == VgtkCore.directions(VgtkCore.model(g)) &&
              has_lens(load_lens("lens:" * lpath), VgtkCore.Division{2})
    end

    # ---- synthetic Schöps yaml files: a smooth ray field on the generic_models grid convention --------------------------------
    function schoeps_yaml(path; noncentral::Bool = false, GW = 7, GH = 6, cmin = (0, 0), cmax = (639, 479), size = cmax .+ 1, origin = SVector(0.0, 0.0, 0.0))
        dirs = [normalize(SVector(0.12 * (i - (GW + 1) / 2), 0.12 * (j - (GH + 1) / 2), 1.0)) for i in 1:GW, j in 1:GH]
        flat(v) = join((string(x) for d in vec(v) for x in d), ", ")        # i fastest (column-major), as generic_models stores it
        head = "type: $(noncentral ? "NoncentralGenericModel" : "CentralGenericModel")\nwidth: $(size[1])\nheight: $(size[2])\ngrid_width: $GW\ngrid_height: $GH\n" *
               "calibration_min_x: $(cmin[1])\ncalibration_min_y: $(cmin[2])\ncalibration_max_x: $(cmax[1])\ncalibration_max_y: $(cmax[2])\n"
        body = noncentral ? "point_grid: [$(flat(fill(origin, GW, GH)))]\ndirection_grid: [$(flat(dirs))]\n" : "grid: [$(flat(dirs))]\n"
        write(path, head * body)
        return dirs
    end

    @testset "Schöps central import" begin
        W, H = 640, 480
        path = tempname() * ".yaml"; schoeps_yaml(path)
        m = VgtkIO.load(VgtkCore.BoundedCamera, path)                        # VisualGeometryToolkitIO's reader: a SplineLens camera
        @test m isa VgtkCore.BoundedCamera{<:VgtkCore.CameraModel{<:VgtkCore.AbstractIntrinsics, <:VgtkCore.SplineLens}}
        @test (vd = VgtkCore.valid_domain(VgtkCore.model(m)); isapprox(SVector(vd.origin..., vd.widths...), SVector(0.0, 0.0, 640.0, 480.0); atol = 1e-9)) && size(VgtkCore.directions(VgtkCore.model(m))) == (6, 7)
        unp(x, y) = (d = RC.backproject(m, SVector(Float64(x), Float64(y))); all(isfinite, d) ? SVector{3, Float64}(d) : nothing)   # NaN: not imaged
        @test unp(-1.0, 10.0) === nothing && unp(10.0, 10.0) !== nothing
        @test norm(unp(320.0, 240.0)) ≈ 1
        # the Schöps surface is a cubic B-spline of the nodes: a smooth field, continuous across the grid
        @test norm(unp(100.0, 100.0) - unp(100.5, 100.0)) < 1e-2
        g = schoeps_gridlens(path, W, H)                                     # his 160 px cells, never subdivided (m = 1, RC 0.11.4)
        @test g isa VgtkCore.BoundedCamera{<:VgtkCore.CameraModel{<:VgtkCore.AbstractIntrinsics, <:VgtkCore.SplineLens}} && grid_dims(g) == (10, 10)   # his 7 × 6, padded to the patch multiple (RC 0.11.2)
        # the source turned onto its optical axis (Estimation's fit_camera, RC 0.11.0): K's principal point the axis's pixel (here the
        # radially symmetric source's centre), the rays the source's (below)
        @test SVector(VgtkCore.principal_point(VgtkCore.model(g).intrinsics)...) ≈ SVector(320.0, 240.0) atol = 0.05
        worst = 0.0
        for x in 20.0:60.0:620.0, y in 20.0:60.0:460.0
            d = normalize(RC.chief_ray(g, SVector(x, y)))
            worst = max(worst, acos(clamp(dot(d, unp(x, y)), -1.0, 1.0)))
        end
        @test worst < 1e-4                                                   # the exact resample: his rays, up to the axis turn
        unp2, label = load_model("schoeps:" * path, W, H)
        @test unp2(SVector(320.0, 240.0)) ≈ unp(320.0, 240.0) && unp2(SVector(704.0, 240.0)) === nothing
    end

    @testset "the initialization region: where a start is defined, stored with its run, never the sensor by default" begin
        W, H = 640, 480
        rect(r) = SVector(r.origin..., r.widths...)
        # a source calibrated on a box inside the sensor (Schöps's calibrated area): the re-gridded start spans the sensor, its
        # initialization region is the source's own imaged region, and the start bounded to it images only there
        path = tempname() * ".yaml"; schoeps_yaml(path; cmin = (100, 80), cmax = (539, 399), size = (W, H))
        src = VgtkIO.load(VgtkCore.BoundedCamera, path)
        st = schoeps_start(path, W, H)
        @test st.init_region isa VgtkCore.Rect2 && rect(st.init_region) ≈ rect(VgtkCore.imaged_region(src)) ≈ SVector(100.0, 80.0, 440.0, 320.0)
        @test rect(VgtkCore.Rect2(VgtkCore.imaged_region(st.lens))) ≈ SVector(0.0, 0.0, 640.0, 480.0)          # the lattice: the sensor
        # the initialization's detections: those inside the region (Core's region membership), with or without an image circle
        det = [(SVector(x, y), SVector(0.0, 0.0, 0.0)) for (x, y) in ((330.0, 230.0), (120.0, 230.0), (90.0, 230.0), (520.0, 390.0), (600.0, 450.0))]
        @test [first(d) for d in RC.in_initialization(det, st.init_region)] == [SVector(330.0, 230.0), SVector(120.0, 230.0), SVector(520.0, 390.0)]
        @test RC.in_initialization(det, nothing) == det                                                   # a parametric start: all
        circ = VgtkCore.Ellipse(VgtkCore.Point2(330.0, 230.0), 200.0, 150.0, 0.0)
        both = circ ∩ st.init_region                                                                      # a Core region value
        for (p, _) in det
            @test (p ∈ both) == (p ∈ circ && p ∈ st.init_region)
        end
        # a full-sensor source: unchanged
        pf = tempname() * ".yaml"; schoeps_yaml(pf)
        sf = schoeps_start(pf, W, H)
        @test rect(sf.init_region) ≈ SVector(0.0, 0.0, 640.0, 480.0)
        @test rect(sf.init_region) ≈ rect(VgtkCore.Rect2(VgtkCore.imaged_region(sf.lens)))                # the same as the lattice's
        # stored beside the grid (rc/initialization_region/v1.0, IO's Rect2 body) and restored with a reloaded grid start
        dir = mktempdir()
        save_grid(joinpath(dir, "grid.json"), st.lens); save_initialization_region(dir, st.init_region)
        @test stored_initialization_region(joinpath(dir, "grid.json")) == st.init_region
        ld = load_start("grid:" * joinpath(dir, "grid.json"))
        @test ld.init_region == st.init_region && grid_dims(ld.lens) == grid_dims(st.lens)
        # any of Core's regions is recorded (IO 0.10's one region body): a run whose initialization region is an intersection (the
        # sensor ∩ an image circle) reloads its own grid as a start, bounded to that region
        d3 = mktempdir(); ri = VgtkCore.RegionIntersection(st.init_region, circ)
        save_grid(joinpath(d3, "grid.json"), st.lens); save_initialization_region(d3, ri)
        @test load_start("grid:" * joinpath(d3, "grid.json")).init_region == ri
        # a grid with no region beside it (a run from a parametric start records none; a run's parameters.json is not the start's
        # record): refused, never the sensor
        d2 = mktempdir()
        save_grid(joinpath(d2, "grid.json"), st.lens)
        save_run_parameters(joinpath(d2, "parameters.json"), (; frontend = "schoeps"))
        @test_throws "no initialization_region.json beside it" load_start("grid:" * joinpath(d2, "grid.json"))
        # its regeneration, named for where the grid came from (review round 3, finding 8): a refinement's io.out holds its
        # pipeline_config.toml; a saved initialization's directory does not — re-run the start that wrote it is re-run
        @test_throws "re-run the start that wrote it" load_start("grid:" * joinpath(d2, "grid.json"))
        # the options beside the grid are named only when they are its writer's (bin/refine.jl writes them into its own io.out): another
        # run's, in a shared directory, or an unreadable file is never named (RC 0.9.0 review, finding 7)
        # (also: a relative io.out — its writer's directory is unknown — and a malformed file, `io` not a table: unreadable, never a
        # MethodError while the refusal is built; review findings 3 and 7)
        for other_cfg in ("", "[io]\nout = \"$(mktempdir())\"\n", "[io]\nsave_init = \"$(joinpath(d2, "state.json"))\"\n",
                          "[io]\nout = \".\"\n", "io = 3\n", "[io]\nout = 3\n")
            write(joinpath(d2, "pipeline_config.toml"), other_cfg)
            m = VgtkCore.thrown_message(() -> load_start("grid:" * joinpath(d2, "grid.json")))
            @test occursin("re-run the start that wrote it", m) && !occursin("--config=", m)
        end
        link = joinpath(mktempdir(), "run"); symlink(d2, link)
        for own in (d2, d2 * "/", joinpath(d2, ".", ""), link)           # its own io.out, however spelled (a symlink to it: realpath)
            write(joinpath(d2, "pipeline_config.toml"), "[io]\nout = \"$own\"\n")
            @test_throws "bin/refine.jl --config=$(joinpath(d2, "pipeline_config.toml"))" load_start("grid:" * joinpath(d2, "grid.json"))
        end
        # the writer of those options, RC's one (`write_pipeline_config`, bin/refine.jl's): io.out absolute, from the writer's working
        # directory, so its own run's records name it (review finding 3: a relative io.out was resolved against the wrong directory)
        rel = mktempdir()
        cd(rel) do
            mkpath("out")
            f = RC.write_pipeline_config("out", RC.pipeline_options(Dict("io" => Dict("out" => "out"))))
            @test RC.pipeline_options(f).io.out == abspath("out") && RC._writer_config(joinpath(rel, "out", "grid.json")) == abspath(f)
        end
        # every grid the pipeline writes is a valid start (review finding 1: a saved initialization and a crashed run's checkpoint
        # held grid.json / grid_ckpt.json without their region): the pipeline's one grid writer, `_save_run_grid`, writes the region
        # beside the grid; a run from a parametric start (`nothing`) writes none
        d4 = mktempdir()
        RC._save_run_grid(joinpath(d4, "grid_ckpt.json"), st.lens, ri; provenance = (; iteration = 5))
        @test load_start("grid:" * joinpath(d4, "grid_ckpt.json")).init_region == ri
        d5 = mktempdir()
        save_initialization_region(d5, ri)                          # another run's region, left in the directory (review round 2, finding 6)
        RC._save_run_grid(joinpath(d5, "grid.json"), st.lens, nothing; provenance = (;))
        @test !isfile(joinpath(d5, "initialization_region.json"))   # removed, never reused
        # a relative path (io.save_init = "state.json"): the grid's own directory, resolved once, never another directory's region
        # (review round 3, finding 1)
        other = mktempdir(); save_initialization_region(other, ri)
        cd(other) do
            mkpath("sub"); save_initialization_region("sub", ri)
            RC._save_run_grid(joinpath("sub", "grid.json"), st.lens, nothing; provenance = (;))
            @test isfile("initialization_region.json") && !isfile(joinpath("sub", "initialization_region.json"))
            RC._save_run_grid("grid.json", st.lens, ri; provenance = (;))
            @test load_start("grid:" * abspath("grid.json")).init_region == ri
        end
        @test_throws "no initialization_region.json" load_start("grid:" * joinpath(d5, "grid.json"))
        # … and it is the pipeline's only grid writer: the checkpoints (calibration/calibrate.jl), the result and its start (io/run_outputs.jl), the start record (start/start_record.jl)
        psrc = join((read(joinpath(pkgdir(RadiometricCamera), "src", f), String) for f in ("io/run_outputs.jl", "calibration/calibrate.jl", "calibration/problem_spec.jl")), "\n")
        @test count("save_grid(", psrc) == 1 && count("_save_run_camera(", psrc) == 4 && !occursin("save_lens(joinpath(opts.io.out, \"grid", psrc)   # the helper's own call; the run's writes through the route
        ssrc = read(joinpath(pkgdir(RadiometricCamera), "src", "start", "start_record.jl"), String)
        @test !occursin("save_grid(", ssrc) && !occursin("save_lens(", ssrc) && count("_save_run_camera(", ssrc) == 1   # the start record's two cameras, one call
        # a parametric start is defined wherever its own valid domain says: no stored region
        lp = tempname() * ".json"; save_lens(lp, cam(VgtkCore.Division(-0.15, 0.03), 540.0, (326.4, 230.4), W, H))
        @test load_start("lens:" * lp).init_region === nothing
    end

    @testset "a Schöps run's non-central model is refused: every camera is central" begin
        W, H = 640, 480
        path = tempname() * ".yaml"; schoeps_yaml(path; noncentral = true, origin = SVector(0.01, -0.02, 0.05))
        @test_throws Exception schoeps_gridlens(path, W, H)            # no non-central start (the model is removed)
        @test_throws Exception load_camera("noncentral:" * path)               # no such model spec
        @test !isdefined(RC, :noncentral_gridlens) && !isdefined(RC, :ApertureSpline) && !isdefined(RC, :NoncentralLens)
        # a non-central RECORD (IO reads the YAML when told its length unit, and writes Core's non-central camera as a camera record):
        # refused by name where a record becomes this package's lens
        nc = RC.VgtkIO.load(VgtkCore.BoundedCamera, path; file_unit = VgtkCore.m)
        dir = mktempdir(); rec = joinpath(dir, "camera.json"); RC.VgtkIO.save(rec, Dict(:cam => nc))
        @test occursin("load_grid: $rec holds a non-central model", VgtkCore.thrown_message(() -> load_grid(rec)))
        @test occursin("holds a non-central model", VgtkCore.thrown_message(() -> load_model("schoeps:" * dir, W, H)))   # a run's imported camera.json
    end
end
