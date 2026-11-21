# The lens coverage of a capture set as ONE keyed step (src/image_region.jl, scripts/image_regions.jl; owner ruling 2026-11-15): its
# key (the frames' sha256, the detector's settings, the code — never the config or the region's name), the opt-in (a set stating the
# sensor has no step and runs no detector), the step's record read by every consumer (the refinement's start, the front ends' bounded
# cameras, the chrome evaluation's radial frame), each refusing without it, and the configs of record: every set
# states its coverage, no coverage number is typed in, every region is measured by the driver's config.

@testitem "image region step: keyed by its frames, opt-in, read by every consumer, refused without its record" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    import PNGFiles
    using ColorTypes: Gray, N0f16
    using StableRNGs: StableRNG
    using StaticArrays: SVector
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitEstimation                     # detect_image_circle's fit: Features' extension on Estimation
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("parametric_starts.jl"))
    include(srcdir("frontends.jl"))
    using VisualGeometryToolkitEval: file_sha256
    using VisualGeometryToolkitCore: thrown_message

    # a synthetic set: photographs of a bright disc on black, each darkened over a part of the rim (16-bit PNG, the renders' kind)
    W, H = 520, 348
    data = mktempdir(); dir = joinpath(data, "set"); mkpath(dir)
    lit(u, v, k) = hypot(u - 260.5, v - 174.5) <= 150 && hypot(u - 260.5 - 150cospi((k - 1) / 2), v - 174.5 - 150sinpi((k - 1) / 2)) >= 60   # the pixel centre (u − ½, v − ½)
    foreach(k -> PNGFiles.save(joinpath(dir, "V$k.png"), Gray{N0f16}.([0.7 * lit(u, v, k) for v in 1:H, u in 1:W])), 1:4)
    frames = Dict{String, Any}("V$k.png" => file_sha256(joinpath(dir, "V$k.png")) for k in 1:4)
    region(; kw...) = merge(Dict{String, Any}("input" => "set", "frames" => frames, "image" => "luma", "outlier_halfwidth_px" => 0.1 * hypot(W, H), "seed" => 1), Dict{String, Any}(String(k) => v for (k, v) in kw))
    cfgof(t; name = "disc") = Dict{String, Any}("experiment" => Dict("name" => "x"),
                                                "inputs" => Dict{String, Any}("set" => Dict{String, Any}("tier" => "rc_synthetic", "path" => "set", "stage" => "detect", "provenance" => "test")),
                                                "image_regions" => Dict{String, Any}(name => t))
    cfg = cfgof(region())
    check_image_regions(cfg, "test")

    @testset "keyed by only what it reads: the frames, the settings, the code" begin
        k = image_region_product(cfg, "disc").key
        @test image_region_product(cfgof(region(); name = "other"), "other").key == k                     # not by the region's name
        @test image_region_product(merge(cfg, Dict("experiment" => Dict("name" => "y"))), "disc").key == k   # nor by the config
        f2 = copy(frames); f2["V1.png"] = "0"^64
        @test image_region_product(cfgof(region(frames = f2)), "disc").key != k                          # a frame's content: another step
        f3 = copy(frames); delete!(f3, "V4.png")
        @test image_region_product(cfgof(region(frames = f3)), "disc").key != k                          # another frame set
        @test image_region_product(cfgof(region(outlier_halfwidth_px = 1.0)), "disc").key != k           # the detector's settings
        @test image_region_product(cfgof(region(seed = 2)), "disc").key != k
        @test image_region_product(cfgof(region(image = "raw_green")), "disc").key != k                  # the geometry image measured on
        @test image_region_product(cfg, "disc").key == Product("image_region", "detect", image_region_product(cfg, "disc").record;
                                                               output = IMAGE_REGION_FILE, code = code_of(IMAGE_REGION_CODE)).key   # its packages' code alone
        @test image_region_product(cfg, "disc").key != Product("image_region", "detect", image_region_product(cfg, "disc").record;
                                                               output = IMAGE_REGION_FILE, code = CODE_VERSION).key                 # not the whole tree's
        @test lens_coverage_key(cfg, "sensor") == "sensor" && lens_coverage_key(cfg, "image_region:disc") == "image_region:" * k
    end

    @testset "the statement: \"sensor\" or a region of the config; a table refused by name" begin
        @test check_lens_coverage(cfg, "sensor", "w") == "sensor" && check_lens_coverage(cfg, "image_region:disc", "w") == "image_region:disc"
        @test occursin("no [image_regions.nope]", thrown_message(() -> check_lens_coverage(cfg, "image_region:nope", "w")))
        @test occursin("\"sensor\"", thrown_message(() -> check_lens_coverage(cfg, "[2128.0, 1424.0, 1680.1]", "w")))   # no circle typed in
        @test occursin("sha256", thrown_message(() -> check_image_regions(cfgof(region(frames = Dict("V1.png" => "abc"))), "t")))
        @test occursin("outlier_halfwidth_px", thrown_message(() -> check_image_regions(cfgof(region(outlier_halfwidth_px = 0.0)), "t")))
        @test occursin("seed", thrown_message(() -> check_image_regions(cfgof(delete!(region(), "seed")), "t")))
        @test occursin("image", thrown_message(() -> check_image_regions(cfgof(delete!(region(), "image")), "t")))   # required: no default
        @test occursin("retired", thrown_message(() -> check_image_regions(cfgof(region(image = "green_cells")), "t")))
        @test occursin("read by no lens_coverage", thrown_message(() -> check_image_regions_read(cfg, "t")))
        @test referenced_image_regions(Dict("cases" => Dict("a" => Dict("pipeline" => Dict("camera" => Dict("lens_coverage" => "image_region:disc")))),
                                            "evaluate" => Dict("lens_coverage" => "sensor"))) == ["disc"]
        @test occursin("retired", thrown_message(() -> check_input_table(Dict("tier" => "rc_baselines", "path" => "vignetting/image_circle/14mm",
                                                                              "stage" => "refine", "provenance" => "p"), "t")))   # the shared PNGs
    end

    withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
        @testset "opt-in: the sensor reads no record and runs no detector" begin
            @test lens_coverage_record(cfg, "sensor"; what = "w") == "sensor" && lens_coverage_path(cfg, "sensor") == "sensor"
            @test lens_coverage_region(cfg, "sensor", W, H; what = "w") == VgtkCore.Rect2{Float64}(0, 0, W, H)
            @test isempty(referenced_image_regions(Dict("cases" => Dict("a" => Dict("pipeline" => Dict("camera" => Dict("lens_coverage" => "sensor")))))))
            pf = profile_frame(VgtkCore.Rect2{Float64}(0, 0, 4256, 2848), 4256, 2848)            # the values typed in before (rmax rounded there)
            @test pf.centre == SVector(2128.0, 1424.0) && isapprox(pf.rmax, 2560.5; atol = 1e-4) && pf.rmax == hypot(2128.0, 1424.0)
        end
        @testset "each consumer produces the record on demand (owner 2026-11-15: an upstream is produced, never refused)" begin
            # each consumer in a results root of its own: the record is missing, the consumer measures it (its step, under its key)
            t = Dict{String, Any}("pipeline" => Dict{String, Any}("camera" => Dict{String, Any}("lens_coverage" => "image_region:disc")))
            @test resolve_pipeline(cfg, t)["camera"]["lens_coverage"] == joinpath(product_dir(image_region_product(cfg, "disc")), IMAGE_REGION_FILE)
            gpu = gpu_listed()                                                                                     # its stage is a GPU's
            for consume in (() -> require_lens_coverage(cfg, t, "case c"),                                          # the refinement's start
                            () -> lens_coverage_region(cfg, "image_region:disc", W, H; what = "[schoeps.s]"))      # the front ends
                withenv("VGTK_RESULTS_ROOT" => mktempdir()) do
                    @test !produced(image_region_product(cfg, "disc"))
                    if gpu                                                                                         # a GPU job: measured
                        consume()
                        @test produced(image_region_product(cfg, "disc")) && isfile(lens_coverage_path(cfg, "image_region:disc"))
                    else                                                                                           # a CPU job: refused, naming the command
                        m = thrown_message(consume)
                        @test occursin("needs a GPU", m) && occursin("image_regions.jl", m) && !produced(image_region_product(cfg, "disc"))
                    end
                end
            end
            withenv("VGTK_RESULTS_ROOT" => mktempdir()) do                                                          # measured once, then loaded by its key
                calls = Ref(0)
                counting(rng, fs; kw...) = (calls[] += 1; VgtkFeatures.detect_image_circle(rng, fs; kw...))
                f = lens_coverage_record(cfg, "image_region:disc"; what = "w", detect = counting, has = true)   # the device list overridden
                @test calls[] == 1 && f == lens_coverage_record(cfg, "image_region:disc"; what = "w", detect = counting, has = true) && calls[] == 1
            end
        end
        @testset "the step: the frames checked, measured (the detector counted), the record read by every consumer" begin
            calls = Ref(0)
            bad = cfgof(region(frames = merge(frames, Dict("V1.png" => "1"^64))))
            @test occursin("the config pins", thrown_message(() -> run_image_region(bad, "disc"; driver = "t", config_path = "t.toml")))
            counting(rng, fs; kw...) = (calls[] += 1; VgtkFeatures.detect_image_circle(rng, fs; kw...))   # Features' detector, counted
            d = run_image_region(cfg, "disc"; driver = "t", config_path = "t.toml", detect = counting)
            @test calls[] == 1 && produced(image_region_product(cfg, "disc"))
            run_image_region(cfg, "disc"; driver = "t", config_path = "t.toml", detect = counting)
            @test calls[] == 1                                                                                   # loaded by its key, not measured again
            r = RadiometricCamera.load_image_region(joinpath(d, IMAGE_REGION_FILE)); e = r.lens_coverage
            @test hypot(e.center[1] - 260, e.center[2] - 174) < 0.5 && 0 < e.a - 150 < 1.5 && r.extent == (W, H) && r.frames == ["V1.png", "V2.png", "V3.png", "V4.png"]
            @test lens_coverage_region(cfg, "image_region:disc", W, H; what = "w") == e                              # the front ends' / chrome's / panoramas'
            @test occursin("photographs, not", thrown_message(() -> lens_coverage_region(cfg, "image_region:disc", W + 2, H; what = "w")))
            @test require_lens_coverage(cfg, Dict("pipeline" => Dict("camera" => Dict("lens_coverage" => "image_region:disc"))), "c") === nothing
            pf = profile_frame(e, W, H)
            @test pf.centre == SVector(e.center...) && isapprox(pf.rmax, max(e.a, e.b); rtol = 1e-6)                           # the chrome radial frame: the rim
            opts = RadiometricCamera.pipeline_options(Dict("camera" => Dict("lens_coverage" => lens_coverage_path(cfg, "image_region:disc"))))
            @test RadiometricCamera.run_lens_coverage(opts, W, H) == e                                                # the refinement's start (bin/refine.jl)
        end
    end
end

@testitem "image region step: the configs of record state every set's coverage, the driver measures every region" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    import VisualGeometryToolkitCore as VgtkCore
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    c = load_image_regions_config(projectdir("configs", "experiments", "image_regions.toml"))
    listed = Set(c["measure"]["configs"])
    excluded = Set(normpath(f) for (_, e) in excluded_sets() for f in get(e, "configs", String[]))
    for f in sort(readdir(projectdir("configs", "experiments")))
        endswith(f, ".toml") || continue
        rel = joinpath("configs", "experiments", f); t = TOML.parsefile(projectdir(rel))
        rel in excluded && continue
        # every config with image regions is the driver's; no shared image-circle photographs, no typed-in circle or ellipse
        haskey(t, "image_regions") && @test rel in listed
        @test !any(v -> v isa AbstractDict && startswith(get(v, "path", ""), RETIRED_IMAGE_CIRCLE_INPUTS), values(get(t, "inputs", Dict())))
        src = read(projectdir(rel), String)
        @test !occursin(r"(?m)^\s*(image_circle_frames|image_circle_outlier_halfwidth_px|circle|centre|rmax)\s*=\s*[\[0-9\"]", src)
        haskey(t, "image_regions") && check_image_regions_read(t, rel)                       # every region of record read by a statement
        # every run of a case states its lens coverage
        for tb in ("cases", "init"), (n, ct) in get(t, tb, Dict())
            haskey(ct, "pipeline") && @test haskey(get(ct["pipeline"], "camera", Dict()), "lens_coverage")
        end
    end
    regions = image_region_products(c)
    names = sort([r.name for r in values(regions)])
    # the classification (the coordinator's, job 18294): measured — the 14 mm's f/4 housing session, the fisheye sets; the sessions'
    # shared frames are one step each (the AprilGrid set and its chrome; the fisheye 30 cm front end and its chrome)
    # (and the board-warping session's two fisheye sets, each on its own frames)
    # and the Double Sphere renders of the detector bias (their image disc)
    @test names == ["fx3_14mm_f4_20261028", "fx3_fisheye_aprilgrid_10cm", "fx3_fisheye_aprilgrid_20261119", "fx3_fisheye_panorama_20261029",
                    "fx3_fisheye_siemens_20261028", "fx3_fisheye_siemens_20261119", "fx3_fisheye_siemens_30cm", "mitsuba_fisheye_disc"]
    vg = TOML.parsefile(projectdir("configs", "experiments", "radiometric_refinement_vignetting.toml"))
    pv = TOML.parsefile(projectdir("configs", "experiments", "panorama_vignetting_fx3_14mm.toml"))                       # the 14 mm sweep: the session's step
    @test image_region_product(pv, "fx3_14mm_f4_20261028").key == image_region_product(vg, "fx3_14mm_f4_20261028").key
    @test TOML.parsefile(projectdir("configs", "experiments", "panorama_vignetting_fx3_tamron.toml"))["lens_coverage"] == "sensor"
    fe = TOML.parsefile(projectdir("configs", "experiments", "frontends.toml"))
    regions = Dict("fisheye30cm_siemens" => "fx3_fisheye_siemens_30cm", "vig14mm_f4_siemens" => "fx3_14mm_f4_20261028",
                    "vigfisheye_f4_siemens" => "fx3_fisheye_siemens_20261028",             # the fisheye sets and the 14 mm housing: measured
                    "bw_fisheye_siemens" => "fx3_fisheye_siemens_20261119")
    @test all(((s, t),) -> t["lens_coverage"] == (haskey(regions, s) ? "image_region:" * regions[s] : "sensor"), fe["cases"])
    @test image_region_product(fe, "fx3_14mm_f4_20261028").key == image_region_product(vg, "fx3_14mm_f4_20261028").key       # the vignetting sets' own steps
    @test image_region_product(fe, "fx3_fisheye_siemens_20261028").key == image_region_product(vg, "fx3_fisheye_siemens_20261028").key
    ch = TOML.parsefile(projectdir("configs", "experiments", "chrome_evaluation_fx3_fisheye_30cm_si.toml"))
    @test image_region_product(fe, "fx3_fisheye_siemens_30cm").key == image_region_product(ch, "fx3_fisheye_siemens_30cm").key   # one step, the session's
    bw = TOML.parsefile(projectdir("configs", "experiments", "radiometric_refinement_board_warping.toml"))
    @test image_region_product(fe, "fx3_fisheye_siemens_20261119").key == image_region_product(bw, "fx3_fisheye_siemens_20261119").key   # the board-warping session's Siemens set: one step
    @test !isfile(projectdir("scripts", "make_panorama_cameras.jl")) && !isfile(projectdir("src", "panorama_cameras.jl")) &&
          !isfile(projectdir("configs", "inputs", "panorama_cameras.toml"))                                     # the typed-in circles' maker: retired
end
