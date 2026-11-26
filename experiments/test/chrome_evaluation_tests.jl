# The chrome evaluation driver (scripts/chrome_evaluation.jl) on its test configs (test/fixtures/thorlabs/: the HX400V prototype
# chrome_evaluation.toml and the FX3 green-photosite decode of 2026-11-02, retired as experiments and kept for the
# driver's generic stages; the paper's configs are configs/experiments/chrome_evaluation_fx3_*.toml): the config loads with every detect_foerstner keyword stated, its inputs are the tier
# rc_thorlabs, a missing or unknown key is refused, the run key ignores the exclusion reasons, the photo list must account for every
# file, the detections CSV holds what detect_foerstner returned, the index stage's settings are checked the same way, its registration
# recovers a synthetic grid seen through the camera of record (up to the grid's 8 symmetries, distractors left out) by RANSAC scored
# in board space (the Hungarian with its dustbin at half a pitch, MSAC), its local optimization on the scorer's matches, frozen, and
# the pixel refine on the final matches; it labels a board whose model error exceeds half a pitch in pixels but not on the board
# and DSC06962's real detections as the gated growth did, refuses what does not register, its outputs read back through VisualGeometryToolkitIO, the evaluation's motion fit starts from those poses and
# writes the motion variant alone, and --plan lists both stages. The run itself needs the
# photographs (tier rc_thorlabs); it is not this suite (REPRODUCE.md).
@testitem "chrome_evaluation" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm, normalize, cross, dot
    using Random: randperm

    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: HuberShape, GaussianShape
    using VisualGeometryToolkitCore: Irls, Triggs
    using VisualGeometryToolkitEstimation: LocalOptimization, Ransac, CountedLO
    using VisualGeometryToolkitCore: StudentTUniformScoring
    import VisualGeometryToolkitEval.Reproduce as Reproduce
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))                         # a refinement case's product (refinement_product), the evaluate stage's model
    include(srcdir("chrome_evaluation.jl"))
    include(srcdir("figures.jl"))
    include(srcdir("chrome_evaluation_paper.jl"))                        # the paper stages: a refinement's evaluation into its table

    path = projectdir("test", "fixtures", "thorlabs", "chrome_evaluation.toml")
    cfg = load_chrome_evaluation_config(path)

    @testset "the config of record" begin
        @test cfg["experiment"]["name"] == "chrome_evaluation"
        @test all(k -> haskey(cfg["detect"], k), CHROME_DETECT_KEYS)
        # the detector's keywords are keys: detect_sfop's significance test and precision test, and the seeded detect_foerstner's
        # derivative; `noise` (the propagation of the image noise) is not set: the config of record keeps the reference's
        # a-posteriori covariance; detect_sfop's scale-space schedule is its default (not a setting of record)
        sfop = Base.kwarg_decl(only(methods(parentmodule(detect_sfop)._detect_sfop)))   # every detect_sfop method's keywords (Features 0.9.6: a matrix, a LinearRaw, a RawMosaic)
        @test issubset([:noise_sigma, :significance_level, :precision_threshold, :polarity], sfop)
        @test issubset([:derivative, :derivative_sigma, :significance_level, :polarity], Base.kwarg_decl(only(methods(parentmodule(detect_foerstner)._detect_foerstner))))
        @test Set(DETECT_KEYWORDS) == Set(["derivative", "derivative_sigma", "precision_threshold", "significance_level", "polarity"])
        @test DETECT_LUMA_KEYS == ("noise_sigma",)                                   # a luma set's grey-level noise; a raw_green set refuses it
        @test all(t -> t["tier"] == "rc_thorlabs", values(cfg["inputs"]))
        @test cfg["inputs"][cfg["detect"]["photos"]]["stage"] == "detect"
        s = detect_settings(cfg, path)
        @test s.T === Float32 && s.window.derivative === :gaussian2d && s.sfop.noise_sigma == 2.0
        @test s.sfop.polarity === :light && !haskey(s.window, :polarity)          # the backlit dots: SFOP's gate on the seeds, once
        @test s.sensor_characterisation == ""                                # decoded photographs: no sensor model (the photo kind decides)
        # the 17 photographs DSC01289 … DSC01305, each run or excluded
        @test sort([s.photographs; collect(keys(s.excluded))]) == ["DSC0$(n).JPG" for n in 1289:1305]
    end

    @testset "a config refuses what is not its own" begin
        dir = mktempdir()
        good = TOML.parsefile(path)
        function refused(edit!)
            c = deepcopy(good); edit!(c)
            p = joinpath(dir, "chrome_evaluation.toml")
            open(io -> TOML.print(io, c), p, "w")
            return try load_chrome_evaluation_config(p); nothing catch err; err end
        end
        @test refused(identity) === nothing
        for k in CHROME_DETECT_KEYS
            e = refused(c -> delete!(c["detect"], k)); @test e isa ArgumentError && occursin(k, e.msg)
        end
        e = refused(c -> c["detect"]["sigma"] = 1.0);                        @test e isa ArgumentError && occursin("sigma", e.msg)
        e = refused(c -> c["detect"]["precision"] = "Float16");              @test e isa ArgumentError && occursin("Float16", e.msg)
        e = refused(c -> c["detect"]["device"] = "metal");                   @test e isa ArgumentError && occursin("metal", e.msg)
        e = refused(c -> c["detect"]["polarity"] = "bright");                @test e isa ArgumentError && occursin("polarity", e.msg)
        @test refused(c -> c["detect"]["device"] = "gpu") === nothing       # the GPU (the configs of record's)
        e = refused(c -> c["detect"]["device"] = "cuda");                    @test e isa ArgumentError && occursin("renamed \"gpu\"", e.msg)
        e = refused(c -> c["detect"]["photos"] = "target");                  @test e isa ArgumentError && occursin("target", e.msg)
        e = refused(c -> c["detect"]["sensor_characterisation"] = "sensor/fx3.json"); @test e isa ArgumentError && occursin("input of stage detect", e.msg)   # an input id, or empty
        e = refused(c -> c["detect"]["sensor_characterisation"] = "none"); @test e isa ArgumentError && occursin("retired", e.msg)   # RC 0.11.2: "none" retired
        e = refused(c -> push!(c["detect"]["photographs"], "DSC01302.JPG"));  @test e isa ArgumentError && occursin("DSC01302", e.msg)
        e = refused(c -> c["detect"]["integration_sigma"] = Dict("DSC01289.JPG" => 4.0)); @test e isa ArgumentError && occursin("integration_sigma", e.msg)   # retired: SFOP's scale
        e = refused(c -> c["inputs"]["photos"]["tier"] = "rc_raw");          @test e isa ArgumentError && occursin("rc_raw", e.msg)
        # the run key: an exclusion reason or an input's provenance never re-keys the run; a setting does
        k(c) = detect_product(c).key
        c2 = deepcopy(good); c2["detect"]["excluded"]["DSC01302.JPG"] = "edited"; c2["inputs"]["photos"]["provenance"] = "edited"
        @test k(c2) == k(good)
        c2 = deepcopy(good); c2["detect"]["precision_threshold"] = 0.25
        @test k(c2) != k(good)
        c2 = deepcopy(good); c2["detect"]["noise_sigma"] = 1.0
        @test k(c2) != k(good)                                                      # a luma set's noise_sigma keys it: it is read
        e = refused(c -> delete!(c["detect"], "noise_sigma"));               @test e isa ArgumentError && occursin("noise_sigma", e.msg)   # required for luma
    end

    @testset "the photo list accounts for every file" begin
        s = detect_settings(cfg, path)
        dir = mktempdir()
        foreach(n -> write(joinpath(dir, n), ""), ["DSC0$(n).JPG" for n in 1289:1305])
        @test check_photo_list(s, dir) == s.photographs
        write(joinpath(dir, "DSC01306.JPG"), "")
        @test_throws ArgumentError check_photo_list(s, dir)
        rm(joinpath(dir, "DSC01306.JPG")); rm(joinpath(dir, "DSC01305.JPG"))
        @test_throws ArgumentError check_photo_list(s, dir)
    end

    @testset "detections: IO's detection codec, read back as written" begin
        s = detect_settings(cfg, path)
        img = [Float32((c - 80.5)^2 + (r - 84.5)^2 < 6.25) for r in 1:168, c in 1:160]   # a dot at pixel (84.5, 80.5): COLMAP (80, 84); SFOP's 3 octaves
        points = detect_photo(img, s)
        @test label_counts(points) == (1, 0, 0)
        f = write_detections(joinpath(mktempdir(), "dot.json"), "dot.png", points)
        @test VgtkIO.read_record(f).schema == "vgio/detection/v1.0" && keys(VgtkIO.load(Dict{String, Vector{VgtkCore.AbstractDetection}}, f)) == Set(["dot.png"])
        r = only(read_detections(f))                                          # IO's detection codec: an Attributed uncertain PointFrame
        d = only(points)
        c = Tuple(VgtkCore.centroid(VgtkCore.geom(r.value)))
        @test c == Tuple(VgtkCore.centroid(d.value)) && round.(c; digits = 1) == (80.0, 84.0)   # bit for bit
        @test cov(r.value) == cov(d.value)
        @test String(r.attributes.label) == "circular" && Tuple(r.attributes.window) == Tuple(d.attributes.window) && r.attributes.F == d.attributes.F &&
              r.attributes.covariance_status == d.attributes.covariance_status
        @test VgtkCore.geom(only(read_detections(f, "circular"))) == SVector(c...) && isempty(read_detections(f, "corner"))
        @test photograph_size(img) == (160, 168)
    end

    # A synthetic chrome photograph on the raw's grid: dots (radius 7 raw px) on a dark ground, each photosite the scene's area mean
    # over its own square (4 × 4 samples), an RGGB mosaic, decoded by stage prepare's Julia path (bin2: write_decoded,
    # write_conversion) into a set's <set>_linear/.
    function synthetic_green_set(; H = 288, W = 320, r = 7.0)                              # cells 144 × 160: SFOP's 3 octaves
        dots = [SVector(x, y) for x in (104.3, 144.0, 181.6) for y in (105.7, 150.2)]      # raw COLMAP px
        lit(x, y) = any(c -> hypot(x - c[1], y - c[2]) < r, dots) ? 0.8 : 0.05
        area(x0, y0) = sum(lit(x0 + (a - 0.5) / 4, y0 + (b - 0.5) / 4) for a in 1:4, b in 1:4) / 16   # photosite (x0, y0) … (x0 + 1, y0 + 1)
        scene = [area(j - 1.0, i - 1.0) for i in 1:H, j in 1:W]                           # the scene each photosite records, linear
        raw = round.(UInt16, 512 .+ scene .* (16383 - 512))
        md = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing, iso = nothing,
              model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
        m = VgtkCore.RawMosaic(raw, UInt8[0 1; 3 2], (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (H, W), md)
        L = VgtkCore.LinearRaw(m; mode = raw_mode("bin2"), gain = 1.0)
        root = mktempdir()
        dirs = (linear = mkpath(joinpath(root, "s_linear")), srgb = nothing)
        meta = Dict{String, Any}("p.png" => write_decoded(L, "p.png", dirs))
        write_conversion(dirs.linear, "bin2", [frame_record(L)], meta)
        return (; dots, scene, linear = dirs.linear, H, W, mosaic = m)
    end
    # the circular points of a detections CSV, raw COLMAP px
    circular(f) = VgtkCore.geom.(read_detections(f, "circular"))
    nearest(p, qs) = minimum(q -> norm(p - q), qs)

    @testset "raw_green: the RAW mosaic detected on Features' bilinear green, in the raw's pixels (Features 0.9.6)" begin
        # the stage's path for a raw_green set: the mosaic and its frame's sensor model (here none: the a-posteriori covariance),
        # Features' detect_sfop(m::RawMosaic) seeds and the seeded detect_foerstner(m, seeds) — no map onto the photograph, the rows are
        # its raw COLMAP pixels; the reference is the same scene detected on the raw's own grid (every photosite)
        s = detect_settings(cfg, path)
        g = synthetic_green_set()
        img = (g.mosaic, VgtkCore.SensorModel())
        @test photograph_size(img) == (g.W, g.H)
        raw = circular(write_detections(joinpath(mktempdir(), "raw.json"), "p.png", detect_photo(img, s)))
        @test length(raw) == length(g.dots)
        @test all(c -> nearest(c, raw) < 0.1, g.dots)                           # the true dot centres, raw px
        extent(ps) = maximum(q -> q[1], ps) - minimum(q -> q[1], ps)
        @test isapprox(extent(raw) / extent(g.dots), 1; atol = 1e-3)            # the raw's scale (never the cells' half)
    end

    @testset "a prepare set whose record states no frame is refused, never guessed" begin
        g = synthetic_green_set()
        rec = VgtkIO.read_record(joinpath(g.linear, "conversion.json"))
        conv(d) = (q = joinpath(mktempdir(), "s_linear"); mkpath(q); VgtkIO.write_record(joinpath(q, "conversion.json"), d); q)
        plain(r) = Dict{String, Any}(String(k) => v for (k, v) in pairs(r) if !(String(k) in ("generator", "units", "conventions")))   # the envelope is the writer's
        nofm = delete!(plain(rec), "frame_map")                               # a decode made before prepare recorded the frame
        e = try prepared_frame_map(conv(nofm)); nothing catch err; err end
        @test e isa ArgumentError && occursin("frame_map", e.msg) && occursin("prepare", e.msg)
        other = merge(plain(rec), Dict("frame_map" => Dict("from" => "PixelFrame{:colmap}", "to" => "PixelFrame{:colmap}", "scale" => 1.0, "translation" => [0.0, 0.0])))
        e = try prepared_frame_map(conv(other)); nothing catch err; err end
        @test e isa ArgumentError && occursin("frame_map", e.msg)              # another frame than Features' for the mode
        full = merge(plain(rec), Dict("mode" => "fullres"))
        @test_throws ArgumentError prepared_frame_map(conv(full))              # fullres (a demosaic) is retired, refused by name
        @test_throws ArgumentError decoded_frame("fullres")
        @test decoded_frame("bin2").record != decoded_frame("raw_green").record                   # the cell grid, the raw grid
    end

    @testset "the FX3 fixture: a decoding of stage prepare" begin
        p3 = projectdir("test", "fixtures", "thorlabs", "chrome_evaluation_fx3_14mm_20261102.toml")
        c3 = load_chrome_evaluation_config(p3)
        @test collect(keys(c3["prepare"])) == ["thorlabs"]
        t = c3["prepare"]["thorlabs"]
        @test t["mode"] == "bin2" && prepare_srgb(t, "[prepare.thorlabs]") === false && t["gain"] == 1.0
        @test all(i -> i["tier"] == "rc_fx3_14mm", values(c3["inputs"])) && c3["inputs"][t["raw"]]["stage"] == "prepare"
        @test !haskey(c3, "detect")                                                   # its detect stage is the 0.10 sets' (configs/experiments)
        cb = merge(c3, Dict("prepare" => Dict("thorlabs" => merge(t, Dict("mode" => "raw_green")))))
        @test product_dir(prepare_product(c3, "thorlabs")) != product_dir(prepare_product(cb, "thorlabs"))   # the decoding keys the run
        dir = mktempdir()
        good = TOML.parsefile(p3)
        function refused3(edit!)
            c = deepcopy(good); edit!(c)
            q = joinpath(dir, "chrome_evaluation_fx3_14mm_20261102.toml")
            open(io -> TOML.print(io, c), q, "w")
            return try load_chrome_evaluation_config(q); nothing catch err; err end
        end
        @test refused3(identity) === nothing
        for k in PREPARE_KEYS
            e = refused3(c -> delete!(c["prepare"]["thorlabs"], k)); @test e isa ArgumentError && occursin(k, e.msg)
        end
        e = refused3(c -> c["prepare"]["thorlabs"]["mode"] = "demosaic");  @test e isa ArgumentError && occursin("demosaic", e.msg)
        e = refused3(c -> (c["prepare"]["thorlabs"]["mode"] = "raw_green"; c["prepare"]["thorlabs"]["srgb"] = "on"));   @test e isa ArgumentError && occursin("sRGB", e.msg)   # raw_green has no colour
        e = refused3(c -> c["prepare"]["thorlabs"]["gain"] = 0);            @test e isa ArgumentError && occursin("gain", e.msg)
        e = refused3(c -> c["inputs"]["raw"]["stage"] = "detect");          @test e isa ArgumentError && occursin("prepare", e.msg)
        # detect on the prepare set: image raw_green exactly for a photosites set (its RAW photographs linked with their EXIF, nothing
        # decoded: detect reads the mosaics); a raw_green detect names a sensor characterisation, an input of stage detect (the zoom
        # set's record of record), and refuses noise_sigma by name (SFOP reads the sensor model's noise law)
        fx3 = TOML.parsefile(projectdir("configs", "experiments", "chrome_evaluation_fx3_zoom48_f10.toml"))["inputs"]["fx3_sensor_characterisation"]
        withdetect(img; sc = img == "raw_green" ? "fx3_sensor_characterisation" : "", noise = img != "raw_green") = c -> begin
            c["inputs"]["fx3_sensor_characterisation"] = deepcopy(fx3)
            d = merge(deepcopy(TOML.parsefile(path)["detect"]), Dict("photos" => "thorlabs", "image" => img, "photographs" => ["DSC05464.ARW"],
                                                                    "excluded" => Dict(), "sensor_characterisation" => sc))
            noise || delete!(d, "noise_sigma")
            c["detect"] = d
            img == "raw_green" && (c["prepare"]["thorlabs"] = Dict{String, Any}("raw" => c["prepare"]["thorlabs"]["raw"], "mode" => "photosites"))
        end
        @test refused3(withdetect("raw_green")) === nothing
        e = refused3(withdetect("raw_green"; sc = ""));                     @test e isa ArgumentError && occursin("sensor model", e.msg)   # never un-modelled RAW
        e = refused3(withdetect("raw_green"; noise = true))
        @test e isa ArgumentError && occursin("noise_sigma", e.msg) && occursin("noise law", e.msg)       # inert under the sensor model: refused by name
        e = refused3(withdetect("luma"));                                   @test e isa ArgumentError && occursin("raw_green", e.msg)
        e = refused3(withdetect("green_cells"));                            @test e isa ArgumentError && occursin("\"green_cells\" is retired", e.msg) && occursin("raw_green", e.msg)   # retired by name, its replacement named
        e = refused3(c -> (withdetect("raw_green")(c); c["prepare"]["thorlabs"] = deepcopy(good["prepare"]["thorlabs"])))   # a decoding
        @test e isa ArgumentError && occursin("raw_green", e.msg) && occursin("photosites", e.msg)
        e = refused3(c -> c["index"] = TOML.parsefile(path)["index"]);      @test e isa ArgumentError && occursin("needs [detect]", e.msg)
        cg = deepcopy(good); withdetect("raw_green")(cg)
        s3 = detect_settings(cg)
        @test s3.prepared && s3.image == "raw_green"
        @test haskey(detect_product(cg).record, "upstream") && !haskey(detect_product(cg).record, "inputs")   # keyed by the prepare product, not a raw input
        @test !haskey(s3.sfop, :noise_sigma) && !haskey(detect_product(cg).record["detect"], "noise_sigma")   # no inert key in the key
        # the prepare product of a raw_green set is the photograph list and their EXIF (photosites.json), no decode: its output
        @test prepare_product(cg, "thorlabs").output == "photosites.json" && !haskey(prepare_product(cg, "thorlabs").record, "frame_map")
        md = (; datetime_original = "2026:11:02 10:00:00", subsec = nothing, exposure_s = 0.002, fnumber = 8.0, focal_mm = 14.0, iso = 100, model = "ILME-FX3")
        @test raw_exif(md, "/r/frames/a.ARW", "/r") == Dict{String, Any}("datetime_original" => "2026:11:02 10:00:00", "subsec" => "", "exposure_s" => 0.002,
                                                                       "fnumber" => 8.0, "focal_mm" => 14.0, "iso" => 100, "model" => "ILME-FX3", "raw" => "frames/a.ARW")
        # a raw_green detect reads the RAW mosaics in raw pixels: no decoded image's frame keys its product (the prepare product's does)
        @test !haskey(detect_product(cg).record, "image_frame") && !haskey(detect_product(cfg).record, "image_frame")   # every detection is its photograph's own pixels
    end

    @testset "index: the config of record" begin
        s = index_settings(cfg, path)
        @test all(k -> haskey(cfg["index"], k), CHROME_INDEX_KEYS)
        @test cfg["inputs"][s.target]["stage"] == "index" && cfg["inputs"][s.camera]["stage"] == "index"
        # MSAC, its threshold not the table's (half the record's pitch, the dustbin: index_scoring)
        @test s.scoring == Dict{String, Any}("type" => "msac") && s.seed_tolerance == 5.0 && s.rng_seed == 1 && s.ransac == RansacConfig(0.999, 500, 20)
        @test s.photos == sort(collect(setdiff(cfg["detect"]["photographs"], keys(cfg["index"]["excluded"]))))
        @test length(s.photos) == 13
        # the local optimization inside RANSAC: plain least squares (Irls at the Gaussian shape) on the scorer's matches; the pixel
        # refine after it: a refine's least squares (nothing)
        @test s.local_optimization_strategy == Irls{GaussianShape}(; max_iter = 10)
        @test s.refine_strategy === nothing
    end

    @testset "index: a config refuses what is not its own" begin
        dir = mktempdir()
        good = TOML.parsefile(path)
        function refused(edit!)
            c = deepcopy(good); edit!(c)
            p = joinpath(dir, "chrome_evaluation.toml")
            open(io -> TOML.print(io, c), p, "w")
            return try load_chrome_evaluation_config(p); nothing catch err; err end
        end
        for k in CHROME_INDEX_KEYS
            e = refused(c -> delete!(c["index"], k)); @test e isa ArgumentError && occursin(k, e.msg)
        end
        for k in RANSAC_KEYS
            e = refused(c -> delete!(c["index"]["ransac"], k)); @test e isa ArgumentError && occursin(k, e.msg)
        end
        # the lattice growth's gate is retired (owner 2026-11-14), refused by name
        e = refused(c -> c["index"]["assignment_level"] = 0.999)
        @test e isa ArgumentError && occursin("assignment_level is retired", e.msg) && occursin("Hungarian", e.msg)
        # RC's retired ICP (round 7): its keys and type are unknown to Estimation's strategy reader (an Irls table)
        for (k, v) in (("max_cycles", 10), ("refine", Dict("type" => "triggs")), ("residual", "radians"), ("least_squares", Dict("type" => "irls")))
            e = refused(c -> c["index"]["local_optimization_strategy"][k] = v); @test e isa ArgumentError && occursin(k, e.msg)
        end
        e = refused(c -> c["index"]["local_optimization_strategy"]["type"] = "icp"); @test e isa ArgumentError && occursin("icp", e.msg)
        e = refused(c -> c["index"]["local_optimization_strategy"]["polish"] = 1)        # Estimation's reader refuses an unknown key
        @test e isa ArgumentError && occursin("polish", e.msg)
        # the local optimization is plain least squares, the pixel refine too: no robust loss in the index
        e = refused(c -> c["index"]["local_optimization_strategy"]["loss"] = "huber")
        @test e isa ArgumentError && occursin("gaussian", e.msg)
        e = refused(c -> (c["index"]["local_optimization_strategy"] = Dict{String, Any}("type" => "triggs", "loss" => "huber", "efficiency_constant" => 3.0)))
        @test e isa ArgumentError && occursin("irls", e.msg)
        e = refused(c -> (c["index"]["refine_strategy"] = Dict{String, Any}("type" => "triggs", "loss" => "huber")))
        @test e isa ArgumentError && occursin("least squares", e.msg)
        # the scorer: MSAC, its threshold the record's (half a pitch), never the table's
        e = refused(c -> c["index"]["scoring"]["threshold"] = 2.5);                  @test e isa ArgumentError && occursin("threshold", e.msg)
        e = refused(c -> c["index"]["scoring"]["outlier_halfwidth"] = 2.5);          @test e isa ArgumentError && occursin("outlier_halfwidth", e.msg)
        e = refused(c -> c["index"]["scoring"] = Dict{String, Any}());               @test e isa ArgumentError && occursin("type", e.msg)   # the scorer is the config's: no code default
        e = refused(c -> c["index"]["scoring"] = Dict("type" => "no_such_scorer"));  @test e !== nothing                                     # Estimation's scoring_from_toml refuses an unknown type
        e = refused(c -> c["index"]["scoring"]["typo"] = 1);                          @test e isa ArgumentError && occursin("typo", e.msg)
        e = refused(c -> c["index"]["backproject"] = false);                 @test e isa ArgumentError && occursin("backproject", e.msg)
        e = refused(c -> c["index"]["ransac"]["max_trials"] = 0);            @test e isa ArgumentError && occursin("max_trials", e.msg)
        e = refused(c -> c["index"]["rng_seed"] = -1);                       @test e isa ArgumentError && occursin("rng_seed", e.msg)
        e = refused(c -> c["index"]["camera"] = "photos");                   @test e isa ArgumentError && occursin("photos", e.msg)
        e = refused(c -> c["index"]["excluded"]["DSC01302.JPG"] = "x");      @test e isa ArgumentError && occursin("DSC01302", e.msg)
        # the run key: an exclusion reason never re-keys the run; a setting, an exclusion and the detect run do
        k(c) = index_product(c).key
        c2 = deepcopy(good); c2["index"]["excluded"]["DSC01304.JPG"] = "edited"
        @test k(c2) == k(good)
        c2 = deepcopy(good); delete!(c2["index"]["excluded"], "DSC01304.JPG")
        @test k(c2) != k(good)
        c2 = deepcopy(good); c2["index"]["seed_tolerance"] = 4.0
        @test k(c2) != k(good)
        c2 = deepcopy(good); c2["index"]["local_optimization_strategy"]["max_iter"] = 3
        @test k(c2) != k(good)
        c2 = deepcopy(good); c2["detect"]["noise_sigma"] = 1.0
        @test k(c2) != k(good)                                                      # a luma set's noise_sigma keys it: it is read
        e = refused(c -> delete!(c["detect"], "noise_sigma"));               @test e isa ArgumentError && occursin("noise_sigma", e.msg)   # required for luma
    end

    # The synthetic photograph: the record's grid (17 × 17 at 5 mm, built here as the record states it) seen through the camera of
    # record (its vgio/cameras/v4 record, verbatim) at a known pose with a small perspective; 0.05 px noise, 4 distractors within
    # 0.2 pitch of interior nodes and 1 off the grid, shuffled.
    camera_json = """{"schema": "vgio/cameras/v4", "pixel_frame": "colmap", "cameras": [{"name": "hx400v-4.3mm", "width": 5184,
        "height": 3888, "K": {"fx": 3794.7162038135452, "fy": 3794.7162038135452, "cx": 2619.9394721718654, "cy": 2116.7547798154837,
        "skew": 0}, "lens": {"lift": {"radial": [[], [0.007863504061156833]]}}}]}"""
    cpath = joinpath(mktempdir(), "cameras.json"); write(cpath, camera_json)
    camera = VgtkCore.model(VgtkIO.load(Dict{Symbol, VgtkCore.Cameras.BoundedCamera}, cpath)[Symbol("hx400v-4.3mm")])
    target = VgtkCore.PlanarTarget([SVector((c - 1) * 5, (r - 1) * 5) .* VgtkCore.mm for r in 1:17 for c in 1:17])
    grid = thorlabs_grid(target, Dict{String,Any}("rows" => 17, "cols" => 17, "spacing_mm" => 5))
    Rt = VgtkCore.exp_so3(SVector(0.25, -0.15, 0.6))
    truth = VgtkCore.EuclideanMap(SMatrix{3,3,Float64}(Rt), SVector(-0.035, -0.03, 0.30))
    proj(n) = VgtkCore.project(camera, truth(SVector(n[1] * 0.005, n[2] * 0.005, 0.0)))
    rng = StableRNG(7)
    dots = [proj(n) + 0.05 * randn(rng, SVector{2,Float64}) for n in grid.nodes]
    distractors = [proj(SVector(8.15, 8.1)), proj(SVector(3.1, 12.15)), proj(SVector(12.12, 3.1)), proj(SVector(10.0, 9.85)),
                   proj(SVector(5.5, 6.5))]
    xy = SVector{2,Float64}[dots; distractors]
    perm = randperm(rng, length(xy))
    xy = xy[perm]
    true_point = [p ≤ 289 ? p : 0 for p in perm]                          # the record point of each shuffled point (0: a distractor)
    s = index_settings(cfg, path)
    reg0 = register_grid(camera, grid, xy, s; name = "synthetic")          # the synthetic grid's registration, read by the items below

    # the labelling is one of the grid's 8 symmetries: record node n ↦ the true node of its point
    D4 = [n -> SVector(a * n[1] + b * n[2], c * n[1] + d * n[2]) for (a, b, c, d) in
          ((1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0), (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0))]
    consistent(Δ) = any(σ -> (k0 = first(keys(Δ)); o = Δ[k0] - σ(grid.nodes[k0]); all(k -> Δ[k] == σ(grid.nodes[k]) + o, keys(Δ))), D4)

    # the window of the record's cols × rows grid (the hypothesis's window, board_assignment): the densest, the first of the
    # candidates on a tie — the same origin as counting every candidate's nodes, on random node sets with holes and strays — and
    # bounded by the nodes, not their spread (a hypothesis's stray rays meet the plane far off the board)
    @testset "index: densest_window is the brute-force search's window" begin
        brute(nodes, c, r) = (held(o) = count(n -> all(o .≤ n .≤ o .+ SVector(c - 1, r - 1)), nodes);
                              argmax(held, [n - SVector(i, j) for n in nodes for i in 0:(c - 1) for j in 0:(r - 1)]))
        wr = StableRNG(11)
        for _ in 1:200
            w = rand(wr, 3:12)
            nodes = keys(Dict(SVector(rand(wr, -w:w), rand(wr, -w:w)) => 0 for _ in 1:rand(wr, 1:50)))
            c, r = rand(wr, 1:7), rand(wr, 1:7)
            @test densest_window(nodes, c, r) == brute(nodes, c, r)
            far = push!(collect(SVector{2,Int}, nodes), SVector(10^6, -10^6))               # a stray far off: the compressed columns and rows
            @test densest_window(far, c, r) == brute(far, c, r)
        end
        @test densest_window([SVector(0, 0), SVector(5, 5)], 2, 2) == SVector(0, 0)     # a tie: the first candidate of the first node
        @test densest_window([SVector(0, 0), SVector(1, 0), SVector(10^9, -10^9)], 17, 17) == SVector(0, 0)   # a node 10⁹ pitches off: no box over it
        # a grazing hypothesis's ray meets the plane beyond what an Int rounds: a miss (Inf), never an InexactError
        Rg = SMatrix{3,3,Float64}(VgtkCore.exp_so3(SVector(pi / 2 - 1e-17, 0.0, 0.0)))
        pg = GridPoseProblem(camera, xy, grid; s.seed_tolerance)
        ag = board_assignment(pg, VgtkCore.EuclideanMap(Rg, SVector(0.0, 0.0, 0.3)))
        @test all(x -> isinf(x) || x < 2.5, ag.distance)
        @test_throws ArgumentError densest_window(SVector{2,Int}[], 3, 3)
    end

    @testset "index: the registration of a synthetic grid" begin
        reg = reg0
        @test length(reg.assignment) == 289
        @test all(i -> true_point[i] != 0, values(reg.assignment))         # the distractors are unassigned
        Δ = Dict(k => grid.nodes[true_point[i]] for (k, i) in reg.assignment)
        @test consistent(Δ)
        # the pose is the pixel refine's on the final matches: it maps the record onto the true projection to the noise (0.05 px)
        @test maximum(k -> norm(VgtkCore.project(camera, reg.pose(SVector(grid.nodes[k][1] * 0.005, grid.nodes[k][2] * 0.005, 0.0))) -
                                proj(Δ[k])), keys(Δ)) < 0.05
        @test maximum(values(reg.residuals)) < 0.25 && sqrt(sum(abs2, values(reg.residuals)) / 289) < 0.1
        @test reg.trials ≥ s.ransac.min_trials && reg.inliers == 289
        # the photograph's noise scale (the evaluation's pooled scale): the MAD of its pixel residuals, the noise (0.05 px per
        # coordinate) to a few per cent, on 2·289 − 6 degrees of freedom; one pixel refine, the matches the local optimization's
        @test 0.04 < reg.noise_scale < 0.06 && reg.dof == 2 * 289 - 6
        @test reg.rounds == 1 && reg.reassigned == 0
    end

    # The scorer (owner 2026-11-14): every hypothesis scored in BOARD space — the rays met with the board, matched one-to-one to the
    # board window's nodes by the Hungarian with a dustbin at half a pitch — MSAC at that half pitch (mm, derived from the record).
    @testset "index: the scorer — the Hungarian with its dustbin, board millimetres, MSAC at half a pitch" begin
        p = GridPoseProblem(camera, xy, grid; s.seed_tolerance)
        sc = index_scoring(p, s.scoring)
        @test sc == VgtkCore.MsacScoring(2; threshold = 2.5) && dustbin_mm(p) == 2.5
        # under the true pose: every dot on its node (to the noise, in mm), each distractor in the dustbin (Inf)
        a = board_assignment(p, truth)
        @test a.window == SVector(0, 0) && length(a.pairs) == 289
        @test all(((i, n),) -> true_point[i] != 0 && n == grid.nodes[true_point[i]], a.pairs)
        @test all(i -> isinf(a.distance[i]), findall(==(0), true_point)) && maximum(a.distance[true_point .!= 0]) < 0.05
        r = zeros(length(xy)); mahalanobis_distances!(r, p, truth)
        @test isequal(r, a.distance)
        # a contested pair: two detections in one node's half-pitch disk (its own dot gone) — one node, one detection: the nearer
        # takes it, the other goes to the dustbin (the Hungarian's dummy at half a pitch), never to the free neighbour 3.5 mm away
        n0 = SVector(8, 8); i0 = findfirst(i -> true_point[i] != 0 && grid.nodes[true_point[i]] == n0, eachindex(xy))
        near, far = proj(SVector(8.06, 8.0)), proj(SVector(8.3, 8.0))
        n1 = SVector(9, 8); i1 = findfirst(i -> true_point[i] != 0 && grid.nodes[true_point[i]] == n1, eachindex(xy))
        keep = [i for i in eachindex(xy) if i != i0 && i != i1]
        pc = GridPoseProblem(camera, SVector{2,Float64}[xy[keep]; [far, near]], grid; s.seed_tolerance)
        ac = board_assignment(pc, truth); m = Dict(ac.pairs)
        @test m[length(keep) + 2] == n0 && !haskey(m, length(keep) + 1) && isinf(ac.distance[length(keep) + 1])
        @test !(n1 in values(m))                                                       # its dot gone, the node stays free
        # MSAC sweeps the dustbin's Inf as an outlier: its cost the threshold's square, a finite score, never NaN
        res = VgtkCore.solve(VgtkCore.EstimationProblem(p), Ransac(; scoring = sc, local_optimization_strategy = s.local_optimization_strategy, config = s.ransac);
                             rng = StableRNG(1))
        ρ = res.resid
        @test count(isinf, ρ) == 5 && all(.!VgtkCore.inlier_mask(res)[isinf.(ρ)]) && count(VgtkCore.inlier_mask(res)) == 289
        @test isfinite(res.metadata.quality) && isapprox(res.metadata.quality, sum(x -> 1 - min(x^2, 2.5^2) / 2.5^2, Float64.(ρ)); rtol = 1e-5)
    end

    @testset "index: the local optimization refits on the scorer's matches, frozen — plain least squares in board millimetres" begin
        # the truth turned 0.004 rad about the optical axis (the rim dots up to 0.3 mm off on the board, all within the dustbin): the
        # local optimization (Estimation's posterior-weight iteration at the Gaussian shape) refits on the matches the scorer made
        # for the seed and measures every refitted pose on them: at the truth to the noise
        on_grid(P) = [norm(VgtkCore.project(camera, P(SVector(n[1] * 0.005, n[2] * 0.005, 0.0))) - proj(n)) for n in grid.nodes]
        Rz = SMatrix{3,3,Float64}(VgtkCore.exp_so3(SVector(0.0, 0.0, 0.004)))
        seed = VgtkCore.EuclideanMap(Rz * SMatrix{3,3,Float64}(Rt), Rz * SVector(-0.035, -0.03, 0.30))
        p = GridPoseProblem(camera, xy, grid; s.seed_tolerance)
        lo = CountedLO(s.local_optimization_strategy)
        r = VgtkCore.solve(VgtkCore.RefineProblem(p, seed), LocalOptimization(lo; scoring = index_scoring(p, s.scoring)))
        @test lo.calls[] == 1
        # the LO weighs the consensus set alone (Estimation 0.11.2+: non-finite residuals weigh 0 under the Gaussian shape), so its
        # refit is accepted: at the truth to the noise (it was @test_broken on Estimation 0.11.1, every datum weighed 1, the cost Inf)
        @test maximum(on_grid(r.u)) < 0.05
        @test count(VgtkCore.inlier_mask(r)) == 289 && all(i -> true_point[i] != 0, findall(VgtkCore.inlier_mask(r)))   # every dot, no distractor
        θ1 = VgtkCore.refine!(nothing, p, seed, ones(length(xy)))                          # the LO's step: on the seed's matches, at the truth
        @test maximum(on_grid(θ1)) < 0.05
        mahalanobis_distances!(zeros(length(xy)), p, θ1)                                   # a refitted pose: measured on the frozen matches
        # the refit is on the seed's matches (the scorer's: board_assignment at the seed), and its residuals are measured on them
        anchor = Dict(board_assignment(p, seed).pairs)
        @test any(q -> _same_pose(q, θ1), p.correspondences.frozen) && Dict(p.correspondences.pairs) == anchor
        # a re-representation of a pose (its matrix and translation a few ulps off) is the same pose; a pose a micrometre away is not
        R1, t1 = SMatrix{3, 3, Float64}(VgtkCore.rotation(θ1)), SVector{3, Float64}(VgtkCore.translation(θ1))
        @test _same_pose(θ1, VgtkCore.EuclideanMap(R1 .* (1 + 4eps()), t1 .* (1 - 4eps())))
        @test !_same_pose(θ1, VgtkCore.EuclideanMap(R1, t1 + SVector(1e-6, 0.0, 0.0)))
        ρ = zeros(length(xy)); mahalanobis_distances!(ρ, p, θ1)
        @test isequal(ρ, frozen_distances!(zeros(length(xy)), p, θ1, anchor))
        # the weighted refit itself: on the scorer's matches of positive weight; too few, nothing; a pose the scorer never scored is
        # scored on demand (the Hungarian under it, its matches the anchor's) — never an error
        mahalanobis_distances!(ρ, p, seed)
        θ1 = VgtkCore.refine!(nothing, p, seed, ones(length(xy)))
        @test θ1 isa typeof(seed) && maximum(on_grid(θ1)) < maximum(on_grid(seed))
        @test VgtkCore.refine!(nothing, p, seed, [i <= 3 ? 1.0 : 0.0 for i in eachindex(xy)]) === nothing
        θt = VgtkCore.refine!(nothing, p, truth, ones(length(xy)))                         # not scored: assigned now, then refitted
        @test θt isa typeof(seed) && maximum(on_grid(θt)) < 0.05
        @test _same_pose(p.correspondences.anchor, truth) && Dict(p.correspondences.pairs) == Dict(board_assignment(p, truth).pairs)
    end

    # Owner's case (2026-11-14): a camera of record whose error exceeds half a pitch in PIXELS at the rim (nearest-node labelling in
    # the image takes the neighbour's node there) but stays within half a pitch on the BOARD. The board seen steeply from 12 cm, so its
    # near rim is magnified; each dot displaced on the board by a radial cubic field about the board's centre, 2.2 mm at the corners.
    @testset "index: a model error beyond half a pitch in pixels at the rim, within it on the board — every dot labelled" begin
        R2 = SMatrix{3,3,Float64}(VgtkCore.exp_so3(SVector(0.8, 0.1, 0.3)))
        truth2 = VgtkCore.EuclideanMap(R2, SVector(0.0, 0.0, 0.12) - R2 * SVector(0.04, 0.04, 0.0))
        p2(b) = VgtkCore.project(camera, truth2(SVector(b[1] * 0.005, b[2] * 0.005, 0.0)))     # b in pitches
        bend(n) = (v = SVector{2,Float64}(n) - SVector(8.0, 8.0); ρ = norm(v);
                   iszero(ρ) ? SVector{2,Float64}(n) : SVector{2,Float64}(n) + (2.2 / 5) * (ρ / (8 * sqrt(2)))^3 * v / ρ)
        pitch_px = norm(p2(SVector(9, 8)) - p2(SVector(8, 8)))                                # the board centre's pitch, px
        err_px = [norm(p2(bend(n)) - p2(n)) for n in grid.nodes]
        @test count(>(pitch_px / 2), err_px) ≥ 4                                             # beyond half a pitch in pixels
        @test maximum(n -> norm(bend(n) - n) * 5, grid.nodes) ≈ 2.2                          # within half a pitch (2.5 mm) on the board
        @test all(u -> 0 < u[1] < 5184 && 0 < u[2] < 3888, p2.(grid.nodes))
        rb = StableRNG(3)
        xyb = SVector{2,Float64}[[p2(bend(n)) + 0.05 * randn(rb, SVector{2,Float64}) for n in grid.nodes]; distractors]
        reg = register_grid(camera, grid, xyb, s; name = "bent")
        @test length(reg.assignment) == 289 && all(<=(289), values(reg.assignment))
        @test consistent(Dict(k => grid.nodes[i] for (k, i) in reg.assignment))
        @test maximum(values(reg.residuals)) > 10                                            # the camera's error remains, in pixels
    end

    @testset "index: the grid's placement — the hypothesis's window; a partial view at any offset; strays off the board left out" begin
        # the board is matched against an INFINITE lattice under a free pose, so a whole-pitch offset of the record's window is another
        # pose with the same residuals, never a refusal: a partial view (the outer column, and the top row, out of the frame) registers
        # every dot, each to a node one labelling gives (a symmetry and a whole-pitch shift)
        for cut in (n -> n[1] == 16, n -> n[1] == 16 || n[2] == 0)
            gone = [i for i in eachindex(xy) if true_point[i] != 0 && cut(grid.nodes[true_point[i]])]
            xyl = deleteat!(copy(xy), sort(gone)); tpl = deleteat!(copy(true_point), sort(gone))
            reg = register_grid(camera, grid, xyl, s; name = "synthetic")
            @test length(reg.assignment) == count(!cut, grid.nodes) && all(i -> tpl[i] != 0, values(reg.assignment))
            @test consistent(Dict(k => grid.nodes[tpl[i]] for (k, i) in reg.assignment))
            @test maximum(values(reg.residuals)) < 0.25
        end
        # strays just off the board: a row of spurious detections a pitch beyond its edge (on lattice positions: the window holding
        # most dots is the record's, they fall outside it) and half a dozen 0.6 pitch beyond the edge (3 mm from the rim's node, beyond
        # the dustbin): none takes a node
        beyond = [[proj(SVector(c, -1)) for c in 0:4]; [proj(SVector(16.6, r)) for r in 2:3:16]; [proj(SVector(c, 16.6)) for c in (3, 9)]]
        xyb = SVector{2,Float64}[xy; beyond]
        reg = register_grid(camera, grid, xyb, s; name = "synthetic")
        @test length(reg.assignment) == 289 && all(i -> i <= length(xy) && true_point[i] != 0, values(reg.assignment))
        @test consistent(Dict(k => grid.nodes[true_point[i]] for (k, i) in reg.assignment))
    end

    # One connected component (owner, 2026-11-14): a gap of whole missing columns splits the matched nodes in two (8-connectivity on
    # the board window). Two components mean a detection or assignment defect: refused with the components' sizes, never resolved.
    # Isolated missing nodes leave one component.
    @testset "index: the labelled dots form one connected component, or the photograph is refused" begin
        for k in (1, 2, 3)
            gap = 7:(6 + k)
            keep = [i for i in eachindex(xy) if true_point[i] != 0 && !(grid.nodes[true_point[i]][1] in gap)]   # the dots alone: a distractor
            e = try register_grid(camera, grid, xy[keep], merge(s, (; min_assigned = 50)); name = "gap"); nothing catch err; err end
            sizes = sort([count(n -> n[1] < gap[1], grid.nodes), count(n -> n[1] > gap[end], grid.nodes)]; rev = true)
            @test e isa LatticeComponents && occursin("gap: the labelled dots form 2 connected components", sprint(showerror, e))
            @test e.sizes == sizes
        end
        # isolated missing nodes (one here and there, a diagonal neighbour still assigned): one component, every other dot assigned
        holes = Set(SVector.([(3, 3), (8, 7), (12, 5), (5, 12), (14, 14), (8, 9)]))   # none a distractor's node (a distractor within half a pitch of a vacant node takes it)
        keep = [i for i in eachindex(xy) if true_point[i] == 0 || !(grid.nodes[true_point[i]] in holes)]
        reg = register_grid(camera, grid, xy[keep], s; name = "holes")
        @test length(reg.assignment) == 289 - length(holes) && all(i -> true_point[keep[i]] != 0, values(reg.assignment))
        @test consistent(Dict(kk => grid.nodes[true_point[keep[i]]] for (kk, i) in reg.assignment))
        # duplicate detections of a 3 × 3 block of dots (0.3 px off each): each node keeps one detection, the duplicates the dustbin
        dup = SVector{2,Float64}[proj(SVector(c, r)) + SVector(0.3, 0.0) for c in 7:9 for r in 7:9]
        reg = register_grid(camera, grid, SVector{2,Float64}[xy; dup], s; name = "dup")
        @test length(reg.assignment) == 289
        @test component_sizes([SVector(0, 0), SVector(1, 1), SVector(5, 5)], SVector(0, 0), 17, 17) == [2, 1]   # diagonal neighbours connect
    end

    # DSC06962 (the fisheye 10 cm set; fixtures/thorlabs/dsc06962: its circular detections and the labels the lattice growth gave
    # them, after2's registration/DSC06962.tsv, and the set's camera of record, fisheye10cm_all_green_kb_cv2.json):
    # the camera of record's error at the rim exceeds half a pitch in pixels (the nearest-node labelling took its top rows one column
    # over). Under RANSAC's board-space pose on Estimation 0.11.1 (no LO step accepted) five rim dots of the top row fell in the dustbin
    # and a highlight 22 px from a dot took its node; refined in pixels, every dot is within 1.7 mm of its node, and the Hungarian under
    # the refined pose labels all 289 as the growth did, the highlight in the dustbin. Since Estimation 0.11.2 the LO's refit is
    # accepted, so the rounds to the fixed point follow RANSAC's pose (asserted ≥ 1); the labels are asserted exactly.
    @testset "index: DSC06962 — every dot labelled as the gated growth labelled it, the highlights left out" begin
        d = projectdir("test", "fixtures", "thorlabs", "dsc06962")
        rows = read_summary(joinpath(d, "registration_growth.tsv"))
        xy6 = [SVector{2,Float64}(r["x"], r["y"]) for r in rows]; grown = [Int(r["point"]) for r in rows]
        cam6 = VgtkIO.load(Cameras.CameraModel, joinpath(d, "camera.json"); name = "cam")
        reg = register_grid(cam6, grid, xy6, merge(s, (; ransac = RansacConfig(0.999, 10000, 100))); name = "DSC06962")
        @test length(reg.assignment) == 289 == count(>(0), grown)
        @test all(i -> grown[i] > 0, values(reg.assignment))                                 # no highlight
        @test consistent(Dict(k => grid.nodes[grown[i]] for (k, i) in reg.assignment))
        @test reg.rounds >= 1                                                               # the Hungarian under the refined pose, to its fixed point
        p6 = GridPoseProblem(cam6, xy6, grid; s.seed_tolerance)
        @test isempty(mislabelled_dots(cam6, reg.pose, p6, Dict(i => grid.nodes[k] for (k, i) in reg.assignment)))   # the camera of record's rim error, no mislabel
    end

    # No mislabel (owner, 2026-11-14): every labelled dot is nearer its own node's image than any other node's under the final pose
    # (register_grid refuses the photograph otherwise: MislabelledDots); a labelling with two neighbours' labels swapped is caught at
    # exactly those two dots, one shifted by a column at the dots of the shifted column
    @testset "index: no assigned dot nearer to another node's image than its own (mislabelled_dots)" begin
        reg = reg0
        p = GridPoseProblem(camera, xy, grid; s.seed_tolerance)
        lab = Dict(i => grid.nodes[k] for (k, i) in reg.assignment)
        @test isempty(mislabelled_dots(camera, reg.pose, p, lab))
        d1, d2 = (only(i for (i, m) in lab if m == n) for n in (SVector(8, 8), SVector(9, 8)))
        swapped = merge(lab, Dict(d1 => lab[d2], d2 => lab[d1]))
        @test mislabelled_dots(camera, reg.pose, p, swapped) == sort([d1, d2])
        col = Dict(i => (n[1] == 16 ? n + SVector(1, 0) : n) for (i, n) in lab)              # the outer column labelled one further out
        @test mislabelled_dots(camera, reg.pose, p, col) == sort([i for (i, n) in lab if n[1] == 16])
        e = MislabelledDots("x: 2 labelled dots", [d1, d2]); @test sprint(showerror, e) == "x: 2 labelled dots" && e.dots == [d1, d2]
    end

    @testset "index: the seed need not be a corner" begin
        # the dot nearest the image's top-left corner removed (a corner of the grid, as when it is undetected or outside the frame): the
        # other 288 dots are assigned consistently and placed in the record's grid by the window holding most of them
        i0 = argmin([p[1] + p[2] for p in xy])
        xy2 = deleteat!(copy(xy), i0); tp2 = deleteat!(copy(true_point), i0)
        reg = register_grid(camera, grid, xy2, s; name = "synthetic")
        @test length(reg.assignment) == 288
        @test all(i -> tp2[i] != 0, values(reg.assignment))
        @test consistent(Dict(k => grid.nodes[tp2[i]] for (k, i) in reg.assignment))
        @test maximum(values(reg.residuals)) < 0.25                         # the refined pose: to the noise
        # stage evaluate keeps a photograph only when every dot is assigned
        @test !_all_dots((; X = [SVector(grid.nodes[k][1] * 0.005, grid.nodes[k][2] * 0.005, 0.0) for k in keys(reg.assignment)]), grid)
        @test _all_dots((; X = [SVector(n[1] * 0.005, n[2] * 0.005, 0.0) for n in grid.nodes]), grid)
        # … or all but `max_unassigned` of them (a few rim dots the camera of record leaves > 8 px off); never a whole row
        # or column, which a labelling shifted by one spacing would leave unassigned too
        r288 = (; X = [SVector(grid.nodes[k][1] * 0.005, grid.nodes[k][2] * 0.005, 0.0) for k in keys(reg.assignment)])
        @test !_all_dots(r288, grid, 0) && _all_dots(r288, grid, 1) && _all_dots(r288, grid, 8)
        @test check_max_unassigned(min(grid.cols, grid.rows) - 1, grid) == min(grid.cols, grid.rows) - 1
        e = try check_max_unassigned(min(grid.cols, grid.rows), grid) catch err; err end
        @test e isa ArgumentError && occursin("max_unassigned", e.msg)
    end

    @testset "index: refusals" begin
        e = try register_grid(camera, grid, xy, merge(s, (; min_assigned = 290)); name = "synthetic") catch err; err end
        @test e isa ArgumentError && occursin("synthetic", e.msg) && occursin("min_assigned", e.msg)
        e = try wound_neighbours([SVector(Float64(i), 0.0) for i in 0:7], 1) catch err; err end   # 8 points: no seed
        @test e isa ArgumentError && occursin("8 neighbours", e.msg)
        e = try register_grid(camera, grid, SVector{2,Float64}[], s; name = "dark") catch err; err end   # none detected (a dark RAW)
        @test e isa ArgumentError
        e = try register_grid(camera, grid, [SVector(100.0 * i, 0.0) for i in 0:9], s; name = "synthetic") catch err; err end
        @test e isa ArgumentError && occursin("no pose", e.msg)                   # a line of dots: no square of neighbours
        @test_throws ArgumentError thorlabs_grid(target, Dict{String,Any}("rows" => 17, "cols" => 17, "spacing_mm" => 4))
        @test_throws ArgumentError thorlabs_grid(target, Dict{String,Any}("rows" => 17, "cols" => 17))
    end

    @testset "index: the outputs read back" begin
        reg = reg0
        dir = mktempdir()
        rig = target_rig(target, s.target_id)
        VgtkIO.save(joinpath(dir, "targets.json"), rig)
        id = VgtkIO.content_id(joinpath(dir, "targets.json"))
        @test id == VgtkIO.content_id(rig)
        Σ = [SMatrix{2,2}(0.01 + i * 1e-5, 1e-4, 1e-4, 0.02) for i in eachindex(xy)]
        u = [VgtkCore.Uncertain(xy[i], VgtkCore.CovMat(Σ[i])) for i in eachindex(xy)]
        write_registration(dir, "synthetic", "synthetic.JPG", reg, u, s, id)
        tp = VgtkIO.load(VgtkCore.TargetObservations, joinpath(dir, "points", "synthetic.json"))
        @test tp.targets == [id] && collect(keys(tp[1])) == [s.target_id]
        obs = tp[1][s.target_id]                                                   # record point index => detection
        @test length(obs) == 289
        @test all(((k, x),) -> VgtkCore.geom(x) == xy[reg.assignment[k]] && cov(x) == Σ[reg.assignment[k]], obs)
        @test issorted([reg.assignment[k] for k in VgtkCore.sources(obs)])     # in the detections' order
        poses = VgtkIO.load(VgtkCore.TargetPoses, joinpath(dir, "poses", "synthetic.json"))
        P = VgtkCore.ustrip(VgtkCore.m, poses[1][s.target_id])
        @test VgtkCore.rotation(P) ≈ VgtkCore.rotation(reg.pose) && VgtkCore.translation(P) ≈ VgtkCore.translation(reg.pose)
        rows = read_summary(joinpath(dir, "registration", "synthetic.tsv"))   # VisualGeometryToolkitEval's table, read back
        @test Set(keys(first(rows))) == Set(REGISTRATION_COLUMNS) && length(rows) == length(xy) && count(r -> r["point"] > 0, rows) == 289
        @test [SVector(r["x"], r["y"]) for r in rows] == xy                     # the pixels back exactly
        # stage evaluate's reader: the registered dots (points/<photo>.json, their covariances with them) and the pose as plain numbers
        # (poses/<photo>.json, IO's TargetPoses: its rotation a quaternion, R back within an ulp)
        r = only(read_registrations(dir, grid))
        @test r.name == "synthetic" && length(r.xy) == 289 && _all_dots(r, grid)
        @test r.xy == [xy[i] for i in sort(collect(values(reg.assignment)))] && r.Σ == [Σ[i] for i in sort(collect(values(reg.assignment)))]
        P0 = VgtkCore.unlabel(reg.pose); R0, t0 = SMatrix{3, 3, Float64}(P0.R), SVector{3, Float64}(P0.t)
        @test maximum(abs.(r.R - R0)) < 1e-15 && maximum(abs.(r.t - t0)) < 1e-15   # a few ulp (measured over 200 poses: R 5.6e-16, t 5.6e-17)
        @test r.R ≈ R0 && r.t ≈ t0 && maximum(i -> norm(r.R * r.X[i] + r.t - (R0 * r.X[i] + t0)), eachindex(r.X)) < 1e-12
        # the scorer's scale read back (the evaluation's pooled scale); a scorer that recorded none writes null, read back NaN
        @test r.noise_scale == reg.noise_scale && r.dof == reg.dof
        write_registration(dir, "noscale", "noscale.JPG", merge(reg, (; noise_scale = NaN)), u, s, id)
        rn = only(filter(x -> x.name == "noscale", read_registrations(dir, grid)))
        @test isnan(rn.noise_scale) && VgtkIO.read_record(joinpath(dir, "registration", "noscale_scale.json"))["noise_scale"] === nothing
        # the pool leaves a photograph without a scale out (Estimation's PooledScale); with none at all it refuses, naming why
        both = read_registrations(dir, grid)
        @test VgtkCore.noise_scale(pooled_index_scale(both)) ≈ reg.noise_scale && pooled_index_scale(both).metadata.left_out == 1
        e = try pooled_index_scale([rn]); nothing catch err; err end
        @test e isa ArgumentError && occursin("noise scale", e.msg) && occursin("originated none", e.msg)
    end

    @testset "evaluate: one target frame — each labelling's symmetry by the set's one rotation centre" begin
        # photographs about one rotation centre (x_cam = R_k (X − c) + d), one rolled far (1.7 rad about the optical axis), each
        # registered in a labelling of its own (a quarter turn of the grid k·90° about its centre): every photograph's chosen symmetry
        # puts its pose on one rotation centre (a rotation nearest the first photograph's alone took the wrong quarter turn for the
        # rolled one)
        d = SVector(0.001, 0.07, -0.07); c = Rt' * (d - SVector(-0.035, -0.03, 0.30))
        syms = grid_symmetries(grid)
        regs = map(enumerate((SVector(0.0, 0.0, 0.0), SVector(0.02, -0.01, 0.0), SVector(-0.015, 0.02, 0.01), SVector(0.0, 0.0, 1.7),
                              SVector(0.01, 0.015, -0.02)))) do (k, ω)
            R = SMatrix{3,3,Float64}(VgtkCore.exp_so3(ω) * Rt); t = d - R * c
            A, b = syms[2 * mod(k, 4) + 1]                                               # a proper quarter turn
            Rs = SMatrix{3,3,Float64}(R * inv(A)); ts = t - Rs * b
            X = [SVector(n[1] * 0.005, n[2] * 0.005, 0.0) for n in grid.nodes]
            (; name = "p$k", xy = [SVector(0.0, 0.0) for _ in X], X = [A * x + b for x in X], R = Rs, t = ts, Σ = nothing, noise_scale = NaN, dof = 0)
        end
        can = canonical_registrations(regs, grid)
        rows = reduce(vcat, [hcat(-r.R, Matrix{Float64}(I, 3, 3)) for r in can]); θ = rows \ reduce(vcat, [r.t for r in can])
        @test maximum(r -> norm(r.t - (θ[4:6] - r.R * θ[1:3])), can) < 1e-9          # one rotation centre for every photograph
        @test all(r -> (-r.R' * r.t)[3] > 0, can)                                     # every camera on the +z side
        @test all(k -> maximum(i -> norm(regs[k].R * regs[k].X[i] + regs[k].t - (can[k].R * can[k].X[i] + can[k].t)), eachindex(can[k].X)) < 1e-12, eachindex(can))
        # a photograph labelled a whole pitch off (the index stage places a partial view on any window: the same dots under another
        # pose, X + δ and t − R δ) is moved back onto the set's one rotation centre with its symmetry: the same labelling — so the
        # same motion fit and residuals — as when it was labelled right
        for (k, Δ) in ((3, SVector(1, 0)), (4, SVector(-2, 1)), (1, SVector(0, -1)), (2, SVector(2, -2)))   # the gauge photograph too
            δ = SVector(Δ[1] * grid.spacing, Δ[2] * grid.spacing, 0.0)
            off = [j == k ? merge(r, (; X = [x + δ for x in r.X], t = r.t - r.R * δ)) : r for (j, r) in enumerate(regs)]
            can2 = canonical_registrations(off, grid)
            @test all(j -> can2[j].X ≈ can[j].X && can2[j].R ≈ can[j].R && can2[j].t ≈ can[j].t, eachindex(can))
            rows2 = reduce(vcat, [hcat(-r.R, Matrix{Float64}(I, 3, 3)) for r in can2]); θ2 = rows2 \ reduce(vcat, [r.t for r in can2])
            @test maximum(r -> norm(r.t - (θ2[4:6] - r.R * θ2[1:3])), can2) < 1e-9 && θ2[4:6] ≈ θ[4:6]   # one centre; the same offset d
        end
        # a full-grid photograph whose index pose is 4 mm off in the target plane (more than half a pitch) keeps its labelling: a
        # whole-pitch shift would move a row of its dots off the finite board
        noisy = [j == 3 ? merge(r, (; t = r.t + r.R * SVector(0.004, 0.0, 0.0))) : r for (j, r) in enumerate(regs)]
        can3 = canonical_registrations(noisy, grid)
        @test all(j -> can3[j].X ≈ can[j].X, eachindex(can)) && all(r -> r.offset == (0, 0), can3)
    end

    @testset "evaluate: the motion fit from the index stage's poses, the motion variants alone" begin
        # four photographs of the grid about one rotation centre (x_cam = R_k (X − c) + d), registered by the index stage (its local optimization's
        # poses) and scored through the camera of record itself: the motion fit starts from the registrations' poses (no free-pose
        # fit), its translation residual is that of the index poses, and the evaluation writes the motion variant only
        dir = mktempdir(); out = mktempdir()
        rig = target_rig(target, s.target_id); VgtkIO.save(joinpath(dir, "targets.json"), rig); id = VgtkIO.content_id(joinpath(dir, "targets.json"))
        d = SVector(0.001, -0.002, 0.003); c = Rt' * (d - SVector(-0.035, -0.03, 0.30))       # photograph 1 is the synthetic photograph's pose
        rngm = StableRNG(3)
        summary = Tuple[]
        for (k, δω) in enumerate((SVector(0.0, 0.0, 0.0), SVector(0.02, -0.01, 0.0), SVector(-0.015, 0.02, 0.01), SVector(0.01, 0.015, -0.02),
                                  SVector(0.012, 0.0, 0.015)))
            Rk = VgtkCore.exp_so3(δω) * Rt
            xyk = [SVector{2,Float64}(VgtkCore.project(camera, Rk * (SVector(n[1] * 0.005, n[2] * 0.005, 0.0) - c) + d)) + 0.05 * randn(rngm, SVector{2,Float64}) for n in grid.nodes]
            regk = register_grid(camera, grid, xyk, s; name = "p$k")
            write_registration(dir, "p$k", "p$k.JPG", regk, [VgtkCore.Uncertain(p, VgtkCore.CovMat(SMatrix{2,2}(0.0025, 0.0, 0.0, 0.0025))) for p in xyk], s, id)
            rs = residual_summary(collect(values(regk.residuals))); rms, mx = rs.rms_px, rs.max_px
            push!(summary, ("p$k.JPG", length(xyk), regk.inliers, length(regk.assignment), 0, regk.trials, rms, mx))
        end
        write_tsv(joinpath(dir, "summary.tsv"), INDEX_SUMMARY_COLUMNS, summary)
        lpath = joinpath(mktempdir(), "cam.json"); write(lpath, replace(camera_json, "hx400v-4.3mm" => "cam"))
        es = (; optimization_strategy = Triggs{HuberShape}(; efficiency_constant = 3.0),
              optimization_strategy_table = Dict{String, Any}("type" => "triggs", "loss" => "huber", "efficiency_constant" => 3.0),
              holdout = false, iterations = 80, profile_bins = 4,
              centre = SVector(2592.0, 1944.0), rmax = 3240.0, centre_rho = 0.4, edge_rho = 0.6, max_unassigned = 0)
        sums = evaluate_grid_models(dir, grid, es, out; W = 5184, H = 3888, models = [("rec", "lens:" * lpath)])
        @test [x.variant for x in sums] == ["motion"]
        @test sort(filter(f -> startswith(f, "vectors_"), readdir(out))) == ["vectors_rec_motion.json"]
        rec = VgtkIO.read_record(joinpath(out, "eval_rec.json"))
        @test !haskey(rec, "free") && !haskey(rec, "fixed") && haskey(rec, "motion")
        f = rec["fit"]["in_sample"]
        # the rotation-centre refine: Core's LevenbergMarquardt on Estimation's CameraPosesProblem, converged, at the index stage's pooled scale handed down
        @test f["converged"] && f["retcode"] == "Success" && f["iterations"] < 80 && f["scale_source"] == "ScaleUpstream"
        @test 0 < f["scale_px"] < 1 && f["scale_px"] ≈ rec["pooled_scale"]["scale_px"] && f["loss_width_px"] ≈ 3.0 * f["scale_px"]
        @test rec["pooled_scale"]["photos"] == 5 && !isempty(rec["fit"]["held_camera"])          # every camera parameter held
        @test rec["pooled_scale"]["left_out"] == 0 && isempty(rec["pooled_scale"]["without_scale"])  # none left out of the pool, and that is recorded
        @test occursin("0 whose index scorer recorded no scale", read(joinpath(out, "summary.md"), String))
        # a photograph whose index scorer recorded no scale (null) is left out of the pool — and named in the record and the summary
        pj = joinpath(dir, "registration", "p5_scale.json"); pr = VgtkIO.read_record(pj)
        VgtkIO.write_record(pj, (; schema = pr["schema"], photo = pr["photo"], noise_scale = nothing, dof = pr["dof"]))
        out2 = mktempdir()
        evaluate_grid_models(dir, grid, es, out2; W = 5184, H = 3888, models = [("rec", "lens:" * lpath)])
        rec2 = VgtkIO.read_record(joinpath(out2, "eval_rec.json"))
        @test rec2["pooled_scale"]["photos"] == 4 && rec2["pooled_scale"]["left_out"] == 1 && rec2["pooled_scale"]["without_scale"] == ["p5"]
        @test occursin("1 whose index scorer recorded no scale were left out of the pool (p5)", read(joinpath(out2, "summary.md"), String))
        @test rec["motion"]["summary"]["median_px"] < 0.1                    # the noise (0.05 px), through the true camera
        @test maximum(rec["motion"]["summary"]["start_translation_residual_mm"]) < 0.5   # the index poses: one rotation centre
        # the start: from every photograph's dots (no admission rule), c, d near the truth before the refine; its conditioning recorded
        st = rec["start"]
        @test st["retcode"] == "Success" && st["stop_rule"] == "ClosedForm" && st["method"] == "fit_rotation_center" && !haskey(st, "flagged")
        @test norm(SVector{3}(st["d_m"]...) - d) < 1e-3 && st["residual_px"]["median"] < 0.2
        @test length(st["singular_values"]) == 6 && length(st["singular_values_unnormalized"]) == 6 && st["condition"] ≥ 1
        @test norm(SVector{3}(rec["motion"]["summary"]["d_m"]...) - d) < 5e-4    # d, camera frame (c is in the canonical labelling's frame)
        @test readline(joinpath(out, "summary.tsv")) == join(EVALUATE_SUMMARY_COLUMNS, '\t') && [r["variant"] for r in read_summary(joinpath(out, "summary.tsv"))] == ["motion"]
        @test isempty(rec["refused"])
        # regression (the fisheye 10 cm's `KeyError: key "DSC06825" not found`: four photographs whose index poses were centimetres
        # off, none of whose dots the spline model's start projected, so the refine had no view of them and the per-photograph table
        # looked them up): a photograph the start projects no dot of — held out, no dot of one parity — is refused, its dots cleared,
        # named with its reason; the others untouched
        # and (the fisheye 30 cm pair, 2026-11-16: "Need at least 3 correspondences, got 2") one the start projects fewer dots of than
        # a pose's minimal set, VIEW_MIN_DOTS (in either parity, held out), is refused the same way — never handed to the refine
        B(v...) = BitVector(Bool.(collect(v)))
        live = [trues(8), falses(3), B(1, 0, 1), B(1, 1, 1, 1, 0, 0, 0, 0)]
        parity = [B(1, 0, 1, 0, 1, 0, 1, 0), B(1, 0, 1), B(1, 0, 1), B(1, 1, 1, 0, 1, 0, 0, 0)]
        @test VIEW_MIN_DOTS == 3
        l1 = deepcopy(live)
        @test start_refusals!(l1, ["a", "b", "c", "d"]) == [(; photo = "b", reason = "the start projects none of its 3 dots"),
                                                            (; photo = "c", reason = "the start projects 2 of its dots, fewer than a pose's 3")]
        @test l1[1] == live[1] && !any(l1[2]) && !any(l1[3]) && l1[4] == live[4]
        l2 = deepcopy(live)
        @test start_refusals!(l2, ["a", "b", "c", "d"], parity) ==
              [(; photo = "b", reason = "the start projects none of its 3 dots"), (; photo = "c", reason = "the start projects 2 of its dots, fewer than a pose's 3"),
               (; photo = "d", reason = "the start projects fewer than 3 of its dots of one parity (the held-out fits)")]
        @test l2[1] == live[1] && !any(l2[2]) && !any(l2[3]) && !any(l2[4])

        # A refinement case's model flows from its refine product, through the evaluate stage, into tab:refinement-chrome (owner 2026-11-14): a
        # chrome set's [evaluate] model of `case` reads the case's refine product of its [upstream] radiometric_refinement (the product
        # refinement_product names under results_root(): here a fixture product holding the camera of record as its lens.json, no parameters.json), the
        # evaluate stage scores it beside its start, and the tables stage prints both rows of that evaluation, the refinement's
        # retcode read from the run ("unfinished": no parameters.json, recorded, never printed)
        withenv("VGTK_RESULTS_ROOT" => mktempdir()) do
            ccfg = Dict{String, Any}("upstream" => Dict{String, Any}("radiometric_refinement" => "configs/experiments/radiometric_refinement.toml"),
                                     "evaluate" => Dict{String, Any}("models" => [Dict{String, Any}("label" => "ref", "kind" => "lens", "case" => "si30_bc5_warped")]))
            rc = upstream_refinements(ccfg)
            rdir = product_dir(refinement_product(rc, "si30_bc5_warped"))
            @test_throws ArgumentError evaluate_model_specs(ccfg)                       # no model in the run yet: named, never guessed
            mkpath(rdir); cp(lpath, joinpath(rdir, "lens.json"))
            specs = evaluate_model_specs(ccfg)
            @test specs == [("ref", "lens:" * joinpath(rdir, "lens.json"))]
            out3 = mktempdir()
            evaluate_grid_models(dir, grid, es, out3; W = 5184, H = 3888, models = [("rec", "lens:" * lpath); specs])
            P = Dict{String, Any}("render" => Dict{String, Any}("significant_digits" => 3, "change_digits" => 2),
                                  "tables" => Dict{String, Any}("radiometric_refinement" => Dict{String, Any}(
                                      "variant" => "motion", "layout" => "blocks", "label" => "tab:refinement-chrome", "caption" => "C.", "worse_color" => "red",
                                      "rows" => [Dict{String, Any}("set" => "HX400V", "model" => "camera of record", "chrome" => "hx", "start" => "rec", "refined" => "ref")])))
            products = (; summary = k -> joinpath(out3, "summary.tsv"), vectors = (k, l, v) -> joinpath(out3, "vectors_$(l)_$(v).json"),
                        retcode = (k, l) -> chrome_refinement_retcode(ccfg, rc, l; where = "chrome $k"), provenance = k -> (; run_key = "k", commit = "c"))
            rows = chrome_tables_rows(P, products)
            @test only(rows["radiometric_refinement"]).refinement_retcode == "unfinished"
            texdir = mktempdir()
            write_chrome_tables(P, (texdir,), rows)
            tex = read(joinpath(texdir, "table_radiometric_refinement.tex"), String)
            sm = Dict(String(r["model"]) => r for r in read_summary(joinpath(out3, "summary.tsv")) if r["variant"] == "motion")
            tfmt(x) = Reproduce.tex_number(x; sigdigits = 3)
            @test occursin("camera of record & " * tfmt(sm["rec"]["median_px"]) * " & ", tex)   # the start's evaluation …
            # … and the refinement's (the same camera here: no improvement, so its cells in worse_color, which marks the refined ones only)
            @test occursin("\\textcolor{red}{\\textbf{" * tfmt(sm["ref"]["rms_px"]) * "}}", tex) && occursin("\\label{tab:refinement-chrome}", tex)
            @test sm["ref"]["photos"] == sm["rec"]["photos"] == 5
        end

        # the head-to-head on the common dots (owner 2026-11-15): a second model, the same camera with a lens coverage (IO's camera
        # store, a circle) that leaves out the grid's rim dots, cannot evaluate them; the head-to-head of both models is on the dots
        # both score, each model's coverage stands beside it, and the camera of record is also scored on all of its own dots
        regs = read_registrations(dir, grid)
        xy_all = reduce(vcat, [r.xy for r in regs]); cen = sum(xy_all) / length(xy_all)
        rim_r = 0.8 * maximum(p -> norm(p - cen), xy_all)
        store = tempname() * ".json"; VgtkIO.save(store, Dict(:cam => RadiometricCamera.load_camera("lens:" * lpath)))
        rpath = joinpath(mktempdir(), "rim.json")
        write(rpath, replace(read(store, String), "\"lens\": {" => "\"lens_coverage\": {\"circle\": [$(cen[1]), $(cen[2]), $rim_r]},\n\"lens\": {"; count = 1))
        @test !RadiometricCamera.in_domain(RadiometricCamera.load_camera("lens:" * rpath), cen + SVector(rim_r + 1, 0.0))
        out4 = mktempdir()
        sums4 = evaluate_grid_models(dir, grid, es, out4; W = 5184, H = 3888, models = [("rec", "lens:" * lpath), ("rim", "lens:" * rpath)])
        T = Dict(String(r["model"]) => r for r in read_summary(joinpath(out4, "summary.tsv")))
        ndots = sum(r -> length(r.xy), regs); inside = count(p -> norm(p - cen) <= rim_r, xy_all)
        @test T["rec"]["coverage_dots"] == ndots && T["rec"]["coverage_fraction"] == 1.0           # the camera of record evaluates every dot
        @test T["rim"]["coverage_dots"] == inside < ndots && T["rim"]["coverage_fraction"] ≈ inside / ndots   # the rim model loses its rim dots, reported
        @test T["rec"]["dots"] == T["rim"]["dots"] == inside                                        # the head-to-head: the same dots for both
        R4 = Dict(l => VgtkIO.read_record(joinpath(out4, "eval_$l.json")) for l in ("rec", "rim"))
        @test R4["rec"]["motion"]["common"]["n"] == R4["rim"]["motion"]["common"]["n"] == inside
        @test R4["rec"]["motion"]["coverage"]["coverage_dots"] == ndots && R4["rim"]["motion"]["coverage"]["coverage_dots"] == inside
        # the camera of record on ALL of its dots: a score of its own (the rim dots in), not the head-to-head's
        @test R4["rec"]["motion"]["summary"]["n"] == ndots && T["rec"]["all_median_px"] ≈ R4["rec"]["motion"]["summary"]["median_px"] &&
              T["rec"]["all_rms_px"] ≈ R4["rec"]["motion"]["summary"]["rms_px"] && T["rec"]["rms_px"] ≈ R4["rec"]["motion"]["common"]["rms_px"] &&
              T["rec"]["all_rms_px"] != T["rec"]["rms_px"]
        @test [(x.label, x.common.n, x.coverage_dots) for x in sums4] == [("rec", inside, ndots), ("rim", inside, inside)]
        @test occursin("on the dots every model above scores: motion $inside of the $ndots registered dots", read(joinpath(out4, "summary.md"), String))
        # the vectors (the figures' maps) hold every dot a model scores: its coverage
        @test length(VgtkIO.read_record(joinpath(out4, "vectors_rec_motion.json"))["x"]) == ndots &&
              length(VgtkIO.read_record(joinpath(out4, "vectors_rim_motion.json"))["x"]) == inside
        # the scoring itself: the intersection by (photograph, registered dot), the statistics on the dots given, the coverage
        a = DotScores(("p1", 1) => (1.0, 1.0, 0.1), ("p1", 2) => (2.0, 2.0, 0.5), ("p2", 1) => (9.0, 9.0, 0.95))
        b = DotScores(("p1", 2) => (4.0, 4.0, 0.5), ("p1", 1) => (3.0, 3.0, 0.1))
        @test common_dots([a, b]) == [("p1", 1), ("p1", 2)] && common_dots([b, a]) == common_dots([a, b])
        h = dot_summary(a, common_dots([a, b]), es)
        @test h.n == 2 && h.median_px == 1.5 && h.photos == 1 && h.centre_median_px == 1.0 && h.edge_median_px === nothing
        @test dot_summary(a, keys(a), es).n == 3 && dot_summary(a, keys(a), es).edge_median_px == 9.0
        @test dot_coverage(b, 4) == (; coverage_dots = 2, coverage_fraction = 0.5)

        # the Schöps baseline is T. Schöps's own camera: a camera that no Schöps run holds as its camera (here a lens record; a grid
        # re-estimated from his alike) is refused — loudly, never skipped as a model not there yet
        @test_throws ArgumentError schoeps_own_camera(lpath)
        @test_throws ArgumentError evaluate_grid_models(dir, grid, es, mktempdir(); W = 5184, H = 3888, models = [("rec", "lens:" * lpath), ("s", "schoeps:" * lpath)])
        run = mktempdir(); write(joinpath(run, "intrinsics0.yaml"), "")                              # a Schöps run: his own intrinsics0.yaml
        cp(lpath, joinpath(run, "camera.json"))                                                        # and the front end's import of it
        @test schoeps_own_camera(run) == joinpath(run, "camera.json") == schoeps_own_camera(joinpath(run, "camera.json"))
        regrid = joinpath(run, "regrid.json"); cp(lpath, regrid)                                       # a camera beside his run, not its camera
        @test_throws ArgumentError schoeps_own_camera(regrid)
    end

    @testset "evaluate: a dot the start does not project is re-tested at the estimate (owner 2026-11-15)" begin
        # The fisheye ag10fe set's 53 rim dots: inside the spline lens's lattice, but out of its field at the rough start pose only;
        # they were dropped, never scored. Here a strongly distorted lens (the camera of record's division lift at k = 3.2: its
        # forward field ends 0.2726 rad off the axis) sees five photographs about one rotation centre at known poses; photograph 5's
        # index rotation is off by ~18 mrad, so its start pose throws its rim corner out of the field, while the refine (rotations
        # free) recovers the true pose, which projects it. One dot of p2 is out of the field at the TRUE pose: its detection is a gross
        # outlier inside the field, unprojected at the start and the estimate alike.
        dir = mktempdir(); out = mktempdir()
        rig = target_rig(target, s.target_id); VgtkIO.save(joinpath(dir, "targets.json"), rig); id = VgtkIO.content_id(joinpath(dir, "targets.json"))
        lpath = joinpath(mktempdir(), "fish.json"); write(lpath, replace(camera_json, "hx400v-4.3mm" => "fish", "0.007863504061156833" => "3.2"))
        fish = VgtkCore.model(RadiometricCamera.load_camera("lens:" * lpath))
        d = SVector(0.001, -0.002, 0.003); c = Rt' * (d - SVector(-0.035, -0.03, 0.30))
        Xs = [SVector(n[1] * 0.005, n[2] * 0.005, 0.0) for n in grid.nodes]
        rngf = StableRNG(5); both = Tuple{String, Int}[]
        for (k, δω) in enumerate((SVector(0.0, 0.0, 0.0), SVector(0.02, -0.01, 0.0), SVector(-0.015, 0.02, 0.01), SVector(0.01, 0.015, -0.02),
                                  SVector(0.012, 0.0, 0.015)))
            R = VgtkCore.exp_so3(δω) * Rt
            xyk = map(enumerate(Xs)) do (i, X)
                x = R * (X - c) + d; p = VgtkCore.project(fish, x)
                all(isfinite, p) && return SVector{2, Float64}(p) + 0.05 * randn(rngf, SVector{2, Float64})
                push!(both, ("p$k", i))                                   # out of the field at the true pose: detected 10 mrad inside it
                θ = acos(normalize(x)[3]) - 0.01; a = normalize(SVector(x[1], x[2]))
                SVector{2, Float64}(VgtkCore.project(fish, SVector(sin(θ) * a[1], sin(θ) * a[2], cos(θ))))
            end
            Ri = k == 5 ? VgtkCore.exp_so3(SVector(0.0, -0.015, 0.01)) * R : R          # photograph 5's index rotation off
            reg = (; assignment = Dict(i => i for i in eachindex(Xs)), pose = VgtkCore.EuclideanMap(SMatrix{3, 3, Float64}(Ri), d - Ri * c),
                   residuals = Dict(i => 0.0 for i in eachindex(Xs)), noise_scale = 0.05, dof = 2 * length(Xs) - 6)
            write_registration(dir, "p$k", "p$k.JPG", reg, [VgtkCore.Uncertain(p, VgtkCore.CovMat(SMatrix{2,2}(0.0025, 0.0, 0.0, 0.0025))) for p in xyk], s, id)
        end
        @test both == [("p2", 273)]
        es = (; optimization_strategy = Triggs{HuberShape}(; efficiency_constant = 3.0),
              optimization_strategy_table = Dict{String, Any}("type" => "triggs", "loss" => "huber", "efficiency_constant" => 3.0),
              holdout = true, iterations = 80, profile_bins = 4,
              centre = SVector(2592.0, 1944.0), rmax = 3240.0, centre_rho = 0.4, edge_rho = 0.6, max_unassigned = 0)
        evaluate_grid_models(dir, grid, es, out; W = 5184, H = 3888, models = [("fish", "lens:" * lpath)])
        rec = VgtkIO.read_record(joinpath(out, "eval_fish.json"))
        regs = read_registrations(dir, grid); ndots = sum(r -> length(r.xy), regs)
        @test ndots == 5 * 289 && isempty(rec["refused"]) && rec["fit"]["in_sample"]["converged"]
        # the start projects neither the outlier nor photograph 5's rim corner: both are left out of the fit (its start residuals) …
        @test rec["start"]["unprojected_at_start"] == 2 && !haskey(rec["start"], "dropped_unprojected")
        @test rec["fit"]["in_sample"]["dots"] == ndots - 2
        outlier = only(r.xy[273] for r in regs if r.name == "p2")
        # held out (the headline): re-tested at the estimate — the corner, which it projects, is scored; the outlier it does not
        # project is counted, not scored
        sm = rec["motion_heldout"]["summary"]
        @test sm["n"] == ndots - 1 && sm["unprojected_at_estimate"] == 1
        @test sum(p -> p["n"], rec["motion_heldout"]["per_photo"]) == ndots - 1
        v = VgtkIO.read_record(joinpath(out, "vectors_fish_motion_heldout.json"))
        scored = Set(zip(v["x"], v["y"]))
        @test length(v["x"]) == ndots - 1 && (outlier[1], outlier[2]) ∉ scored
        @test all(r -> all(p -> (p[1], p[2]) ∈ scored || p == outlier, r.xy), regs)   # every other registered dot scored, the corner too
        @test sm["median_px"] < 0.1                                                   # the true poses recovered
        # in-sample (code review, RC 0.11.4): the dots its fit saw — the start's two misses are out of its sample, not scored there
        si = rec["motion"]["summary"]
        @test si["n"] == ndots - 2 && si["unprojected_at_estimate"] == 0 && si["median_px"] < 0.1
        vi = VgtkIO.read_record(joinpath(out, "vectors_fish_motion.json"))
        @test length(vi["x"]) == ndots - 2
    end

    @testset "evaluate: the local scale" begin
        # a pinhole camera (f 1000 px, centre (500, 400))
        K = VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(1000.0, 0, 0, 0, 1000.0, 0, 500.0, 400.0, 1))
        pin = VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(K), VgtkCore.Lens(VgtkCore.Perspective()))
        # the local scale: the direction's derivative per px, 1/f at the principal point; NaN off the camera
        @test _local_scale(pin, SVector(500.0, 400.0)) ≈ 1e-3 rtol = 1e-12
        @test isnan(_local_scale(pin, SVector(NaN, 400.0)))
        # a bounded camera (what load_camera returns for a schoeps: model) has its model's direction Jacobian
        bpin = VgtkCore.BoundedCamera(pin, VgtkCore.Size2(1000, 800))
        @test _direction_jacobian(bpin, SVector(520.0, 410.0)) == _direction_jacobian(pin, SVector(520.0, 410.0))
        @test _local_scale(bpin, SVector(500.0, 400.0)) ≈ 1e-3 rtol = 1e-12
    end

    @testset "evaluate: rays, foot points and the gauge of the rays' origins" begin
        # central rays (every origin 0): c★ = 0 and the feet 0
        c0, f0 = foot_points([(zero(SVector{3, Float64}), normalize(SVector(x, 0.2, 1.0))) for x in -0.5:0.25:0.5])
        @test c0 == zero(SVector{3, Float64}) && all(iszero, f0)
        # lines through one point c, their origins anywhere along them (the gauge): c★ = c and every foot point c
        c = SVector(0.001, -0.002, 0.003)
        rays = [(d = normalize(SVector(x, y, 1.0)); (c + s * d, d)) for (x, y, s) in ((-0.4, 0.1, 0.02), (0.3, -0.2, -0.05), (0.1, 0.4, 0.3), (-0.2, -0.3, 1.0))]
        cs, fs = foot_points(rays)
        @test isapprox(cs, c; atol = 1e-12)
        @test all(f -> isapprox(f, c; atol = 1e-12), fs)
        # skew lines: each foot point lies on its own line, nearest c★
        rays2 = [(SVector(0.001 * k, 0.0, 0.0), normalize(SVector(0.1 * k, 0.05, 1.0))) for k in -2:2]
        cs2, fs2 = foot_points(rays2)
        @test all(i -> norm(cross(fs2[i] - rays2[i][1], rays2[i][2])) < 1e-12 && abs(dot(cs2 - fs2[i], rays2[i][2])) < 1e-12, eachindex(rays2))
        # the summary's regions and columns
        @test "centre_mean_px" in EVALUATE_SUMMARY_COLUMNS && "edge_median_px" in EVALUATE_SUMMARY_COLUMNS
    end

    @testset "--plan lists the stage" begin
        cmd = `$(Base.julia_cmd()) --project=$(projectdir()) $(projectdir("scripts", "chrome_evaluation.jl")) --config=$path --plan`
        out = IOBuffer()
        ok = success(pipeline(addenv(cmd, "VGTK_DATA_ROOT" => mktempdir(), "VGTK_RESULTS_ROOT" => mktempdir()); stdout = out, stderr = out))
        s = String(take!(out))
        ok || @info s
        @test ok && occursin(r"^\s+detect\s+(check|\d+/\d+ produced)"m, s) && occursin(r"^\s+index\s+(check|\d+/\d+ produced)"m, s) && occursin("plan only", s)
        # the FX3 config runs the stages it has: prepare alone
        cmd3 = `$(Base.julia_cmd()) --project=$(projectdir()) $(projectdir("scripts", "chrome_evaluation.jl")) --config=$(projectdir("test", "fixtures", "thorlabs", "chrome_evaluation_fx3_14mm_20261102.toml")) --plan`
        out3 = IOBuffer()
        ok3 = success(pipeline(addenv(cmd3, "VGTK_DATA_ROOT" => mktempdir(), "VGTK_RESULTS_ROOT" => mktempdir()); stdout = out3, stderr = out3))
        s3 = String(take!(out3))
        ok3 || @info s3
        @test ok3 && occursin(r"^\s+prepare\s+(check|\d+/\d+ produced)"m, s3) && !occursin(r"^\s+detect\s"m, s3)
    end
end

# The rotation-centre refine's strategy and the fit switch of [evaluate] (`holdout`): the
# strategy is required, an Estimation strategy table of type "triggs" read by optimization_strategy_from_toml, and keys the run; the
# retired keys (pose_fit, huber_multiple, weights, and round 7's scoring and bad_pose_mad_multiple) are unknown keys; the switch defaults to
# the paper's protocol and an unknown value is refused; the configs of record state the index's Student-t scorer and Irls local
# optimization; the covariance whitening maps a bearing difference to the whitened pixel residual; the noise floor's maximum likelihood
# recovers a known σ₀²; the registrations carry the detections' covariances.
@testitem "chrome_evaluation evaluate switches" default_imports = false begin
    using Test
    using DrWatson
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm, normalize, cholesky, I
    using Statistics: median, cov
    using TOML: TOML

    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: HuberShape
    using VisualGeometryToolkitCore: Triggs
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("chrome_evaluation.jl"))

    path = projectdir("test", "fixtures", "thorlabs", "chrome_evaluation.toml")
    huber3 = Dict{String, Any}("type" => "triggs", "loss" => "huber", "efficiency_constant" => 3.0)
    st20 = Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => 20.0)

    @testset "evaluate: the motion fit's robust stage and the switches, defaults and refusals" begin
        base = load_chrome_evaluation_config(path)
        base["inputs"]["start"] = Dict{String, Any}("tier" => "rc_thorlabs", "path" => "a/model.json", "stage" => "evaluate", "provenance" => "test")
        base["evaluate"] = Dict{String, Any}("index" => "stage", "target" => base["index"]["target"],
                                             "optimization_strategy" => deepcopy(huber3),
                                             "iterations" => 80, "profile_bins" => 10, "lens_coverage" => "sensor", "centre_rho" => 0.4, "edge_rho" => 0.6,
                                             "holdout" => "off", "bootstrap" => Dict{String, Any}("resamples" => 100, "seed" => 1, "alpha" => 0.05),
                                             "comparisons" => Any[], "models" => [Dict{String, Any}("label" => "m", "kind" => "lens", "input" => "start")])
        with(kv...) = (c = deepcopy(base); foreach(((k, v),) -> c["evaluate"][k] = v, kv); c)
        without(k) = (c = deepcopy(base); delete!(c["evaluate"], k); c)
        refused(c) = try evaluate_settings(c); nothing catch err; err end
        s = evaluate_settings(base)                                                # the paper's protocol
        @test (s.optimization_strategy, s.holdout) == (Triggs{HuberShape}(; efficiency_constant = 3.0), false) &&
              s.optimization_strategy_table == huber3 && !haskey(s, :scoring) && !haskey(s, :weights)
        @test evaluate_settings(with("holdout" => "on")).holdout
        # holdout is a switch (RC 0.11.4): "on" | "off", required; the retired boolean refused naming its new spelling
        # the paired bootstrap's settings and comparisons: required, checked (two different labels of `models`)
        @test s.bootstrap == (; resamples = 100, seed = 1, alpha = 0.05) && isempty(s.comparisons)
        e = refused(without("bootstrap"));                                         @test e isa ArgumentError && occursin("bootstrap", e.msg)
        e = refused(with("bootstrap" => Dict{String, Any}("resamples" => 100, "seed" => 1, "alpha" => 1.5)));   @test e isa ArgumentError && occursin("alpha", e.msg)
        e = refused(with("comparisons" => [Dict{String, Any}("model" => "m", "reference" => "m")]));            @test e isa ArgumentError && occursin("two different labels", e.msg)
        e = refused(with("comparisons" => [Dict{String, Any}("model" => "m", "reference" => "x")]));            @test e isa ArgumentError && occursin("two different labels", e.msg)
        e = refused(without("holdout"));                                           @test e isa ArgumentError && occursin("holdout", e.msg)
        for (b, w) in ((true, "on"), (false, "off"))
            e = refused(with("holdout" => b))
            @test e isa ArgumentError && occursin("holdout = $b` is retired", e.msg) && occursin("use `", e.msg) && occursin("holdout = \"$w\"`", e.msg)
        end
        # the radial frame is the set's lens coverage (src/image_region.jl), never a centre and radius typed in (2026-11-15)
        @test s.lens_coverage == "sensor" && !haskey(s, :centre) && !haskey(s, :rmax)
        e = refused(with("centre" => [100.0, 100.0]));                             @test e isa ArgumentError && occursin("centre", e.msg) && occursin("unknown key", e.msg)
        e = refused(with("lens_coverage" => "image_region:nope"));                 @test e isa ArgumentError && occursin("no [image_regions.nope]", e.msg)
        # the retired keys (round 7): the motion fit's scorer and the start's bad-pose flags, unknown keys (Core check_keys)
        e = refused(with("scoring" => deepcopy(st20)));                            @test e isa ArgumentError && occursin("scoring", e.msg) && occursin("unknown key", e.msg)
        e = refused(with("bad_pose_mad_multiple" => 5.0));                         @test e isa ArgumentError && occursin("bad_pose_mad_multiple", e.msg) && occursin("unknown key", e.msg)
        e = refused(without("optimization_strategy"));                                @test e isa ArgumentError && occursin("optimization_strategy", e.msg)
        e = refused(with("optimization_strategy" => Dict{String, Any}("type" => "irls", "loss" => "huber", "efficiency_constant" => 3.0)))
        @test e isa ArgumentError && occursin("triggs", e.msg)                    # the rotation-centre refine is the Triggs-corrected LM
        e = refused(with("optimization_strategy" => merge(huber3, Dict("width" => 2.0))))
        @test e isa ArgumentError && occursin("width", e.msg)                     # Estimation's reader refuses an unknown key
        e = refused(with("robust" => deepcopy(huber3)));                           @test e isa ArgumentError && occursin("robust", e.msg)     # retired (2026-11-11)
        e = refused(with("pose_fit" => "triggs"));                                 @test e isa ArgumentError && occursin("pose_fit", e.msg)   # retired
        e = refused(with("huber_multiple" => 3.0));                                @test e isa ArgumentError && occursin("huber_multiple", e.msg)   # retired
        e = refused(with("weights" => "none"));                                    @test e isa ArgumentError && occursin("weights", e.msg)   # deleted: the refine whitens no dot
        e = refused(with("holdout" => "yes"));                                     @test e isa ArgumentError && occursin("holdout", e.msg)
        e = refused(with("loss" => "huber"));                                      @test e isa ArgumentError && occursin("loss", e.msg)
        # the Schöps baseline (kind "schoeps") is T. Schöps's OWN camera — the camera record of a front end's Schöps run, or a Schöps
        # run's directory — never a camera re-estimated from his (an index fit's, a refinement case's grid): anything else is refused,
        # and a front end's Schöps run is read as the Schöps baseline only
        runs(r, p) = Dict{String, Any}("config" => "configs/experiments/frontends.toml", "run" => r, "path" => p, "stage" => "evaluate", "provenance" => "test")
        sch(m, kv...) = (c = with("models" => [Dict{String, Any}("label" => "s", m...)]); foreach(((k, v),) -> c["runs"] = merge(get(c, "runs", Dict{String, Any}()), Dict{String, Any}(k => v)), kv); c)
        @test evaluate_settings(sch(("kind" => "schoeps", "input" => "own"), "own" => runs("schoeps.tamron17_siemens_w40", "camera.json"))).models[1]["kind"] == "schoeps"
        base["inputs"]["run_dir"] = Dict{String, Any}("tier" => "rc_baselines", "path" => "a/schoeps_dense_green", "stage" => "evaluate", "provenance" => "test")
        @test evaluate_settings(sch(("kind" => "schoeps", "input" => "run_dir"))).models[1]["input"] == "run_dir"
        e = refused(sch(("kind" => "schoeps", "input" => "fit"), "fit" => runs("fits.tamron17_ds", "camera.json")))
        @test e isa ArgumentError && occursin("T. Schöps's own camera", e.msg) && occursin("fits.tamron17_ds", e.msg)      # an index fit: re-estimated
        e = refused(sch(("kind" => "schoeps", "input" => "own"), "own" => runs("schoeps.tamron17_siemens_w40", "grid.json")))
        @test e isa ArgumentError && occursin("camera.json", e.msg)                                                    # not the run's camera record
        e = refused(sch(("kind" => "schoeps", "case" => "si30_spline_warped")))
        @test e isa ArgumentError && occursin("never a refinement case", e.msg)                                        # our refinement
        e = refused(sch(("kind" => "schoeps", "input" => "start")))
        @test e isa ArgumentError && occursin("a Schöps run's directory", e.msg) && occursin("a/model.json", e.msg)     # a camera file: not his run
        e = refused(sch(("kind" => "grid", "input" => "own"), "own" => runs("schoeps.tamron17_siemens_w40", "camera.json")))
        @test e isa ArgumentError && occursin("the Schöps baseline, kind \"schoeps\"", e.msg)                         # his camera, as another kind
        # the key of the fit: the robust stage and a switch key their own run
        k(c) = evaluate_product(c).key
        @test k(with("optimization_strategy" => merge(huber3, Dict("efficiency_constant" => 2.0)))) != k(base) && k(with("holdout" => "on")) != k(base)
        @test k(deepcopy(base)) == k(base)
        # an evaluate-only change re-keys the evaluation alone: the index, detect and prepare products it reads are the same, loaded
        let c2 = with("optimization_strategy" => merge(huber3, Dict("efficiency_constant" => 2.0)))
            @test haskey(c2, "index") && index_product(c2).key == index_product(base).key && detect_product(c2).key == detect_product(base).key
            @test evaluate_product(c2).record["upstream"]["index"] == index_product(base).key
        end
        # every config of record states its robust tables: the index's scorer and local optimization, the motion fit's robust stage
        E = projectdir("configs", "experiments")
        for f in readdir(E)
            startswith(f, "chrome_evaluation_") && endswith(f, ".toml") && f != "chrome_evaluation_paper.toml" || continue   # the sets' (not the paper config)
            c = TOML.parsefile(joinpath(E, f))
            t = get(c, "evaluate", nothing)
            # the Huber refine at the efficiency table's constant for d = 2 (2026-11-12, owner: no efficiency_constant literal)
            t === nothing || @test get(t, "optimization_strategy", nothing) == Dict("type" => "triggs", "loss" => "huber") && !haskey(t, "scoring") && !haskey(t, "robust") &&
                                   !haskey(t, "bad_pose_mad_multiple")
            t = get(c, "index", nothing)
            t === nothing || @test t["scoring"] == Dict("type" => "msac") &&
                                   t["local_optimization_strategy"] == Dict("type" => "irls", "loss" => "gaussian", "max_iter" => 10) &&
                                   t["refine_strategy"] == Dict("type" => "none") && !haskey(t, "assignment_level")
            # every FX3 chrome set is read from its RAW photographs through Features' raw module (owner 2026-11-14): prepare links them
            # with their EXIF (photosites), detect reads each mosaic under the FX3's sensor model, no noise_sigma; no decoded luma
            # photographs (the rc_baselines chrome/*.photos copies)
            d = c["detect"]
            @test d["image"] == "raw_green" && d["sensor_characterisation"] == "fx3_sensor_characterisation" &&
                  get(get(c, "prepare", Dict()), d["photos"], Dict())["mode"] == "photosites" && !haskey(d, "noise_sigma") &&
                  all(endswith(".ARW"), d["photographs"]) &&
                  c["inputs"][c["prepare"][d["photos"]]["raw"]]["tier"] == "rc_raw_photos" && !haskey(c["inputs"], "photos")
            # the FX3's characterisation of record in every set (the refinements' too): the all-darks record (vgio v2.0), never the
            # single-Dark exifband one it supersedes
            sc = c["inputs"]["fx3_sensor_characterisation"]
            @test sc["path"] == "sensor_characterisation/fx3.sensor_characterisation.gain-exifband-alldarks.json" &&
                  sc["sha256"] == "add8fc120b83806d1e70ecf4252a3b0bcf64d9fe007f42886789212f8c155737" && occursin("v2.0", sc["provenance"])
        end
    end

    # a planar grid of dots (the motion fit's photographs below)
    rng = StableRNG(11)
    X = [SVector(0.01i, 0.01j, 0.0) for i in -5:5 for j in -4:4]

    @testset "evaluate: the registrations carry the detections' covariances" begin
        dir = mktempdir(); det = joinpath(dir, "detections"); mkpath(det)
        xy = [SVector(10.0 + i, 20.0 - 0.5i) for i in 1:5]
        row(p, Σ, label, covariance_status = :a_posteriori) = VgtkCore.Attributed(VgtkCore.Uncertain(VgtkCore.PointFrame(p[1], p[2]), VgtkCore.CovMat(Σ)),
                                               (; label, window = VgtkCore.Point2(p[1], p[2]), F = 1.0, covariance_status))
        rows = [[row(p, SMatrix{2, 2, Float64}(1e-4 * i, 1e-5, 1e-5, 2e-4 * i), :circular) for (i, p) in enumerate(xy)];
                row(SVector(99.0, 99.0), SMatrix{2, 2, Float64}(NaN, NaN, NaN, NaN), :unknown, :unclassified)]   # Förstner's unclassified window: NaN, written without a covariance
        write_detections(joinpath(det, "p.json"), "p.JPG", rows)
        @test any(isnan, cov(last(read_detections(joinpath(det, "p.json"))).value))   # the absent covariance back as NaN
        u = read_detections(joinpath(det, "p.json"), "circular")             # Core's Uncertain pixels, the unknown row left out
        @test length(u) == 5 && VgtkCore.geom.(u) == xy && cov(u[2]) == SMatrix{2, 2, Float64}(2e-4, 1e-5, 1e-5, 4e-4)
        # a registration assigning dots 1, 2, 4, 5 (dot 3 unassigned), written by the index stage's writer
        reg = (; assignment = Dict(i => i for i in (1, 2, 4, 5)), pose = VgtkCore.EuclideanMap(SMatrix{3, 3, Float64}(I), SVector(0.0, 0.0, 0.3)),
               residuals = Dict(i => 0.1 for i in (1, 2, 4, 5)), noise_scale = NaN, dof = 0)
        write_registration(dir, "p", "p.JPG", reg, u, (; target_id = :t, camera_name = "cam"), repeat("a", 64))
        g = thorlabs_grid(VgtkCore.PlanarTarget([SVector((c - 1) * 5, 0) .* VgtkCore.mm for c in 1:5]), Dict{String, Any}("rows" => 1, "cols" => 5, "spacing_mm" => 5))
        r = only(read_registrations(dir, g))
        @test length(r.Σ) == 4 && r.Σ[3] == SMatrix{2, 2, Float64}(4e-4, 1e-5, 1e-5, 8e-4)          # dots 1, 2, 4, 5, in order
        @test r.xy == xy[[1, 2, 4, 5]]
    end
end

# The rotation-centre start (src/chrome_evaluation.jl _fit_rotation_center, round 7): c, d from the dots' bearings through Core's
# constraint protocol and whitening (RotationCenterBearing) and `_linear_least_squares` (Estimation's ReweightedLeastSquares on a LinearModel, its L2 solve), both sides normalized by Core's
# strategies (Hartley on the board, the half-diagonal on the image), its rotations fixed — recovered to rounding from exact data; the
# conditioning the same normalized or not (the image scale cancels in the whitening, the board's scales every column alike) and the
# normalized solve accurate where the unnormalized one loses digits (an ill-scaled set); and on a set whose index poses carry
# depth-heavy translation noise and rotation noise (the lever arm) far closer to the truth than the poses' own least squares
# t_k = d − R_k c.
@testitem "chrome_evaluation rotation-centre start" default_imports = false begin
    using Test
    using DrWatson
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm, normalize, I, svdvals
    using StableRNGs: StableRNG

    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEst
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("chrome_evaluation.jl"))

    # a pinhole (f 3000 px, centre (2128, 1424), a 4256 × 2848 image) turned about one point: rotation centre c (target frame), offset
    # d (camera frame), rotations about all three axes; a 17 × 17 board at 5 mm
    K = VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(3000.0, 0, 0, 0, 3000.0, 0, 2128.0, 1424.0, 1))
    pin = VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(K), VgtkCore.Lens(VgtkCore.Perspective()))
    image = (4256, 2848)
    board = [SVector(0.005 * i, 0.005 * j, 0.0) for i in 0:16 for j in 0:16]
    R0 = SMatrix{3, 3, Float64}(1, 0, 0, 0, -1, 0, 0, 0, -1)                  # the camera looks at the board's +z side
    d = SVector(0.004, -0.006, 0.05); c = R0' * (d - SVector(-0.04, -0.04, 0.30))
    ωs = [SVector(0.0, 0.0, 0.0), SVector(0.08, 0.0, 0.0), SVector(0.0, 0.08, 0.0), SVector(-0.06, 0.05, 0.03), SVector(0.05, -0.07, -0.04),
          SVector(0.0, 0.0, 0.1), SVector(-0.07, -0.03, 0.06), SVector(0.04, 0.06, -0.08)]
    Rs = [SMatrix{3, 3, Float64}(VgtkCore.exp_so3(ω)) * R0 for ω in ωs]
    Rd(Rs) = Dict(k => R for (k, R) in enumerate(Rs))
    rays(u) = (b = VgtkCore.backproject_with_jacobian(pin, u); (SVector{3, Float64}(b[1]), SMatrix{3, 2, Float64}(b[2])))
    # per photograph the PointSet of its dots: bearings, Σ (0.01 px²), X, o = 0, M, P = the ray at the pose (R, t)
    function dots_of(Rs, ts, X, c, d; σpx = 0.0, rng = StableRNG(1))
        out = Dict{Int, Any}()
        for (k, R) in enumerate(Rs)
            fm = [rays(SVector{2, Float64}(VgtkCore.project(pin, R * (x - c) + d)) + σpx * randn(rng, SVector{2, Float64})) for x in X]
            out[k] = VgtkCore.PointSet([p[1] for p in fm]; Σ = fill(SMatrix{2, 2, Float64}(0.01, 0.0, 0.0, 0.01), length(X)), X = X,
                                       o = fill(zero(SVector{3, Float64}), length(X)), M = [p[2] for p in fm], P = [R * x + ts[k] for x in X])
        end
        return out
    end
    ts = [d - R * c for R in Rs]
    # the unnormalized solve, for the comparison (the same rows at s = 1, X̄ = 0, σ = 1)
    unnormalized(Rs, dots) = (ids = sort(collect(keys(dots))); (A, y) = _rotation_center_rows(Rd(Rs), dots, ids, 1.0, zero(SVector{3, Float64}), 1.0);
                              θ = _linear_least_squares(A, y).u; (SVector{3}(θ[1:3]), SVector{3}(θ[4:6])))

    @testset "exact data: (c, d) recovered to rounding, normalized and not; the record" begin
        dots = dots_of(Rs, ts, board, c, d)
        st = _fit_rotation_center(Rd(Rs), dots; image_size = image)
        e = st.u
        @test st.retcode == VgtkCore.ReturnCode.Success && st.stop_rule == VgtkCore.ClosedForm && norm(e.center - c) < 1e-10 && norm(e.offset - d) < 1e-10
        cu, du = unnormalized(Rs, dots)
        @test norm(cu - c) < 1e-10 && norm(du - d) < 1e-10
        md = st.metadata
        @test length(md.singular_values) == 6 && length(md.singular_values_unnormalized) == 6 && st.rms < 1e-8
        @test md.scale_source === VgtkCore.ScaleNone && isnan(md.scale)
        # the image side cancels in the whitening: the rows at the half-diagonal's σ and at 1 are the same, to rounding
        ids = sort(collect(keys(dots)))
        σ = VgtkCore.AffineMap(VgtkCore.DiagonalNormalization(image)).linear[1, 1]
        A1, y1 = _rotation_center_rows(Rd(Rs), dots, ids, 1.0, zero(SVector{3, Float64}), 1.0)
        Aσ, yσ = _rotation_center_rows(Rd(Rs), dots, ids, 1.0, zero(SVector{3, Float64}), σ)
        @test isapprox(Aσ, A1; rtol = 1e-10) && isapprox(yσ, y1; rtol = 1e-10, atol = 1e-12)
        # the board's scale multiplies every column alike: the conditioning is the same normalized or not
        cond(sv) = sv[1] / sv[end]
        @test cond(md.singular_values) ≈ cond(md.singular_values_unnormalized) rtol = 1e-9
        @test_throws ArgumentError _fit_rotation_center(Dict(k => R for (k, R) in enumerate(Rs[1:end-1])), dots; image_size = image)
    end

    @testset "an ill-scaled set: the normalized solve is the accurate one" begin
        # the target frame's origin 3.6 km from the board: the conditioning is the same either way, but unnormalized the right-hand
        # side's cancellation costs c, d accuracy in proportion to the offset; normalized it does not
        far = SVector(3000.0, -2000.0, 0.0)
        Xf = [x + far for x in board]; cf = c + far
        tsf = [d - R * cf for R in Rs]
        dots = dots_of(Rs, tsf, Xf, cf, d)
        e = _fit_rotation_center(Rd(Rs), dots; image_size = image).u
        cu, du = unnormalized(Rs, dots)
        @test norm(e.center - cf) < 1e-10 && norm(e.offset - d) < 1e-12
        @test norm(e.center - cf) < norm(cu - cf) / 100 && norm(e.offset - d) < norm(du - d) / 100
    end

    @testset "depth-heavy pose noise and rotation noise: the dots beat the poses' least squares" begin
        rng = StableRNG(5)
        # the index poses: rotation noise 0.3 mrad (≈ 0.1 mm at the lever arm |c|), translation noise 3 mm along the depth, 0.1 mm across
        Rn = [SMatrix{3, 3, Float64}(VgtkCore.exp_so3(3e-4 * randn(rng, SVector{3, Float64}))) * R for R in Rs]
        tn = [t + SVector(1e-4 * randn(rng), 1e-4 * randn(rng), 3e-3 * randn(rng)) for t in ts]
        dots = dots_of(Rs, tn, board, c, d; σpx = 0.1, rng)                    # the pixels of the truth; P at the noisy poses
        Mls = reduce(vcat, [hcat(Matrix{Float64}(I, 3, 3), -Matrix(R)) for R in Rn]); dc = Mls \ reduce(vcat, collect.(tn))
        c_ls, d_ls = SVector{3}(dc[4:6]), SVector{3}(dc[1:3])
        st = _fit_rotation_center(Rd(Rn), dots; image_size = image)
        e = st.u
        @test st.retcode == VgtkCore.ReturnCode.Success && st.stop_rule == VgtkCore.ClosedForm
        @test norm(e.center - c) < norm(c_ls - c) / 3 && norm(e.offset - d) < norm(d_ls - d) / 3
        @test norm(e.offset - d) < 2e-3
    end
end

# The evaluate stage's photograph size (index_camera_size): the index camera of record is an input or a [runs] entry (the Tamron f/8
# and the 14 mm sets' Kannala–Brandt fits, frontends.toml [fits.*_kb]) — read by config_source, never config_input (a KeyError
# "camera" on those configs before 0.10.1).
@testitem "chrome_evaluation evaluate: the index camera of a [runs] entry" default_imports = false begin
    using Test
    using DrWatson
    using StaticArrays: SMatrix
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("parametric_starts.jl"))
    include(srcdir("radiometric_refinement.jl"))                                     # the config's [upstream] cases
    include(srcdir("chrome_evaluation.jl"))

    path = projectdir("configs", "experiments", "chrome_evaluation_fx3_tamron_largeaperture.toml")
    cfg = load_chrome_evaluation_config(path)
    ix = index_settings(cfg, path)
    @test haskey(cfg["runs"], ix.camera) && !haskey(cfg["inputs"], ix.camera)         # the config of record's camera: a [runs] entry
    @test_throws "a [runs] entry, not an input: read it with config_source" config_input(cfg, ix.camera)   # named, never a KeyError
    withenv("VGTK_RESULTS_ROOT" => mktempdir()) do
        fit = run_source(cfg, ix.camera)                                               # the front end's fit product, produced here
        K = VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(1000.0, 0, 0, 0, 1000.0, 0, 500.0, 400.0, 1))
        produce(fit; driver = "test", config_source = path) do dir
            write_start(joinpath(dir, cfg["runs"][ix.camera]["path"]), VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(K), VgtkCore.Lens(VgtkCore.Perspective())), 1000, 800;
                        lens_coverage = VgtkCore.Rect2{Float64}(0, 0, 1000, 800), provenance = (; source = "test"))
            write(joinpath(dir, fit.output), "fit = \"test\"\n"); nothing
        end
        @test index_camera_size(cfg, ix) == (1000, 800)
    end
end


# The paired photograph bootstrap of stage evaluate (owner 2026-11-15): the unit is the photograph (its dots share its pose, blur and
# lighting), each draw scores both models on the same drawn photographs' same dots; Eval's bootstrap_ci, StableRNG-seeded.
@testitem "chrome_evaluation paired photograph bootstrap" default_imports = false begin
    using Test
    using DrWatson
    using StableRNGs: StableRNG
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("chrome_evaluation.jl"))
    # two models on the same dots: the model's errors shifted by `shift` against the reference's; `corr` of its error variance a
    # per-photograph offset (its dots correlated within the photograph), the total variance the same
    function synth(rng; photos = 40, dots = 60, shift = -0.2, corr = 0.0)
        a = DotScores(); b = DotScores()
        for k in 1:photos
            u = corr * randn(rng)
            for i in 1:dots
                ρ = rand(rng)
                a[("p$k", i)] = (2.0 + shift + u + sqrt(1 - corr^2) * randn(rng), 0.0, ρ)
                b[("p$k", i)] = (2.0 + randn(rng), 0.0, ρ)
            end
        end
        return Dict("motion" => Dict("a" => a, "b" => b))
    end
    s = (; centre_rho = 0.4, edge_rho = 0.6); cmp = [(; model = "a", reference = "b")]
    boot(sc; seed = 7) = paired_differences(sc, Dict("motion" => common_dots(values(sc["motion"]))), cmp, ("motion",), s; resamples = 2000, seed, alpha = 0.05)
    P = boot(synth(StableRNG(1)))
    @test length(P) == length(PAIRED_REGIONS) * length(PAIRED_STATS) && all(p -> p.photos == 40 && p.resamples == 2000 && p.level == 0.95, P)
    # the interval covers the true difference (−0.2 for every statistic: a shift of the same law), and on every dot excludes 0
    @test all(p -> p.lo < -0.2 < p.hi, P) && all(p -> p.excludes_zero, filter(p -> p.region == "all", P))
    @test only(p for p in P if p.region == "all" && p.stat == "mean_px").dots == 2400
    # photograph-level, not dot-level: with the model's error correlated within the photographs the interval widens
    width(P) = [p.hi - p.lo for p in P if p.region == "all"]
    Pc = boot(synth(StableRNG(1); corr = 0.9))
    @test all(width(Pc) .> 2 .* width(P))
    # the seed reproduces; another seed draws otherwise
    @test boot(synth(StableRNG(1))) == P
    @test boot(synth(StableRNG(1)); seed = 8) != P
    # one replicate set per comparison, variant and region (RC 0.11.5, the 0.11.4 review's item 7c): each statistic's interval is
    # Eval's bootstrap_ci of that statistic alone under the group's stream StableRNG(seed + g), g the region's index here — the four
    # statistics on the same draws, each replicate summarized once
    sc = synth(StableRNG(1)); cd = common_dots(values(sc["motion"]))
    for (g, r) in enumerate(PAIRED_REGIONS)
        ps = paired_photos(sc["motion"]["a"], sc["motion"]["b"], cd, r == "all" ? (ρ -> true) : r == "centre" ? (ρ -> ρ < 0.4) : (ρ -> ρ > 0.6))
        for st in PAIRED_STATS
            d = q -> Float64(residual_summary(reduce(vcat, [p.model for p in q]))[Symbol(st)]) -
                     Float64(residual_summary(reduce(vcat, [p.reference for p in q]))[Symbol(st)])
            lo, hi = bootstrap_ci(d, StableRNG(7 + g), ps; n_resamples = 2000, alpha = 0.05)
            p = only(x for x in P if x.region == r && x.stat == st)
            @test (p.lo, p.hi) == (lo, hi) && p.difference == d(ps)
        end
    end
    # no difference: the interval holds 0
    @test all(p -> !p.excludes_zero, boot(synth(StableRNG(2); shift = 0.0)) |> P0 -> [p for p in P0 if p.region == "all"])
    # the photographs are the unit: one entry per photograph, its dots in the region
    ps = paired_photos(synth(StableRNG(3))["motion"]["a"], synth(StableRNG(3))["motion"]["b"], common_dots(values(synth(StableRNG(3))["motion"])), ρ -> ρ > 0.6)
    @test length(ps) == 40 && all(p -> length(p.model) == length(p.reference) && 0 < length(p.model) < 60, ps)
end
