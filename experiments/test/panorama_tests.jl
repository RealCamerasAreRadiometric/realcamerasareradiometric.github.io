# The panoramas of the pure-rotation sweeps (scripts/panorama_vignetting.jl, scripts/panorama_stitching.jl; src/panorama.jl,
# src/panorama_example.jl, src/panorama_stitching.jl): the registration, fit_vignetting, score_seams and render_panorama are
# PanoramaStitching's, tested there on synthetic fisheye sequences. Here: the candidates as vignettings at a sensor point, the
# maps' grid and rings — each on a case whose answer is known; the configs of record (each panorama_stitching config reads its
# sweep's chain and its lens's large-aperture refinement); the sweep's camera (its camera of record bounded by the sweep's lens
# coverage, the record of its image region); the rotation bundle's expected termination; and what the drivers read of
# PanoramaStitching's example and library, checked against the PanoramaStitching this project resolves.

@testitem "panorama radiometry: the maps' grid, the rings, a map as a vignetting, the exposure" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("panorama.jl"))
    using .Panorama: MapVignetting, no_vignetting, vignetting_map, ring_mean_map, equisolid_theta, panorama_exposure
    using StaticArrays: SVector
    using Statistics: mean

    # a radial V about the circle's centre, on the learned maps' grid (spacing 8 px of 4256 × 2848)
    circle = (2144.5, 1436.69, 1072.64); f_px = 929.85; size_ = (4256, 2848); spacing = 8
    Vtrue(u) = exp(-0.6 * (hypot(u[1] - circle[1], u[2] - circle[2]) / 1000)^2)
    M = vignetting_map((u, f) -> Vtrue(u), spacing, size_)
    @test size(M) == (2848 ÷ 8, 4256 ÷ 8)
    @test M[3, 5] == Vtrue(SVector(4.5 * 8, 2.5 * 8))                  # each value at its grid point's centre
    # the map as a vignetting: the grid points exactly, between them bilinear, the scale divided out
    mapped_v = MapVignetting(M, size_, 2.0)
    @test mapped_v(SVector(4.5 * 8, 2.5 * 8), nothing) ≈ M[3, 5] / 2 rtol = 1e-12
    @test abs(mapped_v(SVector(1000.0, 900.0), nothing) - Vtrue(SVector(1000.0, 900.0)) / 2) < 1e-4
    @test no_vignetting(SVector(1000.0, 900.0), nothing) == 1.0
    # rings of the equisolid field angle: a ring's mean is the mean of the analytic V over the grid points inside it
    edges = Float64.([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]); tmax = equisolid_theta(circle[3] - 24, f_px)
    @test tmax ≈ 2 * asind((circle[3] - 24) / (2f_px))
    lo, hi = edges[1:(end - 1)], edges[2:end]
    r = ring_mean_map(M, circle, f_px, size_, lo, hi, tmax)
    pts = [(j - 0.5) * spacing => (i - 0.5) * spacing for i in 1:size(M, 1), j in 1:size(M, 2)]
    inring(x, y, q) = (ρ = hypot(x - circle[1], y - circle[2]); ρ < circle[3] && lo[q] <= equisolid_theta(ρ, f_px) < min(hi[q], tmax))
    @test r[3] ≈ mean(Vtrue(SVector(x, y)) for (x, y) in pts if inring(x, y, 3))
    @test all(diff(filter(isfinite, r)) .< 0) && isnan(r[end])          # a radial map falls outward; a ring beyond the domain is empty
    # a map divided by its central ring's mean is 1 there: the candidate's scale of the panoramas stage
    n1 = ring_mean_map(M, circle, f_px, size_, [0.0], [5.0], tmax)[1]
    @test 0.99 < n1 < 1
    @test MapVignetting(M, size_, n1)(SVector(circle[1], circle[2]), nothing) ≈ 1 atol = 5e-3
    # the exposure: 1 over the percentile of the green plane, the pixels no frame sees left out
    img = cat(fill(0.5, 4, 5), reshape(collect(0.05:0.05:1.0), 4, 5), fill(0.1, 4, 5); dims = 3); img[1, 1, :] .= NaN
    @test panorama_exposure(img, 100) ≈ 1.0 && panorama_exposure(img, 50) ≈ 1 / 0.55
end

@testitem "panorama configs of record: each panorama_stitching config reads its sweep's chain and the lens's vignetting refinement" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    using VisualGeometryToolkitCore: thrown_message
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("panorama_example.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("panorama_stitching.jl"))
    for l in ("14mm", "fisheye", "tamron")
        path = projectdir("configs", "experiments", "panorama_stitching_fx3_$l.toml")
        cfg = load_panorama_stitching_config(path); up = panorama_stitching_upstream(cfg; path)
        pv = up.pv; rv = up.refinement
        @test pv["experiment"]["name"] == "panorama_vignetting" && rv["experiment"]["name"] == "radiometric_refinement"
        @test cfg["upstream"]["panorama_vignetting"] == "configs/experiments/panorama_vignetting_fx3_$l.toml"
        # the rings' disc is the chain's radial V centre and its camera's lens coverage (scripts/panorama_stitching.jl ring_disc): no circle
        # in either config; the Tamron covers the whole sensor, so its camera of record has no image circle and its radial centre is stated
        @test !haskey(pv["example"]["inputs"], "circle") && !haskey(cfg["lens"], "circle")
        rc = pv["example"]["vignetting"]["radial_center"]
        @test l == "tamron" ? (rc == [2128.0, 1424.0] && pv["inputs"]["lens"]["path"] == "geometric_accuracy/records/2710/tamron_opencv5_fit.json") :
                              rc == "circle_center"
        @test pv["example"]["vignetting"]["model"] == "radial" && pv["example"]["vignetting"]["gain_correction"] == "channel"
        @test pv["example"]["domain"]["plane"] == 2
        @test sort(collect(keys(cfg))) == sort(["experiment", "inputs", "upstream", "maps", "evaluate", "lens", "panoramas"])
        @test cfg["evaluate"]["size"] == cfg["lens"]["image_size"]
        @test collect(keys(cfg["maps"])) == ["learned"] && cfg["maps"]["learned"]["case"] == "f4_$(l)_spline_warped"   # the lens's f/4 refinement
        @test cfg["panoramas"]["gains"] == "none"                                              # an indoor sweep at one exposure
        @test cfg["panoramas"]["exposure_candidate"] in keys(cfg["maps"])                      # the PNGs' exposure: a corrected panorama's
        @test cfg["panoramas"]["unit_ring_deg"] == (l == "fisheye" ? [0.0, 10.0] : [0.0, 5.0])
        # the frames are decoded under the record the refinement read its photographs under: one sensor record for both
        fx3 = rv["inputs"]["fx3_sensor_characterisation"]; sc = pv["inputs"]["sensor_characterisation"]
        @test (sc["tier"], sc["path"], sc["sha256"]) == (fx3["tier"], fx3["path"], fx3["sha256"])
        # the products: the panoramas keyed by the sweep's run and the learned maps, the maps by the refinement
        withenv("VGTK_DATA_ROOT" => mktempdir(), "VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
            p = panoramas_product(cfg, up); e = panorama_maps_product(cfg, up)
            @test p.name == "panoramas" && p.output == "panoramas.json" && e.name == "evaluate" && e.output == "maps.json"
            @test p.record["upstream"] == Dict("panorama_vignetting" => panorama_vignetting_product(pv).key, "maps" => e.key)
            @test e.record["upstream"] == Dict("learned" => refine_product(rv, cfg["maps"]["learned"]["case"]).key)
        end
    end
    # a config with a key removed, and one with an unknown key: each refused by name
    path = projectdir("configs", "experiments", "panorama_stitching_fx3_tamron.toml")
    function edited(change)
        c = TOML.parsefile(path); change(c)
        p = joinpath(mktempdir(), "panorama_stitching_x.toml")
        open(io -> TOML.print(io, c), p, "w")
        return p
    end
    refused(change) = thrown_message(() -> load_panorama_stitching_config(edited(change)))
    @test load_panorama_stitching_config(edited(identity)) == load_panorama_stitching_config(path)
    @test occursin("unit_ring_deg", refused(c -> delete!(c["panoramas"], "unit_ring_deg")))
    @test occursin("ring_edges", refused(c -> c["panoramas"]["ring_edges"] = [0, 5, 10]))
    @test occursin("fit", refused(c -> c["fit"] = Dict("planes" => [1, 2, 3])))                 # no table of a stage the driver does not have
    @test occursin("gains", refused(c -> c["panoramas"]["gains"] = "frame"))
    @test occursin("exposure_candidate", refused(c -> c["panoramas"]["exposure_candidate"] = "rooftop"))
    c = load_panorama_stitching_config(edited(c -> c["maps"]["learned"]["case"] = "f4_tamron_spline"))
    @test occursin("is not a case", thrown_message(() -> panorama_stitching_upstream(c; path = "x.toml")))
end

@testitem "panorama camera: the camera of record bounded by the sweep's measured image region (src/panorama_example.jl panorama_lens)" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("panorama_example.jl"))
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using StaticArrays: SMatrix
    # the configs of record: each sweep states its coverage — the 14 mm's the f/4 session's measured region (the image region the
    # vignetting set reads), the fisheye's its own frames', the Tamron the sensor — and reads its camera of record, never a typed-in circle
    for (lens, cov) in (("14mm", "image_region:fx3_14mm_f4_20261028"), ("fisheye", "image_region:fx3_fisheye_panorama_20261029"), ("tamron", "sensor"))
        c = load_panorama_vignetting_config(projectdir("configs", "experiments", "panorama_vignetting_fx3_$(lens).toml"))
        @test c["lens_coverage"] == cov && !endswith(c["inputs"]["lens"]["path"], "_lens_coverage.json") && !haskey(c["example"]["inputs"], "circle")
        withenv("VGTK_DATA_ROOT" => mktempdir(), "VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
            @test panorama_vignetting_product(c).record["lens_coverage"] == lens_coverage_key(c, cov)        # the run is keyed by its coverage's image region
            @test sort(collect(keys(panorama_vignetting_product(c).record["inputs"]))) == ["lens", "raw", "sensor_characterisation"]
        end
    end
    # the camera: the store's camera with the record's coverage, written into the run; the record measured on demand
    data = mktempdir(); res = mktempdir()
    cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(300.0, 0, 0, 0, 300.0, 0, 200.0, 150.0, 1))),
                                                      VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4))), VgtkCore.Size2(400, 300))
    VgtkIO.save(joinpath(data, "cam.json"), Dict(:cam => cam))
    e = VgtkCore.Ellipse(VgtkCore.Point2(201.5, 149.0), 141.0, 139.5, 0.2)
    cfg = Dict{String, Any}("lens_coverage" => "image_region:r", "example" => Dict{String, Any}("inputs" => Dict{String, Any}("lens_camera" => "cam")),
                            "inputs" => Dict{String, Any}("lens" => Dict{String, Any}("tier" => "rc_baselines", "path" => "cam.json", "stage" => "evaluate", "provenance" => "t"),
                                                          "set" => Dict{String, Any}("tier" => "rc_baselines", "path" => "set", "stage" => "evaluate", "provenance" => "t")),
                            "image_regions" => Dict{String, Any}("r" => Dict{String, Any}("input" => "set", "frames" => Dict("a.ARW" => "0"^64), "image" => "raw_green", "outlier_halfwidth_px" => 1000.0, "seed" => 1)))
    withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => res, "VGTK_SCRATCH" => "1") do
        # without the record the camera's first read MEASURES it (produced on demand): in a GPU job its image region's stage reads its
        # frames (their set is not there: the input's error), in a CPU job it is refused naming the command (its stage is a GPU's)
        m = sprint(showerror, try panorama_lens(cfg) catch err; err end)
        @test gpu_listed() ? (!isempty(m) && !occursin("needs a GPU", m) && occursin("provenance: t", m)) :
                             (occursin("needs a GPU", m) && occursin("image_regions.jl", m))
        @test !produced(image_region_product(cfg, "r"))
        produce(image_region_product(cfg, "r"); driver = "test", config_source = "test") do dir
            RadiometricCamera.save_image_region(joinpath(dir, IMAGE_REGION_FILE), (; lens_coverage = e, extent = (400, 300), frames = ["a.ARW"],
                                                center_scale = [0.1, 0.1], axes_scale = [0.2, 0.2], inliers = 10, candidates = 12, rim_gap = 0.3, cut = false))
            nothing
        end
        b = panorama_lens(cfg)
        @test VgtkCore.lens_coverage(b) == e && VgtkCore.model(b) == VgtkCore.model(cam) && VgtkCore.extent(b) == VgtkCore.extent(cam)
        @test VgtkCore.lens_coverage(panorama_lens(merge(cfg, Dict("lens_coverage" => "sensor")))) == VgtkCore.Rect2{Float64}(0, 0, 400, 300)
    end
end

# The rotation bundle's termination as this project checks it (its Core ReturnCode, the registration's bundle_return_code;
# src/panorama_example.jl check_bundle_retcode, check_registration; every panorama_vignetting config's [expected] bundle). The
# bundle itself is PanoramaStitching's own test (refine_rotations), not restated here.
@testitem "rotation bundle termination: the expected ReturnCode of each panorama config, checked against the recorded one" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("panorama_example.jl"))
    import VisualGeometryToolkitCore as VgtkCore

    # the stage's comparison: a match returns, a mismatch either way is an error naming both, an earlier release's outcome is refused by name
    @test check_bundle_retcode(string(VgtkCore.ReturnCode.Success), "Success"; reason = "r") === nothing
    @test check_bundle_retcode("MaxIters", "MaxIters"; reason = "r") === nothing
    m(f) = try f(); "" catch e; sprint(showerror, e) end
    s = m(() -> check_bundle_retcode("Stalled", "Success"; reason = "inside the circle", where = "cfg evaluate"))
    @test occursin("Stalled", s) && occursin("Success", s) && occursin("inside the circle", s)
    @test !isempty(m(() -> check_bundle_retcode("Success", "MaxIters"; reason = "corners")))
    for retired in ("converged", "not_run", "nonfinite_start")
        @test occursin("no VisualGeometryToolkitCore ReturnCode", m(() -> check_bundle_retcode("Success", retired; reason = "x")))
    end
    @test occursin("states no termination record", m(() -> check_bundle_retcode("Success", "unrecorded"; reason = "x")))
    # check_registration reads the termination record of the registration itself (the example's `registration(...)` result, its keyed
    # product): a converged registration as expected is accepted; a non-converged one, against a config expecting Success, is refused
    cfg = Dict{String, Any}("expected" => Dict{String, Any}("bundle" => "Success", "reason" => "the Tamron's corners dropped at the start"),
                            "example" => Dict{String, Any}("registration" => Dict{String, Any}("execution_provider" => "cuda")))
    onp(p) = Dict{String, Any}("descriptor" => Dict{String, Any}("execution_provider" => p), "matcher" => Dict{String, Any}("execution_provider" => p))
    converged = (; bundle_return_code = "Success", bundle_stop_rule = "GradientTolerance", bundle_dropped_start = 3, loma = onp("cuda"))
    @test check_registration(cfg, converged; where = "t") === nothing
    # LoMa on another provider than the config states (a CUDA run that fell back to the CPU) is refused, naming both
    e = m(() -> check_registration(cfg, (; converged..., loma = onp("cpu")); where = "t"))
    @test occursin("\"cpu\"", e) && occursin("\"cuda\"", e)
    e = m(() -> check_registration(cfg, (; converged..., loma = Dict{String, Any}("descriptor" => Dict{String, Any}("execution_provider" => "cuda"),
                                                                                  "matcher" => Dict{String, Any}("execution_provider" => "cpu"))); where = "t"))
    @test occursin("matcher", e)
    for rc in ("MaxIters", "Stalled", "Failure")
        e = m(() -> check_registration(cfg, (; converged..., bundle_return_code = rc); where = "t"))
        @test occursin(rc, e) && occursin("Success", e)
    end
    cfg["expected"]["bundle"] = "MaxIters"                              # and an expected cap the registration did not reach
    @test occursin("Success", m(() -> check_registration(cfg, converged; where = "t")))

    # every panorama_vignetting config states its expected termination (no default), consistent with its bundle_max_iter
    for f in ("panorama_vignetting_fx3_14mm.toml", "panorama_vignetting_fx3_fisheye.toml", "panorama_vignetting_fx3_tamron.toml")
        c = load_panorama_vignetting_config(projectdir("configs", "experiments", f))
        @test c["example"]["registration"]["execution_provider"] == "cuda"     # LoMa on the GPU; never "auto"
        rc = expected_bundle_retcode(c["expected"]["bundle"]; where = f)
        @test !isempty(c["expected"]["reason"])
        @test c["example"]["registration"]["bundle_max_iter"] != 0 || rc == VgtkCore.ReturnCode.MaxIters
    end
    function edited(change)
        c = TOML.parsefile(projectdir("configs", "experiments", "panorama_vignetting_fx3_14mm.toml")); change(c)
        p = joinpath(mktempdir(), "panorama_vignetting_x.toml"); open(io -> TOML.print(io, c), p, "w")
        return p
    end
    @test occursin("no VisualGeometryToolkitCore ReturnCode", m(() -> load_panorama_vignetting_config(edited(c -> c["expected"]["bundle"] = "converged"))))
    @test occursin("no [expected]", m(() -> load_panorama_vignetting_config(edited(c -> delete!(c, "expected")))))
    # a config without [example.registration]: a named ArgumentError, not a KeyError; the config's load only parses: the example's
    # run.jl is not loaded (the module stays unassigned)
    e = try load_panorama_vignetting_config(edited(c -> delete!(c["example"], "registration"))); nothing catch err; err end
    @test e isa ArgumentError && occursin("no [example.registration]", e.msg)
    @test !isassigned(PANORAMA_EXAMPLE_MODULE)
    # a retired [example.registration] key is refused AT LOAD by PanoramaStitching's one public list (refuse_retired_registration, its
    # successor named): such a config neither plans nor keys a run, and the example stays unloaded
    for (key, v) in (("near_duplicate_deg", 4.0), ("overlap_deg", 30.0), ("averaging_loss", "huber"))
        @test occursin("[example.registration]: `$key` is retired", m(() -> load_panorama_vignetting_config(edited(c -> c["example"]["registration"][key] = v))))
    end
    # what the driver supplies from [inputs] is no key of [example]: the frames' folder, the camera, the frames' sensor record (stated
    # once, for the decode and for the detector); a config without the record's input is refused
    @test occursin("is the driver's ([inputs.sensor_characterisation])", m(() -> load_panorama_vignetting_config(edited(c -> c["example"]["decode"]["sensor_characterisation"] = "none"))))
    @test occursin("is the driver's ([inputs.raw])", m(() -> load_panorama_vignetting_config(edited(c -> c["example"]["inputs"]["raw_dir"] = "x"))))
    @test occursin("sensor_characterisation", m(() -> load_panorama_vignetting_config(edited(c -> delete!(c["inputs"], "sensor_characterisation")))))
    @test !isassigned(PANORAMA_EXAMPLE_MODULE)
    # the one refusal point is the load: the example is loaded lazily by the first call into it (panorama_example_call)
    @test !any(f -> occursin("require_panorama_example", read(projectdir("scripts", f), String)), ("panorama_vignetting.jl", "panorama_stitching.jl"))
end

@testitem "the panorama drivers on the current PanoramaStitching: the example's config keys, registration fields and fit keywords they use" default_imports = false begin
    # the drivers' contract with PanoramaStitching's example, read from the example of the PanoramaStitching this project resolves: every
    # key the driver writes into the example's config is one the example requires there, every registration field the drivers read is
    # one of the example's registration result, the example's `run_example` returns its run's directory, and the keywords the drivers
    # hand its functions are theirs — so a PanoramaStitching release that changes one fails here, not on a GPU after the sweep's chain
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("panorama_example.jl"))
    text = read(PANORAMA_EXAMPLE, String); ex = Meta.parseall(text)
    "Every subexpression of `e` (depth first)."
    function exprs(e, out = Any[])
        e isa Expr || return out
        push!(out, e); foreach(a -> exprs(a, out), e.args)
        return out
    end
    "The method definition of `name` in `e` (a `function name(...)` block)."
    definition(e, name) = only(filter(x -> x.head === :function && x.args[1] isa Expr && x.args[1].head === :call && x.args[1].args[1] === name, exprs(e)))
    "The keywords of a method definition's signature."
    keywords(def) = Set(Symbol[a isa Symbol ? a : a.args[1] for a in only(filter(x -> x isa Expr && x.head === :parameters, def.args[1].args)).args])
    # the registration's result: the fields of the named tuple `registration` returns
    ret = last(filter(x -> x.head === :return && x.args[1] isa Expr && x.args[1].head === :tuple, exprs(definition(ex, :registration))))
    fields = Set(Symbol[a.args[1] for a in only(ret.args[1].args).args if a isa Expr && a.head === :kw])
    @test issubset([:rotations, :bundle_return_code, :loma], fields)
    # the fields the drivers read of it: `reg.<field>` and `R.reg.<field>`
    isreg(x) = x === :reg || (x isa Expr && x.head === :. && x.args[2] == QuoteNode(:reg))
    sources = [projectdir("scripts", "panorama_stitching.jl"), projectdir("scripts", "panorama_vignetting.jl"), srcdir("panorama_example.jl")]
    used = Set(Symbol[x.args[2].value for f in sources for x in exprs(Meta.parseall(read(f, String)))
                      if x.head === :. && length(x.args) == 2 && x.args[2] isa QuoteNode && isreg(x.args[1])])
    @test :rotations in used && :bundle_return_code in used && issubset(used, fields)
    # the example's functions take what the drivers hand them: fit a candidate's V held fixed, the registration its log, the decode
    # its masks, the lens the raw grid's size; the decode's result states the frames' sensor record
    @test :vignetting in keywords(definition(ex, :fit))
    @test :log in keywords(definition(ex, :registration)) && :masks in keywords(definition(ex, :decoded_frames))
    @test issubset([:raw_width, :raw_height], keywords(definition(ex, :load_lens)))
    @test occursin("sensor_characterisation_sha256 = sha", text) && occursin("save_srgb(path, img, exposure, wb)", text)
    # run_example returns its run's directory (the stage evaluate records it among its inputs)
    @test occursin(r"function run_example\(config_path; force = false\)", text) && last(filter(x -> x.head === :return, exprs(definition(ex, :run_example)))).args[1] === :dir
    # the config keys the driver supplies are keys the example reads (its sensor records, required for the decode and the detector)
    @test occursin("refuse_undeclared_sensor_record(cfg[\"decode\"]", text) && occursin("refuse_undeclared_sensor_record(cfg[\"registration\"]", text)
    @test occursin("C.cfg[\"inputs\"][\"raw_dir\"]", text) && occursin("C.cfg[\"inputs\"][\"lens\"]", text)
end

@testitem "the panoramas stage hands PanoramaStitching's library calls every keyword they require" default_imports = false begin
    # `score_seams(…; saturation_margin_px)` is required, with no default: a call that left it out would fail with an
    # UndefKeywordError only after the sweep's chain, on a GPU. Every keyword without a default in the resolved PanoramaStitching's
    # definition must be one the stage's call (src/panorama.jl sweep_panoramas) gives, by name or through the `mkw` named tuple it splats.
    using Test
    using DrWatson
    import PanoramaStitching
    "Every subexpression of `e` (depth first)."
    function exprs(e, out = Any[])
        e isa Expr || return out
        push!(out, e); foreach(a -> exprs(a, out), e.args)
        return out
    end
    kwname(a) = a isa Symbol ? a : a.head === :(::) ? a.args[1] : a.head in (:kw, :(=)) ? kwname(a.args[1]) : nothing
    required(a) = a isa Symbol || (a isa Expr && a.head === :(::))
    script = Meta.parseall(read(srcdir("panorama.jl"), String))
    mkw = only(filter(x -> x.head === :(=) && x.args[1] === :mkw, exprs(script))).args[2]
    mkw_keys = Set(Symbol[kwname(a) for a in only(filter(x -> x isa Expr && x.head === :parameters, mkw.args)).args])
    for f in (:score_seams,)
        file, _ = functionloc(only(methods(getfield(PanoramaStitching, f))))
        def = only(filter(x -> x.head === :function && x.args[1] isa Expr && x.args[1].head === :call && x.args[1].args[1] === f,
                          exprs(Meta.parseall(read(file, String)))))
        need = Set(Symbol[kwname(a) for a in only(filter(x -> x isa Expr && x.head === :parameters, def.args[1].args)).args if required(a)])
        calls = filter(x -> x.head === :call && x.args[1] == :(PanoramaStitching.$f), exprs(script))
        @test !isempty(calls)
        for c in calls
            params = filter(x -> x isa Expr && x.head === :parameters, c.args)
            given = Set{Symbol}()
            for a in (isempty(params) ? Any[] : only(params).args)
                a isa Expr && a.head === :... ? (a.args[1] === :mkw && union!(given, mkw_keys)) : push!(given, kwname(a))
            end
            foreach(a -> a isa Expr && a.head === :kw && push!(given, a.args[1]), c.args[2:end])   # `f(x, k = v)`, no semicolon
            @test :saturation_margin_px in need
            @test issubset(need, given)
        end
    end
end
