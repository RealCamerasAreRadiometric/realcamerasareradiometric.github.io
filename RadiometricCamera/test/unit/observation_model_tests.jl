# The observation model of a calibrated camera (RC 0.11.1; owner 2026-11-15: "if a sensor was estimated with a model, then it must be
# used to predict the model"): a camera calibrated under a sensor record predicts photographs read under exactly that record, through
# ONE entry point (`photo_set` / `observe`), by the types of its observation model — never a flag. A prediction under another record,
# of another photograph kind, of a frame the record does not describe, or from a record that states nothing (RC ≤ 0.11.0) is refused.

@testsetup module ObservationFixtures
    using RadiometricCamera, Test
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_camera.jl"))       # cam
    const FX3 = VgtkCore.DeviceIdentity(; make = "Sony", model = "ILME-FX3")
    "A small radiometric camera model (the kernel type) over a Double Sphere lens."
    function small_model()
        W = H = 24
        lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
        focus = RC.Focus(RC.FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), fill(1 / 1.5, 9))
        return RC.RadiometricCameraModel(lens, focus, 0.0386, VgtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2))
    end
    "A characterisation of the FX3 at `iso` with the channel black `black`, saved through IO at `path`; returns its sha256."
    function write_characterisation(path; black = (512.3, 512.2, 512.3, 512.2), iso = 100, body = FX3)
        f = VgtkCore.sensor_fixture(; body, iso, black, gain = 11.33, read_noise = (1.09, 1.10, 1.03, 1.10), source = VgtkCore.SensorSource.measured)
        VgtkIO.save(path, f.characterisation)
        return open(VgtkCore.sha256_hex, path)
    end
    export RC, VgtkCore, VgtkIO, FX3, small_model, write_characterisation
end

@testitem "observation model: its types refuse an inconsistent statement; its record round-trips through IO" setup = [ObservationFixtures] begin
    sha = repeat("0f", 32)
    @test_throws "not a sha256" RC.SensorRecord("add8fc")
    @test_throws "ONE observation each" RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :RGB)
    @test_throws "a sensor record describes RAW photosites" RC.ObservationModel(RC.SensorRecord(sha), RC.Decoded(false), :G)
    @test_throws "not one of" RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :Y)
    models = (RC.ObservationModel(RC.SensorRecord(sha), RC.Photosites(), :G), RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :G),
              RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(true), :RGB), RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :B))
    settings = RC.RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4, seed = 3)
    mktempdir() do dir
        for om in models
            f = joinpath(dir, "camera.json")
            RC.save_radiometric_camera(f, RC.CalibratedCamera(small_model(), om, settings))
            c = RC.load_radiometric_camera(f)
            @test c isa RC.CalibratedCamera && c.observation == om && typeof(c.observation) == typeof(om) && c.settings == settings
            @test c.model.lens == small_model().lens
        end
        @test VgtkIO.read_record(joinpath(dir, "camera.json")).observation_model.photos.kind == "decoded"
    end
    @test models[1] != models[2] && models[3] != RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB)
    @test occursin(sha, RC.observation_line(models[1])) && occursin("RAW photosites", RC.observation_line(models[1]))
end

@testitem "observation model: the entry point refuses another record, another photograph kind, an undescribed frame" setup = [ObservationFixtures] begin
    s = VgtkCore.parse_unit("s")
    mktempdir() do dir
        a = joinpath(dir, "a.sensor_characterisation.json"); b = joinpath(dir, "b.sensor_characterisation.json")
        sha_a = write_characterisation(a); sha_b = write_characterisation(b; black = (512.5, 512.4, 512.5, 512.4))
        @test sha_a != sha_b && RC.sensor_statement(a) == RC.SensorRecord(sha_a)
        settings = RC.RenderSettings()
        raw_cam = RC.CalibratedCamera(small_model(), RC.ObservationModel(RC.SensorRecord(sha_a), RC.Photosites(), :G), settings)
        # a prediction under another record: refused, naming both
        err = try RC.photo_set(raw_cam, ["DSC1.ARW"]; characterisation = b); nothing catch e; e end
        @test err isa ArgumentError && occursin(sha_a, err.msg) && occursin(sha_b, err.msg) && occursin(b, err.msg)
        @test_throws "give its location" RC.photo_set(raw_cam, ["DSC1.ARW"])          # the record's photographs need its location
        @test_throws "calibrated on RAW photographs (their photosites): it predicts those only, not a.png" RC.photo_set(raw_cam, ["a.png"]; characterisation = a)
        # a camera calibrated without a record: a characterisation is refused; a decoded camera refuses RAW photographs
        none_cam = RC.CalibratedCamera(small_model(), RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :G), settings)
        @test_throws "calibrated without a sensor characterisation" RC.photo_set(none_cam, ["DSC1.ARW"]; characterisation = a)
        dec_cam = RC.CalibratedCamera(small_model(), RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB), settings)
        @test_throws "calibrated on decoded images: it predicts those only, not DSC1.ARW" RC.photo_set(dec_cam, ["a.png", "DSC1.ARW"])
        @test_throws "calibrated without a sensor characterisation" RC.photo_set(dec_cam, ["a.png"]; characterisation = a)
        @test RC.photo_set(dec_cam, ["a.png", "b.png"]).models == [VgtkCore.SensorModel(), VgtkCore.SensorModel()]
        # a frame the record does not describe — an ISO without an entry (UnmatchedISO), another body (OtherBody) — is refused by name
        c = VgtkIO.load(VgtkCore.SensorCharacterisation, a)
        at(iso; body = FX3) = VgtkCore.sensor_fixture(; body, iso, black = (512.0, 512.0, 512.0, 512.0)).capture
        @test VgtkCore.component_sources(VgtkCore.sensor_model(c, at(100), 1)).reason == VgtkCore.Matched
        RC._require_described(sha_a, ["DSC1.ARW"], [VgtkCore.sensor_model(c, at(100), 1)])            # described: accepted
        @test_throws "DSC2.ARW (UnmatchedISO)" RC._require_described(sha_a, ["DSC2.ARW"], [VgtkCore.sensor_model(c, at(800), 1)])
        other = VgtkCore.DeviceIdentity(; make = "Sony", model = "ILCE-7M4")
        @test_throws "DSC3.ARW (OtherBody)" RC._require_described(sha_a, ["DSC3.ARW"], [VgtkCore.sensor_model(c, at(100; body = other), 1)])
    end
end

@testitem "observation model: a state predicts only photographs read as it was estimated (load_state_into!)" setup = [RCFixtures, ObservationFixtures] begin
    using StaticArrays
    tex = fixture_board()
    board = RC.PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    camera = small_model(); W = H = 24
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(VgtkCore.exp_so3(SVector(π, 0.0, 0.0))), SVector(0.0, 0.0, 1.4); check = false)]
    a = zeros(3, 9); a[:, 1] .= 1.0
    light = EnvironmentLight(SphericalHarmonicMap(a, 2), RC.𝔟(board), 2.5)
    mat = RC.Material(; 𝛂_ink = 0.2, 𝛂_noink = 0.3, 𝛈_ink = 1.5, 𝛈_noink = 1.4, μ = 0.5)
    settings = RC.RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    s = RC.CalibrationState(RC.Scene(board, W, H), settings, camera, poses, mat, light, [zeros(H, W, 3)])       # decoded renders, no record
    @test RC.observation_model(s) == RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB)
    @test RC.calibrated_camera(s).observation == RC.observation_model(s)
    # a state estimated on RAW photosites under a record continues (or renders) no run of decoded photographs
    raw = RC.ObservationModel(RC.SensorRecord(repeat("ab", 32)), RC.Photosites(), :G)
    sst = (; observation = raw, names = ["V01.png"], poses, camera, target = RC.radiometric_target(s), light)
    err = try RC.load_state_into!(s, sst, ["V01.png"]); nothing catch e; e end
    @test err isa ArgumentError && occursin(repeat("ab", 32), err.msg) && occursin("decoded images", err.msg)
    @test s.camera == camera                                                     # nothing moved
    # and a state's own observation model loads
    RC.load_state_into!(s, merge(sst, (; observation = RC.observation_model(s))), ["V01.png"])
    # a RAW state's photosites are channel G (the state refuses another)
    @test_throws "ONE observation each" RC.CalibrationState(RC.Scene(board, W, H), settings, camera, poses, mat, light, [zeros(H, W, 3)];
                                                           photo_kind = RC.Photosites(), channel = 0)
end

@testitem "observation model: the dark offset a sensor record subtracts — zero mean residual through the entry point, ln-bias δ/s without it" setup = [ObservationFixtures] begin
    # a dark rim of s = 40 codes above a dark offset 0.4 codes over the file's black 512 (the FX3's measured offsets, the audit's §1):
    # the photosites under the record's Dark read s exactly (zero mean residual); under the file's black (no record: the FileBlack
    # decode) they read s + δ, a bias of ln((s + δ)/s) ≈ δ/s = 1 % on log I — the inconsistency the observation model removes
    s_codes = 40.0; δ = 0.4; H = W = 10
    codes = [UInt16(512 + 40 + (mod(i + 2j, 5) < 2 ? 1 : 0)) for i in 1:H, j in 1:W]     # 2 of 5 at +41: the mean 552.4
    md = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing, iso = nothing,
          model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
    m = VgtkCore.RawMosaic(codes, UInt8[0 1; 3 2], (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (H, W), md)
    sec = VgtkCore.parse_unit("s")
    rec = VgtkCore.sensor_fixture(; body = FX3, exposure = 1.0sec, black = (512.0, 512.0, 512.0, 512.0),
                                  dark = (; offset = fill(Float32(512 + δ), H, W), variance = fill(0.1f0, H, W), exposure_band = (0.5sec, 2.0sec), frames = 50)).model
    under_record, mask = RC._observe(RC.Photosites(), m, trues(H, W), "rim", rec)
    file_black, _ = RC._observe(RC.Photosites(), m, trues(H, W), "rim", VgtkCore.SensorModel())
    g = findall(mask)
    @test !isempty(g) && sum(mask) == count(c -> c == 0x01 || c == 0x03, VgtkCore.photosite_colors(m))
    span_rec = 16383.0 - (512 + δ); span_file = 16383.0 - 512
    mean_rec = sum(VgtkCore.value_of(under_record[i]) * span_rec for i in g) / length(g)
    mean_file = sum(file_black[i] * span_file for i in g) / length(g)
    mean_codes = sum(Float64(codes[i]) for i in g) / length(g)
    @test isapprox(mean_rec, mean_codes - 512 - δ; atol = 1e-2)                        # the record's black subtracted (its Float32 offset; ≪ δ)
    @test isapprox(mean_file, mean_codes - 512; atol = 1e-3)                           # the file's: δ codes too bright
    @test isapprox(log(mean_file / mean_rec), δ / mean_rec; rtol = 0.02)               # ln-bias ≈ δ/s
end

@testitem "observation model: the entry point decodes a real RAW under the calibration's own record (VGTK_RAW_TEST_FILE)" setup = [ObservationFixtures] begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    f = get(ENV, "VGTK_RAW_TEST_FILE", "")
    if isempty(f) || !isfile(f)
        @test_skip isfile(f)                                   # no RAW file here: recorded as Broken, not as a success
    else
        Base.require(Main, :LibRaw_jll)
        mktempdir() do dir
            p = joinpath(dir, "frame_1" * lowercase(last(splitext(f)))); symlink(f, p)
            capture = Base.invokelatest(VgtkFeatures.extract_capture_settings, Dict(1 => Base.invokelatest(VgtkFeatures.load_raw_metadata, p)))
            iso = Int(round(capture.frames[1].exposure.iso))
            rec = joinpath(dir, "own.json"); sha = write_characterisation(rec; iso, body = capture.body)
            other = joinpath(dir, "other_iso.json"); write_characterisation(other; iso = iso + 100, body = capture.body)
            om = RC.ObservationModel(RC.SensorRecord(sha), RC.Photosites(), :G)
            set = Base.invokelatest(RC.photo_set, om, [p]; characterisation = rec)
            c = VgtkIO.load(VgtkCore.SensorCharacterisation, rec)
            @test set.models[1] == VgtkCore.sensor_model(c, capture, 1) && VgtkCore.component_sources(set.models[1]).reason == VgtkCore.Matched
            mosaic = Base.invokelatest(VgtkFeatures.load_raw_mosaic, p)
            vals, mask = Base.invokelatest(RC.observe, set, 1, trues(size(mosaic)))
            @test vals == VgtkCore.linearview(mosaic; sensor_model = set.models[1]) && mask == RC.green_photosite_mask(mosaic)
            # the same photograph under a record of another ISO: refused (sha first — another file — then the frame it does not describe)
            @test_throws "given sha256" Base.invokelatest(RC.photo_set, om, [p]; characterisation = other)
            om2 = RC.ObservationModel(RC.SensorRecord(open(VgtkCore.sha256_hex, other)), RC.Photosites(), :G)
            @test_throws "(UnmatchedISO)" Base.invokelatest(RC.photo_set, om2, [p]; characterisation = other)
        end
    end
end

# `camera.sensor_model` (RC 0.11.2, owner 2026-11-15): "auto" (the default) uses a record present and runs without one; "on" requires
# one; "off" does not use a record present, with the loud once-per-run warning. Present = `camera.sensor_characterisation` names a
# record on disk, or the camera the run continues or predicts states one. A named path that does not exist, the retired "none", any
# other value, and every other mismatch (another sha) stay refused.
@testitem "observation model: sensor_model auto / on / off, the six cells and the guards" setup = [ObservationFixtures] begin
    using Test
    mktempdir() do dir
        a = joinpath(dir, "a.sensor_characterisation.json"); b = joinpath(dir, "b.sensor_characterisation.json")
        sha_a = write_characterisation(a); sha_b = write_characterisation(b; black = (512.5, 512.4, 512.5, 512.4))
        A = RC.SensorRecord(sha_a); raw = ["DSC1.ARW", "DSC2.ARW"]
        auto, on, off = RC.AutoSensorModel(), RC.OnSensorModel(), RC.OffSensorModel()
        @test RC.sensor_model_mode.(("auto", "on", "off")) == (auto, on, off) && RC.PipelineOptions().camera.sensor_model == "auto"
        @test_throws "is not one of \"auto\", \"on\", \"off\"" RC.sensor_model_mode("none")
        @test_throws "is not one of" RC.pipeline_options(Dict("camera" => Dict("sensor_model" => "yes")))
        @test RC.pipeline_options(Dict("camera" => Dict("sensor_model" => "off"))).camera.sensor_model == "off"
        # ESTIMATION: the run's observation model by mode × record present
        est(m, spec; stated = RC.NoSensorRecord()) = RC.observation_model(raw, spec; sensor_model = m, stated).sensor
        @test (@test_logs est(auto, a)) == A                                                           # auto + record: used
        @test (@test_logs (:info, r"no sensor characterisation record is named") est(auto, "")) === RC.NoSensorRecord()   # auto + none: runs
        @test (@test_logs est(on, a)) == A                                                             # on + record: used
        @test_throws "requires a sensor model" est(on, "")                                             # on + none: refused
        @test (@test_logs (:warn, Regex("SENSOR MODEL OFF(.|\\n)*$(sha_a)(.|\\n)*sensor_model = \"off\"")) est(off, a)) === RC.NoSensorRecord()   # off + record
        @test (@test_logs (:warn, Regex(sha_a)) est(off, ""; stated = A)) === RC.NoSensorRecord()       # off + the continued camera's record
        @test (@test_logs est(off, "")) === RC.NoSensorRecord()                                         # off + none: runs, no warning
        @test_throws "name its location" est(auto, ""; stated = A)                                      # the camera's record needs its location
        # the guards: a named path that does not exist, the retired "none" (naming the key), decoded images under a record or "on"
        @test_throws "no file" est(auto, joinpath(dir, "typo.json"))
        @test_throws "no file" est(off, joinpath(dir, "typo.json"))
        @test_throws "sensor_model = \"off\"" est(off, "none")
        @test_throws "sensor_model = \"off\"" RC.pipeline_options(Dict("camera" => Dict("sensor_characterisation" => "none")))
        @test_throws "decoded images" RC.observation_model(["a.png"], a; channel = "RGB", sensor_model = off)
        @test_throws "decoded images" RC.observation_model(["a.png"], ""; channel = "RGB", sensor_model = on)
        @test RC.sensor_model_record(off, a, RC.NoSensorRecord(), RC.observation_model(raw, a; sensor_model = off)) ==
              (; mode = "off", used = false, present = (; path = a, sha256 = sha_a))                     # the records state it
        @test RC.sensor_model_record(auto, "", RC.NoSensorRecord(), RC.observation_model(raw, "")) == (; mode = "auto", used = false, present = nothing)
        # PREDICTION: a camera calibrated under record a, and one without a record
        om_a = RC.ObservationModel(A, RC.Photosites(), :G); om_0 = RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :G)
        @test RC.prediction_observation(auto, om_a, a) == (om_a, (; characterisation = a))              # auto + record: used
        @test RC.prediction_observation(on, om_a, a) == (om_a, (; characterisation = a))                # on + record: used
        @test RC.prediction_observation(auto, om_0, nothing) == (om_0, (; characterisation = nothing))  # auto + none: runs
        @test_throws "no sensor model to predict with" RC.prediction_observation(on, om_0, nothing)    # on + none: refused
        @test (@test_logs (:warn, Regex("SENSOR MODEL OFF(.|\\n)*$(sha_a)")) RC.prediction_observation(off, om_a, nothing)) ==
              (om_0, (; characterisation = nothing))                                                    # off + the camera's record: not used
        @test (@test_logs (:warn, Regex(sha_a)) RC.prediction_observation(off, om_a, a)) == (om_0, (; characterisation = nothing))
        @test (@test_logs (:warn, Regex(sha_b)) RC.prediction_observation(off, om_0, b)) == (om_0, (; characterisation = nothing))   # a record on disk, not used
        @test (@test_logs RC.prediction_observation(off, om_0, nothing)) == (om_0, (; characterisation = nothing))   # off + none
        # every other mismatch stays refused: another sha (auto and off), a path that does not exist
        cam_a = RC.CalibratedCamera(small_model(), om_a, RC.RenderSettings())
        err = try RC.photo_set(cam_a, raw; characterisation = b); nothing catch e; e end
        @test err isa ArgumentError && occursin(sha_a, err.msg) && occursin(sha_b, err.msg)
        @test_throws sha_b RC.prediction_observation(off, om_a, b)
        @test_throws "no file" RC.prediction_observation(off, om_a, joinpath(dir, "typo.json"))
        # a state estimated under record a continues an "off" run reading the same photographs without it; under "auto" refused
        @test RC._continues(off, om_a, om_0) && !RC._continues(auto, om_a, om_0) && !RC._continues(off, om_a, RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :G))
    end
end
