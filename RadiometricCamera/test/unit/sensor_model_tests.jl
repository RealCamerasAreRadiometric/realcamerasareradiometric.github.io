# The camera's sensor characterisation on the calibration state (owner order 2026-11-14; spec 2026-11-13-sensor-noise, rulings D1–D14):
# `camera.sensor_characterisation` names an IO record (or none: empty), each RAW photograph's Core `SensorModel` is resolved from it
# by its EXIF capture settings (Core's resolver, a lookup), and the run record states each photograph's reason. The photographs'
# black is the model's and its defects are masked (Core's defect mask). Fixtures through Core's ONE builder, `sensor_fixture` (D13).

@testsetup module SensorFixtures
    using RadiometricCamera, Test
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    const PAT = UInt8[0 1; 3 2]                                   # RGGB
    "A flat synthetic RGGB mosaic (`H × W`, every photosite `code`), black 512 per channel, white 16383."
    function flat_mosaic(H, W, code)
        md = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing, iso = nothing,
              model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
        return VgtkCore.RawMosaic(fill(UInt16(code), H, W), PAT, (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (H, W), md)
    end
    export RC, VgtkCore, flat_mosaic
end

@testitem "sensor model: the characterisation key, its fallback and its refusals" setup = [RCFixtures, SensorFixtures] begin
    o = RC.pipeline_options(Dict("camera" => Dict("sensor_characterisation" => "sensor/fx3.sensor_characterisation.json")))
    @test o.camera.sensor_characterisation == "sensor/fx3.sensor_characterisation.json"
    @test RC.pipeline_options(RC.config_dict(o)) == o
    @test RC.PipelineOptions().camera.sensor_characterisation == ""                       # not stated
    @test "camera.sensor_characterisation" in RC.PIPELINE_PATH_KEYS
    @test RC.names_input_path("camera.sensor_characterisation", "sensor/fx3.json")
    @test !RC.names_input_path("camera.sensor_characterisation", "")
    @test_throws "sensor_model = \"off\"" RC.sensor_statement("none")                  # retired (RC 0.11.2)
    @test_throws "names no record" RC.sensor_statement("")
    # decoded images: no sensor record, the absent model each; a record named for them is refused
    files = ["a.png", "b.png"]
    for spec in ("",)
        om = RC.observation_model(files, spec; channel = "RGB")
        @test om == RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB)
        @test all(==(VgtkCore.SensorModel()), RC.photo_set(om, files).models)
    end
    rec = tempname(); write(rec, "a record")
    @test_throws "decoded images" RC.observation_model(files, rec)
    @test_throws "no file" RC.observation_model(files, "x.json")                         # a named path that does not exist
    @test_throws "mix RAW files and decoded images" RC.observation_model(["a.ARW", "b.png"], "")
    @test RC.observation_model(["a.ARW"], "") == RC.ObservationModel(RC.NoSensorRecord(), RC.Photosites(), :G)   # none named: the file's black (auto)
    @test_throws "ONE observation each" RC.observation_model(["a.ARW"], ""; channel = "RGB")
    @test_throws "a RAW photograph is linear" RC.observation_model(["a.ARW"], ""; linearize = true)
    # the retired names are refused by name with the entry point
    @test_throws "photo_set" RC.photo_sensor_models(files, "")
    @test_throws "photo_set" RC.photo_observations(nothing, nothing, "x")
    @test_throws "sensor_statement" RC.sensor_characterisation("none")
end

@testitem "sensor model: the characterisation through IO, resolved per frame, recorded" setup = [RCFixtures, SensorFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    f = VgtkCore.sensor_fixture(; body = VgtkCore.DeviceIdentity(; make = "Sony", model = "ILME-FX3"), iso = 100,
                                black = (512.3, 512.2, 512.3, 512.2), gain = 11.33, read_noise = (1.09, 1.10, 1.03, 1.10),
                                defects = (; x = [3], y = [2], onset = [0.5VgtkCore.parse_unit("s")], size = (8, 8)),
                                source = VgtkCore.SensorSource.measured)
    mktempdir() do dir
        p = joinpath(dir, "fx3.sensor_characterisation.json")
        VgtkIO.save(p, f.characterisation)
        c = VgtkIO.load(VgtkCore.SensorCharacterisation, p)                                # IO's reader, the one path
        @test c == f.characterisation
        m = VgtkCore.sensor_model(c, f.capture, 1)
        @test m == f.model && VgtkCore.component_sources(m).reason == VgtkCore.Matched
        rec = only(RC.sensor_record(["IMG_1.ARW"], [m]))
        @test rec.view == "IMG_1.ARW" && rec.reason == "Matched" && rec.black == "measured" && rec.noise == "measured" &&
              rec.defects == "measured" && rec.specified_override === false && rec.digest == VgtkCore.content_digest(m)
        @test rec.black_component == "ChannelBlack"
        rec0 = only(RC.sensor_record(["x.png"], [VgtkCore.SensorModel()]))
        @test rec0.reason == "NoCharacterisation" && rec0.noise === nothing && rec0.black == "file" && rec0.black_component == "FileBlack"
    end
end

@testitem "sensor model: a RAW photograph's black is the model's and its defects are masked" setup = [RCFixtures, SensorFixtures] begin
    s = VgtkCore.parse_unit("s")
    m = flat_mosaic(8, 8, 1512)
    f = VgtkCore.sensor_fixture(; black = (502.0, 492.0, 482.0, 472.0), defects = (; x = [2, 5], y = [1, 4], onset = [0.1s, 0.1s], size = (8, 8)))
    vals0, mask0 = RC._observe(RC.Photosites(), m, trues(8, 8), "absent", VgtkCore.SensorModel(), :G)
    vals, mask = RC._observe(RC.Photosites(), m, trues(8, 8), "measured", f.model, :G)
    col = VgtkCore.photosite_colors(m)
    @test vals0 == VgtkCore.linearview(m)                                                  # the absent model: the file's black, bit for bit
    @test vals == VgtkCore.linearview(m; sensor_model = f.model)                           # the model's black (Core's one linearisation)
    @test vals[1, 2] != vals0[1, 2]
    # the defect sites (x = column, y = row) are out of the fit; every other usable green photosite stays
    green = (col .== 0x01) .| (col .== 0x03)
    @test mask0 == green
    expected = copy(green); expected[1, 2] = false; expected[4, 5] = false              # both defect sites are greens (RGGB)
    @test green[1, 2] && green[4, 5] && mask == expected
end

@testitem "sensor model: a real RAW's model from its EXIF (VGTK_RAW_TEST_FILE)" setup = [RCFixtures, SensorFixtures] begin
    f = get(ENV, "VGTK_RAW_TEST_FILE", "")
    if isempty(f) || !isfile(f)
        @test_skip isfile(f)                                   # no RAW file here: recorded as Broken, not as a success
    else
        Base.require(Main, :LibRaw_jll)
        mktempdir() do dir
            p = joinpath(dir, "frame_1" * lowercase(last(splitext(f)))); symlink(f, p)
            ms = Base.invokelatest(RC.photo_set, RC.observation_model([p], ""), [p]).models
            @test VgtkCore.component_sources(only(ms)).reason == VgtkCore.NoCharacterisation
        end
    end
end

@testitem "sensor model: the black is Core's linearisation of the resolved model — a Dark inside its band, its channel black outside" setup = [RCFixtures, SensorFixtures] begin
    s = VgtkCore.parse_unit("s")
    m = flat_mosaic(8, 8, 1512)
    off = Float32[500 + i + 8j for i in 1:8, j in 1:8]; var = fill(1.5f0, 8, 8)
    dark(t) = VgtkCore.sensor_fixture(; exposure = t, black = (502.0, 492.0, 482.0, 472.0),
                                      dark = (; offset = off, variance = var, exposure_band = (0.5s, 2.0s), frames = 50))
    inband = dark(1.0s).model; outband = dark(4.0s).model
    @test VgtkCore.component_sources(inband).reason == VgtkCore.Matched && VgtkCore.component_sources(outband).reason == VgtkCore.DarkOutOfBand
    @test only(RC.sensor_record(["a"], [inband])).black_component == "Dark" && only(RC.sensor_record(["b"], [outband])).black_component == "ChannelBlack"
    vi = RC._observe(RC.Photosites(), m, trues(8, 8), "a", inband, :G)[1]; vo = RC._observe(RC.Photosites(), m, trues(8, 8), "b", outband, :G)[1]
    @test vi == VgtkCore.linearview(m; sensor_model = inband) && vo == VgtkCore.linearview(m; sensor_model = outband)   # Core's one black rule
    @test vi[1, 1] != vi[3, 1] && vo[1, 1] == vo[3, 1]                     # per photosite inside the band; per colour outside it
end

@testitem "sensor model: a windowed mosaic's defects are read at its window (the map covers the visible raw)" setup = [RCFixtures, SensorFixtures] begin
    s = VgtkCore.parse_unit("s")
    md = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing, iso = nothing,
          model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
    # a 4 × 4 window at (2, 2) of an 8 × 8 visible raw; the defects at visible (row 3, col 4) and (4, 5) → window (1, 2) and (2, 3)
    mw = VgtkCore.RawMosaic(fill(UInt16(1512), 4, 4), UInt8[0 1; 3 2], (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (8, 8), md;
                            window_origin = (2, 2))
    f = VgtkCore.sensor_fixture(; black = (502.0, 492.0, 482.0, 472.0), defects = (; x = [4, 5], y = [3, 4], onset = [0.1s, 0.1s], size = (8, 8)))
    _, mask0 = RC._observe(RC.Photosites(), mw, trues(4, 4), "absent", VgtkCore.SensorModel(), :G)
    _, mask = RC._observe(RC.Photosites(), mw, trues(4, 4), "windowed", f.model, :G)
    expected = copy(mask0); expected[1, 2] = false; expected[2, 3] = false
    @test mask == expected
end
