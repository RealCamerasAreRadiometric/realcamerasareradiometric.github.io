# A session's camera settings from its RAW files (`load_session(…; raw_dir, lens_identity)`): the RAW file of each image found by its
# stem, the options `io.raw` and `io.lens_identity`, and the log line; with VGTK_RAW_TEST_FILE, a real RAW's EXIF.
@testitem "capture settings: RAW lookup, options and the log line" setup = [RCFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures

    @testset "the RAW file of an image: its stem, a RAW extension of any case, exactly one" begin
        mktempdir() do dir
            for f in ("IMG_1.ARW", "IMG_2.cr2", "IMG_2.dng", "IMG_4.jpg", "IMG_5.Nef")
                touch(joinpath(dir, f))
            end
            @test RC.raw_file(dir, "IMG_1.png") == joinpath(dir, "IMG_1.ARW")
            @test RC.raw_file(dir, "sub/IMG_5.tif") == joinpath(dir, "IMG_5.Nef")          # the image's stem, not its directory
            @test_throws ArgumentError RC.raw_file(dir, "IMG_2.png")                         # two RAW files of one image
            @test_throws ArgumentError RC.raw_file(dir, "IMG_3.png")                         # none
            @test_throws ArgumentError RC.raw_file(dir, "IMG_4.png")                         # a JPEG is no RAW file
            e = try RC.raw_file(dir, "IMG_3.png") catch err err end
            @test occursin("IMG_3.png", e.msg)
        end
    end

    @testset "the options: io.raw a path key, io.lens_identity a table" begin
        o = RC.pipeline_options(Dict("io" => Dict("raw" => "raw/board", "lens_identity" => Dict("make" => "Sigma", "model" => "14mm F1.4 DG DN"))))
        @test first(o).io.raw == "raw/board" && "io.raw" in RC.PIPELINE_PATH_KEYS
        @test RC.lens_identity(first(o).io.lens_identity) == VgtkCore.DeviceIdentity("Sigma", "14mm F1.4 DG DN", nothing)
        @test RC.pipeline_options(RC.config_dict(o...)) == o                                   # the writer's table reads back
        @test RC.RunOptions().io.lens_identity === nothing                              # absent: EXIF's lens
        @test !haskey(RC.config_dict(RC.RunOptions(), RC.CalibrationSettings())["io"], "lens_identity")
        @test_throws ArgumentError RC.pipeline_options(Dict("io" => Dict("lens_identity" => Dict{String, Any}())))
        @test_throws ArgumentError RC.pipeline_options(Dict("io" => Dict("lens_identity" => Dict("brand" => "Sigma"))))
        @test_throws ArgumentError RC.pipeline_options(Dict("io" => Dict("lens_identity" => "Sigma")))
    end

    @testset "the log line: constant exposure, or the ranges" begin
        sec = VgtkCore.parse_unit("s")
        ls = VgtkCore.LensSettings(14VgtkCore.mm)
        frame(t, iso) = (exposure = VgtkCore.ExposureSettings(t * sec, 4.0, iso), lens_settings = ls)
        body = VgtkCore.DeviceIdentity("SONY", "ILME-FX3", "123")
        lens = VgtkCore.DeviceIdentity("Sigma", "14mm F1.4 DG DN", nothing)
        cs = VgtkCore.CaptureSettings(Dict(1 => frame(1 / 60, 100.0), 2 => frame(1 / 60, 100.0)); body, lens)
        @test RC.settings_line(cs; lens_stated = true) ==
              "2 frames; body SONY ILME-FX3 #123; lens Sigma 14mm F1.4 DG DN (config); exposure constant 1/60 s f/4 ISO 100"
        cv = VgtkCore.CaptureSettings(Dict(1 => frame(1 / 60, 100.0), 2 => frame(2.0, 400.0)); body, lens)
        @test endswith(RC.settings_line(cv), "(EXIF); exposure 1/60 s–2 s, f/4, ISO 100–400")
    end

    @testset "a real RAW's EXIF (VGTK_RAW_TEST_FILE)" begin
        f = get(ENV, "VGTK_RAW_TEST_FILE", "")
        if isempty(f) || !isfile(f)
            @test_skip isfile(f)                       # no RAW file here: recorded as Broken, not as a success
        else
            Base.require(Main, :LibRaw_jll)            # Features' RAW decoder (its package extension), a test dependency
            mktempdir() do dir
                p = joinpath(dir, "frame_7" * lowercase(last(splitext(f))))
                symlink(f, p)
                @test RC.raw_file(dir, "frame_7.png") == p
                cs = VgtkFeatures.extract_capture_settings(Dict(7 => Base.invokelatest(VgtkFeatures.load_raw_metadata, p)))
                @test collect(keys(cs)) == [7] && startswith(RC.settings_line(cs), "1 frame; body ")
            end
        end
    end
end
