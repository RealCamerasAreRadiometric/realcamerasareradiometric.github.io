@testitem "self-check needs no photographs" begin
    const RC = RadiometricCamera
    using PNGFiles, ImageCore
    mktempdir() do dir
        schoeps = mkpath(joinpath(dir, "schoeps")); images = mkpath(joinpath(dir, "images"))
        # a Schöps run's model (a 4 × 4 grid of directions over the 640 × 480 image), read by VisualGeometryToolkitIO
        grid = join((string(x) for i in 1:16 for x in (0.1 * (mod1(i, 4) - 2.5), 0.1 * (cld(i, 4) - 2.5), 1.0)), ", ")
        write(joinpath(schoeps, "intrinsics0.yaml"), "# Schöps run\ntype : CentralGenericModel\nwidth : 640\nheight : 480\n" *
              "calibration_min_x : 0\ncalibration_min_y : 0\ncalibration_max_x : 639\ncalibration_max_y : 479\ngrid_width : 4\ngrid_height : 4\ngrid : [$grid]\n")
        write(joinpath(images, "V01.png"), "not a PNG")          # a photograph that must not be opened by a self-check
        opts(; schoeps = "", images = "", out = "", only = false) =
            RC.RunOptions(; io = RC.IOOptions(; images, out), frontend = RC.FrontendOptions(; type = RC.SchoepsFrontend, schoeps),
                          selfcheck = RC.SelfcheckOptions(; only))
        # the start camera's size, the photograph never opened (review finding 10: a refinement decoded the first photograph, a whole
        # RAW, only to compare its size): a self-check and a refinement alike
        sc = opts(; schoeps, images, only = true)
        @test RC._schoeps_extent(sc, nothing) == (640, 480)                         # the yaml, the photograph ("not a PNG") untouched
        @test RC._schoeps_extent(opts(; schoeps, images), nothing) == (640, 480)
        # without a start camera there is nothing to read the size from: never the photograph instead
        @test_throws "no start camera" RC._schoeps_extent(opts(; schoeps = images, only = true), nothing)
        # a stored start (macro_defocus' held-out self-check, no Schöps run's model; `load_start`, read once by the front end): its size
        none = mkpath(joinpath(dir, "no_model"))
        lens = RC.save_lens(joinpath(dir, "start.json"), RC.load_camera("schoeps:" * joinpath(schoeps, "intrinsics0.yaml")))
        po = RC.RunOptions(; io = RC.IOOptions(; images, out = "", init = "lens:" * joinpath(dir, "start.json")),
                           frontend = RC.FrontendOptions(; type = RC.SchoepsFrontend, schoeps = none))
        @test RC._schoeps_extent(po, RC.load_start(po.io.init)) == (640, 480)
        # the start: the stored one when io.init names it, and then no Schöps run's model is read (review round 2, finding 2: the Schöps
        # start was built eagerly, failing a stored start's self-check whose Schöps folder holds no model)
        st = RC._schoeps_frontend_start(po, RC.load_start(po.io.init), 640, 480)
        @test st.init_region === nothing && RC.VgtkCore.extent(st.lens) == RC.VgtkCore.extent(RC.load_start(po.io.init).lens)
        @test_throws "no Schöps start camera" RC._schoeps_frontend_start(po, nothing, 640, 480)     # no stored start: the Schöps run's model, absent here
        # the size check is where each photograph is read (`observe`, its one decode): a photograph that is not the start
        # camera's size is refused (never one source for the other); the folder's file by `photo_file`, a missing one refused
        good = mkpath(joinpath(dir, "good")); bad = mkpath(joinpath(dir, "bad"))
        PNGFiles.save(joinpath(good, "V01.png"), zeros(Gray{N0f16}, 480, 640)); PNGFiles.save(joinpath(bad, "V01.png"), zeros(Gray{N0f16}, 240, 320))
        mask = trues(480, 640)
        dec = RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB)
        @test RC.observe(RC.photo_set(dec, [RC.photo_file(good, "V01.png")]), 1, mask)[2] == mask
        @test_throws "refusing" RC.observe(RC.photo_set(dec, [RC.photo_file(bad, "V01.png")]), 1, mask)
        @test_throws "no photograph" RC.photo_file(none, "V01.png")
        # io.images is required by the start's photographs only: a self-check reads none and gets past to the front-end's own checks
        out = mkpath(joinpath(dir, "out"))
        e = try
            RC.selfcheck(RC.start_geometry(opts(; out, only = true), RC.CalibrationSettings()); run = opts(; out, only = true)); nothing
        catch err
            err
        end
        @test e isa ErrorException && occursin("frontend.schoeps required", e.msg)
        e2 = try
            RC.start_photographs(opts(; out), ["V01.png"]); nothing
        catch err
            err
        end
        @test e2 isa ErrorException && occursin("io.images required", e2.msg)
    end
end
