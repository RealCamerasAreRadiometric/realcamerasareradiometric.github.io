@testitem "io.channel: one colour channel, one residual per measured pixel" setup = [RCFixtures] begin
    img = reshape(Float32.(1:24), 2, 4, 3)                        # H × W × 3; channel c holds the values 8(c − 1) + 1 … 8c
    @test RC._io_channel("RGB") == 0 && RC._io_channel("R") == 1 && RC._io_channel("G") == 2 && RC._io_channel("B") == 3
    @test_throws ArgumentError RC._io_channel("X")
    px = [(1, 1), (3, 2), (4, 1)]                                 # (u, v)
    # RGB: three entries per pixel, channel fastest; a single channel: ONE entry per pixel (no copy into three)
    @test RC.masked_vector(img, px) == Float64[img[v, u, c] for (u, v) in px for c in 1:3]
    @test RC.masked_vector(img, px; channel = 2) == Float64[img[v, u, 2] for (u, v) in px]
    @test RC.masked_vector(img[:, :, 2], px; channel = 2) == RC.masked_vector(img, px; channel = 2)   # a plane is one channel
    @test_throws ArgumentError RC.masked_vector(img[:, :, 2], px)
    @test RC._residual_channels(0) == (1, 2, 3) && RC._residual_channels(2) == (2,)
    @test RC._channel_rows(collect(1.0:9.0), 0) == collect(1.0:9.0) && RC._channel_rows(collect(1.0:9.0), 2) == [2.0, 5.0, 8.0]
    # the option is validated with the others, and the key is a string option that names no path
    @test first(RC._check_options(RC.RunOptions(; io = RC.IOOptions(; channel = "G")), RC.CalibrationSettings())).io.channel == "G"
    @test_throws ArgumentError RC._check_options(RC.RunOptions(; io = RC.IOOptions(; channel = "g")), RC.CalibrationSettings())
    @test "io.channel" in RC._NOT_PATH_KEYS
end
