# The printed design's edge distance map through VisualGeometryToolkitIO (owner D6, 2026-11-15): a vgio/raster record on the page
# chart, read by `load_distance_map` — its values bit for bit and the page's size in mm; another frame, an offset page or a missing
# file refused by name. (The generator's raw sdf.f32 is converted by the experiments' scripts/distance_maps.jl, a keyed step.)

@testitem "distance map: a vgio/raster record on the page chart, bit for bit" begin
    using RadiometricCamera: chart
    using Test, RadiometricCamera
    import VisualGeometryToolkitIO as VgtkIO, VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    d = mktempdir()
    v = Float32[(-1)^(i + j) * (i + 0.1f0 * j) for i in 1:4, j in 1:6]; v[2, 3] = prevfloat(1.0f0)
    page = 999.9999999999999                                                   # the Siemens generator's page, as its header states it
    p = VgtkIO.save_raster(joinpath(d, "distance_map.json"), v; frame = (chart = :page, area = VgtkCore.Rect2(0.0, 0.0, page / 1000, page / 1000)), nan = :none)
    sdf, page_mm = RC.load_distance_map(p)
    @test sdf == v && isequal(sdf, v) && eltype(sdf) === Float32 && page_mm === (page, page)
    q = VgtkIO.save_raster(joinpath(d, "px.json"), v; frame = VgtkCore.PixelFrame{:colmap}, nan = :none)
    @test_throws "on the page chart" RC.load_distance_map(q)
    o = VgtkIO.save_raster(joinpath(d, "off.json"), v; frame = (chart = :page, area = VgtkCore.Rect2(0.1, 0.0, 1.0, 1.0)), nan = :none)
    @test_throws "lower-left corner" RC.load_distance_map(o)
    @test_throws "no such file" RC.load_distance_map(joinpath(d, "sdf.f32"))
end
