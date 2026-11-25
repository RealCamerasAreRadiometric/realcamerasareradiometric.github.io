# The initialization region's consumers (owner ruling 2026-11-13): every step between the start camera and the refinement that
# evaluates the start at the detections takes the initialization region EXPLICITLY — the detections it evaluates the start at are those
# inside it (`in_initialization`) — never the re-gridded lens's sensor-wide domain; the refinement's masks, present nodes, grid
# smoothness and the focus surface's nodes keep the refinement's (imaged) region: the focus surface describes the optics over what is
# imaged (2026-11-15: laid over the initialization region, the fisheye f/4's border nodes ran to infinite focus). A structural check of the start (src/start/, src/diagnostics/selfcheck.jl): the
# calls are named, so a new consumer on the whole sensor fails here.
@testitem "initialization region: every initialization consumer takes the initialization region" begin
    using RadiometricCamera: bearing_px, BlobFrontend, in_initialization, initialization_region, SchoepsFrontend,
                             vignetting_center
    using Test, RadiometricCamera
    src = join((read(joinpath(pkgdir(RadiometricCamera), "src", f), String) for f in ("start/prepare.jl", "start/calibration_start.jl", "start/frontends/calibration_session.jl", "start/frontends/schoeps_run.jl", "diagnostics/selfcheck.jl", "io/run_outputs.jl")), "\n")
    body(name) = (a = findfirst(name, src); a === nothing && error("no $name"); b = findnext(r"\nend\n", src, last(a)); src[first(a):first(b)])
    pre = body("function _start_state(")
    fronts = [body("function _frontend(::Type{BlobFrontend}"), body("function _frontend(::Type{SchoepsFrontend}")]
    sc = body("function _write_selfcheck(")
    # the front ends: the start's initialization region carried out; every pose fit and self-check on the detections inside it
    for f in fronts
        @test occursin("lens0, init_region, W₀, H₀", f)
        @test !occursin(r"_imaged\(lens0, (?!in_initialization)", f)
        @test !occursin(r"bearing_px\(lens0, poses0\[k\], fs\[k\]", f)
    end
    @test occursin("features = [in_initialization(f, g.init_region) for f in g.observations]", sc)
    # the start: the self-check on the initialization region …
    @test occursin("be0 = _start_bearings(g, fpx)", pre) && occursin("detection_bearings(g.camera, g.poses, [in_initialization(g.observations[k], g.init_region) for k in eachindex(g.names)], fpx)", body("_start_bearings(g::StartGeometry, fpx) ="))
    # … the focus start's nodes, support and box on the imaged region (`focus_start_nodes`), never the initialization region
    @test occursin("= focus_start_nodes(lens0, opts.focus)", pre) && !occursin("region_i", pre)
    # the vignetting centre is the start's principal point, no region of its own (RC 0.11.0: the start is on its optical axis)
    @test occursin("vignetting_center(lens0)", pre)
    # … and the refinement's own region for its masks and the grid
    @test occursin("region = VgtkCore.imaged_region(lens0)", pre) && occursin("VgtkCore.pixel_mask(_sampling_region(lens0, opts.masks.lens_coverage_scale),", pre)
    # the initialization region has ONE record, beside the grids (`_save_run_grid`, review round 2 finding 4): parameters.json holds none
    @test occursin("_save_run_grid(joinpath(opts.io.out, \"grid.json\"), gfin, meta.init_region;", src)
    @test !occursin("initialization_region = meta.init_region", src) && !occursin("initialization_region_unrecorded", src)
end
