# The refinement's route by the camera's TYPE (the dispatch-not-flags rule, RC 0.11.0): a SplineLens camera is refined per tile
# (_lm_tiles!), any other pixel camera per view in its own parameters (_lm_views!). No Bool chooses it: `lm_route(camera)` is the one
# decision, and the problem spec dispatches on the route.

@testitem "LM route: by the camera's type; the problem spec dispatches on it" setup = [RCFixtures] begin
    using RadiometricCamera: focus, _image_start
    using Test
    W, H = 640, 480
    pc = cam(VgtkCore.Division(-0.15, 0.03), 540.0, (326.4, 230.4), W, H)
    gc = _image_start(pc, (35, 25)).camera
    @test RC.lm_route(pc) === RC.PerViewRoute() && RC.lm_route(gc) === RC.PerTileRoute()
    @inferred RC.lm_route(pc); @inferred RC.lm_route(gc)
    o = RC.pipeline_options(Dict("fixed" => ["camera", "focus"]))
    @test RC.refinement_structure(last(o), RC.PerViewRoute()).kw.fixed == (:distortion, :focus)     # the per-view route's lens block
    @test RC.refinement_structure(last(o), RC.PerTileRoute()).kw.fixed == (:grid, :focus)
    @test haskey(RC.refinement_structure(last(o), RC.PerTileRoute()).kw, :height_degree)
    @test_throws MethodError RC.refinement_structure(last(o), true)                                 # the retired Bool: refused
    @test !hasfield(typeof(RC.refinement_structure(last(o), RC.PerViewRoute())), :parametric)
end
