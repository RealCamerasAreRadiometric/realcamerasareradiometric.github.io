# The focus surface's nodes follow the IMAGED region, never the start's initialization region (Ties, 2026-11-15: since RC
# 0.9.0, the nodes and their support were laid over the initialization region ∩ the imaged region; on the fisheye f/4 the clipped
# box put four nodes on the region's border, seen by few pixels, which ran to infinite focus), kept within a margin of an image circle's
# rim by design, and a focus node of the support run to t̂ ≤ 0 or non-finite flags the calibration non-converged, as a fold does.

@testitem "focus start nodes: the imaged region's grid within the rim margin, never the initialization region" setup = [RCFixtures] begin
    using StaticArrays
    W, H = 520, 348
    # a synthetic fisheye: the blob board's Kannala–Brandt fit (its fold at 133°), f = 93.2 px, its image circle inside the sensor
    kb = VgtkCore.KannalaBrandt(-0.041634590670699925, 0.00040687025248368753, 0.00018535670662213487, -7.735440193838873e-5)
    fish = cam(kb, 93.2, (260.0, 174.0), W, H)
    fo = RC.PipelineOptions().focus
    old_rule(lens, region) = filter(RC._focus_support(lens, region), RC.focus_grid_nodes(region))   # RC ≤ 0.11.2: membership alone
    # an AXIS-ALIGNED ellipse: the box's side midpoints lie exactly on its rim — membership keeps 13, the ρ margin 9 (the inner 3 × 3)
    ax = VgtkCore.with_lens_coverage(fish, VgtkCore.Ellipse(VgtkCore.Point2{Float64}(260.0, 174.0), 150.0, 140.0, 0.0))
    @test length(old_rule(ax, VgtkCore.imaged_region(ax))) == 13
    fa = RC.focus_start_nodes(ax, fo)
    @test length(fa.nodes) == 9 && fa.nodes == fa.nodes_box[[7, 8, 9, 12, 13, 14, 17, 18, 19]]
    # a ROTATED one (as a detector returns it): the same 9, by design rather than by rounding
    rot = VgtkCore.with_lens_coverage(fish, VgtkCore.Ellipse(VgtkCore.Point2{Float64}(260.0, 174.0), 150.0, 146.0, 1.74))
    fr = RC.focus_start_nodes(rot, fo)
    @test length(fr.nodes) == 9 && fr.box == VgtkCore.Rect2(VgtkCore.imaged_region(rot)) && all(fr.in_support, fr.nodes)
    # an initialization region smaller than the imaged region (Schöps's calibrated area): the regression's placement over the region ∩
    # it is another box with nodes on its border; the start's nodes are the imaged region's, which it never reads
    init = VgtkCore.Rect2{Float64}(150.0, 60.0, 220.0, 228.0)
    ri = VgtkCore.imaged_region(rot) ∩ init
    @test VgtkCore.Rect2(ri) != fr.box && length(old_rule(rot, ri)) == 21 && SVector(260.0, 60.0) in old_rule(rot, ri)
    @test fr.nodes_box == RC.focus_grid_nodes(VgtkCore.imaged_region(rot))
    src = read(joinpath(pkgdir(RadiometricCamera), "src", "start", "prepare.jl"), String)
    @test occursin("= focus_start_nodes(lens0, opts.focus)", src) && !occursin("region_i", src) && !occursin(r"_focus_support\(|focus_grid_nodes\(", src)
    # the margin is the options' literal, in (0, 1]: the lens coverage's scale, as [masks]'s (RC 0.11.4: was node_rho_max, refused by name)
    @test fo.lens_coverage_scale == 0.9
    @test_throws ArgumentError RC.pipeline_options(Dict("focus" => Dict("lens_coverage_scale" => 1.2)))
    m = try RC.pipeline_options(Dict("focus" => Dict("node_rho_max" => 0.9))); "" catch e; e.msg end
    @test occursin("`focus.node_rho_max` is retired", m) && occursin("lens_coverage_scale", m)
end

@testitem "focus runaway: a support node at t̂ ≤ 0 or non-finite flags the calibration non-converged" setup = [RCFixtures] begin
    using RadiometricCamera: Focus, FocusSpline, focus_runaway, validity_gate, fold_termination
    using StaticArrays
    nodes = [SVector(x, y) for y in (100.0, 200.0, 300.0) for x in (100.0, 200.0, 300.0)]
    in_support = p -> p[1] < 350.0
    dome = Focus(FocusSpline(nodes), fill(1 / 0.7, 9))
    @test isempty(focus_runaway(dome, in_support))
    t = fill(1 / 0.7, 9); t[2] = 0.0; t[4] = -1e-3; t[6] = NaN
    @test focus_runaway(Focus(FocusSpline(nodes), t), in_support) == [2, 4, 6]
    @test isempty(focus_runaway(Focus(FocusSpline(nodes), t), p -> p[2] > 250.0))   # outside the support: not the run's to flag
    # a node the START floored at t̂ = 0 (a back-facing chief ray: `focus_start`) that stays there did not run away (code review,
    # RC 0.11.4): with the start surface, only a node that went from t̂ > 0 to ≤ 0, or a non-finite one, is flagged
    t0 = fill(1 / 0.7, 9); t0[2] = 0.0
    @test focus_runaway(Focus(FocusSpline(nodes), t), in_support, Focus(FocusSpline(nodes), t0)) == [4, 6]
    @test focus_runaway(Focus(FocusSpline(nodes), t0), in_support, Focus(FocusSpline(nodes), t0)) == Int[]
    ok = (VgtkCore.ReturnCode.Success, VgtkCore.CostDecrease)
    @test validity_gate(ok..., false) == ok
    @test validity_gate(ok..., true) == (VgtkCore.ReturnCode.Failure, VgtkCore.ShapeGate)
    @test fold_termination(ok..., (; folded = [1])) == validity_gate(ok..., true)       # the fold check: the same one gate
    # the refinement folds it into its termination record after the fold check, on the run's fixed support
    src = read(joinpath(pkgdir(RadiometricCamera), "src", "calibration", "calibrate.jl"), String)
    @test occursin("runaway = focus_runaway(fs, f.start.in_support, f.start.views0[1].focus)", src)
    @test occursin("code, rule = validity_gate(fold_termination(costs.termination..., folds)..., !isempty(runaway))", src)
end

@testitem "focus runaway after a resume: the start's node set is matched by position, never by index" setup = [RCFixtures] begin
    # A resumed state (`resume.state`, `load_state_into!`) or a start record may hold a focus on another node set than the run's start
    # (owner ruling 2026-11-16): the check compares each node with the start's value AT THE SAME NODE, no BoundsError and no other
    # node's value; a node the start does not hold has no start value, so t̂ ≤ 0 there is flagged as the two-argument check flags it.
    using RadiometricCamera: Focus, FocusSpline, focus_runaway
    using StaticArrays
    nodes = [SVector(x, y) for y in (100.0, 200.0, 300.0) for x in (100.0, 200.0, 300.0)]
    in_support = _ -> true
    t = fill(1 / 0.7, 9); t[2] = 0.0; t[9] = -1e-3                     # node 2 floored at the start; node 9 absent from the start
    keep = [8, 7, 6, 5, 4, 3, 2, 1]                                     # the start: nodes 1–8 in another order, node 2 floored there
    t0 = [i == 2 ? 0.0 : 1 / 0.7 for i in keep]
    f0 = Focus(FocusSpline(nodes[keep]), t0)
    @test focus_runaway(Focus(FocusSpline(nodes), t), in_support, f0) == [9]
    # the same node set reordered: node 5 ran from t̂ > 0 to ≤ 0 (flagged), node 2 stayed at the start's floor (not flagged)
    t5 = copy(t); t5[9] = 1 / 0.7; t5[5] = -1.0
    f9 = Focus(FocusSpline(nodes[vcat(keep, 9)]), vcat(t0, 1 / 0.7))
    @test focus_runaway(Focus(FocusSpline(nodes), t5), in_support, f9) == [5]
end
