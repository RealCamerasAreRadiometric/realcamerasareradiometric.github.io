# The pipeline's options from TOML (src/options/): the documented defaults file, the retired environment
# variables' keys, the refusals, the overrides and the writer's round trip; the version-stable random draws, angle_between and
# load_pipeline_metrics. One item per former test set, each named "pipeline options (TOML): …"
# (`Pkg.test(; test_args = ["pipeline options"])` runs them).

# The repository root and the two helpers the items share.
@testsetup module PipelineOptionsHelpers
    import RadiometricCamera
    const RC_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

    function haskey_path(T, path)        # `table.key` names a field of a table of T (or of the optimizer's, `[optimizer]`); `fixed`, the top-level list
        path == "fixed" && return hasfield(T, :fixed)
        t, k = split(path, '.')
        t == "optimizer" && return hasfield(RadiometricCamera.OptimizerTable, Symbol(k))   # the algorithm's tables
        t == "sweep" && return hasfield(RadiometricCamera.SweepTable, Symbol(k))
        return hasfield(T, Symbol(t)) && hasfield(fieldtype(T, Symbol(t)), Symbol(k))
    end
    using Test                          # (loads Core's TestExt: thrown_message, the message of the ArgumentError `f()` throws)
    using VisualGeometryToolkitCore: thrown_message
    export RC_ROOT, haskey_path, thrown_message
end

@testitem "pipeline options (TOML): configs/default.toml is THE defaults (every key, nothing else)" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using TOML
    # the structs' keyword defaults ARE the file (owner D8, 2026-11-15: Core's config_toml mechanism, baked at precompile): no second
    # set of literals to agree with it; a key missing from the file is an error naming it, by design
    path = pkgdir(RC, "configs", "default.toml")                                     # the file, where users find it
    @test RC._TOML_DEFAULTS == TOML.parsefile(path)
    @test RC.pipeline_options(Dict{String, Any}()) == RC.PipelineOptions()
    @test_throws KeyError RC.toml_default(RC._TOML_DEFAULTS, "grid", "no_such_key")
    src = read(joinpath(RC_ROOT, "src", "options", "tables.jl"), String)
    @test count("_option_default(", src) >= 75                                     # every table field's default reads the file
    # every key of the struct is in the file (the file documents them all)
    doc = TOML.parsefile(path)
    @test Set(keys(doc)) == union(Set(String.(fieldnames(RC.PipelineOptions))), ["optimizer", "sweep", "config_version"])   # the algorithm's tables, the version
    @test doc["config_version"] == RC.VgtkCore.current_config_version(RC.RC_CONFIG_VERSIONS)
    @test Set(keys(doc["optimizer"])) == Set(String.(fieldnames(RC.OptimizerTable))) && Set(keys(doc["sweep"])) == Set(String.(fieldnames(RC.SweepTable)))
    # but for the optional tables, absent by default, which the file documents in a comment
    optional = Dict("grid" => ["start_perturbation", "node_count"], "io" => ["lens_identity"])   # absent unless stated (no default)
    @test doc["fixed"] == [] && RC.PipelineOptions().fixed == String[]                      # the top-level list of held blocks: none
    for f in fieldnames(RC.PipelineOptions)
        f === :fixed && continue
        @test union(Set(keys(doc[String(f)])), get(optional, String(f), String[])) == Set(String.(fieldnames(fieldtype(RC.PipelineOptions, f))))
    end
    @test occursin("# start_perturbation = {sigma_px", read(path, String))
end

@testitem "pipeline options (TOML): every former environment variable has a key" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using RadiometricCamera: focus, material
    # Every environment variable the pipeline, bin/refine.jl and the recipes read before the options came from TOML, and its key (a
    # variable split into several keys lists each). RC_FOCUS_M was inert (read only to print that it is ignored) and has no key;
    # RC_BOARD (the board export beside a session: the session's records carry the target) and RC_IMAGE_CIRCLE (a typed-in circle:
    # the image region is the capture set's measured record, `camera.lens_coverage`) were retired with their design (RETIRED_KNOBS).
    FORMER_KNOBS = [
        "RC_IMAGES" => ["io.images"], "RC_OUT" => ["io.out"], "RC_LINEARIZE" => ["io.linearize"], "RC_VIEWS" => ["io.views"],
        "RC_INIT" => ["io.init"], "RC_CKPT_EACH" => ["io.checkpoint_each"],
        "RC_RENDER_INIT" => ["io.render_init"], "RC_RENDER_STATE" => ["io.render_state"],
        "RC_FRONTEND" => ["frontend.type"], "RC_SESSION" => ["frontend.session"],
        "RC_SCHOEPS" => ["frontend.schoeps"], "RC_TEXTURE" => ["frontend.texture"],
        "RC_TEXFEAT" => ["frontend.texture_features"], "RC_TEXFEAT_SIZE" => ["frontend.texture_features_size"],
        "RC_TEXDS" => ["frontend.texture_downsample"], "RC_PAGE_MM" => ["frontend.page_mm"], "RC_PRINT_SCALE" => ["frontend.print_scale"],
        "RC_POSE_DISTANCES" => ["frontend.pose_distances"],
        "RC_SENSOR_MM" => ["camera.sensor_mm"], "RC_F_MM" => ["camera.f_mm"], "RC_FNUMBER" => ["camera.fnumber"],
        "RC_IMAGE_CIRCLE_FRAMES" => ["camera.lens_coverage"],          # the frames' measurement became the set's image-region step (2026-11-15)
        "RC_GX" => ["grid.node_count"], "RC_GY" => ["grid.node_count"],   # grid.size → node_count (RC 0.11.4)
        "RC_GRID_PRIOR" => ["grid.smoothness"], "RC_GRID_PRIOR_WEIGHT" => ["grid.smoothness"],
        "RC_GRID_PERTURB" => ["grid.start_perturbation"],
        "RC_FOCUS_PRIOR" => ["focus.smoothness"],
        "RC_MASK_FRAC" => ["masks.samples_per_view"], "RC_NPIX" => ["masks.samples_per_view"],   # one count since 2026-11-15
        "RC_MASK_EDGE" => ["masks.gradient_bias"], "RC_SEED" => ["masks.seed"],
        "RC_N" => ["rendering.N"], "RC_NRHO" => ["rendering.n_rho"], "RC_NPHI" => ["rendering.n_phi"], "RC_NP" => ["rendering.n_p"],
        "RC_HAMMERSLEY" => ["rendering.hammersley"], "RC_SH_ORDER" => ["lighting.sh_order"],
        "RC_MATERIAL" => ["material.alpha_ink", "material.alpha_noink", "material.eta_ink", "material.eta_noink"], "RC_MU" => ["material.mu"],
        "RC_MU_BOUNDS" => ["material.mu_bounds"], "RC_FREEZE_MATERIAL" => ["fixed"],
        "RC_VIG" => ["vignetting.model", "vignetting.node_count", "vignetting.degree"],
        "RC_BOARD_HEIGHT" => ["warping.degree"], "RC_HEIGHT" => ["warping.heights_mm"],
        "RC_EDGE_SDF" => ["edge.sdf"], "RC_EDGE" => ["edge.ink_spread_mm", "edge.print_sigma_mm"],
        "RC_EDGE_FREE" => ["fixed"],   # the printed edge once held by "edge" in lm.fixed (edge.free retired 2026-11-11; "edge" refused, RC 0.11.0)
        "RC_MAXITERS" => ["optimizer.max_iter"], "RC_MAXTIME" => ["optimizer.max_time"], "RC_FIX" => ["fixed"],   # [lm] until RC 0.4.0's [optimizer]
        "RC_LM_LOG" => ["optimizer.log"],
        "RC_SWEEP_EACH" => ["sweep.lm_chunk"], "RC_SWEEP_" * "PA" * "SSES" => ["sweep.iterations"], "RC_SWEEP_JOINT" => ["sweep.joint"],   # the second: its name spelled apart (check_naming R7)
        "RC_RESUME_STATE" => ["resume.state"], "RC_IT_OFFSET" => ["resume.iteration_offset"], "RC_RESUME_SWEEP" => ["resume.sweep"],
        "RC_SELFCHECK_ONLY" => ["selfcheck.only"], "RC_SELFCHECK_POSES" => ["selfcheck.poses"],
        "RC_BACKEND" => ["compute.backend"], "RC_JAC_CHUNK" => ["compute.jacobian_chunk"],
    ]

    RETIRED_KNOBS = ["RC_FOCUS_M", "RC_BOARD", "RC_IMAGE_CIRCLE", "RC_VIG_SMOOTH",   # the V penalty, gone 2026-10-31
                     "RC_LM_ITERATOR",   # lm.iterator: retired with RC 0.9.0 (the damping is the algorithm's optimizer's: λ·I by default since 0.10.0)
                     "RC_NONCENTRAL",    # frontend.noncentral: retired with the non-central spline model (every camera is central)
                     "RC_LINEAR_FRAC",   # masks.linear_fraction: not read since 2026-10-31, retired with RC 0.11.0
                     "RC_RMIN", "RC_RMAX",   # camera.r_min / r_max: inert, retired with RC 0.11.0 (a held aperture is fixed = ["aperture"])
                     "RC_PSF",           # rendering.psf_sigma: inert since 2026-11-14, retired with RC 0.11.0
                     "RC_HEIGHT_POINTS", # warping.points: a hand-parsed file, retired with RC 0.11.0
                     "RC_FOCUS", "RC_FOCUS_NODES", "RC_FOCUS_PERTURB"]   # focus.variant / node_count / perturb: removed with the focus A/B experiment
    @test length(FORMER_KNOBS) == 62                                     # RC_SAVE_INIT (io.save_init) retired with the start record, RC_LINEAR_FRAC (masks.linear_fraction) with RC 0.11.0
    @test allunique(vcat(first.(FORMER_KNOBS), RETIRED_KNOBS))
    for (var, paths) in FORMER_KNOBS, p in paths
        @test haskey_path(RC.PipelineOptions, p)
    end
    # keys born after the environment variables were retired (config keys from the start)
    NEW_KEYS = ["grid.nodes", "vignetting.support_min",
                "vignetting.boundary_order", "vignetting.smoothness",              # 2026-11-04: the boundary order; V's Δ³ (P-spline) penalty ([vignetting.smoothness] since RC 0.11.0)
                "io.raw", "io.lens_identity",                                       # 2026-11-04: the capture's camera settings from its RAW files
                "io.channel",                                                       # 2026-11-04: one colour channel alone (the option of 2026-11-02, migrated)
                "warping.inplane", "warping.inplane_start",                         # 2026-11-10: the in-plane print error (Core's Aspect ∘ Skew) and its start
                "vignetting.chart",                                                 # 2026-11-10: the radial profile's chart (its free centre retired 2026-11-11)
                "camera.sensor_characterization",                                   # 2026-11-14: the camera's sensor model (IO record; the weighting follows it)
                "camera.sensor_model",                                              # 2026-11-15: whether the run uses it (auto | on | off; RC 0.11.2)
                "masks.lens_coverage_scale",                                        # 2026-11-15: the photometric sampling's scale of the lens coverage
                "grid.lattice",                                                     # 2026-11-15: the lattice's mode (derived | node_count; RC 0.11.4)
                "focus.support_min", "focus.lens_coverage_scale",                   # 2026-11-15: the focus nodes' support rule and rim margin
                "masks.edge_detector",                                              # 2026-11-15: the samples' edge strength (gradient_bias: RC_MASK_EDGE's)
                "optimizer.ftol", "optimizer.xtol_rel",                             # 2026-11-15: the LM's stop tolerances, the run's TOML (RC 0.11.4: xtol_rel, Core's relative step)
                "optimizer.type",                                                   # 2026-11-26: the refinement's optimizer (RC 0.4.0's [optimizer], [lm] until then)
                "masks.edge_fraction", "masks.addon_samples", "masks.addon_gradient_bias"]   # 2026-11-15: the draw's other modes (the sampling-strategy experiment, RC 0.11.5)
    # and every other key is one of them: no knob without a former variable, none forgotten
    keyset = union(Set(p for (_, ps) in FORMER_KNOBS for p in ps), NEW_KEYS)
    @test keyset == union(Set("$t.$k" for t in fieldnames(RC.PipelineOptions) if t !== :fixed for k in fieldnames(fieldtype(RC.PipelineOptions, t))), ["fixed"],
                          Set("optimizer.$k" for k in fieldnames(RC.OptimizerTable)), Set("sweep.$k" for k in fieldnames(RC.SweepTable)))
end

@testitem "pipeline options (TOML): unknown tables, keys and values are refused" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using RadiometricCamera: focus, material
    m = thrown_message(() -> RC.pipeline_options(Dict("lm" => Dict("fto" => 1))))
    @test occursin("fto", m) && occursin("max_iter", m) && occursin("max_time", m) && occursin("[optimizer]", m)   # (version 1's [lm], read as the current table)
    m = thrown_message(() -> RC.pipeline_options(Dict("lmm" => Dict("max_iter" => 1))))
    @test occursin("lmm", m) && occursin("lm", m)
    @test occursin("grid.node_count", thrown_message(() -> RC.pipeline_options(Dict("grid" => Dict("node_count" => [1, 2, 3])))))
    @test occursin("io.images", thrown_message(() -> RC.pipeline_options(Dict("io" => Dict("images" => 3)))))
    @test occursin("lens", thrown_message(() -> RC.pipeline_options(Dict("fixed" => ["lens"]))))
    # RC 0.11.0: the held blocks are the top-level `fixed`; `lm.fixed` refused by name, "grid" renamed "camera", "edge" refused
    m = thrown_message(() -> RC.pipeline_options(Dict("lm" => Dict("fixed" => ["focus"])))); @test occursin("`lm.fixed` is retired", m) && occursin("top-level `fixed`", m)
    @test occursin("renamed \"camera\"", thrown_message(() -> RC.pipeline_options(Dict("fixed" => ["grid"]))))
    @test occursin("never a parameter", thrown_message(() -> RC.pipeline_options(Dict("fixed" => ["edge"]))))
    for (t, k) in (("camera", "r_min"), ("camera", "r_max"), ("rendering", "psf_sigma"), ("edge", "camera_sigma_px"), ("warping", "points"))
        @test occursin("`$t.$k` is retired", thrown_message(() -> RC.pipeline_options(Dict(t => Dict(k => 0.0)))))
    end
    @test RC.pipeline_options(Dict("fixed" => ["aperture"])).fixed == ["aperture"] && !RC._lm_blocks(RC.pipeline_options(Dict("fixed" => ["aperture"]))).aperture
    @test RC._lm_blocks(RC.PipelineOptions()).aperture                                    # free by default (no bound: the log chart)
    @test RC.config_dict(RC.pipeline_options(Dict("fixed" => ["focus", "aperture"])))["fixed"] == ["focus", "aperture"]
    @test RC.pipeline_options(RC.config_dict(RC.pipeline_options(Dict("fixed" => ["camera"])))).fixed == ["camera"]
    # the retired parameter-holding keys are refused by name (`RETIRED_KEYS`), naming `fixed` (the stack's one parameter-holding keyword)
    m = thrown_message(() -> RC.pipeline_options(Dict("lm" => Dict("fix" => ["focus"])))); @test occursin("`lm.fix` is retired", m) && occursin("`fixed`", m)
    m = thrown_message(() -> RC.pipeline_options(Dict("edge" => Dict("free" => false)))); @test occursin("`edge.free` is retired", m) && occursin("never a parameter", m)
    # a dead knob refused by name (RC 0.11.0): masks.linear_fraction, not read since the closed-form start moved onto the LM's pixels
    m = thrown_message(() -> RC.pipeline_options(Dict("masks" => Dict("linear_fraction" => 0.01))))
    @test occursin("`masks.linear_fraction` is retired", m) && occursin("not read since 2026-10-31", m) && occursin("delete the key", m)
    @test !hasfield(RC.MaskOptions, :linear_fraction)
    # the printed edge is never a parameter (no hold of the LM; "edge" in fixed refused)
    @test_throws ArgumentError RC._surface_free(RC.pipeline_options(Dict("edge" => Dict("sdf" => "board"))), :edge)
    # one encoding (review round 2, finding 9): the model's structure (`_lm_structure`) and the holds (`_lm_surface_holds`), the
    # booleans `_edge_free` / `_inplane_free` gone; the LM's keywords are built from them by `_refinement_lm` alone
    eh = RC.pipeline_options(Dict("edge" => Dict("sdf" => "board"), "warping" => Dict("inplane" => "aspect_skew")))
    @test isempty(eh.fixed) && RC._lm_structure(eh) == (; height_degree = 0, inplane = true) && RC._lm_surface_holds(eh) == ()
    @test RC._refinement_lm(eh, RC.PerTileRoute()).kw.fixed == () && RC._refinement_lm(eh, RC.PerViewRoute()).kw.fixed == () && RC._surface_free(eh, :inplane)
    @test !haskey(RC._refinement_lm(eh, RC.PerTileRoute()).kw, :edge_block) && RC._refinement_lm(eh, RC.PerTileRoute(); smoothness = :p).kw.smoothness === :p
    # one derivation of the held blocks (review round 3, findings 6 and 7): `_refinement_lm` reads `_lm_blocks` itself; "material" is
    # the problem's `fixed` either way — the builders hold Θ_m alone while a surface parameter rides the block (`_hold_material`), the
    # problem's own type (`MaterialVar{…, true}`), never a module global
    hm = RC.pipeline_options(Dict("fixed" => ["focus", "material"], "warping" => Dict("degree" => 1)))
    lh = RC._refinement_lm(hm, RC.PerTileRoute())
    @test lh.kw.fixed == (:focus, :material) && lh.blocks.material && keys(lh) == (:kw, :blocks)       # Θ_m held inside a riding block
    lh2 = RC._refinement_lm(RC.pipeline_options(Dict("fixed" => ["material"])), RC.PerViewRoute())
    @test lh2.kw.fixed == (:material,) && !lh2.blocks.material                                         # nothing rides: the block held
    @test RC._refinement_lm(hm, RC.PerViewRoute()).kw.fixed == (:focus, :material)
    @test RC._hold_material((:focus, :material), true) == ((:material,), (:focus,)) && RC._hold_material((:focus, :material), false) == ((), (:focus, :material))
    @test !isdefined(RC, :_edge_free) && !isdefined(RC, :_inplane_free)
    psrc = join((read(joinpath(pkgdir(RadiometricCamera), "src", f), String) for f in ("calibration/problem_spec.jl", "start/prepare.jl", "start/image_region.jl", "calibration/calibrate.jl")), "\n")
    @test count("_refinement_lm(opts, ", psrc) == 3                                                    # docstring; the GCV, refine!
    # the grid smoothness's GCV builds the refinement's problem from the same keywords (RC 0.9.0 review, finding 4: prepare_refinement
    # left a module-global hold set; there is none now)
    @test count("select_grid_smoothness_weight(f, θ, opts.grid.smoothness, grid_int; damping, lp.kw...)", psrc) == 1
    # the grid smoothness's weight by GCV with the grid held: refused at load (finding 5: GCV linearized the held patches)
    m = thrown_message(() -> RC.pipeline_options(Dict("grid" => Dict("smoothness" => Dict("type" => "pspline", "order" => 3, "weight" => "gcv")), "fixed" => ["camera"])))
    @test occursin("`grid.smoothness.weight = \"gcv\"` with \"camera\" in `fixed`", m)
    @test RC.pipeline_options(Dict("grid" => Dict("smoothness" => Dict("type" => "pspline", "order" => 3, "weight" => 0.1)), "fixed" => ["camera"])).grid.smoothness == RC.PSplineSmoothness(3, 0.1)
    @test RC.pipeline_options(Dict("grid" => Dict("smoothness" => Dict("type" => "none")), "fixed" => ["camera"])).grid.smoothness == RC.NoSmoothness()
    # … on a grid route only: the parametric route (a `lens:` start) has no lattice, so a grid key it states is refused as unusable
    # (RC 0.11.4, owner ruling: never ignored), not as the grid route's GCV conflict
    m = thrown_message(() -> RC.pipeline_options(Dict("io" => Dict("init" => "lens:a/lens.json"), "grid" => Dict("smoothness" => Dict("type" => "pspline", "order" => 3, "weight" => "gcv")),
                                                      "fixed" => ["camera"])))
    @test occursin("`grid.smoothness` cannot be used", m) && !occursin("gcv", m)
    # the start's kind, decided once (finding 6): what prepare_refinement refines and what its io.save_init writes
    @test !RC.start_is_grid(RC.pipeline_options(Dict("io" => Dict("init" => "lens:a/lens.json"))))
    @test all(i -> RC.start_is_grid(RC.pipeline_options(Dict("io" => Dict("init" => i)))), ("", "grid:a/grid.json"))
    @test count("route = lm_route(lens0)", psrc) == 2 && !occursin("parametric = ", psrc)               # the route: the start camera's type, never a Bool
    # GPU first: a refinement's device, and its image region's measurement, is default_backend() (Core's default_device(), warning on the CPU) unless the caller names one (review
    # finding 10: prepare_refinement defaulted to the CPU)
    ssrc = read(joinpath(pkgdir(RadiometricCamera), "src", "start", "calibration_start.jl"), String)
    csrc = read(joinpath(pkgdir(RadiometricCamera), "src", "calibration", "calibrate.jl"), String)
    @test occursin("_calibration_device(::VgtkCore.NullParameters) = default_backend()", csrc)
    @test occursin("function load_calibration_start(dir::AbstractString; options::PipelineOptions, device = default_backend(),", ssrc)
    @test occursin("function measure_image_region(files::AbstractVector{<:AbstractString}; image::AbstractString, outlier_halfwidth_px::Real, seed::Integer,\n                              device = default_backend(),", psrc)
    @test !occursin("_lm_problem_kw", psrc) && count("if !getfield(blocks, k)", psrc) == 1                       # the held blocks: derived once
    # the printed edge is never a parameter, so no route can leave it free and unsolved (review finding 2: a parametric run recorded
    # it free and never solved it): whatever the holds the problem's keywords hold no edge, on either route
    edge_free = RC.pipeline_options(Dict("edge" => Dict("sdf" => "board"))); edge_held = RC.pipeline_options(Dict("edge" => Dict("sdf" => "board"), "fixed" => ["aperture"]))
    @test all(!(:edge in RC._refinement_lm(o, p).kw.fixed) && !haskey(RC._refinement_lm(o, p).kw, :edge_block) for o in (edge_free, edge_held), p in (RC.PerViewRoute(), RC.PerTileRoute()))
    @test !isdefined(RC, :_check_parametric_edge)
    m = thrown_message(() -> RC.pipeline_options(Dict("material" => Dict("freeze" => true)))); @test occursin("[material]", m) && occursin("\"freeze\"", m)
    # the image region is the capture set's own keyed step (owner ruling 2026-11-15): the run names its record (`camera.lens_coverage`) or
    # states the sensor; the frames and the fit's half-width of the retired in-run measurement are refused by name, whatever their value
    for (k, v) in (("image_circle_frames", "frames"), ("image_circle_outlier_halfwidth_px", 512.1), ("image_circle_min_hole_area_px", 8.0))
        m = thrown_message(() -> RC.pipeline_options(Dict("camera" => Dict(k => v)))); @test occursin("`camera.$k` is retired", m)
    end
    @test RC.PipelineOptions().camera.lens_coverage == RC.SENSOR_LENS_COVERAGE == "sensor"                     # default.toml: the sensor
    @test occursin("camera.lens_coverage", thrown_message(() -> RC.pipeline_options(Dict("camera" => Dict("lens_coverage" => "")))))   # stated, never empty
    @test !RC.names_input_path("camera.lens_coverage", "sensor") && RC.names_input_path("camera.lens_coverage", "regions/image_region.json")
    @test occursin("is retired", thrown_message(() -> RC.pipeline_options(Dict{String, Any}(); set = ["camera.image_circle_min_hole_area_px=8"])))
    # RAW or decoded is the photograph's TYPE, by its file (owner ruling 2026-11-13): the retired flag is refused by name, whatever its value
    for v in (true, false)
        m = thrown_message(() -> RC.pipeline_options(Dict("io" => Dict("photosites" => v, "channel" => "G")))); @test occursin("`io.photosites` is retired", m)
    end
    # the LM's damping is the algorithm's optimizer's (λ·I by default): the retired knob is refused by name, whatever its value
    for v in ("levenbergmarquardt", "levenbergmarquardt_dtd")
        m = thrown_message(() -> RC.pipeline_options(Dict("lm" => Dict("iterator" => v)))); @test occursin("`lm.iterator` is retired", m)
    end
    # the printed edge is never a parameter (RC 0.8.0 retired `edge.free`): an old config's flag is refused by name with its migration,
    # whatever its value
    for v in (true, false)
        m = thrown_message(() -> RC.pipeline_options(Dict("edge" => Dict("free" => v, "sdf" => "/e.png"))))
        @test occursin("`edge.free` is retired", m) && occursin("never a parameter", m)
    end
    @test occursin("`lm.fix` is retired: renamed `fixed`", thrown_message(() -> RC.pipeline_options(Dict("lm" => Dict("fix" => ["edge"])))))
    let c = RC.pipeline_options(Dict("camera" => Dict("lens_coverage" => "regions/image_region.json"))).camera
        @test c.lens_coverage == "regions/image_region.json" && !hasfield(RC.CameraOptions, :image_circle_frames) && !hasfield(RC.CameraOptions, :image_circle_outlier_halfwidth_px)
    end
    @test occursin("image_circle", thrown_message(() -> RC.pipeline_options(Dict("camera" => Dict("image_circle" => [1.0, 2.0, 3.0])))))   # retired: measured
    @test occursin("grid.smoothness.type", thrown_message(() -> RC.pipeline_options(Dict("grid" => Dict("smoothness" => Dict("type" => "tikhonov"))))))
    # the focus A/B experiment's keys are removed, each refused by name whatever its value: the focus surface is the sensor-grid thin plate spline
    for (k, v) in ("variant" => "A", "variant" => "C", "node_count" => 30, "perturb" => [0.1, 3.0])
        m = thrown_message(() -> RC.pipeline_options(Dict("focus" => Dict(k => v))))
        @test occursin("`focus.$k` is retired", m) && occursin("removed", m)
        k == "perturb" || @test occursin("the focus surface is the sensor-grid thin plate spline", m)
    end
    @test fieldnames(RC.FocusOptions) == (:lens_coverage_scale, :smoothness, :support_min)
    # warping.inplane: "none" | "aspect_skew"; inplane_start: none or the two finite values (a, u); "inplane" is a fixable block
    @test occursin("warping.inplane", thrown_message(() -> RC.pipeline_options(Dict("warping" => Dict("inplane" => "affine")))))
    @test occursin("warping.inplane_start", thrown_message(() -> RC.pipeline_options(Dict("warping" => Dict("inplane_start" => [0.01])))))
    @test occursin("warping.inplane_start", thrown_message(() -> RC.pipeline_options(Dict("warping" => Dict("inplane_start" => [0.01, NaN])))))
    let o = RC.pipeline_options(Dict("warping" => Dict("inplane" => "aspect_skew", "inplane_start" => [0.01, -0.002]), "fixed" => ["inplane"]))
        @test o.warping.inplane == "aspect_skew" && o.warping.inplane_start == [0.01, -0.002] && "inplane" in o.fixed
        @test RC._inplane_start(o) == [0.01, -0.002] && !RC._surface_free(o, :inplane)
    end
    @test RC._surface_free(RC.pipeline_options(Dict("warping" => Dict("inplane" => "aspect_skew"))), :inplane) && iszero(RC._inplane_start(RC.PipelineOptions()))
    @test !RC._surface_free(RC.PipelineOptions(), :inplane) && RC.PipelineOptions().warping.inplane == "none"
    # grid.start_perturbation: a table, both keys required (no default), no other key, σ ≥ 0
    sp(t) = thrown_message(() -> RC.pipeline_options(Dict("grid" => Dict("start_perturbation" => t))))
    m = sp(Dict("sigma_px" => 0.5))
    @test occursin("grid.start_perturbation", m) && occursin("seed", m) && occursin("required", m)
    m = sp(Dict("seed" => 7))
    @test occursin("grid.start_perturbation", m) && occursin("sigma_px", m)
    m = sp(Dict("sigma_px" => 0.5, "seed" => 7, "sigma" => 1.0))
    @test occursin("grid.start_perturbation", m) && occursin("sigma", m) && occursin("unknown", m)
    @test occursin("not a table", sp([0.5, 7]))
    @test occursin("grid.start_perturbation.seed", sp(Dict("sigma_px" => 0.5, "seed" => "x")))
    @test occursin("sigma_px", sp(Dict("sigma_px" => -1.0, "seed" => 7)))
    @test occursin("grid.start_perturbation", sp(Dict{String, Any}()))
    o = RC.pipeline_options(Dict("grid" => Dict("start_perturbation" => Dict("sigma_px" => 0.5, "seed" => 7))))
    @test o.grid.start_perturbation == RC.StartPerturbationOptions(; sigma_px = 0.5, seed = 7)
    @test RC.pipeline_options(Dict{String, Any}()).grid.start_perturbation === nothing
    @test RC.pipeline_options(Dict{String, Any}(); set = ["grid.start_perturbation.sigma_px=0.25", "grid.start_perturbation.seed=3"]).grid.start_perturbation ==
          RC.StartPerturbationOptions(; sigma_px = 0.25, seed = 3)
    @test occursin("frontend", thrown_message(() -> RC.pipeline_options(Dict("frontend" => Dict("board" => "x")))))           # retired: the session's records
    @test occursin("must be a table", thrown_message(() -> RC.pipeline_options(Dict("lm" => 3))))
    # a string is not an array: no character-wise conversion of a tuple or vector field
    m = thrown_message(() -> RC.pipeline_options(Dict("frontend" => Dict("page_mm" => "A4"))))
    @test occursin("frontend.page_mm", m) && occursin("Tuple{Float64, Float64}", m) && occursin("array of 2 values", m)
    @test occursin("grid.node_count", thrown_message(() -> RC.pipeline_options(Dict("grid" => Dict("node_count" => "ab")))))
    @test occursin("io.views", thrown_message(() -> RC.pipeline_options(Dict("io" => Dict("views" => "V01.png")))))
    @test occursin("frontend.page_mm", thrown_message(() -> RC.pipeline_options(Dict{String, Any}(); set = ["frontend.page_mm=A4"])))
    # the top level is named as such, and an error read from a file names the file
    m = thrown_message(() -> RC.pipeline_options(Dict("lmm" => Dict())))
    @test occursin("[lmm] is not a table of RadiometricCamera's configuration", m) && !occursin("[pipeline]", m)   # the converter's refusal (RC_CONFIG_VERSIONS)
    mktempdir() do dir
        f = joinpath(dir, "bad.toml"); write(f, "[lm]\nfto = 1\n")
        m = thrown_message(() -> RC.pipeline_options(f))
        @test startswith(m, f) && occursin("fto", m)
        @test occursin("lm.max_iter=x", thrown_message(() -> RC.pipeline_options(f; set = ["lm.max_iter=x"])))
    end
end

@testitem "pipeline options (TOML): tags: the registries, an unknown tag, a retired tag" setup = [RCFixtures, PipelineOptionsHelpers] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    for reg in (RC.FRONTEND_TAGS, RC.VIGNETTING_CONFIG_TAGS, RC.BACKEND_TAGS, RC.LATTICE_TAGS, RC.SENSOR_MODEL_TAGS)   # every enumeration a registry (0.11.5)
        @test isempty(VgtkCore.tag_rule_violations(reg))
    end
    @test RC.pipeline_options(Dict("frontend" => Dict("type" => "schoeps"))).frontend.type === RC.SchoepsFrontend
    @test RC.pipeline_options(Dict("vignetting" => Dict("model" => "radial"))).vignetting.model === VrtkCore.RadialVignetting
    # the vignetting field is a tag of Core's VIGNETTING_TAGS, and a refinement fits the radial profile alone: "tensor_product" and
    # "log" are refused by name, the retired "tensor" names its successor
    @test occursin("a refinement fits \"radial\" alone", thrown_message(() -> RC.pipeline_options(Dict("vignetting" => Dict("model" => "tensor_product")))))
    @test occursin("renamed \"tensor_product\"", thrown_message(() -> RC.pipeline_options(Dict("vignetting" => Dict("model" => "tensor")))))
    @test occursin("\"log\"", thrown_message(() -> RC.pipeline_options(Dict("vignetting" => Dict("model" => "log")))))
    @test RC.pipeline_options(Dict("compute" => Dict("backend" => "cpu"))).compute.backend === RC.CPUCompute
    @test RC.pipeline_options(Dict("compute" => Dict("backend" => "gpu"))).compute.backend === RC.GPUCompute === RC.ComputeOptions().backend
    # the backend's retired tag "cuda" (2026-11-13: any KernelAbstractions GPU backend) names its successor
    @test occursin("renamed \"gpu\"", thrown_message(() -> RC.pipeline_options(Dict("compute" => Dict("backend" => "cuda")))))
    m = thrown_message(() -> RC.pipeline_options(Dict("frontend" => Dict("type" => "blobs"))))
    @test occursin("blobs", m) && occursin("\"blob\"", m) && occursin("\"schoeps\"", m)
    # the mechanism the reader uses names the successor of a retired one
    renamed = merge(RC.FRONTEND_TAGS, VgtkCore.TagRegistry(:frontend, "Frontend"; retired = ("blobboards" => "blob",)))
    @test occursin("renamed \"blob\"", thrown_message(() -> renamed["blobboards"]))
    # the shipped configs carry current tags only
    @test isempty(VgtkCore.config_tag_violations((RC.FRONTEND_TAGS, RC.VIGNETTING_CONFIG_TAGS, RC.BACKEND_TAGS),
                                                 [joinpath(RC_ROOT, "configs")]))   # (the paper project's configs: experiments/test, owner D15)
end

@testitem "pipeline options (TOML): --set overrides" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using RadiometricCamera: focus
    using TOML
    alg, stops = RC.read_calibration(Dict{String, Any}("config_version" => 2);
                                     set = ["optimizer.max_time=1e-10", "optimizer.max_iter=2", "io.out=/tmp/rc out", "frontend.type=schoeps",
                                            "io.views=[\"V01.png\", \"V02.png\"]", "camera.lens_coverage=/data/image_region.json",
                                            "grid.smoothness={type = \"bending\", weight = 0.1}", "focus.support_min=0.25"])
    o = alg.refinement.options
    @test stops.max_time == 1e-10 && stops.max_iter == 2 && o.io.out == "/tmp/rc out" && o.frontend.type === RC.SchoepsFrontend
    @test o.io.views == ["V01.png", "V02.png"] && o.camera.lens_coverage == "/data/image_region.json"
    @test o.grid.smoothness == RC.BendingSmoothness(0.1) && o.focus.support_min == 0.25
    # an override replaces a file's value and is refused like a file's unknown key
    @test last(RC.read_calibration(Dict("lm" => Dict("max_iter" => 7)); set = ["lm.max_iter=3"])).max_iter == 3     # a version-1 table, its names
    @test last(RC.read_calibration(Dict("config_version" => 2, "optimizer" => Dict("max_iter" => 7)); set = ["optimizer.max_iter=3"])).max_iter == 3
    @test occursin("fto", thrown_message(() -> RC.pipeline_options(Dict{String, Any}(); set = ["lm.fto=1"])))
    @test occursin("key.path=value", thrown_message(() -> RC.pipeline_options(Dict{String, Any}(); set = ["lm.max_iter"])))
    # one TOML value per override; a quoted string stays a string
    @test occursin("not one TOML value", thrown_message(() -> RC.pipeline_options(Dict{String, Any}(); set = ["lm.max_iter=1\nmax_time = 2"])))
    @test RC.pipeline_options(Dict{String, Any}(); set = ["io.out=\"2024\""]).io.out == "2024"
end

@testitem "pipeline options (TOML): config_dict round-trips through TOML" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using RadiometricCamera: focus, material
    using TOML
    o = RC.PipelineOptions(; io = RC.IOOptions(; images = "imgs", out = "out", views = ["a.png"], linearize = true),
                           frontend = RC.FrontendOptions(; type = RC.SchoepsFrontend, page_mm = (1000.0, 1000.0)),
                           camera = RC.CameraOptions(; lens_coverage = "regions/image_region.json"),
                           grid = RC.GridOptions(; smoothness = RC.BendingSmoothness(1e-2),
                                                 start_perturbation = RC.StartPerturbationOptions(; sigma_px = 0.5, seed = 7)), focus = RC.FocusOptions(; support_min = 0.1, smoothness = RC.BendingSmoothness(0.5)),
                           vignetting = RC.VignettingOptions(; model = VrtkCore.RadialVignetting, node_count = 16),
                           fixed = ["material", "lighting", "aperture"],
                           edge = RC.EdgeOptions(; ink_spread_mm = 0.02, print_sigma_mm = 0.03), compute = RC.ComputeOptions(; backend = RC.CPUCompute))
    back = RC.pipeline_options(TOML.parse(sprint(io -> TOML.print(io, RC.config_dict(o)))))
    @test back == o
    # the calibration (its options, its optimizer) and its solve's stops round-trip too, the record a current version (`config_version`)
    alg = RC.RadiometricCalibration(RC.RadiometricRefinement(o; log = "lm.tsv")); stops = RC._calibration_stops(; max_iter = 2)
    d = TOML.parse(sprint(io -> TOML.print(io, RC.config_dict(alg, stops))))
    @test d["config_version"] == RC.VgtkCore.current_config_version(RC.RC_CONFIG_VERSIONS) && d["optimizer"]["max_iter"] == 2 &&
          d["optimizer"]["type"] == "levenberg_marquardt"
    alg2, stops2 = RC.read_calibration(d)
    @test alg2.refinement.options == o && alg2.refinement.log == "lm.tsv" && alg2.refinement.optimizer == alg.refinement.optimizer && stops2 == stops
    @test RC.pipeline_options(RC.config_dict(RC.PipelineOptions())) == RC.PipelineOptions()
    # the camera's in-focus blur is removed: its two keys are retired (RC 0.11.0), refused by name whatever their value
    @test_throws "`rendering.psf_sigma` is retired" RC.pipeline_options(Dict{String, Any}("rendering" => Dict{String, Any}("psf_sigma" => 0.0)))
    @test_throws "`edge.camera_sigma_px` is retired" RC.pipeline_options(Dict{String, Any}("edge" => Dict{String, Any}("camera_sigma_px" => [0, 0, 0])))
    @test RC.config_dict(o)["frontend"]["type"] == "schoeps" && RC.config_dict(o)["compute"]["backend"] == "cpu"
    @test RC.config_dict(o)["grid"]["smoothness"] == Dict("type" => "bending", "weight" => 1e-2) && !haskey(RC.config_dict(o), "optimizer")
    @test RC.config_dict(o)["grid"]["start_perturbation"] == Dict("sigma_px" => 0.5, "seed" => 7)
    @test !haskey(RC.config_dict(RC.PipelineOptions())["grid"], "start_perturbation")     # absent: not written
end

@testitem "pipeline options (TOML): every string option is declared an input path, an output path or no path" setup = [RCFixtures, PipelineOptionsHelpers] begin
    strings = ["$t.$k" for t in fieldnames(RC.PipelineOptions) for (k, K) in zip(fieldnames(fieldtype(RC.PipelineOptions, t)), fieldtypes(fieldtype(RC.PipelineOptions, t)))
               if K === String]
    append!(strings, "optimizer.$k" for (k, K) in zip(fieldnames(RC.OptimizerTable), fieldtypes(RC.OptimizerTable)) if K === String)
    declared = vcat(collect(RC.PIPELINE_PATH_KEYS), collect(RC.PIPELINE_OUTPUT_KEYS), collect(RC._NOT_PATH_KEYS))
    @test allunique(declared)
    @test sort(strings) == sort(declared)
    @test all(k -> haskey_path(RC.PipelineOptions, k), declared)
end

@testitem "pipeline options (TOML): the pipeline's random draws are version-stable (masks.seed)" setup = [RCFixtures, PipelineOptionsHelpers] begin
    # the LM pixel subsample and the Bernoulli masks draw from StableRNG(masks.seed) by rand(rng, …) only; these values are pinned so a
    # change of the stream (as randperm's in Julia 1.13) fails here rather than moving the pipeline checks
    @test RC._subsample(RC.StableRNG(1), 4774, 8) == [177, 1629, 1875, 2156, 2172, 3009, 3096, 3859]
    @test RC._subsample(RC.StableRNG(1), 10, 10) == 1:10
    @test rand(RC.StableRNG(2), 3) == [0.9753244036873643, 0.12889655761518704, 0.5277137334239088]   # the masks' Float64 stream
    @test RC._subsample(RC.StableRNG(7), 50, 5) == RC._subsample(RC.StableRNG(7), 50, 5)
end

@testitem "pipeline options (TOML): angle_between is well-conditioned at every angle" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using StaticArrays, LinearAlgebra
    for θ in (1e-5, 1e-8, 0.3, 2.0)
        a = SVector(1.0, 0.0, 0.0); b = SVector(cos(θ), sin(θ), 0.0)
        @test abs(RC.angle_between(a, b) - θ) <= 1e-12 * θ
        @test abs(RC.angle_between(3a, 0.5b) - θ) <= 1e-12 * θ                     # not only unit vectors
    end
    @test abs(acos(clamp(dot(SVector(1.0, 0.0, 0.0), SVector(cos(1e-8), sin(1e-8), 0.0)), -1, 1)) - 1e-8) > 1e-12 * 1e-8   # acos fails it
    @test RC.angle_between(SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 2.0)) == 0.0
    @test RC.angle_between(SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, -1.0)) == Float64(π)
    @test RC.angle_between(SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)) == π / 2
end

@testitem "pipeline options (TOML): load_pipeline_metrics reads a run's metrics back" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using RadiometricCamera: cost
    using TOML
    using JSON: JSON
    dir = mktempdir()                                                          # a refinement: its rc/run_parameters/v4.0 record
    RC.save_run_parameters(joinpath(dir, "parameters.json"),
        (; cost = (; init = (; cost = 2.0, rel_rms = 0.2), after_linear_solve = (; cost = 1.5, rel_rms = 0.15), after_refinement = (; cost = 1.0, rel_rms = 0.1)),
           bearing_error = (; init_median = 0.5, init_p90 = 0.9, refined_median = 0.3, refined_p90 = 0.6), camera_moved = [0.001, 0.002],
           iterations = 2, stopped = "ftol", retcode = "Success", stop_rule = "FunctionTolerance", grid = nothing))
    @test VgtkIO.read_record(joinpath(dir, "parameters.json")).schema == "rc/run_parameters/v4.0"
    @test VgtkIO.read_record(joinpath(dir, "parameters.json")).units.length == "m"
    m = RC.load_pipeline_metrics(dir)
    @test sort(collect(keys(m))) == ["bearing_error", "camera_moved", "cost", "iterations", "retcode", "stop_rule", "stopped"]
    @test sort(collect(keys(m["cost"]))) == ["after_linear_solve", "after_refinement", "init"]
    @test m["cost"]["after_refinement"]["rel_rms"] == 0.1 && m["iterations"] == 2 && m["camera_moved"] == [0.001, 0.002] && m["retcode"] == "Success"
    @test sort(collect(keys(m["bearing_error"]))) == ["init_median", "init_p90", "refined_median", "refined_p90"]
    @test TOML.parse(sprint(io -> TOML.print(io, m))) == m                   # TOML-printable as it stands
    # an RC ≤ 0.8 run directory (rc/run_parameters/v1): refused by name, naming how to regenerate it (the run's own options beside it)
    old_dir = mktempdir(); write(joinpath(old_dir, "parameters.json"), "{\"schema\": \"rc/run_parameters/v1\", \"iterations\": 2}")
    m_old = thrown_message(() -> RC.load_pipeline_metrics(old_dir))
    @test occursin("written by an earlier release, on an earlier VisualGeometryToolkitIO envelope", m_old) && !occursin("--config=", m_old)          # no options of its writer: none named
    write(joinpath(old_dir, "pipeline_config.toml"), "[io]\nout = \"$old_dir\"\n")               # its writer's (its io.out is this directory)
    @test occursin("bin/refine.jl --config=$(joinpath(old_dir, "pipeline_config.toml"))", thrown_message(() -> RC.load_pipeline_metrics(old_dir)))

    old = joinpath(mktempdir(), "parameters.json"); write(old, "{\"frontend\": \"blob\", \"image_size\": [5208, 3476]}")
    @test_throws "not a rc/run_parameters/v4.0 record" RC.load_run_parameters(old)                                  # a report before the record
    # a v2.0 run record (RC ≤ 0.10: the grid's `prior*` keys) reads through its version step into the smoothness's keys, values as stored
    v2 = joinpath(mktempdir(), "parameters.json")
    VgtkIO.write_record(v2, (; schema = "rc/run_parameters/v2.0", frontend = "schoeps",
                              grid = (; n_interior = 3, prior = "pspline", prior_weight = "gcv", prior_cost = (; init = 0.0, final = 1.5e-4),
                                      prior_beta = 0.0123, prior_gcv = (; β = 0.0123, gcv = 2.0e-3, kind = "pspline"), start_perturbation = nothing),
                              vignetting_options = (; smoothness_weight = 0.5, boundary_order = 3, smoothness_gcv = nothing)); style = RC._style(VgtkCore.m))
    r3 = RC.load_run_parameters(v2)
    @test r3.schema == "rc/run_parameters/v4.0" && !haskey(r3.grid, :prior) && !haskey(r3.grid, :prior_weight) && r3.grid.n_interior == 3
    @test r3.grid.smoothness.type == "pspline" && r3.grid.smoothness.order == 3 && r3.grid.smoothness.weight == "gcv" && r3.grid.smoothness.beta == 0.0123
    @test r3.grid.smoothness.cost.final == 1.5e-4 && r3.grid.smoothness.selection.β == 0.0123
    @test r3.vignetting_options.smoothness.weight == 0.5 && r3.vignetting_options.smoothness.order == 3 && r3.vignetting_options.boundary_order == 3
    # a v3.0 run record of record (the experiments' golden siemens_coarse_full run, RC 0.11): read through the v3 → v4 step (owner
    # decision 11), its camera option's key `sensor_characterization` and its frames' reason `NoCharacterization`; every other value as stored
    golden = joinpath(pkgdir(RadiometricCamera), "..", "experiments", "test", "pipeline", "golden", "siemens_coarse_full", "parameters.json")
    if isfile(golden)
        raw = JSON.parsefile(golden)
        @test raw["schema"] == "rc/run_parameters/v3.0" && haskey(raw, "sensor_characterisation")   # the recorded file, unchanged
        r4 = RC.load_run_parameters(golden)
        @test r4.schema == "rc/run_parameters/v4.0" && r4.sensor_characterization == raw["sensor_characterisation"] && !haskey(r4, Symbol("sensor_characterisation"))
        @test [m.reason for m in r4.sensor_models] == fill("NoCharacterization", length(raw["sensor_models"]))
        @test [m.digest for m in r4.sensor_models] == [m["digest"] for m in raw["sensor_models"]] && r4.cost.after_refinement.cost == raw["cost"]["after_refinement"]["cost"]
    end
    # the focus A/B experiment's report (rc/focus_ab, focus_ab.json) is a removed kind: a file of it, of either version, is refused by name
    @test !haskey(RC.RECORD_SCHEMAS, :focus_ab)
    for v in ("v1.0", "v2.0")
        fa = joinpath(mktempdir(), "focus_ab.json")
        VgtkIO.write_record(fa, (; schema = "rc/focus_ab/$v", variant = "A"))
        m = thrown_message(() -> VgtkIO.read_record(fa))
        @test occursin("rc/focus_ab", m) && occursin("was removed", m) && occursin("sensor-grid thin plate spline", m)
    end
    sc_dir = mktempdir()                                                       # a self-check: its selfcheck.json (_write_selfcheck's record)
    VgtkIO.write_record(joinpath(sc_dir, "selfcheck.json"),
        (; schema = RC.RECORD_SCHEMAS.selfcheck, init = "", schoeps = "run", views = ["V01.png"], n = [12], median = [0.21], p90 = [0.4], mean = [0.25],
           all = (; n = 12, median = 0.21, p90 = 0.4, mean = 0.25)))
    s = RC.load_pipeline_metrics(sc_dir)
    @test collect(keys(s)) == ["selfcheck"] && sort(collect(keys(s["selfcheck"]))) == ["mean", "median", "n", "p90"]
    @test_throws ArgumentError RC.load_pipeline_metrics(mktempdir())
end

@testitem "pipeline options (TOML): one smoothness vocabulary for the spline components" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using RadiometricCamera: focus
    using TOML
    using VisualGeometryToolkitEstimation: GeneralizedCrossValidation   # a weight chosen by GCV: the algorithm value (the TOML's "gcv")
    # the defaults: the grid's P-spline of its Δ³ at 0.1, the vignetting's Δ³ at 0, the focus's bending at 0 (RC 0.11.0's tables)
    d = RC.PipelineOptions()
    @test d.grid.smoothness == RC.PSplineSmoothness(3, 0.1) && d.vignetting.smoothness == RC.PSplineSmoothness(3, 0.0) && d.focus.smoothness == RC.BendingSmoothness(0.0)
    # each table round-trips through TOML, dispatched by its type (no flag)
    for (comp, sm) in (("grid", RC.PSplineSmoothness(3, GeneralizedCrossValidation())), ("grid", RC.BendingSmoothness(0.5)), ("grid", RC.NoSmoothness()),
                       ("vignetting", RC.PSplineSmoothness(3, GeneralizedCrossValidation())), ("vignetting", RC.NoSmoothness()), ("focus", RC.BendingSmoothness(0.25)))
        o = RC.pipeline_options(Dict(comp => Dict("smoothness" => RC.smoothness_table(sm))))
        @test getfield(o, Symbol(comp)).smoothness == sm
        @test RC.pipeline_options(TOML.parse(sprint(io -> TOML.print(io, RC.config_dict(o))))) == o
    end
    @test RC.smoothness_line(RC.PSplineSmoothness(3, 0.1)) == "pspline order 3, weight 0.1" && RC.smoothness_line(RC.NoSmoothness()) == "none"
    # the retired keys are refused by name, the replacement named
    for (t, k) in (("grid", "prior"), ("grid", "prior_weight"), ("vignetting", "smoothness_weight"), ("focus", "prior"))
        m = thrown_message(() -> RC.pipeline_options(Dict(t => Dict(k => k == "prior" && t == "grid" ? "pspline" : 0.1))))
        @test occursin("`$t.$k` is retired", m) && occursin("[$t.smoothness]", m)
    end
    @test !isdefined(RC, :GRID_PRIORS) && !isdefined(RC, :grid_prior) && !isdefined(RC, :select_grid_prior_weight) && !isdefined(RC, :focus_prior!)
    # a table's keys are its type's, each required; a component takes only the types it implements
    sm(t, x...) = Dict(t => Dict("smoothness" => Dict(x...)))
    @test occursin("order", thrown_message(() -> RC.pipeline_options(sm("grid", "type" => "pspline", "weight" => 0.1))))
    @test occursin("order", thrown_message(() -> RC.pipeline_options(sm("grid", "type" => "bending", "order" => 3, "weight" => 0.1))))
    @test occursin("vignetting.smoothness.type", thrown_message(() -> RC.pipeline_options(sm("vignetting", "type" => "bending", "weight" => 0.1))))
    @test occursin("vignetting.smoothness.order", thrown_message(() -> RC.pipeline_options(sm("vignetting", "type" => "pspline", "order" => 2, "weight" => 0.1))))
    @test occursin("focus.smoothness.weight", thrown_message(() -> RC.pipeline_options(sm("focus", "type" => "bending", "weight" => "gcv"))))
    @test occursin("grid.smoothness.weight", thrown_message(() -> RC.pipeline_options(sm("grid", "type" => "pspline", "order" => 3, "weight" => -1.0))))
end

@testitem "pipeline options (TOML): a switch is \"on\" | \"off\" (RC 0.11.4); a boolean is refused naming its new spelling" setup = [RCFixtures, PipelineOptionsHelpers] begin
    using TOML
    # every Bool option is a switch, and these are all of them
    switches = [(String(t), String(k)) for t in fieldnames(RC.PipelineOptions) if t !== :fixed
                for (k, K) in zip(fieldnames(fieldtype(RC.PipelineOptions, t)), fieldtypes(fieldtype(RC.PipelineOptions, t))) if K === Bool]
    @test Set("$t.$k" for (t, k) in switches) ==
          Set(["io.linearize", "io.checkpoint_each", "io.render_init", "rendering.hammersley", "resume.sweep", "selfcheck.only"])
    # and the algorithm's one, `sweep.joint` (read by read_calibration: the closed-form blocks jointly); the joint solve has no
    # vignetting smoothness, so its "on" is read beside a smoothness of none
    none = Dict{String, Any}("vignetting" => Dict{String, Any}("smoothness" => Dict{String, Any}("type" => "none")))
    jon, _ = RC.read_calibration(merge(none, Dict{String, Any}("sweep" => Dict{String, Any}("joint" => "on"))))
    joff, joff_stops = RC.read_calibration(Dict{String, Any}("sweep" => Dict{String, Any}("joint" => "off")))
    @test jon.refinement.radiometry isa RC.JointRadiometry && jon.refinement.radiometry.damping == jon.refinement.optimizer.damping && joff.refinement.radiometry isa RC.RadiometryBlockDescent
    @test RC.config_dict(jon, joff_stops)["sweep"]["joint"] == "on" && RC.config_dict(joff, joff_stops)["sweep"]["joint"] == "off"
    @test occursin("`sweep.joint = true` is retired", thrown_message(() -> RC.pipeline_options(Dict{String, Any}("sweep" => Dict{String, Any}("joint" => true)))))
    for (t, k) in switches
        cfg(v) = Dict{String, Any}(t => Dict{String, Any}(k => v))
        on, off = RC.pipeline_options(cfg("on")), RC.pipeline_options(cfg("off"))
        @test getfield(getfield(on, Symbol(t)), Symbol(k)) === true && getfield(getfield(off, Symbol(t)), Symbol(k)) === false
        @test RC.config_dict(on)[t][k] == "on" && RC.config_dict(off)[t][k] == "off"          # the writer spells it
        @test RC.pipeline_options(TOML.parse(sprint(io -> TOML.print(io, RC.config_dict(on))))) == on
        @test TOML.parsefile(joinpath(RC_ROOT, "configs", "default.toml"))[t][k] == "off"      # every default: "off"
        for (b, w) in ((true, "on"), (false, "off"))                                           # the old boolean, refused (no alias)
            m = thrown_message(() -> RC.pipeline_options(cfg(b)))
            @test occursin("`$t.$k = $b` is retired", m) && occursin("use `$t.$k = \"$w\"`", m)
        end
        @test occursin("is not one of \"on\", \"off\" (a switch)", thrown_message(() -> RC.pipeline_options(cfg("yes"))))
        @test occursin("is not one of", thrown_message(() -> RC.pipeline_options(cfg(1))))
    end
    # --set: `table.key=on`; the old `=true` refused by the same message
    @test RC.pipeline_options(Dict{String, Any}(); set = ["rendering.hammersley=on"]).rendering.hammersley
    @test occursin("use `io.render_init = \"on\"`", thrown_message(() -> RC.pipeline_options(Dict{String, Any}(); set = ["io.render_init=true"])))
    # the one reader and its inverse
    @test RC.read_switch("on", "k") === true && RC.read_switch("off", "k") === false
    @test RC.switch_spelling(true) == "on" && RC.switch_spelling(false) == "off"
    # ONE reader for every switch, the three-way sensor model's too ("auto" resolved at the parse into a type)
    @test RC.read_switch("auto", "camera.sensor_model", RC.SENSOR_MODEL_TAGS) === RC.AutoSensorModel() === RC.sensor_model_mode("auto")
    @test RC.switch_spelling(RC.OffSensorModel(), RC.SENSOR_MODEL_TAGS) == "off"
    @test_throws "`camera.sensor_model = true` is retired" RC.sensor_model_mode(true)
    @test_throws "`x.y = false` is retired" RC.read_switch(false, "x.y")
    # the lint: a boolean switch in a config is a finding, a boolean declared data is not
    dir = mktempdir()
    write(joinpath(dir, "a.toml"), "[io]\nlinearize = true\n[lens]\nfills_sensor = false\n[[stage]]\ngpu = true\n")
    v = RC.boolean_switch_violations([dir])
    @test length(v) == 3 && any(f -> endswith(f, ":io.linearize = true"), v) && any(f -> endswith(f, ":stage[1].gpu = true"), v)
    @test RC.boolean_switch_violations([dir]; data = [r"lens\.fills_sensor", r"stage\[\d+\]\.gpu"]) == [joinpath(dir, "a.toml") * ":io.linearize = true"]
    write(joinpath(dir, "b.toml"), "[io\n")
    @test any(f -> occursin("does not parse", f), RC.boolean_switch_violations([dir]))
    # ONE key for a spline's node count (RC 0.11.4): the old keys refused by name, naming `node_count`
    for (t, k, names) in (("vignetting", "coefficients", "node_count"), ("focus", "nodes", "sensor-grid thin plate spline"))   # (the focus's node count was a removed direction chart's)
        m = thrown_message(() -> RC.pipeline_options(Dict(t => Dict(k => 16))))
        @test occursin("`$t.$k` is retired", m) && occursin(names, m)
    end
    @test RC.pipeline_options(Dict("vignetting" => Dict("node_count" => 12))).vignetting.node_count == 12
    # the package's configs carry no boolean switch (the paper project's: experiments/test, calibration_tests.jl)
    @test isempty(RC.boolean_switch_violations([joinpath(RC_ROOT, "configs")]))
end

@testitem "pipeline options (TOML): the configuration converter — version 1 ([lm], the old key spellings) reads as version 2" setup = [RCFixtures, PipelineOptionsHelpers] begin
    old = "sensor_characteris" * "ation"                                       # the British spelling, split (check_naming's banned word)
    cur = RC.VgtkCore.current_config_version(RC.RC_CONFIG_VERSIONS)
    @test cur == 2
    # a version-1 file (no config_version): [lm] is the optimizer's table, the old key spelling the current key, values as stored
    v1 = Dict{String, Any}("lm" => Dict{String, Any}("max_iter" => 7, "log" => "x.tsv"), "camera" => Dict{String, Any}(old => "s.json"))
    alg, stops = RC.read_calibration(v1)
    @test stops.max_iter == 7 && alg.refinement.log == "x.tsv" && alg.refinement.options.camera.sensor_characterization == "s.json"
    @test RC.pipeline_options(v1) == alg.refinement.options
    # a version-2 file names them as version 2 does: a version-1 name in it is refused, naming the current one
    @test occursin("optimizer", thrown_message(() -> RC.pipeline_options(Dict{String, Any}("config_version" => 2, "lm" => Dict{String, Any}()))))
    @test occursin("sensor_characterization",
                   thrown_message(() -> RC.pipeline_options(Dict{String, Any}("config_version" => 2, "camera" => Dict{String, Any}(old => "s.json")))))
    @test occursin("newer", thrown_message(() -> RC.pipeline_options(Dict{String, Any}("config_version" => 3))))
    # the writer states the current version, and its record reads back
    d = RC.config_dict(alg, stops)
    @test d["config_version"] == cur && haskey(d, "optimizer") && !haskey(d, "lm") && first(RC.read_calibration(d)).refinement.options == alg.refinement.options
    # a current key as a stored table names it: a table's rename undone, a key renamed inside its table kept
    @test RC.config_key(v1, "optimizer.max_iter") == "lm.max_iter" && RC.config_key(d, "optimizer.max_iter") == "optimizer.max_iter"
    @test RC.config_key(v1, "optimizer") == "lm" && RC.config_key(v1, "io.out") == "io.out"
    @test RC.config_key(v1, "camera.sensor_characterization") == "camera.sensor_characterization"
    # the split of a version-1 table is under its own names; of a version-2 table under the current ones
    @test last(RC.split_refine_only(Dict{String, Any}("lm" => Dict{String, Any}("max_iter" => 2)))) == Dict("lm" => Dict("max_iter" => 2))
    @test last(RC.split_refine_only(Dict{String, Any}("config_version" => 2, "optimizer" => Dict{String, Any}("max_iter" => 2)))) ==
          Dict("optimizer" => Dict("max_iter" => 2))
end
