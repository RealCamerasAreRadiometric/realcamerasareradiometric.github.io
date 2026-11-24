# An upstream product is PRODUCED ON DEMAND by its consumer, never refused with "run its stage first" (owner 2026-11-15, RC 0.11.4):
# the one mechanism (src/drivers.jl `produce_upstream`: `with_upstream`, the GPU gate, the post-check) and its maker for another
# driver (`driver_run`); every consumer → upstream pair goes through it (a table over the pairs); and a lock that no refusal of an
# upstream comes back (`require_produced` is gone; the one allowed `produced(…) || error` is the host's --gpu gate).

@testitem "upstream products: produced on demand through one mechanism, every consumer's, no refusal (owner 2026-11-15)" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    msg(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end

    @testset "the mechanism: made once under its key, loaded after, a GPU's refused in a CPU process, a maker that fails named" begin
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
            n = Ref(0)
            p = Product("up", "detect", Dict{String, Any}("n" => 1); output = "out.txt")
            make() = produce(d -> (n[] += 1; write(joinpath(d, "out.txt"), "x"); nothing), p; driver = "up.jl", config_source = "up.toml")
            @test produce_upstream(p, "the consumer's upstream"; make, command = "julia up.jl --stage=detect") == product_dir(p) && n[] == 1
            @test produce_upstream(p, "the consumer's upstream"; make, command = "julia up.jl --stage=detect") == product_dir(p) && n[] == 1  # loaded
            # a GPU stage in a CPU process: refused naming the command, never made (owner: GPU first, no CPU fallback)
            q = Product("up", "detect", Dict{String, Any}("n" => 2); output = "out.txt")
            m = msg(() -> produce_upstream(q, "the run q"; make = () -> (n[] += 1), command = "julia up.jl --stage=detect --gpu-job", gpu = true, has = false))
            @test occursin("the run q", m) && occursin("needs a GPU", m) && occursin("--gpu-job", m) && n[] == 1 && !produced(q)
            # in a GPU process the same stage is made
            mq() = produce(d -> (n[] += 1; write(joinpath(d, "out.txt"), "y"); nothing), q; driver = "up.jl", config_source = "up.toml")
            @test produce_upstream(q, "the run q"; make = mq, command = "c", gpu = true, has = true) == product_dir(q) && n[] == 2
            # a maker that does not produce it: named with its key and the command that does
            r = Product("up", "detect", Dict{String, Any}("n" => 3); output = "out.txt")
            m = msg(() -> produce_upstream(r, "the run r"; make = () -> nothing, command = "julia up.jl --stage=detect --case=r"))
            @test occursin("the run r", m) && occursin(r.key, m) && occursin("--case=r", m) && occursin("did not make it", m)
        end
        # another driver's maker: its script on the config through the stages, one case, and the same command line
        d = driver_run("scripts/frontends.jl", "configs/experiments/frontends.toml", "prepare,detect"; case = "s")
        @test d.command == "julia --project=experiments experiments/scripts/frontends.jl --config=$(projectdir("configs/experiments/frontends.toml")) --stage=prepare,detect --case=s"
        @test d.make isa Function
    end

    @testset "every consumer → upstream pair reads it through produce_upstream (the table of pairs)" begin
        # (file, the consumer's call that reads its upstream): each wrapper is produce_upstream with the upstream's own producer
        src(f) = read(projectdir(f), String)
        pairs = [("src/frontends.jl", "produce_upstream(frontend_decode_product(cfg, t[\"images\"])"),        # Schöps run ← decode
                 ("src/frontends.jl", "frontend_upstream_schoeps(cfg, t[\"detections\"]"),                    # fit ← Schöps run (GPU)
                 ("src/frontends.jl", "frontend_upstream_schoeps(cfg, t[\"run\"]"),                           # compare ← Schöps run
                 ("src/frontends.jl", "produce_upstream(frontend_fit_product(cfg, t[\"run\"])"),             # compare ← fit
                 ("src/drivers.jl", "require_prepared(cfg, t, \"start \$name\"; driver, config_path)"),       # start ← decode
                 ("src/drivers.jl", "require_prepared(cfg, cfg[\"cases\"][case], \"case \$case\"; driver, config_path)"),   # refine ← decode
                 ("src/drivers.jl", "idir = init_start_dir(cfg, _start_init(cfg, name), name; driver, config_path)"),       # start ← init start
                 ("src/drivers.jl", "idir = init_start_dir(cfg, _start_init(cfg, case), case; driver, config_path)"),       # refine ← init start
                 ("src/drivers.jl", "produce_upstream(r.product"),                                            # a [runs.<id>] input
                 ("src/material_grid.jl", "produce_upstream(start_product(cfg, case)"),                       # material grid ← the case's start (GPU)
                 ("src/material_grid.jl", "require_prepared(cfg, t, \"the material grid of case \$case\"; driver, config_path)"),   # material grid ← decode
                 ("src/material_grid.jl", "produce_upstream(material_grid_product(cfg, case)"),               # the refinement's start record ← material grid (GPU)
                 ("src/drivers.jl", "gdir = material_grid_dir(cfg, case; driver, config_path)"),              # refine ← material grid
                 ("scripts/calibrate.jl", "config_source(CFG, id) for id in table_runs(CFG[\"pipeline\"])"),     # golden case ← its distance map ([runs])
                 ("scripts/calibrate.jl", "resolved(table) = resolve_pipeline(CFG"),                          # its [pipeline]'s run:<id>, produced
                 ("src/chrome_evaluation.jl", "chrome_upstream(cfg, path, \"detect\""),                      # index ← detect
                 ("src/chrome_evaluation.jl", "chrome_upstream(cfg, path, \"index\""),                       # evaluate ← index
                 ("src/convergence.jl", "refinement_upstream(rc, cfg[\"upstream\"][\"radiometric_refinement\"], case, \"refine\""),   # ← refine
                 ("src/convergence.jl", "refinement_upstream(rc, cfg[\"upstream\"][\"radiometric_refinement\"], case, \"start\""),    # ← start
                 ("src/convergence.jl", "chrome_upstream(cc, cpath, \"index\""),                             # ← the chrome index
                 ("src/warping_ablation.jl", "require_prepared(plan.rc, t, \"case \$(r.variant)\"; driver, config_path)"),   # ← decode
                 ("src/warping_ablation.jl", "foreach(id -> config_source(plan.rc, id), table_runs(t[\"pipeline\"]))"),       # ← the front ends' runs
                 ("src/warping_ablation.jl", "chrome_upstream(cc, cpath, \"index\""),                     # ← the chrome index
                 ("src/warping_ablation.jl", "produce_upstream(refine_product(plan.rc, r.variant)"),       # scoring ← refinement (GPU)
                 ("scripts/warping_ablation.jl", "produce_upstream(warping_evaluate_product(CFG, PLAN, k)"),   # record ← scoring
                 ("scripts/warping_ablation.jl", "produce_upstream(rp, \"the warping ablation's record\""),     # figure ← record
                 ("src/image_region.jl", "produce_upstream(image_region_product(cfg, n)"),                  # every lens_coverage consumer
                 ("scripts/chrome_evaluation.jl", "produce_upstream(prepare_product(CFG, DETECT.photos)"),   # detect ← decode
                 ("scripts/convergence.jl", "produce_upstream(convergence_product(CFG, case)"),               # figure ← record
                 ("scripts/texture_recovery.jl", "produce_upstream(texture_product()"),                       # report ← the page solve
                 ("scripts/texture_recovery.jl", "produce_upstream(report_product()"),                        # the paper's bands ← the report
                 ("scripts/texture_recovery.jl", "produce_upstream(poses_product()"),                         # the paper's bands ← the poses
                 ("scripts/board_warping.jl", "refinement_upstream(RC_CFG, RC_PATH, case, \"refine\""),          # the board warping's figure ← the refinements
                 ("scripts/panorama_stitching.jl", "refinement_upstream(REFINEMENT, REFINEMENT_PATH, CFG[\"maps\"][m][\"case\"], \"refine\""),   # the learned maps ← the refinements
                 ("scripts/panorama_stitching.jl", "produce_upstream(rooftop_product()"),                     # the panoramas ← the sweep's run (GPU)
                 ("scripts/panorama_stitching.jl", "produce_upstream(evaluate_product()"),                    # the panoramas ← the learned maps
                 ("src/panorama_stitching.jl", "produce_upstream(panoramas_product(cfg, up)"),                # a figure ← a lens's panoramas
                 ("scripts/panorama_figure.jl", "panoramas_upstream(SOURCES[k].config"),                      # the panoramas figure ← the lenses' panoramas
                 ("scripts/teaser.jl", "panoramas_upstream(PANORAMAS.config"),                                # the teaser ← the lens's panoramas
                 ("scripts/teaser.jl", "produce_upstream(BANDS.product"),                                     # the teaser ← the learned-texture figure's tables
                 ("scripts/detector_bias.jl", "produce_upstream(start_product(CFG, c)"),                      # the detector bias's evaluation ← its start
                 ("scripts/detector_bias.jl", "produce_upstream(refine_product(CFG, c)"),                     # ← its refinement (GPU)
                 ("scripts/detector_bias.jl", "produce_upstream(detector_bias_evaluate_product(CFG, c)")]     # the detector bias's figure ← the evaluations
        for (f, call) in pairs
            @test occursin(call, src(f))
        end
        # the wrappers are produce_upstream themselves
        for (f, w) in (("src/frontends.jl", "frontend_upstream_schoeps"), ("src/drivers.jl", "init_start_dir"), ("src/drivers.jl", "refinement_upstream"),
                       ("src/drivers.jl", "function require_prepared"), ("src/chrome_evaluation.jl", "function chrome_upstream"))
            s = src(f); i = findfirst(w, s)
            @test i !== nothing && occursin("produce_upstream(", s[first(i):min(end, first(i) + 1200)])
        end
    end

    @testset "the lock: no refusal of an upstream product comes back" begin
        files = [joinpath(r, f) for d in ("src", "scripts") for (r, _, fs) in walkdir(projectdir(d)) for f in fs if endswith(f, ".jl")]
        text = Dict(relpath(f, projectdir()) => read(f, String) for f in files)
        @test isempty([f for (f, s) in text if occursin("require_produced", s)])                       # the refusal is gone
        @test isempty([f for (f, s) in text if occursin(r"run its stage[^\n]*first", s) && f != "src/drivers.jl"])
        # `produced(p) || throw/error`: only produce_upstream's post-check and the host's --gpu gate (src/assets.jl host_stage: a GPU
        # stage of an isolated driver without --gpu, the owner's GPU-first refusal); an input with no producer is config_input's
        refusals = [(f, m.match) for (f, s) in text for m in eachmatch(r"produced\([^\n]*\|\|\s*(throw|error)", s)]
        @test sort(first.(refusals)) == ["src/assets.jl", "src/drivers.jl"]
    end
end
