# A `[runs.<id>]` input (`run:<id>`, e.g. `frontend.schoeps = "run:…"`): checked before use, produced on demand, a case's own only
# (owner 2026-11-15, 22:05). The consumer produce_or_loads its upstream: a missing run is made by its driver's resolver (`make`,
# the upstream driver in a module of its own, Eval's run_isolated), under its own key, recursively; a maker that does not produce the
# keyed product is an error naming the run, its key and the command that produces it, never a bare SystemError. A case resolves and
# produces ONLY the runs its own [pipeline] names. A GPU-only upstream made inside a CPU job is refused, never run on the CPU.

@testitem "run: inputs — checked, produced on demand, a case's own upstreams only (owner 2026-11-15)" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
        dir = mktempdir()
        upcfg = joinpath(dir, "fake_upstream.toml")
        write(upcfg, "[experiment]\nname = \"fake_upstream\"\n[things.a]\nn = 1\n[things.b]\nn = 2\n")
        made = Dict("a" => 0, "b" => 0)
        broken = Ref(false)                                         # a maker that produces nothing
        RUN_RESOLVERS["fake_upstream"] = function (cfg, run, path)
            table, name = split(run, '.')
            p = Product("fake_upstream", "detect", Dict{String, Any}("thing" => cfg[table][name]); output = "out.txt")
            make() = broken[] ? nothing :
                produce(d -> (made[name] += 1; write(joinpath(d, "out.txt"), name); nothing), p; driver = "fake_upstream.jl", config_source = path)
            return (; product = p, make, command = "julia --project=experiments fake_upstream.jl --config=$path --run=$run")
        end
        runs(r) = Dict{String, Any}("config" => upcfg, "run" => r, "path" => "out.txt", "stage" => "detect", "provenance" => "a test")
        cfg = Dict{String, Any}("runs" => Dict{String, Any}("A_up" => runs("things.a"), "B_up" => runs("things.b")),
                                "cases" => Dict{String, Any}("A" => Dict{String, Any}("pipeline" => Dict{String, Any}("frontend" => Dict{String, Any}("schoeps" => "run:A_up"))),
                                                             "B" => Dict{String, Any}("pipeline" => Dict{String, Any}("frontend" => Dict{String, Any}("schoeps" => "run:B_up")))))
        # the owner's acceptance test: an upstream product deleted, its consumer run, the product made again under its own key
        p = resolve_pipeline(cfg, cfg["cases"]["A"])
        @test made["a"] == 1 && isfile(p["frontend"]["schoeps"]) && read(p["frontend"]["schoeps"], String) == "a"
        key = run_source(cfg, "A_up").key
        rm(product_dir(run_source(cfg, "A_up")); recursive = true)
        p = resolve_pipeline(cfg, cfg["cases"]["A"])
        @test made["a"] == 2 && isfile(p["frontend"]["schoeps"]) && run_source(cfg, "A_up").key == key
        p = resolve_pipeline(cfg, cfg["cases"]["A"])
        @test made["a"] == 2                                        # produced: loaded, not made again
        # a case resolves and produces only its own upstreams: B's deleted, A's inputs neither refuse nor make it
        @test run_inputs_through(cfg, cfg["cases"]["A"], "start") == [config_source(cfg, "A_up")]
        @test made["b"] == 0 && !produced(run_source(cfg, "B_up"))
        # a maker that does not make the keyed product: refused naming the run, its key and the command that produces it
        broken[] = true
        m = try resolve_pipeline(cfg, cfg["cases"]["B"]); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
        @test occursin("[runs.B_up]", m) && occursin(run_source(cfg, "B_up").key, m) && occursin("fake_upstream.jl --config=", m)
        @test !occursin("SystemError", m)
        broken[] = false
    end
    # a GPU-only upstream inside a CPU job is refused, never run on the CPU (owner: GPU first, no silent fallback)
    m = try require_gpu("the run schoeps.x (Schöps's camera_calibration)", "julia … frontends.jl --stage=prepare,detect --case=x"; has = false); "" catch e; e.msg end
    @test occursin("schoeps.x", m) && occursin("GPU", m) && occursin("--stage=prepare,detect", m)
    @test require_gpu("x", "y"; has = true) === nothing
    @test gpu_listed() isa Bool
end
