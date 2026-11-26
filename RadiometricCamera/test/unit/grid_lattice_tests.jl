# The geometry lattice's mode, `[grid] lattice` (owner ruling 2026-11-15 19:55): "derived" — the start source's own lattice
# (T. Schöps's camera) extended over the lens coverage, never subdivided (m = 1) — or "node_count" — laid over the image region and
# sized by `grid.node_count` (a parametric start). Parsed ONCE into a type (`LATTICE_TAGS`, `read_switch`); every decision a dispatch.
# And the general rule: a TOML value the run's input makes unusable is an ERROR naming the key and the reason, never ignored
# (`INAPPLICABLE_KEYS`): one test per refusal.

@testitem "grid lattice: parsed once into a type, the writer round-trips" setup = [RCFixtures] begin
    using RadiometricCamera: DerivedLattice, NodeCountLattice, LATTICE_TAGS, config_dict, pipeline_options
    @test RC.PipelineOptions().grid.lattice === DerivedLattice() && RC.PipelineOptions().grid.node_count === nothing   # default.toml: "derived"
    o = pipeline_options(Dict("grid" => Dict("lattice" => "node_count", "node_count" => 875)))
    @test o.grid.lattice === NodeCountLattice() && o.grid.node_count == 875
    d = config_dict(o)
    @test d["grid"]["lattice"] == "node_count" && d["grid"]["node_count"] == 875 && pipeline_options(d) == o
    od = pipeline_options(Dict("grid" => Dict("lattice" => "derived")))     # no target: m = 1
    @test od.grid.lattice === DerivedLattice() && !haskey(config_dict(od)["grid"], "node_count") && pipeline_options(config_dict(od)) == od
    @test collect(keys(LATTICE_TAGS)) == ["derived", "node_count"]
end

@testitem "grid lattice: every inapplicable combination refused, naming the key and the reason" setup = [RCFixtures] begin
    msg(d) = try RC.pipeline_options(d); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    # an unknown mode, and the TOML boolean of a switch
    @test occursin("`grid.lattice`", msg(Dict("grid" => Dict("lattice" => "fitted")))) && occursin("\"derived\"", msg(Dict("grid" => Dict("lattice" => "fitted"))))
    # "derived" reads a node count as its target of present nodes (owner 2026-11-15, 23:25)
    @test RC.pipeline_options(Dict("grid" => Dict("lattice" => "derived", "node_count" => 875))).grid.node_count == 875
    # "node_count" needs its count
    m = msg(Dict("grid" => Dict("lattice" => "node_count")))
    @test occursin("grid.node_count", m) && occursin("node_count", m)
    # "node_count" below Core's smallest lattice
    @test occursin("grid.node_count", msg(Dict("grid" => Dict("lattice" => "node_count", "node_count" => 9))))
    # a Schöps start is his lattice: "node_count" refused
    m = msg(Dict("frontend" => Dict("type" => "schoeps"), "grid" => Dict("lattice" => "node_count", "node_count" => 875)))
    @test occursin("grid.lattice", m) && occursin("Schöps", m) && occursin("\"derived\"", m)
    # the retired keys, naming the new spelling
    m = msg(Dict("grid" => Dict("size" => [35, 25])))
    @test occursin("`grid.size` is retired", m) && occursin("grid.lattice", m) && occursin("node_count", m)
    m = msg(Dict("grid" => Dict("lattice_spacing" => 131.0)))
    @test occursin("`grid.lattice_spacing` is retired", m) && occursin("lattice = \"derived\"", m)
    # a parametric record start (io.init = "lens:…") is refined per view: it has no lattice, so every grid key is refused
    for (k, v) in ("lattice" => "derived", "node_count" => 875, "smoothness" => Dict("type" => "none"), "nodes" => "all",
                   "start_perturbation" => Dict("sigma_px" => 0.5, "seed" => 1))
        m = msg(Dict("io" => Dict("init" => "lens:x.json"), "grid" => Dict(k => v)))
        @test occursin("`grid.$k`", m) && occursin("lens:", m)
    end
    # the focus surface held: its smoothness has nothing to act on
    m = msg(Dict("fixed" => ["focus"], "focus" => Dict("smoothness" => Dict("type" => "none"))))
    @test occursin("`focus.smoothness`", m) && occursin("fixed", m)
    # the applicable spellings read
    @test RC.pipeline_options(Dict("io" => Dict("init" => "lens:x.json"))).grid == RC.GridOptions()
end

@testitem "grid lattice: the start dispatches on the source and the mode" setup = [RCFixtures] begin
    using RadiometricCamera: DerivedLattice, NodeCountLattice, _start_grid
    W, H = 640, 480
    kb = cam(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.0, 0.0), 300.0, (320.0, 240.0), W, H)
    # a parametric source has no lattice of its own: "derived" refused at its start, naming the source and the fix
    m = try _start_grid(kb, RC.GridOptions()); "" catch e; e.msg end
    @test occursin("grid.lattice = \"derived\"", m) && occursin("node_count", m)
    @test _start_grid(kb, RC.GridOptions(; lattice = NodeCountLattice(), node_count = 150)).resample === nothing
end
