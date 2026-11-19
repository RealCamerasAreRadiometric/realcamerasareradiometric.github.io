# The one support rule of a spline's nodes (camera/node_support.jl); and the CVT node layout's removal (RC 0.11.4, owner 2026-11-16).

@testitem "node support: one rule for every spline's nodes" setup = [RCFixtures] begin
    using RadiometricCamera: data_weight, node_support, focus_node_support, Focus, FocusSpline
    using StaticArrays
    r = node_support([2.0, 1.0, 0.05, 0.0, 0.95], 0.1)
    @test r.weight ≈ [2.0, 1.0, 0.05, 0.0, 0.95] ./ 0.8 && r.supported == [true, true, false, false, true]
    @test node_support([1.0, 0.0, 0.01], 0.0).supported == [true, false, true]                           # 0: any pixel
    # the data weight Σ b_i(𝐩)² over the masked pixels (per view, Julia pixel positions)
    @test data_weight(p -> (1.0, 2.0), [[(1, 1), (2, 2)], [(3, 3)]], 2) == [3.0, 12.0]
    # the focus TPS's nodes: pixels only on the left two columns of a 3 × 3 node set — the right column is dependent
    ns = [SVector(x, y) for y in (10.0, 50.0, 90.0) for x in (10.0, 50.0, 90.0)]
    s = focus_node_support(Focus(FocusSpline(ns), fill(1.4, 9)), [[(u, v) for u in 1:60 for v in 1:100]], 0.1)
    @test s.supported == Bool[1, 1, 0, 1, 1, 0, 1, 1, 0]
    # the vignetting's rule is the same implementation
    src = read(joinpath(pkgdir(RadiometricCamera), "src", "camera", "vignetting.jl"), String)
    @test occursin("node_support(w, support_min)", src) && !occursin("rel .>= ", src)
end

@testitem "node layout: not in 0.11.4 (no pipeline step called it); Clustering is no dependency" setup = [RCFixtures] begin
    # owner ruling 2026-11-16: the CVT node layout had no caller in src/, bin/ or experiments/ and brought Clustering into a patch release
    import TOML
    @test !isdefined(RadiometricCamera, :node_layout)
    p = TOML.parsefile(joinpath(pkgdir(RadiometricCamera), "Project.toml"))
    @test !haskey(p["deps"], "Clustering") && !haskey(p["compat"], "Clustering")
end
