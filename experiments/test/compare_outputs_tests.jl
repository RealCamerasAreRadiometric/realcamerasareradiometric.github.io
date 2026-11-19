# The golden gate's comparator (experiments/test/pipeline/compare_outputs.jl) in TOLERANCE mode, the mode every run off the recorded environment
# (another device, Julia or Manifest) is judged by. The exact mode is a bit-for-bit leaf comparison. The tolerance mode reads the records'
# shapes (trajectory records, the grid lens's control nodes), so each golden case is compared with itself and with a perturbed copy.

@testitem "compare_outputs: tolerance mode on every golden case" default_imports = false begin
    using Test
    import JSON
    RC_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
    include(joinpath(RC_ROOT, "experiments", "test", "pipeline", "compare_outputs.jl"))
    golden = joinpath(RC_ROOT, "experiments", "test", "pipeline", "golden")
    cases = sort(filter(c -> isdir(joinpath(golden, c)), readdir(golden)))
    @test cases == ["siemens_coarse_full"]
    for c in cases
        r = compare_case(joinpath(golden, c), joinpath(golden, c); exact = false, io = devnull)
        @test r.bad == 0 && r.n > 0
    end

    # the grid lens (siemens_coarse_full): a moved control node is compared, an unmoved one (no observations: its start) is not
    g = joinpath(golden, "siemens_coarse_full")
    G = JSON.parsefile(joinpath(g, "grid.json")); Gi = JSON.parsefile(joinpath(g, "grid_init.json"))
    ctl = G["cameras"][1]["lens"]["spline"]["controls"]["direction"]; ctl0 = Gi["cameras"][1]["lens"]["spline"]["controls"]["direction"]
    imoved = findfirst(i -> ctl[i] != ctl0[i], eachindex(ctl))
    @test imoved !== nothing
    # the case `gc` (the golden, or a copy whose start holds node `still` at its final value: an unmoved node; on the wave-10 golden
    # every node moved, the grid smoothness coupling them) against its output with node i's first component scaled by 1 + rel
    function perturbed(i, rel; still = nothing)
        gc = g
        if still !== nothing
            gc = mktempdir(); foreach(f -> cp(joinpath(g, f), joinpath(gc, f)), readdir(g))
            Gs = deepcopy(Gi); Gs["cameras"][1]["lens"]["spline"]["controls"]["direction"][still] = deepcopy(ctl[still])
            open(io -> JSON.print(io, Gs), joinpath(gc, "grid_init.json"), "w")
        end
        d = mktempdir(); foreach(f -> cp(joinpath(gc, f), joinpath(d, f)), readdir(gc))
        H = deepcopy(G); H["cameras"][1]["lens"]["spline"]["controls"]["direction"][i][1] *= 1 + rel
        open(io -> JSON.print(io, H), joinpath(d, "grid.json"), "w")
        return compare_case(gc, d; exact = false, io = devnull).bad
    end
    @test perturbed(imoved, 1e-6) == 1        # beyond 1e-9 relative: reported
    @test perturbed(imoved, 1e-12) == 0       # within it
    @test perturbed(imoved, 1e-6; still = imoved) == 0   # an unmoved node carries no information: not compared
end
