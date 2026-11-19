# test/runtests.jl's RC_TEST_PRECOMPILE_ONLY modes (the release gate: no compile on a GPU slot). Both modes exit before any item
# is selected, and the guard "a pattern that matches no item is an error" stays the last word of a test run.

@testitem "runtests: RC_TEST_PRECOMPILE_ONLY exits before item selection; the empty-selection guard stays" begin
    src = read(joinpath(@__DIR__, "..", "runtests.jl"), String)
    at(s) = something(findfirst(s, src), 0:-1).start
    check, load, first_tier = at("if PRECOMPILE_ONLY == \"check\""), at("if PRECOMPILE_ONLY == \"1\""), at("include(\"run_tier.jl\")")
    @test 0 < check < at("using ReTestItems") < load < first_tier          # "check" before loading anything, "1" before selection
    @test occursin(r"if PRECOMPILE_ONLY == \"check\"(?s).*?exit\(isempty\(stale\) \? 0 : 1\)", src)
    @test occursin(r"if PRECOMPILE_ONLY == \"1\"(?s).*?import CUDA(?s).*?exit\(0\)", src)
    @test occursin("isempty(PRECOMPILE_ONLY) || error(", src)              # an unknown value is refused, never a silent run
    @test endswith(rstrip(src), "RAN[] || error(\"no test item matches \$(repr(NAME))\")")   # the guard, unchanged and last
end

# test/run_tier.jl: a tier that matches no item is skipped only under a name pattern (that run's items lie in another tier); without
# one, an emptied tier is an error (review finding 3: it went unreported).
@testitem "run_tier: a tier with no item is skipped under a name pattern, an error without one" begin
    import ReTestItems
    include(joinpath(@__DIR__, "..", "run_tier.jl"))
    none = ReTestItems.NoTestException("no test items found")
    @test tier_skipped(none, r"some item")
    @test !tier_skipped(none, nothing)                                     # no pattern: the empty tier is rethrown
    @test !tier_skipped(ErrorException("a failed item"), r"some item")     # any other error, always
    @test !tier_skipped(ErrorException("a failed item"), nothing)
end
