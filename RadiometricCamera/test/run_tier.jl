# test/run_tier.jl — the one ReTestItems tier runner of the package's suite (test/runtests.jl) and the paper project's
# (experiments/test/runtests.jl), included by both. Under a name pattern a tier may match no item (the pattern's items lie in another
# tier), which ReTestItems reports by throwing; that tier is then skipped, and only a pattern that matches no item in any tier is an
# error (the caller's `RAN[] || error(…)`). Without a pattern every tier holds items, and an empty one is an error.
using ReTestItems: ReTestItems, runtests

"""
    run_tier(select, target, name, ran::Ref{Bool}; kw...)

One tier of a suite: ReTestItems' `runtests(select, target; name, testitem_timeout = 3600, kw...)` (RadiometricCamera's item
timeout, BlobBoards' setting); `ran[]` is set when the tier ran. A tier that matches no item under the name pattern `name` is
skipped (`tier_skipped`); without a pattern (`name === nothing`) a tier with no item is an error, never a silent success.
"""
function run_tier(select, target, name, ran::Ref{Bool}; kw...)
    try
        runtests(select, target; name, testitem_timeout = 3600, kw...)
        ran[] = true
    catch e
        tier_skipped(e, name) || rethrow()
    end
end

"""`tier_skipped(e, name)`: the error `e` of a tier's run is its skip: no item matched the name pattern `name` (never `nothing`)."""
tier_skipped(e, name) = e isa ReTestItems.NoTestException && name !== nothing
