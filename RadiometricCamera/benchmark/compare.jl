# Two benchmark records (benchmark/run.jl's `out`) side by side, as a Markdown table: per stage the best time of each, their ratio
# (before / after: > 1 is faster), the host allocation and whether the stage's value (its checksum) is the same number — the
# relative difference when it is not.
#
#   julia --project=benchmark benchmark/compare.jl before.json after.json
import JSON
using Printf

function compare(before_path, after_path)
    a = JSON.parsefile(before_path); b = JSON.parsefile(after_path)
    describe(r) = @sprintf("%s (%s, %s, %d × %d, %d views, %d px)", r["label"], r["commit"], r["device"], r["size"]..., r["views"], r["masked_pixels"])
    println("before: ", describe(a)); println("after:  ", describe(b)); println()
    println("| stage | before (s) | after (s) | speed-up | bytes before | bytes after | value |")
    println("|---|---:|---:|---:|---:|---:|---|")
    after = Dict(s["stage"] => s for s in b["stages"])
    for s in a["stages"]
        t = get(after, s["stage"], nothing)
        t === nothing && continue
        va, vb = s["value"], t["value"]
        same = va == vb || (va === nothing && vb === nothing)
        value = same ? "same" : (va isa Real && vb isa Real) ? @sprintf("rel. diff. %.2e", abs(vb - va) / max(abs(va), floatmin())) : "differs"
        @printf("| %s | %.4f | %.4f | %.2f× | %d | %d | %s |\n", s["stage"], s["best"], t["best"], s["best"] / t["best"], s["bytes"], t["bytes"], value)
    end
end

length(ARGS) == 2 || error("usage: julia --project=benchmark benchmark/compare.jl before.json after.json")
compare(ARGS...)
