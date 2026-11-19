# Write experiments/test/pipeline/golden/RECORDED.toml: the environment the goldens were recorded in (run by run_pipeline_checks.sh --record).
# compare_outputs.jl compares bit for bit only in this environment (same Julia major.minor, same Manifest sha256), otherwise within
# tolerances.   julia --project=RadiometricCamera experiments/test/pipeline/record_environment.jl <golden root> <thread count> <device: gpu | cpu>
import TOML, SHA, Dates
root, threads, device = ARGS
manifest = Base.project_file_manifest_path(Base.active_project())
m = TOML.parsefile(manifest)
entry(name) = only(m["deps"][name])
# a registered stack package is named by its version and tree hash; a path-sourced one (a checkout) by its directory and commit
function stack_source(name)
    e = entry(name)
    src = if haskey(e, "path")
        dir = normpath(joinpath(dirname(manifest), e["path"]))
        commit = try
            readchomp(`git -C $dir rev-parse --short HEAD`)
        catch
            "unknown"
        end
        "at $(e["path"]) (commit $commit)"
    else
        "registered (git-tree-sha1 $(e["git-tree-sha1"]))"
    end
    return "$(e["version"]) $src"
end
rec = Dict{String, Any}(
    "julia_version" => string(VERSION), "julia_minor" => "$(VERSION.major).$(VERSION.minor)",
    "manifest" => basename(manifest), "manifest_sha256" => bytes2hex(SHA.sha256(read(manifest))),
    "nllssolver_version" => entry("NLLSsolver")["version"], "staticarrays_version" => entry("StaticArrays")["version"],
    "forwarddiff_version" => entry("ForwardDiff")["version"], "stablerngs_version" => entry("StableRNGs")["version"],
    "visualgeometrytoolkitcore" => stack_source("VisualGeometryToolkitCore"), "visualgeometrytoolkitio" => stack_source("VisualGeometryToolkitIO"),
    "visualgeometrytoolkitestimation" => stack_source("VisualGeometryToolkitEstimation"), "panoramastitching" => stack_source("PanoramaStitching"),
    "date" => string(Dates.today()), "julia_num_threads" => parse(Int, threads), "device" => device)
open(joinpath(root, "RECORDED.toml"), "w") do io
    println(io, "# The environment the pipeline goldens were recorded in (experiments/test/pipeline/record_environment.jl). compare_outputs.jl compares")
    println(io, "# bit for bit only under the same Julia major.minor and Manifest sha256; elsewhere within its tolerances.")
    TOML.print(io, rec; sorted = true)
end
println("    recorded environment: Julia $(VERSION), $(basename(manifest)) $(first(rec["manifest_sha256"], 12)), device $device")
