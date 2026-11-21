# quality_gpu_first_tests.jl — the GPU-first rule (owner, 2026-11-13: GPU first, configs never pin the CPU; the backend is "gpu", any
# KernelAbstractions GPU backend), checked over this repository: no config or Slurm script pins the CPU outside the test oracles
# listed below, no config names the retired tag "cuda", and no `Threads.@threads` loop in src/ outside the
# reasoned allow-list (a per-pixel step belongs in a KernelAbstractions kernel on the state's backend). The files are the ones
# VisualGeometryToolkitCore's `naming_scope` reads (its quality checkers' one file walk; Core has no GPU-first rule to extend:
# checked here, an upstream candidate).

@testitem "quality: GPU first" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, RadiometricCamera
    root = pkgdir(RadiometricCamera)
    # relative to root: the package's files, and the experiments' beside it in the repository (absent from an installed package)
    exp = normpath(root, "..", "experiments")
    files = [VgtkCore.naming_scope(root).files; isdir(exp) ? [joinpath("..", "experiments", f) for f in VgtkCore.naming_scope(exp).files] : String[]]
    code(line) = first(split(line, '#'; limit = 2))                  # a TOML or shell line without its comment
    hits(f, re) = [(f, i, l) for (i, l) in enumerate(eachline(joinpath(root, f))) if occursin(re, code(l))]

    # 1. the CPU pinned: `backend = "cpu"`, `device = "cpu"`, an empty CUDA_VISIBLE_DEVICES — only in the CPU test oracles
    CPU_ORACLES = Dict(
        "../experiments/test/pipeline/run_pipeline_checks.sh" => "the pipeline checks: the configs (\"gpu\") run with no visible GPU, compared with the goldens recorded on the CPU",
        "../experiments/scripts/slurm/golden.sbatch" => "the CPU golden calibration cases, compared with experiments/test/pipeline/golden (recorded on the CPU)",
        "../experiments/scripts/slurm/prepare.sbatch" => "the RAW decode (Features' load_raw, LibRaw on the CPU) and the input checks: no device stage",
        "../experiments/test/fixtures/thorlabs/chrome_evaluation.toml" => "the chrome evaluation test's fixture: its detect on the CPU oracle")
    pin = r"^\s*(backend|device)\s*=\s*\"cpu\"|CUDA_VISIBLE_DEVICES=(\"\"|''|\s|$)"
    pins = [h for f in files if endswith(f, ".toml") || endswith(f, ".sbatch") || endswith(f, ".sh") for h in hits(f, pin) if !haskey(CPU_ORACLES, f)]
    @test isempty(pins)
    isempty(pins) || foreach(h -> println("CPU pinned: ", h[1], ":", h[2], "  ", strip(h[3])), pins)

    # 2. the retired tag "cuda" as a config value — except PanoramaStitching's own keys in the sweeps' configs (its example reads
    # `[example.compute] device` and `[example.registration] execution_provider` in its own spelling)
    PS_KEYED = ("../experiments/configs/experiments/panorama_vignetting_fx3_14mm.toml", "../experiments/configs/experiments/panorama_vignetting_fx3_fisheye.toml",
                "../experiments/configs/experiments/panorama_vignetting_fx3_tamron.toml")
    cuda = [h for f in files if endswith(f, ".toml") && !(f in PS_KEYED) for h in hits(f, r"=\s*\"cuda\"")]
    @test isempty(cuda)
    isempty(cuda) || foreach(h -> println("retired tag \"cuda\": ", h[1], ":", h[2], "  ", strip(h[3])), cuda)

    # 3. `Threads.@threads` in src/ (the calibration, its start and diagnostics; the former src/calibration_pipeline/): only the reasoned host loops (file => enclosing functions and why)
    # (none since 2026-11-13: the LM's Jacobian is assembled by a device row kernel and its Gauss–Newton blocks formed on the backend —
    # device/jacobian_kernels.jl `_jacobian_row!`, `_gauss_newton_blocks`; the host scatters of the per-view and per-tile Jacobians are gone)
    THREADED = Dict{String, Tuple{Tuple{Vararg{String}}, String}}()
    function threaded(f)
        out = Tuple{String, Int, String}[]; fn = ""
        for (i, l) in enumerate(eachline(joinpath(root, f)))
            m = match(r"^(?:@inline\s+)?function\s+([^\s(]+)", l); m === nothing || (fn = m.captures[1])
            occursin("Threads.@threads", code(l)) && push!(out, (f, i, fn))
        end
        return out
    end
    pipe = filter(f -> startswith(f, "src/") && endswith(f, ".jl"), files)
    loops = [h for f in pipe for h in threaded(f) if !(haskey(THREADED, f) && h[3] in THREADED[f][1])]
    @test isempty(loops)
    isempty(loops) || foreach(h -> println("Threads.@threads outside the allow-list: ", h[1], ":", h[2], " in ", h[3]), loops)
end
