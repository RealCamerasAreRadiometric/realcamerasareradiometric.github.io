# =============================================================================
# RadiometricCamera's benchmark: every stage of a calibration, timed on one synthetic scene.
# =============================================================================
#
#   julia -t 8 --project=benchmark benchmark/run.jl [key=value …]        (from RadiometricCamera/)
#
#     backend=cpu|gpu     the state's device (`rc_backend`); the forward stages that take a backend run on it too      [cpu]
#     size=320x240        the image, pixels                                                                             [320x240]
#     views=3             the photographs                                                                               [3]
#     pixels=0            the measured pixels per view: every `n`-th masked pixel so that about `pixels` remain (a real
#                         run samples 8,192 to 16,384 per photograph); 0: every masked pixel                             [0]
#     warp=0|1            the board warped (a saddle of nodal heights, held) instead of flat: every ray then meets the height
#                         field by Core's Newton instead of the plane                                                    [0]
#     heights=0|1         the board warping's nodal heights free in the LM stages and the Jacobians (`height_degree = 1`)  [0]
#     grid=20x15          the grid stages' spline model: its lattice of controls (multiples of 5; a run's is 65x50)       [20x15]
#     reps=3              timed runs per stage after the first (the minimum is reported)                                [3]
#     stages=<regex>      only the stages whose name matches                                                            [all]
#     out=<file.json>     the results as a record (benchmark/compare.jl reads two of them)                              [none]
#     profile=<stage>     a flat CPU profile of that stage beside `out` (<out>.<stage>.profile.txt)                     [none]
#     label=<text>        a label for the record (a commit, a variant)                                                  [""]
#
# The environment is benchmark/Project.toml, a member of the package's workspace (it resolves with the package; no Manifest is
# committed): `julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'` once.
#
# The scene is the README's: a W × H double-sphere fisheye over a 2 m checkerboard, `views` photographs rendered at a known
# camera, material and lighting; the timed states start from a perturbed camera (a flat vignetting, another μ and lighting), as a
# calibration does. The grid stages see the same scene through the fisheye's spline model (`grid_model`, `grid` controls).
#
# Per stage: `first` (the first call: compilation and caches), `best` (the minimum of `reps` warm calls), `bytes` (the host
# allocation of that call) and `value` (a checksum of the stage's result: two records of the same arithmetic agree in it, so a
# change that moves numbers shows here before any test does). A device stage is timed to the value on the host (its kernels
# awaited). The stages:
#
#   forward        mask, render (on the backend), pixel (1000 × Φ_pixel), flux (the flux kernel of one view), cost (every view's residual)
#   closed form    parts (all_parts), radiometry_normal_equations, linear_blocks (Θ_l → Θ_v → μ), linear_joint (the joint LM)
#   per view       view_jacobian (record, dual chunks, parts, rows), view_products (JᵀJ, Jᵀr of that Jacobian),
#                  view_normal_equations (the two), lm1_view, lm3_view (_lm_views!, 1 and 3 iterations)
#   per tile       tile_normal_equations (every tile of one view: the batch and the tiles' rows and products),
#                  tile_rows_products (the tiles' rows and products alone, the view's batch memoized; its value the cost sum
#                  plus the number of tiles), lm1_grid
#   calibration    the joint closed form, then three LM iterations
#
# `inferred` lists the return type inference gives for the entry points (a non-concrete one is a type instability at the call).
# =============================================================================
using RadiometricCamera, VisualGeometryToolkitCore, VisualRadiometryToolkitCore, StaticArrays, Printf   # (the README's three)
using VisualGeometryToolkitCore: cost
import JSON, NLLSsolver, Profile
import KernelAbstractions as KA
const RC = RadiometricCamera

option(key, default) = (i = findfirst(a -> startswith(a, key * "="), ARGS); i === nothing ? default : String(split(ARGS[i], "="; limit = 2)[2]))
const BACKEND = option("backend", "cpu")
const W, H = parse.(Int, split(option("size", "320x240"), "x"))
const VIEWS = parse(Int, option("views", "3"))
const PIXELS = parse(Int, option("pixels", "0"))
const WARP = option("warp", "0") == "1"
const HEIGHT_DEGREE = option("heights", "0") == "1" ? 1 : 0
const GRID = Tuple(parse.(Int, split(option("grid", "20x15"), "x")))
const REPS = parse(Int, option("reps", "3"))
const STAGES = Regex(option("stages", "."))
const OUT = option("out", "")
const PROFILE = option("profile", "")
const LABEL = option("label", "")
const BE = RC.rc_backend(BACKEND; fallback = false)          # top level: the vendor package's world

# ---- the scene ---------------------------------------------------------------------------------------------------------------------

function benchmark_board()
    texture = [Float64(isodd(fld(i - 1, 32) + fld(j - 1, 32))) for i in 1:256, j in 1:256, c in 1:3]
    board = RC.TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], texture; facing = [0.0, 0.0, 1.5])
    f = 30.9 * W / 24
    K = CameraCalibrationMat(f, 0.0, 0.0, 0.0, f, 0.0, W / 2, H / 2, 1.0)
    lens = BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.25, 0.4))), Size2(W, H))
    surface = WARP ? RC.TargetState(; h = SVector{RC.HEIGHT_NC}([VgtkCore.Synthetic.saddle_height(𝐛) for 𝐛 in RC.HEIGHT_NODES])) : RC.TargetState{Float64}()   # Core's saddle (metres on the 2 m board)
    board = RC.with_state(board, surface)
    settings = RC.RenderSettings(; n_𝐩 = 2, n_ρ = 3, n_φ = 8, N = 4)
    poses = [lookat_pose(Vec3(0.3cos(k), 0.3sin(k), 1.4 + 0.1k), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)) for k in 1:VIEWS]
    b = RC.𝔟(board); r_SH = RC.sphere_radius(b, [p.t for p in poses])
    xs = range(-0.5, 0.5; length = 4)
    flat = TensorProductVignetting((4, 4), Rect(0, 0, W, H); degree = 2)
    vig = VisualRadiometryToolkitCore.with_control_points(flat, [1 - 0.4 * (x^2 + y^2) for x in xs, y in xs])
    a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2)
    light = EnvironmentLight(SphericalHarmonicMap(a, 2), b, r_SH)
    mat = RC.Material(; diffuse_weight = 0.4)
    camera = RC.RadiometricCameraModel(lens, RC.Focus(3, 1.5, W, H), 0.04, vig)
    cpu = KA.CPU()
    masks = [RC.compute_mask(camera, p, board, settings; δ = 0.02, backend = cpu) for p in poses]
    PIXELS > 0 && (masks = map(thinned, masks))
    photos = [RC.render_view(k, camera, poses[k], mat, light, board, settings, masks[k]; backend = cpu) for k in 1:VIEWS]   # the CPU's photographs on every device
    start_light = EnvironmentLight(SphericalHarmonicMap(0.5, 2), b, r_SH)
    # the timed states: a perturbed start on the benchmark's device
    state(lens_ = lens) = RC.CalibrationState(board, settings, RC.RadiometricCameraModel(lens_, camera.focus, camera.aperture_radius, flat), poses,
                                              RC.Material(; diffuse_weight = 0.7), start_light, photos; masks, backend = BE)
    return (; board, settings, poses, light, mat, camera, masks, photos, state, lens)
end

"The mask with every `n`-th of its pixels kept, about `PIXELS` of them."
function thinned(mask)
    ids = findall(mask); n = max(1, length(ids) ÷ PIXELS)
    out = falses(size(mask)); out[ids[1:n:end]] .= true
    return out
end

# ---- the measurement ---------------------------------------------------------------------------------------------------------------

checksum(x::Real) = Float64(x)
checksum(x::Bool) = Float64(x)
checksum(x::Union{Tuple, NamedTuple}) = sum(checksum, values(x); init = 0.0)
checksum(x::AbstractArray{<:Real}) = Float64(sum(x; init = zero(eltype(x))))
checksum(x::AbstractArray) = sum(checksum, x; init = 0.0)
checksum(x) = NaN

"A stage's result `value` with the seconds of the part of it that is the stage (the rest its preparation)."
struct Span{V}
    value::V
    seconds::Float64
end
span(v, seconds) = v, seconds
span(v::Span, seconds) = v.value, v.seconds

function measure(f)
    GC.gc()
    t0 = time_ns(); v, first = span(f(), (time_ns() - t0) / 1e9)
    best = Inf; bytes = 0
    for _ in 1:REPS
        GC.gc(false)
        t = @timed f()
        _, seconds = span(t.value, t.time)
        seconds < best && (best = seconds; bytes = t.bytes)
    end
    return (; first, best, bytes, value = checksum(v))
end

"The return type inference gives `f(args...)`, as a string, and whether it is concrete."
inferred(f, args...) = (T = Base.promote_op(f, map(typeof, args)...); (; type = string(T), concrete = isconcretetype(T)))

# ---- the stages --------------------------------------------------------------------------------------------------------------------

"The first data block of `problem` of the residual type `R`, its variables and the LM's flags (every variable free)."
function first_block(problem, blocks)
    r = first(blocks)
    vars = NLLSsolver.getvars(r, problem.variables)
    return r, vars, 2^length(vars) - 1
end
view_blocks(problem) = (T = only(T for T in keys(problem.problem.costs) if T <: RC.PerViewResidual); get(problem.problem.costs, T))

function stages(c)
    k = 1; u, v = W ÷ 2, H ÷ 3
    list = Pair{String, Any}[]
    add(name, make) = occursin(STAGES, name) && push!(list, name => make)
    # forward
    add("mask", () -> () -> count(RC.compute_mask(c.camera, c.poses[k], c.board, c.settings; δ = 0.02, backend = BE)))
    render(; kw...) = RC.render_view(k, c.camera, c.poses[k], c.mat, c.light, c.board, c.settings, c.masks[k]; kw...)
    on_backend = hasmethod(RC.render_view, Tuple{map(typeof, (k, c.camera, c.poses[k], c.mat, c.light, c.board, c.settings, c.masks[k]))...}, (:backend,))
    add("render", () -> () -> on_backend ? render(; backend = BE) : render())      # (before render_view took a backend: the CPU's)
    add("pixel", () -> () -> (s = zero(SVector{3, Float64});
                              for i in 1:1000; s += RC.Φ_pixel(u, v, k, c.camera, c.poses[k], c.mat, c.light, c.board, c.settings); end; s))
    add("flux", () -> (s = c.state(); () -> RC._flux_device(s, k, RC.view_of(s, k), s.camera.vignetting, s.mat, s.light)))
    add("cost", () -> (s = c.state(); () -> cost(s)[1]))
    # closed form
    add("parts", () -> (s = c.state(); () -> length(RC.all_parts(s))))
    add("radiometry_normal_equations", () -> (s = c.state(); parts = RC.all_parts(s); () -> RC.radiometry_normal_equations(s, parts)))
    radiometry(s, alg) = VgtkCore.solve(VgtkCore.EstimationProblem(RC.RadiometryBlocks, (s, RC.all_parts(s))), alg)
    add("linear_blocks", () -> () -> (s = c.state(); radiometry(s, RC.RadiometryBlockDescent()); cost(s)[1]))
    add("linear_joint", () -> () -> (s = c.state(); radiometry(s, RC.JointRadiometry(; damping = VgtkCore.LevenbergDamping())); cost(s)[1]))
    # per view
    function view_setup()
        s = c.state(); problem, _ = RC.build_problem(s; damping = VgtkCore.LevenbergDamping(), height_degree = HEIGHT_DEGREE)
        r, vars, flags = first_block(problem, view_blocks(problem))
        return s, r, vars, flags
    end
    add("view_jacobian", () -> ((s, r, vars, flags) = view_setup(); () -> (j = RC._view_jacobian(flags, r, vars...); KA.synchronize(BE); sum(abs, j.res))))
    add("view_products", () -> ((s, r, vars, flags) = view_setup(); j = RC._view_jacobian(flags, r, vars...);
                                () -> RC._gauss_newton_blocks(j.J, j.res, j.T, j.lay, s.run)))
    add("view_normal_equations", () -> ((s, r, vars, flags) = view_setup(); () -> RC._block_normal_equations(flags, r, vars)))
    add("lm1_view", () -> () -> (s = c.state(); RC._lm_views!(s; stops = RC._calibration_stops(; max_iter = 1), damping = VgtkCore.LevenbergDamping(), height_degree = HEIGHT_DEGREE); cost(s)[1]))
    add("lm3_view", () -> () -> (s = c.state(); RC._lm_views!(s; stops = RC._calibration_stops(; max_iter = 3), damping = VgtkCore.LevenbergDamping(), height_degree = HEIGHT_DEGREE); cost(s)[1]))
    # per tile
    grid = Ref{Any}(nothing)
    grid_state() = (grid[] === nothing && (grid[] = RC.grid_model(c.lens, GRID...)); c.state(grid[]))
    add("tile_normal_equations", () -> begin
        s = grid_state(); problem, idx = RC.build_problem_grid(s; damping = VgtkCore.LevenbergDamping(), height_degree = HEIGHT_DEGREE)
        tiles = [t for t in RC.tile_costs(problem) if t.k == k]
        flags = 2^10 - 1
        () -> RC._with_registered(s, problem, idx) do
            sum(t -> RC._block_normal_equations(flags, t, NLLSsolver.getvars(t, problem.variables))[1], tiles)
        end
    end)
    add("tile_rows_products", () -> begin           # the same tiles again within one registration: the view's batch is its memoized one
        s = grid_state(); problem, idx = RC.build_problem_grid(s; damping = VgtkCore.LevenbergDamping(), height_degree = HEIGHT_DEGREE)
        tiles = [t for t in RC.tile_costs(problem) if t.k == k]
        flags = 2^10 - 1
        sweep() = sum(t -> RC._block_normal_equations(flags, t, NLLSsolver.getvars(t, problem.variables))[1], tiles)
        () -> RC._with_registered(s, problem, idx) do
            sweep()
            t0 = time_ns(); x = sweep(); Span(x + length(tiles), (time_ns() - t0) / 1e9)   # (not `v`: that rebound the pixel's v, boxed it in every stage, 4.4 kB a Φ_pixel call)
        end
    end)
    add("lm1_grid", () -> () -> (s = grid_state(); RC._lm_tiles!(s; stops = RC._calibration_stops(; max_iter = 1), damping = VgtkCore.LevenbergDamping(), height_degree = HEIGHT_DEGREE); cost(s)[1]))
    # the whole
    add("calibration", () -> () -> (s = c.state(); radiometry(s, RC.JointRadiometry(; damping = VgtkCore.LevenbergDamping())); RC._lm_views!(s; stops = RC._calibration_stops(; max_iter = 3), damping = VgtkCore.LevenbergDamping(), height_degree = HEIGHT_DEGREE); cost(s)[1]))
    return list
end

function inferred_types(c)
    k = 1; s = c.state()
    view = RC.View(c.poses[k], c.camera)
    F = VgtkCore.kernel_float(BE)
    flux = RC._FluxAt(Int32(k), Int32(H), view, c.camera.vignetting, c.mat, c.light, c.board, c.settings)
    return Dict{String, Any}(
        "View(pose, camera)" => inferred(RC.View, c.poses[k], c.camera),
        "Φ_pixel" => inferred(RC.Φ_pixel, 1, 1, k, c.camera, c.poses[k], c.mat, c.light, c.board, c.settings),
        "pixel_parts" => inferred(RC.pixel_parts, 1, 1, k, c.camera, c.poses[k], c.mat, c.light, c.board, c.settings),
        "_FluxAt(id)" => inferred(flux, UInt32(1)),
        "render_view" => inferred(RC.render_view, k, c.camera, c.poses[k], c.mat, c.light, c.board, c.settings, c.masks[k]),
        "world_ray" => inferred(RC.world_ray, c.camera, c.poses[k], SVector(1.0, 1.0), SVector(0.0, 0.0, 0.0)),
        "CalibrationState" => (; type = string(typeof(s)), concrete = isconcretetype(typeof(s))),
        "cost(state)" => inferred(cost, s),
        "all_parts(state)" => inferred(RC.all_parts, s),
        "view_of(state, k)" => inferred(RC.view_of, s, k),
        "isbits(camera)" => (; type = string(isbitstype(typeof(c.camera))), concrete = isbitstype(typeof(c.camera))),
        "isbits(View)" => (; type = string(isbitstype(typeof(view))), concrete = isbitstype(typeof(view))),
        "kernel float" => (; type = string(F), concrete = true))
end

# ---- the run -----------------------------------------------------------------------------------------------------------------------


function main()
    c = benchmark_board()
    npx = sum(count, c.masks)
    @printf("RadiometricCamera benchmark: %d × %d, %d views, %d masked pixels, %s, Julia %s%s\n", W, H, VIEWS, npx, first(VgtkCore.device_names(BE)), VERSION,
            isempty(LABEL) ? "" : ", " * LABEL)
    results = Dict{String, Any}[]
    @printf("%-30s %12s %12s %14s  %s\n", "stage", "first (s)", "best (s)", "bytes", "value")
    for (name, make) in stages(c)
        f = make()
        m = measure(f)
        @printf("%-30s %12.4f %12.4f %14d  %.12g\n", name, m.first, m.best, m.bytes, m.value); flush(stdout)
        push!(results, Dict{String, Any}("stage" => name, "first" => m.first, "best" => m.best, "bytes" => m.bytes, "value" => m.value))
        if name == PROFILE && !isempty(OUT)
            Profile.clear(); Profile.init(; n = 10^8, delay = 0.002)
            Profile.@profile f()
            open(OUT * ".$name.profile.txt", "w") do io
                Profile.print(IOContext(io, :displaysize => (200, 300)); format = :flat, sortedby = :count, mincount = 20)
            end
        end
    end
    inf = inferred_types(c)
    println("\ninferred:")
    for n in sort!(collect(keys(inf)))
        @printf("  %-22s %s %s\n", n, inf[n].concrete ? " " : "!", inf[n].type[1:min(end, 160)])
    end
    if !isempty(OUT)
        commit = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "" end
        record = Dict{String, Any}("schema" => "rc/benchmark/v1.0", "label" => LABEL, "commit" => commit, "julia" => string(VERSION),
                                   "threads" => Threads.nthreads(), "backend" => BACKEND, "device" => first(VgtkCore.device_names(BE)), "size" => [W, H], "views" => VIEWS,
                                   "masked_pixels" => npx, "pixels" => PIXELS, "grid" => collect(GRID), "warp" => WARP, "heights" => HEIGHT_DEGREE, "reps" => REPS, "stages" => results,
                                   "inferred" => Dict(n => Dict("type" => v.type, "concrete" => v.concrete) for (n, v) in inf))
        open(io -> JSON.print(io, record, 2), OUT, "w")
        println("\nwrote ", OUT)
    end
    return nothing
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()          # (benchmark/linearization.jl includes this file for its scene)
