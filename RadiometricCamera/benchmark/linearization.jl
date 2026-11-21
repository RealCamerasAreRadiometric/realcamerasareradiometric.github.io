# =============================================================================
# One view's linearization, kernel by kernel: where the time of `_view_jacobian` goes on a backend.
# =============================================================================
#
#   julia -t 8 --project=benchmark benchmark/linearization.jl [backend=cpu|gpu] [size=320x240] [pixels=0] [reps=5] [out=<file.json>]
#
# The scene and the state are benchmark/run.jl's (included: its options, its scene). Each kernel of the per-view Jacobian
# is launched alone on the state's backend and awaited, with the inputs `_view_jacobian` gives it (this file follows that function:
# when it changes, this follows): the flux, the radiance record in the kernels' float (or its rounding from the `HIGH_FLOAT` record) and in `HIGH_FLOAT`, one dual chunk in each
# float (`chunk` partials: the state's `jacobian_chunk` on a device, every slot on the CPU), the linear blocks' parts, the gather
# of the slots, the Gauss–Newton products, and the whole (`_view_jacobian`, whose chunks overlap on a device). Milliseconds, the
# minimum of `reps` runs after a first; `chunks` is the number of chunk launches of the whole in each float.
# =============================================================================
include(joinpath(@__DIR__, "run.jl"))
using RadiometricCamera: VgtkCore

function linearization()
    c = benchmark_board(); s = c.state(); problem, _ = RC.build_problem(s; height_degree = HEIGHT_DEGREE)
    r, vars, flags = first_block(problem, view_blocks(problem))
    pose, dist, foc, ap, m, vg, lt = vars
    dc = RC._ensure_device!(s); be = s.backend; F = RC._kernel_float(be); G = RC.HIGH_FLOAT
    k = r.k; px = dc.px[k]; idx = dc.idx[k]; npx = length(s.px[k]); Hh = RC._image_height(s)
    NV = Int(RC.n_control_points(VgtkCore.model(vg))); J = Int(RC.sh_count(RC.max_degree(lt.light)))
    NS = length(s.settings.sensor); N = length(VgtkCore.model(foc).t̂_i)
    lay = RC._jacobian_layout(flags, N, NS, VgtkCore.tangent_size(vg), NV, J, VgtkCore.tangent_size(dist); NH = RC.n_height(m), NI = RC.n_inplane(m), fixed = RC.material_fixed(m))
    view0 = RC._view_of(r, pose, dist, foc, ap)
    stk = VgtkCore.convert_eltype(F, s.settings); C = RC._jac_chunk(s, lay.P)
    vigk = VgtkCore.convert_eltype(F, VgtkCore.model(vg)); lk = VgtkCore.convert_eltype(F, lt.light)
    vk = RC.adapt(be, VgtkCore.convert_eltype(F, view0)); mk = VgtkCore.convert_eltype(F, m.mat)
    boardk = RC.with_state(dc.board, VgtkCore.convert_eltype(F, m.surface))
    record(args...) = RC._radiance_record_device(be, px, idx, k, Hh, args...)
    high = G !== F
    hk = high ? RC._height_inputs(s, view0, VgtkCore.model(vg), m, lt.light) : nothing
    hrays = high ? record(hk.view, hk.vig, hk.mat, hk.light, hk.board, hk.st) : record(vk, vigk, mk, lk, boardk, stk)
    rounded = high && isdefined(RC, :_record_as)            # the kernels'-float record: the HIGH_FLOAT record rounded (before: a record of its own)
    rays = !high ? hrays : rounded ? RC._record_as(F, hrays) : record(vk, vigk, mk, lk, boardk, stk)
    c0s = applicable(RC._chunk_starts, lay.P, C) ? RC._chunk_starts(lay.P, C) : RC._chunk_starts(lay.P, C, lay.off_h, RC.n_height(m))
    plan = RC._chunk_plan(c0s, F, (vigk, lk, dc.board, stk, rays), hk === nothing ? nothing : (hk.vig, hk.light, hk.board, hk.st, hrays))
    low = findfirst(p -> !p[4], plan); hi = findfirst(p -> p[4], plan)
    chunk(p) = RC._view_chunk(Val(p[2]), s, r, lay, p[1], C, pose, dist, foc, ap, m, p[3]...)
    function best(f)
        f(); KA.synchronize(be)
        return 1e3 * minimum((t = time_ns(); f(); KA.synchronize(be); (time_ns() - t) / 1e9) for _ in 1:REPS)
    end
    out = Dict{String, Any}("backend" => BACKEND, "device" => device_name(), "pixels" => npx, "slots" => lay.P, "columns" => lay.ncol, "chunk" => C,
                            "warp" => WARP, "heights" => HEIGHT_DEGREE, "chunks_low" => count(p -> !p[4], plan), "chunks_high" => count(p -> p[4], plan), "float" => string(F), "label" => LABEL)
    out["flux_ms"] = best(() -> RC._device_rows(RC._flux_functor(s, k, vk, VgtkCore.model(vg), m.mat, lt.light, m.surface), be, px))
    rounded ? (out["record_round_ms"] = best(() -> RC._record_as(F, hrays))) : (out["record_ms"] = best(() -> record(vk, vigk, mk, lk, boardk, stk)))
    high && (out["record_high_ms"] = best(() -> record(hk.view, hk.vig, hk.mat, hk.light, hk.board, hk.st)))
    low === nothing || (out["chunk_ms"] = best(() -> chunk(plan[low])))
    hi === nothing || (out["chunk_high_ms"] = best(() -> chunk(plan[hi])))
    parts() = applicable(RC._parts_device, be, px, k, Hh, vk, vigk, mk, lk, boardk, stk, J, NV) ?
        RC._parts_device(be, px, k, Hh, vk, vigk, mk, lk, boardk, stk, J, NV) :              # (before the parts took the record's hits)
        RC._parts_device(be, px, idx, k, Hh, vk, vigk, mk, lk, boardk, stk, J, NV; rays)
    out["parts_ms"] = best(parts)
    if !isempty(lay.height_cols)                           # the nodal heights' kernel (free heights)
        hi_in = hk === nothing ? (view = vk, vig = vigk, mat = mk, light = lk, board = boardk, st = stk) : hk
        out["heights_ms"] = best(() -> RC._heights_device(be, px, idx, k, Hh, hi_in.view, hi_in.vig, hi_in.mat, hi_in.light, hi_in.board, hi_in.st, m.surface.h; rays = hrays))
    end
    hs = chunk(plan[something(low, hi)])
    out["slots_ms"] = best(() -> RC._slot_rows(be, [(0, hs, C)], lay.P, npx))
    v = RC._view_jacobian(flags, r, vars...)
    out["products_ms"] = best(() -> RC._gauss_newton_blocks(v.J, v.res, v.T, v.lay, s.run))
    out["view_jacobian_ms"] = best(() -> RC._view_jacobian(flags, r, vars...))
    out["view_normal_equations_ms"] = best(() -> RC._block_normal_equations(flags, r, vars))
    out["residual_ms"] = best(() -> RC._block_residual(r, vars...))
    for n in sort!(collect(keys(out)))
        println(rpad(n, 28), out[n])
    end
    isempty(OUT) || open(io -> JSON.print(io, out, 2), OUT, "w")
    return out
end

linearization()
