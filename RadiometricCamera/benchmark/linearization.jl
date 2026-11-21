# =============================================================================
# One view's linearization, kernel by kernel: where the time of `_view_jacobian` goes on a backend.
# =============================================================================
#
#   julia -t 8 --project=benchmark benchmark/linearization.jl [backend=cpu|gpu] [size=320x240] [pixels=0] [reps=5] [out=<file.json>]
#
# The scene and the state are benchmark/run.jl's (included: its options, its scene). Each kernel of the per-view Jacobian
# is launched alone on the state's backend and awaited, with the inputs `_view_jacobian` gives it (this file follows that function:
# when it changes, this follows): the flux, the radiance record (its geometry in `ANCHOR_FLOAT`, its estimator in the kernels' float),
# one dual chunk (`chunk` partials: the state's `jacobian_chunk` on a device, every slot on the CPU), the linear blocks' parts, the
# gather of the slots, the Gauss–Newton products, and the whole (`_view_jacobian`, whose chunks overlap on a device). Milliseconds,
# the minimum of `reps` runs after a first; `chunks` is the number of chunk launches of the whole.
# =============================================================================
include(joinpath(@__DIR__, "run.jl"))
using RadiometricCamera: VgtkCore

function linearization()
    c = benchmark_board(); s = c.state(); problem, _ = RC.build_problem(s; height_degree = HEIGHT_DEGREE)
    r, vars, flags = first_block(problem, view_blocks(problem))
    pose, dist, foc, ap, m, vg, lt = vars
    dc = RC._ensure_device!(s); be = s.backend; F = RC._kernel_float(be)
    k = r.k; px = dc.px[k]; idx = dc.idx[k]; npx = length(s.px[k]); Hh = RC._image_height(s)
    NV = Int(RC.n_control_points(VgtkCore.model(vg))); J = Int(RC.sh_count(RC.max_degree(lt.light)))
    NS = length(s.settings.sensor); N = length(VgtkCore.model(foc).t̂_i)
    lay = RC._jacobian_layout(flags, N, NS, VgtkCore.tangent_size(vg), NV, J, VgtkCore.tangent_size(dist); NH = RC.n_height(m), NI = RC.n_inplane(m), fixed = RC.material_fixed(m))
    view0 = RC._view_of(r, pose, dist, foc, ap)
    stk = VgtkCore.convert_eltype(F, s.settings); C = RC._jac_chunk(s, lay.P)
    vigk = VgtkCore.convert_eltype(F, VgtkCore.model(vg)); lk = VgtkCore.convert_eltype(F, lt.light)
    vk = RC.adapt(be, VgtkCore.convert_eltype(F, view0)); mk = VgtkCore.convert_eltype(F, m.mat)
    boardk = RC.with_state(dc.board, VgtkCore.convert_eltype(F, m.surface))
    rec = RC._record_inputs(s, view0, VgtkCore.model(vg), m, lt.light, (vk, vigk, mk, lk, boardk, stk))
    record() = RC._radiance_record_device(be, px, idx, k, Hh, rec.inputs...; rec.kw...)
    rays = record()
    c0s = RC._chunk_starts(lay.P, C)
    chunk(c0) = RC._view_chunk(Val(F), s, r, lay, c0, C, pose, dist, foc, ap, m, vigk, lk, dc.board, stk, rays)
    function best(f)
        f(); KA.synchronize(be)
        return 1e3 * minimum((t = time_ns(); f(); KA.synchronize(be); (time_ns() - t) / 1e9) for _ in 1:REPS)
    end
    out = Dict{String, Any}("backend" => BACKEND, "device" => device_name(), "pixels" => npx, "slots" => lay.P, "columns" => lay.ncol, "chunk" => C,
                            "warp" => WARP, "heights" => HEIGHT_DEGREE, "chunks" => length(c0s), "float" => string(F), "label" => LABEL)
    out["flux_ms"] = best(() -> RC._device_rows(RC._flux_functor(s, k, vk, VgtkCore.model(vg), m.mat, lt.light, m.surface), be, px))
    out["record_ms"] = best(record)
    out["chunk_ms"] = best(() -> chunk(first(c0s)))              # the pose's (slots 1 … C)
    out["parts_ms"] = best(() -> RC._parts_device(be, px, idx, k, Hh, vk, vigk, mk, lk, boardk, stk, J, NV; rays))
    if !isempty(lay.height_cols)                           # the nodal heights' kernel (free heights)
        out["heights_ms"] = best(() -> RC._heights_device(be, px, idx, k, Hh, vk, vigk, mk, lk, boardk, stk, m.surface.h; rays))
    end
    hs = chunk(first(c0s))
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
