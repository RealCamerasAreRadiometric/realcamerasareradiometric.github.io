# ---------------------------------------------------------------------------
# The LM hot path on every backend (KernelAbstractions): the per-pixel flux, the radiance record, the dual Jacobian chunks, the nodal
# heights and the analytic parts of the linear blocks, as functors of a masked pixel id (`_FluxAt`, `_RecordAt`, `_JacAt`, `_HeightAt`,
# `_PartsAt`) mapped over the pixel list by VisualGeometryToolkitCore's `pointwise!` on the state's backend VALUE (`OnBackend`, so its
# `always_inline` holds; a broadcast would launch on a default-built backend): one kernel each, no kernel code here, and this file never
# references CUDA. ONE implementation (owner policy, 2026-11-11): the CPU backend runs the same functors in Float64 (Core's `kernel_float`;
# all dual slots in one chunk, Core's `dual_chunk`), a device in Float32 — the default (`default_backend`: Core's `default_device()`
# after `using CUDA`; set CUDA_VISIBLE_DEVICES before loading CUDA), the CPU by `backend = CPU()`. Build a device backend by hand with
# `always_inline = true`: without it the Jacobian kernel needs twice the local memory per thread and runs 1.5× slower at C = 4 (the
# flux 4.4×); values move by ≤ 2.2e-7. Flux and Jacobian share the one backend, so the residual stays bitwise the one
# `computeresidual` returns. The device's Float32 realisation of the common-random-number stream is deterministic but not
# bit-comparable to the CPU's Float64 one.
# ---------------------------------------------------------------------------

import KernelAbstractions as KA

"""
    HIGH_FLOAT = Float64

The float type, on every backend, of the nodal heights' path and of the Jacobian chunks that carry the pose's six slots (the chunks
starting below slot 6; `_chunk_starts`, `_pose_chunk`): the height kernel and a radiance record of its own, the view, the board, the
relief and the estimator all in it, and the pose's chunks with the heights' inputs and record (`_height_inputs`). A height column is
the radiance's derivative along the ray at the ray's board point; on a printed edge (the Siemens print's σ 0.03 mm) that derivative
moves by ≈ 1 % when the board point moves by the 0.1 µm a Float32 ray puts on it, and the estimator's α and η derivatives at the edge
by up to 0.4 % in Float32 (dgx01, 2026-11-11, the paper-like Siemens case: in Float32 the ray parameter, its height partials and the
accumulation hold to ≤ 7e-6, the radiance's ∂L/∂s does not; Float64 rays with a Float32 estimator leave 0.44 % in a tile). In Float32
the pose's columns on that case are 0.37–0.82 % off the CPU's (the Float32 board point on the print's 0.03 mm edge, as the heights'
were); the owner's gate is 0.4 %, stacked and per tile. The other chunks keep Core's `kernel_float` and their own record.
"""
const HIGH_FLOAT = Float64
"""Whether the chunk starting at `c0` carries a pose slot (`off_e = 0`: slots 1 … 6)."""
_pose_chunk(c0::Integer) = c0 < 6
"""Whether a Jacobian in the kernels' float `F` needs the heights' `HIGH_FLOAT` inputs and record (`_height_inputs`): its nodal
heights are free (`heights`), or its pose chunks run in another float than `F` (`_chunk_plan`)."""
_needs_high(F, heights::Bool) = heights || HIGH_FLOAT !== F
"""
    _chunk_plan(c0s, F, lo, hi) -> Vector{Tuple{Int, DataType, Any, Bool}}

THE float plan of a Jacobian's chunks (the per-view `_view_jacobian` and the per-tile `_batch_launch`): per chunk start
`c0` its `(c0, float, inputs, high)` — a chunk carrying a pose slot (`_pose_chunk`) in `HIGH_FLOAT` with the heights' inputs `hi`
and their record (`high`) when the kernels' float `F` is not `HIGH_FLOAT`; every other chunk in `F` with `lo` and the chunks' record.
"""
_chunk_plan(c0s, F, lo, hi) = [(_pose_chunk(c0) && HIGH_FLOAT !== F) ? (Int(c0), HIGH_FLOAT, hi, true) : (Int(c0), F, lo, false) for c0 in c0s]
"""The float type of a kernel's render settings (the kernels' rows are of it)."""
_float_of(::RenderSettings{T}) where {T} = T
"""The scalar type of a kernel row type (a static array, or a tuple of them)."""
_row_float(::Type{<:StaticArray{S, T}}) where {S, T} = T
_row_float(::Type{T}) where {T <: Tuple} = _row_float(fieldtype(T, 1))

# The device work that runs beside the caller's (a Jacobian chunk, the height kernel, the next view's batch) runs on a pool of
# long-lived worker tasks. A device queue is task-local (CUDA.jl creates a stream for each task on its first launch, 2.2 ms per
# `cuStreamCreate`), so a fresh task per chunk created a stream per chunk — 84 per 3-iteration LM of the paper-like case; a worker
# keeps its stream across calls. A worker runs one job at a time and returns to the pool when its caller collects the result, so the
# pool holds as many workers (streams) as the most jobs ever in flight at once.
struct _DeviceWorker
    inbox::Channel{Any}
end
const _DEVICE_POOL = _DeviceWorker[]
const _DEVICE_POOL_LOCK = ReentrantLock()
function _new_device_worker()
    inbox = Channel{Any}(1)
    worker = Threads.@spawn begin
        for (f, out) in inbox
            res = try                             # the latest world: a worker outlives method definitions made after its start
                (true, Base.invokelatest(f))
            catch e
                (false, e)
            end
            put!(out, res)
        end
    end
    errormonitor(worker)
    return _DeviceWorker(inbox)
end

"""
    _on_worker(f) -> job

Start `f()` on a pooled worker task (its own device queue, kept across jobs); `_await(job)` returns its value or rethrows its
exception, and gives the worker back to the pool. Every job must be awaited.
"""
function _on_worker(f)
    w = lock(() -> isempty(_DEVICE_POOL) ? _new_device_worker() : pop!(_DEVICE_POOL), _DEVICE_POOL_LOCK)
    out = Channel{Any}(1)
    put!(w.inbox, (f, out))
    return (w, out)
end
"""Close the pooled workers (their tasks end) and empty the pool. The precompile workload's evaluations start workers on the CPU
backend too, and a package image cannot hold a task (the pool would reach theirs)."""
function _drain_device_pool!()
    ws = lock(() -> (w = copy(_DEVICE_POOL); empty!(_DEVICE_POOL); w), _DEVICE_POOL_LOCK)
    foreach(w -> close(w.inbox), ws)
    return nothing
end
function _await(job)
    w, out = job
    ok, x = take!(out)
    lock(() -> push!(_DEVICE_POOL, w), _DEVICE_POOL_LOCK)
    ok || throw(x)
    return x
end

"""
    _shared_read_only!(backend, x) -> x

Mark the device arrays inside `x` as shared read-only by the concurrent chunks of a device Jacobian (`_each_chunk`): a no-op here;
RC's CUDA extension (ext/RadiometricCameraCUDAExt.jl) synchronizes the calling task's stream once and takes the arrays out of
CUDA.jl's per-buffer ownership synchronization, so the chunks' kernels on their own streams run concurrently instead of each
waiting for the previous chunk's stream. Only for arrays no kernel writes while they are shared (an upload, or a producing kernel
before the call).
"""
_shared_read_only!(be, x) = x

"""
    _each_chunk(f, be, c0s) -> Vector

`f(c0)` for every chunk start in `c0s`, results in chunk order. Every chunk runs on a worker of its own (`_on_worker`: a device
queue each), so on a device the chunks' kernels run concurrently and each worker waits for its own queue once: with 255 registers
per thread a 25k-px view is less than one wave per launch (+21% there; less on larger views). (On the CPU backend a Jacobian has
one chunk beside the heights', Core's `dual_chunk`, and KernelAbstractions' CPU kernels are threaded over pixels.)
"""
function _each_chunk(f, be, c0s)
    jobs = [_on_worker(() -> f(c0)) for c0 in c0s]
    outs = [try (true, _await(j)) catch e; (false, e) end for j in jobs]      # every worker back to the pool before a rethrow
    for (ok, x) in outs; ok || throw(x); end
    return [x for (_, x) in outs]
end
"""Count a device evaluation `k` (`:flux`, `:jac`) of the state `s` (`s.run.device_calls`: instrumentation the batched-path tests read;
a prefetch counts beside the caller's evaluation)."""
_count_device_call!(s, k::Symbol) = lock(() -> (s.run.device_calls[k] += 1; nothing), s.run.lock)

@inline function _decode_px(id::UInt32, H::Int32)
    z = Int32(id) - Int32(1)
    return Int(z ÷ H) + 1, Int(z % H) + 1                # (u, v)
end
