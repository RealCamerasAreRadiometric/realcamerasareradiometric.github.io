# =============================================================================
# RadiometricCameraCUDAExt — RC's device arrays that the concurrent Jacobian chunks share, out of CUDA.jl's ownership sync
# =============================================================================
#
# CUDA.jl tracks, per device buffer, the stream that last touched it, and a kernel on another stream first synchronizes that
# stream (`maybe_synchronize` on `convert(CuPtr, ·)`). The device Jacobian runs its chunks on pooled workers, one stream each
# (`_each_chunk`), and every chunk READS the same arrays — the device scene (texture, edge map, the warping's system), the pixel
# lists, the view's adapted lens and the radiance record — so each launch waited for the previous chunk's stream: the chunks ran
# one after the other. RC takes exactly those arrays out of the tracking (`enable_synchronization!(a, false)`, CUDA.jl's escape
# hatch), and only them: `_shared_read_only!` is called on the arrays a batch shares, after ONE synchronization of the calling
# task's stream, which completes their uploads and the record's kernel. Safe because
#   - no kernel writes them while they are shared: they are written only before that synchronization (an upload, or the radiance
#     record's kernel), and every chunk's own output is a fresh array that keeps its tracking;
#   - every reader finishes before they can be freed: `_each_chunk` returns only after each worker has waited for its own stream,
#     and the long-lived ones (the state's device cache) are never written after their upload.
# Nothing in the stack changes (owner decision, Option A).
# =============================================================================
module RadiometricCameraCUDAExt

import RadiometricCamera
import CUDA
using Adapt: Adapt

struct _SharedReadOnly end
Adapt.adapt_storage(::_SharedReadOnly, a::CUDA.CuArray) = CUDA.enable_synchronization!(a, false)

function RadiometricCamera._shared_read_only!(::CUDA.CUDABackend, x)
    CUDA.synchronize()                         # the uploads and the producing kernels are complete on this task's stream
    Adapt.adapt(_SharedReadOnly(), x)          # (the traversal's rebuilt structure is discarded: only the flags change)
    return x
end

# The tiles of a view in one launch (`_tiles_normal_equations`): their window lenses as one device array of their device forms (the
# caller preserves the host lenses, whose arrays those forms point into), and their Gauss–Newton blocks in the problem's coordinates:
# TᵀJ as one product, then two strided batched gemms over the zero-padded rows (the padding rows are zero in J and r, so they add
# nothing) — instead of two products and a host transform per tile.
RadiometricCamera._device_elements(::CUDA.CUDABackend, xs::AbstractVector) = CUDA.CuArray(map(CUDA.cudaconvert, xs))

function RadiometricCamera._batched_blocks(J3::CUDA.CuArray{Float64, 3}, r2::CUDA.CuMatrix{Float64}, ms, T::Matrix{Float64}, lay, run)
    ncol, M, nt = size(J3); n = size(T, 2)
    Jp = reshape(CUDA.CuArray(T)' * reshape(J3, ncol, M * nt), n, M, nt)   # TᵀJ: the rows in the problem's coordinates, one product
    nH = n * n * nt; nB = nH + n * nt
    B = similar(J3, nB)                                      # H and g in one buffer: one download
    H = reshape(view(B, 1:nH), n, n, nt); g = reshape(view(B, (nH + 1):nB), n, 1, nt)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1.0, Jp, Jp, 0.0, H)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1.0, Jp, reshape(r2, M, 1, nt), 0.0, g)
    buf = _take_pinned(nB)
    copyto!(buf, 1, B, 1, nB)
    Hh = reshape(view(buf, 1:nH), n, n, nt); gh = reshape(view(buf, (nH + 1):nB), n, nt)
    blocks = map(1:nt) do t                                  # a non-finite entry: that tile scrubbed and formed alone
        gt = view(gh, :, t); Ht = view(Hh, :, :, t)
        RadiometricCamera._finite_blocks(gt, Ht) ? (gt, Ht) :
            RadiometricCamera._gauss_newton_blocks(view(J3, :, 1:ms[t], t), view(r2, 1:ms[t], t), T, lay, run)
    end
    return blocks, buf                                       # (views of the buffer: valid until it is given back)
end
# A pool of page-locked host buffers for the batches' downloads (a view's tiles' blocks are some 60 MB: into fresh pageable memory
# they took 65–210 ms on a B200, the driver's staging and the new pages' faults; pinned and reused, the link's rate). A batch takes one
# (`_take_pinned`) and gives it back once its blocks are copied out (`_release_host_buffer!`); as many exist as batches ran at once, and
# the pool is dropped when the grid problem is unregistered (`_release_host_buffers!`: unpinned as they are collected), so no
# memory stays locked after a solve.
const _PINNED = Vector{Vector{Float64}}()
const _PINNED_LOCK = ReentrantLock()
function _take_pinned(n::Integer)
    buf = lock(_PINNED_LOCK) do
        i = findfirst(b -> length(b) >= n, _PINNED)
        i === nothing ? nothing : popat!(_PINNED, i)
    end
    buf === nothing || return buf
    v = Vector{Float64}(undef, n + n ÷ 4)                    # (a quarter more: a view of more tiles reuses it)
    CUDA.pin(v)
    return v
end
RadiometricCamera._release_host_buffer!(::CUDA.CUDABackend, buf::Vector{Float64}) = (lock(() -> push!(_PINNED, buf), _PINNED_LOCK); nothing)
RadiometricCamera._release_host_buffers!(::CUDA.CUDABackend) = (lock(() -> empty!(_PINNED), _PINNED_LOCK); nothing)

end # module
