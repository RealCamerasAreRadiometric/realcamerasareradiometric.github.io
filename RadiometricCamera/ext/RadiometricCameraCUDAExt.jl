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
# caller preserves the host lenses, whose arrays those forms point into), and their Gauss–Newton products as two strided batched
# gemms over the zero-padded rows (the padding rows are zero in J and r, so they add nothing) — one product each for all the tiles
# instead of two per tile.
RadiometricCamera._device_elements(::CUDA.CUDABackend, xs::AbstractVector) = CUDA.CuArray(map(CUDA.cudaconvert, xs))

function RadiometricCamera._batched_products(J3::CUDA.CuArray{Float64, 3}, r2::CUDA.CuMatrix{Float64}, ms)
    n, M, nt = size(J3)
    H = similar(J3, n, n, nt); g = similar(J3, n, 1, nt)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1.0, J3, J3, 0.0, H)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1.0, J3, reshape(r2, M, 1, nt), 0.0, g)
    nH = n * n * nt
    buf = _pinned_host(nH + n * nt)
    copyto!(buf, 1, H, 1, nH); copyto!(buf, nH + 1, g, 1, n * nt)
    Hh = reshape(view(buf, 1:nH), n, n, nt); gh = reshape(view(buf, (nH + 1):(nH + n * nt)), n, nt)
    return [(view(gh, :, t), view(Hh, :, :, t)) for t in 1:nt]              # (views of the task's buffer: valid until its next call)
end
# The calling task's page-locked host buffer of at least `n` floats (grown, kept with the task: the pooled device workers'). A view's
# tiles' blocks are some 60 MB: downloaded into fresh pageable memory they took 65–210 ms on a B200 (the driver's staging and the
# new pages' faults), pinned and reused the link's rate.
function _pinned_host(n::Integer)
    ref = RadiometricCamera.VgtkCore.task_buffers(() -> Ref(Float64[]), :radiometric_camera_pinned_products)
    length(ref[]) < n && (v = Vector{Float64}(undef, n); CUDA.pin(v); ref[] = v)
    return ref[]
end

end # module
