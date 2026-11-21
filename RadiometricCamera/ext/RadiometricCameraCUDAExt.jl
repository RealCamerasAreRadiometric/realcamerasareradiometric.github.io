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

end # module
