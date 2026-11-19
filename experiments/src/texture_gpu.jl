# =============================================================================
# src/texture_gpu.jl — the texture test's device (scripts/texture_recovery.jl, stage refine): the page solve runs on one GPU, Core's
# `PenalizedOperator` and `cgls` on the CUDA backend (Core's CSR kernel: one work item per row, deterministic, no atomics). Included
# after src/texture_recovery.jl.
# =============================================================================
module TextureGPU

import CUDA

export gpu_name

"The device's name. The refine stage's backend is RadiometricCamera's `rc_backend(\"gpu\"; inline = false, fallback = false)` (there is
no CPU fallback: refused without a functional GPU)."
gpu_name() = CUDA.name(CUDA.device())

end # module
