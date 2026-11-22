using TOML: TOML
using VisualGeometryToolkitCore: TagRegistry, tag_of, check_keys, read_table, toml_key, toml_table, allowed_keys, convert_field
import VisualGeometryToolkitCore: toml_field

"""The device a driver runs the refinement on: [`CPUCompute`](@ref) or [`GPUCompute`](@ref) (tags in [`BACKEND_TAGS`](@ref)). A
driver maps it to a KernelAbstractions backend with [`rc_backend`](@ref), which loads the GPU vendor packages for `GPUCompute`."""
abstract type AbstractComputeBackend end
"""The CPU (KernelAbstractions' `CPU()`). Tag `cpu`."""
struct CPUCompute <: AbstractComputeBackend end
"""A GPU, when one is functional (any KernelAbstractions GPU backend: VisualGeometryToolkitCore's `default_device`). Tag `gpu`."""
struct GPUCompute <: AbstractComputeBackend end

"""The backend registry: `compute.backend` tag → [`AbstractComputeBackend`](@ref) type. The tag `cuda` is retired (2026-11-13: a GPU
is any KernelAbstractions GPU backend, not CUDA's): it refuses, naming `gpu`."""
const BACKEND_TAGS = TagRegistry(:backend, "Compute", "cpu" => CPUCompute, "gpu" => GPUCompute; retired = ["cuda" => "gpu"],
                                 paths = [r"compute\.backend"])

"""The GPU vendor packages `rc_backend` loads for [`GPUCompute`](@ref), each registering its device with VisualGeometryToolkitCore's
resolver (`default_device`) at load. THE one vendor-specific statement of RC: another vendor's package is added beside CUDA here."""
const GPU_VENDOR_PACKAGES = (:CUDA,)

"""
    rc_backend(backend; inline = true, fallback = true) -> KernelAbstractions.Backend

THE device resolution of RC and its drivers: `backend` a tag of [`BACKEND_TAGS`](@ref) or its type. [`CPUCompute`](@ref) is
KernelAbstractions' `CPU()`; [`GPUCompute`](@ref) loads the GPU vendor packages (`GPU_VENDOR_PACKAGES`) and takes
VisualGeometryToolkitCore's vendor-agnostic resolver afresh (`reset_default_device!`, then `default_device`: a functional GPU, else
the CPU). With no functional GPU it falls back to the CPU with a loud warning, or refuses (`fallback = false`). `inline`: the device
backend compiling every call inlined (`always_inline`), RC's kernels' fast build — Core's resolver's non-inlined Jacobian kernel
needs twice the local memory (1.5x slower at jacobian_chunk 4, 4.4x on the flux). Called at top level by a driver, so its later calls
run in the world the vendor package's load made.
"""
function rc_backend(backend; inline::Bool = true, fallback::Bool = true)
    B = backend isa AbstractString ? BACKEND_TAGS[backend] : backend
    B === CPUCompute && return KernelAbstractions.CPU()
    B === GPUCompute || throw(ArgumentError("rc_backend: $(repr(backend)) is not a compute backend ($(join(keys(BACKEND_TAGS), ", ")))"))
    for pkg in GPU_VENDOR_PACKAGES
        @eval import $pkg
    end
    VgtkCore.reset_default_device!()          # a resolution memoized before the vendor's load (its resolver registers at load) is forgotten
    d = Base.invokelatest(VgtkCore.default_device)
    if d isa KernelAbstractions.CPU
        fallback || error("compute backend \"gpu\": no functional GPU (set it to \"cpu\" to run on the CPU)")
        @warn "compute backend \"gpu\": NO FUNCTIONAL GPU — running on the CPU, which is very slow"
        return d
    end
    return inline ? Base.invokelatest(() -> typeof(d)(; always_inline = true)) : d
end

"""
    default_backend() -> KernelAbstractions.Backend

The device of RC's entry points when the caller supplies none (the `backend` of [`CalibrationState`](@ref), [`render_view`](@ref)
and [`compute_mask`](@ref)): VisualGeometryToolkitCore's `default_device()`, a functional GPU when its vendor package is loaded
(`using CUDA`), else the CPU, with a warning, once per session, that the run will be very slow. A caller that supplies
`backend = CPU()` has chosen the CPU and is not warned. It does not load the vendor package, as a driver's [`rc_backend`](@ref) does:
a keyword default runs inside the caller's function, which cannot call the methods a package load defines then (world age), so a
script loads it itself, `using CUDA`, before its first call.
"""
function default_backend()
    d = VgtkCore.default_device()
    d isa KernelAbstractions.CPU &&
        @warn "RadiometricCamera: no functional GPU — running on the CPU, which is very slow. Load the GPU's package (`using CUDA`) " *
              "before the first call for the GPU, or supply `backend = CPU()` to choose the CPU." maxlog = 1
    return d
end
