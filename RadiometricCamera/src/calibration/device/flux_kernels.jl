
"""
    _device_rows(f, be, px; workgroupsize = nothing) -> Vector{F}

`f(id)` of every pixel id of the device list `px`, a `_row_type(f)` (a bits value of the kernel's float `F`) each, as one host vector
of `F`, pixel-major: Core's `pointwise!`, one kernel on the backend value `be` (`OnBackend`: its compiler options kept).
"""
function _device_rows(f, be, px; workgroupsize = nothing)
    out = _launch_rows(f, be, px; workgroupsize)
    KA.synchronize(be)
    return Array(out)
end
"""`_device_rows` without the wait: the kernel queued on this task's device queue, its output on the device (`Array` it after a
`KA.synchronize(be)` in the same task)."""
function _launch_rows(f, be, px; workgroupsize = nothing)
    R = _row_type(f)
    F = _row_float(R)
    out = KA.allocate(be, F, (sizeof(R) ÷ sizeof(F)) * length(px))
    VgtkCore.pointwise!(f, reinterpret(R, out), px; exec = VgtkCore.OnBackend(be; workgroupsize = _pixel_workgroup(be, workgroupsize)))
    return out
end
"""
    _pixel_workgroup(be, requested)

The workgroup size of a per-pixel kernel on the backend `be`: 16 pixels on the CPU backend, the kernel's `requested` one on a device
(its register budget; `nothing`: the backend's choice). KernelAbstractions' CPU backend gives each thread whole workgroups, and its
default group is 1024 items: a view of a run's 8,192 to 16,384 measured pixels then occupies 8 to 16 threads, a test scene's 500
one. A pixel's flux is ≈ 50 µs of work (96 rays, four radiance samples each), so 16 pixels a group cost nothing in scheduling and
every thread works: the flux kernel of a 1,728-pixel view 49 ms → 5.5 ms on 32 threads (2026-11-20), the same bits (a pixel's value
does not depend on its group).
"""
_pixel_workgroup(be, requested) = be isa KA.CPU ? 16 : requested

"""The flux `Φ_pixel` of a masked pixel id of view `k` (image height `H`)."""
struct _FluxAt{V, G, M, L, S, T}
    k::Int32
    H::Int32
    view::V
    vig::G
    mat::M
    light::L
    scene::S
    st::T
end
Adapt.@adapt_structure _FluxAt
_row_type(f::_FluxAt) = SVector{3, _float_of(f.st)}
@inline function (f::_FluxAt)(id::UInt32)
    u, v = _decode_px(id, f.H)
    return _Φ_pixel(u, v, Int(f.k), f.view, f.vig, f.mat, f.light, f.scene, f.st)
end

"""Row sums of `A .* B` per channel (helper: closure-free for the GPU compiler)."""
@inline _rowdot(A::SMatrix{3, J}, B::SMatrix{3, J}) where {J} =
    SVector(ntuple(c -> sum(ntuple(j -> A[c, j] * B[c, j], Val(J))), Val(3)))

"""The analytic parts of the linear blocks at the `i`-th pixel of view `k`'s list `px`: `(light, μ, vig)` — the SH lighting columns
(`3 × J`), the μ column (3) and the raw vignetting rows (`3 × NV`), in the kernel's float type: `_linear_rows` of the pixel's
`pixel_parts`, the closed-form solves' one path. `rays`: the radiance record of the same parameters (`_radiance_record_device`), whose
hits the parts' rays take (`_parts_hit`: the height field's root once per ray, the record's)."""
struct _PartsAt{J, NV, V, G, M, L, S, T, P, R}
    k::Int32
    H::Int32
    view::V
    vig::G
    mat::M
    light::L
    scene::S
    st::T
    sizes::Tuple{Val{J}, Val{NV}}
    px::P
    rays::R
end
Adapt.@adapt_structure _PartsAt
_row_type(f::_PartsAt{J, NV}) where {J, NV} = (F = _float_of(f.st); Tuple{SMatrix{3, J, F, 3J}, SVector{3, F}, SMatrix{3, NV, F, 3NV}})
@inline function (f::_PartsAt{J, NV})(i::UInt32) where {J, NV}
    id, lr = _pixel_and_rays(f.px, f.rays, i, f.st)
    u, v = _decode_px(id, f.H)
    p = _pixel_parts(u, v, Int(f.k), f.view, f.mat, f.light, f.scene, f.st, lr)
    return _linear_rows(p, f.vig, coefficients(f.light), f.mat.μ)
end

"""The [`PixelParts`](@ref) of a masked pixel id of view `k` (`pixel_parts`, in the float type of `st`): the closed-form solves'
parts kernel (`all_parts`)."""
struct _PixelPartsAt{V, M, L, S, T}
    k::Int32
    H::Int32
    view::V
    mat::M
    light::L
    scene::S
    st::T
end
Adapt.@adapt_structure _PixelPartsAt
_row_type(f::_PixelPartsAt) = _parts_type(f.light, f.st)
"""The [`PixelParts`](@ref) type of the lighting `light` and the render settings `st`."""
_parts_type(::EnvironmentLight{<:SphericalHarmonicMap{L, C, J}}, st::RenderSettings{TS, NS}) where {L, C, J, TS, NS} =
    PixelParts{J, 3J, NS, _float_of(st)}
_row_float(::Type{<:PixelParts{J, L3, NS, F}}) where {J, L3, NS, F} = F
@inline function (f::_PixelPartsAt)(id::UInt32)
    u, v = _decode_px(id, f.H)
    return _pixel_parts(u, v, Int(f.k), f.view, f.mat, f.light, f.scene, f.st)
end

"""The parts of every pixel of the device list `px` (`_PartsAt`; `idx` its index list, `rays` the record of the same parameters), on
the device (queued on this task's queue): `_parts_stride(J, NV)` floats per pixel."""
_parts_device(be, px, idx, k, H, view, vig, mat, light, scene, st, J::Integer, NV::Integer; rays) =
    _launch_rows(_PartsAt(Int32(k), Int32(H), view, vig, mat, light, scene, st, (Val(J), Val(NV)), px, rays), be, idx)
"""Floats per pixel of the parts."""
_parts_stride(J::Integer, NV::Integer) = 3 * (J + 1 + NV)

"""
Scatter the parts `ph` (`_parts_device`: per pixel the `3 × J` light block, the μ column and the `3 × NV` vignetting block,
each column-major) of view pixel `iv` into the Jacobian rows `row0 + c`: the SH light columns (× `light_scale`), the μ column
and the raw vignetting rows `JP` (mapped by the caller onto the gauge basis Z).
"""
@inline function _scatter_parts!(Jm, JP, row0::Integer, ph, iv::Integer, J::Integer, NV::Integer, lay, light_scale, need_v::Bool)
    base = _parts_stride(J, NV) * (iv - 1)
    lb = isempty(lay.light_cols) ? 0 : first(lay.light_cols) - 1
    @inbounds for c in 1:3
        row = row0 + c
        if lb > 0
            for j in 1:J
                Jm[row, lb + (j - 1) * 3 + c] = light_scale * Float64(ph[base + 3 * (j - 1) + c])
            end
        end
        lay.μ_col > 0 && (Jm[row, lay.μ_col] = Float64(ph[base + 3J + c]))
        if need_v
            for q in 1:NV
                JP[row, q] = Float64(ph[base + 3J + 3 * q + c])
            end
        end
    end
    return nothing
end

"""The dual flux of a masked pixel id of view `k` in one chunk of `C` partials: per channel its value, then its `C` partials
(`3(1 + C)` floats of the kernel's type). The seeds: t̂ per sensor sample at `fbase + ks`; the chief direction's two tangent seeds at `dbase + 2(ks−1) + c` (a sentinel
base gives zero seeds)."""
struct _JacAt{C, V, G, M, L, S, T, P, R}
    k::Int32
    H::Int32
    view::V
    vig::G
    mat::M
    light::L
    scene::S
    st::T
    chunk::Val{C}
    fbase::Int32
    dbase::Int32
    px::P
    rays::R
end
Adapt.@adapt_structure _JacAt
_row_type(f::_JacAt{C}) where {C} = SVector{3 * (1 + C), _float_of(f.st)}
@inline function (f::_JacAt{C})(i::UInt32) where {C}
    id, lr = _pixel_and_rays(f.px, f.rays, i, f.st)
    u, v = _decode_px(id, f.H)
    Φ = _Φ_pixel(u, v, Int(f.k), f.view, f.vig, f.mat, f.light, f.scene, f.st, Int(f.fbase), Int(f.dbase),
                nothing, lr)
    F = _float_of(f.st)
    return SVector(ntuple(q -> (c = (q - 1) ÷ (1 + C) + 1; j = (q - 1) % (1 + C);
                                F(j == 0 ? ForwardDiff.value(Φ[c]) : ForwardDiff.partials(Φ[c], j))), Val(3 * (1 + C))))
end
"""The chunk's dual flux of every pixel of the device list `px` (`_JacAt`; `idx` its index list, `_index_list`), on the device,
complete (this task's queue waited for): `3(1 + C)` floats per pixel; `rays`: the record of the same parameters
(`_radiance_record_device`), the rays' radiance. Workgroups of 128 (the kernel's register budget at C = 4)."""
function _jac_device(be, px, idx, k, H, view, vig, mat, light, scene, st, ::Val{C}, fbase, dbase; rays) where {C}
    out = _launch_rows(_JacAt(Int32(k), Int32(H), view, vig, mat, light, scene, st, Val(C), Int32(fbase), Int32(dbase),
                              px, rays), be, idx; workgroupsize = 128)
    KA.synchronize(be)
    return out
end

"""The pixel id and the radiance-Jacobian rows of the `i`-th pixel of the list `px` (`rays`: a view's record, `_radiance_record_device`)."""
@inline function _pixel_and_rays(px, rays, i::UInt32, st)
    nr = length(st.sensor) * n_disc(st.aperture)
    return (@inbounds px[i]), RadianceRays(rays, (Int(i) - 1) * nr)
end
"""The device list `1 … n` (the functors that read a record map over pixel indices), built once per pixel list: a view's in the
device cache (`_ensure_device!`, `idx`), a tile's in its own (`_tile_dev!`)."""
_index_list(be, n::Integer) = adapt(be, collect(UInt32(1):UInt32(n)))

"""The radiance record of a pixel index (`RadianceRecord`): `Φ_pixel` at the plain parameters writing every ray's `[L | ∂L/∂ℓ]`.
A thread per pixel: the pixel's footprint, sensor samples and focus are formed once for its rays (a thread per ray or per sensor
sample measured 13× and 2× slower on a 12 048-px view)."""
struct _RecordAt{V, G, M, L, S, T, P, O}
    k::Int32
    H::Int32
    view::V
    vig::G
    mat::M
    light::L
    scene::S
    st::T
    px::P
    out::O
end
Adapt.@adapt_structure _RecordAt
_row_type(f::_RecordAt) = SVector{3, _float_of(f.st)}
@inline function (f::_RecordAt)(i::UInt32)
    nr = length(f.st.sensor) * n_disc(f.st.aperture)
    u, v = _decode_px(@inbounds(f.px[i]), f.H)
    return SVector{3, _float_of(f.st)}(_Φ_pixel(u, v, Int(f.k), f.view, f.vig, f.mat, f.light, f.scene, f.st, nothing, nothing,
                                       nothing, RadianceRecord(f.out, (Int(i) - 1) * nr)))
end
"""
    _radiance_record_device(be, px, idx, k, H, v32, vig32, m32, l32, scene32, st32) -> device vector of `RadianceJac`

The radiance record of every pixel of the device list `px` (`idx` its index list, `_index_list`) at the plain parameters (the kernel's float type) (view `v32`, material, lighting, scene with
its surface): per ray `[L | ∂L/∂ℓ | 𝐱 | 𝐧]` (`RadianceRecord`), `n_rays × npx` rows, pixel-major, zeroed where no ray reached. The chunks and
the height kernel of the same parameters read it (`rays`) instead of carrying their partials through the estimator.
"""
function _radiance_record_device(be, px, idx, k, H, v32, vig32, m32, l32, scene32, st32)
    out = _launch_record(be, px, idx, k, H, v32, vig32, m32, l32, scene32, st32)
    KA.synchronize(be)
    return out
end
"""`_radiance_record_device` without the wait (this task's device queue orders the kernels that read it after it)."""
function _launch_record(be, px, idx, k, H, v32, vig32, m32, l32, scene32, st32)
    nr = length(st32.sensor) * n_disc(st32.aperture)
    F = _float_of(st32)
    out = reinterpret(RadianceJac{F}, KA.zeros(be, F, length(RadianceJac{F}) * nr * length(px)))
    f = _RecordAt(Int32(k), Int32(H), v32, vig32, m32, l32, scene32, st32, px, out)
    haskey(_CAPACITY, typeof(be)) || _note_capacity!(be, f, idx, 128)   # the LM's heaviest kernel: its occupancy sizes the window
    _launch_rows(f, be, idx; workgroupsize = 128)
    return out
end

"""The resident threads of the radiance-record kernel by backend type (`_note_capacity!`): the capacity the blocks' window divides
(`_window_width`); `nothing` on a backend without an occupancy query. Keyed by the backend, not one value: the package image's
precompile workload runs on the CPU backend (one thread), and a single slot carried its value into every session's device runs."""
const _CAPACITY = Dict{DataType, Union{Nothing, Int}}()
"""
    _note_capacity!(be, f, idx, workgroupsize)

`_CAPACITY` for the backend `be` from VisualGeometryToolkitCore's `device_resident_threads` (register-aware: the occupancy of the
compiled kernel) of the very kernel `_launch_rows` runs for the functor `f` over the index list `idx` (Core's `pointwise_kernel`, the
launch `pointwise!` makes); `nothing` where Core has no query for the backend.
"""
function _note_capacity!(be, f, idx, workgroupsize)
    R = _row_type(f); F = _row_float(R)
    out = reinterpret(R, KA.allocate(be, F, (sizeof(R) ÷ sizeof(F)) * length(idx)))
    kernel, args, ndrange = VgtkCore.pointwise_kernel(f, out, idx; exec = VgtkCore.OnBackend(be; workgroupsize))
    n = try
        VgtkCore.device_resident_threads(kernel, args...; ndrange)
    catch e
        e isa ArgumentError || rethrow()
        nothing
    end
    lock(() -> (_CAPACITY[typeof(be)] = n), _DEVICE_CACHE_LOCK)
    return nothing
end
"""The device bytes of the arrays `xs` (`nothing`s skipped): a block's job's buffers, the window's memory bound (`_window_width`)."""
_device_bytes(xs...) = sum(x -> x === nothing ? 0 : sizeof(x), xs; init = 0)

"""The nodal-height partials `∂Φ/∂h` of a masked pixel id of view `k` (`Φ_pixel`'s height mode, `HeightRays`): `3 × HEIGHT_NC`
in the kernel's float type, column-major. The device Jacobian's height columns come from this kernel, not from dual slots of the chunks (a height
reaches the flux only through a ray's parameter s: one slot through the radiance instead of `HEIGHT_NC`)."""
struct _HeightAt{V, G, M, L, S, T, H, P, R}
    k::Int32
    H::Int32
    view::V
    vig::G
    mat::M
    light::L
    scene::S
    st::T
    h::H
    px::P
    rays::R
end
Adapt.@adapt_structure _HeightAt
_row_type(f::_HeightAt) = SMatrix{3, HEIGHT_NC, _float_of(f.st), 3 * HEIGHT_NC}
@inline function (f::_HeightAt)(i::UInt32)
    id, lr = _pixel_and_rays(f.px, f.rays, i, f.st)
    u, v = _decode_px(id, f.H)
    return SMatrix{3, HEIGHT_NC, _float_of(f.st)}(_Φ_pixel(u, v, Int(f.k), f.view, f.vig, f.mat, f.light, f.scene, f.st,
                                                  nothing, nothing, HeightRays(f.h), lr))
end
"""The height partials of every pixel of the device list `px` at the plain view `v32` (the kernel's float type, device-capable),
material `m32`, vignetting and lighting, on the scene `scene32` (plain surface) at the nodal heights `h`, seeded here: `3 HEIGHT_NC`
floats per pixel, pixel-major, on the device (queued on this task's queue: `KA.synchronize` before another task reads it). Workgroups of
128 (the Jacobian kernel's register budget)."""
_heights_device(be, px, idx, k, H, v32, vig32, m32, l32, scene32, st32, h::StaticVector{HEIGHT_NC}; rays) =
    _launch_rows(_height_functor(k, H, v32, vig32, m32, l32, scene32, st32, h, px, rays), be, idx; workgroupsize = 128)
"""The height functor (`_HeightAt`): the nodal heights `h` seeded as `HEIGHT_NC`-slot duals."""
function _height_functor(k, H, v32, vig32, m32, l32, scene32, st32, h::StaticVector{HEIGHT_NC}, px, rays)
    F = _float_of(st32)
    D = ForwardDiff.Dual{JacTag, F, HEIGHT_NC}
    hd = SVector{HEIGHT_NC, D}(ntuple(i -> D(F(h[i]), ForwardDiff.Partials(ntuple(j -> i == j ? one(F) : zero(F), Val(HEIGHT_NC)))), Val(HEIGHT_NC)))
    return _HeightAt(Int32(k), Int32(H), v32, vig32, m32, l32, scene32, st32, hd, px, rays)
end
"""The free height columns of a pixel's rows (`rows[c, col]`, `_PixelRows`) from the height partials `hh` (`_heights_device`) of
view pixel `iv`: its raw nodal-height partials projected by the material block's basis `B` (`HEIGHT_NC × HEIGHT_NFREE`, the block's
`dense_basis` on the backend). Device-safe: the row kernel's (`_jacobian_row!`)."""
Base.@propagate_inbounds function _height_row!(rows, hh, iv::Integer, B, height_cols)
    base = 3 * HEIGHT_NC * (iv - 1)
    h0 = first(height_cols) - 1
    for c in 1:3
        _has_channel(rows, c) || continue
        for q in 1:size(B, 2)
            acc = 0.0
            for j in 1:HEIGHT_NC
                acc += Float64(hh[base + 3 * (j - 1) + c]) * B[j, q]
            end
            rows[c, h0 + q] = acc
        end
    end
    return nothing
end
"""The chunk starts of a device Jacobian of `P` dual slots in chunks of `C`: every slot but the nodal heights' (`off_h + 1 … off_h + NH`,
the height kernel's; the heights are not seeded in the chunks), each chunk the slots `c0 + 1 … c0 + C`."""
_chunk_starts(P::Integer, C::Integer, off_h::Integer, NH::Integer) =
    NH == 0 ? collect(0:C:(P - 1)) : vcat(collect(0:C:(off_h - 1)), collect((off_h + NH):C:(P - 1)))
