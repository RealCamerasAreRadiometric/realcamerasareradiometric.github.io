
# ---------------------------------------------------------------------------
# The Jacobian and its Gauss–Newton blocks on the state's backend (owner, 2026-11-13: GPU first — no host Jacobian, no host scatter
# and no host BLAS in the LM iteration). The kernels' outputs stay on the device: the dual chunks' partials are gathered into one slot
# array (`_slot_rows`), ONE row kernel (`_jacobian_row!`, both LM routes: a view's rows and a tile's) writes the measured channels'
# rows of the Jacobian in the layout's full columns, and the blocks' coordinates — the channel ties, the charts' column scales, the
# held focus nodes — are one column map `T` (`_column_map`) applied to the n × n products (`_gauss_newton_blocks`): only `JᵀJ` and
# `Jᵀr` reach the host.
# ---------------------------------------------------------------------------

"""
    _slot_rows(be, chunks, P, npx) -> S

The dual chunks' partials (`chunks`: `(c0, out, C)` per chunk, `out` its `_JacAt` rows on the device, `3(1 + C)` floats per pixel)
gathered into one Float64 array `S[slot, c, i]` on the backend (`P` slots, the three channels, `npx` pixels): the layout the row kernel reads. Queued on this task's queue.
"""
function _slot_rows(be, chunks, P::Integer, npx::Integer)
    S = KA.zeros(be, Float64, P, 3, npx)
    for (c0, o, C) in chunks
        n = min(C, P - c0)
        n > 0 || continue
        h = reshape(o, 1 + C, 3, npx)
        @views S[(c0 + 1):(c0 + n), :, :] .= Float64.(h[2:(1 + n), :, :])
    end
    return S
end

"""
    _whiten_rows!(J, r, ω, base, Val(NC))

A pixel's `NC` rows `base + 1 … base + NC` under their weights (`_whiten`, weighting.jl): the residual `r[e]` and its Jacobian row
`J[:, e]` (columns × rows) times `ω[e]` — the same product the host residual takes (`_block_residual`), so the two stay bitwise one.
`Unweighted()`: nothing (the rows as measured), statically.
"""
@inline _whiten_rows!(J, r, ::Unweighted, base, ::Val) = nothing
Base.@propagate_inbounds function _whiten_rows!(J, r, ω::AbstractVector, base, ::Val{NC}) where {NC}
    for t in 1:NC
        e = base + t
        r[e] = _whiten(ω, e, r[e])
        for col in 1:size(J, 1)
            J[col, e] = _whiten(ω, e, J[col, e])
        end
    end
    return nothing
end

"""
    _PixelRows(J, base, chmap)

One pixel's Jacobian rows as the per-pixel helpers address them (`rows[c, col]`, channel `c = 1:3`: `_focus_chain!`, `_patch_fanout!`,
`_scatter_parts!`, `_height_row!`), on the device Jacobian `J` (columns × rows): channel `c`'s row is `base + chmap[c]`,
and a channel the state does not measure (`chmap[c] = 0`) has no row — a write to it is dropped, a read is zero.
"""
struct _PixelRows{A}
    J::A
    base::Int
    chmap::NTuple{3, Int}
end
@inline _has_channel(p::_PixelRows, c::Integer) = p.chmap[c] != 0
Base.@propagate_inbounds Base.getindex(p::_PixelRows, c::Integer, col::Integer) =
    (m = p.chmap[c]; m == 0 ? zero(eltype(p.J)) : p.J[col, p.base + m])
Base.@propagate_inbounds function Base.setindex!(p::_PixelRows, x, c::Integer, col::Integer)
    m = p.chmap[c]
    m == 0 || (p.J[col, p.base + m] = x)
    return p
end
"""The raw vignetting rows of a pixel (`JP[c, q]`, `_scatter_parts!`) written straight into its vignetting columns: each through the
free directions `Z` (`NV × NF`, the block's basis on the backend), the columns from `col0 + 1`."""
struct _ProjectedRows{R, Z}
    rows::R
    Z::Z
    col0::Int
end
Base.@propagate_inbounds function Base.setindex!(p::_ProjectedRows, x, c::Integer, q::Integer)
    _has_channel(p.rows, c) || return p
    for j in 1:size(p.Z, 2)
        p.rows[c, p.col0 + j] += x * p.Z[q, j]
    end
    return p
end

"""The raw vignetting rows' sink of a held vignetting (`_jacobian_row!`): it keeps nothing, so the row kernel compiles no call on an
absent basis."""
struct _NoRows end
Base.@propagate_inbounds Base.setindex!(p::_NoRows, x, c::Integer, q::Integer) = p

"""The channel map of the measured channels `chs`: per channel `1:3` its position among them, 0 for an unmeasured channel."""
@inline _channel_map(chs::NTuple{NC, Int}) where {NC} =
    ntuple(c -> (m = 0; for t in 1:NC; chs[t] == c && (m = t); end; m), Val(3))

"""
    _kernel_layout(be, lay) -> NamedTuple

The device form of a Jacobian layout (`_jacobian_layout` of a view, `_tile_layout` of a tile): the AD scatter as a backend vector of
`(col, slot, ch)`, the column blocks, the slot offsets the row kernel reads and the tile's patch bookkeeping (absent in a view's
layout: no patch or finite-difference column), all isbits or adapted.
"""
function _kernel_layout(be, lay)
    g(f, d) = hasproperty(lay, f) ? getproperty(lay, f) : d
    free = g(:free, nothing)
    return (; scatter = adapt(be, [(Int32(col), Int32(slot), Int32(ch)) for (col, slot, ch) in lay.scatter]),
            focus_cols = lay.focus_cols, μ_col = lay.μ_col, vig_cols = lay.vig_cols, light_cols = lay.light_cols, height_cols = lay.height_cols,
            patch_cols = Tuple(g(:patch_cols, (0, 0, 0, 0))),
            off_f = lay.off_f, off_dir = g(:off_dir, 0), off_r = lay.off_r,
            free = free === nothing ? nothing : adapt(be, Matrix{Bool}(free)),
            held_rotation_coordinates = g(:held_rotation_coordinates, _NO_ROTATION_GAUGE))
end

"""
    _jacobian_row!(J, i, r, ids, bufidx, gidx, H, S, flux, ϕ, ω, chs, klay, parts, pdims, heights, hB, G, ::Val{N}, sensor, lensw,
                   bases, P0, pxy, light_scale, Z)

THE row kernel of the LM's Jacobian, both routes (`_launch_pixel_rows!` over a block's pixels): pixel `i` of the block (buffer pixel
`bufidx[i]` of the kernels' outputs, its id in `ids`) writes the residual `r` and the Jacobian rows (`J`, columns × rows) of its
measured channels `chs`, in the layout's full columns (`klay`, `_kernel_layout`): the AD scatter from the slots `S` (`_slot_rows`), the
focus chain through the nodal weights `G[:, ks, gidx[i]]` (`_focus_chain!`), a tile's patch fan-out on its window lens `lensw` with
the chief directions' tangent `bases` (`_patch_fanout!`; `pxy === nothing`: a view, no patches), the parts' lighting, μ and vignetting columns (the raw rows through the free directions `Z`, `_scatter_parts!`), and the
nodal heights through the basis `hB` (`_height_row!`). Every column the arithmetic of
the host assembly it replaces; the blocks' coordinates are `_column_map`'s. Last, the pixel's rows — the residual and every Jacobian
column — under their weights `ω` (`_whiten_rows!`: whitened, or as measured).
"""
Base.@propagate_inbounds function _jacobian_row!(J, i, r, ids, bufidx, gidx, H, S, flux, ϕ, ω, chs::NTuple{NC, Int}, klay, parts, pdims,
                                                 heights, hB, G, ::Val{N}, sensor::SensorQuadrature{TS, NS}, lensw, bases, P0, pxy,
                                                 light_scale, Z) where {NC, N, TS, NS}
    iv = Int(bufidx[i])
    u, v = _decode_px(ids[iv], H)
    base = NC * (i - 1)
    rows = _PixelRows(J, base, _channel_map(chs))
    for t in 1:NC
        r[base + t] = Float64(flux[3 * (iv - 1) + chs[t]]) - ϕ[base + t]
    end
    sc = klay.scatter
    for c in 1:3
        _has_channel(rows, c) || continue
        for e in 1:length(sc)
            col, slot, ch = sc[e]
            (ch == 0 || ch == c) && (rows[c, col] = S[slot, c, iv])
        end
    end
    if G !== nothing && !isempty(klay.focus_cols)        # (`nothing`: a held focus — a static branch, so the device compiles no call on it)
        gi = Int(gidx[i])
        g = ntuple(ks -> SVector{N, Float64}(ntuple(n -> G[n, ks, gi], Val(N))), Val(NS))
        dt̂ = SMatrix{3, NS, Float64}(ntuple(q -> S[klay.off_f + (q - 1) ÷ 3 + 1, (q - 1) % 3 + 1, iv], Val(3 * NS)))
        _focus_chain!(rows, 0, dt̂, g, klay.focus_cols)
    end
    𝐩of = ks -> 𝐩_sample(sensor, ks, u, v)
    if pxy !== nothing                                     # a tile's patches: the chief directions' gradients fanned out on its window
        dg = ntuple(ks -> (x = _chief_ray(lensw, 𝐩of(ks));
                           ntuple(c -> _dir_gradient(S[klay.off_dir + 2ks - 1, c, iv], S[klay.off_dir + 2ks, c, iv], x), Val(3))), Val(NS))
        _patch_fanout!(rows, 0, lensw, bases, klay, P0, pxy[1], pxy[2], NS, 𝐩of, (ks, comp, c) -> dg[ks][c][comp])
    end
    if parts !== nothing
        JL, NVv = pdims
        jp = Z === nothing ? _NoRows() : _ProjectedRows(rows, Z, first(klay.vig_cols) - 1)   # (a held vignetting: no rows, statically)
        _scatter_parts!(rows, jp, 0, parts, iv, JL, NVv, klay, light_scale, Z !== nothing)
    end
    heights === nothing || _height_row!(rows, heights, iv, hB, klay.height_cols)
    _whiten_rows!(J, r, ω, base, Val(NC))
    return nothing
end

"""
    _launch_pixel_rows!(row!, out, n, args...) -> out

`VgtkCore._launch_rows!` for a row function whose row is a pixel's work (the Jacobian's row kernel, `_jacobian_row!`): Core's row
kernel on the per-pixel workgroup (`_pixel_workgroup`: on the CPU backend every thread works on a small block, where Core's launch
gives a 1,728-pixel view's rows to two), followed by a synchronize. Core's launch takes no workgroup: an upstream candidate.
"""
function _launch_pixel_rows!(row!::F, out, n::Integer, args::Vararg{Any, N}) where {F, N}
    n > 0 || return out
    be = KA.get_backend(out)
    VgtkCore.OnBackend(be; workgroupsize = _pixel_workgroup(be, nothing))(VgtkCore._row_kernel!)((row!, out, args...); ndrange = n)
    KA.synchronize(be)
    return out
end

"""
    _focus_weights(s, k, tps, lens, vix, vix_d) -> (G, gidx)

The focus chain's nodal weights of view `k`'s pixels on the backend (`G[n, ks, ·]`: node `n`'s weight at sensor sample `ks`,
`_nodal_weights_at`) and the block's index into them (`gidx[i]` for the block's pixel `i`, at view pixel `vix[i]`; `vix_d` the same
list on the backend, the block's cached copy). The sensor chart's
weights depend on the pixel alone: computed once per view and kept in the device cache (`:focus_weights`, keyed by the focus's nodes);
a direction chart's (EXPERIMENT focus-ab) follow the lens, so they are computed for the block's pixels at each call (a host step: see
the audit in the module notes).
"""
function _focus_weights(s::CalibrationState, k::Integer, tps::FocusSpline{N}, lens, vix::AbstractVector{<:Integer}, vix_d) where {N}
    be = s.backend
    NS = length(s.settings.sensor)
    function weights(pix)
        G = Array{Float64}(undef, N, NS, length(pix))
        for (i, iv) in enumerate(pix)
            u, v = s.px[k][iv]
            g = _nodal_weights_at(tps, lens, s.settings.sensor, u, v, Val(NS))
            for ks in 1:NS, n in 1:N
                G[n, ks, i] = g[ks][n]
            end
        end
        return _shared_read_only!(be, adapt(be, G))
    end
    tps.kind == FOCUS_SENSOR || return weights(vix), adapt(be, collect(Int32(1):Int32(length(vix))))
    G = lock(_DEVICE_CACHE_LOCK) do                          # (the views run concurrently: `_windowed`)
        dc = _ensure_device!(s)
        fw = get(dc, :focus_weights, nothing)
        if fw === nothing || fw.nodes != tps.nodes
            fw = (; nodes = tps.nodes, G = Vector{Any}(nothing, n_views(s)))
            s.device_cache = merge(dc, (; focus_weights = fw))
        end
        fw.G[k] === nothing && (fw.G[k] = weights(1:length(s.px[k])))
        fw.G[k]
    end
    return G, vix_d
end

"""`_scrub_entries!` of the device Jacobian `J` (columns × rows: column `j` of the layout `lay` is row `j` of `J`), counted on the
state's run record `run`."""
_scrub!(J, lay, run::_LmRun) = _scrub_entries!(J, 2, lay, run)

"""
    _device_copy(s, A) -> A on the state's backend

The backend's copy of the host array `A` (a block's basis: the vignetting's free directions, the warping's height basis), adapted
once per array object and kept while the object lives (the blocks' updates keep their basis): the per-block Jacobians read it
without an upload each.
"""
function _device_copy(s::CalibrationState, A::AbstractArray)
    be = s.backend
    be isa KA.CPU && return A
    return lock(_DEVICE_CACHE_LOCK) do
        get!(() -> _shared_read_only!(be, adapt(be, A)), _DEVICE_COPIES, A)
    end
end
const _DEVICE_COPIES = WeakKeyDict{Any, Any}()
"""Held while a lazily built entry of a state's device cache or a device copy is formed (`_device_copy`, `_cached_layout`,
`_view_layouts`, `_focus_weights`): the views' jobs run concurrently (`_windowed`)."""
const _DEVICE_CACHE_LOCK = ReentrantLock()
"""The device form (`_kernel_layout`) of the layout `lay` in the dictionary `cache` (the state's device cache for a view, a tile's
own), formed once. The key is the layout itself (compared by value): varflags and `material_fixed` alone do not determine it — the
free board heights (`NH`), the in-plane print error (`NI`) and the block sizes change its slots, and a layout cached under a
coarser key was read against another layout's slot rows (`S`), past their end."""
_cached_layout(cache::AbstractDict, be, lay) =
    lock(() -> get!(() -> (k = _kernel_layout(be, lay); _shared_read_only!(be, k.scatter); k.free === nothing || _shared_read_only!(be, k.free); k),
                    cache, lay), _DEVICE_CACHE_LOCK)

"""
    _column_map(lay, c, foc, ap, m) -> T

The map from the layout's full columns (`lay`) to the problem's coordinates of the blocks: the aperture's log chart (`_log_r_column!`)
and the material's charts (`_chart_material_columns!`) as column scales, a channel-`c` state's tie bases (`_tie_columns`) and the held
focus nodes left out (`_restricted_focus`) — the column operations of the Jacobian, applied to the identity (`ncol × n`). The problem's
Jacobian is `J_full T`.
"""
function _column_map(lay, c::Integer, foc, ap, m)
    T = Matrix{Float64}(I, lay.ncol, lay.ncol)
    _log_r_column!(T, lay, ap)
    _chart_material_columns!(T, lay, m)
    return _restricted_focus(_tie_columns(T, lay, c), lay.focus_cols, foc)
end

"""
    _gauss_newton_blocks(J, r, T, lay, run) -> (Jᵀr, JᵀJ)

The Gauss–Newton blocks of a block's rows in the problem's coordinates: `Tᵀ(J r)` and `Tᵀ(J Jᵀ)T` of the device Jacobian `J` (full
columns × rows) and residual `r`. The products over the rows, `J r` and `J Jᵀ`, are formed on their backend (BLAS on the CPU, the
device's library on a GPU: one code path) and only their `ncol` and `ncol × ncol` values reach the host; the column map `T`
(`_column_map`: a column scale, a selection, a channel tie — a few entries per column) is applied there as a sparse matrix. (Applied
on the device as a dense matrix, `T` was an upload and three small dense products per block: 0.84 ms of a tile's 1.43 ms on a B200,
2026-11-20, and on the CPU two dense `ncol³` products per tile. A scale or a selection gives the same entries as the dense
products, whose other terms are exact zeros; a channel tie's sums may round differently.)
"""
function _gauss_newton_blocks(J, r, T::Matrix{Float64}, lay, run::_LmRun)
    g, H = Array(_rows_times(J, r)), Array(J * J')
    if !(all(isfinite, g) && all(isfinite, H))
        # a non-finite Jacobian entry makes its column's diagonal of J Jᵀ non-finite: scrubbed (`_scrub_entries!`) and formed again —
        # the blocks of the scrubbed Jacobian; a finite Jacobian (every block but a defect's) skips the scrub's sweep over it
        _scrub!(J, lay, run)
        g, H = Array(_rows_times(J, r)), Array(J * J')
    end
    return _in_problem_coordinates(SparseArrays.sparse(T), g, H)
end
"""
    _in_problem_coordinates(T, g, H) -> (Tᵀg, TᵀHT)

The full-column products `g = J r` and `H = J Jᵀ` in the problem's coordinates under the sparse column map `T`
(`ncol × n`): one pass over `T`'s stored entries per output entry, written straight into the `n` and `n × n` results (no
`ncol × n` intermediate: a tile's 289² values twice per block were 3.5 GB of host allocation per grid linearization of 864 tiles).
"""
function _in_problem_coordinates(T::SparseArrays.SparseMatrixCSC{Float64}, g::Vector{Float64}, H::Matrix{Float64})
    n = size(T, 2); rows = SparseArrays.rowvals(T); vals = SparseArrays.nonzeros(T)
    gp = Vector{Float64}(undef, n); Hp = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n
        rj = SparseArrays.nzrange(T, j)
        acc = 0.0
        for q in rj
            acc += vals[q] * g[rows[q]]
        end
        gp[j] = acc
        for i in 1:n
            acc = 0.0
            for p in SparseArrays.nzrange(T, i)             # (Tᵀ H T)[i, j] = Σ_p T[a_p, i] Σ_q H[a_p, b_q] T[b_q, j]
                a = rows[p]; inner = 0.0
                for q in rj
                    inner += H[a, rows[q]] * vals[q]
                end
                acc += vals[p] * inner
            end
            Hp[i, j] = acc
        end
    end
    return gp, Hp
end
"""
    _rows_times(J, r) -> J r

The device Jacobian `J` (columns × rows) times the residual `r`: `J * r` (the host's BLAS on the CPU backend). RC's CUDA extension
gives the device's method: the vendor library's matrix–vector product of a wide matrix is slow (73 × 230,400 Float64 on a B200:
14.9 ms, its matrix–matrix product of the same sizes 0.33 ms; 2026-11-20), so the product is taken as a one-column matrix product
there (0.10 ms; the same sums to rounding, 3.5e-14 relative).
"""
_rows_times(J, r) = J * r

"""The block's Jacobian in the problem's coordinates on the host (`J_fullᵀ T`, rows × n): `computeresjac`'s, outside the LM iteration
(the tests, the stall postcondition, the grid smoothness's weight selection)."""
_host_jacobian(J, T::Matrix{Float64}, lay, run::_LmRun) = (_scrub!(J, lay, run); Array(transpose(J) * adapt(KA.get_backend(J), T)))

"""The device Jacobian and residual buffers of a block of the state `s` (`ncol × M`, `M`) on its backend, zeroed, reused per task:
valid until the task's next block. Within a registered problem's solve (`_register_window!`) they are the problem's, one pair per
task, released with it (`_unregister_window!`), so the pooled device workers hold no device memory after a solve; outside one, the
task's own (Core's `task_buffers`, the stack's one reuse policy)."""
function _jacobian_buffers(s::CalibrationState, ncol::Integer, M::Integer)
    be = s.backend
    alloc() = (KA.allocate(be, Float64, 0), KA.allocate(be, Float64, 0))
    w = _device_entry(s, :window)
    bufs = w === nothing ? VgtkCore.task_buffers(alloc, (:radiometric_camera_jacobian, typeof(be))) :
                           lock(() -> get!(alloc, w.buffers, current_task()), w.buffers_lock)
    Jb, rb = bufs
    length(Jb) < ncol * M && resize!(Jb, ncol * M)
    length(rb) < M && resize!(rb, M)
    J = reshape(view(Jb, 1:(ncol * M)), Int(ncol), Int(M))
    fill!(J, 0.0)
    return J, view(rb, 1:M)
end

"""Per-state kernel data (the scene in the kernels' float type, adapted to the backend; the pixel-id lists and their index lists,
`_index_list`), built lazily."""
function _ensure_device!(s::CalibrationState)
    s.device_cache === nothing || return s.device_cache
    be = s.backend
    F = _kernel_float(be)
    # the CPU backend's kernels read the host scene itself (Float64 already: no copy of the board's maps)
    scene32 = be isa KA.CPU && F === _float_of(s.settings) ? s.scene : adapt(be, VgtkCore.convert_eltype(F, s.scene))
    px = [adapt(be, UInt32.(UInt32.((getindex.(p, 1) .- 1) .* s.scene.H .+ getindex.(p, 2)))) for p in s.px]
    idx = [_index_list(be, length(p)) for p in px]
    # read by every chunk of every view, written only here: out of the device's per-buffer queue tracking, the lists one by one (an
    # adaptor does not descend into a Vector of device arrays — the lists stayed tracked, and a tile's row kernel on the main queue
    # waited for the next view's whole batch on its worker's queue: 85 % of a grid linearization, 2026-11-13)
    foreach(a -> _shared_read_only!(be, a), px); foreach(a -> _shared_read_only!(be, a), idx)
    s.device_cache = _shared_read_only!(be, (; scene = scene32, px, idx))
    return s.device_cache
end

"""The scene of the nodal heights' path (`HIGH_FLOAT`) on the state's backend, adapted once and kept in the device cache (the
kernels' scene itself where the two float types agree)."""
function _height_scene(s::CalibrationState)
    dc = _ensure_device!(s)
    hasproperty(dc, :scene_h) && return dc.scene_h
    be = s.backend
    G = HIGH_FLOAT
    sh = G === _kernel_float(be) ? dc.scene : _shared_read_only!(be, adapt(be, VgtkCore.convert_eltype(G, s.scene)))
    s.device_cache = merge(s.device_cache, (; scene_h = sh))
    return sh
end
"""
    _height_inputs(s, view, vig, m::MaterialVar, light[, lens]) -> (; view, vig, mat, light, scene, st)

The inputs of the nodal heights' radiance record and height kernel at the trial parameters, in `HIGH_FLOAT` (Float64). `view` is
host-side; its lens is adapted here (`lens`: an already assembled grid, `_lens32` of it in the float type).
"""
function _height_inputs(s::CalibrationState, view::View, vig, m::MaterialVar, light, lens = nothing)
    be = s.backend
    G = HIGH_FLOAT
    scene = with_surface(_height_scene(s), VgtkCore.convert_eltype(G, m.surface))
    vG = lens === nothing ? adapt(be, VgtkCore.convert_eltype(G, view)) :
         (l = _lens32(be, lens, G); View{G, length(view.focus.t̂_i), typeof(l)}(_pose_eltype(G, view.pose), l, VgtkCore.convert_eltype(G, view.focus), G(view.r)))
    return (; view = vG, vig = VgtkCore.convert_eltype(G, vig), mat = VgtkCore.convert_eltype(G, m.mat), light = VgtkCore.convert_eltype(G, light),
            scene, st = VgtkCore.convert_eltype(G, s.settings))
end
"""The parameter pack of a per-view state in the kernels' float type."""
_pack32(s::CalibrationState, view::View, vig::AbstractVignetting, mat::Material, light::EnvironmentLight) =
    (F = _kernel_float(s.backend); (VgtkCore.convert_eltype(F, view), VgtkCore.convert_eltype(F, vig), VgtkCore.convert_eltype(F, mat),
                                    VgtkCore.convert_eltype(F, light), VgtkCore.convert_eltype(F, s.settings)))

"""Masked flux vector (3·npx, Float64 host) of view `k` at (view, vig, mat, light, surface) on the device backend (`surf` defaults to the
state's accepted surface; the LM gives the trial block's)."""
function _flux_device(s::CalibrationState, k::Integer, view::View, vig::AbstractVignetting, mat::Material, light::EnvironmentLight,
                      surf::BoardSurface = s.scene.surface)
    v32, _, _, _, _ = _pack32(s, view, vig, mat, light)
    return _flux_device_px(s, k, _ensure_device!(s).px[k], adapt(s.backend, v32), vig, mat, light, surf)   # adapt: a grid lens carries arrays
end

"""As above over an explicit device pixel-id list with a view already in the kernels' float type (device-capable)."""
function _flux_device_px(s::CalibrationState, k::Integer, px_d, v32, vig::AbstractVignetting, mat::Material, light::EnvironmentLight,
                         surf::BoardSurface = s.scene.surface)
    _count_device_call!(s, :flux)
    be = s.backend
    npx = length(px_d)
    npx == 0 && return Float64[]                          # a view without masked pixels (its board outside the image circle): no launch (ndrange 0 fails)
    return Float64.(_device_rows(_flux_functor(s, k, v32, vig, mat, light, surf), be, px_d))
end
"""The flux functor (`_FluxAt`) of view `k` at the view `v32` in the kernels' float type (device-capable) and the host vignetting, material, lighting and
surface (converted here)."""
function _flux_functor(s::CalibrationState, k::Integer, v32, vig::AbstractVignetting, mat::Material, light::EnvironmentLight,
                       surf::BoardSurface = s.scene.surface)
    F = _kernel_float(s.backend)
    vig32, m32, l32, st32 = VgtkCore.convert_eltype(F, vig), VgtkCore.convert_eltype(F, mat), VgtkCore.convert_eltype(F, light), VgtkCore.convert_eltype(F, s.settings)
    scene32 = with_surface(_ensure_device!(s).scene, VgtkCore.convert_eltype(F, surf))
    return _FluxAt(Int32(k), Int32(s.scene.H), v32, vig32, m32, l32, scene32, st32)
end

@inline _chunk_dual(::Type{Tag}, ::Type{T}, x, gslot::Integer, c0::Integer, ::Val{C}, w = 1.0) where {Tag, T, C} =
    ForwardDiff.Dual{Tag}(T(x), ForwardDiff.Partials(ntuple(i -> (c0 + i == gslot) ? T(w) : zero(T), Val(C))))

"""
    _view_jacobian(varflags, r, pose, dist, foc, ap, m, vg, lt) -> (; J, res, flux, lay, T)

A view's Jacobian on the state's backend (`PerViewResidual`), every backend: chunked dual kernels over all blocks (the aperture radius
is one of the dual slots) in the kernels' float type (`_kernel_float`: Float32 on a device, Float64 on the CPU backend; `_jac_chunk`
partials per chunk), the nodal heights by the height kernel, the linear blocks' analytic columns by the parts kernel, the flux by the
flux kernel — all kept on the device and assembled by the row kernel (`_jacobian_row!`): `J` (the layout's full columns × the measured
rows) and the residual `res` on the backend, the flux (device), the layout and the column map `T` to the problem's coordinates
(`_column_map`).
"""
function _view_jacobian(varflags, r::PerViewResidual{N}, pose::_PoseBlock, dist::_LensBlock,
                        foc::_FocusBlock, ap::ApertureVar, m::MaterialVar, vg::VgtkCore.LinearSubspaceManifold, lt::LightingVar) where {N}
    _count_device_call!(r.state, :jac)
    s = r.state
    dc = _ensure_device!(s)
    be = s.backend
    F = _kernel_float(be)
    NV = Int(n_control_points(VgtkCore.model(vg))); J = Int(sh_count(max_degree(lt.light)))
    NS = length(s.settings.sensor); ND = VgtkCore.tangent_size(dist); NH = n_height(m); NI = n_inplane(m)
    lay = _jacobian_layout(varflags, N, NS, VgtkCore.tangent_size(vg), NV, J, ND; NH, NI, fixed = material_fixed(m))
    P = lay.P
    view0 = _view_of(r, pose, dist, foc, ap)
    npx = length(s.px[r.k])
    need_v = !isempty(lay.vig_cols)
    need_lin = need_v || !isempty(lay.light_cols) || lay.μ_col > 0
    need_f = !isempty(lay.focus_cols)
    need_h = !isempty(lay.height_cols)
    stk = VgtkCore.convert_eltype(F, s.settings)
    C = _jac_chunk(s, P)
    c0s = _chunk_starts(P, C)                                        # (the heights: the height kernel's, below; no dual slots)
    # linear blocks: analytic columns from the parts kernel, so vignetting and lighting are plain here (not duals with zero
    # partials: the same values and partials, fewer registers)
    vigk = VgtkCore.convert_eltype(F, VgtkCore.model(vg)); lk = VgtkCore.convert_eltype(F, lt.light)
    vk = adapt(be, VgtkCore.convert_eltype(F, view0))                 # adapt: a grid lens carries arrays
    mk = VgtkCore.convert_eltype(F, m.mat); scenek = with_surface(dc.scene, VgtkCore.convert_eltype(F, m.surface))
    G = HIGH_FLOAT                                             # the heights' and the pose chunks' float type and inputs
    hk = _needs_high(F, need_h) ? _height_inputs(s, view0, VgtkCore.model(vg), m, lt.light) : nothing
    # the estimator once per ray: in `HIGH_FLOAT` where a chunk needs it (a device: the pose's), and then that record's numbers
    # rounded to the kernels' float for the other chunks and the parts (`_record_as`), not a second record
    if G === F || hk === nothing                               # (the CPU backend: the chunks' record is already Float64)
        rays = _radiance_record_device(be, dc.px[r.k], dc.idx[r.k], r.k, s.scene.H, vk, vigk, mk, lk, scenek, stk)
        hrays = rays
    else
        hrays = _radiance_record_device(be, dc.px[r.k], dc.idx[r.k], r.k, s.scene.H, hk.view, hk.vig, hk.mat, hk.light, hk.scene, hk.st)
        rays = _record_as(F, hrays)
    end
    _shared_read_only!(be, (vk, rays))                                          # the view's lens and the record: read by every chunk below
    hrays === rays || _shared_read_only!(be, hrays)
    plan = _chunk_plan(c0s, F, (vigk, lk, dc.scene, stk, rays), hk === nothing ? nothing : (hk.vig, hk.light, hk.scene, hk.st, hrays))
    hs = _each_chunk(be, plan) do (c0, Fc, inputs, _)                 # the chunks' launches overlap (one worker queue each)
        _view_chunk(Val(Fc), s, r, lay, c0, C, pose, dist, foc, ap, m, inputs...)
    end
    S = _slot_rows(be, [(c0, h, C) for (c0, h) in zip(c0s, hs)], P, npx)
    hh = need_h ? _heights_device(be, dc.px[r.k], dc.idx[r.k], r.k, s.scene.H, hk.view, hk.vig, hk.mat, hk.light, hk.scene, hk.st, m.surface.h; rays = hrays) :
         nothing                                           # the nodal heights: the height kernel's partials
    ph = need_lin ? _parts_device(be, dc.px[r.k], dc.idx[r.k], r.k, s.scene.H, vk, vigk, mk, lk, scenek, stk, J, NV; rays) : nothing
    # The residual must be BITWISE the one `computeresidual` returns for the same parameters: NLLSsolver
    # recomputes the cost through this function after every accepted step and stops ("internal discrepancy
    # in cost computations") if it differs from the trial cost beyond isapprox — and the value part of the
    # dual kernel differs from the plain flux kernel in its last bits. So take it from the flux kernel.
    flux = _launch_rows(_flux_functor(s, r.k, vk, VgtkCore.model(vg), m.mat, lt.light, m.surface), be, dc.px[r.k])
    Gw, gidx = need_f ? _focus_weights(s, r.k, FocusSpline(VgtkCore.model(foc)), view0.lens, 1:npx, dc.idx[r.k]) : (nothing, nothing)
    chs = fit_channels(s)
    Jd, rd = _jacobian_buffers(s, lay.ncol, length(chs) * npx)
    KA.synchronize(be)
    _launch_pixel_rows!(_jacobian_row!, Jd, npx, rd, dc.px[r.k], dc.idx[r.k], gidx, Int32(s.scene.H), S, flux, _view_phi(s, r.k), _view_weights_device(s, r.k), chs,
                           _cached_layout(_view_layouts(s), be, lay), ph, (J, NV), hh,
                           need_h ? _device_copy(s, m.dense_basis) : nothing, Gw, Val(N), s.settings.sensor, nothing, nothing, (0, 0),
                           nothing, lt.scale, need_v ? _device_copy(s, vg.basis) : nothing)
    _note_block_bytes!(s, _device_bytes(rays, hrays === rays ? nothing : hrays, S, hh, ph, flux, Jd, rd) +
                       sum(sizeof, hs; init = 0))
    return (; J = Jd, res = rd, flux, lay, T = _column_map(lay, s.channel, foc, ap, m))
end
"""The device layouts of the views' Jacobians, keyed by the layout (`_cached_layout`; every view of a state has the same layout for
the same free blocks), kept in the device cache."""
function _view_layouts(s::CalibrationState)
    return lock(_DEVICE_CACHE_LOCK) do
        dc = _ensure_device!(s)
        d = get(dc, :view_layouts, nothing)
        d === nothing && (d = Dict{Any, Any}(); s.device_cache = merge(dc, (; view_layouts = d)))
        d
    end
end
"""The measured values of view `k` on the state's backend (`s.ϕ[k]`), adapted once and kept in the device cache."""
function _view_phi(s::CalibrationState, k::Integer)
    dc = _ensure_device!(s)
    ϕs = get(dc, :phi, nothing)
    ϕs === nothing && (ϕs = [_shared_read_only!(s.backend, adapt(s.backend, p)) for p in s.ϕ]; s.device_cache = merge(dc, (; phi = ϕs)))
    return ϕs[k]
end

"""The rows' weights of view `k` on the state's backend (`_view_weights` of `s.weights`: `Unweighted()`, or its 1/σ vector adapted
once and kept in the device cache)."""
_view_weights_device(s::CalibrationState, k::Integer) = _view_weights_device(s, s.weights, k)
_view_weights_device(s::CalibrationState, w::Unweighted, k::Integer) = w
_view_weights_device(s::CalibrationState, w::AbstractVector, k::Integer) =
    lock(_DEVICE_CACHE_LOCK) do                              # (the views run concurrently: `_windowed`)
        dc = _ensure_device!(s)
        ws = get(dc, :weights, nothing)
        ws === nothing && (ws = [_device_weights(s.backend, p) for p in w]; s.device_cache = merge(dc, (; weights = ws)))
        ws[k]
    end

"""The per-view `computeresjac` (`PerViewResidual`; NLLSsolver's method, per_tile.jl) as host arrays: dual numbers over the free AD
blocks (pose 6, distortion ND, focus N, material 8, the aperture radius seeded through the unit-disc rule), the nodal heights through
the ray parameter, the analytic columns of the linear blocks — ONE implementation on every backend (`_view_jacobian`): the residual of
the flux kernel (bitwise `computeresidual`'s) and the view's Jacobian in the problem's coordinates (`_host_jacobian`), for the tests,
the stall postcondition and the diagnostics; the LM's iterations take the Gauss–Newton blocks on the device (`computecostgradhess`)."""
function _block_jacobian(varflags, r::PerViewResidual, vars...)
    v = _view_jacobian(varflags, r, vars...)
    return _whiten(_view_weights(r.state.weights, r.k), _channel_rows(Float64.(Array(v.flux)), r.state.channel) .- r.state.ϕ[r.k]),
           _host_jacobian(v.J, v.T, v.lay, r.state.run)
end

"""One chunk of the per-view Jacobian (`_view_jacobian`) in the float type `F`: the chunk's slots `c0 + 1 … c0 + C` seeded,
the plain vignetting, lighting and settings, the scene `scene` (its surface replaced by the seeded one) and the record `rays` all in
`F` (`_kernel_float`, or `HIGH_FLOAT` for the pose's chunks)."""
function _view_chunk(::Val{F}, s::CalibrationState, r::PerViewResidual{N}, lay, c0, C, pose, dist, foc, ap, m, vigk, lk, scene, stk,
                     rays) where {F, N}
    be = s.backend
    D, cd, pose_d, focus_d, mat_d, scene_d = _seed_chunk(Val(F), c0, C, lay, pose, foc, m, scene, n_inplane(m))
    lens_d = _seeded_lens(dist, s.camera.lens, x -> cd(x, 0), j -> cd(0.0, lay.off_d + j))   # a pixel camera is isbits: to the kernel as-is
    view_d = View{D, N, typeof(lens_d)}(pose_d, lens_d, focus_d, cd(ap.r, lay.col_r > 0 ? lay.off_r + 1 : 0))   # r seeded: AD through r² w̃ and r â
    fbase = Int32(!isempty(lay.focus_cols) ? lay.off_f - c0 : -1000000)
    return _jac_device(be, s.device_cache.px[r.k], s.device_cache.idx[r.k], r.k, s.scene.H, view_d, vigk, mat_d, lk, scene_d, stk, Val(C), fbase, -1000000;
                       rays)   # waits on this chunk's stream only
end

"""
    _seed_chunk(::Val{F}, c0, C, lay, pose, foc, m, scene, NI) -> (D, cd, pose_d, focus_d, mat_d, scene_d)

The dual seeding shared by a Jacobian chunk of a view (`_view_chunk`) and of a tile (`_tile_chunk`) in the float type `F`: the chunk's
dual type `D` and seeder `cd(x, gslot)` (slots `c0 + 1 … c0 + C`), the seeded pose (`lay.off_e`), the focus (`t̂` seeded per sensor
sample in-kernel, the chain rule), the material and the scene `scene` with the seeded surface (its nodal heights unseeded: the height
kernel's); `NI` in-plane parameters.
"""
function _seed_chunk(::Val{F}, c0, C, lay, pose, foc, m, scene, NI) where {F}
    Tag = JacTag
    N = length(VgtkCore.model(foc).t̂_i)
    D = ForwardDiff.Dual{Tag, F, C}
    cd(x, gslot) = _chunk_dual(Tag, F, x, gslot, c0, Val(C))
    coeff = Matrix{F}(FocusSpline(VgtkCore.model(foc)).coeff)
    nodes = SVector{N}(SVector{2, F}.(VgtkCore.model(foc).nodes))
    ω = SVector(ntuple(i -> cd(0.0, lay.off_e + i), Val(3)))
    ρ = SVector(ntuple(i -> cd(0.0, lay.off_e + 3 + i), Val(3)))
    pose_d = _seeded_pose(VgtkCore.model(pose), x -> cd(x, 0), ω, ρ)
    t̂d = SVector(ntuple(i -> cd(VgtkCore.model(foc).t̂_i[i], 0), Val(N)))
    wa = coeff * collect(t̂d)
    focus_d = Focus{D, N}(nodes, t̂d, SVector{N, D}(ntuple(i -> wa[i], Val(N))), SVector{3, D}(wa[N + 1], wa[N + 2], wa[N + 3]), VgtkCore.model(foc).kind)
    mat_d, surf_d = _material_dual(m.mat, m.surface, cd, lay.off_m, D; off_h = 0, off_i = NI > 0 ? lay.off_i : 0)
    return D, cd, pose_d, focus_d, mat_d, with_surface(scene, surf_d)
end

"""GPU render of view `k` at the current state (returns the H×W×3 Float64 image)."""
function _render_view_device(s::CalibrationState, k::Integer)
    Φ = _flux_device(s, k, view_of(s, k), s.camera.vignetting, s.mat, s.light)
    img = zeros(Float64, s.scene.H, s.scene.W, 3)
    for (i, (u, v)) in enumerate(s.px[k]), c in 1:3
        img[v, u, c] = Φ[3 * (i - 1) + c]
    end
    return img
end

