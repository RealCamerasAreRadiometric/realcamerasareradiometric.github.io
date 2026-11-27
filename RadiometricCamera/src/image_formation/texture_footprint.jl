# ---------------------------------------------------------------------------
# A pixel's footprint over a printed page's texture: the blur row of the renderer's quadrature (sensor samples × aperture-disc
# samples, each ray through the board onto the page's texels, bilinearly), the linear model of a photograph of a texture
# (I = V · Σ w x). One KernelAbstractions kernel on every backend (Core's `pointwise!`), its record the kernels' View.
# ---------------------------------------------------------------------------

"""
    texel_weights(𝐛, TW, TH) -> ((texel, weight) × 4) or nothing

The bilinear weights of the page position `𝐛` (∈ [0, 1]², the design's chart) over the `TW × TH` texels (texel `(i, j)` is column
`(j − 1) TW + i`, x fastest): Core's `bilinear_weights` (the half-texel chart of `bilinear_sample`), the position strictly inside the
texel centres' grid, `nothing` otherwise (Core's clamp would extend the edge; a footprint off the grid is dropped).
"""
@inline function texel_weights(𝐛, TW::Integer, TH::Integer)
    uv = SVector(𝐛[1], 𝐛[2])
    VgtkCore.is_inside_texel_centers(uv, (TW, TH)) || return nothing
    i, j, fx, fy = VgtkCore.bilinear_weights(uv, (TW, TH))
    tid(a, b) = (b - 1) * TW + a
    return ((tid(i, j), (1 - fx) * (1 - fy)), (tid(i + 1, j), fx * (1 - fy)), (tid(i, j + 1), (1 - fx) * fy), (tid(i + 1, j + 1), fx * fy))
end

"""
    texture_footprint_slots(settings) -> Int

The slots of a pixel's footprint row ([`texture_footprints!`](@ref)) under the render settings `settings`: 4 bilinear texels per
sensor sample × aperture-disc sample (its most distinct texels).
"""
texture_footprint_slots(st) = 4 * length(st.sensor) * VrtkCore.n_disc(st.aperture)

# The footprint row of a sampled pixel `(q, u, v)` (`q` its index in the launch) under `view`: a functor of Core's `pointwise!`.
struct _TextureFootprint{V, S, T, G, C, W}
    view::V
    board::S
    st::T
    vig::G
    TW::Int32
    TH::Int32
    cols::C
    vals::W
end
Adapt.@adapt_structure _TextureFootprint
@inline function (f::_TextureFootprint)(q::Int32, u::Int32, v::Int32)
    st, view = f.st, f.view
    bd = f.board; rays = VrtkCore.BoardRays(f.board)
    TW, TH = Int(f.TW), Int(f.TH)
    base = (Int(q) - 1) * texture_footprint_slots(st); n = 0; tot = 0.0
    @inbounds for ks in 1:length(st.sensor)
        𝐩 = VrtkCore.sensor_sample(st.sensor, ks, Int(u), Int(v)); rs = view.r; sw = st.sensor.weights[ks]
        for la in 1:VrtkCore.n_disc(st.aperture)
            𝐚 = VrtkCore.aperture_sample(st.aperture, la, rs)
            o, d = VrtkCore.world_ray(view, 𝐩, 𝐚); sr, 𝐱 = VrtkCore.intersect_board(rays, o, d)
            (isfinite(sr) && sr > 0) || return Int32(0)
            tw = texel_weights(VgtkCore.chart(bd, 𝐱), TW, TH); tw === nothing && return Int32(0)
            w = sw * VrtkCore.aperture_weight(st.aperture, la, rs)
            for (t, b) in tw                                           # the texel's slot (a linear search of the ≤ n so far), or a new one
                c = Int32(t); i = base + 1
                while i <= base + n && f.cols[i] != c
                    i += 1
                end
                if i > base + n
                    n += 1; f.cols[i] = c; f.vals[i] = 0.0
                end
                f.vals[i] += w * b
            end
            tot += w
        end
    end
    tot > 0 || return Int32(0)
    V = f.vig(VrtkCore.sensor_position(Int(u), Int(v)))
    @inbounds for i in (base + 1):(base + n)
        f.vals[i] = V * (f.vals[i] / tot)
    end
    @inbounds for i in (base + 2):(base + n)                              # the row's texels ascending: the order of a sparse operator's row
        c = f.cols[i]; x = f.vals[i]; j = i - 1
        while j > base && f.cols[j] > c
            f.cols[j + 1] = f.cols[j]; f.vals[j + 1] = f.vals[j]; j -= 1
        end
        f.cols[j + 1] = c; f.vals[j + 1] = x
    end
    return Int32(n)
end

"""
    texture_footprints!(n, cols, vals, camera, pose, board, settings, texture_size, us, vs; device) -> n

The footprint rows over a page's texture of the pixels `(us[q], vs[q])` of a photograph of `camera` (a `RadiometricCameraModel`,
the blur model) at `pose` (camera → world), the page the textured `board` and its texture `texture_size = (TW, TH)` texels, under
the render settings `settings` (the renderer's quadrature): for every sensor sample and every aperture-disc sample the ray
(`world_ray`), its intersection with the board and the bilinear texels there ([`texel_weights`](@ref)), weighted by the sensor and
aperture weights and accumulated per texel in sample order, normalised to a sum of 1 and scaled by the chief ray's vignetting
`V(𝐩)` (a pixel's model is I = V · Σ w x). Pixel `q`'s `n[q]` distinct texels go into the first `n[q]` of its
[`texture_footprint_slots`](@ref)`(settings)` slots from `(q − 1) · slots + 1` of `cols` (the texel, ascending) and `vals` (its
weight); `n[q] = 0` when a sample misses the board or the texel grid (the pixel is dropped). One kernel on `device` (Core's
`pointwise!`): `n`, `cols`, `vals`, `us` and `vs` live there.
"""
function texture_footprints!(n::AbstractVector{Int32}, cols::AbstractVector{Int32}, vals::AbstractVector{Float64}, camera::RadiometricCameraModel,
                             pose, board::TexturedTarget, settings::RenderSettings, texture_size::Tuple{Integer, Integer},
                             us::AbstractVector{Int32}, vs::AbstractVector{Int32}; device)
    f = _TextureFootprint(View(pose, camera), board, settings, camera.vignetting, Int32(texture_size[1]), Int32(texture_size[2]), cols, vals)
    VgtkCore.pointwise!(f, n, Adapt.adapt(device, Int32.(1:length(n))), us, vs; exec = VgtkCore.OnBackend(device))
    return n
end
