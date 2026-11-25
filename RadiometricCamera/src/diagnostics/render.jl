# ---- render dumps --------------------------------------------------------------------------------------------------------------
"""Render the masked pixels of every view at the state (no fitting): per-view level ratios on ink and paper, Float32 photo / model dumps
(green channel, column-major H × W, NaN outside the mask) and Float16 RGB dumps in `out`."""
function render_dump!(s::CalibrationState, names, out::AbstractString, label::AbstractString)
    K = n_views(s); npx = sum(length.(s.px))
    c, rms = cost(s); @printf("render from %s: %d views, %d masked pixels, cost %.4e rel-RMS %.4f\n", label, K, npx, c, rms); flush(stdout)
    r = photometric_residuals(s)                                   # in the photographs' units: the model is ϕ + r
    sse_all = 0.0; sse_gain = 0.0; ϕ2 = 0.0
    for k in 1:K
        Pk = s.ϕ[k]; Mk = Pk .+ r[k]; g = dot(Mk, Pk) / dot(Mk, Mk); lvl = median(Pk)
        ink = Pk .< 0.4lvl; paper = Pk .> 1.3lvl
        e0 = sum(abs2, r[k]); e1 = sum(abs2, g .* Mk .- Pk); sse_all += e0; sse_gain += e1; ϕ2 += sum(abs2, Pk)
        @printf("  %-14s level ratio model/photo %.3f  rel-RMS %.4f;  model/photo on paper %.3f, on ink %.3f\n", names[k], 1 / g, sqrt(e0 / sum(abs2, Pk)),
                mean(Mk[paper]) / mean(Pk[paper]), mean(Mk[ink]) / mean(Pk[ink]))
    end
    @printf("all views: rel-RMS %.4f (%.0f %% of the squared residual is a per-view level)\n", sqrt(sse_all / ϕ2), 100 * (1 - sse_gain / sse_all)); flush(stdout)
    chs = residual_channels(s); nc = length(chs); g1 = something(findfirst(==(2), chs), 1)   # the green channel's entry (a single channel: its own)
    for k in 1:K
        P = fill(NaN32, _image_height(s), _image_width(s)); M = fill(NaN32, _image_height(s), _image_width(s))
        for (i, (u, v)) in enumerate(s.px[k]); e = nc * (i - 1) + g1; P[v, u] = s.ϕ[k][e]; M[v, u] = s.ϕ[k][e] + r[k][e]; end
        base = joinpath(out, "render_" * replace(names[k], ".png" => ""))
        write(base * "_photo.f32", P); write(base * "_model.f32", M)
        P3 = fill(Float16(NaN), _image_height(s), _image_width(s), 3); M3 = fill(Float16(NaN), _image_height(s), _image_width(s), 3)
        for (i, (u, v)) in enumerate(s.px[k]), (j, c) in enumerate(chs)
            e = nc * (i - 1) + j; P3[v, u, c] = Float16(s.ϕ[k][e]); M3[v, u, c] = Float16(s.ϕ[k][e] + r[k][e])
        end
        write(base * "_photo_rgb.f16", P3); write(base * "_model_rgb.f16", M3); P3 = M3 = nothing
        @printf("  %s: %d px, rel-RMS %.4f → %s_{photo,model}.f32 (Float32 %d × %d, column-major)\n", names[k], length(s.px[k]), sqrt(sum(abs2, r[k]) / sum(abs2, s.ϕ[k])), base, _image_height(s), _image_width(s))
    end
    return nothing
end
