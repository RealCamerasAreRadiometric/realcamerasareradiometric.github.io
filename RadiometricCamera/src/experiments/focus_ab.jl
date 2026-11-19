# ---------------------------------------------------------------------------
# EXPERIMENT (focus-ab branch only): the A/B of the focus surface's parameterisation, selected by the `[focus]` table of the
# pipeline options (`FocusOptions`, read in `start_calibration`):
#
#   focus.variant     A  the sensor-grid TPS, every node free (the port as is; the default)
#                     B  the sensor-grid TPS, the nodes outside the run's support FROZEN at their fill values
#                     C  the direction-chart TPS (focus_charts.jl, kind 1) on `focus.node_count` nodes of the cap θ ≤ θ_max
#                     D  Wahba's spherical spline (kind 2) on the same nodes
#   focus.node_count  node count of C / D (0 = A's, 25)
#   focus.smoothness  [focus.smoothness] type = "bending": β of the bending smoothness ½ β (Σϕ²/N) t̂ᵀQ̃t̂ (Q̃ = Q / mean positive eigenvalue; 0 = none)
#   focus.perturb     [σ, seed]: the start's free nodal values × exp(σ z), z ~ N(0, 1) (the robustness start)
# θ_max is the largest chief-ray angle of the start lens on its image circle (Core's `lens_coverage` of the run's lens) or, without
# one, 1.05 × the largest chief-ray angle of the detections (the border of a grid lens is an extrapolation: siemens_coarse_full's
# reaches 178°).
# ---------------------------------------------------------------------------

"""Largest and smallest chief-ray angle (to +z) of `lens0` on its image circle (Core's `lens_coverage`, an `Ellipse`: half a pixel
inside it; its points on the sensor) or, when the lens lights the whole sensor, on the image border."""
function focus_theta_max(lens0, W::Integer, H::Integer)
    e = VgtkCore.lens_coverage(lens0)
    pts = if !(e isa VgtkCore.Ellipse)
        bx = range(0.0, Float64(W); length = 401); by = range(0.0, Float64(H); length = 401)
        vcat([SVector(x, 0.0) for x in bx], [SVector(x, Float64(H)) for x in bx], [SVector(0.0, y) for y in by], [SVector(Float64(W), y) for y in by])
    else
        e_in = VgtkCore.Ellipse(e.center, e.a - 0.5, e.b - 0.5, e.θ)
        [SVector{2, Float64}(q[1], q[2]) for q in (VgtkCore.boundary_point(e_in, φ) for φ in range(0, 2π; length = 721)[1:720])
         if 0 <= q[1] <= W && 0 <= q[2] <= H]
    end
    θ = Float64[]
    for 𝐩 in pts
        d = _chief_ray(lens0, 𝐩)
        all(isfinite, d) && push!(θ, acos(clamp(normalize(d)[3], -1.0, 1.0)))
    end
    isempty(θ) && error("focus_theta_max: the start lens images no point of the border")
    return maximum(θ), minimum(θ)
end

"""
    focus_variant(fo::FocusOptions, lens0, nodes, focusA, in_support, d_c, W, H) -> (focus, in_support, info)

The start focus of a variant (`focusA` is A's, `nodes` its 5 × 5 sensor grid; `info[:free]` its free nodes, the state's
`focus_fixed` the others — B holds the nodes outside the support); the smoothness's rows are `focus_smoothness!`'s, once the observations are
known.
"""
function focus_variant(fo::FocusOptions, lens0, nodes, focusA::Focus, in_support, d_c, W::Integer, H::Integer; features = nothing)
    variant = uppercase(fo.variant)
    info = Dict{Symbol, Any}(:variant => variant)
    if variant == "A" || variant == "B"
        f = focusA
        free = variant == "B" ? in_support.(nodes) : fill(true, length(nodes))
        info[:kind] = 0; info[:free] = free; sup = in_support
    elseif variant == "C" || variant == "D"
        n = fo.node_count > 0 ? fo.node_count : length(nodes)
        θmax, θmin = focus_theta_max(lens0, W, H)
        if !(VgtkCore.lens_coverage(lens0) isa VgtkCore.Ellipse) && features !== nothing          # no image circle: the start lens's border can be an extrapolation (a
            θf = Float64[]                                            # grid lens past its calibrated region): the cap is the detections'
            for fv in features, (𝐩, _) in fv                          # largest chief-ray angle + 5 %
                d = _chief_ray(lens0, 𝐩); all(isfinite, d) && push!(θf, acos(clamp(normalize(d)[3], -1.0, 1.0)))
            end
            info[:theta_max_border] = θmax; info[:theta_max_detections] = maximum(θf)
            θmax = min(1.05 * maximum(θf), θmax)
            @printf("  focus cap: border θ %.2f° … %.2f°, detections up to %.2f° → θ_max %.2f°\n", rad2deg(θmin), rad2deg(info[:theta_max_border]), rad2deg(maximum(θf)), rad2deg(θmax))
        end
        cn = polar_cap_nodes(n, θmax)
        kind = variant == "C" ? FOCUS_DIRECTION_CHART : FOCUS_SPHERE
        t0 = [max(ae_dir(ξ)[3] / d_c, 0.0) for ξ in cn]                    # the plane start cos θᵢ / 𝔱₀, floored at 0 (back rays)
        f = Focus(FocusSpline(cn, kind), t0)
        info[:kind] = kind; info[:free] = fill(true, n); info[:theta_max] = θmax; info[:theta_min_border] = θmin
        sup = _ -> true                                                    # every node is a direction inside the cap
    else
        error("focus.variant must be A, B, C or D (got $variant)")
    end
    pert = isempty(fo.perturb) ? "" : join(fo.perturb, ",")
    if !isempty(fo.perturb)
        σ, seed = fo.perturb[1], Int(fo.perturb[2])
        z = randn(MersenneTwister(seed), length(f.t̂_i))
        t = [info[:free][i] ? f.t̂_i[i] * exp(σ * z[i]) : f.t̂_i[i] for i in eachindex(f.t̂_i)]
        f = Focus(FocusSpline(f), t)
        info[:perturb] = (; σ, seed)
    end
    info[:start] = collect(f.t̂_i)
    @printf("FOCUS A/B variant %s: %s, %d nodes, %d free%s%s\n", variant,
            info[:kind] == 0 ? "sensor-grid TPS" : info[:kind] == 1 ? "direction-chart TPS (azimuthal-equidistant)" : "spherical spline (Wahba m = 2)",
            length(f.t̂_i), count(info[:free]),
            haskey(info, :theta_max) ? @sprintf("; cap θ_max %.2f° (border θ %.2f° … %.2f°)", rad2deg(info[:theta_max]), rad2deg(info[:theta_min_border]), rad2deg(info[:theta_max])) : "",
            isempty(pert) ? "" : "; START PERTURBED ($pert)"); flush(stdout)
    return f, sup, info
end

"""The bending smoothness rows of the run from `[focus.smoothness]` (its weight β), written into the state `s` (its `focus_smoothness`, every focus block's smoothness
residual reads it; none for β = 0): `√(β Σϕ²/N) L/√m̄`, `L` Core's `bending_rows` of the
focus spline's system (one row per positive eigenvalue of the bending energy Q) and m̄ Core's `mean_positive_eigenvalue` of it:
`‖L t̂‖²/m̄ = t̂ᵀQt̂/m̄`, the bending energy relative to its mean positive eigenvalue."""
function focus_smoothness!(s, info, f::Focus, ϕ²::Real, sm::AbstractSmoothness)
    β = Float64(smoothness_weight(sm))
    info[:smoothness] = smoothness_record(sm)                     # focus_ab.json's (rc/focus_ab v2.0)
    β > 0 || return info
    N = length(f.t̂_i)
    Lc = VgtkCore.bending_rows(FocusSpline(f).sys)
    L = sqrt(β * ϕ² / N / VgtkCore.mean_positive_eigenvalue(Lc)) .* Lc
    s.focus_smoothness = L
    info[:smoothness_row_weight] = β * ϕ² / N
    @printf("focus smoothness %s: row weight β Σϕ²/N %.4g, start smoothness cost %.4g\n", smoothness_line(sm), β * ϕ² / N, 0.5 * sum(abs2, L * collect(f.t̂_i))); flush(stdout)
    return info
end

"""Per-node pixel counts of the masked pixels (nearest node of their chief direction through `lens`; the sensor's pixels for kind 0), the
largest chief-ray angle of a masked pixel, and the number of masked pixels the lens gives no ray (a non-finite back-projection: a
parametric lens refined past its invertible range at a corner, 2026-11-08) — counted, in no node, not in the angle."""
function focus_node_pixel_counts(f::Focus, lens, px, W::Integer, H::Integer)
    N = length(f.t̂_i)
    cnt = zeros(Int, N); θpx = 0.0; noray = 0
    dirs = [ae_dir(SVector{2, Float64}(n)) for n in f.nodes]
    for pv in px, (u, v) in pv
        𝐩 = 𝐩_of_pixel(u, v)
        d = normalize(SVector{3, Float64}(_chief_ray(lens, 𝐩)))
        all(isfinite, d) || (noray += 1; continue)
        θpx = max(θpx, acos(clamp(d[3], -1.0, 1.0)))
        i = f.kind == 0 ? argmin([sum(abs2, 𝐩 - n) for n in f.nodes]) : argmax([dot(d, e) for e in dirs])
        cnt[i] += 1
    end
    return cnt, θpx, noray
end

"""focus_ab.json: the variant as built, its start and final nodal values, and the node pixel counts."""
function write_focus_ab(path, info, s::CalibrationState, W::Integer, H::Integer)
    f = focus(s)
    cnt, θpx, noray = focus_node_pixel_counts(f, lens(s), s.px, W, H)
    noray > 0 && (@printf("focus_ab.json: %d of %d masked pixels have no ray through the lens (non-finite back-projection): in no node count\n",
                          noray, sum(length, s.px)); flush(stdout))
    VgtkIO.write_record(path, (; schema = RECORD_SCHEMAS.focus_ab, variant = info[:variant], kind = info[:kind], nodes = [collect(n) for n in f.nodes],
            node_dirs = f.kind == 0 ? nothing : [collect(ae_dir(SVector{2, Float64}(n))) for n in f.nodes],
            free = collect(info[:free]), n_free = count(info[:free]), start = info[:start], final = collect(f.t̂_i),
            theta_max = get(info, :theta_max, nothing), theta_min_border = get(info, :theta_min_border, nothing), theta_max_border = get(info, :theta_max_border, nothing), theta_max_detections = get(info, :theta_max_detections, nothing), theta_max_masked_px = θpx,
            smoothness = (; get(info, :smoothness, smoothness_record(NoSmoothness()))..., row_weight = get(info, :smoothness_row_weight, 0.0)),
            perturb = get(info, :perturb, nothing), node_pixel_counts = cnt, nodes_without_pixels = count(iszero, cnt)))
    return nothing
end
