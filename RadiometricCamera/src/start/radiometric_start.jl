"""The start `(a, u)` of the in-plane print error from the options (`warping.inplane_start`; none = the design as printed)."""
_inplane_start(opts::PipelineOptions) =
    isempty(opts.warping.inplane_start) ? zero(SVector{INPLANE_NC, Float64}) : SVector{INPLANE_NC, Float64}(opts.warping.inplane_start)

"""The nodal heights of the board warping from the options: `warping.heights_mm` (the nodal heights, mm), else flat."""
function _height_start(opts::PipelineOptions)
    h = zeros(HEIGHT_NC)
    if !isempty(opts.warping.heights_mm)
        c = VgtkCore.ustrip.(VgtkCore.m, opts.warping.heights_mm .* VgtkCore.mm)
        length(c) == HEIGHT_NC || error("warping.heights_mm: $(length(c)) values — expected the $(HEIGHT_NC) nodal heights (the 3 / 7 legacy " *
                                         "polynomial coefficients of an earlier release are retired)")
        h = c
    end
    return h
end

# ---- the vignetting field of the options --------------------------------------------------------------------------------------

"""The vignetting field of `[vignetting]`: `(make_vig, description)` — the radial profile V(𝐩) = Σ N_{i,p}(ρ(‖𝐩 − 𝐜‖)) P_i on Core's `chart` with `coefficients` coefficients of degree `degree`
about `center` 𝐜, the camera's principal point (`vignetting_center`, fixed), its radius Core's `farthest_radius` from 𝐜 of the sensor ∩ the run's lens coverage (the
image circle's radius seen from 𝐜, or the sensor's farthest corner): ρ = 1 there on the linear chart, ½ on the projective."""
function _vignetting_spec(opts::PipelineOptions, W₀, H₀, coverage, center)
    n = opts.vignetting.node_count; p = opts.vignetting.degree; chart = opts.vignetting.chart
    c = SVector{2, Float64}(center[1], center[2])
    sensor = VgtkCore.Rect(0, 0, W₀, H₀)                            # the image circle clipped by the sensor; without one the sensor (its corner)
    rmax = VgtkCore.farthest_radius(coverage isa VgtkCore.Ellipse ? sensor ∩ coverage : sensor, c)
    return () -> RadialVignetting(n, c, rmax; degree = p, chart),
           @sprintf("radial profile, %d nodes (degree %d, P_0 = P_1) about the principal point (%.1f, %.1f) px, %s chart with %s", n, p, c...,
                    chart, chart == :linear ? @sprintf("ρ = 1 at %.1f px", rmax) : @sprintf("ρ = ½ at %.1f px (pixel-uniform knots to there, one tail span beyond)", rmax))
end
