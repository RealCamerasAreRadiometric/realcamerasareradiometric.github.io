# =============================================================================
# The project icon, drawn from measured camera parameters when the site is built
# =============================================================================
#
# make.jl calls `write_icons(dir)` before `makedocs`. The icon is the measured
# fisheye IMAGE CIRCLE with its VIGNETTING FALL-OFF: a square tile holding a
# disc whose brightness falls off radially towards a soft rim, with the faint
# lattice of the calibration spline grid over it — a real lens's lit circle.
# Every file is regenerated on every build into src/assets/landing/ and is
# gitignored: icon.svg and icon_dark.svg (the hero, light and dark),
# favicon.png (32 px) and avatar.png (500×500, the GitHub organization's
# avatar, uploaded by hand).

using CairoMakie

# --- Measured parameters -----------------------------------------------------
#
# Provenance: Sony FX3 with the Sigma 8 mm fisheye at f/4, the RadiometricCamera
# vignetting refinement of 2026-11-07. The image circle is the domain of the
# learned grid: an ellipse with centre `CIRCLE_CENTRE` and semi-axes
# `CIRCLE_AXES` (px), its first axis at angle `CIRCLE_THETA` (rad) from the
# image x axis, on a `SENSOR` (width × height) px sensor. The grid is the
# calibration spline grid of `GRID` (columns × rows) nodes spanning the sensor.
const SENSOR = (4256, 2848)
const CIRCLE_CENTRE = (2130.975, 1429.296)
const CIRCLE_AXES = (1091.363, 1083.617)
const CIRCLE_THETA = 1.5097
const GRID = (35, 25)

# The radial fall-off, measured to θ = 68.3° and illustrative beyond it.
#
# `V_RINGS` is the measured relative vignetting V(θ) of the same lens (the
# file does not record the aperture; the rotation fit of 2026-10-29; green
# channel, all cells, normalized to the 0–10° ring), against the equisolid field
# angle θ about the image-circle centre, on the rings `V_EDGES` (degrees). Read
# with plain JSON from
#   /home/jbpritts/data/rc-cvpr2027/vignetting/results/logs/fit/rotation_vignetting.json
#   (fits.all_G.V, ring_edges), the rooftop rotation sequence DSC00456–DSC00465
# and copied here so the site builds without the private data. The samples
# reach θ = 68.3°; the rings beyond are unmeasured (NaN in the file).
const V_EDGES = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0)
const V_RINGS = (1.0, 0.9985908794070764, 0.9961322607755136, 0.9943575628294486, 0.9910267457898146,
    0.9729401521067639, 0.9356826882323029, 0.8871858815529197, 0.8349593736339558,
    0.7805869599572546, 0.7245175370201514, 0.6665913779356815, 0.6041448458995331)
# The fisheye's field of view (180°: the rim of the image circle is θ = 90°).
const THETA_RIM = 90.0
# ILLUSTRATIVE beyond the last measured ring: the measured slope continued to
# the rim, then a soft edge over the outer `RIM_SOFT` of the radius.
const RIM_SOFT = 0.05

# V(θ): piecewise linear through the ring centres (flat inside the first).
function vignetting(θ)
    cs = [(V_EDGES[i] + V_EDGES[i + 1]) / 2 for i in eachindex(V_RINGS)]
    vs = collect(V_RINGS)
    θ <= cs[1] && return vs[1]
    i = searchsortedlast(cs, θ)
    i < length(cs) && return vs[i] + (vs[i + 1] - vs[i]) * (θ - cs[i]) / (cs[i + 1] - cs[i])
    slope = (vs[end] - vs[end - 1]) / (cs[end] - cs[end - 1])   # illustrative extrapolation
    return max(vs[end] + slope * (θ - cs[end]), 0.0)
end

# Brightness at the normalized radius ρ = r / R of the image circle: the
# equisolid projection r ∝ sin(θ/2) gives θ, and the soft edge closes the rim.
function brightness(ρ)
    ρ >= 1 && return 0.0
    θ = 2 * asind(ρ * sind(THETA_RIM / 2))
    edge = clamp((1 - ρ) / RIM_SOFT, 0, 1)
    return vignetting(θ) * edge^2 * (3 - 2edge)
end

# --- Drawing -----------------------------------------------------------------

# The square crop of the sensor the tile shows (px): centred on the circle.
const HALF = 1.12 * maximum(CIRCLE_AXES)

# Normalized elliptical radius of the sensor point (x, y).
function _rho(x, y)
    dx, dy = x - CIRCLE_CENTRE[1], y - CIRCLE_CENTRE[2]
    c, s = cos(CIRCLE_THETA), sin(CIRCLE_THETA)
    u, v = c * dx + s * dy, -s * dx + c * dy
    return sqrt((u / CIRCLE_AXES[1])^2 + (v / CIRCLE_AXES[2])^2)
end

# The disc as an n×n image of brightness, supersampled `ss`×`ss` per pixel.
function _disc(n; ss = 4)
    x0, y0 = CIRCLE_CENTRE .- HALF
    step = 2HALF / n
    img = zeros(Float64, n, n)
    for j in 1:n, i in 1:n, b in 1:ss, a in 1:ss
        x = x0 + (i - 1 + (a - 0.5) / ss) * step
        y = y0 + (j - 1 + (b - 0.5) / ss) * step
        img[i, j] += brightness(_rho(x, y)) / ss^2
    end
    return img
end

# The chord of the sensor line p + t·d inside the image circle, or nothing.
function _chord(p, d)
    c, s = cos(CIRCLE_THETA), sin(CIRCLE_THETA)
    rot(v) = (c * v[1] + s * v[2], -s * v[1] + c * v[2])
    q, e = rot(p .- CIRCLE_CENTRE), rot(d)
    A, B = CIRCLE_AXES
    qa = (e[1] / A)^2 + (e[2] / B)^2
    qb = 2 * (q[1] * e[1] / A^2 + q[2] * e[2] / B^2)
    qc = (q[1] / A)^2 + (q[2] / B)^2 - 1
    disc = qb^2 - 4qa * qc
    disc <= 0 && return nothing
    t1, t2 = (-qb - sqrt(disc)) / 2qa, (-qb + sqrt(disc)) / 2qa
    return (p .+ t1 .* d, p .+ t2 .* d)
end

# The grid lines, clipped to the image circle: a vector of segments.
function _lattice()
    (W, H), (gx, gy) = SENSOR, GRID
    segs = Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    for i in 0:(gx - 1)
        ch = _chord((i * W / (gx - 1), 0.0), (0.0, 1.0))
        ch === nothing || push!(segs, ch)
    end
    for j in 0:(gy - 1)
        ch = _chord((0.0, j * H / (gy - 1)), (1.0, 0.0))
        ch === nothing || push!(segs, ch)
    end
    return segs
end

# A rounded square of side 1 at the origin, corner radius r, as a polygon.
function _rounded_square(r; k = 16)
    pts = Point2f[]
    for (cx, cy, a0) in ((1 - r, 1 - r, 0), (r, 1 - r, 90), (r, r, 180), (1 - r, r, 270))
        for t in range(a0, a0 + 90; length = k)
            push!(pts, Point2f(cx + r * cosd(t), cy + r * sind(t)))
        end
    end
    return pts
end

# The palette: the tile is the unlit sensor beyond the image circle, the disc
# its warm lit circle, the lattice a faint dark grid on the light.
const LIGHT = (tile = RGBf(0.08, 0.09, 0.11), outline = nothing)
const DARK = (tile = RGBf(0.05, 0.055, 0.07), outline = RGBf(0.30, 0.33, 0.40))
const WARM = RGBf(1.0, 0.965, 0.88)

# The icon at `px`×`px`. `rounded` gives the hero's rounded tile on a
# transparent page (a full square otherwise); `lattice` draws the grid, whose
# 1-px lines are left out of the 32-px favicon, where they would only grey the disc.
function icon_figure(px; theme = LIGHT, rounded = true, lattice = true, n = px)
    fig = Figure(; size = (px, px), figure_padding = 0, backgroundcolor = rounded ? :transparent : theme.tile)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = :transparent)
    hidedecorations!(ax); hidespines!(ax)
    limits!(ax, 0, 1, 0, 1)
    pad = rounded ? 0.02 : 0.0
    if rounded
        sq = [Point2f(pad + (1 - 2pad) * p[1], pad + (1 - 2pad) * p[2]) for p in _rounded_square(0.14)]
        poly!(ax, sq; color = theme.tile, strokecolor = something(theme.outline, theme.tile),
            strokewidth = theme.outline === nothing ? 0 : max(1, px / 256))
    end
    # The disc, in tile coordinates (y up: the sensor's rows are flipped).
    v = _disc(n)
    rgba = [RGBAf(WARM.r * b, WARM.g * b, WARM.b * b, min(1, 4b)) for b in reverse(v; dims = 2)]
    image!(ax, pad .. (1 - pad), pad .. (1 - pad), rgba; interpolate = true)
    if lattice
        x0, y0 = CIRCLE_CENTRE .- HALF
        tile(p) = Point2f(pad + (1 - 2pad) * (p[1] - x0) / 2HALF, 1 - pad - (1 - 2pad) * (p[2] - y0) / 2HALF)
        pts = Point2f[]
        for (a, b) in _lattice()
            push!(pts, tile(a), tile(b))
        end
        linesegments!(ax, pts; color = RGBAf(0.10, 0.11, 0.16, 0.22), linewidth = max(0.6, px / 420))
    end
    return fig
end

function write_icons(dir)
    mkpath(dir)
    save(joinpath(dir, "icon.svg"), icon_figure(320; theme = LIGHT, n = 640))
    save(joinpath(dir, "icon_dark.svg"), icon_figure(320; theme = DARK, n = 640))
    save(joinpath(dir, "favicon.png"), icon_figure(32; theme = LIGHT, rounded = false, lattice = false); px_per_unit = 1)
    save(joinpath(dir, "avatar.png"), icon_figure(500; theme = LIGHT, rounded = false); px_per_unit = 1)
    return dir
end
