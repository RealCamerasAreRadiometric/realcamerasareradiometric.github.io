# =============================================================================
# src/flow_colour.jl — the one colour coding of a residual vector, direction + length (optical-flow style, flowcolor.py's
# `flow_colors`): every colour-wheel figure of the project draws its vectors with it. Included by the module of src/figures.jl
# and by the paper figures' common file (src/paper_figures/common.jl), as src/residual_summary.jl is shared.
# =============================================================================

# matplotlib.colors.hsv_to_rgb, element for element (the same float operations, so the colours are the Python figures' colours).
function _hsv_to_rgb(h::Float64, s::Float64, v::Float64)
    i = floor(Int, h * 6.0); f = h * 6.0 - i
    p = v * (1.0 - s); q = v * (1.0 - s * f); t = v * (1.0 - s * (1.0 - f))
    k = mod(i, 6)
    k == 0 && return (v, t, p)
    k == 1 && return (q, v, p)
    k == 2 && return (p, v, t)
    k == 3 && return (p, q, v)
    k == 4 && return (t, p, v)
    return (v, p, q)
end

"""
    flow_rgb(dx, dy, vmax) -> (r, g, b)

The direction + length colour coding of a residual vector (optical-flow style, flowcolor.py's `flow_colors`): hue = the direction
of the vector as drawn in the image (x right, y down), saturation = length / `vmax` (clipped at 1: short = white, ≥ `vmax` = fully
saturated), value = 1. Float64 components in [0, 1].
"""
function flow_rgb(dx::Real, dy::Real, vmax::Real)
    h = mod(atan(Float64(dy), Float64(dx)) / (2π), 1.0)
    s = clamp(hypot(Float64(dx), Float64(dy)) / Float64(vmax), 0.0, 1.0)
    return _hsv_to_rgb(h, s, 1.0)
end
