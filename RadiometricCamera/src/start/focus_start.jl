"""The support predicate of the focus nodes, fixed at the start of a run: a sensor position 𝐩 can see light when the START lens images
it (π⁻¹ finite) and it lies in the run's region (Core's `imaged_region`). Being imaged by the model is not enough: the refined blob_kb lens
images all 25 focus nodes, 16 of them outside its image circle and dark."""
_focus_support(lens0, region) = 𝐩 -> all(isfinite, backproject(lens0, 𝐩)) && _in_region(region, 𝐩...)
"""
    focus_grid_nodes(region; n = 5) -> Vector{SVector{2, Float64}}

The candidate nodes of the focus surface: the `n × n` equally spaced grid over the bounding box of the run's region (Core's
`Rect2(region)`: the sensor when the lens lights it all, the sensor ∩ the image circle's box otherwise; corners included).
"""
focus_grid_nodes(region; n::Integer = 5) = grid_nodes(n, VgtkCore.Rect2(region))
"""Whether the position 𝐩 lies within `ρ_max` of the lens coverage's normalised radius: inside the coverage scaled by `ρ_max` about its
centre (`_scaled_coverage`, the sampling margin's: an image circle's semi-axes, whatever its orientation); the sensor's rectangle is not
scaled (its edges are the photographs' own, exact: no rim to round on)."""
_within_coverage(coverage, ρ_max) = (c = _scaled_coverage(coverage, Float64(ρ_max)); 𝐩 -> _in_region(c, 𝐩...))
"""
    focus_start_nodes(lens0, fo::FocusOptions) -> (; in_support, nodes_box, nodes, box)

The focus surface's nodes of a run, fixed at its start: the `5 × 5` candidates `focus_grid_nodes` over the bounding box `box`
of the start lens's IMAGED region (Core's `imaged_region`: the sensor ∩ its lens coverage), kept where the start lens images them inside
the region (`in_support`, `_focus_support`) AND inside the lens coverage scaled by `fo.lens_coverage_scale` (`_within_coverage`: the same `_scaled_coverage` as `[masks] lens_coverage_scale`), so
no node sits on the circle's rim by design, whatever the ellipse's orientation. The focus surface describes the optics over what is
imaged, so it never reads the start's initialization region (where the start is defined: the pose fits and the self-checks read that
one). Before (Ties, 2026-11-15): laid over the initialization region ∩ the imaged region's box (RC 0.9.0–0.11.2), the clipped
box's four side midpoints lay inside the region, on its border, seen by 8 314–35 955 pixels against 58 893–195 292 for the inner nodes,
and ran to infinite focus; over the imaged region's own box the same four lay 0.006 px OUTSIDE the rotated circle (rounding alone kept
9 nodes; an axis-aligned circle kept 13). A runaway node is flagged (`focus_runaway`).
"""
function focus_start_nodes(lens0, fo)
    region = VgtkCore.imaged_region(lens0)
    in_support = _focus_support(lens0, region)
    within = _within_coverage(VgtkCore.lens_coverage(lens0), fo.lens_coverage_scale)
    nodes_box = focus_grid_nodes(region)
    return (; in_support, nodes_box, nodes = _check_focus_nodes(filter(p -> in_support(p) && within(p), nodes_box), lens0),
            box = VgtkCore.Rect2(region))
end
"""The nodes of a focus TPS: at least 4 and not collinear (the system matrix [K P; Pᵀ 0] is singular otherwise)."""
function _check_focus_nodes(nodes, lens0)
    ok = length(nodes) >= 4 && rank([ones(length(nodes)) getindex.(nodes, 1) getindex.(nodes, 2)]) == 3
    ok || throw(ArgumentError("focus nodes: $(length(nodes)) node(s) of the grid over the imaged region are in the support of the start lens " *
                              "$(nameof(typeof(lens0))) — the thin plate spline needs at least 4, not collinear"))
    return nodes
end
"""The focus surface's nodal values t̂ᵢ at its nodes in the run's fixed support (`in_support`): a summary of the focus (`focus_m`, the
monitor's "1/t̂ mean") averages only these, so its node count does not change during the LM."""
_support_focus_values(f::Focus, in_support) = [f.t̂_i[i] for i in eachindex(f.t̂_i) if in_support(f.nodes[i])]
"""
    focus_runaway(f::Focus, in_support) -> Vector{Int}

The calibrated focus surface's check on output: the nodes of the run's support (`in_support`, fixed at the start) whose inverse depth
t̂ᵢ is non-finite or ≤ 0 — a focus distance 1/t̂ᵢ at or past infinity, which no lens focused on the target has. A node seen by few
pixels (one on the border of the imaged region) is free to run there when nothing ties it to its neighbours (`[focus.smoothness]`
weight 0, the default): the fisheye f/4 regression of RC 0.9.0 (2026-11-15). A runaway node flags the calibration
non-converged (`validity_gate`), as a fold does (`fold_termination`).
"""
focus_runaway(f::Focus, in_support) = [i for i in eachindex(f.t̂_i) if in_support(f.nodes[i]) && !(isfinite(f.t̂_i[i]) && f.t̂_i[i] > 0)]
# with the START surface `f0`: a node at t̂ ≤ 0 is flagged only when it RAN there (its start t̂ > 0) — the start floors a back-facing
# chief ray's node to t̂ = 0 (`focus_start`), which is no runaway (code review, RC 0.11.4); a non-finite one always. The start's value is
# the one AT THE SAME NODE (matched by position): a resumed state or a start record may hold another node set or order than the run's
# start (owner ruling 2026-11-16), and a node the start does not hold has no start value, so it is flagged as the check above flags it.
function focus_runaway(f::Focus, in_support, f0::Focus)
    at_start(i) = (j = findfirst(==(f.nodes[i]), f0.nodes); j === nothing ? nothing : f0.t̂_i[j])
    return [i for i in focus_runaway(f, in_support) if !isfinite(f.t̂_i[i]) || something(at_start(i), Inf) > 0]
end
"""
    focus_node_support(f::Focus, px, support_min) -> (; weight, supported)

The data support of the focus surface's nodes under the run's masked pixels `px`: the camera's one support rule (`node_support`,
the vignetting's) on the nodes' data weights `Σ_pixels b_i(𝐩)²` (`data_weight`), `b_i` the TPS's cardinal function of node `i`
(the surface with t̂ = 1 at node `i`, 0 at the others: ∂t̂(𝐩)/∂t̂ᵢ). An unsupported node is DEPENDENT: `start_calibration` builds the
TPS on the supported nodes alone, which continues the surface over it (the minimal-bending interpolant of the supported nodes is the
TPS of all of them with the dependent values continued) — never a free parameter (2026-11-15: the fisheye f/4's four nodes on the
region's border, a weak share of the data, ran to infinite focus).
"""
function focus_node_support(f::Focus, px, support_min::Real)
    sys = FocusSpline(f).sys                                    # the cardinal functions at 𝐩: Core's `nodal_weights`, ∂t̂(𝐩)/∂t̂ᵢ (one SVector)
    return node_support(data_weight(𝐩 -> VgtkCore.nodal_weights(sys, 𝐩), px, length(f.t̂_i)), support_min)
end
