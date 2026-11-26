# ---- the random draws -----------------------------------------------------------------------------------------------------------
"""
    _subsample(rng, n, m) -> Vector{Int}

`m` of the indices `1:n`, sorted, drawn uniformly without replacement: the `m` smallest of `n` uniform keys `rand(rng, n)`. Every random
draw of the pipeline uses a `StableRNG` and only `rand(rng, …)` of Float64 — a stream StableRNGs keeps fixed across Julia versions.
`randperm` is not used: its algorithm changed in Julia 1.13 (a different permutation from the same generator state), which moved
the LM's pixel subsample and with it every refined result of the pipeline checks.
"""
_subsample(rng, n::Integer, m::Integer) = _subsample(rng, n, m, nothing)
"""
    _subsample(rng, n, m, w) -> Vector{Int}

`_subsample`'s weighted generalization, THE draw: `m` of `1:n` without replacement with probability ∝ `w` at every step (successive
sampling, Efraimidis–Spirakis), from the same stream — the `m` smallest exponential keys −log(1 − U_i)/w_i of the same `rand(rng, n)`:
an item's key is the first arrival of a Poisson clock of rate w_i (−log(1 − U) is Exp(1)). The key is increasing in U_i, so equal
weights order the items exactly as `_subsample`'s U_i do: `w = nothing` (or any constant `w`) draws `_subsample`'s very indices.
"""
_subsample(rng, n::Integer, m::Integer, w) = sort!(partialsortperm(_draw_keys(rand(rng, n), w), 1:m))
# THE draw's keys of the uniforms `U` (`_subsample`, `draw_samples`): `U` itself for the uniform draw (`w = nothing`), otherwise the
# first arrivals −log(1 − U_i)/w_i of Poisson clocks of rate w_i
_draw_keys(U, ::Nothing) = U
_draw_keys(U, w) = (length(w) == length(U) || throw(DimensionMismatch("_subsample: $(length(w)) weights for $(length(U)) items")); -log1p.(-U) ./ w)

"""
    _edge_select(mask, img, n; channels = axes(img, 3), diagonal = false) -> BitMatrix

The `n` masked pixels with the strongest squared gradient |∇ϕ|² of the photo (central differences from the four neighbours, summed
over the measured `channels` of `img`, H × W × C or one H × W plane; an uncertain image's values): the rule of 2026-10-31 (an earlier release's `masks.edge_fraction`,
restored verbatim for the sampling-strategy experiment, `draw_samples`' `edge_fraction`). Border pixels have no
gradient and rank last. Deterministic, no random draw.
"""
function _edge_select(mask::AbstractMatrix{Bool}, img::AbstractArray, n::Integer; channels = axes(img, 3), diagonal::Bool = false)
    H, W = size(mask); px = masked_pixels(mask)
    x(i, j, c) = VgtkCore.value_of(img[i, j, c])                     # an uncertain image's values, read in place (Core's `value_of`)
    g = map(px) do (u, v)
        (2 <= u <= W - 1 && 2 <= v <= H - 1) || return -1.0
        # `diagonal`: a RAW photograph's photosites (`observe`), whose green photosites' four diagonal neighbours are green (their
        # edge-neighbours red or blue): the squared gradient from the diagonal differences (spacing √2 px)
        diagonal ? sum(((x(v + 1, u + 1, c) - x(v - 1, u - 1, c))^2 + (x(v + 1, u - 1, c) - x(v - 1, u + 1, c))^2) / 8 for c in channels) :
                   sum(((x(v, u + 1, c) - x(v, u - 1, c))^2 + (x(v + 1, u, c) - x(v - 1, u, c))^2) / 4 for c in channels)
    end
    n = min(n, length(px)); out = falses(H, W)
    for i in (n == 0 ? Int[] : partialsortperm(g, 1:n; rev = true)); u, v = px[i]; out[v, u] = true; end
    return out
end

"""
    draw_samples(rng, [valid,] candidates, img, n; gradient_bias = 0, edge_detector, edge_fraction = 0, addon_samples = 0,
                 addon_gradient_bias = 0, channels = axes(img, 3), diagonal = false, device) -> BitMatrix

THE draw of a view's samples (owner, 2026-11-15): `n` of its USABLE observations — the board-valid pixels of the sampling region that
the photograph's observation model keeps (`observe`: a RAW photograph's green photosites, no defect, uncensored; a decoded image's
pixels) — so every camera gets the same count whatever its image circle or channel, drawn without replacement from the `StableRNG`
stream `rng` (in view order). The usable observations are the `candidates` (the sampling region's pixels the observation model
keeps) valid under the mask test `valid` (`ids -> Bool` per id: the view's `BoardValidity`; without it every candidate is
usable). Every candidate gets its uniform key U (`rand(rng, #candidates)`, column-major: the stream consumed once per view, whatever the
mode or the count) and the draw reads the keys of the usable ones. Its modes, one draw (owner: no second sampler):

- the weighted draw (the default): each usable observation ∝ w = u^α (owner design, 2026-11-15), u the ECDF rank of the photograph's
  edge strength over the usable observations (`edge_detector` on `img`'s measured `channels`, `edge_strength` / `edge_rank`, on
  `device`), α = `gradient_bias` (> 0 favours edges, < 0 flat regions): `_subsample`'s weighted generalization (exponential keys of
  the same uniforms). α = 0 is the uniform draw itself (`_subsample`, no detector run), bit for bit;
- `edge_fraction = q ∈ (0, 1)` (with α = 0): the rule of 2026-10-31 (an earlier release's, verbatim): a uniform pool of ⌈n / q⌉,
  then its `n` strongest |∇ϕ|² (`_edge_select`; `diagonal` for a RAW photograph's photosites);
- `addon_samples = M ≥ 1`: the draw above (n ∝ u^α) PLUS M more of the other usable observations ∝ u^β, β = `addon_gradient_bias`,
  from the SAME uniforms: the keys are the first arrivals of Poisson clocks (E = −log(1 − U) ~ Exp(1), at rate w); given the n
  that arrived first (by τ), every other clock's residual E − w τ is a fresh Exp(1) (memoryless), so the M are the others' own
  weighted draw — and the n are exactly the n of the draw without the add-on (the same stream: no extra random number, so every
  view's n samples are those of the draw without it, bit for bit).

The mask test: where no weight reads the edge ranks (α = β = 0: the uniform draw, the rule
of 2026-10-31's pool, the uniform add-on) the draw is the m smallest keys among the usable observations (m = n + M, or the pool),
so `valid` tests the candidates in key order, in batches, until m are valid (`_smallest_valid`) — the same m as the test of every
candidate, at a few times m pixels tested instead of the whole region. A weight that reads the ranks (α ≠ 0 or β ≠ 0) needs every
usable observation: every candidate is tested.

A view with fewer usable observations than its count uses them all; never padded (its shortfall is the masks' record's).
"""
draw_samples(rng, usable::AbstractMatrix{Bool}, img::AbstractArray, n::Integer; kw...) = draw_samples(rng, _every_candidate, usable, img, n; kw...)
function draw_samples(rng, valid, candidates::AbstractMatrix{Bool}, img::AbstractArray, n::Integer; gradient_bias::Real = 0.0,
                      edge_detector::AbstractEdgeDetector, edge_fraction::Real = 0.0, addon_samples::Integer = 0, addon_gradient_bias::Real = 0.0,
                      channels = axes(img, 3), diagonal::Bool = false, device = default_backend())
    check_draw(n, gradient_bias, edge_fraction, addon_samples, addon_gradient_bias)
    cand = findall(vec(candidates))                                   # the candidates' linear (column-major) indices
    Uc = rand(rng, length(cand))                                      # the view's uniforms: every candidate's key, the stream consumed once
    α = Float64(gradient_bias); β = Float64(addon_gradient_bias)
    ranked = !(iszero(α) && iszero(β))                                # a weight reads the edge ranks: every usable observation
    m = edge_fraction > 0 ? ceil(Int, n / edge_fraction) : n + addon_samples   # the smallest keys the draw reads, where no rank is read
    keep = ranked ? findall(valid(cand)) : _smallest_valid(valid, cand, Uc, m)  # the usable candidates (all, or the m smallest keys')
    idx = cand[keep]; U = Uc[keep]; N = length(idx)
    out = falses(size(candidates))
    if edge_fraction > 0                                              # the rule of 2026-10-31 (verbatim): its uniform pool, the
        out[idx] .= true                                              #   ⌈n / q⌉ smallest keys among the usable (all when fewer; α = 0)
        return _edge_select(out, img, n; channels, diagonal)
    end
    # the uniform draw: the n smallest keys (the uniform add-on the next M: the residuals E − τ order the rest as their U); a
    # weighted one with fewer than n (+ M) usable: all of them (the add-on takes the rest)
    (!ranked || n + addon_samples >= N) && (out[idx] .= true; return out)
    u = ranked ? _usable_ranks(idx, size(candidates), img, channels, edge_detector, device) : Float64[]   # the edge ranks, only where a weight reads them
    w = iszero(α) ? nothing : u .^ α                                  # the draw's weights (none: the uniform draw)
    key = _draw_keys(U, w)                                            # `_subsample(rng, N, n, w)`'s keys
    sel = partialsortperm(key, 1:n)
    out[idx[sel]] .= true
    addon_samples == 0 && return out
    # the add-on: given the n drawn (the n first arrivals, by τ), each other clock's residual E − w τ is a fresh Exp(1) (memoryless),
    # so the M are the others' own weighted draw on the same stream
    rest = findall(!, view(out, idx))
    τ = w === nothing ? -log1p(-maximum(view(U, sel))) : maximum(view(key, sel))
    E2 = -log1p.(-view(U, rest)) .- (w === nothing ? τ : view(w, rest) .* τ)
    key2 = iszero(β) ? E2 : E2 ./ view(u, rest) .^ β
    out[idx[rest[partialsortperm(key2, 1:addon_samples)]]] .= true
    return out
end
# No mask test: every candidate is usable (`draw_samples` on the usable observations themselves)
_every_candidate(ids) = trues(length(ids))
# The ECDF ranks of the edge strength of the usable observations `idx` among themselves (`edge_rank` of `edge_strength` read over
# them alone), in `idx`'s order
function _usable_ranks(idx, dims, img, channels, detector, device)
    support = falses(dims); support[idx] .= true
    return edge_rank(edge_strength(detector, edge_plane(img, channels), support; device), idx)::Vector{Float64}
end

"""
    _smallest_valid(valid, ids, U, m) -> Vector{Int}

The positions in `ids` of the `m` smallest keys `U` among the ids valid under the mask test `valid` (all the valid ones when fewer are), in
increasing key order: the candidates tested in key order, in batches. The first batch is the `m` smallest keys (at least `m`
candidates are needed); each further one the count the observed valid share predicts for the rest, `(m − found) · tested / found`
(twice the tested count while none was valid), never fewer than `m − found` — derived from `m` and the view's own valid share, no
setting. The batching changes the number of pixels tested, never the result: the result is the first `m` valid ids in key order.
"""
function _smallest_valid(valid, ids::AbstractVector, U::AbstractVector{Float64}, m::Integer)
    L = length(ids); order = collect(1:L); found = Int[]; tested = 0
    while length(found) < m && tested < L
        need = m - length(found)
        b = isempty(found) ? (tested == 0 ? need : tested) : ceil(Int, need * tested / length(found))
        hi = min(L, tested + max(b, need))
        partialsortperm!(order, U, 1:hi; initialized = true)          # the keys' order up to `hi` (its first `tested` as before: distinct keys)
        batch = view(order, tested+1:hi)
        ok = valid(view(ids, batch))
        for (j, isvalid) in zip(batch, ok)
            isvalid && length(found) < m && push!(found, j)
        end
        tested = hi
    end
    return found
end

"""
    check_draw(n, gradient_bias, edge_fraction, addon_samples, addon_gradient_bias)

The draw's settings (`[masks]`, `draw_samples`) checked together: `n ≥ 1`; α finite; `edge_fraction` in [0, 1), and > 0 only with
α = 0 (the rule of 2026-10-31 draws its pool uniformly) and no add-on; `addon_samples ≥ 0`; a stated `addon_gradient_bias` (≠ 0)
only with an add-on to bias (an inapplicable setting is an error, never ignored).
"""
function check_draw(n, gradient_bias, edge_fraction, addon_samples, addon_gradient_bias)
    n >= 1 || throw(ArgumentError("masks.samples_per_view = $n: at least one sample per view"))
    isfinite(gradient_bias) || throw(ArgumentError("masks.gradient_bias = $gradient_bias: a finite exponent (0 = uniform samples)"))
    isfinite(addon_gradient_bias) || throw(ArgumentError("masks.addon_gradient_bias = $addon_gradient_bias: a finite exponent"))
    0 <= edge_fraction < 1 || throw(ArgumentError("masks.edge_fraction = $edge_fraction: a share in [0, 1) (0 = the weighted draw)"))
    addon_samples >= 0 || throw(ArgumentError("masks.addon_samples = $addon_samples: a count ≥ 0 (0 = no add-on)"))
    edge_fraction > 0 && !iszero(gradient_bias) &&
        throw(ArgumentError("masks.edge_fraction = $edge_fraction with masks.gradient_bias = $gradient_bias: the rule of 2026-10-31 draws its pool uniformly — state gradient_bias = 0.0"))
    edge_fraction > 0 && addon_samples > 0 && throw(ArgumentError("masks.edge_fraction and masks.addon_samples: two modes of the draw, one at a time"))
    addon_samples == 0 && !iszero(addon_gradient_bias) &&
        throw(ArgumentError("masks.addon_gradient_bias = $addon_gradient_bias without masks.addon_samples: no add-on to bias (inapplicable)"))
    return nothing
end

"""
    sample_record(n, masks) -> (; per_view, drawn, shortfall)

The masks' record of a run's draw (`draw_samples`): the samples per view `n` (`samples_per_view`, plus `addon_samples` with an
add-on), the samples each view's final mask holds, and each view's shortfall `max(n − drawn, 0)` (a view with fewer usable
observations than `n` uses them all; never padded).
"""
sample_record(n::Integer, masks) = (; per_view = Int(n), drawn = [count(m) for m in masks], shortfall = [max(n - count(m), 0) for m in masks])

"""
    _sampling_region(camera, scale) -> region

Where the photometric pixels are sampled (`masks.lens_coverage_scale`): Core's imaged region of `camera` with its lens coverage — an
image circle (a `Circle` or an `Ellipse`) — scaled about its centre by `scale` (both semi-axes); a rectangle coverage (the sensor's:
no image circle, no rim) is not scaled. Sampling only: the camera, its lattice's present nodes and the vignetting chart keep the
measured coverage. `scale = 1` is the imaged region itself.
"""
function _sampling_region(camera, scale::Real)
    0 < scale <= 1 || throw(ArgumentError("masks.lens_coverage_scale = $scale: a factor in (0, 1]"))
    isone(scale) && return VgtkCore.imaged_region(camera)
    return VgtkCore.imaged_region(VgtkCore.with_lens_coverage(camera, _scaled_coverage(VgtkCore.lens_coverage(camera), Float64(scale))))
end
_scaled_coverage(c::Union{VgtkCore.Ellipse, VgtkCore.Circle}, s::Float64) = VgtkCore.dilate(c, s)   # Core's: the semi-axes (radius) scaled about the centre
_scaled_coverage(r::VgtkCore.Rect2, s::Float64) = r                     # the sensor's rectangle: no rim to cut
