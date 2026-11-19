# =============================================================================
# src/residual_summary.jl — the one summary of a residual vector, (n, median, mean, rms, p95, max), every record of the project
# states. Included by the prelude (src/experiments.jl) and by the modules of src/figures.jl and
# src/evaluation.jl (as src/return_codes.jl).
# =============================================================================
using Statistics: median, mean, quantile

"""
    residual_summary(x) -> NamedTuple

THE summary of a vector of residuals (px) every record of the project states: `(; n, median_px, mean_px, rms_px, p95_px, max_px)`,
`rms` = √mean(x .^ 2) (the chrome tables' arithmetic; `mean(abs2, x)` can differ in the last ulp) and `p95` the 0.95 quantile;
`nothing`s (and `n = 0`) for an empty `x`.
"""
residual_summary(x) = isempty(x) ? (; n = 0, median_px = nothing, mean_px = nothing, rms_px = nothing, p95_px = nothing, max_px = nothing) :
    (; n = length(x), median_px = median(x), mean_px = mean(x), rms_px = sqrt(mean(x .^ 2)), p95_px = quantile(x, 0.95), max_px = maximum(x))
