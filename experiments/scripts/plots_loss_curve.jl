# The loss-curve figure of a refinement trace (a FitResult's `metadata.trace`: refine_calibration!'s): research plotting, out of the
# package (the headless rule; RC 0.11.0 moved it from bin/, where bin/refine.jl drew it by `@eval using CairoMakie`, owner D6).
# Include after `using CairoMakie`. Its input, a run's trace, is not a record yet (a follow-up): call it in the process that refined.

"""Scale-free residual series of a trace: rel-RMS = √(2 cost / Σϕ²) per LM iteration."""
rel_rms_series(trace) = [sqrt(2 * t.cost / t.ϕ²) for t in trace]

"""rel-RMS over the LM iterations of `run_lm_grid!` (log scale)."""
function loss_curve_figure(trace)
    fig = Figure(size = (800, 340))
    ax = Axis(fig[1, 1]; xlabel = "LM iteration", ylabel = "rel-RMS residual", yscale = log10, title = "radiometric grid refinement")
    ys = rel_rms_series(trace)
    lines!(ax, 1:length(ys), ys; color = :gray60)
    scatter!(ax, 1:length(ys), ys; color = Makie.wong_colors()[1], markersize = 7)
    return fig
end
