# =============================================================================
# src/summary_table.jl — a run's summary table, read back: VisualGeometryToolkitEval's tab-separated table (`Render.write_tsv`
# writes it, `read_tsv` reads it as text) with each cell as its value. Included by src/chrome_evaluation.jl and by the module of
# src/figures.jl (as src/residual_summary.jl is).
# =============================================================================
using VisualGeometryToolkitEval.Render: write_tsv, read_tsv

"""
    tsv_value(s) -> Union{Int, Float64, String}

A cell of a tab-separated table as its value: an integer, a number (`NaN` included), or the text as it is; an empty cell `""`.
"""
tsv_value(s::AbstractString) = isempty(s) ? "" : something(tryparse(Int, s), tryparse(Float64, s), String(s))

"""
    read_summary(path) -> Vector{Dict{String, Any}}

The rows of a summary table (`write_tsv`'s file, `read_tsv`) in file order, column name => [`tsv_value`](@ref) of its cell.
"""
read_summary(path::AbstractString) = [Dict{String, Any}(k => tsv_value(v) for (k, v) in r) for r in read_tsv(path)]

"""A row's cell as `write_tsv` writes it: `nothing` (no value) as an empty cell, anything else as it is (`string`)."""
tsv_cell(x) = x === nothing ? "" : x
