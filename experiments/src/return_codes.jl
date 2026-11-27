# =============================================================================
# src/return_codes.jl — the VisualGeometryToolkitCore ReturnCode a run record names. Included where it is read: inside module
# Figures (src/figures.jl, the tables' records: a row's convergence, never printed).
# =============================================================================
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO

"""
    recorded_retcode(name) -> Union{ReturnCode.T, Nothing}

The VisualGeometryToolkitCore `ReturnCode` a run record names (its `string`, as the drivers write it: `"Success"`, `"MaxIters"`, …);
`nothing` for a fit without a record (`"unfinished"`, `"unrecorded"`). A record written before Core 0.14's termination record holds a
legacy spelling (`"converged"`, `"max_iterations"`): it reads through VisualGeometryToolkitIO's one reader of those,
`termination_from_legacy`. Any other name is an `ArgumentError` (that reader's, listing the spellings).
"""
function recorded_retcode(name::AbstractString)
    name in ("unfinished", "unrecorded") && return nothing
    for c in instances(VgtkCore.ReturnCode.T)
        string(c) == name && return c
    end
    return first(VgtkIO.termination_from_legacy(name))
end
