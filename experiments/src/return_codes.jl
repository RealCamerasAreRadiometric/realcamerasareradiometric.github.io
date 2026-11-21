# =============================================================================
# src/return_codes.jl — the VisualGeometryToolkitCore ReturnCode a run record names. Included where it is read: inside module
# Figures (src/figures.jl, the tables' records: a row's convergence, never printed) and by src/panorama_example.jl (the rotation bundle's [expected] record).
# =============================================================================
import VisualGeometryToolkitCore as VgtkCore

"""
    recorded_retcode(name) -> Union{ReturnCode.T, Nothing}

The VisualGeometryToolkitCore `ReturnCode` a run record names (its `string`, as the drivers write it: `"Success"`, `"MaxIters"`, …);
`nothing` for a fit without a record (`"unfinished"`, `"unrecorded"`). Any other name is an `ArgumentError`: a record before Core's
§14 termination record holds a retcode Symbol (`"converged"`, `"max_iterations"`), and its run is rerun, not read.
"""
function recorded_retcode(name::AbstractString)
    name in ("unfinished", "unrecorded") && return nothing
    for c in instances(VgtkCore.ReturnCode.T)
        string(c) == name && return c
    end
    throw(ArgumentError("retcode \"$name\" is no VisualGeometryToolkitCore ReturnCode: a run record before Core's §14 termination " *
                        "record (a retcode Symbol); rerun it"))
end
