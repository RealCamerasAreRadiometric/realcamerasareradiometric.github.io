# the median and 90 % quantile of a diagnostic, NaN for none (a session without detections)
_med(x) = isempty(x) ? NaN : median(x)
_q90(x) = isempty(x) ? NaN : quantile(x, 0.9)

# ---- lengths in mm for print and report ---------------------------------------------------------------------------------------
# a length the pipeline holds as a plain number in metres, printed or reported in mm / µm: FlexUnits' `ustrip` (units.md), no numeric
# factor; angles stay radians and are converted where they are displayed (`rad2deg`)
_mm(x) = VgtkCore.ustrip(VgtkCore.mm, x * VgtkCore.m)
_μm(x) = VgtkCore.ustrip(VgtkCore.μm, x * VgtkCore.m)

# ---- refine --------------------------------------------------------------------------------------------------------------------
"""V averaged on rings of 500 / 750 / 900 px about the frame centre, relative to the centre value (the report line of the sweeps)."""
_Vprof(vig) = (c = VgtkCore.center(vig); join([@sprintf("%.3f", mean(vig(c + SVector(r * cos(φ), r * sin(φ))) for φ in 0:π/8:2π-1e-9) / vig(c))
                            for r in (500.0, 750.0, 900.0)], " / "))
