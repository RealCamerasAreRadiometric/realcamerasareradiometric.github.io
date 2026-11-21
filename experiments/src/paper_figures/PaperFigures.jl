# PaperFigures.jl — the generators of the paper's figures that were first drawn outside this project, ported from T. Harder's
# figure folders (tools/<figure>.py with tools/common.py) to Julia. Each generator reads small tables and images (the figure's data
# folder) and writes the paper's files under the name its caller hands it (the `asset_dir` of the figure's config): <asset_dir>.tex,
# the folder <asset_dir>/ with one PGFPlots file (and, where it has one, an image) per panel, and <asset_dir>-caption.tex. What a
# figure shows is its config's to state: a generator holds the figure's drawing, no case, no title and no setting of its own.
#
# One file per figure beside this one; each defines its generator and registers it:
#     FIGURES["<figure>"] = figure_<name>          # figure_<name>(; data, out, settings...) -> the files it wrote
# A generator is run by a stage of its figure's driver through `generate`. common.jl is what they share. Each figure's file is
# loaded into a module of its own (<name>.jl -> PaperFigures.<Name>), so that two generators may use the same helper names; what
# common.jl defines is visible in each. The files are listed in FIGURE_FILES: a new figure's file is added there.
module PaperFigures

import ColorTypes, PNGFiles, Printf
using VisualGeometryToolkitEval.Render: HOUSE_AXIS_COMMON     # the house's axis options of every panel (src/figure_style.jl's `rc axis`)
export HOUSE_AXIS_COMMON

"The generators by the name a driver's stage runs them under (`generate`)."
const FIGURES = Dict{String, Function}()

include("common.jl")
for n in names(@__MODULE__; all = true)                     # everything defined so far is visible in the figures' modules
    s = string(n)
    (startswith(s, "#") || n in (:eval, :include, :PaperFigures)) && continue
    @eval export $n
end

"The figures' files beside this one, each with the name of the module it is loaded into, in the order they are loaded."
const FIGURE_FILES = (:BoardWarping => "board_warping.jl", :FocusSurfaces => "focus_surfaces.jl", :TextureLearned => "texture_learned.jl",
                      :Vignetting => "vignetting.jl")

for (name, file) in FIGURE_FILES                            # a file that does not load is an error: no figure is silently missing
    path = joinpath(@__DIR__, file)
    @eval module $name
        import JSON, Printf, Statistics, TOML
        using ..PaperFigures
        include($path)
    end
end

"Runs the generator of `figure` with the keyword options `opts`."
function generate(figure::AbstractString; opts...)
    haskey(FIGURES, figure) || error("no generator for the figure \"$figure\"; there are: $(join(sort(collect(keys(FIGURES))), ", "))")
    return Base.invokelatest(FIGURES[figure]; opts...)
end

end # module
