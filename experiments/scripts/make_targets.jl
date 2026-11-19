# =============================================================================
# scripts/make_targets.jl — AUTHOR-SIDE: write the target record of each manufactured target of configs/targets/ into data_root()
# (src/boards.jl, save_dot_grid_board). A manufactured target's dimensions are the maker's and fixed, so its record is its design:
# rerunning rewrites it byte for byte. The records are inputs of tier rc_thorlabs (configs/data_artifacts.toml).
#
#   VGTK_DATA_ROOT=<the author's data root> julia --project=experiments experiments/scripts/make_targets.jl
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("boards.jl"))
using .Boards: save_dot_grid_board

for f in sort(filter(endswith(".toml"), readdir(projectdir("configs", "targets"); join = true)))
    design = TOML.parsefile(f)
    dir = joinpath(data_root(), design["output"])
    mkpath(dir)
    println(save_dot_grid_board(dir, design))
end
