# Record the golden outputs of the scenes in golden_scenes.jl.
#   julia --project=. test/regression/record_golden.jl test/regression/golden_migration_snd_try.json          (the reference: THIS branch's code,
#       GOLDEN_OLD_STACK=0, the default) — test/regression/golden_tests.jl compares the package against it, so a migration (a VisualGeometryToolkitCore /
#       NLLSsolver / ForwardDiff / Julia bump, a refactor) that changes any value of the pathwise route shows up
#   GOLDEN_OLD_STACK=1 julia --project=<board-space worktree> test/regression/record_golden.jl <out.json>    (the package before the second
#       migration, kept as golden_board_space.json: the record that the second migration changed no value except the intended
#       printed-edge derivative, see README.md)
# Re-record only on purpose (an intended numerical change), and say why in the commit.
const OLD_STACK = get(ENV, "GOLDEN_OLD_STACK", "0") == "1"
include(joinpath(@__DIR__, "golden_scenes.jl"))
import VisualGeometryToolkitIO as VgtkIO
out = gold_outputs()
VgtkIO.write_record(ARGS[1], Dict("schema" => GOLDEN_SCHEMA, "stack" => OLD_STACK ? "board-space (before the second migration)" : "migration-snd-try (the current package)",
                                  "julia" => string(VERSION), "outputs" => out))
println("golden outputs written to ", ARGS[1])
