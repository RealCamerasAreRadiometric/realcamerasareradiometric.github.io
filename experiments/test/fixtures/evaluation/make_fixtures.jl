# Writes the regression fixture of the ported evaluation (src/evaluation.jl; test/drivers_tests.jl "the ported
# evaluation against the stored runs"): a few kilobytes cut from the STORED runs of record, with the stored results the functions
# must reproduce. Run once, from the repository root, with data_root() bound to the stored runs (experiments/, their Git LFS objects
# present):
#
#   VGTK_DATA_ROOT=$PWD/experiments julia --project=experiments experiments/test/fixtures/evaluation/make_fixtures.jl
#
# The lenses and poses are VisualGeometryToolkitIO records (the tracked data's one-time migration):
#   gt/poses.json           two views of the Siemens ground truth's report (detector_bias/data/fisheye_full/gt/poses.json), every
#                           50th corner; gt/camera.json and gt/trajectory.json its records (the true lens, the true poses of the two)
#   truth_lens.json         the render lens's camera record (data/target/lenses/fisheye_equisolid_932px.json)
#   kb_lens.json, kb_poses.json      db_kb_siemens's refined lens and poses at its last checkpoint (refine_kb_bias_truthrad_pathwise, it 11)
#   grid_lens.json, grid_poses.json  db_coarse_siemens's 35 × 25 grid and poses at its last checkpoint (refine_bias_truthrad_35x25_pathwise, it 9)
#   expected_reprojection.json       the stored iter_eval/learned_it<N>.json vectors at the fixture's corners
#   vignetting.json         the learned V of the fine radial refinement (the radial profile's parameters, COLMAP pixels): written
#                           once from the stored state of an earlier package version (state_refined_noncentral_pathwise_it3.jls,
#                           which the current package does not read), not by this script
#   expected_vignetting.json  the stored export of that state (vignetting/results/logs/maps/V_refined_noncentral_pathwise.csv, step
#                           8 px on 4256 × 2848), every 37th row and column
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
import VisualGeometryToolkitIO as VgtkIO

const OUT = @__DIR__
const GA = joinpath(data_root(), "geometric_accuracy")
# read strictly: allownan would read every number as a Float64, so the ids (and the trajectory records' integers) would be written
# back as floats; a stored run holding a NaN is refused, naming its field
readj(p) = VgtkIO.read_record(p)
# IO 0.10's envelope: a record states its schema (the source's, else this fixture set's), never the writer's generator, units, conventions
writej(name, x) = VgtkIO.write_record(joinpath(OUT, name),
    merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.evaluation_fixture),
          Dict{String, Any}(String(k) => v for (k, v) in pairs(x) if !(String(k) in ("generator", "units", "conventions")))); allownan = true)

# ---- detector bias ------------------------------------------------------------------------------------------------------------------
GT = joinpath(GA, "detector_bias/data/fisheye_full/gt")
gt = readj(joinpath(GT, "poses.json"))
views = [Dict{String,Any}(v..., "corners" => v["corners"][1:50:end]) for v in gt["views"][1:2]]
ids = Set(Int(c["id"]) for v in views for c in v["corners"])
names = Set(String(v["name"]) for v in views)
mkpath(joinpath(OUT, "gt"))
writej("gt/poses.json", Dict("pixel_convention" => gt["pixel_convention"],
                             "board" => Dict("corners" => [c for c in gt["board"]["corners"] if Int(c["id"]) in ids]),
                             "views" => [Dict(k => v[k] for k in ("name", "corners")) for v in views]))
cp(joinpath(GT, "camera.json"), joinpath(OUT, "gt", "camera.json"); force = true)
# a trajectory record cut to the fixture's views (the frames' entries as stored)
cut_trajectory(src, name) = (T = readj(src); T["frames"] = [f for f in T["frames"] if String(f["image"]) in names]; writej(name, T))
cut_trajectory(joinpath(GT, "trajectory.json"), "gt/trajectory.json")
cp(joinpath(GA, "data/target/lenses/fisheye_equisolid_932px.json"), joinpath(OUT, "truth_lens.json"); force = true)
expected = Dict{String,Any}()
for (tag, run, kind, it) in (("kb", "refine_kb_bias_truthrad_pathwise", "lens", 11), ("grid", "refine_bias_truthrad_35x25_pathwise", "grid", 9))
    dir = joinpath(GA, "results/logs/detector_bias/fisheye_full", run)
    cp(joinpath(dir, "$(kind)_it$it.json"), joinpath(OUT, "$(tag)_lens.json"); force = true)
    cut_trajectory(joinpath(dir, "poses_it$it.json"), "$(tag)_poses.json")
    S = readj(joinpath(dir, "iter_eval", "learned_it$it.json"))
    keep = [i for i in eachindex(S["x"]) if String(S["view"][i]) in names && Int(S["id"][i]) in ids]
    expected[tag] = Dict("source" => "results/logs/detector_bias/fisheye_full/$run/iter_eval/learned_it$it.json",
                         (k => S[k][keep] for k in ("view", "id", "x", "y", "dx", "dy"))...)
end
writej("expected_reprojection.json", expected)

println("fixture written to ", OUT, ": ", join(sort(readdir(OUT)), ", "))
