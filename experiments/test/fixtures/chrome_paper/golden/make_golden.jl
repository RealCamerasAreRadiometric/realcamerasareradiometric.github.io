# The golden of the chrome paper tables: the driver of record (scripts/chrome_evaluation_paper.jl's tables stage: chrome_tables_rows,
# write_chrome_tables, chrome_table_macros of src/chrome_evaluation_paper.jl) writes its tables and number macros from the fixture
# products (test/fixtures/chrome_paper), the refinements' retcodes the fixture's — exactly the test's calls. Re-recorded 2026-11-15 (owner)
# after the fisheye 27.10 Siemens set's exclusion (configs/excluded.toml [excluded.fisheye2710_siemens]: superseded by the fisheye
# 30 cm recapture; Schöps's folded camera): the paper's tables drop its rows. Before, the golden was the pre-move driver's
# (scripts/geometric_figures.jl, dgx01 job 17793), which this driver matched byte for byte.
#
#   julia --project=experiments experiments/test/fixtures/chrome_paper/golden/make_golden.jl      (from the repository root)
using DrWatson
using TOML: TOML
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("chrome_evaluation.jl"))
include(srcdir("figures.jl"))
include(srcdir("chrome_evaluation_paper.jl"))
import VisualGeometryToolkitEval.Reproduce as Reproduce
F = projectdir("test", "fixtures", "chrome_paper"); G = joinpath(F, "golden")
path = projectdir("configs", "experiments", "chrome_evaluation_paper.toml")
P = load_chrome_paper_config(path)
# the fixture products predate the held-out headline (variants motion and free, no paired.tsv): the tables at the variant they were
# recorded with, no in-sample column, no interval (RC 0.11.4; chrome_evaluation_paper_tests.jl does the same)
foreach(t -> (t["variant"] = "motion"; delete!(t, "in_sample_variant"); t["confidence"] = "off"), values(P["tables"]))
ret = TOML.parsefile(joinpath(F, "refinement_retcodes.toml"))
products = (; summary = k -> joinpath(F, k, "summary.tsv"), vectors = (k, l, v) -> error("no vectors in the fixture"), retcode = (k, l) -> ret[k][l],
            provenance = k -> (; run_key = "fixture-key-$k", commit = String(TOML.parsefile(joinpath(F, k, "_provenance.toml"))["gitcommit"])))
rows = chrome_tables_rows(P, products)
out = mktempdir()
write_chrome_tables(P, (out, mktempdir()), rows)
for f in readdir(out)
    startswith(f, "table_") && cp(joinpath(out, f), joinpath(G, f); force = true)
end
Reproduce.write_numbers(joinpath(G, "numbers.tex"), chrome_table_macros(P, path, rows, products.provenance; driver = "scripts/chrome_evaluation_paper.jl"))
println("GOLDEN ", join(readdir(G), " "))
