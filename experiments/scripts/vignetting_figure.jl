#!/usr/bin/env julia
# reproduce: writes ../paper/fig/vignetting.tex
# reproduce: reads configs/experiments/vignetting_figure.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/vignetting_figure.jl — the paper's vignetting figure (fig:vignetting): per lens, along the sensor's diagonal through the
# centre 𝔠 of the vignetting its large-aperture refinement learned, the flat field of a diffuser photograph, the learned vignetting
# and the thin lens model's cos⁴α (its focal length the refinement case's own), each over its value at 𝔠
# (src/vignetting_figure.jl), drawn as the paper's PGFPlots files (src/paper_figures/vignetting.jl). Config of record:
# configs/experiments/vignetting_figure.toml.
#
# STAGES (each a product keyed by its config record and the keys of the products it reads; datadir("exp_pro", <experiment>, <stage>, <key>))
#   figures   per lens its curves from the refined camera's record and its diffuser photograph (tables/<lens>_diagonal.tsv,
#             tables/meta.toml), the figure's files from them → plotsdir(<asset_dir>.tex, <asset_dir>/<panel>-diagonal.tex,
#             <asset_dir>-caption.tex), numbers.tex (empty: the figure cites no number)
#
#   julia --project=experiments experiments/scripts/vignetting_figure.jl [--config=<toml>] [--plan] [--force]
#
# The stage reads the refine product of each lens's case (scripts/radiometric_refinement.jl on the lens's refinement config); a
# refinement that is not produced is produced first, which needs a GPU.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))                                  # the refinement configs the lenses name (refine_product)
include(srcdir("vignetting_figure.jl"))
include(srcdir("paper_figures", "PaperFigures.jl"))                           # the figure's PGFPlots text (its generator: vignetting.jl there)
using VisualGeometryToolkitEval.Experiments: run_isolated                     # a missing refinement is produced by its own driver (driver_run)

const DRIVER = "scripts/vignetting_figure.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "vignetting_figure.toml")))   # default: the config of record
const CFG = load_vignetting_figure_config(CFG_PATH)
"Per lens its refinement: the config of record, the case and its refine product (`vignetting_refinements`), resolved once."
const REFINEMENTS = vignetting_refinements(CFG; path = CFG_PATH)

"The figure's product: keyed by the config's tables, its inputs and the keys of the lenses' refine products (`vignetting_figure_product`)."
figures_product() = vignetting_figure_product(CFG, REFINEMENTS)

"""Per lens the location of the sensor characterisation record its diffuser photograph is read under: the one its refinement read
its photographs under (`case_characterisation`: an input of the refinement's config, its sha256 checked; `nothing` for a case
without one)."""
lens_characterisations() = Dict{String, Union{Nothing, String}}(k => case_characterisation(REFINEMENTS[k].rc, REFINEMENTS[k].case) for k in CFG["figure"]["order"])

"""
    lens_curves(k, characterisation) -> (; curves, f_mm, f_px, refine_key, photograph)

The curves of lens `k` (`lens_diagonal`): from the camera record of its refinement — the refine product, produced first when it is
missing (`refinement_upstream`) — and its one diffuser photograph, read under the sensor record the refinement read its photographs
under (`characterisation`, its location: `lens_characterisations`). The thin lens model's focal length is the refinement case's own
(`case_focal_length`: `f_mm`, and `f_px` in the camera's pixels).
"""
function lens_curves(k::AbstractString, characterisation::Union{Nothing, AbstractString})
    t = CFG["lenses"][k]; r = REFINEMENTS[k]
    dir = refinement_upstream(r.rc, r.config, r.case, "refine", "[lenses.$k]: the refinement of case $(r.case) of $(r.config)")
    record = joinpath(dir, "radiometric_camera.json")
    isfile(record) || throw(ArgumentError("[lenses.$k] case $(r.case): $dir has no radiometric_camera.json (the refinement is not complete)"))
    photograph = diffuser_photograph(config_input(CFG, t["diffuser"]), "[lenses.$k] diffuser $(t["diffuser"])")
    camera = RadiometricCamera.load_radiometric_camera(record)
    focal = case_focal_length(r.rc, r.case, camera)
    curves = lens_diagonal(camera, photograph, focal.f_px, CFG["diagonal"]; characterisation)
    println("   $k ($(r.case), $(basename(photograph))): 𝔠 ($(round(curves.centre[1]; digits = 2)), $(round(curves.centre[2]; digits = 2))), " *
            "$(length(curves.t)) samples in [$(round(minimum(curves.t); digits = 1)), $(round(maximum(curves.t); digits = 1))] px, " *
            "$(curves.photosites) green photosites; the thin lens model's focal length $(focal.f_mm) mm = $(round(focal.f_px; digits = 1)) px")
    return (; curves, focal..., refine_key = r.product.key, photograph = basename(photograph))
end

"""Stage figures: the lenses' curves as tables (`write_vignetting_tables`, into the product's tables/) and the figure drawn from
them by its generator (src/paper_figures/vignetting.jl, with the settings of `[figure]` and `[lenses]`; its files name this stage
as their generator), kept in the product (`assets/`) and published to plotsdir() (`produce_figures`); numbers.tex, the output,
written last. What it reads is declared to the product's provenance: the lenses' refine products, the diffuser photographs and
the sensor characterisation records the photographs are read under."""
function figures(p)
    order = CFG["figure"]["order"]; records = lens_characterisations()
    inputs = [[product_dir(REFINEMENTS[k].product) for k in order]; require_stage_inputs(CFG, "figures");
              unique(String[records[k] for k in order if records[k] !== nothing])]
    F = CFG["figure"]; L = CFG["lenses"]
    return produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH, inputs) do dir
        lenses = Dict{String, Any}(k => lens_curves(k, records[k]) for k in order)
        tables = write_vignetting_tables(joinpath(dir, "tables"), CFG, lenses)        # the data folder the generator reads
        written = PaperFigures.generate("vignetting"; data = tables, out = joinpath(dir, "assets"),
                                        generator = "experiments/$DRIVER (stage figures; T. Harder's vignetting.py)",
                                        asset_dir = F["asset_dir"], samples = F["samples"],
                                        panels = Dict{String, String}(k => L[k]["panel"] for k in order),
                                        caption_labels = Dict{String, String}(k => L[k]["caption_label"] for k in order))
        foreach(f -> println(">> figures → ", f), written)
        Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
    end
end

"The stage runs on no device (the refined cameras are records, the photographs are read on the host)."
const GPU_STAGES = Symbol[]

const STAGES = [
    Stage(:figures, "the paper's figure fig:vignetting (plotsdir(<asset_dir>.tex, <asset_dir>/<panel>-diagonal.tex, <asset_dir>-caption.tex)), its tables, numbers.tex";
          items = () -> [figures_product()], product = identity, run = figures),
]

run_stages("vignetting_figure", STAGES)
