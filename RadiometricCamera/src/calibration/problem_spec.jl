"""
    _lm_structure(opts) -> (; height_degree, inplane)

The surface parameters the run's model HAS (its structure, the LMs' keywords of that name): the board warping of `warping.degree`,
the in-plane print error of `warping.inplane = "aspect_skew"` (the printed edge is never a parameter). With
the holds (`_lm_surface_holds`) the one encoding of the surface's free parameters: a parameter is free where the model has it and
`fixed` does not hold it (`_surface_free`).
"""
_lm_structure(opts::PipelineOptions) = (; height_degree = opts.warping.degree, inplane = opts.warping.inplane == "aspect_skew")
"""The surface parameters `fixed` holds (`SURFACE_HOLDS`: "inplane"), as the LMs' `fixed` entries ("edge" in `fixed` is not one: the
printed edge is never a parameter, and the name is accepted for the configs of record)."""
_lm_surface_holds(opts::PipelineOptions) = Tuple(Symbol(k) for k in opts.fixed if Symbol(k) in SURFACE_HOLDS)
"""Whether the surface parameter `k` (`:height`, `:inplane`) is free: the model has it (`_lm_structure`) and `fixed` does
not hold it (`_lm_surface_holds`) — read off the one encoding, for the reports and the checks."""
function _surface_free(opts::PipelineOptions, k::Symbol)
    st = _lm_structure(opts)
    has = k === :height ? st.height_degree > 0 : k === :inplane ? st.inplane :
          throw(ArgumentError("_surface_free: $(repr(k)) is not a surface parameter ($(join(repr.(SURFACE_HOLDS), ", ")))"))
    return has && !(k in _lm_surface_holds(opts))
end
"""
    _refinement_lm(opts, route; smoothness = nothing) -> (; kw, blocks)

THE refinement's LM problem, derived once from the options: `blocks` (`_lm_blocks`: which blocks are solved) and `kw`, the keywords
of the route's builder and LM (`lm_route`: `build_problem` / `_lm_views!` per view, `build_problem_grid` / `_lm_tiles!` per tile): the held blocks as `fixed` (`:grid` the
per-view route's `:distortion`; `:material` whenever "material" is in `fixed` — the builders hold Θ_m alone inside a block that
stays variable for its free surface parameters, `_hold_material`, as `_material_rides` reports it), the
surface's holds from `fixed`, the model's structure (`_lm_structure`) and the grid smoothness `smoothness` (the LM's `damping`, the
optimizer's, is the caller's: a builder takes it beside these). The calibration's refinement, the grid smoothness's GCV, the Jacobian parity
check and the tests build their problems from it alone.
"""
function _refinement_lm(opts::PipelineOptions, route::LmRoute; smoothness = nothing)
    blocks = _lm_blocks(opts)
    held = Tuple(k for k in keys(blocks) if !getfield(blocks, k) || (k === :material && "material" in opts.fixed))
    kw = _route_kw(route, held, opts)
    smoothness === nothing || (kw = (; kw..., smoothness))
    return (; kw, blocks)
end
# the route's builder keywords: the per-view route names the lens block `:distortion` (`build_problem`), the per-tile route `:grid`
function _route_kw(::PerViewRoute, held, opts::PipelineOptions)
    st = _lm_structure(opts)
    return (; fixed = (Tuple(k === :grid ? :distortion : k for k in held)..., _lm_surface_holds(opts)...),
              height_degree = st.height_degree, inplane = st.inplane)
end
_route_kw(::PerTileRoute, held, opts::PipelineOptions) =
    (; fixed = (held..., _lm_surface_holds(opts)...), _lm_structure(opts)...)
"""The route's LM (`lm_route`): `_lm_views!` per view, `_lm_tiles!` per tile."""
_lm_route!(f::CalibrationData, θ::CalibrationParameters, ::PerViewRoute; kwargs...) = _lm_views!(f, θ; kwargs...)
_lm_route!(f::CalibrationData, θ::CalibrationParameters, ::PerTileRoute; kwargs...) = _lm_tiles!(f, θ; kwargs...)
"""Whether the run models an in-plane print error on the surface `surf`: `warping.inplane` names one, or the surface carries one
(`inplane_on`; a start or a resumed state's). Considered: `_surface_free(opts, :inplane)` (free, not modelled — a held print error is modelled)."""
_inplane_modelled(opts::PipelineOptions, surf) = opts.warping.inplane != "none" || inplane_on(surf)
"""The board features under the surface `surf`: `meta.features_for(surf)` when the board is warped or carries an in-plane print error
(the features follow them), else `features` (the design's)."""
_board_features(meta, surf, features) = height_on(surf) || inplane_on(surf) ? meta.features_for(surf) : features
"""Whether the material block carries free surface parameters besides Θ_m (the board warping, the in-plane print
error): then "material" in `fixed` holds Θ_m alone (the builders' `_hold_material`: its columns zero) and the block stays variable;
else it holds the block."""
_material_rides(opts::PipelineOptions) = any(k -> _surface_free(opts, k), SURFACE_HOLDS)

"""`_lm_blocks(opts) -> NamedTuple`: per block of the refinement's LM (`poses`, `grid` — the lens —, `focus`, `aperture`, `material`,
`vignetting`, `lighting`) whether it is solved: held by `fixed`, "material" in
`fixed` holds Θ_m alone while the block carries a free surface parameter (`_material_rides`)."""
function _lm_blocks(opts::PipelineOptions)
    base = (; poses = true, grid = true, focus = true, aperture = true,
            material = !("material" in opts.fixed) || _material_rides(opts), vignetting = true, lighting = true)
    return (; (k => (k !== :material && _fixed_name(k) in opts.fixed ? false : getfield(base, k)) for k in keys(base))...)
end
"""The name of the LM block `k` in `fixed` (`FIXABLE_BLOCKS`): the camera's block is `:grid` inside the LM, `"camera"` in the options."""
_fixed_name(k::Symbol) = k === :grid ? "camera" : String(k)

# ---- the route's camera: its record, its label, its log line ---------------------------------------------------------------------
"""The stem of the route's camera record in a run's directory: `lens` (`lens.json`, a parametric camera) or `grid` (`grid.json`)."""
_camera_file(::PerViewRoute) = "lens"
_camera_file(::PerTileRoute) = "grid"
"""The camera's name in the records' model labels: its lens's (`lens_type_name`) per view, `grid` per tile."""
_model_label(::PerViewRoute, camera) = lens_type_name(camera)
_model_label(::PerTileRoute, camera) = "grid"
"""A camera record of the run (a result, a start, a checkpoint): `save_lens` per view; per tile `_save_run_grid`, the start's
initialization region `init_region` beside it."""
_save_run_camera(::PerViewRoute, path::AbstractString, camera, init_region; provenance::NamedTuple) = save_lens(path, camera; provenance)
_save_run_camera(::PerTileRoute, path::AbstractString, camera, init_region; provenance::NamedTuple) = _save_run_grid(path, camera, init_region; provenance)
"""The lens block in the free-block line: `lens(<n parameters>)` per view, `grid` per tile."""
_lens_block_label(::PerViewRoute, camera) = "lens($(length(VgtkCore.parameter_names(VgtkCore.model(camera)))))"
_lens_block_label(::PerTileRoute, camera) = "grid"
"""The monitor's camera line: a parametric camera's pinhole and lens parameters; a grid's none."""
function _print_camera_parameters(::PerViewRoute, camera)
    pp = VgtkCore.parameters(VgtkCore.model(camera))
    @printf("      lens: f (%.2f, %.2f) px  c (%.2f, %.2f) px  %s\n", pp.f, pp.f * pp.aspect, pp.cx, pp.cy,
            join([@sprintf("%s %+.4g", n, v) for (n, v) in pairs(VgtkCore.parameters(VgtkCore.lens(camera)))], "  "))
    return nothing
end
_print_camera_parameters(::PerTileRoute, camera) = nothing
"""The start's route line: a parametric camera's lens and parameters (refined in its own parameters, per view); a grid's none (its
nodes' line follows the masks)."""
_print_start_route(::PerViewRoute, camera) =
    @printf("parametric lens %s (%d parameters: %s) — refined in its own parameters (per-view LM)\n", lens_type_name(camera),
            length(VgtkCore.parameter_names(VgtkCore.model(camera))), join(String.(VgtkCore.parameter_names(VgtkCore.model(camera))), " "))
_print_start_route(::PerTileRoute, camera) = nothing
"""The lattice nodes a grid start holds under the region (`grid_present`, per_tile_smoothness.jl, by `nodes`); a parametric camera has none."""
_present_nodes(::PerViewRoute, camera, nodes::Symbol) = nothing
_present_nodes(::PerTileRoute, camera, nodes::Symbol) = grid_present(camera; nodes)
