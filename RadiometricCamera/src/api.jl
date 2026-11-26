# ---------------------------------------------------------------------------
# RadiometricCamera's API surface — the one file that states it.
#
#   export   the headline API: the radiometric calibration (its algorithm, solved by VisualGeometryToolkitCore's `init`/`solve!`/`solve`,
#            and the one-call calibrate), its inputs, cache and result types, the options and their TOML reader. Everything a user writes unqualified.
#   public   the user-facing names beyond it (the image formation, the camera and target models, the records, the diagnostics, the
#            LM's blocks for those who build their own problem, the paper's symbols — D14: demoted, names kept), reached qualified
#            (`RadiometricCamera.name`) or imported by name (`using RadiometricCamera: name`).
# Everything else is internal. The names of Core's functions RC adds methods to (`chart`, `cost`, `residuals`) are Core's:
# reach them through VisualGeometryToolkitCore. The generic radiometric types and functions are VisualRadiometryToolkitCore's (the
# vignetting fields and the light, which were VisualGeometryToolkitCore's, and what moved there from this package: the camera's
# parts and maps, the focus surface, the sensor domain and quadrature, the material, the textured target with its state — the board):
# reach them through
# VisualRadiometryToolkitCore — they are not public names of this package. The one re-export is the headline's camera type,
# `RadiometricCameraModel`, which every verb here takes.
# ---------------------------------------------------------------------------

export calibrate, RadiometricCalibration, CalibrationCache, read_calibration, StartGeometry, start_geometry, load_calibration_start,
       save_calibration_start, save_radiometric_calibration, start_photographs, CalibrationEstimate, CalibrationMetadata,
       CalibrationState, RadiometricCameraModel, RadiometricTarget, PipelineOptions, pipeline_options, CalibratedCamera

public flat_vignetting, center_constraint, in_domain, lm_box, lm_scale, is_grid_lens, grid_dims, GridStart, ImageLattice,
       coverage_region, check_coverage, calibrated_folds, grid_angular_error,
       load_start, schoeps_run_file, save_grid, load_grid, load_model, load_camera,
       save_lens, load_parametric_lens, load_lens, lens_type_name, TargetRadiometry, material, edge_parameters,
       target_radiometry, board_surface, radiometric_target, set_radiometric_target!, HEIGHT_NFREE, height_manifold,
       height_basis, height_nparams, sphere_radius, RadianceSampler, SampleKey, draws, key_hash, L_c, RadianceParts,
       radiance_parts, L_from_parts, RenderSettings, pixel_id, Φ_pixel, compute_mask, masked_pixels, render_view,
       masked_vector, n_views, set_surface!, set_vignetting!, set_pose!, render_all, photometric_residuals, PixelParts,
       pixel_parts, RadiometryParts, all_parts, RadiometryBlocks, RadiometryBlockDescent, JointRadiometry,
       lens_free, lens_manifold, focus_block, ApertureVar, MaterialVar, LightingVar,
       PerViewResidual, build_problem, read_back!, extract_variables, Θ̄_snapshot, tile_costs,
       build_problem_grid, extract_variables_grid, load_session, thin_lens_focus_distance,
       dataset_features, REFINE_ONLY_KEYS, split_refine_only, start_options, start_pipeline_table, check_start_options,
       bearing_px, detection_bearings, state_bearings, fpx_of, residual_vectors, board_hits, render_dump!,
       load_state_into!, save_poses, load_pipeline_metrics, RECORD_SCHEMAS, save_run_parameters, load_run_parameters,
       save_initialization_region, measure_image_region, save_image_region, load_image_region, run_lens_coverage,
       SENSOR_LENS_COVERAGE, GEOMETRY_IMAGES, check_geometry_image, raw_green_image, stored_initialization_region,
       save_state, load_state, save_radiometric_camera, load_radiometric_camera, save_radiometric_target,
       load_radiometric_target, initialization_region, in_initialization, selfcheck, settings_line, RAW_EXTENSIONS,
       photosite_mask, is_raw_file, is_png_file, photo_files, raw_file, raw_image_size, photo_file, load_photo,
       sensor_record, NO_SENSOR_CHARACTERIZATION, calibrated_camera, ObservationModel, photo_set, observe,
       SensorStatement, SensorRecord, NoSensorRecord, PhotoKind, SensorModelMode, AutoSensorModel, OnSensorModel,
       OffSensorModel, sensor_model_mode, warn_sensor_model_off, Photosites, Decoded, OBSERVATION_CHANNELS, PhotoSet,
       observation_model, sensor_statement, observation_line, names_input_path, load_distance_map,
       config_dict, write_pipeline_config, IOOptions, LensIdentityOptions, FrontendOptions, CameraOptions, GridOptions,
       StartPerturbationOptions, FocusOptions, MaskOptions, RenderingOptions, LightingOptions, MaterialOptions,
       VignettingOptions, WarpingOptions, EdgeOptions, ResumeOptions, SelfcheckOptions,
       ComputeOptions, OPTIMIZER_TAGS, RC_CONFIG_VERSIONS, config_key, FRONTEND_TAGS, VIGNETTING_CONFIG_TAGS, BACKEND_TAGS, rc_backend, default_backend, FIXABLE_BLOCKS,
       PIPELINE_PATH_KEYS, PIPELINE_OUTPUT_KEYS, START_RECORD_FILE, START_MASKS_DIR, load_refinement_start, masked_observations, Unweighted, AbstractFrontend, BlobFrontend, vignetting_center, SchoepsFrontend, AbstractComputeBackend,
       CPUCompute, GPUCompute, start_is_grid, read_switch, switch_spelling, boolean_switch_violations, LatticeRule,
       DerivedLattice, NodeCountLattice, LATTICE_TAGS, INAPPLICABLE_KEYS, inapplicable_reason

# ---- retired names (an earlier release's), each refused by name with its replacement (no alias, no forwarding)
"""Retired in an earlier release: VisualGeometryToolkitCore's `backproject(camera, 𝐩)`."""
π⁻¹(args...; kwargs...) = throw(ArgumentError("π⁻¹ is retired: use VisualGeometryToolkitCore's `backproject(camera, 𝐩)` " *
                                               "(a unit ray, NaN where the lens does not image 𝐩)"))
"""Retired in an earlier release: VisualGeometryToolkitCore's `project`."""
project_pixel(args...; kwargs...) = throw(ArgumentError("project_pixel is retired: use VisualGeometryToolkitCore's `project(camera, X)`; " *
                                                        "a SplineLens camera's batched `project(model(camera), rays, seeds)`, the detections as the seeds"))
# (ONE entry point for photographs, the calibrated camera's observation model — no second decode path)
"""Retired in an earlier release: photographs are read through THE entry point, `photo_set(camera_or_observation_model, files; characterization)` and `observe(set, k, mask)`."""
photo_observations(args...; kwargs...) = throw(ArgumentError("photo_observations is retired: read photographs through the entry point, " *
                                                             "`observe(photo_set(camera, files; characterization), k, mask)` — the camera's observation model decides the decode"))
"""Retired in an earlier release: each photograph's sensor model is `photo_set(...).models`."""
photo_sensor_models(args...; kwargs...) = throw(ArgumentError("photo_sensor_models is retired: `photo_set(observation_model(files, spec), files; characterization).models` " *
                                                              "— the frames' models of the camera's own record"))

# ---- retired call forms (the camera-first API). A radiometric camera is ONE object, a `RadiometricCameraModel`
# (geometric camera, focus surface, aperture radius, vignetting); a photograph is the camera at a pose. The forms that took a `View` assembled
# from the camera's parts, with the vignetting as a separate argument, are refused by name with their replacement (no alias, no
# forwarding). `View` itself is the kernels' internal record (`View(pose, camera)`).
_retired_view_form(name, replacement) =
    throw(ArgumentError("$name of a View (the camera's parts, the vignetting apart) is retired: a radiometric camera is one object. " *
                        "Give the camera and the pose: $replacement"))
CalibrationState(::TexturedTarget, ::RenderSettings, ::AbstractVector{<:View}, ::AbstractVignetting, args...; kwargs...) =
    _retired_view_form("CalibrationState", "`CalibrationState(board, settings, camera, poses, mat, light, targets; …)`, " *
                       "`camera = RadiometricCameraModel(geometric, focus, aperture_radius, vignetting)`")
render_view(::Integer, ::View, ::AbstractVignetting, args...) =
    _retired_view_form("render_view", "`render_view(k, camera, pose, mat, light, board, settings, mask)`")
compute_mask(::View, ::TexturedTarget, ::RenderSettings; kwargs...) =
    _retired_view_form("compute_mask", "`compute_mask(camera, pose, board, settings; δ, external, backend)`")
Φ_pixel(::Integer, ::Integer, ::Integer, ::View, ::AbstractVignetting, args...) =
    _retired_view_form("Φ_pixel", "`Φ_pixel(u, v, k, camera, pose, mat, light, board, settings)`")
pixel_parts(::Integer, ::Integer, ::Integer, ::View, args...) =
    _retired_view_form("pixel_parts", "`pixel_parts(u, v, k, camera, pose, mat, light, board, settings)`")

# ---- retired names (`lens` is VisualGeometryToolkitCore's lens model everywhere). A radiometric camera's geometric
# camera (Core's bounded camera or a grid model) is its `geometric`: the field, the accessor of a camera and of a state, the replacer.
# The old spellings are refused by name with their replacement (no alias, no forwarding).
"""Retired: VisualRadiometryToolkitCore's `with_geometric`."""
with_lens(args...; kwargs...) = throw(ArgumentError("with_lens is retired: a radiometric camera's geometric camera is its `geometric` — " *
                                                    "use VisualRadiometryToolkitCore's `with_geometric(camera, geometric)`"))
_retired_lens_accessor(what) =
    throw(ArgumentError("lens($what) is retired: `lens` is VisualGeometryToolkitCore's lens model of a geometric camera. The geometric camera of " *
                        "a radiometric camera or a calibration state is `VisualRadiometryToolkitCore.geometric($what)`; its lens model `lens(geometric($what))`"))
VgtkCore.lens(::CalibrationState) = _retired_lens_accessor("state")

# ---- moved names (the generic radiometric layer is VisualRadiometryToolkitCore's). The names this package uses are
# imported from there (src/RadiometricCamera.jl) and are no longer public here; the ones it does not use are refused by name with
# their replacement (no alias, no forwarding).
# VisualRadiometryToolkitCore's public names are descriptive; a symbol of the paper is its public alias of the same function, and a
# plain old spelling has no alias: each row names what to call there.
_moved_name_refusal(name::Symbol, replacement::AbstractString) = throw(ArgumentError(
    "RadiometricCamera.$name moved to VisualRadiometryToolkitCore: $replacement"))
# each moved name with what to call there
const _MOVED_REPLACEMENTS = (
        :with_geometric => "`with_geometric`", :with_focus => "`with_focus`", :with_aperture_radius => "`with_aperture_radius`",
        :aperture_samples => "`aperture_samples`", :aperture_border => "`aperture_border`", :luma => "`luma`", :n_nodes => "`n_nodes`",
        :pixel_of_𝐩 => "`pixel_position` (the paper's symbol stays its alias, `pixel_of_𝐩`)",
        :ℐ_𝐮 => "`pixel_footprint` (the paper's symbol stays its alias, `ℐ_𝐮`)",
        :𝔱 => "`focus_distance` (the paper's symbol stays its alias, `𝔱`)",
        :Λ_ggx => "`ggx_lambda` (the paper's symbol stays its alias, `Λ_ggx`)",
        :G1_ggx => "`ggx_masking`", :D_ggx => "`ggx_distribution`", :G2_ggx => "`ggx_masking_shadowing`",
        :F_dielectric => "`fresnel_dielectric`", :D_v => "`visible_normal_density`",
        :chief_dir => "`chief_direction`", :sample_vndf_caps => "`sample_visible_normal`", :PDF_reflect => "`reflection_pdf`",
        :f_CT_parts => "`cook_torrance_terms`", :f_CT => "`cook_torrance`",
        :ae_chart => "the azimuthal-equidistant chart is VisualGeometryToolkitCore's equidistant lens, `project(Lens(Equidistant()), d)`",
        :ae_dir => "the azimuthal-equidistant chart is VisualGeometryToolkitCore's equidistant lens, `backproject(Lens(Equidistant()), ξ)`",
        # the textured target's names this package does not use itself (the ones it uses are imported, no longer public here)
        :posed_target => "`posed_target(target)` (under its state) or `posed_target(target, state)`", :texture_size => "`texture_size`",
        :has_sdf => "`has_sdf`", :warping_field => "`warping_field`", :board_relief => "`board_relief`", :physical_chart => "`physical_chart`",
        :ρ_c => "`albedo` (the paper's symbol stays its alias, `ρ_c`)",
        :edge_albedo => "`edge_albedo(target, 𝐛)`: the printed edge is the target's state's (`with_edge(target, (δ, σ_print))`)")
"""The names this package refuses by name since they moved to VisualRadiometryToolkitCore, each with what to call there (the
stack's `MOVED_NAMES` convention: VisualGeometryToolkitCore's `check_duplication` reads `MOVED_NAMES` and counts a listed name as a
refusal, not a definition, so its new owner's is the one definition)."""
const MOVED_NAMES = map(first, _MOVED_REPLACEMENTS)
for (name, replacement) in _MOVED_REPLACEMENTS
    @eval begin
        @doc $("Moved to VisualRadiometryToolkitCore: $replacement.")
        $name(args...; kwargs...) = _moved_name_refusal($(QuoteNode(name)), $replacement)
    end
end

# ---- retired names (the board is VisualRadiometryToolkitCore's textured target, and there is no scene). The board's
# surface parameters are the target's OWN state; the image size is the camera's; a render takes the camera, the pose and the board.
# Each old name is refused by name with its replacement (no alias, no forwarding).
"""Retired: VisualRadiometryToolkitCore's `TexturedTarget`."""
PrintedTarget(args...; kwargs...) = throw(ArgumentError(
    "PrintedTarget is retired: the board is VisualRadiometryToolkitCore's `TexturedTarget` (the same constructors: " *
    "`TexturedTarget(target, texture; facing, sdf, page_mm)`, `TexturedTarget(𝐨, 𝐮_B, 𝐯_B, texture; …)`), which holds its own state"))
"""Retired: VisualRadiometryToolkitCore's `TargetState`, the state of a `TexturedTarget`."""
BoardSurface(args...; kwargs...) = throw(ArgumentError(
    "BoardSurface is retired: the board's warping h, printed edge and in-plane print error are the textured target's own state, " *
    "VisualRadiometryToolkitCore's `TargetState(; h, edge, inplane)` — `with_state(board, state)`, or `with_h`, `with_edge`, `with_inplane` on the board"))
"""Retired: there is no scene; a render takes the camera, the pose and the board."""
Scene(args...; kwargs...) = throw(ArgumentError(
    "Scene is retired: there is no scene type. The image size is the camera's (`extent(geometric(camera))`) and the board holds its own " *
    "state: give the board itself, a `TexturedTarget` (`with_state(board, state)` for a warped or printed-edge board) — " *
    "`render_view(k, camera, pose, mat, light, board, settings, mask)`, `CalibrationState(board, settings, camera, poses, …)`"))
"""Retired: VisualRadiometryToolkitCore's `with_state`."""
with_surface(args...; kwargs...) = throw(ArgumentError(
    "with_surface is retired: the surface parameters are the board's own state — VisualRadiometryToolkitCore's `with_state(board, state)`"))

# ---- retired name: the photosites of any one colour filter are read by one function
"""Retired: `photosite_mask(m, :G)`."""
green_photosite_mask(args...; kwargs...) = throw(ArgumentError(
    "green_photosite_mask is retired: the usable photosites of one colour filter are `photosite_mask(m, channel)`, channel :R, :G or :B — " *
    "the green photosites `photosite_mask(m, :G)`"))

# ---- removed name (the focus A/B experiment is decided: the focus surface is the sensor-grid thin plate spline). The focus surfaces
# on the chief directions are deleted, here and in VisualRadiometryToolkitCore; the old name is refused by name.
"""Removed with the focus surfaces on the chief directions: the inverse focus distance is read at a sensor position."""
t̂_dir(args...; kwargs...) = throw(ArgumentError(
    "t̂_dir was removed with the focus surfaces on the chief directions: the focus surface is the sensor-grid thin plate spline, read at " *
    "a sensor position — VisualRadiometryToolkitCore's `inverse_focus_distance(focus, 𝐩)` (the paper's `t̂`, its alias)"))
