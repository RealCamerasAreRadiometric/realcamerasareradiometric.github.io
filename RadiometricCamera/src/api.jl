# ---------------------------------------------------------------------------
# RadiometricCamera's API surface — the one file that states it (owner D13, 2026-11-15).
#
#   export   the headline API: the verbs of a radiometric calibration (start, refine, the one-call calibrate), their inputs and
#            result types, the options and their TOML reader. Everything a user writes unqualified.
#   public   the user-facing names beyond it (the image formation, the camera and target models, the records, the diagnostics, the
#            LM's blocks for those who build their own problem, the paper's symbols — D14: demoted, names kept), reached qualified
#            (`RadiometricCamera.name`) or imported by name (`using RadiometricCamera: name`).
# Everything else is internal. The names of Core's functions RC adds methods to (`lens`, `chart`, `cost`, `residuals`) are Core's:
# reach them through VisualGeometryToolkitCore.
# ---------------------------------------------------------------------------

export calibrate, start_calibration, refine_calibration!, StartGeometry, start_geometry, load_calibration_start,
       save_calibration_start, save_radiometric_calibration, start_photographs, CalibrationEstimate, CalibrationMetadata,
       CalibrationState, RadiometricCameraModel, RadiometricTarget, PipelineOptions, pipeline_options, CalibratedCamera

public 𝐩_of_pixel, pixel_of_𝐩, ℐ_𝐮, luma, luma_image, SensorQuadrature, ApertureQuadrature, sensor_samples, aperture_samples,
       aperture_border, 𝐩_sample, 𝐚_sample, 𝐚_weight, 𝐚_border, n_disc, n_border, center_constraint, FocusSpline, Focus, t̂, 𝔱,
       n_nodes, in_domain, lm_box, lm_scale, is_grid_lens, grid_dims, grid_model, exact_start, start_grid, region_start, coverage_region,
       check_coverage, calibrated_folds, grid_angular_error, schoeps_gridlens,
       schoeps_start, load_start, schoeps_run_file, save_grid, load_grid, load_model, load_camera, save_lens,
       load_parametric_lens, load_lens, lens_type_name, 𝔭, 𝔣, π̂⁻¹, chief_dir, world_ray, TargetRadiometry, material,
       edge_parameters, target_radiometry, board_surface, radiometric_target, set_radiometric_target!, with_vignetting, with_lens,
       with_focus, with_aperture_radius,
       PrintedTarget, posed_target, board_normal, 𝔟, texture_size, corners, intersect_board, board_plane_hit, world_point,
       inside, ρ_c, with_sdf, has_sdf, BoardSurface, with_h, with_edge, height_on, HEIGHT_NC, HEIGHT_NFREE, EDGE_NC,
       warping_nodes, warping_aspect, warping_field, page_points, height_manifold, height_basis, board_height, board_relief,
       height_nparams, edge_albedo, INPLANE_NC, with_inplane, inplane_on, inplane_calibration, design_chart, physical_chart,
       sphere_radius, Material, Θ_m_nonlinear, with_nonlinear, with_μ, texture_weight, 𝛂, 𝛈, D_ggx, Λ_ggx, G1_ggx, G2_ggx,
       F_dielectric, sample_vndf_caps, D_v, PDF_reflect, reflect, f_CT_parts, f_CT, RadianceSampler, SampleKey, draws,
       key_hash, L_c, RadianceParts, radiance_parts, L_from_parts, RenderSettings, Scene, with_surface, pixel_id, Φ_pixel,
       compute_mask, masked_pixels, render_view, masked_vector, n_views, focus, aperture_radius, set_surface!,
       set_vignetting!, set_pose!, render_all, photometric_residuals, PixelParts, pixel_parts, RadiometryParts, all_parts,
       solve_Θ_l!, solve_Θ_v!, solve_μ!, solve_radiometry_gn!, solve_linear_blocks!, lens_free, lens_manifold, focus_block,
       ApertureVar, MaterialVar, LightingVar, PerViewResidual, build_problem, read_back!, run_lm!, extract_variables,
       Θ̄_snapshot, tile_costs, build_problem_grid, run_lm_grid!, extract_variables_grid, load_session,
       thin_lens_focus_distance, dataset_features, REFINE_ONLY_KEYS, split_refine_only, start_options, start_pipeline_table,
       check_start_options, bearing_px, detection_bearings, state_bearings, fpx_of, residual_vectors, board_hits, render_dump!, load_state_into!, save_poses,
       load_pipeline_metrics, RECORD_SCHEMAS, save_run_parameters, load_run_parameters, save_initialization_region,
       measure_image_region, save_image_region, load_image_region, run_lens_coverage, SENSOR_LENS_COVERAGE,
       GEOMETRY_IMAGES, check_geometry_image, raw_green_image,
       stored_initialization_region, save_state, load_state, save_radiometric_camera, load_radiometric_camera,
       save_radiometric_target, load_radiometric_target, initialization_region, in_initialization, selfcheck, settings_line,
       RAW_EXTENSIONS, green_photosite_mask, is_raw_file, is_png_file, photo_files, raw_file, raw_image_size, photo_file,
       load_photo, sensor_record, NO_SENSOR_CHARACTERISATION, calibrated_camera, ObservationModel, photo_set, observe, SensorStatement, SensorRecord, NoSensorRecord, PhotoKind,
       SensorModelMode, AutoSensorModel, OnSensorModel, OffSensorModel, sensor_model_mode, warn_sensor_model_off,
       Photosites, Decoded, OBSERVATION_CHANNELS, PhotoSet, observation_model, sensor_statement, observation_line,
       names_input_path, PRINT_PLANE, load_distance_map, config_dict, write_pipeline_config, IOOptions,
       LensIdentityOptions, FrontendOptions, CameraOptions, GridOptions, StartPerturbationOptions, FocusOptions, MaskOptions,
       RenderingOptions, LightingOptions, MaterialOptions, VignettingOptions, WarpingOptions, EdgeOptions, LMOptions,
       SweepOptions, ResumeOptions, SelfcheckOptions, ComputeOptions, FRONTEND_TAGS, VIGNETTING_CONFIG_TAGS, BACKEND_TAGS,
       rc_backend, FIXABLE_BLOCKS, PIPELINE_PATH_KEYS, AbstractFrontend, BlobFrontend, vignetting_center, SchoepsFrontend,
       AbstractComputeBackend, CPUCompute, GPUCompute, start_is_grid, read_switch, switch_spelling, boolean_switch_violations,
       LatticeRule, DerivedLattice, NodeCountLattice, LATTICE_TAGS, INAPPLICABLE_KEYS, inapplicable_reason

# ---- retired names (RC 0.11.0), each refused by name with its replacement (no alias, no forwarding)
"""Retired (RC 0.11.0, owner 2026-11-15): VisualGeometryToolkitCore's `backproject(camera, 𝐩)`."""
π⁻¹(args...; kwargs...) = throw(ArgumentError("π⁻¹ is retired (RC 0.11.0): use VisualGeometryToolkitCore's `backproject(camera, 𝐩)` " *
                                               "(a unit ray, NaN where the lens does not image 𝐩)"))
"""Retired (RC 0.11.0, owner 2026-11-15): VisualGeometryToolkitCore's `project`."""
project_pixel(args...; kwargs...) = throw(ArgumentError("project_pixel is retired (RC 0.11.0): use VisualGeometryToolkitCore's `project(camera, X)`; " *
                                                        "a SplineLens camera's batched `project(model(camera), rays, seeds)`, the detections as the seeds"))
# (RC 0.11.1, owner 2026-11-15: ONE entry point for photographs, the calibrated camera's observation model — no second decode path)
"""Retired (RC 0.11.1): photographs are read through THE entry point, `photo_set(camera_or_observation_model, files; characterisation)` and `observe(set, k, mask)`."""
photo_observations(args...; kwargs...) = throw(ArgumentError("photo_observations is retired (RC 0.11.1): read photographs through the entry point, " *
                                                             "`observe(photo_set(camera, files; characterisation), k, mask)` — the camera's observation model decides the decode"))
"""Retired (RC 0.11.1): each photograph's sensor model is `photo_set(...).models`."""
photo_sensor_models(args...; kwargs...) = throw(ArgumentError("photo_sensor_models is retired (RC 0.11.1): `photo_set(observation_model(files, spec), files; characterisation).models` " *
                                                              "— the frames' models of the camera's own record"))
"""Retired (RC 0.11.1): a characterisation is stated by `sensor_statement(spec)` and read by `photo_set`."""
sensor_characterisation(args...; kwargs...) = throw(ArgumentError("sensor_characterisation is retired (RC 0.11.1): `sensor_statement(spec)` states the record " *
                                                                  "(its sha256), `photo_set` reads it at the location given"))

# ---- retired call forms (owner 2026-11-20: the camera-first API). A radiometric camera is ONE object, a `RadiometricCameraModel`
# (lens, focus surface, aperture radius, vignetting); a photograph is the camera at a pose. The forms that took a `View` assembled
# from the camera's parts, with the vignetting as a separate argument, are refused by name with their replacement (no alias, no
# forwarding). `View` itself is the kernels' internal record (`View(pose, camera)`).
_retired_view_form(name, replacement) =
    throw(ArgumentError("$name of a View (the camera's parts, the vignetting apart) is retired: a radiometric camera is one object. " *
                        "Give the camera and the pose: $replacement"))
CalibrationState(::Scene, ::RenderSettings, ::AbstractVector{<:View}, ::AbstractVignetting, args...; kwargs...) =
    _retired_view_form("CalibrationState", "`CalibrationState(scene, settings, camera, poses, mat, light, targets; …)`, " *
                       "`camera = RadiometricCameraModel(lens, focus, aperture_radius, vignetting)`")
render_view(::Integer, ::View, ::AbstractVignetting, args...) =
    _retired_view_form("render_view", "`render_view(k, camera, pose, mat, light, scene, settings, mask)`")
compute_mask(::View, ::Scene, ::RenderSettings; kwargs...) =
    _retired_view_form("compute_mask", "`compute_mask(camera, pose, scene, settings; δ, external, backend)`")
Φ_pixel(::Integer, ::Integer, ::Integer, ::View, ::AbstractVignetting, args...) =
    _retired_view_form("Φ_pixel", "`Φ_pixel(u, v, k, camera, pose, mat, light, scene, settings)`")
pixel_parts(::Integer, ::Integer, ::Integer, ::View, args...) =
    _retired_view_form("pixel_parts", "`pixel_parts(u, v, k, camera, pose, mat, light, scene, settings)`")
