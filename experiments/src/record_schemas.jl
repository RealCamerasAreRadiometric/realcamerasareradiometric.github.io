# =============================================================================
# src/record_schemas.jl — the schemas of the experiments' own result records, ONE table (as RadiometricCamera's RECORD_SCHEMAS):
# VisualGeometryToolkitIO 0.10's `write_record` requires each record's `schema` ("<owner>/<kind>/v<major>.<minor>") and writes
# the envelope (generator, units, conventions) itself. Included by the prelude (src/experiments.jl) and by the modules that write a
# record (src/evaluation.jl), as src/residual_summary.jl is.
# =============================================================================
const EXPERIMENT_RECORD_SCHEMAS = (
    chrome_evaluation = "experiments/chrome_evaluation/v1.0",
    chrome_index_grid_rows = "experiments/chrome_index_grid_rows/v1.0",
    chrome_index_grid_table = "experiments/chrome_index_grid_table/v1.0",
    chrome_index_photos = "experiments/chrome_index_photos/v1.0",
    conversion = "experiments/conversion/v1.0",
    convergence = "experiments/convergence/v1.0",
    evaluation_fixture = "experiments/evaluation_fixture/v1.0",
    exif = "experiments/exif/v1.0",
    night_stars = "experiments/night_stars/v1.0",
    panorama_maps = "experiments/panorama_maps/v1.0",
    panoramas = "experiments/panoramas/v1.0",
    photosites = "experiments/photosites/v1.0",
    sampling_strategy = "experiments/sampling_strategy/v1.0",
    sampling_strategy_configs = "experiments/sampling_strategy_configs/v1.0",
    sensor_models = "experiments/sensor_models/v1.0",
    registration_scale = "experiments/registration_scale/v1.0",
    residual_vectors = "experiments/residual_vectors/v1.0",
    texture_page = "experiments/texture_page/v1.0",
    texture_page_variant = "experiments/texture_page_variant/v1.0",
    texture_poses = "experiments/texture_poses/v1.0",
    texture_recovery_chosen = "experiments/texture_recovery_chosen/v1.0",
    texture_recovery_curves = "experiments/texture_recovery_curves/v1.0",
    texture_recovery_folds = "experiments/texture_recovery_folds/v1.0",
    texture_recovery_report = "experiments/texture_recovery_report/v1.0",
    textures = "experiments/textures/v1.0",
    timing = "experiments/timing/v1.0",
    timing_summary = "experiments/timing_summary/v1.0",
    tool_images = "experiments/tool_images/v1.0",
    warping_ablation = "experiments/warping_ablation/v1.0",
)
