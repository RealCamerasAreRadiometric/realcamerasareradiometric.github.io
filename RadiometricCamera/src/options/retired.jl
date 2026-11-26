# The retired keys of the pipeline's configs, each refused by name, whatever its value, with its migration (VisualGeometryToolkitCore's `check_keys` would
# refuse them as unknown keys without one; its `retired` warns and ignores, which would change an old config's run silently).
# the migration of a key of the removed focus A/B experiment: what was removed and what the focus surface is, then the key's own note
_focus_ab_key_removed(note::AbstractString) =
    "removed: the focus A/B experiment is decided and its other parameterisations (the nodes outside the support frozen, the " *
    "direction-chart thin plate spline, the spherical spline) are deleted — the focus surface is the sensor-grid thin plate spline, " *
    "every node free" * note
const RETIRED_KEYS = (
    ("io", "photosites") => "a RAW photograph is read as one by its file (`load_photo`: VisualGeometryToolkitCore's RawMosaic, its green photosites; " *
                            "`io.channel = \"G\"`), a decoded image as one — never by a flag; delete the key",
    ("lm", "iterator")   => "the LM is VisualGeometryToolkitCore's one (`lmsolve_chart!`), its damping the algorithm's (`RadiometricCalibration`'s optimizer, Levenberg's λ·I by default); " *
                            "delete the key",
    ("io", "save_init")  => "the start is a step of its own: `bin/refine.jl --stage=start` writes the start record (start.json, state.json, " *
                            "grid.json | lens.json, masks/; a `RadiometricCalibration` start), `--start=<dir>` refines from it; delete the key",
    ("masks", "linear_fraction") => "not read since 2026-10-31: the closed-form start runs on the LM's pixels; delete the key",
    ("masks", "lm_fraction") => "retired: the samples are a fixed count of each view's usable observations, " *
                            "`masks.samples_per_view` (a Bernoulli share of the raw grid gave a RAW photosite run half an RGB run's samples)",
    ("masks", "importance_weights") => "retired: the importance weights 1/p undid the edge bias's " *
                            "focus gain on the fisheye sweep and gave no Thorlabs gain; the samples count alike — delete the key",
    ("masks", "max_pixels_per_view") => "retired: the count per view is `masks.samples_per_view` (drawn from the " *
                            "usable observations; no cap of a Bernoulli draw)",
    ("grid", "prior")    => "renamed `[grid.smoothness] type` (the components' one smoothness vocabulary): type = \"pspline\", order = 3, weight = …",
    ("grid", "prior_weight") => "renamed `[grid.smoothness] weight` (the components' one smoothness vocabulary): a number β ≥ 0 or \"gcv\"",
    ("vignetting", "smoothness_weight") => "renamed `[vignetting.smoothness] weight` (the components' one smoothness vocabulary): type = \"pspline\", order = 3, weight = …",
    ("focus", "prior")   => "renamed `[focus.smoothness] weight` (the components' one smoothness vocabulary): type = \"bending\", weight = …",
    ("lm", "fix")        => "renamed `fixed`, the options' top-level list (the stack's one parameter-holding keyword; no alias)",
    ("vignetting", "size") => "the tensor-product vignetting surface is no option of the pipeline (a refinement fits the radial " *
                            "profile about the principal point, `vignetting.node_count`); delete the key",
    ("edge", "free")     => "the printed edge is never a parameter (its δ and σ_print are held at `edge.ink_spread_mm` and " *
                            "`edge.print_sigma_mm`); delete the key",
    ("refine", "weighting") => "weighting follows the camera's sensor model: give a characterisation with a gain to weight " *
                            "(`camera.sensor_characterization`; no flag); delete the key",
    ("camera", "image_circle_min_hole_area_px") => "the image circle is measured on the rim's log-polar profiles (VisualGeometryToolkitFeatures' " *
                            "`detect_image_circle`), with no dark-region hull and no hole to drop; delete the key",
    ("camera", "image_circle_frames") => "the image region is a step of its own, per capture set: its record " *
                            "(`measure_image_region` → `save_image_region`, rc/image_region/v1.0) is named by `camera.lens_coverage`, or the " *
                            "lens lights the whole sensor: `camera.lens_coverage = \"sensor\"`",
    ("camera", "image_circle_outlier_halfwidth_px") => "a setting of the image-region step (`measure_image_region`'s `outlier_halfwidth_px`), " *
                            "not of a run: the run reads the step's record through `camera.lens_coverage`; delete the key",
    ("frontend", "noncentral") => "the non-central spline model is removed (the paper no longer uses it): every camera is central; a Schöps " *
                            "central-generic start is `frontend.schoeps`; delete the key",
    # the inert keys refused, the held blocks one top-level list
    ("lm", "fixed")      => "moved to the options' top-level `fixed` (the stack's one parameter-holding keyword): `fixed = [\"focus\", …]` " *
                            "beside the tables; \"grid\" is renamed \"camera\", \"edge\" is refused (never a parameter)",
    ("camera", "r_min")  => "retired (inert: no bound of r since 2026-10-31); a held aperture is `fixed = [\"aperture\"]` (formerly " *
                            "r_min == r_max); delete the key",
    ("camera", "r_max")  => "retired (inert: no bound of r since 2026-10-31); a held aperture is `fixed = [\"aperture\"]` (formerly " *
                            "r_min == r_max); delete the key",
    ("rendering", "psf_sigma") => "retired (inert since 2026-11-14: the camera's in-focus blur is removed, the sensor rule is the pixel box); delete the key",
    ("edge", "camera_sigma_px") => "retired (inert since 2026-11-14: the printed edge's width is `edge.print_sigma_mm`); delete the key",
    ("grid", "size")     => "retired, never ignored (on the exact-start route an earlier release silently did not read it): " *
                            "state the lattice's mode `grid.lattice` — \"derived\" for a Schöps start (his own lattice, m = 1) or " *
                            "\"node_count\" with `grid.node_count` for a parametric start (the nodes its image region supports: 105 × 75 is " *
                            "node_count = 7875, 35 × 25 is 875)",
    ("grid", "lattice_spacing") => "retired: a Schöps start is never subdivided — state `grid.lattice = \"derived\"` " *
                            "(his own lattice extended over the lens coverage, m = 1); a parametric start states `grid.lattice = \"node_count\"` " *
                            "and `grid.node_count`",
    ("warping", "points") => "retired: a board-warp points file was read by a hand parser (every input is read through VisualGeometryToolkitIO) and no " *
                            "config of record named one; start the nodal heights with `warping.heights_mm`",
    # ONE word for a spline's free values, "node", and ONE key for how many, `node_count`
    ("vignetting", "coefficients") => "renamed `[vignetting] node_count` (the stack's one word for a spline's free values, node): node_count = 16",
    ("focus", "nodes")   => _focus_ab_key_removed(" (the key was a direction chart's node count); delete the key"),
    ("lm", "xtol")       => "renamed `[lm] xtol_rel` (VisualGeometryToolkitCore's relative step test, Ceres' parameter " *
                            "tolerance, under VisualGeometryToolkitCore's name; the absolute step test is VisualGeometryToolkitCore's of record, no key): xtol_rel = 1.0e-8",
    ("focus", "node_rho_max") => "renamed `[focus] lens_coverage_scale` (the same scaling of the lens coverage about its centre as " *
                            "`[masks] lens_coverage_scale`, `_scaled_coverage`): lens_coverage_scale = 0.9",
    # the focus A/B is decided (the sensor-grid thin plate spline fits best): its keys are removed
    ("focus", "variant") => _focus_ab_key_removed(" (the former variant \"A\"); delete the key"),
    ("focus", "node_count") => _focus_ab_key_removed(", its nodes the 5 × 5 sensor grid's inside the lens coverage " *
                            "(`focus.lens_coverage_scale`, `focus.support_min`); delete the key"),
    ("focus", "perturb") => "removed with the focus A/B experiment (a perturbed focus start was its robustness test); delete the key")

# The stack's switch convention: a switch of a config is a string — "on" | "off" — and TOML's true/false
# is refused wherever a switch is read (`read_switch`), naming the new spelling: no alias, no silent reading of an old boolean.
"""The message refusing the retired boolean spelling `v` of the switch `key`: its new spelling named, `"on"` or `"off"`."""
retired_boolean_switch(key::AbstractString, v::Bool) =
    "`$key = $v` is retired (a switch is the string \"on\" or \"off\", TOML true/false no longer accepted): use `$key = \"$(switch_spelling(v))\"`"
