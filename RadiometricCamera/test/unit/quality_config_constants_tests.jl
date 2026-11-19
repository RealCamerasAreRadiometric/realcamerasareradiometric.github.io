# The no-configuring-constant rule (owner, 2026-11-16: "NO global constants in code that configure an experiment or a run.
# Experiments are driven by their TOMLs, so the configuration must be visible there."): a mode, a threshold, a tolerance that changes
# a result, a scoring type, an iteration cap, a sample size, a seed or a setting's default is a REQUIRED key of the config of record
# (RC's [pipeline] tables, configs/default.toml; a experiments driver's configs/experiments/*.toml), never a `const`. This item scans RC's
# src/ and the paper project's src/ and scripts/ for a `const` whose right-hand side LOOKS like a setting — a number, a string, a
# symbol, a Bool, a literal tuple/vector of numbers, a qualified zero-argument constructor (a strategy object) — and fails on any name
# that is not in the allow-list below: schema and vocabulary (file names, driver paths, retired-value spellings), parsed CLI flags,
# model structure fixed by static array sizes, numerical machinery no experiment varies, each with its reason. The list only shrinks;
# an entry that matches nothing is stale. DEFERRED names belong to files another branch rewrites (they are classified, not locked).

@testitem "quality: no const configures a run" begin
    using Test, RadiometricCamera
    ALLOWED = Dict(
        "CORE_LM_XTOL" => "Core's absolute step tolerance of record (LmStops().xtol, no literal): RC 0.11.4's ruling — the absolute [lm] xtol key refused, xtol_rel the run's",
        "ROR_PREFIX" => "a job-name prefix (the run of record's Slurm names), no setting",
        "VIEW_MIN_DOTS" => "model structure: a pose's minimal correspondence set (P3P, 3 dots), the refusal of a photograph that cannot be fitted",
        # ---- schema: file and directory names, record texts ----
        "INITIALIZATION_REGION_FILE" => "a record's file name",
        "START_RECORD_FILE" => "a record's file name",
        "START_MASKS_DIR" => "a record's directory name",
        "PIPELINE_CONFIG_FILE" => "a record's file name",
        "_NO_OBSERVATION_MODEL" => "the provenance text a pre-0.11.1 record is read with, no setting",
        "IMAGE_REGION_FILE" => "a record's file name",
        "DISTANCE_MAP_FILE" => "a record's file name (the distance-map step's raster record)",
        "DISTANCE_MAPS_DRIVER" => "a driver path (scripts/distance_maps.jl, the distance-map step's maker)",
        "WARPING_RECORD_FILE" => "a record's file name",
        "WARPING_TABLE_FILE" => "a table's file name",
        "CONVERGENCE_RECORD_FILE" => "a record's file name",
        "TIMING_REPLICATE_FILE" => "a record's file name", "TIMING_RECORD_FILE" => "a record's file name",
        "PAIRED_FILE" => "a table's file name",
        "LFS_POINTER_HEAD" => "the git-lfs pointer header a stub file is recognised by (git-lfs's format)",
        "INPUT_PROVENANCE" => "a provenance text of calibrate.jl's records",
        "USAGE" => "a script's usage text",
        # ---- vocabulary: the spellings of legal or retired values ----
        "NO_SENSOR_CHARACTERISATION" => "the retired value's spelling, refused by name",
        "SENSOR_LENS_COVERAGE" => "the spelling of `camera.lens_coverage`'s whole-sensor value (the TOML states it)",
        "RETIRED_IMAGE_CIRCLE_INPUTS" => "the retired input's path, refused by name",
        "BOUNDARY_ORDERS" => "the legal values of `grid.boundary_order` / `vignetting.boundary_order` (the TOML picks one)",
        "FOCUS_SENSOR" => "an enumeration value of the focus chart kinds",
        "FOCUS_DIRECTION_CHART" => "an enumeration value of the focus chart kinds",
        "FOCUS_SPHERE" => "an enumeration value of the focus chart kinds",
        # ---- driver paths and parsed CLI flags ----
        "DRIVER" => "a driver's own path (its records' provenance)",
        "IMAGE_REGIONS_DRIVER" => "a driver's path",
        "REFINEMENT_DRIVER" => "a driver's path",
        "CHROME_EVALUATION_DRIVER" => "a driver's path",
        "TEXTURE_STAGES" => "the stage list a driver hands the texture driver (which stages run, not how)",
        "RESULTS_BRANCH" => "a parsed CLI flag (where the renders are deployed)",
        "GPU" => "a parsed CLI flag (whether the GPU stages run here)",
        "FORCE" => "a parsed CLI flag (recompute a keyed product)",
        # ---- model structure: static array sizes, gauge slices ----
        "EDGE_NC" => "the printed edge's parameter count (δ, σ_print): the model's structure, a static size",
        "INPLANE_NC" => "the in-plane print error's parameter count (aspect, skew): the model's structure, a static size",
        "HEIGHT_N" => "the warping height field's node grid per page axis: static sizes (HEIGHT_NC, HEIGHT_NFREE) of the board model (judged structure, 2026-11-16)",
        "RAD_NL" => "the radiance Jacobian's column count (the lighting parameters): a static size",
        "GRID_PATCH" => "the grid route's static patch block size: a larger block explodes codegen (judged structure, 2026-11-16)",
        "_NO_ROTATION_GAUGE" => "the empty gauge slice (no held coordinate), a sentinel",
        # ---- numerical machinery no experiment varies ----
        "EDGE_EPS_MM" => "the floor of the edge blur σ_c that keeps a sharp edge's derivative finite (a guard, not a setting)",
        "WINDOW_MEMORY_FOOTPRINT" => "the device scheduler's memory estimate: how many blocks are in flight, never what they compute",
        "LM_DAMPING" => "Levenberg's λ·I: the owner's ruling 2026-11-14 'one value, no option' (the retired key lm.iterator names it)",
        "_SAME_POSE_RTOL" => "a numerical identity: whether two representations of one pose are the same map (~4500 ulps), not a run setting")
    DEFERRED = Dict(
        "FRONTEND_DECODE_MODE" => "experiments/src/frontends.jl, frontends_runs.jl: removed by the schoeps-window branch",
        "SCHOEPS_BINARY" => "experiments/src/frontends.jl: a machine path, the schoeps-window branch's",
        "DOUBLE_SPHERE_STARTS" => "experiments/src/parametric_starts.jl: the Double Sphere multi-start; its config of record is frontends.toml (schoeps-window branch)")
    root = pkgdir(RadiometricCamera)
    # the package, and the experiments beside it in the repository (absent from an installed package)
    dirs = filter(isdir, [joinpath(root, d) for d in ("src", joinpath("..", "experiments", "src"), joinpath("..", "experiments", "scripts"))])
    re = r"^\s*const\s+([A-Za-z_][A-Za-z_0-9]*)\s*(?:::[^=]+)?=\s*(?:[-+]?\.?\d|\"|:[A-Za-z_]|true\b|false\b|[\(\[]+\s*[-+]?\.?\d|(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\(\s*\))"
    found = [(relpath(f, root), i, m[1]) for dir in dirs for (d, _, fs) in walkdir(dir) for n in fs if endswith(n, ".jl")
             for f in (joinpath(d, n),) for (i, l) in enumerate(eachline(f)) for m in (match(re, l),) if m !== nothing]
    bad = filter(h -> !haskey(ALLOWED, h[3]) && !haskey(DEFERRED, h[3]), found)
    @test isempty(bad)
    isempty(bad) || foreach(h -> println("a const that looks like a run setting (make it a required TOML key, or allow-list it with its reason): ",
                                         h[1], ":", h[2], " ", h[3]), bad)
    @test all(n -> any(h -> h[3] == n, found), keys(ALLOWED))        # an allow-list entry that matches nothing is stale
end
