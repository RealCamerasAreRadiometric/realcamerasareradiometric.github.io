# ---------------------------------------------------------------------------
# The start's state (paper §Calibration Pipeline): from a geometric calibration — a StartGeometry, the poses Θ_e, a geometric camera π⁻¹,
# the observations of the target (a BlobBoards session, a Schöps central-generic run, a Kannala–Brandt or OpenCV fit, start/frontends/) —
# and the photographs ϕ to the started radiometric state: the focus plane through the mean target centre (s′ from the thin-lens
# equation, for the log), r from the f-number, Θ_m, V flat, the masks (the target's validity ∩ the imaged region ∩ a random pixel
# subset), the streamed observations ϕ, L^SVSH by the closed-form Θ_l solve and one closed-form sweep of Θ_l, Θ_v, μ, the grid
# smoothness (`_start_state`, behind a `RadiometricCalibration`'s `init` and `load_calibration_start`, start/calibration_start.jl). The refinement is
# the `RadiometricCalibration` solve (calibration/calibrate.jl), its records `save_radiometric_calibration` (io/run_outputs.jl). Every file is read
# and written through VisualGeometryToolkitIO: the records of Core's values and RC's (records.jl), the reports as JSON documents
# (`write_record` / `read_record`).
# ---------------------------------------------------------------------------

# A start record's value `k` (`fpx`, `f_m`, `d_c`, `be0`), checked against the value recomputed from its stored lens and poses: the same
# to the record's rounding (its lengths and rotations through IO: 1e-6 relative, 1e-6 absolute), else the record is of
# another front end or code — refused by name. The record's value is the one taken (the computed start's, bit for bit).
function _recorded(start, k::Symbol, x)
    r = getfield(start, k)
    size(r) == size(x) && isapprox(r, x; rtol = 1e-6, atol = 1e-6) ||
        throw(ArgumentError("the start record $(start.dir): its $k ($(k === :be0 ? "$(length(r)) bearing errors" : r)) is not its stored lens and poses' " *
                            "($(k === :be0 ? "$(length(x))" : x)): a start of another front end or code — produce its start again"))
    return r
end

# A start record's poses of the views `names` (its state's, camera → world).
function _stored_poses(rec, names)
    names == rec.state.names || throw(ArgumentError("the start record $(rec.dir) holds the views $(rec.state.names), the run $(names)"))
    return rec.state.poses
end

# A state written under another region (e.g. the whole sensor before the image circle was measured) resumed under the run's region
# (2026-10-30): the view's grid lens takes the run's lens coverage (what the run's interior nodes, smoothness and in_domain read), its
# values the state's.
function _with_run_region(lens, run_lens)
    is_grid_lens(lens) && is_grid_lens(run_lens) ||
        throw(ArgumentError("state: the state's lens $(nameof(typeof(lens))) does not match the run's $(nameof(typeof(run_lens)))"))
    return VgtkCore.with_lens_coverage(lens, VgtkCore.lens_coverage(run_lens))
end

# ---- prepare -------------------------------------------------------------------------------------------------------------------
"""
    start_is_grid(run) -> Bool

Whether the start of the run `run` is a grid camera (a bounded SplineLens camera), refined per tile,
rather than a parametric one refined in its own parameters: every start is a grid but a parametric record's (`io.init = "lens:…"`)
— the blob front end's session camera and the Schöps front end's are resampled into the grid. THE decision of the start's kind:
the start takes it (and refuses a front end whose start disagrees); its start record holds `grid.json` beside its
`initialization_region.json`, else `lens.json`.
"""
start_is_grid(run::RunOptions) = !startswith(run.io.init, "lens:")

function _start_state(g::StartGeometry, photographs::AbstractVector{<:AbstractString}, run::RunOptions, settings::CalibrationSettings, start; backend, radiometry,
                      observations = nothing)
    isempty(run.io.out) && error("io.out required")
    mkpath(run.io.out)
    t_start = time()
    R0 = VgtkCore.ustrip(VgtkCore.m, _focal_length(run)) / (2 * run.camera.fnumber)
    height0 = _height_start(settings)
    inplane0 = _inplane_start(settings)
    names = g.names; K = length(names); W₀, H₀ = g.image_size; lens0 = g.camera; poses0 = g.poses; features = g.observations
    length(photographs) == K || throw(ArgumentError("$(length(photographs)) photographs for the $(K) views $(names)"))
    start === nothing || (W₀, H₀) == (start.W₀, start.H₀) ||
        throw(ArgumentError("the start record $(start.dir) is of $(start.W₀) × $(start.H₀) photographs, the front end's $(W₀) × $(H₀)"))
    fpx = start === nothing ? fpx_of(lens0, W₀, H₀) : _recorded(start, :fpx, fpx_of(lens0, W₀, H₀))
    route = lm_route(lens0)                                # the refinement's route, by the start camera's type (a parametric camera: per view, _lm_views!)
    is_grid_lens(lens0) == start_is_grid(run) || error("io.init = $(repr(run.io.init)): the front end's start is a $(_route_label(route)) " *
                                                        "camera, but start_is_grid says otherwise (one decision of the start's kind: fix start_is_grid)")
    _print_start_route(route, lens0)
    be0 = _start_bearings(g, fpx)                                            # on the initialization region
    be0 = start === nothing ? be0 : _recorded(start, :be0, be0)
    @printf("SELF-CHECK: %d detections; bearing error through the start model under the start poses: median %.3f px, p90 %.3f px (expect ≈ the pipeline's own residual)\n",
            length(be0), _med(be0), _q90(be0)); flush(stdout)
    _med(be0) > 2.0 && @warn "bearing self-check > 2 px: suspect a pixel/board convention error" median = median(be0)

    # focus / sensor start: the focus surface is a plane parallel to the sensor through the MEAN BOARD CENTRE (the axial distance d_c of the
    # board centre averaged over the views); the image distance s′ (no parameter of the model) is derived from the thin-lens equation of
    # d_c for the log and parameters.json
    sensor = run_sensor(run, W₀, H₀)                            # Core's Sensor: the image's pixels, the pitch of the sensor width
    f_m = start === nothing ? fpx * _pixel_pitch(sensor) : _recorded(start, :f_m, fpx * _pixel_pitch(sensor))   # the focal length (m)
    board = g.target
    𝐛_c = 𝔟(board)
    d_c = start === nothing ? mean(dot(p.R[:, 3], 𝐛_c - p.t) for p in poses0) : _recorded(start, :d_c, mean(dot(p.R[:, 3], 𝐛_c - p.t) for p in poses0))
    d_s = f_m * d_c / (d_c - f_m)
    # the run's one support: Core's imaged region of the lens whose lens coverage is the capture set's, as `camera.lens_coverage` states
    # it (run_lens_coverage: the sensor, or the set's measured image-region record), which a grid's interior nodes, its smoothness and
    # in_domain read
    start === nothing && (lens0 = VgtkCore.with_lens_coverage(lens0, run_lens_coverage(run, W₀, H₀)))   # a start record's lens holds it
    lens_start = lens0                                            # the start lens with its image circle: the start record's
    region = VgtkCore.imaged_region(lens0)                        # the refinement's region (its masks, present nodes, grid smoothness and focus nodes)
    coverage = VgtkCore.lens_coverage(lens0)                     # the image circle (an Ellipse), or the sensor's rect
    circ = coverage isa VgtkCore.Ellipse ? coverage : nothing
    vig_c = start !== nothing ? start.vig_c : vignetting_center(lens0)   # the radial profile's centre: the start's principal point (a SplineLens start's the axis)
    circ === nothing ? println("imaged region: the whole sensor") :
                       @printf("imaged region: the sensor ∩ the image circle, centre (%.1f, %.1f), semi-axes %.1f / %.1f px\n", circ.center..., circ.a, circ.b)
    # the focus nodes: the 5 × 5 grid over the IMAGED region's bounding box (never the initialization region's), kept where the START
    # lens images them inside the region (`in_support`, fixed once for the run) and inside the lens coverage scaled by `focus.lens_coverage_scale` (an image
    # circle's semi-axes, `_scaled_coverage`: no node on the rim; `focus_start_nodes`); the TPS is built on these alone. `focus_m` and the monitor average
    # the support nodes.
    (; in_support, nodes_box, nodes, box) = focus_start_nodes(lens0, settings.focus)
    focus0 = Focus(FocusSpline(nodes), focus_start(lens0, nodes, d_c))
    @printf("focus start: %d of the %d grid nodes over the imaged region's box [%.0f, %.0f] × [%.0f, %.0f] px are in the support (imaged by the start lens, inside the region%s) and are the TPS nodes; %d start values floored at 0\n",
            length(nodes), length(nodes_box), box.origin[1], box.origin[1] + box.widths[1], box.origin[2], box.origin[2] + box.widths[2],
            circ === nothing ? "" : @sprintf(", inside the image circle scaled by %.2f", settings.focus.lens_coverage_scale), count(iszero, focus0.t̂_i))
    views0 = [View(poses0[k], lens0, focus0, R0) for k in 1:K]
    @printf("f %.2f mm (%.1f px), sensor %.1f mm, s′ %.3f mm (derived, thin lens), focus plane through the mean board centre %.3f m, aperture r₀ %.3f mm (f/%.1f); backend %s\n",
            _mm(f_m), fpx, VgtkCore.ustrip(VgtkCore.mm, _sensor_width(run)), _mm(d_s), d_c, _mm(R0), run.camera.fnumber, nameof(typeof(backend))); flush(stdout)

    # board (+ the printed edge), settings, masks, streamed observations
    if !isempty(settings.edge.sdf)
        sdf, page_mm = load_distance_map(settings.edge.sdf)
        board = with_sdf(board, sdf, page_mm)
        @printf("printed edge: distance map %s (%d × %d texels, %.0f × %.0f mm); δ %.4f mm, σ_print %.4f mm (held)\n",
                settings.edge.sdf, size(sdf, 2), size(sdf, 1), page_mm..., _edge_start(settings)...)
        sdf = nothing
    end
    surface0 = TargetState(; h = height0, edge = isempty(settings.edge.sdf) ? nothing : collect(_edge_start(settings)), inplane = inplane0)
    board = with_state(board, surface0)
    Tuple(VgtkCore.extent(lens0)) == (W₀, H₀) ||   # the image is the camera's: the run's size and the start camera's are one
        throw(ArgumentError("the start's camera is of a $(join(Tuple(VgtkCore.extent(lens0)), " × ")) image, the run's photographs are $(W₀) × $(H₀)"))
    st = RenderSettings(; n_𝐩 = settings.rendering.n_p, n_ρ = settings.rendering.n_rho, n_φ = settings.rendering.n_phi, N = settings.rendering.N,
                        hammersley = settings.rendering.hammersley)
    settings.rendering.hammersley && @printf("radiance samples: the Hammersley set of %d, shifted per (view, pixel, sub-pixel, aperture) key\n", settings.rendering.N)
    b = 𝔟(board); r_SH = sphere_radius(board, [v.pose.t for v in views0])          # ≥ the page's corners too (2026-11-04)
    @printf("light sphere: r_SH %.3f m about the board centre (farthest camera %.3f m, farthest page corner %.3f m)\n", r_SH,
            sphere_radius(b, [v.pose.t for v in views0]), maximum(norm(c - b) for c in corners(board)))
    rng = StableRNG(settings.masks.seed)                                            # version-stable stream (_subsample)
    inreg = VgtkCore.pixel_mask(_sampling_region(lens0, settings.masks.lens_coverage_scale), W₀, H₀)   # the pixels sampled: the coverage scaled
    circ === nothing || @printf("imaged region: %d of %d pixels inside\n", count(inreg), length(inreg))
    mo = settings.masks                                                             # the draw (`draw_samples`): ∝ u^α of the edge strength's rank (0 = uniform),
    α = mo.gradient_bias                                                        #   or the rule of 2026-10-31 (edge_fraction), with an optional add-on
    N = mo.samples_per_view                                                     # the samples per view, the same for every camera (`draw_samples`)
    # the one pixel set of the run: the LM's, and the closed-form start's (2026-10-31). The usable observations are the pixels of the
    # sampling region that the photograph's observation model keeps (`observe`: the green photosites, no defect, uncensored) and that
    # are valid under the mask test of the start view (`BoardValidity`: the board-valid pixels, one kernel on the refinement's device), and
    # `samples_per_view` of THOSE are drawn per view (owner, 2026-11-15: conditioned on the usable observations, so a small image circle
    # and a lens filling the sensor get the same count; `draw_samples` tests the candidates it needs, in its keys' order).
    # A start record's masks are its final ones (the drawn samples): read, not drawn
    mask_views = views0                                                         # the mask test's views: the unperturbed start's
    if start === nothing
        masks = [inreg for _ in 1:K]                                            # the candidates, before the observation model
    else
        masks = copy(start.masks)
        all(m -> size(m) == (H₀, W₀), masks) || throw(ArgumentError("the start record $(start.dir): its masks are not the $(W₀) × $(H₀) photographs'"))
    end
    # the start perturbed (`grid.start_perturbation`; perturb_start, per_tile_smoothness.jl): after the masks, which are the unperturbed
    # start's, and before the views, the present nodes and the smoothness, whose reference d̂₀ is so the PERTURBED start
    sp = settings.grid.start_perturbation
    lens0 = perturb_start(lens0, sp, fpx)
    sp === nothing || @printf("grid start PERTURBED: every direction control rotated by σ %.3g px / f %.1f px = %.3g mrad RMS per tangent axis (seed %d)\n",
                              sp.sigma_px, fpx, 1e3 * sp.sigma_px / fpx, sp.seed)
    # the grid's nodes under the imaged region (per_tile_smoothness.jl): the lens holds those the region reaches, the parameters of the run
    grid_held = _present_nodes(route, lens0, settings.grid.nodes)                  # nothing per view (no lattice)
    grid_held === nothing || (lens0 = VgtkCore.with_present(lens0, grid_held))
    views0 = [View(poses0[k], lens0, focus0, R0) for k in 1:K]
    px = Vector{Vector{Tuple{Int, Int}}}(undef, K)                              # each view's samples, set below (observed or loaded)
    ϕ = Vector{Vector{Float64}}(undef, K); ω = Vector{Any}(undef, K); qs = Vector{Any}(undef, K)          # the measured values and their rows' weights (weighting.jl)
    ch = _io_channel(run.io.channel)                                           # 0: RGB; c: channel c alone, one residual per pixel
    ch == 0 || @printf("photographs: the %s channel alone, one residual per measured pixel (io.channel)\n", run.io.channel)
    # the photographs' observation model (observation_model.jl: the sensor statement of `camera.sensor_characterization`, RAW photosites
    # or decoded images, the channel) and each photograph's sensor model through THE entry point (`photo_set`: Core's resolver by its
    # EXIF; the absent model for a decoded image): its black and defects, and its reason, recorded per photograph (`sensor_record`)
    # `camera.sensor_model` (parsed once): whether a record present — the named one, or the start record's statement — is used
    files = photographs
    smode = sensor_model_mode(run.camera.sensor_model)
    stated = start === nothing ? NoSensorRecord() : start.state.observation.sensor
    om = observation_model(files, run.camera.sensor_characterization; run.io.linearize, channel = run.io.channel, sensor_model = smode, stated)
    sensor_model = sensor_model_record(smode, run.camera.sensor_characterization, stated, om)
    pset = observations === nothing ? photo_set(om, files; characterization = _characterization_path(run.camera.sensor_characterization, om),
                                                lens = run.io.lens_identity === nothing ? nothing : lens_identity(run.io.lens_identity)) : nothing
    sensor = observations !== nothing ? observations.sensor : (pset::PhotoSet).models
    any(is_raw_file, files) && println("sensor models (", om.sensor isa SensorRecord ? run.camera.sensor_characterization : "none used", "; sensor_model = ", sensor_model.mode, "): ",
                                       join(["$(n) × $(r)" for (r, n) in sort!(collect(pairs(_count_reasons(sensor))); by = string ∘ first)], ", "))
    if observations !== nothing                                                 # the start's own streamed observations (`load_calibration_start`):
        observations.masks == masks || throw(ArgumentError("the start's observations are of other masks than its record's"))   # no photograph read again
        for k in 1:K
            px[k] = masked_pixels(masks[k]); ϕ[k], ω[k], qs[k] = observations.ϕ[k], observations.ω[k], observations.qs[k]
        end
    end
    photosites = om.kind isa Photosites                                         # the rule of 2026-10-31 reads a RAW photograph's diagonal neighbours
    photosites && om.channel !== :G && mo.edge_fraction > 0 &&                  # (a red or blue photosite's same-filter neighbours are two photosites along the axes)
        throw(ArgumentError("`masks.edge_fraction` > 0 selects a RAW photograph's photosites by their diagonal neighbours, which are of the " *
                            "same colour filter for the green photosites only: not with `io.channel = \"$(om.channel)\"` (the weighted draw, " *
                            "`masks.gradient_bias` with `masks.edge_detector`, reads the photosites of any one filter)"))
    for k in (observations === nothing ? (1:K) : (1:0))                         # one decode per photo: the edge selection, the cap, the observations
        # THE entry point (`observe`): what the photograph contributes, by the observation model's TYPE — a RAW mosaic the photosites of its
        # channel's colour filter on the raw grid (uncensored, defects masked), a decoded image its pixels — never a flag (owner ruling 2026-11-13)
        img, masks[k] = observe(pset::PhotoSet, k, masks[k]; name = names[k])   # the candidates the model keeps (a start record's: its samples)
        chs = ndims(img) == 2 ? (1,) : _residual_channels(ch)                   # the edge selection reads the measured channels
        start === nothing && (masks[k] = draw_samples(rng, BoardValidity(mask_views[k], board, st; δ = 0.01, backend), masks[k], img, N;
                                                      gradient_bias = α, edge_detector = mo.edge_detector, mo.edge_fraction, mo.addon_samples,
                                                      mo.addon_gradient_bias, channels = chs, diagonal = photosites, device = backend))
        px[k] = masked_pixels(masks[k])
        ϕ[k], ω[k], qs[k] = masked_observations(img, px[k]; channel = ch); img = nothing; GC.gc()
    end
    npx = sum(length.(px))
    samples = sample_record(N + mo.addon_samples, masks)                       # the masks' record: a short view uses all it has, never padded
    mo.edge_fraction > 0 && @printf("edge-selective samples (the rule of 2026-10-31): the strongest %.0f %% by the photo's |∇ϕ|² of a uniform pool of the usable observations\n", 100mo.edge_fraction)
    iszero(α) || @printf("edge-biased samples: drawn ∝ u^%g of the photograph's edge-strength rank (%s)\n", α, edge_detector_table(mo.edge_detector)["type"])
    mo.addon_samples > 0 && @printf("add-on samples: %d per view more, drawn ∝ u^%g of the edge-strength rank from the other usable observations\n", mo.addon_samples, mo.addon_gradient_bias)
    @printf("masks: %d samples (%d per view%s); photos streamed in %.0f s\n", npx, N + mo.addon_samples,
            any(>(0), samples.shortfall) ? @sprintf("; %d views short of it by %s", count(>(0), samples.shortfall), join(filter(>(0), samples.shortfall), ", ")) : "",
            time() - t_start); flush(stdout)
    any(isempty, px) && @warn "views without masked pixels" views = names[isempty.(px)]
    all(isempty, px) && error("no masked pixels at all: the board is not inside the views (pose/board frame error?) — check the SELF-CHECK line")
    # the focus nodes' data support (the vignetting's rule, `node_support`): a node with less than `focus.support_min` of the mean data
    # weight is dependent — the TPS is rebuilt on the supported nodes, which continues the surface over it
    focus_support = nothing
    if settings.focus.support_min > 0
        fsup = focus_node_support(focus0, px, settings.focus.support_min)
        @printf("focus support: %d of %d nodes carry ≥ %g of the mean data weight (relative weights %s)%s\n", count(fsup.supported), length(fsup.supported),
                settings.focus.support_min, join([@sprintf("%.2g", x) for x in fsup.weight], " "),
                all(fsup.supported) ? "" : "; the others are dependent: the TPS is built on the supported nodes"); flush(stdout)
        dropped = [collect(n) for (n, ok) in zip(focus0.nodes, fsup.supported) if !ok]
        if !all(fsup.supported)
            nodes = _check_focus_nodes(focus0.nodes[fsup.supported], lens0)
            focus0 = Focus(FocusSpline(nodes), focus_start(lens0, nodes, d_c))
            views0 = [View(poses0[k], lens0, focus0, R0) for k in 1:K]
            fsup = focus_node_support(focus0, px, settings.focus.support_min)      # the record describes the REBUILT node set (owner 2026-11-16)
        end
        # one entry per node of the run's focus (weight, support); the candidates the rule dropped, by position
        focus_support = (; weight = fsup.weight, supported = collect(fsup.supported), support_min = settings.focus.support_min,
                         dropped = length(dropped), dropped_nodes = dropped)
    end
    empty_targets = [Array{Float64, 3}(undef, 0, 0, 0) for _ in 1:K]
    material0 = Material(; roughness_ink = settings.material.alpha_ink, roughness_paper = settings.material.alpha_noink, refractive_index_ink = settings.material.eta_ink, refractive_index_paper = settings.material.eta_noink, diffuse_weight = settings.material.mu)
    light0 = EnvironmentLight(SphericalHarmonicMap(1.0, settings.lighting.sh_order), b, r_SH)   # L ≡ 1 in every channel
    make_vig, vig_desc = _vignetting_spec(settings, W₀, H₀, coverage, vig_c)
    println("vignetting field: ", vig_desc, "; gauge V(𝐜) = 1 held ", "exactly (the centre's free direction eliminated)"); flush(stdout)
    camera0 = RadiometricCameraModel(lens0, focus0, R0, make_vig()); poses_start = [v.pose for v in views0]
    f = CalibrationData(board, st, camera0, poses_start, empty_targets;
                         masks, ϕ, backend, region, channel = ch,
                         sensor_models = sensor, sensor = om.sensor, photo_kind = om.kind, weights = _run_weights(ω), quanta = qs,
                         support_min = settings.vignetting.support_min, boundary_order = settings.vignetting.boundary_order,
                         mu_bounds = settings.material.mu_bounds, jacobian_chunk = run.compute.jacobian_chunk)
    θ = CalibrationParameters(f, camera0, poses_start, material0, light0)
    f.vig_rule isa RadialSupportRule && @printf("vignetting support: %d of %d nodes are parameters — the pole pair and those carrying ≥ %g of the mean data weight (relative weights %s); the others continue them with a zero Δ^%d across the border (Core's BoundaryRule, vignetting.boundary_order)\n",
                                                count(f.vig_rule.supported), length(f.vig_rule.supported), settings.vignetting.support_min,
                                                join([@sprintf("%.2g", x) for x in f.vig_rule.weight], " "), f.vig_rule.rule.order)
    focus_row_weight = focus_smoothness!(f, focus0, data_norm2(f), settings.focus.smoothness)   # β Σϕ²/N: the run record's (`_run_parameters`)
    # the grid's smoothness over the present nodes (per_tile_smoothness.jl): fixed at the start, its reference the start grid. A weight
    # chosen by GCV (`[grid.smoothness] weight = "gcv"`) is chosen after the closed-form start of the radiometry, on the linearization
    # there (`select_grid_smoothness_weight`)
    grid_int = grid_held === nothing ? nothing : interior_nodes(grid_held)
    grid_gcv = _grid_gcv(run, settings)
    grid_sm = grid_held === nothing || grid_gcv ? nothing : grid_smoothness(settings.grid.smoothness, lens0, data_norm2(f), grid_int)
    grid_held === nothing || @printf("grid: %d of %d nodes present (the parameters; %d absent); smoothness %s%s\n", count(grid_int), length(grid_int),
                          length(grid_int) - count(grid_int), smoothness_line(settings.grid.smoothness),
                          grid_gcv ? ", by GCV after the closed-form start" :
                          grid_sm === nothing ? "" : @sprintf(" (row weight %.4g, m̄ %.4g, %d rows)", grid_sm.w, grid_sm.m̄, 3 * length(grid_sm.rows)))
    meta = (; focus_support, focus_row_weight, grid_int, grid_smoothness = grid_sm, grid_gcv = nothing, vignetting_gcv = NamedTuple[], region, coverage, init_region = g.init_region, resample = start === nothing ? g.resample : start.resample, names, views0, lens0, lens_start, fpx, in_support, features, features_for = g.features_for, W₀, H₀, be0, world = g.world, page_flip = g.page_flip,
            board_y_sign = g.board_y_sign, vig_c, make_vig, vig_desc, height0, inplane0, npx, samples, R0, d_c, f_m, t_start,
            sensor_model, observations = start === nothing ? (; masks = copy(masks), ϕ, ω, qs, sensor) : nothing)   # a start's streamed photographs (`load_calibration_start`)
    start === nothing || return _from_start(f, θ, meta, start, run, settings)   # a start record: its closed-form start and GCV weights loaded, not computed
    # the radiometric start, on the LM's own pixels with the parts rendered once (decision 2026-10-31; before: a separate sparse subset,
    # `masks.linear_fraction`, and the sweep kept only if it lowered the cost on the LM's pixels — a test that a least-squares solve on
    # the same pixels cannot fail): the closed-form Θ_l solve sets the lighting's scale (no scalar light gain), then ONE closed-form
    # sweep of Θ_l, Θ_v, μ. Every solve is exact on these pixels, so the cost cannot rise.
    parts = all_parts(f, θ)
    _update_radiometry!(f, θ, RadiometryBlockDescent(), (; light = true, vignetting = false, mu = false); parts)
    c0, rms0 = cost(f, θ)
    @printf("lighting solve; init cost %.4e rel-RMS %.4f\n", c0, rms0); flush(stdout)
    t1 = time()
    _update_radiometry!(f, θ, _with_schedule(radiometry; sweeps = 1, vignetting_smoothness = smoothness_weight(settings.vignetting.smoothness), verbose = false),
                        (; light = true, vignetting = true, mu = !("material" in settings.fixed)); parts, gcv_record = meta.vignetting_gcv)
    parts = nothing; GC.gc()
    c1, rms1 = cost(f, θ); μ_lin = θ.mat.diffuse_weight
    @printf("linear sweep (%.0f s): cost %.4e rel-RMS %.4f; μ %.3f; V centre %.3f\n", time() - t1, c1, rms1, μ_lin, θ.camera.vignetting(VgtkCore.center(θ.camera.vignetting))); flush(stdout)
    isempty(meta.vignetting_gcv) || (g = last(meta.vignetting_gcv);
        @printf("vignetting smoothness: β %.4g by GCV (minimum %.6g, %.1f effective parameters of %d%s)\n", g.β, g.gcv, g.dof, g.n_free,
                g.at_bracket_end ? "; AT THE END of the bracket" : ""))
    meta = (; meta..., costs = (; init = (c0, rms0), linear = (c1, rms1)), μ_lin)
    if grid_gcv                                                 # the grid smoothness's weight by GCV, on the linearization at this start
        tg = time()
        lp = _refinement_lm(settings, route)                          # the refinement's problem (its blocks and holds, Θ_m's among them)
        grid_sm, sel = select_grid_smoothness_weight(f, θ, settings.grid.smoothness, grid_int; lp.kw...)
        @printf("grid smoothness %s: weight %.4g by GCV (%.0f s; minimum %.6g, %.1f effective parameters of %d columns, %d observations%s)\n",
                smoothness_line(with_weight(settings.grid.smoothness, sel.β)), sel.β, time() - tg, sel.gcv, sel.dof, sel.n_columns, sel.n_observations,
                sel.at_bracket_end ? "; AT THE END of the bracket" : ""); flush(stdout)
        meta = (; meta..., grid_smoothness = grid_sm, grid_gcv = sel)
    end
    f.start = meta
    return f, θ
end

"""The state `s` (its front end, masks and photographs read; `meta` its start's) of a start record `start`: the closed-form start's
state loaded into it (`load_state_into!`), its costs, μ and GCV selections recorded, the grid smoothness at the recorded weight — nothing of
the start computed again."""
function _from_start(f::CalibrationData, θ::CalibrationParameters, meta, start, run::RunOptions, settings::CalibrationSettings)
    vig_ok, light_ok = load_state_into!(f, θ, start.state, meta.names; sensor_model = sensor_model_mode(run.camera.sensor_model))
    vig_ok && light_ok || throw(ArgumentError("the start record $(start.dir): its $(vig_ok ? "lighting" : "vignetting field") is not the run's " *
                                              "(another size or geometry): a start of other code or options — produce its start again"))
    grid_sm, grid_gcv = meta.grid_smoothness, nothing
    if _grid_gcv(run, settings)                                          # the GCV's weight of the start (a grid start's: `_grid_gcv`), its smoothness rebuilt there
        grid_gcv = start.grid_gcv
        grid_gcv === nothing && throw(ArgumentError("the start record $(start.dir) records no grid-smoothness GCV, the run chooses its weight by GCV"))
        grid_sm = grid_smoothness(settings.grid.smoothness, θ.camera.geometric, data_norm2(f), meta.grid_int; weight = grid_gcv.β)
    end
    c1, rms1 = start.costs.linear
    @printf("start loaded from %s: closed-form start rel-RMS %.4f; μ %.3f%s\n", start.dir, rms1, start.μ_lin,
            grid_gcv === nothing ? "" : @sprintf("; grid smoothness weight %.4g (GCV of the start)", grid_gcv.β)); flush(stdout)
    f.start = (; meta..., costs = start.costs, μ_lin = start.μ_lin, vignetting_gcv = NamedTuple[start.vignetting_gcv...],
               grid_smoothness = grid_sm, grid_gcv)
    return f, θ
end
