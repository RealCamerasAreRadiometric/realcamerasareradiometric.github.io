"""The blob front-end: a calibration session's records (`load_session`: the camera → the start grid, the board's target and printed
image, the frames' poses; the world is the board's frame) and its observations (`observations.json`, the session's
VisualGeometryToolkitIO observations record: `_session_observations`)."""
function _frontend(::Type{BlobFrontend}, run::RunOptions, settings::CalibrationSettings, start::NamedTuple)   # reads start.R0 (a blob session's frame is its records';
    R0 = start.R0                                                                       #   it places no board features of its own)
    isempty(run.frontend.session) && error("frontend.session (the session's records) required for the blob front-end")
    obs_file = _session_observations_file(run.frontend.session)        # refused by name before any record is read
    ds = load_session(run.frontend.session, run.io.images; lens = :camera, focus_init = :board_center, downsample = 1,
                      texture_downsample = run.frontend.texture_downsample, sensor_width = _sensor_width(run), r₀ = R0, linearize = run.io.linearize,
                      load_targets = false,   # the photographs are streamed below; decoding all of them here cost ≈ 2 s per view
                      raw_dir = isempty(run.io.raw) ? nothing : run.io.raw, lens_identity = run.io.lens_identity === nothing ? nothing : lens_identity(run.io.lens_identity))
    capture = ds.settings                                        # read once: the test below narrows it (`ds` may infer as a Union)
    capture === nothing || println("camera settings: ", settings_line(capture; lens_stated = run.io.lens_identity !== nothing))
    board = ds.board; names = copy(ds.names); W₀, H₀ = ds.extent
    poses0 = copy(ds.poses)
    stored = start.stored                                         # a start record (`load_refinement_start`): its lens, region and poses, no fit
    st0 = stored !== nothing ? (; stored.lens, stored.init_region, resample = nothing) :
          isempty(run.io.init) ? _session_start(run, settings, ds.lens₀) :   # the session's camera onto the run's lattice (`_start_grid`, by its type)
                                  (; load_start(run.io.init)..., resample = nothing)
    lens0 = st0.lens; init_region = st0.init_region              # the initialization region: where the start is defined (a stage input)
    @printf("blob front-end: %d views of %s, image %d × %d, init grid %s from %s\n", length(names), run.frontend.session, W₀, H₀, _lattice_line(lens0),
            isempty(run.io.init) ? "the session's camera" : run.io.init); flush(stdout)
    # the observations (the session's IO record: the image point, COLMAP px, of each target point id) → features with the target's
    # points (its frame, the world, plain metres); the board's y sign is chosen by the self-check
    frame_of = Dict(n => t for (n, t) in zip(ds.names, ds.frames))
    raw = _session_observations(obs_file, board.target)
    fpx0 = fpx_of(lens0, W₀, H₀)
    bestm = Inf; features = Vector{Vector{Tuple{SVector{2, Float64}, SVector{3, Float64}}}}(); board_y_sign = 1.0
    board_features(sy) = [[(SVector(x, y), SVector(bx, sy * by, bz)) for (x, y, bx, by, bz) in get(raw, frame_of[names[k]], NTuple{5, Float64}[])]
                          for k in eachindex(names)]
    if stored !== nothing                                # the start's choice of the board's y sign: no bearing errors computed to choose it
        board_y_sign = stored.board_y_sign; features = board_features(board_y_sign)
    else
        for sy in (1.0, -1.0)
            fs = board_features(sy)
            be = detection_bearings(lens0, poses0, [in_initialization(fs[k], init_region) for k in eachindex(names)], fpx0)
            m = isempty(be) ? Inf : median(be); @printf("  board y sign %+.0f: bearing self-check median %.3f px over %d detections\n", sy, m, length(be))
            if m < bestm; bestm = m; features = fs; board_y_sign = sy; end
        end
    end
    isempty(features) && (features = [Tuple{SVector{2, Float64}, SVector{3, Float64}}[] for _ in names])   # no detections
    if stored !== nothing                                # the start's poses: none refitted
    elseif run.selfcheck.only || !isempty(run.io.init)        # another model than the session's: the session's poses belong to its own model → refit each
        for k in eachindex(names)
            fk = _imaged(lens0, in_initialization(features[k], init_region)); length(fk) < 6 && continue
            Xs = [X for (_, X) in fk]; vs = [normalize(backproject(lens0, 𝐩)) for (𝐩, _) in fk]
            r = _refined_pose(inv(poses0[k]), _ray_correspondences(Xs, vs))     # world → camera
            VgtkCore.successful_retcode(r) || (@printf("  %s: the pose refit did not succeed (%s, %s): the session's pose kept\n", names[k], r.retcode, r.stop_rule); continue)
            e0 = median(bearing_px(lens0, poses0[k], fk, fpx0)); poses0[k] = _pose_eltype(Float64, inv(r.u))
            @printf("  %s: pose refit through the lens under test, bearing median %.3f → %.3f px\n", names[k], e0, median(bearing_px(lens0, poses0[k], fk, fpx0)))
        end
        flush(stdout)
    end
    if !isempty(run.io.views)                                  # io.views subsets the session's views too
        keep = [k for k in eachindex(names) if names[k] in run.io.views]
        isempty(keep) && error("io.views matches none of the session's views")
        names = names[keep]; poses0 = poses0[keep]; features = features[keep]
    end
    stored === nothing || (poses0 = _stored_poses(stored, names))
    return (; board, names, poses0, lens0, init_region, W₀, H₀, st0.resample, features, features_for = _ -> features, page_flip = 1.0, board_y_sign,
            world = "the board's target frame (the session's records)")
end

"""The session's observations record, `<session>/observations.json` (vgio/observations, read by VisualGeometryToolkitIO's
`load(TargetObservations, …)`): the one form the blob front end reads its detections in (owner D4, 2026-11-15: no hand parser;
a BlobBoards `sightings.json` is not read). A session without it is refused by name, with the record it needs — its producer
(CaliBlobs' session records, or an IO importer of BlobBoards' export) writes it beside camera.json."""
function _session_observations_file(session::AbstractString)
    f = joinpath(session, "observations.json")
    isfile(f) || throw(ArgumentError("the calibration session $session holds no observations.json: the blob front end reads its detections only as the " *
                                     "session's VisualGeometryToolkitIO observations record (vgio/observations, beside camera.json; " *
                                     "`VisualGeometryToolkitIO.load(TargetObservations, …)`) — a BlobBoards sightings.json is not read"))
    return f
end

"""The observations of `file` (`_session_observations_file`) by frame: `(x, y, X, Y, Z)` per observation — its image point (COLMAP px,
the observation's value) and its target point (`points(target)` by the point's index, the target's frame, plain metres; a planar
target's z = 0). One target's: a record of more than one target is refused (two targets' indices would collide)."""
function _session_observations(file::AbstractString, target)
    tp = VgtkIO.load(VgtkCore.TargetObservations, file)
    ids = unique(id for t in keys(tp) for id in keys(tp[t]))
    length(ids) ≤ 1 || throw(ArgumentError("$file: the observations of $(length(ids)) targets ($(join(sort(String.(ids)), ", "))); the blob front end reads one target's"))
    P = VgtkCore.points(target)
    X(i) = (p = P[i]; (Float64(p[1]), Float64(p[2]), length(p) == 3 ? Float64(p[3]) : 0.0))
    out = Dict{Int, Vector{NTuple{5, Float64}}}()
    for t in keys(tp), cs in values(tp[t]), (i, xy) in cs
        q = VgtkCore.geom(xy)
        push!(get!(() -> NTuple{5, Float64}[], out, Int(t)), (Float64(q[1]), Float64(q[2]), X(Int(i))...))
    end
    return out
end

"""The blob front end's start from the session's camera `src`: `_start_grid` over the run's lens coverage (`run_lens_coverage`) — a
SplineLens camera resampled exactly, a parametric one fitted over the image region at `grid.node_count` ([`GridStart`](@ref)) — and `src`'s initialization region."""
function _session_start(run::RunOptions, settings::CalibrationSettings, src)
    st = _start_grid(src, settings.grid; coverage = run_lens_coverage(run, Tuple(VgtkCore.extent(src))...))
    return (; lens = st.camera, init_region = initialization_region(src), st.resample)
end
