# ---- the Schöps detections and the pose refine: the owners' reader and refiner, in the shapes the front-ends read -------------
"""
    dataset_features(path) -> [(image name, [(x, y, id)])]

The detections of a Schöps `dataset.bin` (a run's, or the texture's own detector run) in file order: COLMAP px (Schöps' pixel-corner
convention is COLMAP's, `PixelFrame{:colmap}`), 0-based feature ids — VisualGeometryToolkitIO's `load(TargetObservations, path; target)`;
or a run's import, `detections.json` ([`schoeps_run_file`](@ref): a vgio observations file in COLMAP px — v3.0, or an earlier
target_points2d/v1 read through IO's version step — which names its targets itself, so IO's reader takes no `target`). The ids are
one target's point indices: a file holding the detections of more than one target is refused (two targets' ids would collide).
"""
function dataset_features(path::AbstractString)
    tp = basename(path) == "detections.json" ? VgtkIO.load(VgtkCore.TargetObservations, path) : VgtkIO.load(VgtkCore.TargetObservations, path; target = :board)
    ids = unique(id for t in keys(tp) for id in keys(tp[t]))
    length(ids) ≤ 1 || throw(ArgumentError("$path: the detections of $(length(ids)) targets ($(join(sort(String.(ids)), ", "))): " *
                                           "dataset_features reads one target's, whose 0-based ids are its point indices"))
    pixels(xy) = (q = VgtkCore.geom(xy); (Float64(q[1]), Float64(q[2])))
    return [(tp.images[t], [(pixels(xy)..., Int(i) - 1) for cs in values(tp[t]) for (i, xy) in cs]) for t in sort!(collect(keys(tp)))]
end

"""
    require_featured_views(counts, views, run)

The one policy for a view without front-end features: it has no start pose, so the case's views list (`io.views`) must not name it —
the config of record drops it, with a comment naming the front-end run; the solver never drops a view silently. `counts` is the
front-end run's photographs with their usable feature counts (`name => n`), `views` the case's list (empty: every photograph of the
run). An `ArgumentError` names each photograph of `views` (or, with none given, of the run) with 0 features, and each one `views`
names that the run has no detections of, with the reason.
"""
function require_featured_views(counts::AbstractVector{<:Pair{String, Int}}, views::AbstractVector{<:AbstractString}, run::AbstractString)
    n = Dict(counts)
    absent = sort([String(v) for v in views if !haskey(n, v)])
    isempty(absent) || throw(ArgumentError("io.views names $(join(absent, ", ")), of which the front-end run $run holds no detections: " *
                                           "a view without front-end features has no start pose; drop it from the case's views list"))
    named = isempty(views) ? first.(counts) : String.(views)
    empty = sort([v for v in named if n[v] == 0])
    isempty(empty) || throw(ArgumentError("$(join(empty, ", ")): 0 front-end features in $run — a view without features has no start pose; " *
                                          "drop $(length(empty) == 1 ? "it" : "them") from the case's views list (io.views), naming the front-end run"))
    return nothing
end

"""The image size `(W₀, H₀)` of the Schöps front-end: the start camera's, the fact — the stored start's (`start`, `io.init`'s
`load_start`: its camera's extent), else the Schöps run's model
(`frontend.schoeps`'s intrinsics0.yaml, VisualGeometryToolkitIO's reader). No photograph is opened: each photograph is checked
against the start camera's size where it is read (`observe`, one decode per photograph)."""
function _schoeps_extent(opts::PipelineOptions, start)
    model(prefix, y) = (
                          isfile(y) || error("no start camera to read the image size from: no $y");
                          (Tuple(VgtkCore.extent(load_camera(prefix * y))), y))
    (W, H), src = start !== nothing ? (Tuple(VgtkCore.extent(start.lens)), opts.io.init) :
                  model("schoeps:", schoeps_run_file(opts.frontend.schoeps, :camera))   # the run's import, or its intrinsics0.yaml
    @printf("  image size %d × %d from the start camera (%s)\n", W, H, src)
    return (W, H)
end

"""`_schoeps_frontend_start(opts, stored, W, H)`: the Schöps front end's start — the stored start `stored` (`io.init`, `load_start`)
when there is one, and then no Schöps run's model is read (`resample = nothing`); else the Schöps run's spline model resampled
exactly on his own lattice (`grid.lattice = "derived"`, m = 1) over the sensor ∩ the run's lens coverage (`_schoeps_start`; the coverage
`run_lens_coverage`'s: the sensor, or the capture set's image-region record)."""
function _schoeps_frontend_start(opts::PipelineOptions, stored, W, H)
    stored === nothing || return (; stored.lens, stored.init_region, resample = nothing)
    f = schoeps_run_file(opts.frontend.schoeps, :camera)
    isfile(f) || error("no Schöps start camera: no stored start (io.init) and no $f (frontend.schoeps: the run's import camera.json or its intrinsics0.yaml)")
    src = _schoeps_model(f, W, H, "the Schöps start")
    return _schoeps_start(f, W, H; coverage = run_lens_coverage(opts, W, H))
end

"""The Schöps front-end: a central-generic run on a printed page (the board = the page textured with the design;
star ids → texture positions from the detector run on the texture; poses by a multi-start ray-domain fit through the start model).
`start`: the run's start, `(; R0, height0, page_flip, inplane0)` (every front end takes the one NamedTuple and reads its own)."""
function _frontend(::Type{SchoepsFrontend}, opts::PipelineOptions, start::NamedTuple)
    (; R0, height0, page_flip, inplane0) = start
    isempty(opts.frontend.schoeps) && error("frontend.schoeps required for the schoeps front-end")
    (isempty(opts.frontend.texture) || isempty(opts.frontend.texture_features)) && error("frontend.texture and frontend.texture_features required for the schoeps front-end")
    sets = dataset_features(schoeps_run_file(opts.frontend.schoeps, :detections))
    texture = VgtkFeatures.load_linear_image(opts.frontend.texture); TH_t, TW_t = size(texture, 1), size(texture, 2)
    sx_t, sy_t = TW_t / opts.frontend.texture_features_size[1], TH_t / opts.frontend.texture_features_size[2]
    tsets = dataset_features(opts.frontend.texture_features); tex = Dict(id => (x * sx_t, y * sy_t) for (x, y, id) in tsets[1][2])
    # the views of the run: every view the case names must have front-end features (the one policy, `require_featured_views`)
    require_featured_views([String(s[1]) => count(f -> haskey(tex, f[3]), s[2]) for s in sets], opts.io.views, opts.frontend.schoeps)
    isempty(opts.io.views) || (sets = [s for s in sets if String(s[1]) in opts.io.views])
    names = [String(s[1]) for s in sets]
    (sx_t != 1 || sy_t != 1) && @printf("  texture features rescaled by (%.4f, %.4f) to the %d × %d texture\n", sx_t, sy_t, TW_t, TH_t)
    opts.frontend.texture_downsample > 1 && (texture = VgtkFeatures.box_downsample(texture, opts.frontend.texture_downsample))
    page_w, page_h = VgtkCore.ustrip(VgtkCore.m, _page_size(opts)[1]) * opts.frontend.print_scale, VgtkCore.ustrip(VgtkCore.m, _page_size(opts)[2]) * opts.frontend.print_scale
    rec = start.stored                                                     # a start record (`load_refinement_start`): its lens, region and poses, no fit
    stored = rec !== nothing ? (; rec.lens, rec.init_region) : isempty(opts.io.init) ? nothing : load_start(opts.io.init)   # the stored start, read once
    W₀, H₀ = _schoeps_extent(opts, stored)
    st0 = _schoeps_frontend_start(opts, stored, W₀, H₀)
    lens0 = st0.lens; init_region = st0.init_region              # the initialization region: Schöps's calibrated area (a stage input)
    @printf("schoeps front-end: %d views of %s, image %d × %d, page %.1f × %.1f mm (print scale %.4f), texture %d × %d (ds %d), %d texture star ids, init grid %s\n",
            length(names), opts.frontend.schoeps, W₀, H₀, _mm(page_w), _mm(page_h), opts.frontend.print_scale, TW_t, TH_t, opts.frontend.texture_downsample, length(tex), _lattice_line(lens0)); flush(stdout)
    fpx0 = fpx_of(lens0, W₀, H₀)
    # page frame: x = columns (right), rows along +y (down), the printed side's normal 𝐧f = −(𝐮 × 𝐯f) = −z: the world "printed page
    # (origin top-left, x right, y down)", one labelling, no search. Rows along −y with the camera on the other side is the SAME scene in
    # a frame rotated 180° about x (F = diag(1, −1, −1)): every pose fit scores the two alike, and until 2026-11-05 a ±y search broke that
    # tie by rounding (Julia 1.12 and 1.13 chose differently). The camera is on the printed side (cheirality, `_fit_page_pose`); a
    # mirrored texture — the print seen from behind — has no such pose and fails the fit. Under the board warping h the features sit
    # on the surface, z along 𝐧f (the board's `facing` normal); under the in-plane print error (a, u) a design feature is printed at
    # `physical_chart` of its page fraction (Core's Aspect ∘ Skew in the placed chart) and the height is the surface's THERE. A state
    # written before 2026-11-05 with `page_flip = −1` holds its poses in the rotated frame: it is rendered or resumed in that frame
    # (𝐯f = −y, 𝐧f = +z), never re-chosen.
    𝐨 = SVector(0.0, 0.0, 0.0); 𝐮 = SVector(page_w, 0.0, 0.0)
    function feature_sets(𝐯f, h, θ = zero(SVector{INPLANE_NC, Float64}))
        𝐧f = -normalize(cross(𝐮, 𝐯f))
        return map(sets) do s
            kept = [(x, y, id) for (x, y, id) in s[2] if haskey(tex, id)]
            Xs = page_points(𝐨, 𝐮, 𝐯f, 𝐧f, h, [SVector(tex[id][1] / TW_t, tex[id][2] / TH_t) for (_, _, id) in kept]; inplane = θ)
            [(SVector(x, y), X) for ((x, y, _), X) in zip(kept, Xs)]
        end
    end
    flip = page_flip === nothing ? 1.0 : Float64(sign(page_flip))
    flip == 0 && throw(ArgumentError("page_flip must be ±1, got $page_flip"))
    𝐯 = SVector(0.0, flip * page_h, 0.0); bc = 𝐨 + 0.5𝐮 + 0.5𝐯; 𝐧f = -normalize(cross(𝐮, 𝐯))
    features = feature_sets(𝐯, height0, inplane0)
    if rec !== nothing                                                      # the start record's poses: no page-pose fit
        poses0 = _stored_poses(rec, names)
    else
        res_best = map(enumerate(_imaged(lens0, in_initialization(f, init_region)) for f in features)) do (k, fk)
            r = _fit_page_pose([X for (_, X) in fk], [normalize(backproject(lens0, 𝐩)) for (𝐩, _) in fk], 𝐨, 𝐧f, bc, opts.frontend.pose_distances)
            isfinite(r[1]) || error("$(names[k]): no pose with the camera on the printed side of the page fits its $(length(fk)) stars " *
                                    "(a mirrored texture: the print seen from behind?)")
            r
        end
        @printf("  page rows along %+.0fy%s: pose-fit bearing median %.3f px over %d views\n", flip,
                page_flip === nothing ? "" : " (the state's frame)", median(fpx0 * r[1] for r in res_best), length(res_best)); flush(stdout)
        any(!=(0), height0) && @printf("  board features placed on the warping h = %s mm (z towards the camera)\n", join([@sprintf("%+.3f", _mm(c)) for c in height0], " "))
        iszero(inplane0) || @printf("  board features placed with the in-plane print error a %+.4e, u %+.4e (Core's Aspect ∘ Skew in the placed chart)\n", inplane0...)
        poses0 = VgtkCore.PlainEuclideanMap{3, Float64}[]
        for k in eachindex(names)
            e, R, t = res_best[k]; push!(poses0, EuclideanMap(SMatrix{3, 3, Float64}(R'), SVector{3, Float64}(-R' * t); check = false))
            @printf("  %s: %d stars, pose fit bearing median %.3f px, camera at (%.3f, %.3f, %.3f) m\n", names[k], length(features[k]), fpx0 * e, poses0[k].t...)
        end
    end
    board = TexturedTarget(𝐨, 𝐮, 𝐯, texture; facing = poses0[1].t, design = Symbol(splitext(basename(opts.frontend.texture))[1]))
    flip < 0 && @warn "the state's page frame has rows along −y (page_flip = −1, a state written before 2026-11-05): the same board as +y rotated 180° about x; a new run uses +y"
    return (; board, names, poses0, lens0, init_region, W₀, H₀, st0.resample, features, features_for = (surf::TargetState) -> feature_sets(𝐯, surf.h, surf.inplane), page_flip = flip,
            board_y_sign = 1.0, world = "printed page (origin top-left, x right, y down)")
end
