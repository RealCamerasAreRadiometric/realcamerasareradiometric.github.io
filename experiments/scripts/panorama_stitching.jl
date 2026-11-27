#!/usr/bin/env julia
# reproduce: reads configs/experiments/panorama_stitching_fx3_14mm.toml
# reproduce: reads configs/experiments/panorama_stitching_fx3_fisheye.toml
# reproduce: reads configs/experiments/panorama_stitching_fx3_tamron.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/panorama_stitching.jl — the learned vignetting on a panorama, one lens per config: the frames of a pure-rotation sweep
# composed into a panorama twice, as they are and divided by the V(𝐩) the lens's large-aperture refinement learned ([upstream]
# refinement: a config of scripts/radiometric_refinement.jl, configs/experiments/radiometric_refinement_vignetting.toml), so that the
# seams show whether the learned vignetting is right (the supplement's panoramas of the three lenses:
# scripts/panorama_figure.jl). The sweep's chain is PanoramaStitching's direct sampling: every pixel of the panorama through the
# lens's projection into every frame, no feature blending (score_seams, render_panorama). Its decoded frames, rotations and lens
# are the run of scripts/panorama_vignetting.jl on [upstream] panorama_vignetting (PanoramaStitching's example of record on the
# lens's sweep); this driver reads that run through the example's functions (src/panorama_example.jl) and src/panorama.jl adds the
# candidates. Configs of record: configs/experiments/panorama_stitching_fx3_<lens>.toml (14mm, fisheye, tamron).
#
# STAGES (each run keyed by its config record and the keys of the runs it reads; <results>/<experiment>/<stage>/<key>/)
#   refine     the learned maps' refinements: the cases of [maps] in [upstream] refinement (scripts/radiometric_refinement.jl's)
#   evaluate   the learned vignettings of the refinements' states, through IO (V_<map>.json: vgio/vignetting; maps.json)
#   rooftop    the sweep's chain of [upstream] panorama_vignetting (scripts/panorama_vignetting.jl's: decode, register, fit)
#   panoramas  per candidate V — none, and each learned map —: the frames' gains, the seam score (score_seams) and the
#              equirectangular panorama (render_panorama) (panoramas.json, panorama_<slug>.png)
#
#   julia --project=experiments experiments/scripts/panorama_stitching.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--force]
#
# The refinement and the sweep's chain are their own drivers' GPU runs: each is read by its key, and one that is missing is
# produced by its driver where this process holds a GPU, or named with the command that does (CUDA_VISIBLE_DEVICES=0).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("panorama_example.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("panorama.jl"))
include(srcdir("panorama_stitching.jl"))
using .Panorama: MapVignetting, no_vignetting, vignetting_map, ring_mean_map, equisolid_theta, sweep_panoramas, panorama_exposure
import PanoramaStitching
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
import VisualRadiometryToolkitCore as VrtkCore
using StaticArrays: SVector
using VisualGeometryToolkitEval.Experiments: run_isolated                     # a missing upstream is produced by its own driver (driver_run)

const DRIVER = "scripts/panorama_stitching.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "panorama_stitching_fx3_14mm.toml")))   # default: a config of record
const CFG = load_panorama_stitching_config(CFG_PATH)
"The two configs it reads: the refinement config whose cases learn the maps, and the sweep's chain (the example's config of record on the sweep, whose camera and radial V centre this experiment's rings)."
const UP = panorama_stitching_upstream(CFG; path = CFG_PATH)
const REFINEMENT_PATH = UP.refinement_path
const REFINEMENT = UP.refinement
const PV_PATH = UP.pv_path
const PV = UP.pv
const MAPS = sort(collect(keys(CFG["maps"])))
"""The panoramas stage reads the sweep's registration through the example's own `registration`, which resolves LoMa's ONNX Runtime
provider before it reads its keyed record: with [upstream] panorama_vignetting's [example.registration] execution_provider = "cuda"
the stage runs in a process that holds a GPU (its seam scores and renders run on the host)."""
const GPU_STAGES = PV["example"]["registration"]["execution_provider"] == "cuda" ? [:panoramas] : Symbol[]

"The sweep's run of [upstream] panorama_vignetting (its driver's evaluate product)."
rooftop_product() = panorama_vignetting_product(PV)
"""The directory of the sweep's run, produced on demand by its own driver (`produce_upstream`: scripts/panorama_vignetting.jl's
stage evaluate, a GPU's when its [example.compute] device is "cuda")."""
rooftop_dir() = produce_upstream(rooftop_product(), "the sweep's run of $(relpath(PV_PATH, projectdir()))";
                                 driver_run("scripts/panorama_vignetting.jl", PV_PATH, "evaluate")..., gpu = PV["example"]["compute"]["device"] == "cuda")
"The sweep's camera: the camera its run read (`panorama_lens_file`: its camera of record bounded by the sweep's lens coverage)."
lens_camera() = VgtkIO.load(VgtkCore.BoundedCamera, panorama_lens_file(rooftop_dir()); name = PV["example"]["inputs"]["lens_camera"])
"""
    lens_circle() -> (cx, cy, r)

The sweep's image circle, camera pixels: the centre of its camera's lens coverage (the sweep's measured image region: an `Ellipse`
or a `Circle`) and the radius of the disc inscribed in it (an ellipse's minor semi-axis); an error for a camera that covers the
whole sensor (no image circle).
"""
function lens_circle()
    c = VgtkCore.lens_coverage(lens_camera())
    c isa VgtkCore.Circle && return (Float64(c.center[1]), Float64(c.center[2]), Float64(c.r))
    c isa VgtkCore.Ellipse && return (Float64(c.center[1]), Float64(c.center[2]), Float64(min(c.a, c.b)))
    error("$PV_PATH: the camera's lens coverage is the sensor ($(typeof(c))), not an image circle")
end
"""
    ring_disc() -> (cx, cy, r)

The disc the rings span, camera pixels: about the sweep's radial V centre ([upstream] panorama_vignetting [example.vignetting]
radial_center: "circle_center" the centre of its camera's image circle, or a stated [x, y]), out to the farthest point of its
camera's lens coverage (Core's `farthest_radius`) — the image circle's rim, or, for a camera that covers the whole sensor (its lens
coverage the sensor), the farthest sensor corner.
"""
function ring_disc()
    cam = lens_camera(); rc = PV["example"]["vignetting"]["radial_center"]
    o = rc == "circle_center" ? SVector{2, Float64}(lens_circle()[1:2]) :
        rc isa AbstractVector && length(rc) == 2 ? SVector{2, Float64}(rc...) :
        error("$PV_PATH [example.vignetting] radial_center = $(repr(rc)): the rings need \"circle_center\" or [x, y]")
    return (o[1], o[2], VgtkCore.farthest_radius(VgtkCore.lens_coverage(cam), o))
end
lens_opts() = (; circle = ring_disc(), f_px = Float64(CFG["lens"]["f_px"]), image_size = Tuple(CFG["lens"]["image_size"]))
function logger(dir)
    mkpath(dir)
    return msg -> (println(msg); open(io -> println(io, msg), joinpath(dir, "run.log"), "a"))
end

# ---- refine and evaluate: the learned maps -------------------------------------------------------------------------------------
"A vignetting on the learned maps' grid ([evaluate] spacing over [evaluate] size: Panorama's `vignetting_map`)."
evaluate_map(v) = vignetting_map(v, CFG["evaluate"]["spacing"], Tuple(CFG["evaluate"]["size"]))
"""The state of map `m`'s refinement ([maps.<m>] state of its case's refine product), the refinement produced on demand by its
own driver (`refinement_upstream`)."""
map_state(m) = joinpath(refinement_upstream(REFINEMENT, REFINEMENT_PATH, CFG["maps"][m]["case"], "refine", "[maps.$m]: the refinement of case $(CFG["maps"][m]["case"])"),
                        CFG["maps"][m]["state"])
"Stage refine: every learned map's refinement is there (produced on demand)."
refine() = (foreach(map_state, MAPS); nothing)
evaluate_product() = panorama_maps_product(CFG, UP)
evaluate(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    stats = Dict{String,Any}(); states = map_state.(MAPS)
    for (m, state) in zip(MAPS, states)
        v = RadiometricCamera.load_state(state).camera.vignetting      # the refinement's learned V, the model itself
        VgtkIO.save(joinpath(dir, "V_$m.json"), v; provenance = (; state, case = CFG["maps"][m]["case"]))
        M = evaluate_map(v)
        stats[m] = Dict{String,Any}("rows" => size(M, 1), "cols" => size(M, 2), "min" => minimum(M), "max" => maximum(M),
                                    "centre" => M[cld(size(M, 1), 2), cld(size(M, 2), 2)])
    end
    VgtkIO.write_record(joinpath(dir, "maps.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.panorama_maps), stats); allownan = true)
    Produced(; inputs = states)
end

"""
    learned_camera() -> CalibratedCamera

The learned maps' calibrated camera — each map's refinement state as a complete calibration (`calibrated_camera`), one observation
model for every map (maps of different models are refused: the sweep's frames are read under one).
"""
function learned_camera()
    cams = [RadiometricCamera.calibrated_camera(RadiometricCamera.load_state(map_state(m))) for m in MAPS]
    all(c -> c.observation == first(cams).observation, cams) ||
        throw(ArgumentError("$CFG_PATH: the learned maps $(MAPS) were calibrated under different observation models " *
                            "($(join(unique(RadiometricCamera.observation_line.(getfield.(cams, :observation))), "; "))): one model reads the frames"))
    return first(cams)
end

# ---- rooftop: the sweep's run ---------------------------------------------------------------------------------------------------
"Stage rooftop: the sweep's run is there (produced on demand)."
rooftop_stage() = (rooftop_dir(); nothing)

"""
    rooftop() -> (; C, inp, lens, reg, W, H)

The sweep's run of [upstream] panorama_vignetting (its registration's bundle termination checked against the upstream's
[expected]), read through the example's functions: its context `C` (the run's example_config.toml), the decoded frames and
saturation masks (the example's decode cache), the lens, the registration (the example's keyed product) and the camera size.
"""
function rooftop()
    C = panorama_example_call(:load_config, joinpath(rooftop_dir(), "example_config.toml"))
    (; inp, lens, reg, W, H) = example_registration(C)   # the example's keyed registration: its key is the run's settings
    check_registration(PV, reg; where = "$PV_PATH (the sweep's run)")
    require_frames_decoded_as(inp, learned_camera())   # the frames under the learned camera's own sensor record
    (W, H) == Tuple(CFG["lens"]["image_size"]) || throw(ArgumentError("$CFG_PATH [lens] image_size = $(CFG["lens"]["image_size"]) but the sweep's frames are $W × $H camera pixels"))
    return (; C, inp, lens, reg, W, H)
end
"The largest equisolid field angle the sweep's domain reaches: the rings' disc less the sweep's rim margin."
theta_max() = equisolid_theta(ring_disc()[3] - PV["example"]["domain"]["rim_margin_px"], CFG["lens"]["f_px"])

# ---- panoramas ------------------------------------------------------------------------------------------------------------------
"The learned map `m`'s vignetting (vgio/vignetting: the evaluate stage's V_<map>.json), that stage produced on demand."
learned_vignetting(m) = VgtkIO.load(VrtkCore.AbstractVignetting,
                                    joinpath(produce_upstream(evaluate_product(), "the learned vignettings (stage evaluate)"; make = () -> evaluate(evaluate_product()),
                                                              command = driver_run(DRIVER, CFG_PATH, "evaluate").command), "V_$m.json"))
the_maps() = Dict(m => (CFG["maps"][m]["label"], evaluate_map(learned_vignetting(m))) for m in MAPS)
panoramas_product() = panoramas_product(CFG, UP)
"""The candidate vignettings at a sensor point: none, and every learned map, on the maps' grid and divided by its mean over the
ring [panoramas] unit_ring_deg of the equisolid field angle about the sweep's vignetting centre."""
function candidates()
    lo = lens_opts(); ring = Float64.(CFG["panoramas"]["unit_ring_deg"]); tmax = theta_max()
    c = Any[("original", "no vignetting correction", no_vignetting)]
    for (slug, (label, M)) in the_maps()
        n1 = ring_mean_map(M, lo.circle, lo.f_px, lo.image_size, ring[1:1], ring[2:2], tmax)[1]
        push!(c, (slug, label, MapVignetting(M, lo.image_size, n1)))
    end
    return c
end
panoramas(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    say = logger(dir)
    R = rooftop(); t = CFG["panoramas"]; d = R.C.cfg["domain"]; K = length(R.inp.frames)
    # the candidates' vignettings are host functions (a learned map is a host array): the frames on the host
    sweep = (; R.lens, frames = R.inp.frames, saturated = R.inp.saturated, rotations = R.reg.rotations, R.inp.frame_map, R.inp.plane_maps)
    gc = (; cfg = merge(R.C.cfg, Dict("vignetting" => merge(R.C.cfg["vignetting"], Dict("gain_correction" => "channel")))), R.C.out, R.C.path)
    gains(v) = t["gains"] == "fit" ?   # PanoramaStitching's fit with the candidate's V held fixed: the frames' gains alone
        Matrix{Float64}(panorama_example_call(:fit, "radial", gc, R.lens, (; sweep.frames, sweep.saturated, sweep.frame_map, sweep.plane_maps), sweep.rotations, R.W, R.H;
                                              vignetting = v).metadata.gains) :
        ones(K, 3)                       # every frame and plane at one exposure
    res = sweep_panoramas(sweep, candidates(); gains, seam_deg_per_pixel = t["seam_deg_per_pixel"], deg_per_pixel = t["deg_per_pixel"], chunk = t["chunk"],
                          plane = d["plane"], rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], saturation_margin_px = d["saturation_margin_px"], say)
    # the PNGs as PanoramaStitching's example chain renders its panoramas ("corrected and uncorrected at one exposure"): one exposure
    # for every candidate, 1 / the `exposure_percentile` of the G of the corrected panorama `exposure_candidate`, LibRaw's daylight white
    # balance, the example's own writer save_srgb (sRGB, clamped; no frame: black)
    wb = R.inp.daylight_wb ./ R.inp.daylight_wb[2]
    exposure = panorama_exposure(res.images[t["exposure_candidate"]], t["exposure_percentile"])
    for (slug, img) in res.images
        panorama_example_call(:save_srgb, joinpath(dir, "panorama_$slug.png"), img, exposure, wb)
    end
    say("  exposure (every candidate) $(round(exposure; digits = 4)): 1 / the $(t["exposure_percentile"]) % G of $(t["exposure_candidate"])")
    out = Dict{String, Any}("frames" => R.inp.names, "extent" => collect(res.extent), "seam_deg_per_pixel" => t["seam_deg_per_pixel"],
                            "deg_per_pixel" => t["deg_per_pixel"], "white_balance" => wb, "gains_mode" => t["gains"], "exposure_percentile" => t["exposure_percentile"],
                            "exposure_candidate" => t["exposure_candidate"], "candidates" => res.candidates, "exposure" => exposure)
    VgtkIO.write_record(joinpath(dir, "panoramas.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.panoramas), out); allownan = true)
    nothing
end

const STAGES = [
    Stage(:refine, "the refinements of the learned maps (scripts/radiometric_refinement.jl on [upstream] refinement)"; check = refine),
    Stage(:evaluate, "the learned vignettings through IO (V_<map>.json)"; items = () -> [evaluate_product()], product = identity, run = evaluate),
    Stage(:rooftop, "the sweep's chain (scripts/panorama_vignetting.jl on [upstream] panorama_vignetting)"; check = rooftop_stage),
    Stage(:panoramas, "the panoramas and seam scores under each candidate (panoramas.json, panorama_<slug>.png)"; items = () -> [panoramas_product()], product = identity, run = panoramas),
]

run_stages("panorama_stitching", STAGES)
