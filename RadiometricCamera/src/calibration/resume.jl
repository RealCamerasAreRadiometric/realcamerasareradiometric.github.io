# ---- the state as a file -------------------------------------------------------------------------------------------------------
"""The page's ±y choice (`page_flip`) of the state a run renders (`io.render_state`) or resumes (`resume.state`), to hold in the front end;
`nothing` without such a state, or for a state that records none (one written before 2026-11-02; a warning: the tie is broken again)."""
function _state_page_flip(run::RunOptions, load_state)
    path = !isempty(run.io.render_state) ? run.io.render_state : run.resume.state
    isempty(path) && return nothing
    sst = load_state(path)
    sst.page_flip === nothing && (@warn "$path carries no page_flip (a state written before 2026-11-02): the page's ±y tie is broken again on this run's views"; return nothing)
    return Float64(sst.page_flip)
end
"""Load a state ([`load_state`](@ref)) into `s` (the poses of the views by name; the camera's lens, focus and aperture; the target's
material, warping and printed edge; the vignetting and the lighting when their sizes match). Returns `(vig_ok, light_ok)`. The state's
observation model must be the run's (`observation_model(f)`: the same sensor record, photograph kind and channel): a state continues,
or predicts, only photographs read as it was estimated on — another is refused naming both. Under `sensor_model =`
[`OffSensorModel`](@ref)`()` (`camera.sensor_model = "off"`) a state estimated under a sensor record continues a run that reads the
same photographs without it (the run's resolution warned, `observation_model`'s `stated`)."""
function load_state_into!(f::CalibrationData, θ::CalibrationParameters, sst, names; sensor_model::SensorModelMode = AutoSensorModel())
    _state_continues(sensor_model, sst.observation, observation_model(f)) ||
        throw(ArgumentError("state: estimated under the observation model $(observation_line(sst.observation)), the run reads its photographs " *
                            "under $(observation_line(observation_model(f))): a state continues or predicts only photographs read as it was estimated on"))
    idx = [findfirst(==(n), sst.names) for n in names]
    any(isnothing, idx) && error("state: views not in the state: $(names[isnothing.(idx)])")
    set_radiometric_target!(f, θ, sst.target)                          # the material, the board warping h and the print's edge (first: a warping of another page is refused before anything moves)
    for k in eachindex(names)
        set_pose!(θ, k, sst.poses[idx[k]])
    end
    c = sst.camera
    lens = c.geometric
    if !(typeof(lens) === typeof(θ.camera.geometric))
        lens = _with_run_region(lens, θ.camera.geometric)
        println("state: written under another imaged region — its grid lens takes the run's region")
    end
    light_ok = size(coefficients(sst.light)) == size(coefficients(θ.light))
    light_ok && (θ.light = sst.light)
    vig_ok = _same_field(c.vignetting, θ.camera.vignetting) && light_ok
    θ.camera = RadiometricCameraModel(lens, c.focus, c.aperture_radius, vig_ok ? c.vignetting : θ.camera.vignetting)
    return vig_ok, light_ok
end

# whether a state estimated under `a` continues a run reading under `b`: the same observation model; "off" also a record not used
_state_continues(::SensorModelMode, a::ObservationModel, b::ObservationModel) = a == b
_state_continues(::OffSensorModel, a::ObservationModel{SensorRecord}, b::ObservationModel{NoSensorRecord}) = a.kind == b.kind && a.channel === b.channel
# The state's vignetting field is the run's only when its geometry is (2026-10-30: a 16-node profile over the whole sensor,
# r_max 2560 px, was taken for the 16-node profile over a measured image circle, r_max 1592 px, by its size alone): the same
# type, number of nodes, degree and knots, and for a radial profile the same centre and r_max.
_same_field(a::AbstractVignetting, b::AbstractVignetting) = false
_same_field(a::RadialVignetting, b::RadialVignetting) =
    n_control_points(a) == n_control_points(b) && VrtkCore.spline_degree(a) == VrtkCore.spline_degree(b) &&
    VrtkCore.radial_chart(a) == VrtkCore.radial_chart(b) &&
    all(map(≈, VrtkCore.knot_vectors(a), VrtkCore.knot_vectors(b))) && VgtkCore.center(a) ≈ VgtkCore.center(b) &&
    isapprox(VrtkCore.field_radius(a), VrtkCore.field_radius(b); rtol = 1e-9)
