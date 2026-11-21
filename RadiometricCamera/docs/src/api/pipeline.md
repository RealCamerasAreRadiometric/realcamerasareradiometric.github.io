# Pipeline and options

The refinement pipeline of `bin/refine.jl`: the sessions in, the options from TOML, the stages, the records out.

```@meta
CurrentModule = RadiometricCamera
```

## Loading

```@autodocs
Modules = [RadiometricCamera]
Pages = ["io/photographs.jl"]
Private = false
```

## Options

```@autodocs
Modules = [RadiometricCamera]
Pages = ["options/device.jl", "options/tables.jl", "options/toml.jl", "options/retired.jl", "calibration/smoothness.jl"]
Private = false
```

## The pipeline

```@autodocs
Modules = [RadiometricCamera]
Pages = ["start/masks.jl", "start/image_region.jl", "start/focus_start.jl", "start/start_poses.jl", "start/radiometric_start.jl",
         "start/frontends/calibration_session.jl", "start/frontends/schoeps_run.jl", "start/start_calibration.jl", "start/prepare.jl", "start/start_record.jl",
         "diagnostics/monitor.jl", "diagnostics/selfcheck.jl", "diagnostics/render.jl", "io/run_outputs.jl"]
Private = false
```

## Records

RadiometricCamera's kinds, read and written through VisualGeometryToolkitIO (`write_record`, `read_record`), each nesting IO's
records of the Core values it holds, under IO 0.10's envelope (one `units` and one `conventions` object): `rc/radiometric_camera/v3.1`
(a `CalibratedCamera`: the model, its observation model and its render settings), `rc/radiometric_target/v2.0`,
`rc/calibration_state/v3.1` (the state files `state.json`, `state_it<N>.json` and a start record's start state; with the
photographs' observation model) and `rc/run_parameters/v3.0`. A record of an earlier RC version is refused by name — a camera or a
state of RC 0.9–0.11.0 (v3.0) too, which states no observation model (RC 0.11.1: refused, never migrated, naming the refinement step
that regenerates it) — but an `rc/run_parameters/v2.0` (its grid `prior*` keys renamed to the smoothness vocabulary, RC 0.11.0),
which reads through its version step. A camera's or a state's focus surface is the sensor-grid thin plate spline
(`"type": "sensor_thin_plate_spline"`); a record of a removed focus surface (`direction_thin_plate_spline`,
`direction_spherical_spline`) and a file of the removed kind `rc/focus_ab` (the focus A/B experiment's report) are refused by name.

## The observation model and the one entry point of photographs

A calibrated camera states how its photographs were read (`ObservationModel`: the sensor characterisation's sha256 or none, RAW
photosites or decoded images, the channel); every photograph a camera is calibrated on or predicts is read through `photo_set` /
`observe` under it, by dispatch on its types (RC 0.11.1).

```@autodocs
Modules = [RadiometricCamera]
Pages = ["camera/observation_model.jl"]
```

```@autodocs
Modules = [RadiometricCamera]
Pages = ["io/records.jl"]
Private = false
```
