# Calibration

Image formation, the calibration state, the closed-form solves and the joint Levenberg–Marquardt over all blocks, on the CPU
or a device.

```@meta
CurrentModule = RadiometricCamera
```

## Image formation

```@autodocs
Modules = [RadiometricCamera]
Pages = ["image_formation/flux.jl"]
Private = false
```

## The state

```@autodocs
Modules = [RadiometricCamera]
Pages = ["calibration/weighting.jl", "calibration/state.jl", "calibration/precision.jl"]
Private = false
```

## Closed-form solves

```@autodocs
Modules = [RadiometricCamera]
Pages = ["calibration/closed_form.jl"]
Private = false
```

## Levenberg–Marquardt

```@autodocs
Modules = [RadiometricCamera]
Pages = ["calibration/lm/per_view.jl", "calibration/problem_spec.jl", "calibration/calibrate.jl", "calibration/resume.jl",
         "calibration/device/workers.jl", "calibration/device/flux_kernels.jl", "calibration/device/jacobian_kernels.jl"]
Private = false
```

## The grid models' tiles

```@autodocs
Modules = [RadiometricCamera]
Pages = ["calibration/lm/per_tile.jl", "calibration/lm/per_tile_smoothness.jl"]
Private = false
```
