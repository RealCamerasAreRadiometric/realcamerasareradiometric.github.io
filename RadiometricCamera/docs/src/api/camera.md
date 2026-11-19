# Camera

The geometric chief-ray models, the radiometric back-projection around them, the focus surface, the vignetting and the
quadrature rules on the pixel and the aperture.

```@meta
CurrentModule = RadiometricCamera
```

## Notation and quadrature

```@autodocs
Modules = [RadiometricCamera]
Pages = ["sensor/notation.jl", "sensor/quadrature.jl"]
Private = false
```

## Vignetting

```@autodocs
Modules = [RadiometricCamera]
Pages = ["camera/vignetting.jl"]
Private = false
```

## Focus

```@autodocs
Modules = [RadiometricCamera]
Pages = ["camera/focus.jl", "experiments/focus_charts.jl"]
Private = false
```

## Geometric models and the radiometric back-projection

```@autodocs
Modules = [RadiometricCamera]
Pages = ["camera/pose.jl", "camera/models/geometric.jl", "camera/radiometric_camera.jl", "camera/models/pixel_camera.jl",
         "camera/models/grid_camera.jl", "io/camera_models.jl"]
Private = false
```
