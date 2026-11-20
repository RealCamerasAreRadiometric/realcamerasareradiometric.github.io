# Camera

The geometric chief-ray models, the kernels' view of the radiometric camera at a pose, and the vignetting's rules and gauge.

The radiometric camera itself (`RadiometricCameraModel`: a geometric camera with its focus surface, aperture radius and
vignetting), its back-projection from the aperture (`chief_dir`, `𝔣`, `π̂⁻¹`, `world_ray`), the focus surface (`Focus`), the
vignetting fields, the sensor domain and the quadrature rules on the pixel and the aperture are
VisualRadiometryToolkitCore's: its manual documents them, and they are reached through that package
(`using VisualRadiometryToolkitCore: Focus`). `RadiometricCameraModel` alone is re-exported here. A camera composed from its
geometric camera has no vignetting (`NoVignetting`); a calibration estimates V, so its camera states a field —
[`RadiometricCamera.flat_vignetting`](@ref) is the flat start.

```@meta
CurrentModule = RadiometricCamera
```

## Vignetting

```@autodocs
Modules = [RadiometricCamera]
Pages = ["camera/vignetting.jl"]
Private = false
```

## Geometric models and the camera at a pose

```@autodocs
Modules = [RadiometricCamera]
Pages = ["camera/pose.jl", "camera/models/geometric.jl", "camera/view.jl", "camera/models/pixel_camera.jl",
         "camera/models/grid_camera.jl", "io/camera_models.jl"]
Private = false
```
