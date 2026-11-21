# Target and lighting

The printed board 𝓑 and its warping h, the Cook–Torrance SVBSDF and the radiance L_c. The light is VisualRadiometryToolkitCore's
`EnvironmentLight` (real spherical harmonics per channel on a sphere about the board centre, `SphericalHarmonicMap`); its
documentation is that package's.

```@meta
CurrentModule = RadiometricCamera
```

## The board and its warping

```@autodocs
Modules = [RadiometricCamera]
Pages = ["target/board.jl", "target/target_radiometry.jl"]
Private = false
```

## SVBSDF and radiance

```@autodocs
Modules = [RadiometricCamera]
Pages = ["image_formation/radiance.jl"]
Private = false
```
