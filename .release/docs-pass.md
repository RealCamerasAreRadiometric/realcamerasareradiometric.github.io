docs-pass: c7bd5f9
landing: site/ of this repository (site/src/index.md, deployed by the Site workflow with the package manual under /docs/): the install paragraph and the example's imports changed in this pass

# RadiometricCamera 0.2.1 docs pass (julia-docs-pass), on the release tree

A docs-only pass after 0.2.0: no code behaviour changed (every edited file under src/ parses to the same code with its strings
removed; only docstrings, comments and error-message wording moved). The package builds on VisualGeometryToolkitCore 0.13.1 (IO,
Features, Estimation; General) and VisualRadiometryToolkitCore / IO 0.1.1 (CauRegistry until their General registration).

[x] Compat: docs/, benchmark/ and test/integration_blobboards/ state VisualGeometryToolkitCore 0.13.1 and
    VisualRadiometryToolkitCore 0.1.1, as the package does (and the integration member Estimation 0.12.1, Features 0.9.9, IO 0.11).
    No Project.toml edited.
[x] Examples import each name from the module that exports it (checked against `Base.isexported` / `Base.ispublic` of
    RadiometricCamera, VisualRadiometryToolkitCore and VisualGeometryToolkitCore): `𝔟` is VisualRadiometryToolkitCore's (README,
    docs/src/index.md and site/src/index.md, one example shown three times; guide/getting-started.md; guide/calibrating.md, which also
    loads VisualRadiometryToolkitCore for `RadialVignetting`, `EnvironmentLight`, `SphericalHarmonicMap`);
    examples/radiometric_calibration.jl imports from VisualRadiometryToolkitCore and VisualGeometryToolkitCore directly (and
    KernelAbstractions), not through the package's internal aliases.
[x] The chief ray is VisualRadiometryToolkitCore's `chief_ray` (one of its CONTRACT_GENERICS; `backproject` for a
    VisualGeometryToolkitCore camera): guide/calibrating.md, examples/README.md, guide/model.md, the module docstring's table.
[x] docs/src/index.md: `RadiometricCameraModel`, `Focus`, `π̂⁻¹` and `Material` are VisualRadiometryToolkitCore's (the camera
    re-exported); the package adds `center_constraint`, `render_view` and the calibration. README and landing: the package lifts a
    Visual Geometry Toolkit camera into `RadiometricCameraModel` and builds on VisualRadiometryToolkitCore and IO; the README's stack
    paragraph names VisualGeometryToolkitEstimation; the API surface page is in both reference lists.
[x] Install text (README, docs/src/index.md, guide/setup.md, site/src/index.md): RadiometricCamera in CauRegistry, the Visual
    Geometry Toolkit in General, and CauRegistry serving the Visual Radiometry Toolkit until it is registered in General: true
    before and after that registration. guide/setup.md adds the registry before the clone's instantiate.
[x] Wording: "scene" retired where it meant the board (load_session, the walkthrough, getting-started's heading, internal
    docstrings and comments of the kernels, the LM blocks, the target radiometry); examples/README.md names `EnvironmentLight` /
    `SphericalHarmonicMap` and the camera's `geometric` part; `View(pose, camera)` in the module docstring.
[x] load_session's docstring states its return value (the NamedTuple's fields).
[x] No pre-public version numbers, owner rulings or host paths in public text: api/pipeline.md, and the docstrings, comments
    and error messages of src/ (api.jl, options/retired.jl, options/toml.jl, io/records.jl, camera/observation_model.jl,
    camera/models/grid_camera.jl and others) say "an earlier release" where the meaning needs it. Two test expectations followed
    the reworded record refusal (test/unit/records_tests.jl, test/unit/pipeline_options_tests.jl); no other test asserts a changed
    string (every literal of an `occursin` / `@test_throws` checked against the old and new sources). docs/design/camera-first-design.md
    names no host and no cluster path.
[x] Bare "Core" / "IO" qualified as VisualGeometryToolkitCore / VisualGeometryToolkitIO: api/pipeline.md, io/records.jl,
    io/camera_models.jl, guide/model.md, the module docstring.
[x] api/pipeline.md: `Private = false` on every @autodocs block; api/surface.md: `RadiometricTarget` listed as exported, no
    internal owner tag.
[x] Docs build: `julia --project=RadiometricCamera/docs RadiometricCamera/docs/make.jl` (checkdocs = :public, doctests, the
    @example blocks, linkcheck, no warnonly): green.
[x] Test items (CPU, RC_TEST_GPU=0): docs, README (the fresh-environment run of the install line and the example, compared with
    the landing page and the site), API surface, quality: naming, layout, records, pipeline options, the lattice, focus-start,
    vignetting-centre, sensor-model, refinement-start and image-region items: all pass (one pre-existing broken).

Open, not part of this pass:
- examples/radiometric_calibration.ipynb is built from the script and executed on the published data with a GPU
  (build_notebook.py, then nbconvert): not rebuilt here, so its import cell still shows the earlier form until it is re-run.
- RadiometricCamera/Project.toml still states version 0.2.0; the bump to 0.2.1 belongs to the release.
