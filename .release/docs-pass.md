docs-pass: 4f6998a
landing: site/ of this repository (site/src/index.md, deployed by the Site workflow with the package manual under /docs/): the install lines changed in this pass

# RadiometricCamera 0.2.0 docs pass at the release (julia-docs-pass), resolved from the registries alone

The release tree is the port onto the Visual Radiometry Toolkit (branch `vrtk/port`) with no temporary [sources] in the
package or its workspace members (only the members' own path to the package). Registered before it: 0.1.0 (CauRegistry;
the code of the private 0.11.6). The next minor of a 0.x package: 0.2.0, breaking.

[x] API diff since 0.1.0 (CHANGELOG 0.2.0, against `git diff 247c089..HEAD -- RadiometricCamera` and the composition
    commits before it):
    - moved to VisualRadiometryToolkitCore, no longer public here: the radiometric camera's functions, the focus surface,
      the sensor quadrature, the material, the board (`PrintedTarget` → `TexturedTarget`, `BoardSurface` → `TargetState`),
      the vignetting models and the light (were VisualGeometryToolkitCore's); `RadiometricCameraModel` stays exported;
    - removed: `Scene`, `with_surface`, the alternative focus parameterisations with `focus.variant`, `focus.node_count`,
      `focus.perturb` and the report `focus_ab.json`; `lens` of a camera or a state (`geometric`); `green_photosite_mask`;
    - re-signatured: the entry points take (camera, pose, board, …); the composed constructor's vignetting is `NoVignetting`;
    - added: `flat_vignetting`, `photosite_mask(m, channel)`, `io.channel = "R" | "G" | "B"` (fit the photosites of any one
      colour filter).
    Every removed name is refused by name with its replacement (src/api.jl), and no manual page, guide, README or site text
    names one as current (grep of docs/src, README.md, the repository README, site/src/index.md, experiments/*.md).
[x] Docstrings: the docs item (`test_docs`: every public name documented, attached, signatures accepted) and the API
    surface item (docs/src/api/surface.md against the module) in the release gate. By reading: four statements that still
    named VisualGeometryToolkitCore as the owner of the light or the radial vignetting, or the board's state as "the scene's",
    corrected (src/io/records.jl `load_state`, src/target/target_radiometry.jl `RadiometricTarget` and `board_surface`,
    comments in src/camera/vignetting.jl and src/options/tables.jl).
[x] Guides and API pages: docs/src/index.md (the vignetting models and the light are VisualRadiometryToolkitCore's; the
    packages it builds on name the Visual Radiometry Toolkit), api/target.md (the light's owner), guide/calibrating.md
    (a session gives the board, not "the scene"; the light's owner), guide/getting-started.md (the light's owner).
    api/camera.md, api/calibration.md, api/pipeline.md, api/surface.md, guide/model.md, guide/setup.md, guide/testing.md:
    read against the API diff, unchanged.
[x] README and landing pages: the install lines are the registries' (`registry add General <CauRegistry>`, then
    `add RadiometricCamera VisualGeometryToolkitCore VisualRadiometryToolkitCore`) in RadiometricCamera/README.md,
    docs/src/index.md and site/src/index.md; the repository README installs the same way. The example is unchanged (one
    example, shown three times, compared by the README item). The README item runs in the release gate: the install line's
    packages resolve from the depot's registries. True for a user once 0.2.0 is in CauRegistry (0.1.0 there is on
    VisualGeometryToolkitCore 0.11, which VisualRadiometryToolkitCore 0.1 does not admit).
[x] site/: only the install paragraph and line changed (they named the repository URLs of the package and of
    VisualRadiometryToolkitCore). Nothing else there states the API.
[x] docs/design/camera-first-design.md: kept as a dated report (2026-11-20), marked at its top: it names `Scene`,
    `BoardSurface` and this package as the owner of the generic layer.
[x] CHANGELOG: "Unreleased" is 0.2.0 (2026-11-21), with the breaking changes in short, the installation, the dependencies
    and the records' vignetting and light bodies (VisualRadiometryToolkitIO's); the "Temporary [sources]" entry is gone.
[x] Docs build (docs/make.jl: checkdocs = :public, doctests, linkcheck, no warnonly): in the release gate.

Open, not part of this pass:
- CHANGELOG 0.1.0 says "the first version registered in Julia's General registry"; 0.1.0 is registered in CauRegistry and
  General holds no RadiometricCamera. Left as written (a past release's section; the owner's wording).
- The CompatHelper workflow adds General only: it does not see CauRegistry's packages.
