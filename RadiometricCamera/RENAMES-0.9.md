# RC 0.9.0: public renames and removals (running list for experiments / Agent C)

Old → new. No aliases are kept.

| Old | New | Why |
|---|---|---|
| `VgtkEstimation.image_size(camera)` (Estimation deleted it) | `VgtkCore.extent(camera)` (a `Size2{Int}`; `W, H = extent(c)` destructures) | Est 0.11 E5: one name for a camera's pixel extent |
| `load_session(...)`'s field `image_size::Tuple{Int,Int}` | field `extent::Size2{Int}` = `VgtkCore.extent` of the records' `camera.json` camera (full resolution, not downsampled) | the same concept as Core's `extent`; named and typed as Core's |
| `RC._schoeps_image_size(opts, name)` (private) | `RC._schoeps_extent(opts, name)` | the start camera's extent (checked against the first photograph) |
| `READ_SCHEMAS` (exported) | removed: each kind reads one version, `RECORD_SCHEMAS.<kind>` (`get(read_record(p), :schema, nothing) == RC.RECORD_SCHEMAS.calibration_state`) | IO 0.10 envelope; ruling (A) |
| `rc/radiometric_camera/v2`, `rc/radiometric_target/v1`, `rc/calibration_state/v2`, `rc/run_parameters/v1` | `rc/radiometric_camera/v3.0`, `rc/radiometric_target/v2.0`, `rc/calibration_state/v3.0`, `rc/run_parameters/v2.0` (IO 0.10's envelope: `units`/`conventions`, no top-level `length_unit`/`pixel_frame`) | an older RC record is refused by name ("written by RC ≤ 0.8 on IO 0.9: regenerate it"), except rc/calibration_state v1/v2 (one version step, ruling (A'), for the detector-bias GT) |
| a record's blur blocks `{"unit": "px^2" \| "mm^2", "covariance"}` | `{"pixel_frame": "colmap", "covariance"}` (in-focus blur) / `{"frame": "target:board", "covariance"}` (print blur; `RC.PRINT_PLANE`) | IO 0.10 blur body names its plane |

Kept, with a different concept:
- `raw_image_size(path)`: the size of a RAW file's mosaic on the raw grid (a photograph's size, not a camera's extent).
- the run record `parameters.json` key `"image_size"`: a key in the stored record, not a binding.
