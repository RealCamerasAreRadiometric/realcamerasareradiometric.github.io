# Fixture `lenssim_fisheye3` — three synthetic fisheye views of blob board 291 with rendered truth

Used by the BlobBoards-stack migration tests (`test/integration_blobboards/`): the only fixture on which a moved blob
detection can be told from a moved camera, because every pixel's true board position is known.

| file | what |
|---|---|
| `lsc_v1_ref.png`, `lsc_v2_ref.png`, `lsc_v3_ref.png` | the three training views (1024², 8-bit sRGB), BlobBoards' session input |
| `gtrays/board_<view>_uv.exr`, `board_<view>_flat.exr` | rendered truth of the training views and of the held-out views `val1`–`val3`: per pixel the board UV that pixel imaged, premultiplied by the coverage (`uv = board_uv / board_flat`; Blender UV, `v` bottom-up, so the board chart is `(u, 1 − v)`); pixels off the board have `flat = 0` |
| `manifest.json` | the render's lens (LensSim model of the Sigma 15 mm f/2.8 fisheye, `f_model_px` 426.7, 1024² sensor of 36 mm), the board frame in Blender world (origin, `u_axis`, `v_axis`; the board is 2 × 2 m), every view's true pose (`pose_true`: `R_cw_cv` rows, `t` = camera centre, metres) and split |
| `board_291/` | the BlobBoards export of the printed board 291 (`blobboards_board.jl pattern`): `pattern.png` at native print resolution, `pattern_layout.json` (blob centres and radii in chart units), `geometry.json` (the pattern rectangle in BlobBoards' board frame, mm) — the texture the views show, and the board every consumer of a session on this board needs |
| `session_io/` | the session of the three training views as VisualGeometryToolkitIO records — `camera.json` (cameras/v4: the refined division camera, COLMAP pixels), `targets_1.json` (rig/v6: board 291, its canvas and texture; `textures/291.png` is a link to `board_291/pattern.png`, the same bytes), `target_poses.json`, `trajectory_1.json` — written once by CaliBlobs' `scripts/migrate_rc_session.jl` from the BlobBoards session export (2026-10-28); the input of stage C |

The images were rendered with Blender + LensSim by the old repository's `scripts/blender_lenssim_targets.py` and the truth maps
by `scripts/blender_lenssim_gtboard.py` (branch `experiments-old`, dataset `data/lenssim_fisheye_blobs`; 15 training and
6 held-out views there, of which 3 + 3 are kept here, 30 MB). The entrance pupil of the traced lens sits 43.25 mm in front
of Blender's camera origin (`PUPIL_Z` in the tests). BlobBoards reconstructs the scene in board millimetres (the 2 m board is
the 130 mm board 291), so a session's metric distances are 130/2000 of the render's; the tests compare shapes, not scales.
