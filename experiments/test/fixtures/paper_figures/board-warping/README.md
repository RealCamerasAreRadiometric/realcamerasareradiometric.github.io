The supplement's board-warping figure (fig:board-warping; scripts/board_warping.jl, src/board_warping.jl, the drawing code
src/paper_figures/board_warping.jl — the port of T. Harder's board_warping.py) from the calibrated targets of the refinements it
was drawn from. The test board_warping_tests.jl runs the figures stage's work on these records and checks that it writes the table
and the paper's files: every line but the first, which names the generator.

- products/<key>/radiometric_target.json: the calibrated target of a refinement (the page, the warping's 5 × 5 nodal heights, the
  print's radiometry), copied unchanged from its refine product `<key>`. The six refinements are the ones the figure shows: the
  spline camera's on the six capture sets (AprilGrid and Siemens stars, each with the Sigma 14 mm, the fisheye and the Tamron 20 mm)
  of RC 0.11.5's run of record (commit 16f7dcd9, 2026-11-16). They are 3 kB each; the height fields the stage samples from them
  (201 × 201 samples, 320 kB each) are not stored.
- cases.tsv: the six cases in the figure's order, row by row — case, set, camera model and the product's folder, relative to this
  one.
- expected/meta.toml: the per-case tables of the figure folder for these six cases (board_warping.py tables on the height fields
  that RC 0.11.5 sampled from the same records; copied from its data/meta.toml): the page's size and the surface's range and RMS.
- expected/board-warping.tex, expected/board-warping/<set>.tex, expected/board-warping/key.tex: the paper's files (board_warping.py
  figure3d). The paper's caption is written by hand and is not here.
