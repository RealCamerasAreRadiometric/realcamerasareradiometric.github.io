The board warping across lenses (scripts/board_warping.jl on configs/supplement/board_warping_lenses.toml) on REDUCED height fields:
the paper's figure with the board's offsets (the drawing code src/paper_figures/board_warping_offsets.jl — the port of T. Harder's
board_warping_offsets.py) and the session's heights alone (src/paper_figures/board_warping_lenses.jl — the port of his
board_warping_lenses.py). The six
height fields are 201 × 201 samples each (2 MB together). The test board_warping_tests.jl checks that the drawing code writes from
the data below what the Python generator wrote from them (Python 3 with NumPy 2.2.6): every line but the first, which names the
generator.

- data/dump/<case>.f64: every 5th sample of the case's height field (41 × 41 samples; the warping's 5 × 5 nodes stay on samples);
  data/dump/<case>_nodes.tsv: the nodes as dumped. The six cases are the spline refinements of the board-warping session's sets:
  two boards (AprilGrid, Siemens stars), three lenses (Sigma 14 mm, fisheye, Tamron 20 mm), refined with RC 0.11.5.
- data/cases.tsv: the cases, each refine product named relative to the data folder (products/<key>).
- data/products/<key>/radiometric_target.json: of the refinement's target record the one entry the tables read
  (inplane_print_error).
- expected/3d/: board_warping_lenses.py figure3d <data> <out> — the frame, the six panels, the colour key, the caption and
  source.toml.
- expected/tables/meta.toml: board_warping_lenses.py tables <data>, run inside data/ (the product folders are relative to it).
- expected/offsets/: the paper's figure of the same data, the board's offsets as surfaces and maps (src/paper_figures/
  board_warping_offsets.jl — the port of T. Harder's board_warping_offsets.py, its variant b6 with large surfaces, the surfaces
  first and maps of 0.40 of a surface's width, the view's elevation 25 degrees), run inside data/: the frame, the six surfaces, the
  six maps' pictures, the two keys, the caption and source.toml as the drawing code writes them — the Python generator's files
  of the same data in every line but the first, with its two folders as one (its maps' key.tex is maps-key.tex here).
  pixels.sha256: of each map's image (123 x 123 pixels) the SHA-256 of its decoded pixels; the images are the Python's in
  every pixel and are not stored.
- expected/offsets-tables/meta.toml: that drawing's tables of the same data — expected/tables/meta.toml's blocks in the figure's
  order and, per board and pair of lenses, the difference between the board's offsets learned with the two lenses.
