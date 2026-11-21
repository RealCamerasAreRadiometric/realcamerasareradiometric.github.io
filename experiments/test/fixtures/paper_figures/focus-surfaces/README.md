The paper's focus-surface figure (fig:focus-surfaces; T. Harder's focus_surfaces.py) and the tables it is drawn from. The test
focus_surfaces_tests.jl checks that src/paper_figures/focus_surfaces.jl writes the files of expected/ from tables/: every line but
the generator line.

- tables/: the figure's tables as stage panels of scripts/focus_surfaces.jl writes them (`focus_panels`: the driver's
  `focus_surface_points` every 16 px, then the generator's part `tables`), here from the refined cameras of the two refinements the
  paper's figure shows — f4_fisheye_spline_warped (refine product 350c8458977a2efb) and sitla_spline_warped (c2b5da93643d54fe) of
  the run of 2026-11-16 (RadiometricCamera 0.11.5, commit 16f7dcd9), read by RadiometricCamera 0.2.0. fisheye.tsv and tamron.tsv are
  byte for byte the tables the paper's figure was drawn from (the Python's step `tables` on the dumps of the same two cameras);
  meta.toml differs from that one in the line that names its writer, in the key of the lattice's spacing (`spacing_px`, and the
  comment that names it) and in the two panel labels, which are the panels' titles (`label` in the config of record's `[panels]`).
- expected/focus-surfaces.tex and expected/focus-surfaces/*.tex: the submitted figure's files as they are (the manuscript's
  src/fig/; the copies under paper/fig/ of this repository differ from them in the generator line and in the default of \rcfigdir,
  fig instead of src/fig).
- expected/focus-surfaces-caption.tex: the caption the Python generator wrote from these tables (the paper's caption is written by
  hand).

The dumps the part `tables` reads (<case>.f64, 1.5 MB per camera) are not stored: the test of the stage writes them from two small
synthetic cameras and checks the tables against them.
