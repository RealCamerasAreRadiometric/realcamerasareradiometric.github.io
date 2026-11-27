The paper's learned-texture figure (fig:texture-learned, Fig. 7 right; T. Harder's texture_learned.py) with the records of the
texture run it is drawn from, cut to what the figure reads. The test texture_learned_figure_tests.jl checks that the generator's
two parts (src/paper_figures/texture_learned.jl) and the stage bands of scripts/texture_recovery.jl write the files of expected/
from run/ and photograph/: every line but the generator line, every pixel.

- run/: the three records of the texture run of 2026-11-16 (RadiometricCamera 0.11.5, commit 16f7dcd9) the stage reads, reduced.
  - run/evaluate/ (the evaluate product 16f9d84088c7714b): design.png, texture_pinhole.png and texture_learned.png are the run's
    827 × 1169 pages with every texel white but those of the text block (x 100 … 727) in the rows 100 … 214 — the text lines 1 to
    3, which the figure shows, and the first rows of line 4, which end the gap the bands are cut in (74 kB instead of 1.2 MB).
    report.json holds the three numbers read of the run's report (tex_mm, TW, TH).
  - run/init/poses.json (the init product a2902f6141e57d0e): of every photograph its name, rotation, translation and distance.
  - run/refine/page.json (the refine product 270c783b87521b12): the photographs and their roles.
- photograph/: the band "training photograph" as the author-side step wrote it (experiments/tools/texture_learned_photograph.py on
  that run; configs/inputs/texture_learned_photograph_20261121.toml): photograph.png and its record. They are the inputs of record
  texture_learned_photograph and texture_learned_photograph_record of configs/experiments/texture_recovery_14mm_small.toml, byte
  for byte (the test reads them through the config's sha256). The record is the step's meta.toml; it is stored as meta.toml.txt
  because it states two facts as TOML booleans (read_on, rectified) and the suite's lint of the switch convention reads every
  .toml under test/fixtures.
- tables/meta.toml.txt: the tables' record the submitted figure was drawn from (the Python's step `tables` on the whole run;
  stored as .txt for the same reason). From run/ the step writes every line of it but the first (it names the writer) and the six
  numbers of [errors.*]: those are taken over the whole text block, which run/ holds the first lines of.
- expected/texture-learned.tex and expected/texture-learned/*.tex: the submitted figure's files as they are (the manuscript's
  src/fig/; the copies under paper/fig/ of this repository differ from them in the default of \rcfigdir, fig instead of src/fig).
- expected/pixels.sha256: the submitted figure's four images (texture-learned/<band>.png, 8-bit grey), not stored as files: per
  image a line with the SHA-256 of its decoded pixels (row by row, one byte per pixel), its size as <width>x<height> and its path.
- expected/texture-learned-caption.tex: the caption the Python generator wrote from these tables (the paper's caption is written by
  hand).

The pages of run/evaluate/ were cut by a few lines of Python (Pillow) from the run's files; the other files of run/ are the run's
JSON records with the keys the stage does not read left out, every number as the run wrote it.
