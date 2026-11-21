The paper's teaser (fig:teaser; T. Harder's teaser.py) and the pieces it is made of. The test teaser_tests.jl checks that the
stage's crops and src/paper_figures/teaser.jl write the files of expected/ from them: every line but the generator line, every
pixel of the strip.

- texture/: the four bands of the strip, as the learned-texture figure's tables hold them (truth.png, photograph.png,
  pinhole.png, ours.png; 8-bit grey, 77 kB).
- panoramas/: in the place of a lens's panoramas product, panorama_original.png and panorama_learned.png — stand-ins of the size
  of the paper's part of the panorama (460 × 250 px; every block of 10 × 10 pixels of the paper's two panels at its mean colour,
  5 kB instead of 215 kB; the panoramas themselves are 3600 × 725 px, 4.7 MB). The test crops them whole and unchanged, so the
  figure's files state the paper's panel size.
- expected/teaser.tex and expected/teaser/*.tex: the submitted figure's files as they are (the manuscript's src/fig/; the copies
  under paper/fig/ of this repository differ from them in the default of \rcfigdir, fig instead of src/fig).
- expected/pixels.sha256: of the submitted strip teaser/texture-strip.png (40 kB as a file) the SHA-256 of its decoded pixels
  (8-bit, as RGB, row by row, three bytes per pixel), its size and its path.
- expected/teaser-caption.tex: the caption the Python generator wrote for these pieces with the panels brightened by 1.2 (the
  paper's caption is written by hand).
