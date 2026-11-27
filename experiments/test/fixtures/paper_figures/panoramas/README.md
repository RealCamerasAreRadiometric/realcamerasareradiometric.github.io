The supplement's panoramas figure (panoramas-james.tex; T. Harder's panorama_pgf.py) on SYNTHETIC panoramas. The paper's six
panoramas are 3600 px wide (20 MB) and are not stored here. The test panorama_figure_tests.jl checks that
src/paper_figures/panoramas.jl writes from the small panoramas below what the original Python generator wrote from them (Python 3
with NumPy 2.2.6 and Pillow 12.3.0; the same Python, run on the paper's panoramas, writes the paper's files byte for byte): every
line but the generator line, every pixel.

- panoramas/<lens>/panorama_original.png and panorama_learned.png (fourteen 252 × 99 px, fisheye 252 × 117 px, tamron
  200 × 50 px), each folder in the place of a lens's panoramas product: a made-up scene of smooth ramps over the whole range of
  values, a texture and hard bright and dark edges; "original" darkens every frame towards its edges, as a panorama of uncorrected
  frames does. The sizes are chosen so that each panel's height falls on a half (38.5, 45.5 and 24.5 px at a width of 98 px),
  which Python rounds to the even number.
- expected/three-lenses/: what panorama_pgf.py wrote for the three lenses at a panel width of 98 px with the display curve of
  record (gain 2.5, gamma 1.5): the figure's file, its six panels' files and its caption.
- expected/two-lenses/: what it wrote for tamron and fourteen, in this order, with gain 4, gamma 2.2 and a panel width of 64 px.
- expected/<case>/pixels.sha256: the panels' images the Python wrote (panoramas-james/<panel>-<column>.png, 8-bit RGB), not stored
  as files: per image a line with the SHA-256 of its decoded pixels (row by row, three bytes per pixel), its size as
  <width>x<height> and its path.
