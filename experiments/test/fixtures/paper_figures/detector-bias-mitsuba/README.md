The paper's detector-bias figure (fig:detector-bias, Fig. 4; T. Harder's detector_bias.py) and the tables it is drawn from. The
test detector_bias_tests.jl checks that the part `figure` of src/paper_figures/detector_bias.jl, with the settings of the config of
record (configs/experiments/detector_bias.toml [figure]), writes the files of expected/ from data/: every line but the generator
line, every pixel.

- data/: the figure's tables as the part `tables` writes them from the evaluations of the paper's runs — rows.json and the ten bin
  tables <row>-<panel>.tsv. The paper's runs are the two cases of the config of record on RadiometricCamera 0.11.5 (commit
  16f7dcd9), with 6,000 and 16,528 samples per photograph: db_ocv14_mitsuba_fov (36 iterations) and
  db_spline_mitsuba_siemens_fisheye (25 iterations; the lens coverage scaled by 0.9 for the samples, the grid smoothness weight
  0.01). From those runs' evaluation records, the render pipeline's two data.npy and the two renders the part `tables` writes
  the submitted figure's thirteen table files byte for byte; the eleven text files here are those. The suite does not run that
  part: its inputs are up to 22 MB per evaluation record and 295 MB of detections. The two renders <row>-render.jpg here are
  reduced copies (266 × 178 px, 16 kB; the paper's are 1064 × 712 px, 264 kB): the part `figure` copies a render into the
  figure's folder and never reads it, and the test checks the copy.
- expected/detector-bias-mitsuba.tex and expected/detector-bias-mitsuba/*.tex: the submitted figure's files as they are (the
  manuscript's src/fig/; the copies under paper/fig/ of this repository differ from them in the generator line and in the default
  of \rcfigdir, fig instead of src/fig).
- expected/pixels.sha256: of each of the submitted figure's ten panel images (515 kB as files) the SHA-256 of its decoded pixels
  (8-bit RGB, row by row, three bytes per pixel), its size and its path.
- expected/detector-bias-mitsuba-caption.tex: the caption the Python generator wrote from these tables (the paper's caption is
  written by hand).
