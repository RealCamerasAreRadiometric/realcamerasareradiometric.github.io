Expected values for what the paper-figure generators share (src/paper_figures/common.jl), each written by what the Python generators
used: Python's own number formatting, NumPy 2.2.6, Pillow 12.3.0 and the figure folders' tools/common.py (T. Harder). The test
paper_figures_tests.jl checks that common.jl gives the same, digit for digit and pixel for pixel.

- python_format.tsv: 95 values and Python's formatting of each (the columns are named in its first line). A selection of a list
  of 36,602 values: its 11 special values (99.6, 0.105, 995.0, …), lines on which Julia's own formatting (Printf's %g, the string
  of a Float64) differs from Python's, and every 900th line.
- numpy_sum.tsv: numpy.sum of a[i] = (-1)^i · 1e6 / i + 1/3, i = 1 … n, for eleven lengths n around NumPy's block sizes (8, 128,
  8192), and the same vector summed from left to right (it shows that the test's vector is NumPy's, bit for bit).
- python_misc.txt: for the nine values 0.00012345, 0.0456, 1.5, 12.6, 123.4, 99.6, 1e-5, 0.105, 3.0 common.py's num(v), then
  repr(nice(v)); then xcol(flow_rgb(0.3, -0.2, 0.5)), keybox("t") and headbox("h").
- pillow_in.png: a 24 × 16 image with smooth parts and hard edges; pillow_lanczos_<w>x<h>.png: its Image.resize((w, h),
  Image.LANCZOS).
- bin_points.tsv: 300 vectors (x, y, dx, dy) on a 4000 × 3000 sensor, six of them on or beyond its edges; python_bins.tsv: their
  bin means in 4 × 3 bins of at least 20 vectors (common.py's bin_means and save_bins; three bins hold too few);
  python_field_0.png: those bins as an image of 48 × 36 px under the key radius 0.1 (field_rgb).
- python_panel.tex: common.py's image_panel, called as the test calls it.
