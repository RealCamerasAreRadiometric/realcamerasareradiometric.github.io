Expected values for what the paper-figure generators share (src/paper_figures/common.jl), each written by what the Python generators
used: Python's own number formatting and NumPy 2.2.6. The test paper_figures_tests.jl checks that common.jl gives the same, digit
for digit.

- python_format.tsv: 95 values and Python's formatting of each (the columns are named in its first line). A selection of a list
  of 36,602 values: its 11 special values (99.6, 0.105, 995.0, …), lines on which Julia's own formatting (Printf's %g, the string
  of a Float64) differs from Python's, and every 900th line.
- numpy_sum.tsv: numpy.sum of a[i] = (-1)^i · 1e6 / i + 1/3, i = 1 … n, for eleven lengths n around NumPy's block sizes (8, 128,
  8192), and the same vector summed from left to right (it shows that the test's vector is NumPy's, bit for bit).
