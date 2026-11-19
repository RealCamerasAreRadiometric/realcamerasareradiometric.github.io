The paper's convergence figure as in the paper (paper/fig/convergence-rms*, T. Harder's rms_figure.py)
and the per-iteration medians it plots (paper_conv.tsv: the detections' true reprojection medians and the held-out Thorlabs medians of
si30_spline_warped, convergence 83de712a60fd92ef, and ag30_spline_warped, convergence 9878a81935d84404; dgx01 2026-11-17). The test
convergence_tests.jl checks that src/convergence_panels.jl writes these files from these numbers, every line but the generator line.
