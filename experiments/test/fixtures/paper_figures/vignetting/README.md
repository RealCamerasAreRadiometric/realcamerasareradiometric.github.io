The paper's vignetting figure (fig:vignetting; T. Harder's vignetting.py), the tables it is drawn from and the learned vignettings
behind them. The test vignetting_figure_tests.jl checks that src/paper_figures/vignetting.jl writes the files of expected/ from
tables/ (every line but the generator line), and that the stage's functions (src/vignetting_figure.jl) give the samples, the learned
and the thin lens curves of tables/ from learned/ and write the folder tables/ byte for byte.

- learned/<lens>.json: the vignetting each lens's refinement learned, a vgio/vignetting record (VisualGeometryToolkitIO.save): the
  cases f4_14mm_spline_warped (refine product 64950cb2ad025100), f4_fisheye_spline_warped (350c8458977a2efb) and
  f4_tamron_spline_warped (0ca9c03ba11ff939) of configs/experiments/radiometric_refinement_vignetting.toml in the run of 2026-11-16
  (RadiometricCamera 0.11.5, commit 16f7dcd9), read from their radiometric_camera.json by RadiometricCamera 0.2.0.
- tables/<lens>_diagonal.tsv: the tables the paper's figure was drawn from, as they are (t, flat, learned, cos4). Their columns t,
  learned and cos4 are what `vignetting_diagonal` gives from learned/ with the settings of the config of record. Their column flat
  cannot be made again here: it was read from a decoded copy of each diffuser photograph (the green channel of a demosaiced 16-bit
  PNG, 33 to 53 MB per lens), where the stage reads the RAW photograph's green photosites; and their diagonals end 15 px inside an image
  circle the figure's folder carried per lens (14 mm: centre (2131.1, 1407.9), radius 1680.1 px; fisheye: (2126.8, 1434.9),
  1060.3 px; the Tamron's in the frame), where the stage asks the refined camera for its lens coverage. The test states both.
- tables/meta.toml: the tables' description as the stage's writer (`write_vignetting_tables`) writes it for these curves. Its f_px
  are the thin lens model's focal lengths the submitted figure was drawn with (1755.0, 933.2 and 2484.1 px; carried in pixels from
  an earlier run's tables), its f_mm the same in millimetres on the FX3's sensor (35.6 mm over 4256 px), to four decimals. The stage
  takes the focal length from each lens's refinement case instead (its camera.f_mm: 14.6776, 7.8057 and 21.1975 mm, that is 1754.7,
  933.2 and 2534.2 px), so its cos4 column differs from these tables': by at most 9.7e-5 (14 mm), 1.6e-5 (fisheye) and 0.012 (Tamron)
  over the tables' samples.
- The stage's own figure is therefore not the submitted one, on purpose. On the refinements above its tables differ from tables/
  in three ways: the flat field read at the RAW green photosites differs from the decoded copy's by at most 8.4e-4; the diagonals
  of the 14 mm and the fisheye end at the refined camera's lens coverage less 15 px and are 78 px and 36 px longer (1705 and 1063
  samples for 1666 and 1045); and the thin lens curve is drawn with the refinement's focal length (the differences above).
- expected/vignetting.tex and expected/vignetting/*.tex: the submitted figure's files as they are (the manuscript's src/fig/; the
  copies under paper/fig/ of this repository differ from them in the generator line and in the default of \rcfigdir, fig instead
  of src/fig).
- expected/vignetting-caption.tex: the caption the Python generator wrote from these tables (the paper's caption is written by hand).
