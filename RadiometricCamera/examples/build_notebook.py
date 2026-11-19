#!/usr/bin/env python3
"""Build radiometric_calibration.ipynb from radiometric_calibration.jl: every block of `# ` comment lines becomes a markdown cell,
every other block a Julia code cell (Literate.jl convention, no dependency). Execute afterwards with
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=<julia kernel of the experiments environment> \
      --ExecutePreprocessor.timeout=3600 examples/radiometric_calibration.ipynb
"""
import json, os, uuid
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radiometric_calibration.jl")
cells, kind, buf = [], None, []
def flush():
    global buf
    text = "\n".join(buf).strip("\n")
    if text:
        if kind == "md":
            cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text})
        else:
            cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [], "source": text})
    buf = []
for line in open(src, encoding="utf-8"):
    line = line.rstrip("\n")
    k = "md" if line.startswith("# ") or line == "#" else "code"
    if k != kind:
        flush(); kind = k
    buf.append(line[2:] if k == "md" else line)
flush()
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Julia (experiments)", "language": "julia", "name": "julia-hyper-accuracy"},
                                   "language_info": {"name": "julia"}}, "nbformat": 4, "nbformat_minor": 5}
out = src.replace(".jl", ".ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells")
