# Makefile for the paper (the release build; main.tex in this directory)
#   make              # = make all: build/paper.pdf and build/supp.pdf
#   make pdf          # build main.tex -> build/main.pdf (paper + supplement as one document, so the
#                     #   cross-references between them resolve, as the author kit intends)
#   make paper        # build/paper.pdf: the pages of build/main.pdf before the supplement
#   make supp         # build/supp.pdf: the supplement, from its "Supplementary Material" title page on
#   make figures      # regenerate every generated asset (../experiments/scripts/regenerate_paper.jl)
#   make figures-gpu  # ... plus the CUDA figures
#   make check        # the experiments' suite (../experiments/test/runtests.jl: a paper project has no UUID, Pkg.test refuses)
#                     # (REPRODUCE.md contract, configs)
#   make clean        # remove the build directory

# Every path in main.tex and the files it inputs is relative to this directory (sec/..., fig/..., latex/...).
PAPER_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BUILD     := build
LATEXMK   := latexmk
LFLAGS    := -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD)
TEXENV    := TEXINPUTS=.: BIBINPUTS=.: BSTINPUTS=.:

PROJ_DIR := $(PAPER_DIR)/../experiments
JL := julia --startup-file=no --project=.

.PHONY: all pdf paper supp figures figures-gpu check figcheck clean

all: paper supp

pdf: figcheck
	cd $(PAPER_DIR) && $(TEXENV) $(LATEXMK) $(LFLAGS) main.tex

# The supplement starts on the page cvpr.sty's \maketitlesupplementary prints ("Supplementary Material");
# the split reads that page from the compiled PDF, so main.tex keeps the author kit's one-document layout.
SUPP_PAGE = $$(n=$$(pdfinfo $(BUILD)/main.pdf | awk '/^Pages:/{print $$2}'); \
	for p in $$(seq 2 $$n); do pdftotext -f $$p -l $$p $(BUILD)/main.pdf - | grep -qx 'Supplementary Material' && { echo $$p; break; }; done)

paper: pdf
	@cd $(PAPER_DIR) && s=$(SUPP_PAGE); \
	if [ -z "$$s" ]; then cp $(BUILD)/main.pdf $(BUILD)/paper.pdf; echo "paper: no supplement in main.pdf; build/paper.pdf = main.pdf"; \
	else qpdf --empty --pages $(BUILD)/main.pdf 1-$$((s-1)) -- $(BUILD)/paper.pdf && echo "paper: build/paper.pdf = pages 1-$$((s-1))"; fi

supp: pdf
	@cd $(PAPER_DIR) && s=$(SUPP_PAGE); \
	if [ -z "$$s" ]; then echo "supp: build/main.pdf has no 'Supplementary Material' page (is sec/X_suppl.tex input in main.tex?)" >&2; exit 1; fi; \
	qpdf --empty --pages $(BUILD)/main.pdf $$s-z -- $(BUILD)/supp.pdf && echo "supp: build/supp.pdf = pages $$s-end"

# Staleness guard: warn when result-affecting code or a config of record changed
# in a commit after the last one that regenerated the numbers (latex/numbers.tex
# is the marker, regenerated in the same batch as every other asset). WARN, never fail.
figcheck:
	@root=$$(git -C $(PAPER_DIR) rev-parse --show-toplevel); \
	marker=$$(git -C $$root log -1 --format=%H -- paper/latex/numbers.tex 2>/dev/null); \
	if [ -n "$$marker" ] && ! git -C $$root diff --quiet "$$marker"..HEAD -- RadiometricCamera/src RadiometricCamera/ext RadiometricCamera/configs experiments/configs experiments/src 2>/dev/null; then \
	  echo "WARNING: result-affecting code or a config of record (RadiometricCamera/src/ ext/ configs/,"; \
	  echo "         experiments/configs/ experiments/src/) changed since the numbers were last"; \
	  echo "         regenerated ($$marker) - run 'make figures' before trusting the compiled numbers"; \
	fi

figures:
	cd $(PROJ_DIR) && $(JL) scripts/regenerate_paper.jl

figures-gpu:
	cd $(PROJ_DIR) && $(JL) scripts/regenerate_paper.jl --gpu

check:
	cd $(PROJ_DIR) && $(JL) test/runtests.jl

clean:
	rm -rf $(PAPER_DIR)/$(BUILD)
