# Makefile for brain-like learning paper

PAPER = brain_like_learning

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc

.PHONY: all clean
