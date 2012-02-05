all: preface.pdf acknowledgements.pdf technical.pdf morphology.pdf rgb_cmyk.pdf

.SUFFIXES: .tex .dvi .ps .pdf .mp .eps

%.pdf: %.tex hogg_atlas.tex
	pdflatex $<
	pdflatex $<
	pdflatex $<

GSRES= 200
hogg_hdf_label.ps: hogg_hdf_label.tex
	latex hogg_hdf_label
	dvips -E -Ppdf hogg_hdf_label -o

clean:
	\rm -vf *.log *.aux
