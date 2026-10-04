# Numerical Chan recursion for hypervolume and dominated-set magnitude

This report is written as a self-contained research manuscript for both
evolutionary multiobjective optimization and computational geometry readers.
It uses 12-point text, A4 pages, a vector illustration, and inline `bibitem`
references.

- [PDF](numerical_product_chan.pdf)
- [Main LaTeX source](numerical_product_chan.tex)
- [Experimental section](numerical_product_chan_experiments.tex)
- [Complete LaTeX source bundle](numerical_product_chan_source.zip)

The report explains the metric definition of magnitude, proves its product
computing-measure formula on finite dominated regions, and gives the HV
transformation in both directions, including zero coordinates and scaling.
It derives exact cell-mass discretization, monotone-cutoff contraction,
block-mass compression, and the geometric recursion.

Theorem 10.1 proves that a locally restarted compression schedule retains
Chan's `O(n^(d/3) polylog(n))` arithmetic bound for fixed `d >= 4`.
It allows a compression to be skipped only when the current stored state
certifies the required size bound. Both outcomes restart the local budget.
The compiled implementation is
[`numerical_chan_local_compiled.py`](../numerical_chan_local_compiled.py),
with [verification records](../local_compiled_verification.json).

The archived timing tables concern the earlier global-interval adaptive
driver at commit `38587e7`. They are explicitly not timings of the new local
policy. The report does not infer complexity from those measurements.

From this directory, using a standard TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error numerical_product_chan.tex
pdflatex -interaction=nonstopmode -halt-on-error numerical_product_chan.tex
```

Alternatively, `tectonic numerical_product_chan.tex` handles the repeat
passes automatically. The two `.tex` files are sufficient: there is no
external bibliography database or image dependency. The source bundle
contains both files and this guide. This is an arXiv-style source package;
it has not been submitted to arXiv.
