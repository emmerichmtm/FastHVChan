# A Numerical Formulation of Chan's Hypervolume Algorithm

This report is written as a self-contained research manuscript for readers
from the Multiobjective Optimization field and computational geometry.
It uses 12-point text, A4 pages, a vector illustration, and inline `bibitem`
references.

- [PDF](numerical_product_chan.pdf)
- [Main LaTeX source](numerical_product_chan.tex)
- [Experimental section](numerical_product_chan_experiments.tex)
- [Larger 4D–6D experimental section](numerical_product_chan_larger.tex)
- [Product-measure and magnitude appendix](numerical_product_chan_magnitude.tex)
- [Complete LaTeX source bundle](numerical_product_chan_source.zip)

The main text develops the algorithm directly for ordinary hypervolume
under Lebesgue measure: interval lengths, prefix sums, monotone-cutoff
contraction, block-mass compression, and the geometric recursion. No
magnitude transformation is required by the algorithm or its proof.

The magnitude viewpoint encouraged this algebraic formulation, as documented
in the earlier [HV4DMagnitude report](https://github.com/emmerichmtm/HV4DMagnitude/blob/main/paper/magnitude_chan_pair_elimination_report.pdf).
Appendix A preserves the metric definition, product computing-measure proof,
both HV transformations, boundary cases, and use of the shared engine.
This is an additional application, not a claimed source of an HV speedup.

The main complexity theorem proves that a locally restarted compression schedule retains
Chan's `O(n^(d/3) polylog(n))` arithmetic bound for fixed `d >= 4`.
It allows a compression to be skipped only when the current stored state
certifies the required size bound. Both outcomes restart the local budget.
The compiled implementation is
[`numerical_chan_local_compiled.py`](../numerical_chan_local_compiled.py),
with [verification records](../local_compiled_verification.json).

The first experimental tables now compare the proved local-checkpoint
policy with the earlier compiled policy and both Python Chan references
at 64–1,024 points in 4D–6D, plus a 4,096-point 4D stress test. They retain
partial results and timeouts explicitly. See the
[larger benchmark report](../benchmarks/larger_4d_6d/RESULTS.md).
The archived small-input timing tables remain separate and concern the
earlier global-interval adaptive driver at commit `38587e7`. The report
does not infer complexity from either set of measurements.

From this directory, using a standard TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error numerical_product_chan.tex
pdflatex -interaction=nonstopmode -halt-on-error numerical_product_chan.tex
```

Alternatively, `tectonic numerical_product_chan.tex` handles the repeat
passes automatically. The four `.tex` files are sufficient: there is no
external bibliography database or image dependency. The source bundle
contains all four files and this guide. This is an arXiv-style source package;
it has not been submitted to arXiv.
