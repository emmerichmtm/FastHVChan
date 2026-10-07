# A Numerical Formulation of Chan's Hypervolume Algorithm

Michael Emmerich. 5 October 2026; revised 7 October 2026.

This report is written as a self-contained research manuscript for readers
from the Multiobjective Optimization field and computational geometry.
It uses 12-point text, A4 pages, a vector illustration, and inline `bibitem`
references.

- [PDF](numerical_product_chan.pdf)
- [Main LaTeX source](numerical_product_chan.tex)
- [Main 4D–6D benchmark comparison](numerical_product_chan_larger.tex)
- [Structural validation and archive appendix](numerical_product_chan_experiments.tex)
- [Product-measure and magnitude appendix](numerical_product_chan_magnitude.tex)
- [Worked eight-point 4D example](numerical_product_chan_example.tex)
- [Example verification script](eight_point_example.py) and [checked results](eight_point_example.json)
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

Appendix B explains eight nondominated points in `{1,2,3,4,5}^4`, with
gridded TikZ projections, a hand-checkable planar table, the actual recursion,
a complete numerical leaf calculation, and an illustrative compression.
Their hypervolume is 192. The example script independently checks every
displayed numerical result; `--compiled` additionally checks both solvers
and the compiled recursion. Compression is illustrated separately because
this small input does not trigger a scheduled compression.

The report distinguishes Chan's geometric algorithm and symbolic functional
formulation from Michael Emmerich's FastHVChan implementation project.
The contribution is a compilable numerical implementation and its explicit
representation and analysis, preserving Chan's exponent.

The main complexity theorem proves that a locally restarted compression schedule retains
Chan's `O(n^(d/3) polylog(n))` arithmetic bound for fixed `d >= 4`.
It allows a compression to be skipped only when the current stored state
certifies the required size bound. Both outcomes restart the local budget.
The compiled implementation is
[`numerical_chan_local_compiled.py`](../numerical_chan_local_compiled.py),
with [verification records](../local_compiled_verification.json).

The two benchmark tables compare **Numerical Chan (Numba)** with **Chan
d/3 (Python)** and **Chan d/2 (Python)**. Numerical Chan is the current
local-checkpoint implementation covered by the theorem. The tables cover
64–1,024 points in 4D–6D, plus a 4,096-point 4D stress test, with timeouts
and unrun cases kept visible. See the
[larger benchmark report](../benchmarks/larger_4d_6d/RESULTS.md).
Earlier timing tables and additional implementation variants remain in
the repository archive, linked in Appendix D. They are omitted from the
main comparison. Raw measurements and solver code are unchanged; the
report does not infer complexity from timings.

From this directory, using a standard TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error numerical_product_chan.tex
pdflatex -interaction=nonstopmode -halt-on-error numerical_product_chan.tex
```

Alternatively, `tectonic numerical_product_chan.tex` handles the repeat
passes automatically. The five `.tex` files are sufficient: there is no
external bibliography database or image dependency. The source bundle
contains all five files, this guide, and the example verification files. This is an arXiv-style source package;
it has not been submitted to arXiv.
