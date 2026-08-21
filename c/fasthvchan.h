/* fasthvchan.h -- exact hypervolume and Klee's measure, Chan (FOCS 2013).
 *
 * C port of chan_hypervolume.py from https://github.com/emmerichmtm/FastHVChan
 * (the Section-2 "practical" algorithm).  Semantics follow the Python source
 * exactly, including its tie-breaking and degenerate-input behaviour.
 *
 * Points and box corners are plain double arrays of length `dim`; a set of `n`
 * points is a single contiguous array of n*dim doubles, row-major.
 */
#ifndef FASTHVCHAN_H
#define FASTHVCHAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of recursive nodes used by the most recent call, for experimenting
 * with the base case.  Mirrors ChanMeasure.node_count. */
extern long fhv_node_count;

/* Base case of the recursion: instances with at most this many boxes are
 * settled by inclusion-exclusion.  Defaults to 2, as in the Python source.
 * Values above ~20 make the base case explode; it is exponential. */
extern int fhv_base_case;

/* Hypervolume indicator of `points` (n x dim, row-major) w.r.t. `reference`.
 *
 * Objectives are minimised unless `maximize` is nonzero.  Only points strictly
 * better than the reference in every objective contribute.  `prefilter` drops
 * dominated points first: it never changes the result but usually shrinks the
 * input, at O(n^2 d) cost.  Returns 0.0 if no point dominates the reference.
 *
 * Returns a negative value on allocation failure (FHV_ENOMEM). */
double fhv_hypervolume(const double *points, size_t n, int dim,
                       const double *reference, int maximize, int prefilter);

/* Volume of the union of `n` axis-parallel boxes (Klee's measure problem).
 *
 * `los` and `his` are each n x dim, row-major.  `domain_lo`/`domain_hi` restrict
 * the volume to a box; pass NULL for both to use the bounding box of the input,
 * which restricts nothing.  Degenerate (non-positive volume) boxes are ignored,
 * and an empty or fully degenerate input gives 0.0. */
double fhv_union_volume(const double *los, const double *his, size_t n, int dim,
                        const double *domain_lo, const double *domain_hi);

/* Filter `points` to its non-dominated front (minimisation; ties dropped).
 *
 * Writes the kept points into `out` (capacity n*dim) and returns how many were
 * kept, or a negative value on allocation failure.  `out` may not alias
 * `points`.  Negate the objectives to use this for maximisation. */
long fhv_nondominated(const double *points, size_t n, int dim, double *out);

/* Brute-force hypervolume by inclusion-exclusion over all 2^n subsets, for
 * cross-checking.  Exponential -- keep n below about 20. */
double fhv_reference_hypervolume(const double *points, size_t n, int dim,
                                 const double *reference);

#define FHV_ENOMEM (-1.0)

#ifdef __cplusplus
}
#endif

#endif /* FASTHVCHAN_H */
