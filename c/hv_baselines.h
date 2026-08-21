/* hv_baselines.h -- classical exact hypervolume baselines in C99.
 *
 * C ports of hv_baselines.py from https://github.com/emmerichmtm/FastHVChan:
 *
 *  - fhv_hypervolume_ds : dimension sweep.  d=3 base case is the classical
 *    O(n log n) sweep over a 2-D staircase; higher dimensions sweep the last
 *    objective and maintain the cross-section volume through exclusive
 *    contributions, following Guerreiro, Fonseca, Emmerich (CCCG 2012) in
 *    structure (their specialised 4-D bookkeeping achieves O(n^2); this port
 *    recomputes contributions, giving O(n^2 log n) at d=4).
 *
 *  - fhv_hypervolume_wfg : the WFG algorithm of While, Bradstreet, Barone
 *    (IEEE Trans. Evolutionary Computation 16(1):86-95, 2012): recursive
 *    exclusive-hypervolume over dominance-pruned limit sets.
 *
 * Same conventions as fasthvchan.h: points are n x dim row-major double
 * arrays; objectives are minimised unless `maximize` is nonzero; `prefilter`
 * drops dominated points first; only points strictly better than the
 * reference in every objective contribute.  Returns FHV_ENOMEM (negative)
 * on allocation failure.  Not thread-safe (shared qsort key globals).
 */
#ifndef HV_BASELINES_H
#define HV_BASELINES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

double fhv_hypervolume_ds(const double *points, size_t n, int dim,
                          const double *reference, int maximize, int prefilter);

double fhv_hypervolume_wfg(const double *points, size_t n, int dim,
                           const double *reference, int maximize, int prefilter);

#ifdef __cplusplus
}
#endif

#endif /* HV_BASELINES_H */
