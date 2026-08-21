/* hv_baselines.c -- see hv_baselines.h.
 *
 * Ports the logic of hv_baselines.py exactly, including tie handling: the
 * dominance filter drops duplicates, 2-D-dominated staircase insertions gain
 * zero area, and clipped points touching the reference are discarded.
 */
#include "hv_baselines.h"
#include "fasthvchan.h" /* fhv_nondominated, FHV_ENOMEM */

#include <stdlib.h>
#include <string.h>

static int bl_err = 0; /* sticky allocation-failure flag */

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

static double pt_box_volume(const double *p, const double *ref, int dim)
{
    double volume = 1.0;
    int k;
    for (k = 0; k < dim; k++)
        volume *= ref[k] - p[k];
    return volume;
}

/* qsort context: the coordinate to order rows by. */
static int bl_sort_key = 0;

static int cmp_by_key(const void *a, const void *b)
{
    const double *p = (const double *)a, *q = (const double *)b;
    double x = p[bl_sort_key], y = q[bl_sort_key];
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}

static void sort_by_coordinate(double *points, size_t n, int dim, int key)
{
    bl_sort_key = key;
    qsort(points, n, (size_t)dim * sizeof(double), cmp_by_key);
}

/* Shared front end: mirror for maximisation, keep only points strictly
 * dominating the reference, optionally dominance-filter.  Writes the kept
 * points (n_out x dim) into a fresh allocation the caller frees. */
static double *prepare(const double *points, size_t n, int dim,
                       const double *reference, int maximize, int prefilter,
                       double *ref_out, size_t *n_out)
{
    double *front = (double *)malloc((n ? n : 1) * (size_t)dim * sizeof(double));
    size_t m = 0, i;
    int k;
    if (!front)
        return NULL;
    for (k = 0; k < dim; k++)
        ref_out[k] = maximize ? -reference[k] : reference[k];
    for (i = 0; i < n; i++) {
        const double *p = points + i * (size_t)dim;
        int inside = 1;
        for (k = 0; k < dim; k++) {
            double v = maximize ? -p[k] : p[k];
            front[m * (size_t)dim + k] = v;
            if (v >= ref_out[k])
                inside = 0;
        }
        if (inside)
            m++;
    }
    if (prefilter && m > 1) {
        double *filtered = (double *)malloc(m * (size_t)dim * sizeof(double));
        long kept;
        if (!filtered) {
            free(front);
            return NULL;
        }
        kept = fhv_nondominated(front, m, dim, filtered);
        free(front);
        if (kept < 0) {
            free(filtered);
            return NULL;
        }
        *n_out = (size_t)kept;
        return filtered;
    }
    *n_out = m;
    return front;
}

/* ------------------------------------------------------------------ */
/* Dimension sweep                                                     */
/* ------------------------------------------------------------------ */

static double hv2d(double *points, size_t n, const double *ref)
{
    double best, area = 0.0;
    size_t i;
    sort_by_coordinate(points, n, 2, 0);
    best = ref[1];
    for (i = 0; i < n; i++) {
        double y = points[i * 2 + 1];
        if (y < best) {
            area += (ref[0] - points[i * 2]) * (best - y);
            best = y;
        }
    }
    return area;
}

/* Insert (x, y) into the staircase (xs/ys sorted by x ascending, y strictly
 * descending); return the covered-area gain.  Mirrors the Python
 * _staircase_insert exactly. */
static double staircase_insert(double *xs, double *ys, size_t *len, double x,
                               double y, double rx, double ry)
{
    size_t i, j, lo = 0, hi = *len;
    double cover, gain = 0.0, cur, nx;
    while (lo < hi) { /* bisect_left on xs */
        size_t mid = (lo + hi) / 2;
        if (xs[mid] < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    i = lo;
    cover = i > 0 ? ys[i - 1] : ry;
    if (y >= cover)
        return 0.0; /* 2-D dominated: no new area */
    cur = x;
    j = i;
    while (j < *len && ys[j] >= y) {
        nx = xs[j];
        gain += (nx - cur) * (cover - y);
        cover = ys[j];
        cur = nx;
        j++;
    }
    nx = j < *len ? xs[j] : rx;
    gain += (nx - cur) * (cover - y);
    /* replace entries [i, j) with the single new entry; j may equal i, in
     * which case this is a plain insertion (avoid size_t underflow) */
    memmove(xs + i + 1, xs + j, (*len - j) * sizeof(double));
    memmove(ys + i + 1, ys + j, (*len - j) * sizeof(double));
    *len = *len - (j - i) + 1;
    xs[i] = x;
    ys[i] = y;
    return gain;
}

static double hv3d(double *points, size_t n, const double *ref)
{
    double *xs, *ys, area = 0.0, volume = 0.0;
    size_t len = 0, i;
    sort_by_coordinate(points, n, 3, 2);
    xs = (double *)malloc(n * sizeof(double));
    ys = (double *)malloc(n * sizeof(double));
    if (!xs || !ys) {
        bl_err = 1;
        free(xs);
        free(ys);
        return 0.0;
    }
    for (i = 0; i < n; i++) {
        const double *p = points + i * 3;
        double next_z = i + 1 < n ? points[(i + 1) * 3 + 2] : ref[2];
        area += staircase_insert(xs, ys, &len, p[0], p[1], ref[0], ref[1]);
        volume += area * (next_z - p[2]);
    }
    free(xs);
    free(ys);
    return volume;
}

/* Exact hypervolume of `points` (destroyed: reordered) strictly dominating
 * `ref`, by sweeping the last objective. */
static double sweep(double *points, size_t n, int dim, const double *ref)
{
    double *clips, *pruned, volume = 0.0, cross_section = 0.0;
    int sub = dim - 1;
    size_t i;

    if (n == 0 || bl_err)
        return 0.0;
    if (dim == 1) {
        double best = points[0];
        for (i = 1; i < n; i++)
            if (points[i] < best)
                best = points[i];
        return ref[0] - best;
    }
    if (dim == 2)
        return hv2d(points, n, ref);
    if (dim == 3)
        return hv3d(points, n, ref);

    sort_by_coordinate(points, n, dim, dim - 1);
    clips = (double *)malloc(n * (size_t)sub * sizeof(double));
    pruned = (double *)malloc(n * (size_t)sub * sizeof(double));
    if (!clips || !pruned) {
        bl_err = 1;
        free(clips);
        free(pruned);
        return 0.0;
    }
    for (i = 0; i < n && !bl_err; i++) {
        const double *p = points + i * (size_t)dim;
        double next_z = i + 1 < n ? points[(i + 1) * (size_t)dim + dim - 1]
                                  : ref[dim - 1];
        double exclusive = pt_box_volume(p, ref, sub);
        size_t m = 0, t;
        int k;
        for (t = 0; t < i; t++) { /* clip against every swept point */
            const double *q = points + t * (size_t)dim;
            int inside = 1;
            for (k = 0; k < sub; k++) {
                double v = p[k] > q[k] ? p[k] : q[k];
                clips[m * (size_t)sub + k] = v;
                if (v >= ref[k])
                    inside = 0;
            }
            if (inside)
                m++;
        }
        if (m > 0) {
            long kept = fhv_nondominated(clips, m, sub, pruned);
            if (kept < 0) {
                bl_err = 1;
                break;
            }
            exclusive -= sweep(pruned, (size_t)kept, sub, ref);
        }
        cross_section += exclusive;
        volume += cross_section * (next_z - p[dim - 1]);
    }
    free(clips);
    free(pruned);
    return volume;
}

double fhv_hypervolume_ds(const double *points, size_t n, int dim,
                          const double *reference, int maximize, int prefilter)
{
    double *front, *ref, result;
    size_t m = 0;
    if (dim <= 0)
        return 0.0;
    ref = (double *)malloc((size_t)dim * sizeof(double));
    if (!ref)
        return FHV_ENOMEM;
    front = prepare(points, n, dim, reference, maximize, prefilter, ref, &m);
    if (!front) {
        free(ref);
        return FHV_ENOMEM;
    }
    bl_err = 0;
    result = m ? sweep(front, m, dim, ref) : 0.0;
    free(front);
    free(ref);
    return bl_err ? FHV_ENOMEM : result;
}

/* ------------------------------------------------------------------ */
/* WFG                                                                 */
/* ------------------------------------------------------------------ */

/* Sum of exclusive hypervolumes of `front` (m x dim, sorted by the last
 * objective ascending, all strictly dominating `ref`). */
static double wfg(const double *front, size_t m, int dim, const double *ref)
{
    double *limit, *pruned, total = 0.0;
    size_t i;

    if (m == 0 || bl_err)
        return 0.0;
    limit = (double *)malloc(m * (size_t)dim * sizeof(double));
    pruned = (double *)malloc(m * (size_t)dim * sizeof(double));
    if (!limit || !pruned) {
        bl_err = 1;
        free(limit);
        free(pruned);
        return 0.0;
    }
    for (i = 0; i < m && !bl_err; i++) {
        const double *p = front + i * (size_t)dim;
        double exclusive = pt_box_volume(p, ref, dim);
        size_t nl = 0, t;
        int k;
        for (t = i + 1; t < m; t++) { /* limit set against later points */
            const double *q = front + t * (size_t)dim;
            int inside = 1;
            for (k = 0; k < dim; k++) {
                double v = p[k] > q[k] ? p[k] : q[k];
                limit[nl * (size_t)dim + k] = v;
                if (v >= ref[k])
                    inside = 0;
            }
            if (inside)
                nl++;
        }
        if (nl > 0) {
            long kept = fhv_nondominated(limit, nl, dim, pruned);
            if (kept < 0) {
                bl_err = 1;
                break;
            }
            /* fhv_nondominated returns the limit set lexicographically
             * sorted, matching the Python version's recursion order */
            exclusive -= wfg(pruned, (size_t)kept, dim, ref);
        }
        total += exclusive;
    }
    free(limit);
    free(pruned);
    return total;
}

double fhv_hypervolume_wfg(const double *points, size_t n, int dim,
                           const double *reference, int maximize, int prefilter)
{
    double *front, *ref, result;
    size_t m = 0;
    if (dim <= 0)
        return 0.0;
    ref = (double *)malloc((size_t)dim * sizeof(double));
    if (!ref)
        return FHV_ENOMEM;
    front = prepare(points, n, dim, reference, maximize, prefilter, ref, &m);
    if (!front) {
        free(ref);
        return FHV_ENOMEM;
    }
    bl_err = 0;
    if (m > 1)
        sort_by_coordinate(front, m, dim, dim - 1);
    result = m ? wfg(front, m, dim, ref) : 0.0;
    free(front);
    free(ref);
    return bl_err ? FHV_ENOMEM : result;
}
