/* fasthvchan.c -- see fasthvchan.h.
 *
 * A box is 2*dim contiguous doubles: lo[0..dim), then hi[0..dim).  An array of
 * m boxes is m*2*dim doubles.  Recursion nodes own their box array and their
 * cell and free both, so each node's working set dies with it.
 *
 * Not thread-safe: fhv_node_count, fhv_base_case and the sort width below are
 * globals, matching the Python original's per-instance mutable state.
 */
#include "fasthvchan.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

long fhv_node_count = 0;
int fhv_base_case = 2;

static int fhv_err = 0; /* sticky allocation-failure flag */

#define LO(boxes, i, dim) ((boxes) + (size_t)(i) * 2 * (dim))
#define HI(boxes, i, dim) ((boxes) + (size_t)(i) * 2 * (dim) + (dim))

/* ------------------------------------------------------------------ */
/* Geometry primitives                                                 */
/* ------------------------------------------------------------------ */

static double box_volume(const double *b, int dim)
{
    double volume = 1.0;
    int k;
    for (k = 0; k < dim; k++) {
        if (b[dim + k] <= b[k])
            return 0.0;
        volume *= b[dim + k] - b[k];
    }
    return volume;
}

/* Intersection of `b` with `cell` into `out`; 0 if the overlap is flat. */
static int box_clip_to(const double *b, const double *cell, int dim, double *out)
{
    int k;
    for (k = 0; k < dim; k++) {
        double low = b[k] > cell[k] ? b[k] : cell[k];
        double high = b[dim + k] < cell[dim + k] ? b[dim + k] : cell[dim + k];
        if (high <= low)
            return 0;
        out[k] = low;
        out[dim + k] = high;
    }
    return 1;
}

/* True if `b` covers `cell` along every axis other than `axis`. */
static int box_spans(const double *b, const double *cell, int dim, int axis)
{
    int k;
    for (k = 0; k < dim; k++) {
        if (k == axis)
            continue;
        if (b[k] > cell[k] || b[dim + k] < cell[dim + k])
            return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Interval union and the collapse map                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    double a, b;
} interval_t;

static int cmp_interval(const void *p, const void *q)
{
    const interval_t *x = (const interval_t *)p, *y = (const interval_t *)q;
    if (x->a < y->a) return -1;
    if (x->a > y->a) return 1;
    if (x->b < y->b) return -1;
    if (x->b > y->b) return 1;
    return 0;
}

/* Sorts in place and merges overlaps; returns the number of disjoint runs. */
static size_t merge_intervals(interval_t *iv, size_t n)
{
    size_t i, m = 0;
    if (n == 0)
        return 0;
    qsort(iv, n, sizeof(interval_t), cmp_interval);
    for (i = 0; i < n; i++) {
        if (m > 0 && iv[i].a <= iv[m - 1].b) {
            if (iv[i].b > iv[m - 1].b)
                iv[m - 1].b = iv[i].b;
        } else {
            iv[m++] = iv[i];
        }
    }
    return m;
}

typedef struct {
    const interval_t *iv; /* sorted, disjoint */
    double *removed;      /* total covered length strictly before iv[i] */
    size_t n;
} collapse_t;

static int collapse_init(collapse_t *c, const interval_t *iv, size_t n)
{
    double total = 0.0;
    size_t i;
    c->iv = iv;
    c->n = n;
    c->removed = (double *)malloc((n ? n : 1) * sizeof(double));
    if (!c->removed) {
        fhv_err = 1;
        return 0;
    }
    for (i = 0; i < n; i++) {
        c->removed[i] = total;
        total += iv[i].b - iv[i].a;
    }
    return 1;
}

static void collapse_free(collapse_t *c) { free(c->removed); }

/* Monotone map shrinking every covered interval to zero length; the identity
 * below the first one.  Distances outside the covered set are preserved. */
static double collapse_apply(const collapse_t *c, double x)
{
    size_t lo = 0, hi = c->n, index; /* bisect_right over interval starts */
    double inside;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (c->iv[mid].a <= x)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return x;
    index = lo - 1;
    inside = (c->iv[index].b < x ? c->iv[index].b : x) - c->iv[index].a;
    return x - c->removed[index] - inside;
}

/* ------------------------------------------------------------------ */
/* Base case                                                           */
/* ------------------------------------------------------------------ */

/* Measure of `cell` minus the union of `boxes`, already clipped to it.
 * Exponential in m, so only ever called with m <= fhv_base_case. */
static double complement_incl_excl(const double *boxes, size_t m,
                                   const double *cell, int dim)
{
    double measure = box_volume(cell, dim);
    double *common, *scratch;
    unsigned long long mask, limit;

    if (m == 0)
        return measure;
    if (m > 62) { /* refuse to overflow the subset counter */
        fhv_err = 1;
        return 0.0;
    }
    common = (double *)malloc((size_t)4 * dim * sizeof(double));
    if (!common) {
        fhv_err = 1;
        return 0.0;
    }
    scratch = common + 2 * dim;

    limit = 1ULL << m;
    for (mask = 1; mask < limit; mask++) {
        int bits = 0, alive = 1, first = 1;
        size_t i;
        for (i = 0; i < m; i++) {
            if (!((mask >> i) & 1ULL))
                continue;
            bits++;
            if (first) {
                memcpy(common, LO(boxes, i, dim), (size_t)2 * dim * sizeof(double));
                first = 0;
            } else if (box_clip_to(LO(boxes, i, dim), common, dim, scratch)) {
                memcpy(common, scratch, (size_t)2 * dim * sizeof(double));
            } else {
                alive = 0;
                break;
            }
        }
        if (alive)
            measure += (bits % 2 ? -1.0 : 1.0) * box_volume(common, dim);
    }
    free(common);
    return measure;
}

/* ------------------------------------------------------------------ */
/* Chan's divide and conquer                                           */
/* ------------------------------------------------------------------ */

/* Collapse away every box that is a slab of `cell`, repeating until stable.
 * Rewrites `boxes` in place (only ever shrinking) and `cell`; returns the
 * surviving box count.  Afterwards every box has a (d-2)-face crossing cell. */
static size_t simplify(int dim, double *boxes, size_t m, double *cell)
{
    interval_t *iv = (interval_t *)malloc((m ? m : 1) * sizeof(interval_t));
    if (!iv) {
        fhv_err = 1;
        return m;
    }
    while (m > 0) {
        int collapsed_any = 0, axis;
        for (axis = 0; axis < dim; axis++) {
            size_t i, w = 0, nslab = 0, nrun;
            collapse_t c;

            /* Partition: slab intervals out, survivors compacted to the front. */
            for (i = 0; i < m; i++) {
                const double *b = LO(boxes, i, dim);
                if (box_spans(b, cell, dim, axis)) {
                    iv[nslab].a = b[axis];
                    iv[nslab].b = b[dim + axis];
                    nslab++;
                } else {
                    if (w != i)
                        memmove(LO(boxes, w, dim), b,
                                (size_t)2 * dim * sizeof(double));
                    w++;
                }
            }
            if (nslab == 0)
                continue;
            collapsed_any = 1;
            m = w;

            nrun = merge_intervals(iv, nslab);
            if (!collapse_init(&c, iv, nrun)) {
                free(iv);
                return m;
            }
            cell[axis] = collapse_apply(&c, cell[axis]);
            cell[dim + axis] = collapse_apply(&c, cell[dim + axis]);
            for (i = 0; i < m; i++) {
                double *b = LO(boxes, i, dim);
                b[axis] = collapse_apply(&c, b[axis]);
                b[dim + axis] = collapse_apply(&c, b[dim + axis]);
            }
            collapse_free(&c);

            /* Collapsing can flatten a survivor. */
            w = 0;
            for (i = 0; i < m; i++) {
                double *b = LO(boxes, i, dim);
                if (b[axis] < b[dim + axis]) {
                    if (w != i)
                        memmove(LO(boxes, w, dim), b,
                                (size_t)2 * dim * sizeof(double));
                    w++;
                }
            }
            m = w;
            if (m == 0)
                break;
        }
        if (!collapsed_any)
            break;
    }
    free(iv);
    return m;
}

typedef struct {
    double x, w;
} cut_t;

static int cmp_cut(const void *p, const void *q)
{
    const cut_t *a = (const cut_t *)p, *b = (const cut_t *)q;
    if (a->x < b->x) return -1;
    if (a->x > b->x) return 1;
    if (a->w < b->w) return -1;
    if (a->w > b->w) return 1;
    return 0;
}

/* Coordinates and weights of the (d-2)-faces orthogonal to `axis`.
 *
 * After `depth` cuts the paper's renumbering puts original axis k at position
 * ((k - depth) mod d) + 1, with the cut axis at position 1; a face orthogonal
 * to positions i and j weighs 2^((i+j)/d). */
static size_t cut_candidates(int dim, const double *boxes, size_t m,
                             const double *cell, int axis, int depth,
                             cut_t *cuts, double *weights)
{
    size_t i, n = 0;
    int k;
    for (k = 0; k < dim; k++) {
        int pos = (k - depth) % dim;
        if (pos < 0)
            pos += dim; /* Python's modulo is never negative */
        weights[k] = pow(2.0, (double)(pos + 2) / (double)dim);
    }
    for (i = 0; i < m; i++) {
        const double *b = LO(boxes, i, dim);
        double coords[2], weight = 0.0;
        int ncoord = 0, j;
        if (cell[axis] < b[axis] && b[axis] < cell[dim + axis])
            coords[ncoord++] = b[axis];
        if (cell[axis] < b[dim + axis] && b[dim + axis] < cell[dim + axis])
            coords[ncoord++] = b[dim + axis];
        if (ncoord == 0)
            continue;
        /* One face per boundary of the box crossing the cell along another
         * axis.  A box with no such boundary is a slab and is already gone. */
        for (k = 0; k < dim; k++) {
            int crossings;
            if (k == axis)
                continue;
            crossings = (b[k] > cell[k]) + (b[dim + k] < cell[dim + k]);
            if (crossings)
                weight += crossings * weights[k];
        }
        if (weight != 0.0) {
            for (j = 0; j < ncoord; j++) {
                cuts[n].x = coords[j];
                cuts[n].w = weight;
                n++;
            }
        }
    }
    return n;
}

/* Smallest coordinate whose weight prefix reaches half the total, so both open
 * subcells keep at most half: faces on the hyperplane belong to neither. */
static double weighted_median(cut_t *cuts, size_t n)
{
    double half = 0.0, accumulated = 0.0;
    size_t i;
    qsort(cuts, n, sizeof(cut_t), cmp_cut);
    for (i = 0; i < n; i++)
        half += cuts[i].w;
    half /= 2.0;
    for (i = 0; i < n; i++) {
        accumulated += cuts[i].w;
        if (accumulated >= half)
            return cuts[i].x;
    }
    return cuts[n - 1].x;
}

/* Clip every box to `cell`, dropping flat overlaps.  Allocates the result. */
static size_t clip_all(int dim, const double *boxes, size_t m,
                       const double *cell, double **out)
{
    double *dst = (double *)malloc((m ? m : 1) * 2 * dim * sizeof(double));
    size_t i, w = 0;
    if (!dst) {
        fhv_err = 1;
        *out = NULL;
        return 0;
    }
    for (i = 0; i < m; i++) {
        if (box_clip_to(LO(boxes, i, dim), cell, dim, LO(dst, w, dim)))
            w++;
    }
    *out = dst;
    return w;
}

/* Takes ownership of `boxes` and `cell`, which must be malloc'd; frees both. */
static double measure(int dim, double *boxes, size_t m, double *cell, int depth)
{
    double result, value, *left = NULL, *right = NULL, *lb = NULL, *rb = NULL;
    double *weights = NULL;
    cut_t *cuts = NULL;
    size_t ncuts = 0, lm, rm;
    int axis = 0, tries;

    fhv_node_count++;

    /* Step 0: trivial instances. */
    if (fhv_err || m == 0)
        goto trivial;
    if (m <= (size_t)fhv_base_case) {
        result = complement_incl_excl(boxes, m, cell, dim);
        free(boxes);
        free(cell);
        return result;
    }

    /* Step 1: collapse away every slab-shaped box. */
    m = simplify(dim, boxes, m, cell);
    if (fhv_err || m == 0)
        goto trivial;

    /* Step 2: cut at the weighted median, rotating the axis until one carries
     * a face.  Some axis always does, since every simplified box has one. */
    cuts = (cut_t *)malloc((size_t)2 * m * sizeof(cut_t));
    weights = (double *)malloc((size_t)dim * sizeof(double));
    if (!cuts || !weights) {
        fhv_err = 1;
        goto trivial;
    }
    for (tries = 0; tries < dim; tries++) {
        axis = depth % dim;
        ncuts = cut_candidates(dim, boxes, m, cell, axis, depth, cuts, weights);
        if (ncuts)
            break;
        depth++;
    }
    if (ncuts == 0) { /* unreachable if simplify() held up its end */
        fhv_err = 1;
        goto trivial;
    }
    value = weighted_median(cuts, ncuts);
    free(cuts);
    cuts = NULL;
    free(weights);
    weights = NULL;

    /* Step 3: recurse.  The cut axis cycles through the dimensions, which is
     * the paper's axis renumbering. */
    left = (double *)malloc((size_t)2 * dim * sizeof(double));
    right = (double *)malloc((size_t)2 * dim * sizeof(double));
    if (!left || !right) {
        fhv_err = 1;
        free(left);
        free(right);
        goto trivial;
    }
    memcpy(left, cell, (size_t)2 * dim * sizeof(double));
    memcpy(right, cell, (size_t)2 * dim * sizeof(double));
    left[dim + axis] = value;
    right[axis] = value;

    lm = clip_all(dim, boxes, m, left, &lb);
    rm = clip_all(dim, boxes, m, right, &rb);
    free(boxes);
    free(cell);
    if (fhv_err) {
        free(lb);
        free(rb);
        free(left);
        free(right);
        return 0.0;
    }
    return measure(dim, lb, lm, left, depth + 1) +
           measure(dim, rb, rm, right, depth + 1);

trivial:
    result = fhv_err ? 0.0 : box_volume(cell, dim);
    free(cuts);
    free(weights);
    free(boxes);
    free(cell);
    return result;
}

/* Volume of `cell` not covered by any box, clipping the input first. */
static double complement_measure(int dim, const double *boxes, size_t m,
                                 const double *cell)
{
    double *clipped, *own_cell;
    size_t kept;

    fhv_node_count = 0;
    own_cell = (double *)malloc((size_t)2 * dim * sizeof(double));
    if (!own_cell) {
        fhv_err = 1;
        return 0.0;
    }
    memcpy(own_cell, cell, (size_t)2 * dim * sizeof(double));
    kept = clip_all(dim, boxes, m, cell, &clipped);
    if (fhv_err) {
        free(own_cell);
        free(clipped);
        return 0.0;
    }
    return measure(dim, clipped, kept, own_cell, 0);
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

static int g_sort_dim = 0;

static int cmp_point_lex(const void *p, const void *q)
{
    const double *a = (const double *)p, *b = (const double *)q;
    int k;
    for (k = 0; k < g_sort_dim; k++) {
        if (a[k] < b[k]) return -1;
        if (a[k] > b[k]) return 1;
    }
    return 0;
}

long fhv_nondominated(const double *points, size_t n, int dim, double *out)
{
    double *sorted;
    size_t i, kept = 0;

    if (n == 0 || dim <= 0)
        return 0;
    sorted = (double *)malloc(n * dim * sizeof(double));
    if (!sorted)
        return -1;
    memcpy(sorted, points, n * dim * sizeof(double));
    g_sort_dim = dim;
    qsort(sorted, n, (size_t)dim * sizeof(double), cmp_point_lex);

    /* Lexicographic order puts any dominator before its victim, so a single
     * forward scan suffices.  Dominance is weak, so exact duplicates drop. */
    for (i = 0; i < n; i++) {
        const double *p = sorted + i * dim;
        size_t j;
        int dominated = 0;
        for (j = 0; j < kept && !dominated; j++) {
            const double *other = out + j * dim;
            int k, all = 1;
            for (k = 0; k < dim; k++) {
                if (!(other[k] <= p[k])) {
                    all = 0;
                    break;
                }
            }
            dominated = all;
        }
        if (!dominated) {
            memcpy(out + kept * dim, p, (size_t)dim * sizeof(double));
            kept++;
        }
    }
    free(sorted);
    return (long)kept;
}

double fhv_hypervolume(const double *points, size_t n, int dim,
                       const double *reference, int maximize, int prefilter)
{
    double sign = maximize ? -1.0 : 1.0;
    double *ref = NULL, *front = NULL, *boxes = NULL, *domain = NULL;
    size_t i, nfront = 0;
    int k;
    double result;

    fhv_err = 0;
    if (dim <= 0 || n == 0)
        return 0.0;

    ref = (double *)malloc((size_t)dim * sizeof(double));
    front = (double *)malloc(n * dim * sizeof(double));
    if (!ref || !front) {
        free(ref);
        free(front);
        return FHV_ENOMEM;
    }
    for (k = 0; k < dim; k++)
        ref[k] = sign * reference[k];

    /* Only points strictly better than the reference in every objective. */
    for (i = 0; i < n; i++) {
        double *dst = front + nfront * dim;
        int ok = 1;
        for (k = 0; k < dim; k++) {
            dst[k] = sign * points[i * dim + k];
            if (!(dst[k] < ref[k]))
                ok = 0;
        }
        if (ok)
            nfront++;
    }
    if (nfront == 0) {
        free(ref);
        free(front);
        return 0.0;
    }
    if (prefilter) {
        double *filtered = (double *)malloc(nfront * dim * sizeof(double));
        long kept;
        if (!filtered) {
            free(ref);
            free(front);
            return FHV_ENOMEM;
        }
        kept = fhv_nondominated(front, nfront, dim, filtered);
        if (kept < 0) {
            free(ref);
            free(front);
            free(filtered);
            return FHV_ENOMEM;
        }
        free(front);
        front = filtered;
        nfront = (size_t)kept;
    }

    boxes = (double *)malloc(nfront * 2 * dim * sizeof(double));
    domain = (double *)malloc((size_t)2 * dim * sizeof(double));
    if (!boxes || !domain) {
        free(ref);
        free(front);
        free(boxes);
        free(domain);
        return FHV_ENOMEM;
    }
    for (i = 0; i < nfront; i++) {
        memcpy(LO(boxes, i, dim), front + i * dim, (size_t)dim * sizeof(double));
        memcpy(HI(boxes, i, dim), ref, (size_t)dim * sizeof(double));
    }
    for (k = 0; k < dim; k++) {
        double lo = front[k];
        for (i = 1; i < nfront; i++)
            if (front[i * dim + k] < lo)
                lo = front[i * dim + k];
        domain[k] = lo;
        domain[dim + k] = ref[k];
    }

    result = box_volume(domain, dim) -
             complement_measure(dim, boxes, nfront, domain);
    free(ref);
    free(front);
    free(boxes);
    free(domain);
    return fhv_err ? FHV_ENOMEM : result;
}

double fhv_union_volume(const double *los, const double *his, size_t n, int dim,
                        const double *domain_lo, const double *domain_hi)
{
    double *boxes, *domain;
    size_t i, m = 0;
    int k;
    double result;

    fhv_err = 0;
    if (dim <= 0 || n == 0)
        return 0.0;

    boxes = (double *)malloc(n * 2 * dim * sizeof(double));
    domain = (double *)malloc((size_t)2 * dim * sizeof(double));
    if (!boxes || !domain) {
        free(boxes);
        free(domain);
        return FHV_ENOMEM;
    }
    for (i = 0; i < n; i++) { /* degenerate boxes drop, as in the original */
        double *dst = LO(boxes, m, dim);
        memcpy(dst, los + i * dim, (size_t)dim * sizeof(double));
        memcpy(dst + dim, his + i * dim, (size_t)dim * sizeof(double));
        if (box_volume(dst, dim) > 0.0)
            m++;
    }
    if (m == 0) {
        free(boxes);
        free(domain);
        return 0.0;
    }
    if (domain_lo && domain_hi) {
        memcpy(domain, domain_lo, (size_t)dim * sizeof(double));
        memcpy(domain + dim, domain_hi, (size_t)dim * sizeof(double));
    } else { /* bounding box of the input restricts nothing */
        for (k = 0; k < dim; k++) {
            double lo = LO(boxes, 0, dim)[k], hi = HI(boxes, 0, dim)[k];
            for (i = 1; i < m; i++) {
                if (LO(boxes, i, dim)[k] < lo)
                    lo = LO(boxes, i, dim)[k];
                if (HI(boxes, i, dim)[k] > hi)
                    hi = HI(boxes, i, dim)[k];
            }
            domain[k] = lo;
            domain[dim + k] = hi;
        }
    }

    result = box_volume(domain, dim) - complement_measure(dim, boxes, m, domain);
    free(boxes);
    free(domain);
    return fhv_err ? FHV_ENOMEM : result;
}

double fhv_reference_hypervolume(const double *points, size_t n, int dim,
                                 const double *reference)
{
    double *boxes, total = 0.0, *common, *scratch;
    size_t i, m = 0;
    unsigned long long mask, limit;

    if (dim <= 0 || n == 0)
        return 0.0;
    if (n > 62)
        return FHV_ENOMEM;

    boxes = (double *)malloc(n * 2 * dim * sizeof(double));
    common = (double *)malloc((size_t)4 * dim * sizeof(double));
    if (!boxes || !common) {
        free(boxes);
        free(common);
        return FHV_ENOMEM;
    }
    scratch = common + 2 * dim;
    for (i = 0; i < n; i++) {
        double *dst = LO(boxes, m, dim);
        memcpy(dst, points + i * dim, (size_t)dim * sizeof(double));
        memcpy(dst + dim, reference, (size_t)dim * sizeof(double));
        if (box_volume(dst, dim) > 0.0)
            m++;
    }

    limit = 1ULL << m;
    for (mask = 1; mask < limit; mask++) {
        int bits = 0, alive = 1, first = 1;
        for (i = 0; i < m; i++) {
            if (!((mask >> i) & 1ULL))
                continue;
            bits++;
            if (first) {
                memcpy(common, LO(boxes, i, dim), (size_t)2 * dim * sizeof(double));
                first = 0;
            } else if (box_clip_to(LO(boxes, i, dim), common, dim, scratch)) {
                memcpy(common, scratch, (size_t)2 * dim * sizeof(double));
            } else {
                alive = 0;
                break;
            }
        }
        if (alive)
            total += (bits % 2 ? 1.0 : -1.0) * box_volume(common, dim);
    }
    free(boxes);
    free(common);
    return total;
}
