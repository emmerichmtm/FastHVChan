/* test_hv_baselines.c -- cross-checks for the dimension-sweep and WFG ports.
 *
 * Self-contained: validates both baselines against brute-force
 * inclusion-exclusion (fhv_reference_hypervolume) on random instances and
 * tie-heavy integer grids, and against the Chan port (fhv_hypervolume) on
 * larger spherical fronts where brute force is infeasible.
 *
 *   ./test_hv_baselines            run the suite
 */
#include "fasthvchan.h"
#include "hv_baselines.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long rng_state = 20120808ULL; /* deterministic, portable */

static double rng_uniform(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((rng_state >> 16) & 0xffffffffULL) / 4294967296.0;
}

static int failures = 0;

static void check(const char *name, double got, double expected)
{
    double scale = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
    if (fabs(got - expected) <= 1e-9 * scale) {
        printf("  [ok ] %-40s %.10f\n", name, got);
    } else {
        printf("  [FAIL] %-40s got %.12f expected %.12f\n", name, got, expected);
        failures++;
    }
}

static void random_points(double *out, size_t n, int dim)
{
    size_t i;
    for (i = 0; i < n * (size_t)dim; i++)
        out[i] = rng_uniform();
}

static void grid_points(double *out, size_t n, int dim)
{
    size_t i;
    for (i = 0; i < n * (size_t)dim; i++)
        out[i] = (double)(1 + (int)(rng_uniform() * 3.0)); /* in {1,2,3} */
}

static void sphere_points(double *out, size_t n, int dim)
{
    size_t i;
    int k;
    for (i = 0; i < n; i++) {
        double norm = 0.0;
        for (k = 0; k < dim; k++) {
            double v = rng_uniform() + 1e-12;
            out[i * (size_t)dim + k] = v;
            norm += v * v;
        }
        norm = sqrt(norm);
        for (k = 0; k < dim; k++)
            out[i * (size_t)dim + k] /= norm;
    }
}

int main(void)
{
    double pts[16 * 8], ref[8];
    int dim, k;
    size_t n;
    char name[128];

    printf("hv_baselines C port tests\n");

    for (dim = 1; dim <= 6; dim++) {
        for (k = 0; k < dim; k++)
            ref[k] = 1.0;
        for (n = 1; n <= 12; n += 5) { /* n = 1, 6, 11 */
            double want;
            random_points(pts, n, dim);
            want = fhv_reference_hypervolume(pts, n, dim, ref);
            sprintf(name, "ds  random d=%d n=%d", dim, (int)n);
            check(name, fhv_hypervolume_ds(pts, n, dim, ref, 0, 1), want);
            sprintf(name, "wfg random d=%d n=%d", dim, (int)n);
            check(name, fhv_hypervolume_wfg(pts, n, dim, ref, 0, 1), want);
        }
    }

    for (dim = 2; dim <= 5; dim++) { /* tie-heavy integer coordinates */
        double want;
        for (k = 0; k < dim; k++)
            ref[k] = 4.0;
        n = 10;
        grid_points(pts, n, dim);
        want = fhv_reference_hypervolume(pts, n, dim, ref);
        sprintf(name, "ds  grid d=%d", dim);
        check(name, fhv_hypervolume_ds(pts, n, dim, ref, 0, 1), want);
        sprintf(name, "wfg grid d=%d", dim);
        check(name, fhv_hypervolume_wfg(pts, n, dim, ref, 0, 1), want);
    }

    { /* larger spherical fronts vs the Chan port */
        static double big[200 * 6];
        static const int dims[] = {3, 4, 5, 6};
        static const int sizes[] = {120, 60, 40, 25};
        int t;
        for (t = 0; t < 4; t++) {
            double want;
            dim = dims[t];
            n = (size_t)sizes[t];
            for (k = 0; k < dim; k++)
                ref[k] = 1.2;
            sphere_points(big, n, dim);
            want = fhv_hypervolume(big, n, dim, ref, 0, 1);
            sprintf(name, "ds  sphere d=%d n=%d vs Chan", dim, (int)n);
            check(name, fhv_hypervolume_ds(big, n, dim, ref, 0, 1), want);
            sprintf(name, "wfg sphere d=%d n=%d vs Chan", dim, (int)n);
            check(name, fhv_hypervolume_wfg(big, n, dim, ref, 0, 1), want);
        }
    }

    { /* maximisation and degenerate inputs */
        double p2[4] = {2.0, 2.0, 0.0, 0.0};
        double r2[2] = {0.0, 0.0};
        check("maximisation d=2", fhv_hypervolume_ds(p2, 1, 2, r2, 1, 1), 4.0);
        check("wfg maximisation", fhv_hypervolume_wfg(p2, 1, 2, r2, 1, 1), 4.0);
        r2[0] = r2[1] = 1.0;
        check("empty input ds", fhv_hypervolume_ds(p2, 0, 2, r2, 0, 1), 0.0);
        check("point on ref wfg",
              fhv_hypervolume_wfg(r2, 1, 2, r2, 0, 1), 0.0);
    }

    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("All baseline tests passed.\n");
    return 0;
}
