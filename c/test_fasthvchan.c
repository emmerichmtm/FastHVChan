/* test_fasthvchan.c -- self-tests and a differential oracle.
 *
 *   test_fasthvchan            run the built-in suite
 *   test_fasthvchan --check f  read cases from file `f` and compare
 *   test_fasthvchan --bench    timings on spherical fronts
 *
 * The built-in suite checks the divide-and-conquer against the brute-force
 * inclusion-exclusion baseline, which is the same thing the Python self-test
 * does; --check compares against values produced by the Python original.
 */
#include "fasthvchan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures = 0;
static int checks = 0;
static double total_secs = 0.0; /* compute time only, excluding parsing */

static void check(const char *name, double got, double expected, double tol)
{
    int ok = fabs(got - expected) <= tol;
    checks++;
    if (!ok)
        failures++;
    printf("  [%s] %-44s got %14.8f  expected %14.8f\n", ok ? "ok " : "FAIL",
           name, got, expected);
}

/* xorshift64*, so the C and Python sides can be kept independent. */
static unsigned long long rng_state = 88172645463325252ULL;

static void rng_seed(unsigned long long s) { rng_state = s ? s : 1; }

static double rng_next(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 2685821657736338717ULL) >> 11) /
           (double)(1ULL << 53);
}

/* ------------------------------------------------------------------ */

static void test_known_values(void)
{
    /* The README's worked example. */
    double pts[9] = {0.2, 0.7, 0.3, 0.5, 0.2, 0.6, 0.8, 0.4, 0.1};
    double ref[3] = {1.0, 1.0, 1.0};
    double los[4] = {0.0, 0.0, 1.0, 1.0};
    double his[4] = {2.0, 2.0, 3.0, 3.0};
    double dlo[2] = {0.0, 0.0}, dhi[2] = {2.0, 2.0};
    double los2[4] = {0.0, 0.0, 5.0, 5.0};
    double his2[4] = {1.0, 1.0, 7.0, 8.0};

    puts("Known values");
    check("README example, d=3", fhv_hypervolume(pts, 3, 3, ref, 0, 1), 0.31, 1e-12);
    check("README example, prefilter off",
          fhv_hypervolume(pts, 3, 3, ref, 0, 0), 0.31, 1e-12);
    check("brute force agrees", fhv_reference_hypervolume(pts, 3, 3, ref), 0.31,
          1e-12);
    check("maximize -> nothing dominates ref",
          fhv_hypervolume(pts, 3, 3, ref, 1, 1), 0.0, 1e-12);
    check("two overlapping unit squares",
          fhv_union_volume(los, his, 2, 2, NULL, NULL), 7.0, 1e-12);
    check("same, restricted to [0,2]^2",
          fhv_union_volume(los, his, 2, 2, dlo, dhi), 4.0, 1e-12);
    check("README disjoint boxes",
          fhv_union_volume(los2, his2, 2, 2, NULL, NULL), 7.0, 1e-12);
}

static void test_against_brute_force(void)
{
    int dims[4] = {2, 3, 4, 5};
    int counts[3] = {3, 7, 10};
    int di, ci;

    puts("Divide and conquer vs inclusion-exclusion");
    rng_seed(20260821ULL);
    for (di = 0; di < 4; di++) {
        for (ci = 0; ci < 3; ci++) {
            int dim = dims[di], n = counts[ci], i, k;
            double *pts = (double *)malloc((size_t)n * dim * sizeof(double));
            double *ref = (double *)malloc((size_t)dim * sizeof(double));
            char name[64];
            double got, want;

            for (i = 0; i < n; i++)
                for (k = 0; k < dim; k++)
                    pts[i * dim + k] = rng_next();
            for (k = 0; k < dim; k++)
                ref[k] = 1.0;

            got = fhv_hypervolume(pts, n, dim, ref, 0, 1);
            want = fhv_reference_hypervolume(pts, n, dim, ref);
            sprintf(name, "d=%d, n=%d points", dim, n);
            check(name, got, want, 1e-9);
            free(pts);
            free(ref);
        }
    }
}

static void test_union_against_brute_force(void)
{
    int dims[3] = {2, 3, 4};
    int di;

    puts("Box unions vs inclusion-exclusion");
    rng_seed(99887766ULL);
    for (di = 0; di < 3; di++) {
        int dim = dims[di], n = 8, i, k;
        double *los = (double *)malloc((size_t)n * dim * sizeof(double));
        double *his = (double *)malloc((size_t)n * dim * sizeof(double));
        /* Brute-force the union as a hypervolume with a shared upper corner is
         * not possible for general boxes, so inclusion-exclusion inline. */
        double total = 0.0;
        unsigned long long mask;
        char name[64];

        for (i = 0; i < n; i++) {
            for (k = 0; k < dim; k++) {
                double a = rng_next(), b = rng_next();
                los[i * dim + k] = a < b ? a : b;
                his[i * dim + k] = a < b ? b : a;
            }
        }
        for (mask = 1; mask < (1ULL << n); mask++) {
            double lo[8], hi[8], vol = 1.0;
            int bits = 0, alive = 1;
            for (k = 0; k < dim; k++) {
                lo[k] = -1e300;
                hi[k] = 1e300;
            }
            for (i = 0; i < n; i++) {
                if (!((mask >> i) & 1ULL))
                    continue;
                bits++;
                for (k = 0; k < dim; k++) {
                    if (los[i * dim + k] > lo[k]) lo[k] = los[i * dim + k];
                    if (his[i * dim + k] < hi[k]) hi[k] = his[i * dim + k];
                }
            }
            for (k = 0; k < dim; k++) {
                if (hi[k] <= lo[k]) { alive = 0; break; }
                vol *= hi[k] - lo[k];
            }
            if (alive)
                total += (bits % 2 ? 1.0 : -1.0) * vol;
        }
        sprintf(name, "d=%d, n=%d random boxes", dim, n);
        check(name, fhv_union_volume(los, his, n, dim, NULL, NULL), total, 1e-9);
        free(los);
        free(his);
    }
}

static void test_edge_cases(void)
{
    double p[4] = {0.5, 0.5, 0.5, 0.5};
    double ref[2] = {1.0, 1.0};
    double worse[2] = {0.0, 0.0};
    double flat_lo[2] = {0.0, 0.0}, flat_hi[2] = {0.0, 1.0};
    double dup[6] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    double out[6];
    long kept;

    puts("Edge cases");
    check("duplicate points counted once",
          fhv_hypervolume(p, 2, 2, ref, 0, 1), 0.25, 1e-12);
    check("duplicates, prefilter off",
          fhv_hypervolume(p, 2, 2, ref, 0, 0), 0.25, 1e-12);
    check("reference dominates everything",
          fhv_hypervolume(p, 2, 2, worse, 0, 1), 0.0, 1e-12);
    check("empty input", fhv_hypervolume(p, 0, 2, ref, 0, 1), 0.0, 1e-12);
    check("degenerate box ignored",
          fhv_union_volume(flat_lo, flat_hi, 1, 2, NULL, NULL), 0.0, 1e-12);

    kept = fhv_nondominated(dup, 3, 2, out);
    check("nondominated drops exact duplicates", (double)kept, 1.0, 0);
}

static void test_nondominated(void)
{
    double pts[8] = {1, 2, 2, 1, 3, 3, 0.5, 5};
    double out[8];
    long kept;

    puts("Non-dominated filtering");
    kept = fhv_nondominated(pts, 4, 2, out);
    check("front size", (double)kept, 3.0, 0);
    check("sorted lexicographically, first point", out[0], 0.5, 1e-12);
    check("dominated point dropped", out[2 * 2 + 0], 2.0, 1e-12);
}

/* ------------------------------------------------------------------ */

static void bench(void)
{
    int dims[5] = {3, 4, 5, 6, 7};
    int counts[5] = {500, 200, 100, 60, 40};
    int di;

    puts("\nTiming on spherical fronts (all points mutually non-dominated)");
    puts("   d      n     nodes   seconds  hypervolume");
    rng_seed(4242ULL);
    for (di = 0; di < 5; di++) {
        int dim = dims[di], n = counts[di], i, k;
        double *pts = (double *)malloc((size_t)n * dim * sizeof(double));
        double *ref = (double *)malloc((size_t)dim * sizeof(double));
        clock_t t0;
        double hv, secs;

        for (i = 0; i < n; i++) {
            double norm = 0.0;
            for (k = 0; k < dim; k++) {
                double g = rng_next() + 1e-12;
                pts[i * dim + k] = g;
                norm += g * g;
            }
            norm = sqrt(norm);
            for (k = 0; k < dim; k++)
                pts[i * dim + k] /= norm;
        }
        for (k = 0; k < dim; k++)
            ref[k] = 1.0;

        t0 = clock();
        hv = fhv_hypervolume(pts, n, dim, ref, 0, 1);
        secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("%4d %6d %9ld %9.3f  %.8f\n", dim, n, fhv_node_count, secs, hv);
        free(pts);
        free(ref);
    }
}

/* Differential oracle.  Each case is one line:
 *     hv <dim> <n> <maximize> <prefilter> <expected> <ref...> <points...>
 *     uv <dim> <n> <expected> <lo,hi per box...>
 */
static int run_check_file(const char *path)
{
    FILE *f = fopen(path, "r");
    char kind[8];
    int cases = 0;

    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    while (fscanf(f, "%7s", kind) == 1) {
        int dim, n, i, k;
        double expected, got;
        char name[64];

        if (strcmp(kind, "hv") == 0) {
            int maximize, prefilter;
            double *ref, *pts;
            if (fscanf(f, "%d %d %d %d %lf", &dim, &n, &maximize, &prefilter,
                       &expected) != 5)
                break;
            ref = (double *)malloc((size_t)dim * sizeof(double));
            pts = (double *)malloc((size_t)(n ? n : 1) * dim * sizeof(double));
            for (k = 0; k < dim; k++)
                if (fscanf(f, "%lf", &ref[k]) != 1) { free(ref); free(pts); goto done; }
            for (i = 0; i < n * dim; i++)
                if (fscanf(f, "%lf", &pts[i]) != 1) { free(ref); free(pts); goto done; }
            { clock_t t0 = clock();
              got = fhv_hypervolume(pts, n, dim, ref, maximize, prefilter);
              total_secs += (double)(clock() - t0) / CLOCKS_PER_SEC; }
            sprintf(name, "hv d=%d n=%d max=%d pre=%d", dim, n, maximize, prefilter);
            check(name, got, expected, 1e-9);
            free(ref);
            free(pts);
        } else if (strcmp(kind, "uv") == 0) {
            double *los, *his;
            if (fscanf(f, "%d %d %lf", &dim, &n, &expected) != 3)
                break;
            los = (double *)malloc((size_t)(n ? n : 1) * dim * sizeof(double));
            his = (double *)malloc((size_t)(n ? n : 1) * dim * sizeof(double));
            for (i = 0; i < n; i++) {
                for (k = 0; k < dim; k++)
                    if (fscanf(f, "%lf", &los[i * dim + k]) != 1) goto uv_done;
                for (k = 0; k < dim; k++)
                    if (fscanf(f, "%lf", &his[i * dim + k]) != 1) goto uv_done;
            }
            { clock_t t0 = clock();
              got = fhv_union_volume(los, his, n, dim, NULL, NULL);
              total_secs += (double)(clock() - t0) / CLOCKS_PER_SEC; }
            sprintf(name, "uv d=%d n=%d", dim, n);
            check(name, got, expected, 1e-9);
        uv_done:
            free(los);
            free(his);
        } else {
            fprintf(stderr, "unknown case kind '%s'\n", kind);
            break;
        }
        cases++;
    }
done:
    fclose(f);
    printf("\n%d case(s) from %s, %d check(s), %d failure(s), %.4f s compute\n",
           cases, path, checks, failures, total_secs);
    return failures ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[1], "--check") == 0)
        return run_check_file(argv[2]);

    puts("fasthvchan self-tests");
    test_known_values();
    test_against_brute_force();
    test_union_against_brute_force();
    test_nondominated();
    test_edge_cases();

    printf("\n%d check(s), %d failure(s)\n", checks, failures);
    if (failures == 0)
        puts("All self-tests passed.");

    if (argc >= 2 && strcmp(argv[1], "--bench") == 0)
        bench();

    return failures ? 1 : 0;
}
