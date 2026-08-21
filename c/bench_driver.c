/* bench_driver.c -- time one C implementation on one dataset file.
 *
 *   bench_driver <file> [--base-case K] [--algo sec2|ds|wfg]
 *
 * --algo selects the implementation: sec2 (default) is the Chan Section-2
 * port, ds the dimension sweep, wfg the WFG algorithm (see hv_baselines.h).
 *
 * The file is the format written by benchmarks/crossbench.py:
 *
 *   dim n
 *   ref_0 ... ref_{dim-1}
 *   p_0_0 ... p_0_{dim-1}
 *   ...
 *
 * Prints one line of `key=value` pairs so the Python runner can parse it
 * without caring about formatting.  The call is repeated until it has run for
 * at least MIN_SECONDS, so that fast cases are not lost in clock granularity;
 * the reported time is per call.
 */
#include "fasthvchan.h"
#include "hv_baselines.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIN_SECONDS 0.05
#define MAX_REPS 200

int main(int argc, char **argv)
{
    const char *path = NULL, *algo = "sec2";
    int dim = 0, i, base_case = -1;
    long n = 0;
    double *pts = NULL, *ref = NULL, hv = 0.0, elapsed;
    long reps = 0;
    clock_t t0;
    FILE *f;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base-case") == 0 && i + 1 < argc)
            base_case = atoi(argv[++i]);
        else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc)
            algo = argv[++i];
        else if (!path)
            path = argv[i];
    }
    if (!path) {
        fprintf(stderr,
                "usage: bench_driver <file> [--base-case K] [--algo sec2|ds|wfg]\n");
        return 2;
    }
    if (strcmp(algo, "sec2") != 0 && strcmp(algo, "ds") != 0 &&
        strcmp(algo, "wfg") != 0) {
        fprintf(stderr, "unknown --algo %s\n", algo);
        return 2;
    }
    if (base_case >= 0)
        fhv_base_case = base_case;

    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }
    if (fscanf(f, "%d %ld", &dim, &n) != 2 || dim <= 0 || n < 0) {
        fprintf(stderr, "bad header in %s\n", path);
        fclose(f);
        return 2;
    }
    ref = (double *)malloc((size_t)dim * sizeof(double));
    pts = (double *)malloc((size_t)(n ? n : 1) * dim * sizeof(double));
    if (!ref || !pts) {
        fprintf(stderr, "out of memory\n");
        fclose(f);
        free(ref);
        free(pts);
        return 2;
    }
    for (i = 0; i < dim; i++) {
        if (fscanf(f, "%lf", &ref[i]) != 1) {
            fprintf(stderr, "truncated reference point\n");
            fclose(f);
            free(ref);
            free(pts);
            return 2;
        }
    }
    for (i = 0; i < (int)(n * dim); i++) {
        if (fscanf(f, "%lf", &pts[i]) != 1) {
            fprintf(stderr, "truncated point data\n");
            fclose(f);
            free(ref);
            free(pts);
            return 2;
        }
    }
    fclose(f);

    t0 = clock();
    do {
        if (strcmp(algo, "ds") == 0)
            hv = fhv_hypervolume_ds(pts, (size_t)n, dim, ref, 0, 1);
        else if (strcmp(algo, "wfg") == 0)
            hv = fhv_hypervolume_wfg(pts, (size_t)n, dim, ref, 0, 1);
        else
            hv = fhv_hypervolume(pts, (size_t)n, dim, ref, 0, 1);
        reps++;
        elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    } while (elapsed < MIN_SECONDS && reps < MAX_REPS);

    printf("hv=%.17g nodes=%ld seconds=%.9f reps=%ld base_case=%d algo=%s\n",
           hv, fhv_node_count, elapsed / (double)reps, reps, fhv_base_case,
           algo);
    free(ref);
    free(pts);
    return 0;
}
