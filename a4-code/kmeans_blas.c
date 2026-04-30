// kmeans_blas.c for Problem 3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);

typedef struct {
    int ndata;
    int dim;
    float *features;
    int *assigns;
    int *labels;
    int nlabels;
} KMData;

typedef struct {
    int nclust;
    int dim;
    float *features;
    int *counts;
} KMClust;

KMData kmdata_load(const char *datafile) {
    KMData data;
    memset(&data, 0, sizeof(KMData));

    ssize_t tot_tokens = 0, tot_lines = 0;
    if (filestats((char *)datafile, &tot_tokens, &tot_lines) != 0) {
        printf("Failed to open file '%s'\n", datafile);
        exit(1);
    }

    int tokens_per_line = (int)(tot_tokens / tot_lines);
    data.ndata = (int)tot_lines;
    data.dim = tokens_per_line - 2;

    data.features = (float *)malloc((size_t)data.ndata * (size_t)data.dim * sizeof(float));
    data.assigns = (int *)malloc((size_t)data.ndata * sizeof(int));
    data.labels = (int *)malloc((size_t)data.ndata * sizeof(int));

    if (!data.features || !data.assigns || !data.labels) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    FILE *fin = fopen(datafile, "r");
    if (!fin) {
        printf("Failed to open file '%s'\n", datafile);
        exit(1);
    }

    int max_label = -1;
    for (int i = 0; i < data.ndata; i++) {
        int label;
        char colon[8];
        fscanf(fin, "%d %7s", &label, colon);
        data.labels[i] = label;
        if (label > max_label) max_label = label;

        for (int d = 0; d < data.dim; d++) {
            float val;
            fscanf(fin, "%f", &val);
            data.features[(size_t)i * (size_t)data.dim + (size_t)d] = val;
        }
    }

    fclose(fin);
    data.nlabels = max_label + 1;
    return data;
}

KMClust kmclust_new(int nclust, int dim) {
    KMClust clust;
    clust.nclust = nclust;
    clust.dim = dim;

    clust.features = (float *)malloc((size_t)nclust * (size_t)dim * sizeof(float));
    clust.counts = (int *)malloc((size_t)nclust * sizeof(int));

    if (!clust.features || !clust.counts) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < nclust * dim; i++) clust.features[i] = 0.0f;
    for (int i = 0; i < nclust; i++) clust.counts[i] = 0;

    return clust;
}

void save_pgm_files(KMClust *clust, const char *savedir) {
    int dim_root = (int)(sqrt((double)clust->dim));
    if (dim_root * dim_root != clust->dim) return;

    float maxfeat = 0.0f;
    for (int i = 0; i < clust->nclust * clust->dim; i++) {
        if (clust->features[i] > maxfeat) maxfeat = clust->features[i];
    }
    if (maxfeat < 1.0f) maxfeat = 1.0f;

    printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);

    for (int c = 0; c < clust->nclust; c++) {
        char outfile[512];
        sprintf(outfile, "%s/cent_%04d.pgm", savedir, c);

        FILE *pgm = fopen(outfile, "w");
        if (!pgm) {
            printf("Failed to open file '%s'\n", outfile);
            exit(1);
        }

        fprintf(pgm, "P2\n");
        fprintf(pgm, "%d %d\n", dim_root, dim_root);
        fprintf(pgm, "%.0f\n", maxfeat);

        for (int d = 0; d < clust->dim; d++) {
            if (d > 0 && d % dim_root == 0) fprintf(pgm, "\n");
            fprintf(pgm, "%3.0f ", clust->features[c * clust->dim + d]);
        }
        fprintf(pgm, "\n");
        fclose(pgm);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: kmeans_blas <datafile> <nclust> [savedir] [maxiter]\n");
        return 1;
    }

    const char *datafile = argv[1];
    int nclust = atoi(argv[2]);
    const char *savedir = ".";
    int MAXITER = 100;

    if (argc > 3) savedir = argv[3];
    if (argc > 4) MAXITER = atoi(argv[4]);

    char cmd[512];
    sprintf(cmd, "mkdir -p %s", savedir);
    system(cmd);

    printf("datafile: %s\n", datafile);
    printf("nclust: %d\n", nclust);
    printf("savedir: %s\n", savedir);

    KMData data = kmdata_load(datafile);
    KMClust clust = kmclust_new(nclust, data.dim);

    printf("ndata: %d\n", data.ndata);
    printf("dim: %d\n\n", data.dim);

    for (int i = 0; i < data.ndata; i++)
        data.assigns[i] = i % nclust;

    for (int c = 0; c < nclust; c++) {
        int icount = data.ndata / nclust;
        int extra = (c < (data.ndata % nclust)) ? 1 : 0;
        clust.counts[c] = icount + extra;
    }

    int curiter = 1;
    int nchanges = data.ndata;

    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    while (nchanges > 0 && curiter <= MAXITER) {
        for (int i = 0; i < nclust * clust.dim; i++)
            clust.features[i] = 0.0f;

        for (int i = 0; i < data.ndata; i++) {
            int c = data.assigns[i];
            cblas_saxpy(clust.dim,
                        1.0f,
                        &data.features[(size_t)i * (size_t)data.dim],
                        1,
                        &clust.features[(size_t)c * (size_t)clust.dim],
                        1);
        }

        for (int c = 0; c < nclust; c++) {
            if (clust.counts[c] > 0) {
                for (int d = 0; d < clust.dim; d++) {
                    clust.features[(size_t)c * (size_t)clust.dim + (size_t)d] /=
                        (float)clust.counts[c];
                }
            }
        }

        for (int c = 0; c < nclust; c++) clust.counts[c] = 0;

        nchanges = 0;

        for (int i = 0; i < data.ndata; i++) {
            int best_clust = 0;
            float best_distsq = INFINITY;
            for (int c = 0; c < nclust; c++) {
                float distsq = 0.0f;
                for (int d = 0; d < clust.dim; d++) {
                    float diff =
                        data.features[(size_t)i * (size_t)data.dim + (size_t)d] -
                        clust.features[(size_t)c * (size_t)clust.dim + (size_t)d];
                    distsq += diff * diff;
                }
                if (distsq < best_distsq) {
                    best_distsq = distsq;
                    best_clust = c;
                }
            }

            clust.counts[best_clust]++;
            if (best_clust != data.assigns[i]) {
                nchanges++;
                data.assigns[i] = best_clust;
            }
        }

        printf("%3d: %5d |", curiter, nchanges);
        for (int c = 0; c < nclust; c++)
            printf(" %4d", clust.counts[c]);
        printf("\n");

        curiter++;
    }

    if (curiter > MAXITER)
        printf("WARNING: maximum iteration %d exceeded\n", MAXITER);
    else
        printf("CONVERGED: after %d iterations\n", curiter);

    int *confusion = (int *)calloc((size_t)data.nlabels * (size_t)nclust, sizeof(int));
    for (int i = 0; i < data.ndata; i++)
        confusion[(size_t)data.labels[i] * (size_t)nclust + (size_t)data.assigns[i]]++;

    printf("\n==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST\n");
    printf("   ");
    for (int c = 0; c < nclust; c++) printf(" %4d", c);
    printf("  TOT\n");

    for (int l = 0; l < data.nlabels; l++) {
        int tot = 0;
        printf("%2d:", l);
        for (int c = 0; c < nclust; c++) {
            int v = confusion[(size_t)l * (size_t)nclust + (size_t)c];
            printf(" %4d", v);
            tot += v;
        }
        printf(" %4d\n", tot);
    }

    printf("TOT");
    int grand = 0;
    for (int c = 0; c < nclust; c++) {
        printf(" %4d", clust.counts[c]);
        grand += clust.counts[c];
    }
    printf(" %4d\n", grand);

    char outfile[512];
    sprintf(outfile, "%s/labels.txt", savedir);
    printf("\nSaving cluster labels to file %s\n", outfile);

    save_pgm_files(&clust, savedir);

    FILE *fout = fopen(outfile, "w");
    if (!fout) {
        printf("Failed to open file '%s'\n", outfile);
        exit(1);
    }
    for (int i = 0; i < data.ndata; i++)
        fprintf(fout, "%2d %2d\n", data.labels[i], data.assigns[i]);
    fclose(fout);

    free(confusion);
    free(data.features);
    free(data.assigns);
    free(data.labels);
    free(clust.features);
    free(clust.counts);

    return 0;
}
