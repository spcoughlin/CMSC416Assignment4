// kmeans_cuda.cu for problem 2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/types.h>

// filestats is in kmeans_util.cu (symlinked from kmeans_util.c)
int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _err = (call);                                       \
        if (_err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(_err));       \
            exit(1);                                                     \
        }                                                                \
    } while (0)

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

#define BLOCK_SIZE 256

// zeros out cluster feature sums and counts before each iteration
// launched with enough threads to cover max(nclust*dim, nclust)
__global__ void reset_clust_kernel(float *clust_feat, int *clust_cnt,
                                   int nclust, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nclust * dim)
        clust_feat[idx] = 0.0f;
    if (idx < nclust)
        clust_cnt[idx] = 0;
}

// Phase A part 1: one thread per data point accumulates its features into
// its assigned cluster using atomicAdd to handle concurrent writes
__global__ void accumulate_kernel(const float *data_feat,
                                  const int *data_assigns,
                                  float *clust_feat,
                                  int *clust_cnt,
                                  int ndata, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ndata) return;

    int c = data_assigns[i];
    atomicAdd(&clust_cnt[c], 1);
    for (int d = 0; d < dim; d++) {
        atomicAdd(&clust_feat[(size_t)c * dim + d],
                  data_feat[(size_t)i * dim + d]);
    }
}

// Phase A part 2: one thread per (cluster, dim) cell divides the accumulated
// sum by the cluster count to get the mean center
__global__ void normalize_kernel(float *clust_feat, const int *clust_cnt,
                                 int nclust, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nclust * dim) return;

    int c = idx / dim;
    if (clust_cnt[c] > 0)
        clust_feat[idx] /= (float)clust_cnt[c];
}

// Phase B: one thread per data point finds the nearest cluster center,
// updates the assignment, tallies the new cluster count, and counts changes
__global__ void assign_kernel(const float *data_feat,
                              int *data_assigns,
                              const float *clust_feat,
                              int *clust_cnt,
                              int *d_nchanges,
                              int ndata, int dim, int nclust) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ndata) return;

    int best_clust = 0;
    float best_distsq = FLT_MAX;

    for (int c = 0; c < nclust; c++) {
        float distsq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = data_feat[(size_t)i * dim + d]
                       - clust_feat[(size_t)c * dim + d];
            distsq += diff * diff;
        }
        if (distsq < best_distsq) {
            best_distsq = distsq;
            best_clust = c;
        }
    }

    atomicAdd(&clust_cnt[best_clust], 1);

    if (best_clust != data_assigns[i]) {
        atomicAdd(d_nchanges, 1);
        data_assigns[i] = best_clust;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: kmeans_cuda <datafile> <nclust> [savedir] [maxiter]\n");
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

    // round-robin initial assignment (same as serial/omp)
    for (int i = 0; i < data.ndata; i++)
        data.assigns[i] = i % nclust;
    for (int c = 0; c < nclust; c++) {
        int icount = data.ndata / nclust;
        int extra = (c < (data.ndata % nclust)) ? 1 : 0;
        clust.counts[c] = icount + extra;
    }

    // allocate device memory
    int ndata_h = data.ndata;
    int dim_h = data.dim;
    size_t feat_bytes = (size_t)ndata_h * dim_h * sizeof(float);
    size_t asgn_bytes = (size_t)ndata_h * sizeof(int);
    size_t cfeat_bytes = (size_t)nclust * dim_h * sizeof(float);
    size_t ccnt_bytes = (size_t)nclust * sizeof(int);

    float *d_data_feat;
    int *d_data_assigns;
    float *d_clust_feat;
    int *d_clust_cnt;
    int *d_nchanges;

    CUDA_CHECK(cudaMalloc((void **)&d_data_feat, feat_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_data_assigns, asgn_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_clust_feat, cfeat_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_clust_cnt, ccnt_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_nchanges, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data_feat, data.features, feat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_assigns, data.assigns, asgn_bytes, cudaMemcpyHostToDevice));

    // grid dimensions derived from data size
    int blocks_data = (ndata_h + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_cfeat = nclust * dim_h;
    int blocks_cfeat = (total_cfeat + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int reset_elems = (total_cfeat > nclust) ? total_cfeat : nclust;
    int blocks_reset = (reset_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int curiter = 1;
    int nchanges = ndata_h;

    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    while (nchanges > 0 && curiter <= MAXITER) {

        // Phase A: compute new cluster centers
        reset_clust_kernel<<<blocks_reset, BLOCK_SIZE>>>(
            d_clust_feat, d_clust_cnt, nclust, dim_h);
        CUDA_CHECK(cudaGetLastError());

        accumulate_kernel<<<blocks_data, BLOCK_SIZE>>>(
            d_data_feat, d_data_assigns, d_clust_feat, d_clust_cnt,
            ndata_h, dim_h);
        CUDA_CHECK(cudaGetLastError());

        normalize_kernel<<<blocks_cfeat, BLOCK_SIZE>>>(
            d_clust_feat, d_clust_cnt, nclust, dim_h);
        CUDA_CHECK(cudaGetLastError());

        // copy centers to host so we can restore them after resetting counts
        CUDA_CHECK(cudaMemcpy(clust.features, d_clust_feat,
                              cfeat_bytes, cudaMemcpyDeviceToHost));

        // Phase B: assign each data point to nearest center
        // reset counts and change counter, restore the centers
        CUDA_CHECK(cudaMemset(d_clust_cnt, 0, ccnt_bytes));
        CUDA_CHECK(cudaMemset(d_nchanges, 0, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_clust_feat, clust.features,
                              cfeat_bytes, cudaMemcpyHostToDevice));

        assign_kernel<<<blocks_data, BLOCK_SIZE>>>(
            d_data_feat, d_data_assigns, d_clust_feat, d_clust_cnt,
            d_nchanges, ndata_h, dim_h, nclust);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&nchanges, d_nchanges, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(clust.counts, d_clust_cnt, ccnt_bytes, cudaMemcpyDeviceToHost));

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

    // copy final assignments back for output
    CUDA_CHECK(cudaMemcpy(data.assigns, d_data_assigns,
                          asgn_bytes, cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < data.ndata; i++)
        fprintf(fout, "%2d %2d\n", data.labels[i], data.assigns[i]);
    fclose(fout);

    cudaFree(d_data_feat);
    cudaFree(d_data_assigns);
    cudaFree(d_clust_feat);
    cudaFree(d_clust_cnt);
    cudaFree(d_nchanges);

    free(confusion);
    free(data.features);
    free(data.assigns);
    free(data.labels);
    free(clust.features);
    free(clust.counts);

    return 0;
}
