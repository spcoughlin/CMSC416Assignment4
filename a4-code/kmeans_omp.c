// kmeans_omp.c for problem 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>

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

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: kmeans_serial <datafile> <nclust> [savedir] [maxiter]\n");
        return 1;
    }

    const char *datafile = argv[1];
    int nclust = atoi(argv[2]);
    const char *savedir = ".";
    int MAXITER = 100;