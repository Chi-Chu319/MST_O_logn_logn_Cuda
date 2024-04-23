#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

double* generate_clique_graph(int n) {
    srand(time(0)); 

    double* vertices = (double*)malloc(n * n * sizeof(double)); 

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            vertices[i * n + j] = (double)rand() / RAND_MAX;
            vertices[j * n + i] = vertices[i * n + j];
        }
    }

    for (int i = 0; i < n; ++i) {
        vertices[i * n + i] = 200;
    }

    return vertices;
}

// double* generate_sparse_graph(int n, int m) {

// }