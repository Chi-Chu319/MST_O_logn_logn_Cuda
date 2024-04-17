#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "algo.cuh"
#include <float.h>

int main() {
    int n_block = 4;
    int n_thread = 1024;
    const int n = 4096;

    double vertices[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            vertices[i * n + j] = (double)rand() / RAND_MAX;
            vertices[j * n + i] = vertices[i * n + j];
        }
    }

    for (int i = 0; i < n; ++i) {
        vertices[i * n + i] = 200;
    }

    std::vector<ClusterEdge> cuda_result = MSTSolver::algo_cuda(vertices, n, n_block, n_thread);
    std::vector<int> prim_parents = MSTSolver::algo_prim(vertices, n);

    double cuda_weights = 0;
    for (int i = 0; i < cuda_result.size(); i++) {
        cuda_weights += cuda_result[i].weight;
    }

    double prim_weights = 0;
    for (int i = 1; i < n; i++) {
        prim_weights += vertices[i * n + prim_parents[i]];
    }

    printf("CUDA: %f\n", cuda_weights);
    printf("Prim: %f\n", prim_weights);
}
