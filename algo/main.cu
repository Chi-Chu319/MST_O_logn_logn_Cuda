#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "algo.cuh"
#include <float.h>
#include <iostream>
#include <cstdlib> // For atoi()

int main(int argc, char* argv[]) {
    if (argc < 5) {
        return 1;
    }

    // Parse the arguments
    int n_block = std::atoi(argv[1]);
    int n_thread = std::atoi(argv[2]);
    const int n = std::atoi(argv[3]);
    int num_vertex_local = std::atoi(argv[4]);

    // int n_block = 8;
    // int n_thread = 1024;
    // const int n = 8192;
    // int num_vertex_local = 1;

    // double *vertices = generate_clique_graph(n);
    double* vertices = generate_clique_graph(n);

    // start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    std::vector<ClusterEdge> cuda_result = MSTSolver::algo_cuda(vertices, n, n_block, n_thread, num_vertex_local);
    std::vector<int> prim_parents = MSTSolver::algo_prim(vertices, n);

    // end timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f\n", milliseconds);

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

