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


    // printf("generating graph-1 \n");

    SparseGraph graph = generate_sparse_graph(n, 10 * n);


    int* prefixsum_sizes = new int[n];
    prefixsum_sizes[0] = graph.sizes[0];
    for (int i = 1; i < n; ++i) {
        prefixsum_sizes[i] = prefixsum_sizes[i - 1] + graph.sizes[i];
    }

    std::vector<ClusterEdge> cuda_result = MSTSolver::algo_cuda_sparse(graph, n_block, n_thread, num_vertex_local);
    float prim_weights = MSTSolver::algo_prim_sparse(graph, prefixsum_sizes);

    // float* vertices = generate_clique_graph(n);

    // // start timer
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    // std::vector<ClusterEdge> cuda_result = MSTSolver::algo_cuda(vertices, n, n_block, n_thread, num_vertex_local);
    // std::vector<int> prim_parents = MSTSolver::algo_prim(vertices, n);

    // // end timer
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Time: %f\n", milliseconds);

    float cuda_weights = 0;
    for (int i = 0; i < cuda_result.size(); i++) {
        cuda_weights += cuda_result[i].weight;
    }

    // float prim_weights = 0;
    // for (int i = 1; i < n; i++) {
    //     prim_weights += graph.edges[i * n + prim_parents[i]].weight;
    // }

    printf("CUDA: %f\n", cuda_weights);
    printf("Prim: %f\n", prim_weights);
}

