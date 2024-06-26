#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "algo.cuh"
#include <float.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        return 1;
    }

    // Parse the arguments
    int n_block = std::atoi(argv[1]);
    int n_thread = std::atoi(argv[2]);
    const int n = std::atoi(argv[3]);
    int num_vertex_local = std::atoi(argv[4]);

    /*
    ------------------------------------------------Sparse------------------------------------------------
    */

    int factor = 40;
    // make sure the number of edges is larger than the max number of vertices
    int m = factor * n;
    // print m
    SparseGraph graph = generate_sparse_graph(n, m);

    // start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::vector<ClusterEdge> cuda_result = MSTSolver::algo_cuda_sparse(graph, n_block, n_thread, num_vertex_local);
    // sort the edges by from_v
    // std::sort(cuda_result.begin(), cuda_result.end(), [](const ClusterEdge& a, const ClusterEdge& b) {
    //     return a.weight < b.weight;
    // });
    // std::vector<int> parent = MSTSolver::algo_prim_sparse(graph);

    // end timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f\n", milliseconds);

    /*
    ------------------------------------------------Clique------------------------------------------------
    */

    // float* vertices = generate_clique_graph(n);

    // // start timer
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    // std::vector<ClusterEdge> cuda_result = MSTSolver::algo_cuda(vertices, n, n_block, n_thread, num_vertex_local);
    // // std::vector<int> parent = MSTSolver::algo_prim(vertices, n);

    // // end timer
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Time: %f\n", milliseconds);


    /*
    ------------------------------------------------Test the correctness of the result------------------------------------------------
    */


    // fully-connected 
    // double prim_weights = 0;
    // for (int i = 1; i < n; i++) {
    //     prim_weights += vertices[i * n + parent[i]];
    // }

    // // sparse
    // std::vector<ClusterEdge> prims_result(0);
    // for (int i = 1; i < n; i++) {
    //     // from i to parent[i]
    //     int i_start = graph.v_indices[i];
    //     int i_end = graph.v_indices[i + 1];

    //     // loop from i_start to i_end
    //     for (int j = i_start; j < i_end; j++) {
    //         if (graph.edges[j].to_v == parent[i]) {
    //             prims_result.push_back(ClusterEdge(i, parent[i], graph.edges[j].weight));
    //             break;
    //         }
    //     }
    // }

    // std::sort(prims_result.begin(), prims_result.end(), [](const ClusterEdge& a, const ClusterEdge& b) {
    //     return a.weight < b.weight;
    // });

    // float cuda_weight = 0;
    // for (int i = 0; i < cuda_result.size(); i++) {
    //     cuda_weight += cuda_result[i].weight;
    // }

    // float prims_weight = 0;
    // for (int i = 0; i < prims_result.size(); i++) {
    //     prims_weight += prims_result[i].weight;
    // }

    // printf("CUDA: %f\n", cuda_weight);
    // printf("Prim: %f\n", prims_weight);
}
