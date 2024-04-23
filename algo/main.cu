#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "algo.cuh"
#include <float.h>

// int main() {
//     int n_block = 1;
//     int n_thread = 8192;
//     const int n = 8192;
//     int num_vertex_local = 1;

//     // double *vertices = generate_clique_graph(n);
//     double vertices[n * n];
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             vertices[i * n + j] = (double)rand() / RAND_MAX;
//             vertices[j * n + i] = vertices[i * n + j];
//         }
//     }

//     for (int i = 0; i < n; ++i) {
//         vertices[i * n + i] = 200;
//     }

//     // start timer
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);


//     std::vector<ClusterEdge> cuda_result = MSTSolver::algo_cuda(vertices, n, n_block, n_thread, num_vertex_local);
//     // std::vector<int> prim_parents = MSTSolver::algo_prim(vertices, n);

//     // end timer
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Time: %f\n", milliseconds);


//     // double cuda_weights = 0;
//     // for (int i = 0; i < cuda_result.size(); i++) {
//     //     cuda_weights += cuda_result[i].weight;
//     // }

//     // double prim_weights = 0;
//     // for (int i = 1; i < n; i++) {
//     //     prim_weights += vertices[i * n + prim_parents[i]];
//     // }

//     // printf("CUDA: %f\n", cuda_weights);
//     // printf("Prim: %f\n", prim_weights);
// }

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}
