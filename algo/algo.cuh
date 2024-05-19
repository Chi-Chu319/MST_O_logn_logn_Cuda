#ifndef ALGO_H
#define ALGO_H

#include <vector>
#include <random>

struct ClusterEdge {
    int from_v;    // Starting vertex of an edge
    int to_v;      // Ending vertex of an edge
    double weight; // Weight of the edge

    __host__ __device__ ClusterEdge() : from_v(-1), to_v(-1), weight(100.0) {}
    __host__ __device__ ClusterEdge(int from, int to, double w) : from_v(from), to_v(to), weight(w) {}
};

struct SparseGraph {
    int* edges;
    double* weights;
};

__device__ int get_cluster_machine(int num_vertex_local, int v);
__global__ void min_to_cluster_kernel(ClusterEdge* to_cluster_buf, const double* vertices, int* cluster_ids, const  int n, int num_vertices_local);
__global__ void min_from_cluster_kernel(const ClusterEdge* to_cluster_buf, ClusterEdge* from_cluster_buf, int* cluster_ids, int* cluster_sizes, const int n, int num_vertices_local);
__global__ void speedup_kernel(const double* vertices, ClusterEdge* from_cluster_buf, const int n, int num_vertices_local);


double* generate_clique_graph(int n);
double* generate_sparse_graph(int n, int m);

namespace MSTSolver {
    std::vector<ClusterEdge> algo_cuda(const double* vertices, const int n, int n_block, int n_thread, int num_vertex_local);
    std::vector<int> algo_prim(const double* vertices, const int n);
}

#endif // ALGO_H
