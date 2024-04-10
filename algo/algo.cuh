#ifndef ALGO_H
#define ALGO_H

#include <vector>
#include <random>

struct ClusterEdge {
    int from_v;    // Starting vertex of an edge
    int to_v;      // Ending vertex of an edge
    double weight; // Weight of the edge

    __host__ __device__ ClusterEdge() : from_v(-1), to_v(-1), weight(0.0) {}
    __host__ __device__ ClusterEdge(int from, int to, double w) : from_v(from), to_v(to), weight(w) {}
};

__device__ int get_cluster_machine(int num_vertex_local, int v);
__global__ void min_to_cluster_kernel(ClusterEdge* to_cluster_buf, const double* vertices, int* cluster_ids, int n, int num_vertices_local);
__global__ void min_from_cluster_kernel(const ClusterEdge* to_cluster_buf, ClusterEdge* from_cluster_buf, int* cluster_ids, int n, int num_vertices_local);


#endif // ALGO_H
