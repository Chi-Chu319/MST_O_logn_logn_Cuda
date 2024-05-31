#ifndef ALGO_H
#define ALGO_H

#include <vector>
#include <set>
#include <random>
#include <iostream>

struct ClusterEdge {
    int from_v;    // Starting vertex of an edge
    int to_v;      // Ending vertex of an edge
    float weight; // Weight of the edge

    __host__ __device__ ClusterEdge() : from_v(-1), to_v(-1), weight(100.0) {}
    __host__ __device__ ClusterEdge(int from, int to, float w) : from_v(from), to_v(to), weight(w) {}
};

struct SparseGraphEdge {
    int to_v;
    float weight;
};

struct SparseGraph {
    SparseGraphEdge* edges;
    int* sizes;
    int n;
    int m;
};


class SparseGraphBuilder {
public:
    int V;
    int edge_count = 0;
    std::vector<std::set<int>> adjList;
    std::vector<std::vector<SparseGraphEdge>> weights;

    SparseGraphBuilder(int vertices) : V(vertices) {
        adjList.resize(vertices);
        weights.resize(vertices);
    }

    void addEdge(int u, int v, float weight) {
        adjList[u].insert(v);
        adjList[v].insert(u);

        SparseGraphEdge edge1;
        edge1.to_v = v;
        edge1.weight = weight;
        weights[u].push_back(edge1);

        SparseGraphEdge edge2;
        edge2.to_v = u;
        edge2.weight = weight;
        weights[v].push_back(edge2);
        edge_count++;
    }

    SparseGraph toSparseGraph () {
        SparseGraph graph;
        graph.edges = new SparseGraphEdge[edge_count * 2];
        graph.sizes = new int[V];
        graph.n = V;
        graph.m = edge_count;


        int edge_idx = 0;
        for (int i = 0; i < V; ++i) {
            std::vector<SparseGraphEdge> edges = weights[i];
            graph.sizes[i] = edges.size();

            for (int j = 0; j < edges.size(); ++j) {
                graph.edges[edge_idx] = edges[j];

                edge_idx++;
            }
        }

        return graph;
    }

private:

};

__device__ int get_cluster_machine(int num_vertex_local, int v);
__global__ void min_to_cluster_kernel(ClusterEdge* to_cluster_buf, ClusterEdge* min_edges_buf, const float* vertices, int* cluster_ids, const  int n, int num_vertices_local);
__global__ void min_from_cluster_kernel(const ClusterEdge* to_cluster_buf, ClusterEdge* from_cluster_buf, ClusterEdge* min_edges_bufGPU, int* min_edges_stack_bufGPU, int* cluster_ids, int* cluster_sizes, const int n, int num_vertices_local);

__global__ void min_to_cluster_kernel_sparse(ClusterEdge* to_cluster_buf, ClusterEdge* min_edges_buf, SparseGraphEdge* edges, int* sizes, int* prefix_sum_sizes, int* cluster_ids, const int n, int num_vertices_local);
__global__ void min_from_cluster_kernel_sparse(const ClusterEdge* to_cluster_buf, ClusterEdge* from_cluster_buf, ClusterEdge* min_edges_bufGPU, int* min_edges_stack_bufGPU, int* sizes, int* prefix_sum_sizes, int* cluster_ids, int* cluster_sizes, const int n, int num_vertices_local);

__global__ void speedup_kernel(const float* vertices, ClusterEdge* from_cluster_buf, const int n, int num_vertices_local);


float* generate_clique_graph(int n);
SparseGraph generate_sparse_graph(int n, int m);

namespace MSTSolver {
    std::vector<ClusterEdge> algo_cuda(const float* vertices, const int n, int n_block, int n_thread, int num_vertex_local);
    std::vector<ClusterEdge> algo_cuda_sparse(const SparseGraph graph, int n_block, int n_thread, int num_vertices_local);
    std::vector<int> algo_prim(const float* vertices, const int n);
    float algo_prim_sparse(const SparseGraph graph, int* prefixsum_sizes);
}

#endif // ALGO_H
