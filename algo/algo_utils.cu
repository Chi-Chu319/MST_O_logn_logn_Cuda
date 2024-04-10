#include "algo.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__device__ int get_cluster_leader(int* cluster_ids, int v) {
    int leader = v;
    while (cluster_ids[leader] != leader) {
        leader = cluster_ids[leader];
    }

    return leader;
}

__device__ int get_cluster_machine(int num_vertex_local, int v) {
    return v / num_vertex_local;
}

__global__ void min_to_cluster_kernel(ClusterEdge* to_cluster_buf, const double* vertices, int* cluster_ids, int n, int num_vertices_local) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    int vertex_local_start = (i * n + j) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int from_v = vertex_local + vertex_local_start;

        // init an array of cluster edges with length n
        ClusterEdge cluster_edges[n];

        for (int to_v = 0; to_v < n; ++to_v) {
            double weight = vertices[vertex_local * n + to_v];
            int from_cluster = get_cluster_leader(cluster_ids, from_v);
            int to_cluster = get_cluster_leader(cluster_ids, to_v);

            if (from_cluster != to_cluster) {
                if (cluster_edges[to_cluster].from_v == -1) {
                    cluster_edges[to_cluster] = ClusterEdge(from_v, to_v, weight);
                } else if (weight < cluster_edges[to_cluster].weight) {
                    cluster_edges[to_cluster] = ClusterEdge(from_v, to_v, weight);
                }
            }
        }

        //  loop cluster_edges for non empty item and update cluster_ids
        for (int k = 0; k < n; ++k) {
            if (cluster_edges[k].from_v != -1) {

                ClusterEdge edge = cluster_edges[k];

                int to_cluster = get_cluster_leader(cluster_ids, edge.to_v);
                to_cluster_buf[to_cluster * n + edge.from_v] = edge;
            }
        }
    }
}

__global__ void min_from_cluster_kernel(const ClusterEdge* to_cluster_buf, ClusterEdge* from_cluster_buf, int* cluster_ids, int n, int num_vertices_local) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    int vertex_local_start = (i * n + j) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int vertex = vertex_local + vertex_local_start;
        ClusterEdge cluster_edges[n];

        for (int from_v = 0; from_v < n; ++from_v) {
            ClusterEdge edge = to_cluster_buf[vertex * n + from_v];

            if (edge.from_v != -1) {
                int from_cluster = get_cluster_leader(cluster_ids, edge.from_v);

                if (cluster_edges[from_cluster].from_v == -1) {
                    cluster_edges[from_cluster] = edge;
                } else if (edge.weight < cluster_edges[from_cluster].weight) {
                    cluster_edges[from_cluster] = edge;
                }
            }
        }

        // For al non empty cluster_edges, update cluster_ids
        for (int k = 0; k < n; ++k) {
            if (cluster_edges[k].from_v != -1) {
                from_cluster_buf[vertex * n + k] = cluster_edges[k];
            }
        }
    }

}