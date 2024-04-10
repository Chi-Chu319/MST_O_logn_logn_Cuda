#include "algo.cuh"
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <algorithm>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

int get_cluster_leader(int* cluster_ids, int v) {
    int leader = v;
    while (cluster_ids[leader] != leader) {
        leader = cluster_ids[leader];
    }

    return leader;
}

bool get_cluster_finished(int* cluster_ids, bool* cluster_finished, int v) {
    int leader = get_cluster_leader(cluster_ids, v);
    return cluster_finished[leader];
}

bool cluster_safe_union(int* cluster_ids, int* cluster_size, int p, int q) {
    int i = get_cluster_leader(cluster_ids, p);
    int j = get_cluster_leader(cluster_ids, q);

    if (i == j) {
        return false;
    }

    if (cluster_size[i] < cluster_size[j]) {
        cluster_ids[i] = j;
        cluster_size[j] += cluster_size[i];
        return true;
    } else {
        cluster_ids[j] = i;
        cluster_size[i] += cluster_size[j];
        return true;
    }
}

void cluster_set_finished(int* cluster_ids, bool* cluster_finished, int i) {
    cluster_finished[get_cluster_leader(cluster_ids, i)] = true;
}

namespace MSTSolverCuda {

    //  n is the number of vertices
    std::vector<ClusterEdge> algo_cuda(const double* vertices, int n) {
        int num_vertices = n;
        int num_vertices_local = 1;

        int* cluster_ids = new int[n];

        for (int i = 0; i < n; ++i) {
            cluster_ids[i] = i;
        }

        int* cluster_sizes = new int[n];

        for (int i = 0; i < n; ++i) {
            cluster_sizes[i] = 1;
        }

        std::vector<ClusterEdge> mst_edges = std::vector<ClusterEdge>();
        int* cluster_idsGPU = NULL;
        CHECK(cudaMalloc((void**)&cluster_idsGPU, n * sizeof(int)));
        CHECK(cudaMemcpy(cluster_idsGPU, cluster_ids, n * sizeof(int), cudaMemcpyHostToDevice));
        
        double* verticesGPU = NULL;
        CHECK(cudaMalloc((void**)&verticesGPU, n * n * sizeof(double)));
        CHECK(cudaMemcpy(verticesGPU, vertices, n * n * sizeof(int), cudaMemcpyHostToDevice));
        

        while (true) {
            ClusterEdge* to_cluster_bufGPU = NULL;
            CHECK(cudaMalloc((void**)&to_cluster_bufGPU, n * n * sizeof(ClusterEdge)));

            ClusterEdge* from_cluster_bufGPU = NULL;
            CHECK(cudaMalloc((void**)&from_cluster_bufGPU, n * n * sizeof(ClusterEdge)));

            min_to_cluster_kernel<<<16, 1024>>>(to_cluster_bufGPU, verticesGPU, cluster_idsGPU, n, num_vertices_local);
        
            min_from_cluster_kernel<<<16, 1024>>>(to_cluster_bufGPU, from_cluster_bufGPU, cluster_idsGPU, n, num_vertices_local);

            ClusterEdge* from_cluster_buf = new ClusterEdge[n * n];
            CHECK(cudaMemcpy(from_cluster_buf, from_cluster_bufGPU, n * n * sizeof(ClusterEdge), cudaMemcpyDeviceToHost));

            CHECK(cudaFree(from_cluster_bufGPU));
            CHECK(cudaFree(to_cluster_bufGPU));

            // rank 0 merge edges
            std::vector<ClusterEdge> edges_to_add;

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (from_cluster_buf[i * n + j].from_v != -1) {
                        edges_to_add.push_back(from_cluster_buf[i * n + j]);
                    }
                }
            }

            std::sort(edges_to_add.begin(), edges_to_add.end(), [](ClusterEdge a, ClusterEdge b) {
                return a.weight < b.weight;
            });

            std::vector<bool> heaviest_edges(edges_to_add.size());
            std::fill(heaviest_edges.begin(), heaviest_edges.end(), false);
            std::map<int, bool> encountered_clusters;

            for (int i = edges_to_add.size() - 1; i >= 0; --i) {
                ClusterEdge edge = edges_to_add[i];
                int to_cluster = get_cluster_leader(cluster_ids, edge.to_v);

                if (encountered_clusters.find(to_cluster) == encountered_clusters.end()) {
                    encountered_clusters[to_cluster] = true;
                    heaviest_edges[i] = true;
                }
            }

            // declare a bool array with size n and fill it with false
            bool* cluster_finished = new bool[n];
            std::fill(cluster_finished, cluster_finished + n, false);

            for (int i = 0; i < edges_to_add.size(); ++i) {
                ClusterEdge edge = edges_to_add[i];
                int from_cluster = get_cluster_leader(cluster_ids, edge.from_v);
                int to_cluster = get_cluster_leader(cluster_ids, edge.to_v);

                bool from_cluster_finished = get_cluster_finished(cluster_ids, cluster_finished, from_cluster);
                bool to_cluster_finished = get_cluster_finished(cluster_ids, cluster_finished, to_cluster);
                
                if (to_cluster_finished && from_cluster_finished) {
                    continue;
                }

                bool merged = cluster_safe_union(cluster_ids, cluster_sizes, from_cluster, to_cluster);

                if (merged) {
                    mst_edges.push_back(edge);
                    if (heaviest_edges[i] || (from_cluster_finished || to_cluster_finished)) {
                        cluster_set_finished(cluster_ids, cluster_finished, from_cluster);
                        cluster_set_finished(cluster_ids, cluster_finished, to_cluster);
                    }
                } else {
                    if (heaviest_edges[i]) {
                        cluster_set_finished(cluster_ids, cluster_finished, to_cluster);
                    }
                }
            }

            delete[] cluster_finished;

            // flatten?
        }

        CHECK(cudaFree(verticesGPU));
        CHECK(cudaFree(cluster_idsGPU));


        delete[] cluster_ids;
        delete[] cluster_sizes;

        return mst_edges;
    }
}

