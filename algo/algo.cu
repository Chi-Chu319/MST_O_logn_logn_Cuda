#include "algo.cuh"
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <algorithm>
#include <set>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

int get_cluster_leader_host(int* cluster_ids, int v) {
    int leader = v;
    while (cluster_ids[leader] != leader) {
        leader = cluster_ids[leader];
    }

    return leader;
}

bool get_cluster_finished(int* cluster_ids, bool* cluster_finished, int v) {
    int leader = get_cluster_leader_host(cluster_ids, v);
    return cluster_finished[leader];
}

bool cluster_safe_union(int* cluster_ids, int* cluster_size, int p, int q) {
    int i = get_cluster_leader_host(cluster_ids, p);
    int j = get_cluster_leader_host(cluster_ids, q);

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
    cluster_finished[get_cluster_leader_host(cluster_ids, i)] = true;
}

namespace MSTSolver {

    //  n is the number of vertices
    std::vector<ClusterEdge> algo_cuda(const double* vertices, int n, int n_block, int n_thread, int num_vertices_local) {
        float cpu_time = 0;

        int num_vertices = n;
        int k = 0;

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
        CHECK(cudaMemcpy(verticesGPU, vertices, n * n * sizeof(double), cudaMemcpyHostToDevice));
        
        ClusterEdge* to_cluster_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&to_cluster_bufGPU, n * n * sizeof(ClusterEdge)));

        ClusterEdge* from_cluster_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&from_cluster_bufGPU, n * n * sizeof(ClusterEdge)));

        int num_clusters = num_vertices;

        int *deviceResult; // Device pointer to memory

        // Allocate device memory for the result
        cudaMalloc((void **)&deviceResult, sizeof(int));

        ClusterEdge* from_cluster_buf = new ClusterEdge[n * n];

        while (true) {
            CHECK(cudaMemset(to_cluster_bufGPU, 0, n * n * sizeof(ClusterEdge)));
            CHECK(cudaMemset(from_cluster_bufGPU, 0, n * n * sizeof(ClusterEdge)));
            CHECK(cudaMemcpy(cluster_idsGPU, cluster_ids, n * sizeof(int), cudaMemcpyHostToDevice));


            // if (k == 0) {
            //     speedup_kernel<<<n_block, n_thread>>>(vertices, from_cluster_buf, n, num_vertices_local);
            //     CHECK(cudaGetLastError());
            // }
            // else {
                min_to_cluster_kernel<<<n_block, n_thread>>>(to_cluster_bufGPU, verticesGPU, cluster_idsGPU, n, num_vertices_local, k);
                CHECK(cudaGetLastError());

                min_from_cluster_kernel<<<n_block, n_thread>>>(to_cluster_bufGPU, from_cluster_bufGPU, cluster_idsGPU, n, num_vertices_local);
                CHECK(cudaGetLastError());
            // }

            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(from_cluster_buf, from_cluster_bufGPU, n * n * sizeof(ClusterEdge), cudaMemcpyDeviceToHost));

            // start timer
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventSynchronize(start);

            // rank 0 merge edges
            std::vector<ClusterEdge> edges_to_add;

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (from_cluster_buf[i * n + j].from_v != 0) {
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
                int to_cluster = get_cluster_leader_host(cluster_ids, edge.to_v);

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
                int from_cluster = get_cluster_leader_host(cluster_ids, edge.from_v);
                int to_cluster = get_cluster_leader_host(cluster_ids, edge.to_v);

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

            // count the number of unique numbers in cluster_ids
            std::set<int> unique_cluster_finder_id(cluster_ids, cluster_ids + num_vertices);
            num_clusters = unique_cluster_finder_id.size();

            k++;

            if (k >= 10) {
                throw std::runtime_error("k >= 10");
            }

            if (num_clusters == 1) {
                break;
            }

            // flatten cluster_ids
            int* new_cluster_ids = new int[n];
            for (int i = 0; i < n; i++) {
                new_cluster_ids[i] = get_cluster_leader_host(cluster_ids, i);
            }

            delete[] cluster_ids;
            cluster_ids = new_cluster_ids;

            for (int i = 0; i < n; ++i) {
                cluster_sizes[i] = 1;
            }

            // end timer
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cpu_time += milliseconds;
        }

        CHECK(cudaFree(verticesGPU));
        CHECK(cudaFree(cluster_idsGPU));
        CHECK(cudaFree(to_cluster_bufGPU));
        CHECK(cudaFree(from_cluster_bufGPU));


        delete[] cluster_ids;
        delete[] cluster_sizes;
        delete[] from_cluster_buf;

        // print cpu time
        std::cout << "CPU time: " << cpu_time << std::endl;

        return mst_edges;
    }

    std::vector<int> algo_prim(const double* vertices, const int n) {
        std::vector<int> parent(n, -1);
        std::vector<double> key(n, std::numeric_limits<double>::max());
        std::vector<bool> mstSet(n, false);

        key[0] = 0;
        for (int count = 0; count < n - 1; count++) {
            int u = -1;
            double min_key = std::numeric_limits<double>::max();
            for (int i = 0; i < n; i++) {
                if (!mstSet[i] && key[i] < min_key) {
                    u = i;
                    min_key = key[i];
                }
            }
            mstSet[u] = true;
            for (int v = 0; v < n; v++) {
                if (!mstSet[v] && vertices[u * n + v] < key[v]) {
                    parent[v] = u;
                    key[v] = vertices[u * n + v];
                }
            }
        }

        return parent;
    }

}

