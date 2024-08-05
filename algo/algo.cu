#include "algo.cuh"
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <algorithm>
#include <set>
#include <chrono>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

int get_cluster_leader_host(int* cluster_ids, int v) {
    if (cluster_ids[v] != v) {
        cluster_ids[v] = get_cluster_leader_host(cluster_ids, cluster_ids[v]);
    }

    return cluster_ids[v];
}

bool get_cluster_finished(int* cluster_ids, bool* cluster_finished, int v) {
    int leader = get_cluster_leader_host(cluster_ids, v);
    return cluster_finished[leader];
}

void get_prefix_sum(int* array, int* result, int n) {
    result[0] = 0;
    for (int i = 1; i < n; ++i) {
        result[i] = array[i - 1] + result[i - 1];
    }
}

void compute_cluster_leader_sizes(int* cluster_leader_sizes, int* cluster_ids, int* v_indices, int n) {
    for (int i = 0; i < n + 1; ++i) {
        cluster_leader_sizes[i] = 0;
    }

    for (int i = 0; i < n; ++i) {
        int leader = get_cluster_leader_host(cluster_ids, i);
        //  shifted to the right by 1
        cluster_leader_sizes[leader + 1] += (v_indices[i + 1] - v_indices[i]);
    }

    // compute prefix sum of cluster_leader_sizes
    for (int i = 1; i < n + 1; ++i) {
        cluster_leader_sizes[i] += cluster_leader_sizes[i - 1];
    }
}

bool cluster_safe_union(int* cluster_ids, int* cluster_sizes, int p, int q) {
    int i = get_cluster_leader_host(cluster_ids, p);
    int j = get_cluster_leader_host(cluster_ids, q);

    if (i == j) {
        return false;
    }

    if (cluster_sizes[i] < cluster_sizes[j]) {
        cluster_ids[i] = j;
        cluster_sizes[j] += cluster_sizes[i];
        cluster_sizes[i] = 0;
        return true;
    } else {
        cluster_ids[j] = i;
        cluster_sizes[i] += cluster_sizes[j];
        cluster_sizes[j] = 0;
        return true;
    }
}

void cluster_set_finished(int* cluster_ids, bool* cluster_finished, int i) {
    cluster_finished[get_cluster_leader_host(cluster_ids, i)] = true;
}

namespace MSTSolver {

    //  n is the number of vertices
    std::vector<ClusterEdge> algo_cuda(const float* vertices, int n, int n_block, int n_thread, int num_vertices_local) {
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
        
        int* cluster_sizesGPU = NULL;
        CHECK(cudaMalloc((void**)&cluster_sizesGPU, n * sizeof(int)));
        CHECK(cudaMemcpy(cluster_sizesGPU, cluster_sizes, n * sizeof(int), cudaMemcpyHostToDevice));

        float* verticesGPU = NULL;
        CHECK(cudaMalloc((void**)&verticesGPU, n * n * sizeof(float)));
        CHECK(cudaMemcpy(verticesGPU, vertices, n * n * sizeof(float), cudaMemcpyHostToDevice));
        
        ClusterEdge* to_cluster_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&to_cluster_bufGPU, n * n * sizeof(ClusterEdge)));

        ClusterEdge* from_cluster_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&from_cluster_bufGPU, n * n * sizeof(ClusterEdge)));

        ClusterEdge* min_edges_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&min_edges_bufGPU, n * n * sizeof(ClusterEdge)));

        int* min_edges_stack_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&min_edges_stack_bufGPU, n * n * sizeof(int)));

        int num_clusters = num_vertices;

        ClusterEdge* from_cluster_buf = new ClusterEdge[n * n];

        while (true) {
            CHECK(cudaMemset(to_cluster_bufGPU, -1, n * n * sizeof(ClusterEdge)));
            CHECK(cudaMemset(from_cluster_bufGPU, -1, n * n * sizeof(ClusterEdge)));
            CHECK(cudaMemset(min_edges_bufGPU, -1, n * n * sizeof(ClusterEdge)));
            CHECK(cudaMemset(min_edges_stack_bufGPU, -1, n * n * sizeof(int)));
            CHECK(cudaMemcpy(cluster_idsGPU, cluster_ids, n * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(cluster_sizesGPU, cluster_sizes, n * sizeof(int), cudaMemcpyHostToDevice));

            if (k == 100) {
                speedup_kernel<<<n_block, n_thread>>>(verticesGPU, from_cluster_bufGPU, n, num_vertices_local);
                CHECK(cudaGetLastError());
            }
            else {
                min_to_cluster_kernel<<<n_block, n_thread>>>(to_cluster_bufGPU, min_edges_bufGPU, verticesGPU, cluster_idsGPU, n, num_vertices_local);
                CHECK(cudaGetLastError());
                
                CHECK(cudaMemset(min_edges_bufGPU, -1, n * n * sizeof(ClusterEdge)));

                min_from_cluster_kernel<<<n_block, n_thread>>>(to_cluster_bufGPU, from_cluster_bufGPU, min_edges_bufGPU, min_edges_stack_bufGPU, cluster_idsGPU, cluster_sizesGPU, n, num_vertices_local);
                CHECK(cudaGetLastError());
            }

            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(from_cluster_buf, from_cluster_bufGPU, n * n * sizeof(ClusterEdge), cudaMemcpyDeviceToHost));

            // start timer in C++ way
            auto start_cpu = std::chrono::high_resolution_clock::now();

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
            // init bool arr with size n called encountered_clusters
            bool encountered_clusters[n] = {false};

            for (int i = edges_to_add.size() - 1; i >= 0; --i) {
                ClusterEdge edge = edges_to_add[i];
                int to_cluster = get_cluster_leader_host(cluster_ids, edge.to_v);

                if (!encountered_clusters[to_cluster]) {
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
                    num_clusters--;
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

            k++;

            // end timer
            auto end_cpu = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

            cpu_time += duration.count();

            if (k >= 10) {
                throw std::runtime_error("k >= 10");
            }

            if (num_clusters == 1) {
                break;
            }
        }

        CHECK(cudaFree(verticesGPU));
        CHECK(cudaFree(cluster_idsGPU));
        CHECK(cudaFree(to_cluster_bufGPU));
        CHECK(cudaFree(from_cluster_bufGPU));
        CHECK(cudaFree(min_edges_bufGPU));
        CHECK(cudaFree(min_edges_stack_bufGPU));

        delete[] cluster_ids;
        delete[] cluster_sizes;
        delete[] from_cluster_buf;

        // print cpu time
        std::cout << "CPU time: " << cpu_time << std::endl;

        return mst_edges;
    }

    //  n is the number of vertices
    std::vector<ClusterEdge> algo_cuda_sparse(const SparseGraph graph, int n_block, int n_thread, int num_vertices_local) {
        float cpu_time = 0;

        int n = graph.n;
        int m = graph.m;
        int k = 0;

        int* cluster_ids = new int[n];
        // a sequential listing of cluster members.
        int* cluster_members = new int[n];

        for (int i = 0; i < n; ++i) {
            cluster_ids[i] = i;
            cluster_members[i] = i;
        }

        int* cluster_sizes = new int[n];
        int* prefix_sum_cluster_sizes = new int[n + 1];

        for (int i = 0; i < n; ++i) {
            cluster_sizes[i] = 1;
            prefix_sum_cluster_sizes[i] = i;
        }

        prefix_sum_cluster_sizes[n] = n;

        // Total degree of a cluster 
        // cluster leader index => buffer start (from cluster buffer)
        // cluster leader index + ! => buffer end (end cluster buffer)
        int* cluster_leader_sizes = new int[n + 1];

        std::vector<ClusterEdge> mst_edges = std::vector<ClusterEdge>();
        int* cluster_idsGPU = NULL;
        CHECK(cudaMalloc((void**)&cluster_idsGPU, n * sizeof(int)));
        CHECK(cudaMemcpy(cluster_idsGPU, cluster_ids, n * sizeof(int), cudaMemcpyHostToDevice));
        
        int* cluster_sizesGPU = NULL;
        CHECK(cudaMalloc((void**)&cluster_sizesGPU, n * sizeof(int)));
        CHECK(cudaMemcpy(cluster_sizesGPU, cluster_sizes, n * sizeof(int), cudaMemcpyHostToDevice));

        SparseGraphEdge* edgesGPU = NULL;
        CHECK(cudaMalloc((void**)&edgesGPU, 2 * m * sizeof(SparseGraphEdge)));
        CHECK(cudaMemcpy(edgesGPU, graph.edges, 2 * m * sizeof(SparseGraphEdge), cudaMemcpyHostToDevice));
        
        int* v_indicesGPU = NULL;
        CHECK(cudaMalloc((void**)&v_indicesGPU, (n + 1) * sizeof(int)));
        CHECK(cudaMemcpy(v_indicesGPU, graph.v_indices, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        
        int* cluster_membersGPU = NULL;
        CHECK(cudaMalloc((void**)&cluster_membersGPU, (n) * sizeof(int)));

        int* prefix_sum_cluster_sizesGPU = NULL;
        CHECK(cudaMalloc((void**)&prefix_sum_cluster_sizesGPU, (n + 1) * sizeof(int)));

        ClusterEdge* min_edges_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&min_edges_bufGPU, 2 * m * sizeof(ClusterEdge)));
 
        ClusterEdge* to_cluster_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&to_cluster_bufGPU, 2 * m * sizeof(ClusterEdge)));

        ClusterEdge* from_cluster_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&from_cluster_bufGPU, 2 * m * sizeof(ClusterEdge)));

        int* min_edges_stack_bufGPU = NULL;
        CHECK(cudaMalloc((void**)&min_edges_stack_bufGPU, 2 * m * sizeof(int)));

        int* cluster_leader_sizesGPU = NULL;
        CHECK(cudaMalloc((void**)&cluster_leader_sizesGPU, (n + 1) * sizeof(int)));

        int num_clusters = n;

        ClusterEdge* from_cluster_buf = new ClusterEdge[2 * m];

        while (true) {
            compute_cluster_leader_sizes(cluster_leader_sizes, cluster_ids, graph.v_indices, n);

            CHECK(cudaMemset(to_cluster_bufGPU, -1, 2 * m * sizeof(ClusterEdge)));
            CHECK(cudaMemset(from_cluster_bufGPU, -1, 2 * m * sizeof(ClusterEdge)));
            CHECK(cudaMemset(min_edges_bufGPU, -1, 2 * m * sizeof(ClusterEdge)));
            CHECK(cudaMemset(min_edges_stack_bufGPU, -1, 2 * m * sizeof(int)));

            CHECK(cudaMemcpy(cluster_idsGPU, cluster_ids, n * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(cluster_sizesGPU, cluster_sizes, n * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(cluster_leader_sizesGPU, cluster_leader_sizes, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(cluster_membersGPU, cluster_members, (n) * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(prefix_sum_cluster_sizesGPU, prefix_sum_cluster_sizes, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));


            if (k == 100) {
                // speedup_kernel<<<n_block, n_thread>>>(verticesGPU, from_cluster_bufGPU, n, num_vertices_local);
                // CHECK(cudaGetLastError());
            }
            else {
                min_to_cluster_kernel_sparse<<<n_block, n_thread>>>(
                    to_cluster_bufGPU,
                    min_edges_bufGPU,
                    edgesGPU,
                    v_indicesGPU,
                    cluster_idsGPU,
                    n,
                    num_vertices_local
                );
                CHECK(cudaGetLastError());

                CHECK(cudaMemset(min_edges_bufGPU, -1, 2 * m * sizeof(ClusterEdge)));

                min_from_cluster_kernel_sparse<<<n_block, n_thread>>>(
                    to_cluster_bufGPU,
                    from_cluster_bufGPU,
                    min_edges_bufGPU,
                    min_edges_stack_bufGPU,
                    v_indicesGPU,
                    cluster_leader_sizesGPU,
                    cluster_idsGPU,
                    cluster_sizesGPU,
                    prefix_sum_cluster_sizesGPU,
                    cluster_membersGPU,
                    n,
                    num_vertices_local
                );
                CHECK(cudaGetLastError());
            }

            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(from_cluster_buf, from_cluster_bufGPU, 2 * m * sizeof(ClusterEdge), cudaMemcpyDeviceToHost));

            // start timer in C++ way
            auto start_cpu = std::chrono::high_resolution_clock::now();

            // rank 0 merge edges
            std::vector<ClusterEdge> edges_to_add;

            for (int i = 0; i < 2 * m; ++i) {
                if (from_cluster_buf[i].from_v != -1) {
                    edges_to_add.push_back(from_cluster_buf[i]);
                }
            }

            std::sort(edges_to_add.begin(), edges_to_add.end(), [](ClusterEdge a, ClusterEdge b) {
                return a.weight < b.weight;
            });

            std::vector<bool> heaviest_edges(edges_to_add.size());
            std::fill(heaviest_edges.begin(), heaviest_edges.end(), false);
            // init bool arr with size n called encountered_clusters
            bool encountered_clusters[n] = {false};

            for (int i = edges_to_add.size() - 1; i >= 0; --i) {
                ClusterEdge edge = edges_to_add[i];
                int to_cluster = get_cluster_leader_host(cluster_ids, edge.to_v);

                if (!encountered_clusters[to_cluster]) {
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
                    num_clusters--;
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

            k++;

            get_prefix_sum(cluster_sizes, prefix_sum_cluster_sizes, n + 1);
            
            for (int vertex = 0; vertex < n; ++vertex) {
                int cluster_id = get_cluster_leader_host(cluster_ids, vertex);
                cluster_members[prefix_sum_cluster_sizes[cluster_id]] = vertex;
                prefix_sum_cluster_sizes[cluster_id]++;
            }

            get_prefix_sum(cluster_sizes, prefix_sum_cluster_sizes, n + 1);

            // end timer
            auto end_cpu = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

            cpu_time += duration.count();

            if (k >= 10) {
                // throw std::runtime_error("k >= 10");
                return mst_edges;
            }

            if (num_clusters == 1) {
                break;
            }
        }

        CHECK(cudaFree(edgesGPU));
        CHECK(cudaFree(cluster_idsGPU));
        CHECK(cudaFree(to_cluster_bufGPU));
        CHECK(cudaFree(from_cluster_bufGPU));
        CHECK(cudaFree(v_indicesGPU));
        CHECK(cudaFree(min_edges_bufGPU));
        CHECK(cudaFree(min_edges_stack_bufGPU));
        CHECK(cudaFree(cluster_membersGPU));
        CHECK(cudaFree(prefix_sum_cluster_sizesGPU));

        delete[] cluster_ids;
        delete[] cluster_sizes;
        delete[] from_cluster_buf;
        delete[] prefix_sum_cluster_sizes;
        delete[] cluster_members;
        
        // print cpu time
        std::cout << "CPU time: " << cpu_time << std::endl;

        return mst_edges;
    }

    std::vector<int> algo_prim(const float* vertices, const int n) {
        std::vector<int> parent(n, -1);
        std::vector<float> key(n, std::numeric_limits<float>::max());
        std::vector<bool> mstSet(n, false);

        key[0] = 0;
        for (int count = 0; count < n - 1; count++) {
            int u = -1;
            float min_key = std::numeric_limits<float>::max();
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

    std::vector<int> algo_prim_sparse(const SparseGraph graph) {
        int n = graph.n;
        std::vector<int> parent(n, -1);

        SparseGraphEdge placeHolder;
        placeHolder.to_v = -1;
        placeHolder.weight = std::numeric_limits<float>::max();

        std::vector<SparseGraphEdge> key(n, placeHolder);
        std::vector<bool> mstSet(n, false);
        
        SparseGraphEdge initialEdge;
        initialEdge.to_v = 0;
        initialEdge.weight = 0;
        key[0] = initialEdge;
        for (int count = 0; count < n - 1; count++) {
            int u = -1;
            float min_key = std::numeric_limits<float>::max();

            for (int i = 0; i < n; i++) {
                if (!mstSet[i] && key[i].weight < min_key) {
                    u = i;
                    min_key = key[i].weight;
                }
            }

            mstSet[u] = true;

            int v_edge_size = graph.v_indices[u + 1] - graph.v_indices[u];
            for (int v = 0; v < v_edge_size; v++) {
                int u_edge_start = graph.v_indices[u];

                int v_vertex = graph.edges[u_edge_start + v].to_v;
                if (!mstSet[v_vertex] && graph.edges[u_edge_start + v].weight < key[v_vertex].weight) {
                    parent[v_vertex] = u;
                    key[v_vertex] = graph.edges[u_edge_start + v];
                }
            }
        }

        return parent;
    }
}

