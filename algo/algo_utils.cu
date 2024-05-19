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

__device__ void swap(ClusterEdge* a, ClusterEdge* b) {
    ClusterEdge temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int partition(ClusterEdge* arr, int low, int high) {
    float pivot = arr[high].weight;
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j].weight <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

__device__ void quickSortIterative(int* stack, ClusterEdge* arr, int l, int h) {
    int top = -1;

    stack[++top] = l;
    stack[++top] = h;

    while (top >= 0) {
        // Pop h and l
        h = stack[top--];
        l = stack[top--];

        int p = partition(arr, l, h);

        if (p - 1 > l) {
            stack[++top] = l;
            stack[++top] = p - 1;
        }

        if (p + 1 < h) {
            stack[++top] = p + 1;
            stack[++top] = h;
        }
    }
}

__global__ void min_to_cluster_kernel(ClusterEdge* to_cluster_buf, const double* vertices, int* cluster_ids, const int n, int num_vertices_local) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    int vertex_local_start = (j * blockDim.x + i) * num_vertices_local;

    // TODO change n
    ClusterEdge cluster_edges[8192];

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int from_v = vertex_local + vertex_local_start;

        for (int k = 0; k < n; ++k) {
            cluster_edges[k] = ClusterEdge();
        }

        for (int to_v = 0; to_v < n; ++to_v) {
            double weight = vertices[from_v * n + to_v];
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
                to_cluster_buf[edge.from_v * n + to_cluster] = edge;
            }
        }
    }
}

__global__ void min_from_cluster_kernel(const ClusterEdge* to_cluster_buf, ClusterEdge* from_cluster_buf, int* cluster_ids, int* cluster_sizes, const int n, int num_vertices_local) {
    int vertex_local_start = (blockIdx.x * blockDim.x + threadIdx.x) * num_vertices_local;

    // TODO change n
    ClusterEdge cluster_edges[8192];
    int stack[8192];

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int vertex = vertex_local + vertex_local_start;
        
        if (get_cluster_leader(cluster_ids, vertex) != vertex) {
            continue;
        }

        for (int k = 0; k < n; ++k) {
            cluster_edges[k] = ClusterEdge();
        }

        for (int from_v = 0; from_v < n; ++from_v) {
            ClusterEdge edge = to_cluster_buf[from_v * n + vertex];

            if (edge.from_v != -1) {
                int from_cluster = get_cluster_leader(cluster_ids, edge.from_v);
                if (cluster_edges[from_cluster].from_v == -1) {
                    cluster_edges[from_cluster] = edge;
                } else if (edge.weight < cluster_edges[from_cluster].weight) {
                    cluster_edges[from_cluster] = edge;
                }
            }
        }

        int edge_size = 0;
        for (int k = 0; k < n; ++k) {
            if (cluster_edges[k].from_v != -1) {
                edge_size++;
            }
        }

        quickSortIterative(stack, cluster_edges, 0, n - 1);

        int mu = min(cluster_sizes[vertex], edge_size);

        // For all non empty cluster_edges, update cluster_ids
        for (int k = 0; k < mu; ++k) {
            from_cluster_buf[vertex * n + k] = cluster_edges[k];
        }
    }
}

__global__ void speedup_kernel(const double* vertices, ClusterEdge* from_cluster_buf, const int n, int num_vertices_local) {
    int i = threadIdx.x;
    int j = blockIdx.x; 
    
    int vertex_local_start = (j * blockDim.x + i) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int from_v = vertex_local + vertex_local_start;

        double min_weight = DBL_MAX;
        int min_to_v;

        for (int to_v = 0; to_v < n; ++to_v) {
            if (from_v == to_v) {
                continue;
            }

            double weight = vertices[from_v * n + to_v];
            // update min_weight
            if (weight < min_weight) {
                min_weight = weight;
                min_to_v = to_v;
            } 
        }

        from_cluster_buf[from_v * n + min_to_v] = ClusterEdge(from_v, min_to_v, min_weight);
    }
}