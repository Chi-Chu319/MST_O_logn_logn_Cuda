#include "algo.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__device__ int get_cluster_leader(int* cluster_ids, int v) {
    if (v < 0) {
        return v;
    }

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

// removes all the empty slots in the array returns the size of the array after squeezing
__device__ int squeezeArray(ClusterEdge* edges, int n) {
    // points to the first non-empty slot after empty slots
    int ptr1 = 0;
    // points to the first empty slot
    int ptr2 = 0;

    while (ptr1 < n) {
        if (edges[ptr2].from_v != -1) {
            ptr2++;
            if (ptr1 < ptr2) {
                ptr1 = ptr2;
            }
        } else if (edges[ptr1].from_v == -1) {
            ptr1++;
        } else if (ptr2 < n && ptr1 < n) {
            edges[ptr2] = edges[ptr1];
            edges[ptr1] = ClusterEdge();
        }
    }

    return ptr2;
}

__global__ void min_to_cluster_kernel(ClusterEdge* to_cluster_buf, ClusterEdge* min_edges_buf, const float* vertices, int* cluster_ids, const int n, int num_vertices_local) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    int vertex_local_start = (j * blockDim.x + i) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int from_v = vertex_local + vertex_local_start;
        int min_edge_start = from_v * n;
        ClusterEdge* cluster_edges = min_edges_buf + min_edge_start;

        for (int k = 0; k < n; ++k) {
            cluster_edges[from_v * num_vertices_local + k] = ClusterEdge();
        }

        for (int to_v = 0; to_v < n; ++to_v) {
            float weight = vertices[from_v * n + to_v];
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

__global__ void min_from_cluster_kernel(const ClusterEdge* to_cluster_buf, ClusterEdge* from_cluster_buf, ClusterEdge* min_edges_bufGPU, int* min_edges_stack_bufGPU, int* cluster_ids, int* cluster_sizes, const int n, int num_vertices_local) {
    int vertex_local_start = (blockIdx.x * blockDim.x + threadIdx.x) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int vertex = vertex_local + vertex_local_start;
        int min_edge_start = vertex * n;

        ClusterEdge* cluster_edges = min_edges_bufGPU + min_edge_start;
        int* stack = min_edges_stack_bufGPU + min_edge_start;

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

        int squeezedSize = squeezeArray(cluster_edges, n);

        quickSortIterative(stack, cluster_edges, 0, squeezedSize - 1);

        int mu = min(cluster_sizes[vertex], edge_size);

        // For all non empty cluster_edges, update cluster_ids
        for (int k = 0; k < mu; ++k) {
            from_cluster_buf[vertex * n + k] = cluster_edges[k];
        }
    }
}

__global__ void speedup_kernel(const float* vertices, ClusterEdge* from_cluster_buf, const int n, int num_vertices_local) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    int vertex_local_start = (j * blockDim.x + i) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int from_v = vertex_local + vertex_local_start;

        float min_weight = DBL_MAX;
        int min_to_v;

        for (int to_v = 0; to_v < n; ++to_v) {
            if (from_v == to_v) {
                continue;
            }

            float weight = vertices[from_v * n + to_v];
            // update min_weight
            if (weight < min_weight) {
                min_weight = weight;
                min_to_v = to_v;
            }
        }

        from_cluster_buf[from_v * n + min_to_v] = ClusterEdge(from_v, min_to_v, min_weight);
    }
}

__global__ void min_to_cluster_kernel_sparse(ClusterEdge* to_cluster_buf, ClusterEdge* min_edges_buf, SparseGraphEdge* edges, int* v_indices, int* cluster_ids, const int n, int num_vertices_local) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    int vertex_local_start = (j * blockDim.x + i) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int from_v = vertex_local + vertex_local_start;

        ClusterEdge* cluster_edges = min_edges_buf + v_indices[from_v];
        SparseGraphEdge* edges_local = edges + v_indices[from_v];

        for (int k = 0; k < v_indices[from_v + 1] - v_indices[from_v]; ++k) {
            cluster_edges[k] = ClusterEdge();
        }

        int edge_count = 0;

        for (int i = 0; i < v_indices[from_v + 1] - v_indices[from_v]; ++i) {
            SparseGraphEdge edge = edges_local[i];
            float weight = edge.weight;
            int to_v = edge.to_v;

            int from_cluster = get_cluster_leader(cluster_ids, from_v);
            int to_cluster = get_cluster_leader(cluster_ids, to_v);

            if (from_cluster != to_cluster) {
                for (int k = 0; k < edge_count + 1; ++k) {
                    if (to_cluster == get_cluster_leader(cluster_ids, cluster_edges[k].to_v)) {
                        if (weight < cluster_edges[k].weight) {
                            cluster_edges[k] = ClusterEdge(from_v, to_v, weight);
                            edge_count++;
                        } 
                        break;
                    // empty slot
                    } else if (k == edge_count) {
                        cluster_edges[k] = ClusterEdge(from_v, to_v, weight);
                        edge_count++;
                        break;
                    }
                }
            }
        }

        //  loop cluster_edges for non empty item and update cluster_ids
        edge_count = 0;
        for (int k = 0; k < v_indices[from_v + 1] - v_indices[from_v]; ++k) {
            if (cluster_edges[k].from_v != -1) {
                ClusterEdge edge = cluster_edges[k];
                int to_cluster = get_cluster_leader(cluster_ids, edge.to_v);
                to_cluster_buf[v_indices[from_v] + edge_count] = edge;
                edge_count++;
            }
        }
    }
}

__global__ void min_from_cluster_kernel_sparse(
    const ClusterEdge* to_cluster_buf,
    ClusterEdge* from_cluster_buf,
    ClusterEdge* min_edges_bufGPU,
    int* min_edges_stack_bufGPU,
    int* v_indices,
    int* cluster_leader_sizesGPU,
    int* cluster_ids,
    int* cluster_sizes,
    const int n,
    int num_vertices_local
) {
    int vertex_local_start = (blockIdx.x * blockDim.x + threadIdx.x) * num_vertices_local;

    for (int vertex_local = 0; vertex_local < num_vertices_local; ++vertex_local) {
        int to_v = vertex_local + vertex_local_start;

        if (get_cluster_leader(cluster_ids, to_v) != to_v) {
            continue;
        }

        ClusterEdge* cluster_edges = min_edges_bufGPU + cluster_leader_sizesGPU[to_v];
        int* stack = min_edges_stack_bufGPU + cluster_leader_sizesGPU[to_v];

        for (int k = 0; k < cluster_leader_sizesGPU[to_v + 1] - cluster_leader_sizesGPU[to_v]; ++k) {
            cluster_edges[k] = ClusterEdge();
        }

        int edge_count = 0;

        for (int from_v = 0; from_v < n; ++from_v) {
            ClusterEdge edge = ClusterEdge();

            //  the most inefficient part for sparse graph
            for (int i = v_indices[from_v]; i < v_indices[from_v + 1]; ++i) {
                ClusterEdge e = to_cluster_buf[i];
                if (get_cluster_leader(cluster_ids, e.to_v) == to_v) {
                    edge = e;
                    break;
                }
            }

            if (edge.from_v != -1) {
                for (int k = 0; k < edge_count + 1; ++k) {
                    if (get_cluster_leader(cluster_ids, edge.from_v) == get_cluster_leader(cluster_ids, cluster_edges[k].from_v)) {
                        if (edge.weight < cluster_edges[k].weight) {
                            cluster_edges[k] = edge;
                            edge_count++;
                        } 
                        break;
                    // empty slot
                    } else if (k == edge_count) {
                        cluster_edges[k] = edge;
                        edge_count++;
                        break;
                    }
                }
            }
        }

        quickSortIterative(stack, cluster_edges, 0, edge_count - 1);

        int mu = min(cluster_sizes[to_v], edge_count);

        // For all non empty cluster_edges, update cluster_ids
        for (int k = 0; k < mu; ++k) {
            from_cluster_buf[cluster_leader_sizesGPU[to_v] + k] = cluster_edges[k];
        }
    }
}