#include <iostream>
#include "algo.cuh"
#include <vector>
#include <queue>

float* generate_clique_graph(int n) {
    srand(time(0)); 

    float* vertices = (float*)malloc(n * n * sizeof(float)); 

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            vertices[i * n + j] = (float)rand() / RAND_MAX;
            vertices[j * n + i] = vertices[i * n + j];
        }
    }

    for (int i = 0; i < n; ++i) {
        vertices[i * n + i] = 200;
    }

    return vertices;
}

SparseGraph generate_sparse_graph(int n, int m) {
    SparseGraphBuilder builder(n);

    // Construct the tree
    std::vector<int> prufer(n - 2);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, n-1);

    for (int i = 0; i < n - 2; i++)  {
        prufer[i] = distr(gen);
    }

    std::vector<int> vertex_set(n);
 
    for (int i = 0; i < n; i++) {
        vertex_set[i] = 0;
    }
 
    for (int i = 0; i < n - 2; i++) {
        vertex_set[prufer[i]] += 1;
    }
 
    int j = 0;

    for (int i = 0; i < n - 2; i++) {
        for (j = 0; j < n; j++) {
            if (vertex_set[j] == 0) {
                vertex_set[j] = -1;
                builder.addEdge(j, prufer[i], (float)rand() / RAND_MAX);

                vertex_set[prufer[i]]--;
 
                break;
            }
        }
    }
 
    j = 0;
 
    int v_buf;
    // For the last element
    for (int i = 0; i < n; i++) {
        if (vertex_set[i] == 0 && j == 0) {
            v_buf = i;
            j++;
        }
        else if (vertex_set[i] == 0 && j == 1) {
            builder.addEdge(v_buf, i, (float)rand() / RAND_MAX);
        }
    }

    // Add the rest of the edges 
    int edgeCount = 0;
    while (edgeCount < m - (n - 1)) {
        int u = distr(gen);
        int v = distr(gen);

        if (u != v && builder.adjList[u].find(v) == builder.adjList[u].end()) {
            builder.addEdge(u, v, (float)rand() / RAND_MAX);
            edgeCount++;
        }
    }

    return builder.toSparseGraph();
}