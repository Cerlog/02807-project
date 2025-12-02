import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from typing import Dict, List, Any, Union

def betweenness_centrality_normalized(graph: Dict[Any, List[Any]]) -> Dict[Any, float]:
    """
    Calculate the normalized betweenness centrality for an unweighted, undirected graph.

    Args:
        graph: Adjacency dictionary where keys are nodes and values are lists of neighbors.

    Returns:
        Dictionary mapping nodes to their normalized betweenness centrality scores.
    """
    betweenness = dict.fromkeys(graph.keys(), 0.0)
    n = len(graph)

    for s in graph:
        stack = []
        predecessors = {v: [] for v in graph}
        sigma = dict.fromkeys(graph, 0)
        sigma[s] = 1
        distance = dict.fromkeys(graph, -1)
        distance[s] = 0

        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph[v]:
                if distance[w] < 0:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta = dict.fromkeys(graph, 0.0)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # compensate for undirected double-counting
    for v in betweenness:
        betweenness[v] /= 2.0

    # normalize like NetworkX for undirected graphs
    if n > 2:
        normalization_factor = 1 / ((n - 1) * (n - 2) / 2)
        for v in betweenness:
            betweenness[v] *= normalization_factor

    return betweenness


def spectral_clustering(graph: Dict[Any, List[Any]], k: int) -> Dict[Any, int]:
    """
    Perform spectral clustering on an unweighted, undirected graph.

    Args:
        graph: Adjacency dictionary where keys are nodes and values are lists of neighbors.
        k: Number of clusters.

    Returns:
        Dictionary mapping nodes to cluster labels.
    """
    nodes = list(graph.keys())
    n = len(nodes)
   
    # Adjacency matrix
    A = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for v in graph[u]:
            if v in nodes: # Ensure neighbor is in the graph
                j = nodes.index(v)
                A[i, j] = 1
   
    # Degree matrix
    D = np.diag(A.sum(axis=1))
   
    # Laplacian
    L = D - A
   
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
   
    # Take first k eigenvectors (smallest eigenvalues)
    # Note: The first eigenvector corresponds to eigenvalue 0 for connected components
    U = eigvecs[:, :k]

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42).fit(U)
    labels = kmeans.labels_
   
    return dict(zip(nodes, labels))
 