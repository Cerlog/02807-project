import numpy as np
from sklearn.cluster import KMeans

def betweenness_centrality_normalized(graph):
    betweenness = dict.fromkeys(graph.keys(), 0.0)
    n = len(graph)

    for s in graph:
        stack = []
        predecessors = {v: [] for v in graph}
        sigma = dict.fromkeys(graph, 0)
        sigma[s] = 1
        distance = dict.fromkeys(graph, -1)
        distance[s] = 0

        from collections import deque
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
    normalization_factor = 1 / ((n - 1) * (n - 2) / 2)
    for v in betweenness:
        betweenness[v] *= normalization_factor

    return betweenness


 
def spectral_clustering(graph, k):
    """
    Perform spectral clustering on an unweighted, undirected graph.
    graph: dict where keys are nodes and values are lists of neighbors.
    k: number of clusters.
    Returns: dict of node -> cluster label.
    """
    nodes = list(graph.keys())
    n = len(nodes)
   
    # Adjacency matrix
    A = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for v in graph[u]:
            j = nodes.index(v)
            A[i, j] = 1
   
    # Degree matrix
    D = np.diag(A.sum(axis=1))
   
    # Laplacian
    L = D - A
   
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
   
    # Take first k eigenvectors (smallest eigenvalues)
    U = eigvecs[:, :k]

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42).fit(U)
    labels = kmeans.labels_
   
    return dict(zip(nodes, labels))
 