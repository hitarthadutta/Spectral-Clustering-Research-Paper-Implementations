#!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csgraph
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import networkx as nx


wine = fetch_ucirepo(id=109)
X = wine.data.features.values
y_true = wine.data.targets.values.ravel()

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Create sparse affinity matrix using k-nearest neighbors
knn_graph = kneighbors_graph(X_scaled, n_neighbors=10, include_self=False)
W = 0.5 * (knn_graph.toarray() + knn_graph.toarray().T)  # make symmetric


D = np.diag(W.sum(axis=1))
L = D - W  # unnormalized


eigvals, eigvecs = eigh(L)
k = 3
X_spec = eigvecs[:, :k]

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42).fit(X_spec)
y_pred = kmeans.labels_

# evaluation
def clustering_accuracy(y_true, y_pred):
    D = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-D)
    return D[row_ind, col_ind].sum() / y_true.size

ari = adjusted_rand_score(y_true, y_pred)
acc = clustering_accuracy(y_true, y_pred)

# Graph cut value
cut_val = 0
for i in range(len(X)):
    for j in range(i + 1, len(X)):
        if y_pred[i] != y_pred[j]:
            cut_val += W[i, j]

print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Clustering Accuracy: {acc:.3f}")
print(f"Graph Cut Value: {cut_val:.3f}")

# PCA Plots
X_pca = PCA(n_components=2).fit_transform(X_scaled)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='plasma')
ax[0].set_title("True Labels (PCA 2D)")
ax[0].set_xlabel("PCA 1")
ax[0].set_ylabel("PCA 2")

ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
ax[1].set_title("Unnormalized Spectral Clustering (PCA 2D)")
ax[1].set_xlabel("PCA 1")
ax[1].set_ylabel("PCA 2")

plt.tight_layout()
plt.savefig('wine_unnormalized_clustering_results.png')
plt.show()  # Keep this for interactive viewing if needed

# Network graph
G = nx.from_numpy_array(W)
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=y_pred, cmap='viridis', node_size=40)
nx.draw_networkx_edges(G, pos, alpha=0.05)
plt.title("Graph View - Unnormalized Spectral Clustering (KNN Graph)")
plt.axis('off')
plt.savefig('wine_unnormalized_graph_view.png')
plt.show()  # Keep this for interactive viewing if needed