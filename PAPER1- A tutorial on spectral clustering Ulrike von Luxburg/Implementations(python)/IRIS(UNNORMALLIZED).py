import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

iris = datasets.load_iris()
X = iris.data
y_true = iris.target
n = X.shape[0]
k = 3

def gaussian_similarity(X, sigma=1.0):
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    W = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    np.fill_diagonal(W, 0)  # no self loops
    return W

W = gaussian_similarity(X, sigma=1.0)

D = np.diag(W.sum(axis=1))
L = D - W

eigvals, eigvecs = np.linalg.eigh(L)
U = eigvecs[:, :k]

kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
labels_pred = kmeans.fit_predict(U)

def best_map(true_labels, pred_labels):
    from sklearn.utils.linear_assignment_ import linear_assignment  # for older sklearn
    D = max(true_labels.max(), pred_labels.max()) + 1
    cost = np.zeros((D, D), dtype=int)
    for i in range(len(true_labels)):
        cost[true_labels[i], pred_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)  # maximize matches
    new_pred = np.zeros_like(pred_labels)
    for i in range(D):
        new_pred[pred_labels == col_ind[i]] = row_ind[i]
    return new_pred

row_ind, col_ind = linear_sum_assignment(-np.histogram2d(y_true, labels_pred, bins=(3, 3))[0])
label_map = dict(zip(col_ind, row_ind))
labels_aligned = np.array([label_map[label] for label in labels_pred])

ari = adjusted_rand_score(y_true, labels_pred)
acc = accuracy_score(y_true, labels_aligned)

print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Clustering Accuracy: {acc:.3f}")

def compute_cut(W, labels):
    cut_value = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] != labels[j]:
                cut_value += W[i, j]
    return cut_value / 2

cut_val = compute_cut(W, labels_pred)
print(f"Graph Cut Value: {cut_val:.3f}")

import networkx as nx
import matplotlib.pyplot as plt
W_thresh = np.copy(W)
threshold = np.percentile(W, 95)
W_thresh[W < threshold] = 0
G = nx.from_numpy_array(W_thresh)
colors = [labels_pred[i] for i in range(n)]

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)  # force-directed layout
nx.draw_networkx_nodes(G, pos, node_color=colors, cmap='viridis', node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.2)
plt.title("Unnormalized Spectral Clustering on Iris Dataset (Graph View)")
plt.axis('off')
plt.savefig('iris_unnormalized_graph_view.png')
plt.show()  # Keep this for interactive viewing if needed