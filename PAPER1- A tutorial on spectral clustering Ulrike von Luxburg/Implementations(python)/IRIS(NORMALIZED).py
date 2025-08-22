import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import networkx as nx


iris = datasets.load_iris()
X = iris.data
y_true = iris.target
n = X.shape[0]
k = 3

# Gaussian similarity matrix W
def gaussian_similarity(X, sigma=1.0):
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    W = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    return W

W = gaussian_similarity(X, sigma=1.0)

# Degree matrix D and symmetric normalized Laplacian
D = np.diag(W.sum(axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

# first k eigenvectors of L_sym
eigvals, eigvecs = np.linalg.eigh(L_sym)
U = eigvecs[:, :k]  # first k eigenvectors

# Normalize each row to unit length put that in matrix T
T = normalize(U, norm='l2', axis=1)

# k-means to rows of T
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
labels_pred = kmeans.fit_predict(T)

# Align predicted labels with true labels
def match_labels(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    label_map = dict(zip(col_ind, row_ind))
    return np.array([label_map[label] for label in y_pred])

labels_aligned = match_labels(y_true, labels_pred)

# Evaluation
ari = adjusted_rand_score(y_true, labels_pred)
acc = accuracy_score(y_true, labels_aligned)

print(" Ng, Jordan & Weiss Normalized Spectral Clustering")
print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Clustering Accuracy: {acc:.3f}")

# cut value
def compute_cut(W, labels):
    cut_value = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] != labels[j]:
                cut_value += W[i, j]
    return cut_value / 2

cut_val = compute_cut(W, labels_pred)
print(f"Graph Cut Value: {cut_val:.3f}")


# GRAPH PLOT (Thresholded W)
W_thresh = np.copy(W)
threshold = np.percentile(W, 95)
W_thresh[W < threshold] = 0
G = nx.from_numpy_array(W_thresh)
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=labels_pred, cmap='viridis', node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.2)
plt.title("Ng et al. Normalized Spectral Clustering (Graph View)")
plt.axis('off')
plt.savefig('iris_normalized_graph_view.png')
plt.show()  # Keep this for interactive viewing if needed

# SCATTER PLOT
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title("True Iris Labels")
plt.xlabel("Sepal Length"); plt.ylabel("Sepal Width")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='viridis')
plt.title("Ng et al. Spectral Clustering")
plt.xlabel("Sepal Length"); plt.ylabel("Sepal Width")
plt.tight_layout()
plt.savefig('iris_normalized_scatter_plot.png')
plt.show()  # Keep this for interactive viewing if needed
