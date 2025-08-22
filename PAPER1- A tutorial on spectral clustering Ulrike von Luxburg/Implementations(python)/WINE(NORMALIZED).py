# Install dependencies if running for the first time:
    # !pip install ucimlrepo scikit-learn matplotlib numpy scipy

from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

# Fetch and prepare Wine dataset
wine = fetch_ucirepo(id=109)
X = wine.data.features.to_numpy()
y_true = wine.data.targets.to_numpy().ravel()

# Standardize features
X = StandardScaler().fit_transform(X)

def gaussian_similarity(X, gamma=0.1):
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    return np.exp(-gamma * pairwise_sq_dists)

# Similarity matrix
W = gaussian_similarity(X, gamma=0.1)

# Construct symmetric normalized Laplacian
D = np.diag(W.sum(axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
L_sym = np.eye(X.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

# Eigen decomposition (np.linalg.eigh guarantees sorted eigenvalues in ascending order)
eigvals, eigvecs = eigh(L_sym)
U = eigvecs[:, :3]  # Use first 3 eigenvectors

# Ng et al. normalization: row-normalize U
U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

# Cluster using k-means
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(U_normalized)

# Reduce data to 2D for plotting
X_pca = PCA(n_components=2, random_state=42).fit_transform(X)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='plasma')
axes[0].set_title("True Labels (PCA 2D)")
axes[0].set_xlabel("PCA 1")
axes[0].set_ylabel("PCA 2")

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
axes[1].set_title("Ng Spectral Clustering (k=3, γ=0.1)")
axes[1].set_xlabel("PCA 1")
axes[1].set_ylabel("PCA 2")

plt.tight_layout()
plt.savefig('wine_clustering_results.png')
plt.show()  # Keep this for interactive viewing if needed

# Evaluation metrics
def best_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def graph_cut(W, labels):
    cut_value = 0.0
    N = len(W)
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] != labels[j]:
                cut_value += W[i, j]
    return cut_value

print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, y_pred):.4f}")
print(f"Clustering Accuracy: {best_accuracy(y_true, y_pred):.4f}")
print(f"Graph Cut Value: {graph_cut(W, y_pred):.4f}")
