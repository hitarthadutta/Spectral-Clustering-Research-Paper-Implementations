import numpy as np
from scipy.linalg import eig
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import rand_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import random
import warnings
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# --- Algorithm 1: CSP

def constrained_spectral_clustering(X, Q, beta, n_clusters=3):
    n = X.shape[0]
    A = rbf_kernel(X, gamma=0.5)
    np.fill_diagonal(A, 0)

    D_diag = np.sum(A, axis=1)
    volG = np.sum(D_diag)

    D_inv_sqrt_diag = 1.0 / np.sqrt(D_diag + 1e-9)
    D_inv_sqrt = np.diag(D_inv_sqrt_diag)

    L_bar = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    Q_bar = D_inv_sqrt @ Q @ D_inv_sqrt

    M = Q_bar - (beta / volG) * np.eye(n)

    try:
        eigvals, eigvecs = eig(L_bar, M)
        eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
    except np.linalg.LinAlgError:
        warnings.warn("Eigendecomposition failed. Falling back to unconstrained clustering.")
        sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42)
        return sc.fit_predict(X)

    positive_indices = np.where(eigvals > 1e-9)[0]

    if len(positive_indices) < n_clusters:
        warnings.warn(f"Found only {len(positive_indices)} positive eigenvalues (need {n_clusters}). Falling back to unconstrained clustering.")
        sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42)
        return sc.fit_predict(X)

    feasible_vectors = eigvecs[:, positive_indices]

    costs = np.array([vec.T @ L_bar @ vec for vec in feasible_vectors.T])

    best_k_indices = np.argsort(costs)[:n_clusters]
    embedding = feasible_vectors[:, best_k_indices]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embedding)

# Algorithm 2: ModAff (Modify Affinity)

def modaff_spectral_clustering(X, must_links, cannot_links, n_clusters=3):
    A = rbf_kernel(X, gamma=0.5)
    for i, j in must_links: A[i, j] = A[j, i] = 1.0
    for i, j in cannot_links: A[i, j] = A[j, i] = 0.0
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    return sc.fit_predict(A)

# Algorithm 3: GrBias (Grouping Bias)

def grbias_spectral_clustering(X, must_links, n_clusters=3):
    A = rbf_kernel(X, gamma=0.5)
    for i, j in must_links: A[i, j] = A[j, i] = 2.0
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    return sc.fit_predict(A)

# Experiment Setup and Execution

def run_all_experiments(X, y_true, p, n_clusters=3):
    n = len(y_true)
    known_count = int(p * n)

    if known_count < n_clusters:
        sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=0.5, random_state=42)
        labels = sc.fit_predict(X)
        ri = rand_score(y_true, labels)
        return ri, ri, ri

    known_indices = random.sample(range(n), known_count)

    must_links, cannot_links = [], []
    Q = np.zeros((n, n))
    for i in range(len(known_indices)):
        for j in range(i + 1, len(known_indices)):
            idx1, idx2 = known_indices[i], known_indices[j]
            if y_true[idx1] == y_true[idx2]:
                must_links.append((idx1, idx2))
                Q[idx1, idx2] = Q[idx2, idx1] = 1
            else:
                cannot_links.append((idx1, idx2))
                Q[idx1, idx2] = Q[idx2, idx1] = -1

    beta_csp = 0.5

    labels_csp = constrained_spectral_clustering(X, Q, beta_csp, n_clusters=n_clusters)
    ri_csp = rand_score(y_true, labels_csp)

    labels_modaff = modaff_spectral_clustering(X, must_links, cannot_links, n_clusters=n_clusters)
    ri_modaff = rand_score(y_true, labels_modaff)

    labels_grbias = grbias_spectral_clustering(X, must_links, n_clusters=n_clusters)
    ri_grbias = rand_score(y_true, labels_grbias)

    return ri_csp, ri_modaff, ri_grbias

# execution block

if __name__ == "__main__":

    wine = load_wine()
    X, y_true = wine.data, wine.target
    n_clusters = len(np.unique(y_true))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    percentages = np.linspace(0, 1, 11)
    num_trials = 20

    results = {'csp': [], 'modaff': [], 'grbias': []}

    print("Running final, corrected comparison of all methods on the Wine dataset...")
    for p in percentages:
        trial_results = {'csp': [], 'modaff': [], 'grbias': []}
        for _ in range(num_trials):
            try:
                ri_csp, ri_modaff, ri_grbias = run_all_experiments(X_scaled, y_true, p, n_clusters)
                trial_results['csp'].append(ri_csp)
                trial_results['modaff'].append(ri_modaff)
                trial_results['grbias'].append(ri_grbias)
            except Exception as e:
                warnings.warn(f"A trial failed at {p*100}%: {e}")
                continue

        for method in results.keys():
            if trial_results[method]:
                results[method].append((np.mean(trial_results[method]), np.std(trial_results[method])))
            else:
                results[method].append((np.nan, np.nan))

        print(f"Known Labels: {p*100:.0f}%\t-> CSP: {results['csp'][-1][0]:.3f}, ModAff: {results['modaff'][-1][0]:.3f}, GrBias: {results['grbias'][-1][0]:.3f}")

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    mean_csp, std_csp = np.array(results['csp']).T
    mean_modaff, std_modaff = np.array(results['modaff']).T
    mean_grbias, std_grbias = np.array(results['grbias']).T
    x_axis = percentages * 100

    plt.plot(x_axis, mean_csp, marker='o', linestyle='-', label='CSP (Proposed)', zorder=3, color='#1f77b4', linewidth=2)
    plt.fill_between(x_axis, mean_csp - std_csp, mean_csp + std_csp, alpha=0.2, color='#1f77b4')

    plt.plot(x_axis, mean_modaff, marker='x', linestyle='--', color='red', label='ModAff')
    plt.fill_between(x_axis, mean_modaff - std_modaff, mean_modaff + std_modaff, alpha=0.2, color='red')

    plt.plot(x_axis, mean_grbias, marker='s', linestyle=':', color='green', label='GrBias')
    plt.fill_between(x_axis, mean_grbias - std_grbias, mean_grbias + std_grbias, alpha=0.2, color='green')

    plt.title('Comparison of Constrained Spectral Clustering Methods on Wine Dataset', fontsize=16)
    plt.xlabel('Percentage of Known Labels (%)', fontsize=12)
    plt.ylabel('Rand Index', fontsize=12)
    plt.xticks(x_axis)
    plt.ylim(0.7, 1.02)
    plt.legend(fontsize=12)
    plt.show()
