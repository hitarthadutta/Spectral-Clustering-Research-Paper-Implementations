import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import rand_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import random
import warnings
import matplotlib.pyplot as plt

# Set the CSP (Constrained Spectral Clustering) hyperparameter
alpha_csp = 0.5

random.seed(42)
np.random.seed(42)

def constrained_spectral_clustering(X, Q, alpha, n_clusters=2):
    n = X.shape[0]
    A = rbf_kernel(X, gamma=0.5)
    np.fill_diagonal(A, 0)
    D_diag = np.sum(A, axis=1)
    D_inv_sqrt_diag = 1.0 / np.sqrt(D_diag + 1e-9)
    D_inv_sqrt = np.diag(D_inv_sqrt_diag)
    L_bar = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    Q_bar = D_inv_sqrt @ Q @ D_inv_sqrt

    trace_L = np.trace(L_bar)
    trace_Q = np.trace(np.abs(Q_bar))
    if trace_Q > 1e-9:
        Q_bar_scaled = Q_bar * (trace_L / trace_Q)
    else:
        Q_bar_scaled = Q_bar

    L_mod = L_bar - alpha * Q_bar_scaled

    try:
        eigvals, eigvecs = eigh(L_mod)
    except np.linalg.LinAlgError:
        warnings.warn("Standard eigendecomposition failed. Falling back to unconstrained clustering.")
        sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42)
        return sc.fit_predict(X)

    indices = np.argsort(eigvals)[:n_clusters]
    embedding = eigvecs[:, indices]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embedding)

def run_sc_on_possibly_disconnected_graph(X, y_true, A_mod, n_clusters=2):
    """
    Spectral Clustering on an affinity matrix that might be disconnected.
    Finds the largest connected component, clusters it, and calculates Rand Index.
    """
    n_components, component_labels = connected_components(A_mod, directed=False)
    if n_components > 1:
        unique_labels, counts = np.unique(component_labels, return_counts=True)
        largest_comp_label = unique_labels[np.argmax(counts)]
        indices_to_keep = np.where(component_labels == largest_comp_label)[0]
        if len(indices_to_keep) < n_clusters:
            return rand_score(y_true, np.zeros_like(y_true))
        X_conn = X[indices_to_keep]
        A_conn = A_mod[np.ix_(indices_to_keep, indices_to_keep)]
        y_conn = y_true[indices_to_keep]
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        pred_labels_conn = sc.fit_predict(A_conn)
        return rand_score(y_conn, pred_labels_conn)
    else:
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        pred_labels = sc.fit_predict(A_mod)
        return rand_score(y_true, pred_labels)

def run_all_experiments(X, y_true, p, n_clusters=2):
    n = len(y_true)
    known_count = int(p * n)
    if known_count < n_clusters:
        A = rbf_kernel(X, gamma=0.5)
        ri = run_sc_on_possibly_disconnected_graph(X, y_true, A, n_clusters)
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

    pred_csp = constrained_spectral_clustering(X, Q, alpha_csp, n_clusters=n_clusters)
    ri_csp = rand_score(y_true, pred_csp)

    # --- Run ModAff ---
    A_modaff = rbf_kernel(X, gamma=0.5)
    for i, j in must_links: A_modaff[i, j] = A_modaff[j, i] = 1.0
    for i, j in cannot_links: A_modaff[i, j] = A_modaff[j, i] = 0.0
    ri_modaff = run_sc_on_possibly_disconnected_graph(X, y_true, A_modaff, n_clusters)

    # --- Run GrBias ---
    A_grbias = rbf_kernel(X, gamma=0.5)
    for i, j in must_links: A_grbias[i, j] = A_grbias[j, i] = 2.0
    ri_grbias = run_sc_on_possibly_disconnected_graph(X, y_true, A_grbias, n_clusters)

    return ri_csp, ri_modaff, ri_grbias

if __name__ == "__main__":
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        df = pd.read_csv(url, header=None)
    except Exception as e:
        print(f"Could not download dataset. Error: {e}")
        exit()

    X_raw, y_str = df.iloc[:, :-1].values, df.iloc[:, -1].values
    y_true = LabelEncoder().fit_transform(y_str)
    X_scaled = StandardScaler().fit_transform(X_raw)
    n_clusters = len(np.unique(y_true))

    percentages = np.linspace(0, 1, 11)
    num_trials = 20

    results = {'csp': [], 'modaff': [], 'grbias': []}

    print("\nRunning final, definitive comparison on the Ionosphere dataset...")
    for p in percentages:
        trial_results = {'csp': [], 'modaff': [], 'grbias': []}
        for _ in range(num_trials):
            try:
                ri_csp, ri_modaff, ri_grbias = run_all_experiments(X_scaled, y_true, p, n_clusters)
                trial_results['csp'].append(ri_csp)
                trial_results['modaff'].append(ri_modaff)
                trial_results['grbias'].append(ri_grbias)
            except Exception as e:
                warnings.warn(f"A trial at {p*100:.1f}% failed: {e}")
                continue

        for method in results.keys():
            mean_ri = np.mean(trial_results[method]) if trial_results[method] else np.nan
            std_ri = np.std(trial_results[method]) if trial_results[method] else np.nan
            results[method].append((mean_ri, std_ri))

        print(f"Known Labels: {p*100:.0f}%\t-> CSP: {results['csp'][-1][0]:.3f}, ModAff: {results['modaff'][-1]:.3f}, GrBias: {results['grbias'][-1]:.3f}")

    # Plot
    plt.style.use('seaborn-whitegrid')
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

    plt.title('Comparison of Constrained Spectral Clustering Methods on Ionosphere Dataset', fontsize=16)
    plt.xlabel('Percentage of Known Labels (%)', fontsize=12)
    plt.ylabel('Rand Index', fontsize=12)
    plt.xticks(x_axis)
    plt.ylim(0.4, 1.02)
    plt.legend(fontsize=12)
    plt.savefig('ionosphere_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'ionosphere_results.png'")
    plt.show()
