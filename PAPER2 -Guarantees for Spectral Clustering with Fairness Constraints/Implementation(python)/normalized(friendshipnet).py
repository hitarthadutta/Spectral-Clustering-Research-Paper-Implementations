import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.linalg import null_space, eigh, inv, sqrtm
import matplotlib.pyplot as plt
import os
import sys

# Define paths to the data files
META_BASENAME = 'metadatareal.csv'  # Changed to CSV format
EDGES_BASENAME = 'Friendship-network_data_2013.csv'

# Look for data files in the current directory and parent directories
data_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Looking for data files in: {data_dir}")

# Check if the data files exist
meta_exists = os.path.isfile(os.path.join(data_dir, META_BASENAME))
edges_exists = os.path.isfile(os.path.join(data_dir, EDGES_BASENAME))

print(f"Metadata file: {meta_exists}")
print(f"Edges file: {edges_exists}")


def load_and_preprocess_data(meta_filename, edges_filename):

    meta_df = pd.read_csv(meta_filename, header=None, names=['id', 'class', 'gender'])
    meta_df['id'] = meta_df['id'].astype(str)

    # P:Remove 'Unknown' genders
    meta_df = meta_df[meta_df['gender'] != 'Unknown'].copy()

    with open(edges_filename, 'r') as f:
        edges = [line.strip().split() for line in f.readlines() if line.strip()]

    # Create initial graph
    G_initial = nx.Graph()
    G_initial.add_edges_from(edges)

    # p:Filter graph to include only nodes for which we have metadata
    nodes_with_meta = set(meta_df['id'])
    G_filtered = G_initial.subgraph(nodes_with_meta).copy()

    # p:Remove isolated vertices
    G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))

    # p:Keep only the largest connected component
    if not nx.is_connected(G_filtered):
        largest_cc_nodes = max(nx.connected_components(G_filtered), key=len)
        G_final = G_filtered.subgraph(largest_cc_nodes).copy()
    else:
        G_final = G_filtered

    print(f"\nPreprocessing complete. Analysis will run on the largest connected component, which has {G_final.number_of_nodes()} vertices.")

    gender_map = {node_id: gender for node_id, gender in zip(meta_df['id'], meta_df['gender']) if node_id in G_final.nodes()}

    return G_final, gender_map

def compute_balance(labels, node_list, gender_map):
    num_clusters = np.max(labels) + 1
    cluster_balances = []
    for i in range(num_clusters):
        cluster_nodes_idx = np.where(labels == i)[0]
        if len(cluster_nodes_idx) == 0: continue
        cluster_nodes = [node_list[j] for j in cluster_nodes_idx]
        num_males = sum(1 for node in cluster_nodes if gender_map.get(node) == 'M')
        num_females = sum(1 for node in cluster_nodes if gender_map.get(node) == 'F')
        if num_males == 0 or num_females == 0:
            cluster_balances.append(0)
        else:
            balance = min(num_males / num_females, num_females / num_males)
            cluster_balances.append(balance)
    return np.mean(cluster_balances) if cluster_balances else 0

def compute_ncut(W, D, labels):

    num_clusters = np.max(labels) + 1
    total_ncut = 0
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0: continue

        other_indices = np.where(labels != i)[0]
        cut_value = W[np.ix_(cluster_indices, other_indices)].sum()

        # Volume of the cluster is the sum of degrees of its nodes
        vol_cluster = D[cluster_indices, cluster_indices].sum()
        if vol_cluster > 0:
            total_ncut += cut_value / vol_cluster

    return total_ncut

def run_normalized_analysis(G, gender_map, n_iterations=100):

    node_list = list(G.nodes())
    n = G.number_of_nodes()
    W = nx.to_numpy_array(G, nodelist=node_list)
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # --- Standard Normalized SC ---
    # Compute D^(-1/2) L D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    eigvals_norm, eigvecs_norm = eigh(L_sym)

    # --- Fair Normalized SC (Algorithm 3) ---
    group_f = np.array([1 if gender_map.get(node) == 'F' else 0 for node in node_list])
    prop_f = np.sum(group_f) / n
    f_vector = group_f - prop_f
    F = f_vector.reshape(-1, 1)
    Z = null_space(F.T)

    Q_squared = Z.T @ D @ Z
    # Use real part to handle potential small imaginary parts from numerical errors
    Q = sqrtm(Q_squared).real
    Q_inv = inv(Q)

    L_fair_sym = Q_inv @ Z.T @ L @ Z @ Q_inv
    eigvals_fair_norm, eigvecs_fair_norm = eigh(L_fair_sym)

    results = {key: [] for key in ['k', 'balance_sc', 'ncut_sc', 'balance_fair_sc', 'ncut_fair_sc']}
    print(f"Running NORMALIZED analysis ({n_iterations} iterations per k)...")
    for k in range(2, 16):
        # Standard normalized embedding
        H_sc = eigvecs_norm[:, :k]
        H_sc_norm = H_sc / (np.linalg.norm(H_sc, axis=1, keepdims=True) + 1e-9)

        # Fair normalized embedding
        X = eigvecs_fair_norm[:, :k]
        H_fair = Z @ Q_inv @ X
        H_fair_norm = H_fair / (np.linalg.norm(H_fair, axis=1, keepdims=True) + 1e-9)

        iter_metrics = {key: [] for key in results if key != 'k'}
        for _ in range(n_iterations):
            kmeans_sc = KMeans(n_clusters=k, random_state=None, n_init=1).fit(H_sc_norm)
            kmeans_fair = KMeans(n_clusters=k, random_state=None, n_init=1).fit(H_fair_norm)

            iter_metrics['balance_sc'].append(compute_balance(kmeans_sc.labels_, node_list, gender_map))
            iter_metrics['ncut_sc'].append(compute_ncut(W, D, kmeans_sc.labels_))
            iter_metrics['balance_fair_sc'].append(compute_balance(kmeans_fair.labels_, node_list, gender_map))
            iter_metrics['ncut_fair_sc'].append(compute_ncut(W, D, kmeans_fair.labels_))

        results['k'].append(k)
        for key in iter_metrics: results[key].append(np.mean(iter_metrics[key]))

    return pd.DataFrame(results), prop_f

def print_and_plot_normalized(results, prop_f):
    """Prints the final table and generates the plot for normalized SC."""
    print("\n--- Computed Averaged Results (Normalized SC) ---")
    print("k  | Plain Balance | Fair Balance")
    print("---|---------------|-------------")
    for i, row in results.iterrows():
        print(f"{int(row['k']):<2} | {row['balance_sc']:<13.3f} | {row['balance_fair_sc']:<11.3f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('k', fontsize=14)
    ax1.set_ylabel('Balance', color='blue', fontsize=14)
    ax1.plot(results['k'], results['balance_sc'], 'b-', marker='o', markersize=4, label='Normalized SC')
    ax1.plot(results['k'], results['balance_fair_sc'], 'b--', marker='x', markersize=5, label='Fair Norm. SC (Alg. 3)')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0.2, 0.8)

    balance_of_dataset = min(prop_f / (1 - prop_f), (1 - prop_f) / prop_f)
    ax1.axhline(y=balance_of_dataset, color='green', linestyle=':', linewidth=3, label=f'Balance of data set (~{balance_of_dataset:.2f})')

    ax2 = ax1.twinx()
    ax2.set_ylabel('NCut', color='red', fontsize=14)
    ax2.plot(results['k'], results['ncut_sc'], 'r-', marker='o', markersize=4)
    ax2.plot(results['k'], results['ncut_fair_sc'], 'r--', marker='x', markersize=5)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 4)

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right', fontsize=12)
    plt.title('FriendshipNet --- Normalized SC (on Your Preprocessed Data)', fontsize=16)
    fig.tight_layout()
    plt.savefig('friendshipnet_normalized_results.png')
    plt.show()

try:
    if meta_exists and edges_exists:
        meta_path = os.path.join(data_dir, META_BASENAME)
        edges_path = os.path.join(data_dir, EDGES_BASENAME)
        final_graph, final_gender_map = load_and_preprocess_data(meta_path, edges_path)
        computed_results, computed_prop_f = run_normalized_analysis(final_graph, final_gender_map, n_iterations=100)
        print_and_plot_normalized(computed_results, computed_prop_f)
    else:
        print(f"Error: Could not find one or both official data files. Please ensure you have '{META_BASENAME}' and '{EDGES_BASENAME}' in the directory.")
        sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

