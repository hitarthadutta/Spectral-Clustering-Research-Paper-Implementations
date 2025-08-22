import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import os

# Define paths to the data files
META_BASENAME = 'metadatareal.csv'  # Changed to CSV format
EDGES_BASENAME = 'Friendship-network_data_2013.csv'

# Check if files exist in the current directory
data_dir = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(data_dir, META_BASENAME)
edges_path = os.path.join(data_dir, EDGES_BASENAME)

print(f"Looking for data files in: {data_dir}")
print(f"Metadata file: {os.path.exists(meta_path)}")
print(f"Edges file: {os.path.exists(edges_path)}")


def load_and_preprocess_data(meta_filename, edges_filename):

    meta_df = pd.read_csv(meta_filename, header=None, names=['id', 'class', 'gender'])
    meta_df['id'] = meta_df['id'].astype(str)

    # Preprocessing: Remove 'Unknown' genders ---
    meta_df = meta_df[meta_df['gender'] != 'Unknown'].copy()

    with open(edges_filename, 'r') as f:
        edges = [line.strip().split() for line in f.readlines() if line.strip()]

    # Create initial graph
    G_initial = nx.Graph()
    G_initial.add_edges_from(edges)

    # filter graph to include only nodes for which we have metadata
    nodes_with_meta = set(meta_df['id'])
    G_filtered = G_initial.subgraph(nodes_with_meta).copy()

    # remove isolated vertices
    G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))

    # keep only the largest connected component
    if not nx.is_connected(G_filtered):
        largest_cc_nodes = max(nx.connected_components(G_filtered), key=len)
        G_final = G_filtered.subgraph(largest_cc_nodes).copy()
    else:
        G_final = G_filtered

    print(f"\nPreprocessing complete. Analysis will run on the largest connected component, which has {G_final.number_of_nodes()} vertices.")

    # create the gender map for the final set of nodes
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

def compute_ratiocut(W, labels):
    num_clusters = np.max(labels) + 1
    total_ratio_cut = 0
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0: continue
        other_indices = np.where(labels != i)[0]
        cut_value = W[np.ix_(cluster_indices, other_indices)].sum()
        total_ratio_cut += cut_value / len(cluster_indices)
    return total_ratio_cut

def run_analysis_and_get_results(G, gender_map, n_iterations=100):

    node_list = list(G.nodes())
    W = nx.to_numpy_array(G, nodelist=node_list)
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    group_f = np.array([1 if gender_map.get(node) == 'F' else 0 for node in node_list])
    prop_f = np.sum(group_f) / len(node_list)
    f_vector = group_f - prop_f
    F = f_vector.reshape(-1, 1)
    Z = null_space(F.T)
    L_fair = Z.T @ L @ Z

    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals_fair, eigvecs_fair = np.linalg.eigh(L_fair)

    results = {key: [] for key in ['k', 'balance_sc', 'ratiocut_sc', 'balance_fair_sc', 'ratiocut_fair_sc']}
    print(f"Running analysis ({n_iterations} iterations per k)...")
    for k in range(2, 16):
        H_sc = eigvecs[:, :k]
        H_sc_norm = H_sc / (np.linalg.norm(H_sc, axis=1, keepdims=True) + 1e-8)
        Y = eigvecs_fair[:, :k]
        H_fair = Z @ Y
        H_fair_norm = H_fair / (np.linalg.norm(H_fair, axis=1, keepdims=True) + 1e-8)

        iter_metrics = {key: [] for key in results if key != 'k'}
        for _ in range(n_iterations):
            kmeans_sc = KMeans(n_clusters=k, random_state=None, n_init=1).fit(H_sc_norm)
            kmeans_fair = KMeans(n_clusters=k, random_state=None, n_init=1).fit(H_fair_norm)
            iter_metrics['balance_sc'].append(compute_balance(kmeans_sc.labels_, node_list, gender_map))
            iter_metrics['ratiocut_sc'].append(compute_ratiocut(W, kmeans_sc.labels_))
            iter_metrics['balance_fair_sc'].append(compute_balance(kmeans_fair.labels_, node_list, gender_map))
            iter_metrics['ratiocut_fair_sc'].append(compute_ratiocut(W, kmeans_fair.labels_))

        results['k'].append(k)
        for key in iter_metrics: results[key].append(np.mean(iter_metrics[key]))

    return pd.DataFrame(results), prop_f

def print_and_plot(results, prop_f):
    """Prints the final table and generates the plot from the computed results."""
    print("\n--- Computed Averaged Results (100 Iterations on Preprocessed Data) ---")
    print("k  | Plain Balance | Fair Balance")
    print("---|---------------|-------------")
    for i, row in results.iterrows():
        print(f"{int(row['k']):<2} | {row['balance_sc']:<13.3f} | {row['balance_fair_sc']:<11.3f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('k', fontsize=14)
    ax1.set_ylabel('Balance', color='blue', fontsize=14)
    ax1.plot(results['k'], results['balance_sc'], 'b-', marker='o', markersize=4, label='SC (Alg. 1)')
    ax1.plot(results['k'], results['balance_fair_sc'], 'b--', marker='x', markersize=5, label='FairSC (Alg. 2)')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0.2, 0.8)

    balance_of_dataset = min(prop_f / (1 - prop_f), (1 - prop_f) / prop_f)
    ax1.axhline(y=balance_of_dataset, color='green', linestyle=':', linewidth=3, label=f'Balance of data set (~{balance_of_dataset:.2f})')

    ax2 = ax1.twinx()
    ax2.set_ylabel('RatioCut', color='red', fontsize=14)
    ax2.plot(results['k'], results['ratiocut_sc'], 'r-', marker='o', markersize=4)
    ax2.plot(results['k'], results['ratiocut_fair_sc'], 'r--', marker='x', markersize=5)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 15)

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right', fontsize=12)
    plt.title('FriendshipNet --- Unnormalized SC (on Your Preprocessed Data)', fontsize=16)
    fig.tight_layout()
    plt.savefig('friendshipnet_unnormalized_results.png')
    plt.show()

try:
    if os.path.exists(meta_path) and os.path.exists(edges_path):
        # Create the final graph and gender map using the pre-processing function
        final_graph, final_gender_map = load_and_preprocess_data(meta_path, edges_path)

        # Run the analysis to get the real computed results
        computed_results, computed_prop_f = run_analysis_and_get_results(final_graph, final_gender_map, n_iterations=100)

        # Generate the final table and plot
        print_and_plot(computed_results, computed_prop_f)
    else:
        print("Error: Could not find one or both official data files. Please ensure you upload 'metadatareal.csv' and 'Friendship-network_data_2013.csv'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

