import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tqdm import tqdm

print("=== Phase 1: Full 32-Layer Hyperbolic Sweep ===")

# 1. Load the Data
data_path = "./llama3_hyperbolic/saved_features_and_labels.pt"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

data = torch.load(data_path, map_location="cpu")
activations = data["background_layered_activations"] # Shape: (1600, 32, 4096)
labels = data["labels"].numpy()
num_layers = activations.shape[1]

def calc_lorentz_distance_matrix(X):
    time_prods = np.outer(X[:, 0], X[:, 0])
    space_prods = np.dot(X[:, 1:], X[:, 1:].T)
    minkowski = -time_prods + space_prods
    minkowski = np.clip(minkowski, a_max=-1.0, a_min=None)
    dist_mat = np.arccosh(-minkowski)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat

def knn_purity(dist_mat, labels, k=10):
    n = dist_mat.shape[0]
    correct = 0
    for i in range(n):
        neighbors = np.argsort(dist_mat[i])[1:k+1]
        neighbor_labels = labels[neighbors]
        correct += np.sum(neighbor_labels == labels[i]) / k
    return correct / n

def estimate_gromov_delta(dist_mat, num_samples=2000):
    N = dist_mat.shape[0]
    deltas = []
    for _ in range(num_samples):
        p = np.random.choice(N, 4, replace=False)
        x, y, z, w = p[0], p[1], p[2], p[3]
        sums = [
            dist_mat[x, y] + dist_mat[z, w],
            dist_mat[x, z] + dist_mat[y, w],
            dist_mat[x, w] + dist_mat[y, z]
        ]
        sums.sort()
        delta = (sums[2] - sums[1]) / 2.0
        deltas.append(delta)
    return np.mean(deltas)

# We will track 4 metrics across all 32 layers
purity_euc_4096 = []
purity_hyp_4096 = []
purity_euc_64 = []
purity_hyp_64 = []
cosine_similarities = []

scale_factor = 15.0

print(f"{'L':<3} | {'Cos Sim':<7} | {'Euc(4K)':<8} | {'Hyp(4K)':<8} | {'Euc(64)':<8} | {'Hyp(64)':<8} | {'Δ Euc(4K)':<9} | {'Δ Hyp(64)':<9}")
print("-" * 90)

for l in range(num_layers):
    # CRITICAL FIX: Cast to float64 before converting to numpy to prevent float16 overflow!
    X_raw = activations[:, l, :].to(dtype=torch.float64).numpy()
    
    # 1. Cosine Similarity
    avg_cos_sim = np.mean(cosine_similarity(X_raw))
    cosine_similarities.append(avg_cos_sim)
    
    # 2. Raw 4096 Dimensions (No PCA)
    dist_matrix_euc_4096 = pairwise_distances(X_raw, metric='euclidean')
    
    X_scaled_4096 = X_raw * scale_factor
    time_coords_4096 = np.sqrt(1 + np.sum(X_scaled_4096**2, axis=1, keepdims=True))
    X_lorentz_4096 = np.hstack((time_coords_4096, X_scaled_4096))
    dist_matrix_hyp_4096 = calc_lorentz_distance_matrix(X_lorentz_4096)
    
    p_euc_4096 = knn_purity(dist_matrix_euc_4096, labels)
    p_hyp_4096 = knn_purity(dist_matrix_hyp_4096, labels)
    purity_euc_4096.append(p_euc_4096)
    purity_hyp_4096.append(p_hyp_4096)
    
    # 3. PCA 64 Dimensions
    pca = PCA(n_components=64, whiten=False)
    X_pca_64 = pca.fit_transform(X_raw)
    
    dist_matrix_euc_64 = pairwise_distances(X_pca_64, metric='euclidean')
    
    X_scaled_64 = X_pca_64 * scale_factor
    time_coords_64 = np.sqrt(1 + np.sum(X_scaled_64**2, axis=1, keepdims=True))
    X_lorentz_64 = np.hstack((time_coords_64, X_scaled_64))
    dist_matrix_hyp_64 = calc_lorentz_distance_matrix(X_lorentz_64)
    
    p_euc_64 = knn_purity(dist_matrix_euc_64, labels)
    p_hyp_64 = knn_purity(dist_matrix_hyp_64, labels)
    purity_euc_64.append(p_euc_64)
    purity_hyp_64.append(p_hyp_64)
    
    delta_euc_4096 = estimate_gromov_delta(dist_matrix_euc_4096, 2000)
    delta_hyp_64 = estimate_gromov_delta(dist_matrix_hyp_64, 2000)
    
    print(f"{l:<3} | {avg_cos_sim:.4f}  | {p_euc_4096:.4f}   | {p_hyp_4096:.4f}   | {p_euc_64:.4f}   | {p_hyp_64:.4f}   | {delta_euc_4096:.4f}     | {delta_hyp_64:.4f}")

# 4. Create the Line Graph
os.makedirs("visualizations", exist_ok=True)
plt.figure(figsize=(10, 6))

layers = np.arange(num_layers)
plt.plot(layers, purity_euc_4096, label='Euclidean (Raw 4096)', linestyle='--', color='blue')
plt.plot(layers, purity_hyp_4096, label='Hyperbolic (Raw 4096)', linestyle='-', color='cyan')
plt.plot(layers, purity_euc_64, label='Euclidean (PCA 64)', linestyle='--', color='red')
plt.plot(layers, purity_hyp_64, label='Hyperbolic (PCA 64)', linestyle='-', color='orange')

plt.title('K-NN Purity Across All 32 LLaMA-3 Layers')
plt.xlabel('Transformer Layer')
plt.ylabel('K-NN Purity (k=10)')
plt.legend()
plt.grid(True, alpha=0.3)

try:
    plt.savefig("visualizations/layer_purity_plot.png")
    print("\nSaved line graph to visualizations/layer_purity_plot.png")
except Exception as e:
    print(f"\nWarning: Could not save visualization due to error: {e}")
