import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os
from sklearn.metrics import pairwise_distances

print("=== Phase 1: Hyperparameter Sweep (Empirical Validation) ===")

# 1. Load the Data
data_path = "./llama3_hyperbolic/saved_features_and_labels.pt"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

data = torch.load(data_path, map_location="cpu")
activations = data["background_layered_activations"] # Shape: (1600, 32, 4096)
labels = data["labels"].numpy()

target_layer = 15
X_raw = activations[:, target_layer, :].numpy()

print(f"\nLoaded layer {target_layer} with shape {X_raw.shape}")

avg_cos_sim = np.mean(cosine_similarity(X_raw))
print(f"Average Cosine Similarity (Raw): {avg_cos_sim:.4f}")

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

dimensions_to_test = [4096, 128, 64, 32, 16, 5]
scale_factor = 15.0

print(f"\n{'Dim':<5} | {'Euc Purity':<12} | {'Hyp Purity':<12} | {'Gromov Delta':<12}")
print("-" * 50)

for dim in dimensions_to_test:
    if dim == 4096:
        X_pca = X_raw
    else:
        pca = PCA(n_components=dim, whiten=False)
        X_pca = pca.fit_transform(X_raw)
        
    X_scaled = X_pca * scale_factor
    time_coords = np.sqrt(1 + np.sum(X_scaled**2, axis=1, keepdims=True))
    X_lorentz = np.hstack((time_coords, X_scaled))
    
    dist_matrix_lorentz = calc_lorentz_distance_matrix(X_lorentz)
    dist_matrix_euclidean = pairwise_distances(X_pca, metric='euclidean')
    
    euclidean_p = knn_purity(dist_matrix_euclidean, labels, k=10)
    hyperbolic_p = knn_purity(dist_matrix_lorentz, labels, k=10)
    delta = estimate_gromov_delta(dist_matrix_lorentz, num_samples=2000)
    
    print(f"{dim:<5} | {euclidean_p:.4f}       | {hyperbolic_p:.4f}       | {delta:.4f}")

print("\nSweep Complete!")
