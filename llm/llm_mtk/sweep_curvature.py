import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os

print("=== Hyperbolic Curvature (C) Sweep ===")

data_path = "./llama3_hyperbolic/saved_features_and_labels.pt"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

data = torch.load(data_path, map_location="cpu")
activations = data["background_layered_activations"] # Shape: (1600, 32, 4096)
labels = data["labels"].numpy()

def calc_lorentz_distance_matrix(X, C):
    time_prods = np.outer(X[:, 0], X[:, 0])
    space_prods = np.dot(X[:, 1:], X[:, 1:].T)
    
    # Minkowski inner product
    minkowski = -time_prods + space_prods
    
    # For curvature C, distance = (1/sqrt(C)) * arccosh(-C * minkowski)
    # We must clamp -C * minkowski to >= 1.0 to avoid NaNs
    val_inside_arccosh = np.clip(-C * minkowski, a_min=1.0, a_max=None)
    
    dist_mat = (1.0 / np.sqrt(C)) * np.arccosh(val_inside_arccosh)
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

# We will sweep the Curvature on Layer 15
layer = 15
X_raw = activations[:, layer, :].to(dtype=torch.float64).numpy()

pca = PCA(n_components=64, whiten=False)
X_pca_64 = pca.fit_transform(X_raw)

# C values: Small C = gently curved (closer to flat). Large C = highly curved.
C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

print(f"Testing Curvature Parameters (C) on Layer 15 (PCA 64):")
print(f"{'Curvature (C)':<15} | {'Hyp(64) Purity':<15}")
print("-" * 35)

for C in C_values:
    # Proper mapping to Lorentz manifold with curvature C
    # t = sqrt(1/C + ||x||^2)
    time_coords_64 = np.sqrt((1.0 / C) + np.sum(X_pca_64**2, axis=1, keepdims=True))
    X_lorentz_64 = np.hstack((time_coords_64, X_pca_64))
    
    dist_matrix_hyp_64 = calc_lorentz_distance_matrix(X_lorentz_64, C)
    p_hyp_64 = knn_purity(dist_matrix_hyp_64, labels)
    
    print(f"{C:<15.3f} | {p_hyp_64:.4f}")
