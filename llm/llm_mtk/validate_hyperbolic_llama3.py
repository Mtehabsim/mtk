import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

print("=== Phase 1: Empirical Validation of Hyperbolic MTK Assumptions ===")

# 1. Load the Data
data_path = "./llama3_hyperbolic/saved_features_and_labels.pt"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. You must run extract_trainset_hiddenstates first!")
    exit(1)

data = torch.load(data_path, map_location="cpu")
activations = data["background_layered_activations"] # Shape: (1600, 32, 4096)
labels = data["labels"].numpy()

# We will test on a middle layer where semantic hierarchy is strongest
target_layer = 15
X_raw = activations[:, target_layer, :].numpy()

print(f"\n[Step 1] Loaded layer {target_layer} with shape {X_raw.shape}")

# 2. Check the "Narrow Cone" Anisotropy
# Calculate how tightly clustered all 1600 points are in the original Euclidean space.
avg_cos_sim = np.mean(cosine_similarity(X_raw))
print(f"Average Cosine Similarity (Raw): {avg_cos_sim:.4f}")
if avg_cos_sim > 0.90:
    print("-> Result: High Anisotropy detected. The vectors are clustered in a narrow cone. Whitening is required.")

# 3. PCA Without Whitening (Preserve the safety axis!)
pca_dim = 64
pca = PCA(n_components=pca_dim, whiten=False)  # <-- Changed to False
X_pca = pca.fit_transform(X_raw)

print(f"\n[Step 2] Applied PCA. Reduced from 4096 to {pca_dim} dimensions.")

# 4. Project to Lorentz Model with a Scale Factor
# We multiply by a scalar to push the points outward into the exponential space
scale_factor = 15.0  # <-- Increased to push vectors deeper into hyperbolic space
X_scaled = X_pca * scale_factor

# Add the time coordinate t = sqrt(1 + ||x||^2)
time_coords = np.sqrt(1 + np.sum(X_scaled**2, axis=1, keepdims=True))
X_lorentz = np.hstack((time_coords, X_scaled))

# 5. Lorentz Distance Matrix
print("\n[Step 3] Calculating Lorentz Distance Matrix...")
def calc_lorentz_distance_matrix(X):
    N = X.shape[0]
    # U0*V0
    time_prods = np.outer(X[:, 0], X[:, 0])
    # Space inner products
    space_prods = np.dot(X[:, 1:], X[:, 1:].T)
    # Minkowski inner product
    minkowski = -time_prods + space_prods
    # Clamp to avoid math domain errors
    minkowski = np.clip(minkowski, a_max=-1.0, a_min=None)
    # Arcosh
    return np.arccosh(-minkowski)

dist_matrix_lorentz = calc_lorentz_distance_matrix(X_lorentz)
np.fill_diagonal(dist_matrix_lorentz, 0.0)

# 6. Static Neighborhood Purity (K-NN Purity for k=10)
# Note: Silhouette score fails in Hyperbolic space because it uses average distances,
# and Hyperbolic space expands exponentially, causing intra-cluster distances to look huge.
# MTK uses K-Nearest Neighbors (K-NN), so we must measure K-NN Purity!
print("\n[Step 4] Calculating Static Neighborhood Purity (K-NN Purity, k=10)")
def knn_purity(dist_mat, labels, k=10):
    n = dist_mat.shape[0]
    correct = 0
    for i in range(n):
        # Find k nearest neighbors (excluding self, which is distance 0)
        # argsort sorts ascending. Index 0 is self. Indices 1 to k are neighbors.
        neighbors = np.argsort(dist_mat[i])[1:k+1]
        neighbor_labels = labels[neighbors]
        # Purity for this point is the fraction of neighbors with the same label
        correct += np.sum(neighbor_labels == labels[i]) / k
    return correct / n

# We need a standard Euclidean distance matrix for the baseline
from sklearn.metrics import pairwise_distances
dist_matrix_euclidean = pairwise_distances(X_pca, metric='euclidean')

euclidean_purity = knn_purity(dist_matrix_euclidean, labels, k=10)
hyperbolic_purity = knn_purity(dist_matrix_lorentz, labels, k=10)

print(f"Euclidean K-NN Purity: {euclidean_purity:.4f}")
print(f"Hyperbolic (Lorentz) K-NN Purity: {hyperbolic_purity:.4f}")
if hyperbolic_purity >= euclidean_purity:
    print("-> Result: Hyperbolic space maintains or improves local neighborhood purity!")

# 7. Gromov's Delta-Hyperbolicity Estimation
print("\n[Step 5] Estimating Gromov's Delta-Hyperbolicity...")
def estimate_gromov_delta(dist_mat, num_samples=5000):
    N = dist_mat.shape[0]
    deltas = []
    for _ in range(num_samples):
        # Pick 4 random points
        p = np.random.choice(N, 4, replace=False)
        x, y, z, w = p[0], p[1], p[2], p[3]
        
        # Calculate sums of opposite pairs
        sums = [
            dist_mat[x, y] + dist_mat[z, w],
            dist_mat[x, z] + dist_mat[y, w],
            dist_mat[x, w] + dist_mat[y, z]
        ]
        sums.sort()
        # Delta is the difference between the two largest sums divided by 2
        delta = (sums[2] - sums[1]) / 2.0
        deltas.append(delta)
    return np.mean(deltas)

avg_delta = estimate_gromov_delta(dist_matrix_lorentz)
print(f"Average Gromov's Delta: {avg_delta:.4f}")
print("-> Note: Values close to 0 indicate strong hierarchical/tree-like structure.")

# 8. Visualizations
print("\n[Step 6] Saving Visualizations...")
os.makedirs("visualizations", exist_ok=True)
plt.figure(figsize=(8, 6))
colors = ['green' if l == 0 else 'red' for l in labels]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=10)
plt.title(f"PCA Representations (Layer {target_layer})")
try:
    plt.savefig("visualizations/pca_layer15.png")
    print("Saved scatter plot to visualizations/pca_layer15.png")
except Exception as e:
    print(f"Warning: Could not save visualization due to error: {e}")
print("\nValidation Complete. If metrics look good, you are ready to run the pipeline!")
