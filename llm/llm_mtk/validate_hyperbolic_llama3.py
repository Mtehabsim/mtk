import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

print("=== Phase 1: Empirical Validation of Hyperbolic MTK Assumptions ===")

# 1. Load the Data
data_path = "./llama3/saved_features_and_labels.pt"
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

# 3. PCA Whitening & Dimensionality Reduction
pca_dim = 64
pca = PCA(n_components=pca_dim, whiten=True)
X_whitened = pca.fit_transform(X_raw)

print(f"\n[Step 2] Applied PCA Whitening. Reduced from 4096 to {pca_dim} dimensions.")

# 4. Project to Lorentz Model
# Add the time coordinate t = sqrt(1 + ||x||^2)
time_coords = np.sqrt(1 + np.sum(X_whitened**2, axis=1, keepdims=True))
X_lorentz = np.hstack((time_coords, X_whitened))

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

# 6. Static Neighborhood Purity (Silhouette Score)
print("\n[Step 4] Calculating Static Neighborhood Purity (Silhouette Score)")
euclidean_score = silhouette_score(X_raw, labels, metric='euclidean')
hyperbolic_score = silhouette_score(dist_matrix_lorentz, labels, metric='precomputed')

print(f"Euclidean Purity: {euclidean_score:.4f}")
print(f"Hyperbolic (Lorentz) Purity: {hyperbolic_score:.4f}")
if hyperbolic_score > euclidean_score:
    print("-> Result: Hyperbolic space successfully pushed the Benign and Malicious anchors further apart!")

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
plt.scatter(X_whitened[:, 0], X_whitened[:, 1], c=colors, alpha=0.5, s=10)
plt.title(f"PCA-Whitened Representations (Layer {target_layer})")
plt.savefig("visualizations/pca_whitened_layer15.png")
print("Saved scatter plot to visualizations/pca_whitened_layer15.png")
print("\nValidation Complete. If metrics look good, you are ready to run the pipeline!")
