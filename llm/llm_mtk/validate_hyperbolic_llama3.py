"""
mtk_hyperbolic_validation.py
=============================
Layer-by-layer diagnostic for whether hyperbolic embeddings add anything over
Euclidean distance for MTK-style benign/malicious anchor separation.

Two questions, kept deliberately separate:

  (A) Is the *native* (Euclidean) activation geometry at this layer tree-like
      at all? -> relative Gromov delta-hyperbolicity, normalized by the
      manifold's own diameter so it's comparable across layers.

      IMPORTANT CAVEAT, confirmed empirically before shipping this script:
      Gromov delta is a global, max-based statistic, and it is badly
      distorted by distance concentration in high dimensions -- a synthetic
      test (hierarchical/tree-shaped point clouds vs. isotropic Gaussian
      noise, same n) shows the tree-vs-flat gap in relative delta shrinking
      from ~0.037 at d=8 to ~0.0009 at d=256 to statistically zero at
      d=4096. So `rel_delta_euc_4k` on the raw 4096-d activations is close
      to meaningless as an absolute "how tree-like is this layer" number --
      concentration of measure swamps it regardless of the data. We compute
      it anyway for continuity, but the trustworthy version of this
      diagnostic is `rel_delta_low_dim`, computed on an 8-d PCA projection
      where the metric still discriminates. Treat the 4096-d and 64-d
      versions as informal / for ranking layers against each other at fixed
      dimensionality only, never as an absolute claim.

  (B) If (A) suggests there's structure to exploit, does actually lifting
      into the hyperboloid model improve the thing MTK's Phase 2 depends on:
      clean, stable local neighborhoods in a fixed benign/malicious
      reference bank? Measured via:
        - k-NN purity (as before, but now per-sample so it can be tested,
          not just averaged)
        - a class-separation ratio (inter-class / intra-class mean distance)
          that stays informative once purity saturates near 1.0
        - k-NN hubness (skew of the in-degree distribution -- a few
          pathological hub points corrupt rank-based signals, which is
          exactly the signal MTK's Phase 2 rank-sequence encoding relies on)

Every headline comparison is a PAIRED test against Euclidean, on the same
points at the same layer, via Wilcoxon signed-rank -- so "hyperbolic wins"
means something other than two noisy means landing 0.001 apart.

Two more differences from the original sweep:
  - The Lorentz lift now normalizes activations by their mean norm before
    applying `scale`, so a given scale value means the same "how far from
    the hyperboloid's apex" thing at every layer and in both the 4096-d and
    64-d spaces. Applying a fixed raw scale (e.g. 15.0) to un-normalized
    activations confounds curvature effects with the fact that residual-
    stream norms grow across depth.
  - `scale` (curvature) is swept rather than fixed, on the cheap 64-d space,
    and the best-found scale is then confirmed once on the full 4096-d
    space (sweeping 4096-d directly is ~64x more expensive per embedding
    and not needed once the right curvature range is known).
"""

import os
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH = "./llama3_hyperbolic/saved_features_and_labels.pt"
OUT_DIR = "validation_results"
K = 10
# Curvature/scale sweep, applied to norm-normalized activations (see
# lorentz_lift) so these are directly comparable across layers/dims.
SCALE_FACTORS = [5.0, 10.0, 15.0, 25.0, 40.0, 60.0]
N_DELTA_SAMPLES = 3000
LOW_DIM = 8  # dimensionality for the trustworthy tree-likeness diagnostic;
              # see the concentration-of-measure caveat above
RNG_SEED = 42
NORM_OUTLIER_Z = 5.0  # flag activations more than this many std devs from
                       # the layer's mean norm ("massive activation" check)

rng = np.random.default_rng(RNG_SEED)


# --------------------------------------------------------------------------
# Geometry primitives
# --------------------------------------------------------------------------
def lorentz_lift(X, scale):
    """Lift Euclidean points onto the hyperboloid model (curvature -1),
    after normalizing by mean norm so `scale` means the same thing at every
    layer and in every dimensionality."""
    X_norm = X / np.mean(np.linalg.norm(X, axis=1))
    X_scaled = X_norm * scale
    time = np.sqrt(1 + np.sum(X_scaled ** 2, axis=1, keepdims=True))
    return np.hstack([time, X_scaled])


def lorentz_distance_matrix(X_lorentz):
    time = X_lorentz[:, 0]
    space = X_lorentz[:, 1:]
    mink = -np.outer(time, time) + space @ space.T
    mink = np.clip(mink, a_max=-1.0, a_min=None)
    d = np.arccosh(-mink)
    np.fill_diagonal(d, 0.0)
    return d


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def knn_indices(dist_mat, k):
    """Indices of the k nearest neighbors of every point, excluding self
    (the unique zero on the diagonal). Vectorized, no per-row Python loop."""
    return np.argsort(dist_mat, axis=1, kind="stable")[:, 1:k + 1]


def per_sample_purity(nbr_idx, labels):
    nbr_labels = labels[nbr_idx]
    return (nbr_labels == labels[:, None]).mean(axis=1)


def hubness_skew(nbr_idx, n):
    """Skew of the k-NN in-degree distribution. Higher = more pathological
    hub points dominating as everyone's nearest neighbor, which corrupts
    rank-based signals like MTK's."""
    indeg = np.bincount(nbr_idx.ravel(), minlength=n)
    return float(stats.skew(indeg))


def class_separation_ratio(dist_mat, labels):
    """Mean inter-class distance / mean intra-class distance. Stays
    informative once purity has saturated near 1.0, since it captures how
    much margin separates the two clusters, not just whether the top-k
    neighbors happen to be correct."""
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same = (labels[:, None] == labels[None, :])[iu]
    d = dist_mat[iu]
    intra = d[same].mean()
    inter = d[~same].mean()
    return float(inter / intra) if intra > 0 else float("nan")


def relative_delta_hyperbolicity(dist_mat, num_samples=N_DELTA_SAMPLES):
    """Gromov 4-point delta, normalized by mean pairwise distance (a
    diameter proxy) so it's comparable across layers/spaces of different
    overall scale. This is the headline 'is this layer's native geometry
    tree-like' number -- compute it on the raw Euclidean distances, not on
    an embedding you've already forced onto a hyperboloid."""
    n = dist_mat.shape[0]
    deltas = np.empty(num_samples)
    for i in range(num_samples):
        x, y, z, w = rng.choice(n, 4, replace=False)
        s = sorted([
            dist_mat[x, y] + dist_mat[z, w],
            dist_mat[x, z] + dist_mat[y, w],
            dist_mat[x, w] + dist_mat[y, z],
        ])
        deltas[i] = (s[2] - s[1]) / 2.0
    iu = np.triu_indices(n, k=1)
    diam = dist_mat[iu].mean()
    delta_mean = float(deltas.mean())
    return delta_mean, (delta_mean / diam if diam > 0 else float("nan"))


def paired_test(purity_a, purity_b):
    """Wilcoxon signed-rank test on per-sample purity: is b systematically
    different from a on the SAME points, beyond what noise would produce?"""
    diff = purity_b - purity_a
    nz = diff[diff != 0]
    if len(nz) < 10:
        return float(diff.mean()), float("nan")
    _, p = stats.wilcoxon(nz)
    return float(diff.mean()), float(p)


def confusable_overlap(purity_a, purity_b):
    """Does metric b fix genuinely hard points, or just relocate the error
    mass? 'confusable' = at least one wrong neighbor in the top-k."""
    conf_a = purity_a < 1.0
    conf_b = purity_b < 1.0
    return dict(
        n_confusable_a=int(conf_a.sum()),
        n_confusable_b=int(conf_b.sum()),
        fixed_by_b=int((conf_a & ~conf_b).sum()),
        newly_broken_by_b=int((~conf_a & conf_b).sum()),
    )


def norm_outlier_report(X, z_thresh=NORM_OUTLIER_Z):
    """Cheap check for 'massive activation' style outliers that can blow up
    max-based statistics like Gromov delta without meaningfully changing
    local neighborhoods."""
    norms = np.linalg.norm(X, axis=1)
    z = (norms - norms.mean()) / norms.std()
    return dict(
        mean_norm=float(norms.mean()),
        max_norm=float(norms.max()),
        max_norm_ratio_to_median=float(norms.max() / np.median(norms)),
        n_outliers=int(np.sum(np.abs(z) > z_thresh)),
    )


# --------------------------------------------------------------------------
# Per-layer analysis (pure numpy in, dict out -- no torch dependency here,
# so this is independently testable/importable without a torch install)
# --------------------------------------------------------------------------
def analyze_layer(X_raw, labels, k=K, scale_factors=SCALE_FACTORS):
    cos_sim = float(np.mean(cosine_similarity(X_raw)))
    outlier_info = norm_outlier_report(X_raw)

    # --- native (Euclidean) geometry, raw dimensionality --------------------
    dist_euc_4k = pairwise_distances(X_raw, metric="euclidean")
    nbr_euc_4k = knn_indices(dist_euc_4k, k)
    purity_euc_4k = per_sample_purity(nbr_euc_4k, labels)
    sep_euc_4k = class_separation_ratio(dist_euc_4k, labels)
    hub_euc_4k = hubness_skew(nbr_euc_4k, len(labels))
    _, delta_rel_4k = relative_delta_hyperbolicity(dist_euc_4k)

    # --- native geometry, PCA-64 --------------------------------------------
    pca = PCA(n_components=64, whiten=False)
    X_pca = pca.fit_transform(X_raw)
    dist_euc_64 = pairwise_distances(X_pca, metric="euclidean")
    nbr_euc_64 = knn_indices(dist_euc_64, k)
    purity_euc_64 = per_sample_purity(nbr_euc_64, labels)
    sep_euc_64 = class_separation_ratio(dist_euc_64, labels)
    hub_euc_64 = hubness_skew(nbr_euc_64, len(labels))
    _, delta_rel_64 = relative_delta_hyperbolicity(dist_euc_64)

    # --- the TRUSTWORTHY tree-likeness diagnostic, at low dimensionality
    # where Gromov delta isn't swamped by distance concentration -----------
    X_low = PCA(n_components=LOW_DIM, whiten=False).fit_transform(X_raw)
    dist_low = pairwise_distances(X_low, metric="euclidean")
    _, delta_rel_low = relative_delta_hyperbolicity(dist_low)

    # --- curvature sweep on PCA-64 (cheap) to find the best scale ----------
    best_scale, best_mean_purity, best_purity, best_dist, best_nbr = (
        None, -np.inf, None, None, None
    )
    sweep_log = []
    for s in scale_factors:
        d_hyp = lorentz_distance_matrix(lorentz_lift(X_pca, s))
        nbr_hyp = knn_indices(d_hyp, k)
        p_hyp = per_sample_purity(nbr_hyp, labels)
        sweep_log.append((s, float(p_hyp.mean())))
        if p_hyp.mean() > best_mean_purity:
            best_scale, best_mean_purity = s, float(p_hyp.mean())
            best_purity, best_dist, best_nbr = p_hyp, d_hyp, nbr_hyp

    sep_hyp_64 = class_separation_ratio(best_dist, labels)
    hub_hyp_64 = hubness_skew(best_nbr, len(labels))
    diff_64, p_64 = paired_test(purity_euc_64, best_purity)
    conf_64 = confusable_overlap(purity_euc_64, best_purity)

    # --- confirm best scale on raw 4096-d (one embedding, not a full sweep) -
    dist_hyp_4k = lorentz_distance_matrix(lorentz_lift(X_raw, best_scale))
    nbr_hyp_4k = knn_indices(dist_hyp_4k, k)
    purity_hyp_4k = per_sample_purity(nbr_hyp_4k, labels)
    sep_hyp_4k = class_separation_ratio(dist_hyp_4k, labels)
    hub_hyp_4k = hubness_skew(nbr_hyp_4k, len(labels))
    diff_4k, p_4k = paired_test(purity_euc_4k, purity_hyp_4k)
    conf_4k = confusable_overlap(purity_euc_4k, purity_hyp_4k)

    row = dict(
        cos_sim=cos_sim,
        rel_delta_low_dim=delta_rel_low,  # trustworthy tree-likeness signal
        rel_delta_euc_4k=delta_rel_4k, rel_delta_euc_64=delta_rel_64,  # informal only, see caveat
        best_scale=best_scale, scale_sweep=sweep_log,
        purity_euc_4k=float(purity_euc_4k.mean()), purity_hyp_4k=float(purity_hyp_4k.mean()),
        diff_purity_4k=diff_4k, p_value_4k=p_4k,
        sep_euc_4k=sep_euc_4k, sep_hyp_4k=sep_hyp_4k,
        hub_euc_4k=hub_euc_4k, hub_hyp_4k=hub_hyp_4k,
        purity_euc_64=float(purity_euc_64.mean()), purity_hyp_64=best_mean_purity,
        diff_purity_64=diff_64, p_value_64=p_64,
        sep_euc_64=sep_euc_64, sep_hyp_64=sep_hyp_64,
        hub_euc_64=hub_euc_64, hub_hyp_64=hub_hyp_64,
    )
    row.update({f"conf_4k_{kk}": v for kk, v in conf_4k.items()})
    row.update({f"conf_64_{kk}": v for kk, v in conf_64.items()})
    row.update({f"norm_{kk}": v for kk, v in outlier_info.items()})
    return row


# --------------------------------------------------------------------------
# Main sweep (torch import is local to this function -- everything above is
# testable/importable without torch installed)
# --------------------------------------------------------------------------
def main():
    import torch
    import pandas as pd

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    data = torch.load(DATA_PATH, map_location="cpu")
    activations = data["background_layered_activations"]  # (N, L, D)
    labels = data["labels"].numpy()
    num_layers = activations.shape[1]

    rows = []
    for l in range(num_layers):
        X_raw = activations[:, l, :].to(dtype=torch.float64).numpy()
        row = analyze_layer(X_raw, labels)
        row["layer"] = l
        rows.append(row)

        print(f"L{l:>2} | relδ(low-dim)={row['rel_delta_low_dim']:.4f} | "
              f"purity euc/hyp(4K)={row['purity_euc_4k']:.4f}/{row['purity_hyp_4k']:.4f} "
              f"(Δ={row['diff_purity_4k']:+.4f}, p={row['p_value_4k']:.3g}) | "
              f"sep euc/hyp={row['sep_euc_4k']:.3f}/{row['sep_hyp_4k']:.3f} | "
              f"hub euc/hyp={row['hub_euc_4k']:.2f}/{row['hub_hyp_4k']:.2f} | "
              f"scale*={row['best_scale']:g} | norm outliers={row['norm_n_outliers']}")

    df = pd.DataFrame(rows)
    df_out = df.drop(columns=["scale_sweep"])  # list column, keep CSV flat
    csv_path = os.path.join(OUT_DIR, "layerwise_headroom.csv")
    df_out.to_csv(csv_path, index=False)

    sig = df[(df.p_value_4k < 0.05) & (df.diff_purity_4k > 0)]
    print("\n=== Summary ===")
    print(f"Layers with a statistically significant (p<0.05) hyperbolic "
          f"purity gain on raw-4096: {len(sig)}/{len(df)}")
    if len(sig):
        print(sig[["layer", "diff_purity_4k", "p_value_4k",
                    "sep_euc_4k", "sep_hyp_4k"]].to_string(index=False))
    print(f"\nFull per-layer table written to {csv_path}")


if __name__ == "__main__":
    main()