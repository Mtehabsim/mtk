"""
mtk_hybrid_geometry.py
=======================
Builds and validates a dual-geometry MTK trajectory feature: for a fixed
benign/malicious reference bank, computes each test prompt's layer-wise
"distance to benign cluster minus distance to malicious cluster" (delta),
defined via rank as in the paper's Sec 5.1, under BOTH Euclidean and
Hyperbolic distance -- then asks, layer by layer and attack by attack,
which geometry actually carries discriminative signal.

WHY THIS EXISTS
---------------
The end-to-end AUROC table (Euclidean-PCA64 / Hyperbolic-PCA64 /
Hyperbolic-raw4096 / paper baseline) bundles two decisions -- dimensionality
and geometry -- into single numbers you can't take apart. This script gives
you the per-layer, per-attack breakdown underneath those numbers, so you can
see WHERE and for WHICH attacks each geometry actually helps, instead of
guessing from attack names.

TWO OUTPUTS, delivered as two different design philosophies:

  1. Concatenated dual-geometry trajectory (`build_dual_trajectory`):
     don't choose a geometry per layer -- compute both, double the feature
     width, and let the downstream anomaly detector (IsolationForest,
     trained on benign trajectories only, exactly as the paper does) decide
     how to weight them. This can't be overfit to the known attack suite
     because no attack labels are used to build it.

  2. Per-layer worst-case geometry mask (`select_layer_mask`): if you want
     a smaller, single-geometry-per-layer feature vector instead, this
     picks each layer's geometry by the MINIMUM (not mean) AUROC across
     attack types -- optimizing for the worst-case attack, not the average
     one -- and validates the resulting mask with leave-one-attack-out
     cross-validation, so you can see whether it's actually generalizing or
     just memorizing the 10 attacks you have.

CAVEAT THIS SCRIPT CANNOT RESOLVE:
Everything here measures detection of a STATIC attack suite. It says
nothing about robustness to an attacker who knows your final geometry
choice and optimizes against it end-to-end (the paper's own adaptive-attack
protocol). Treat improvements here as "broader coverage of known attack
families," not "proven harder to optimize against" -- that claim needs the
GCG_adapt-style evaluation run separately against whatever this script
recommends.

EXPECTED DATA FORMAT (adjust the loader in main() if yours differs):
  Reference bank (reuse the one from the earlier script):
    {"background_layered_activations": (1600, L, D) tensor,
     "labels": (1600,) tensor, 0=benign, 1=malicious}
  Test set (new):
    {"test_layered_activations": (N_test, L, D) tensor,
     "test_source": list[str] of length N_test, values are "benign" or an
                     attack name (e.g. "SAA", "AutoDAN", "IJP", ...)}
"""

import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

REFERENCE_PATH = "./llama3_hyperbolic/saved_features_and_labels.pt"
TEST_PATH = "./llama3_hyperbolic/test_attack_activations.pt"
HEADROOM_CSV = "validation_results/layerwise_headroom.csv"  # optional, for per-layer best_scale
OUT_DIR = "hybrid_results"
DEFAULT_SCALE = 15.0


# --------------------------------------------------------------------------
# Geometry primitives (bipartite: test points scored against a FIXED
# reference bank, not against each other)
# --------------------------------------------------------------------------
def lift(X, ref_mean_norm, scale):
    """Lift points onto the hyperboloid, normalized by the REFERENCE bank's
    mean norm (not the test batch's own norm) so a given scale means the
    same thing regardless of which test batch you're scoring, and so
    reference and test points are lifted consistently."""
    X_norm = X / ref_mean_norm
    X_scaled = X_norm * scale
    time = np.sqrt(1 + np.sum(X_scaled ** 2, axis=1, keepdims=True))
    return time, X_scaled


def euclidean_bipartite(X_test, X_ref):
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, vectorized
    a2 = np.sum(X_test ** 2, axis=1, keepdims=True)
    b2 = np.sum(X_ref ** 2, axis=1, keepdims=True).T
    d2 = np.clip(a2 + b2 - 2 * X_test @ X_ref.T, a_min=0, a_max=None)
    return np.sqrt(d2)


def hyperbolic_bipartite(X_test, X_ref, scale):
    ref_mean_norm = np.mean(np.linalg.norm(X_ref, axis=1))
    t_test, s_test = lift(X_test, ref_mean_norm, scale)
    t_ref, s_ref = lift(X_ref, ref_mean_norm, scale)
    mink = -np.outer(t_test.ravel(), t_ref.ravel()) + s_test @ s_ref.T
    mink = np.clip(mink, a_max=-1.0, a_min=None)
    return np.arccosh(-mink)


def rank_delta(dist_mat, ref_labels):
    """For each test point (row), the rank position of the nearest benign
    reference point minus the rank position of the nearest malicious
    reference point, where rank = position in the distance-sorted reference
    list (0 = closest overall). Matches the paper's Sec 5.1 definition of
    'distance' as a rank index. Negative = benign-like, positive =
    malicious-like."""
    order = np.argsort(dist_mat, axis=1, kind="stable")  # (n_test, n_ref)
    sorted_labels = ref_labels[order]  # (n_test, n_ref)
    rank_benign = np.argmax(sorted_labels == 0, axis=1)
    rank_malicious = np.argmax(sorted_labels == 1, axis=1)
    return (rank_benign - rank_malicious).astype(float)


# --------------------------------------------------------------------------
# Trajectory construction across all layers
# --------------------------------------------------------------------------
def build_delta_trajectories(ref_activations, ref_labels, test_activations,
                              scales_per_layer):
    """
    ref_activations:  (N_ref, L, D) numpy float64
    test_activations: (N_test, L, D) numpy float64
    scales_per_layer: list of length L (hyperbolic curvature per layer)
    Returns: delta_euc (N_test, L), delta_hyp (N_test, L)
    """
    n_test, L, D = test_activations.shape
    delta_euc = np.empty((n_test, L))
    delta_hyp = np.empty((n_test, L))
    for l in range(L):
        X_ref = ref_activations[:, l, :]
        X_test = test_activations[:, l, :]
        d_euc = euclidean_bipartite(X_test, X_ref)
        d_hyp = hyperbolic_bipartite(X_test, X_ref, scales_per_layer[l])
        delta_euc[:, l] = rank_delta(d_euc, ref_labels)
        delta_hyp[:, l] = rank_delta(d_hyp, ref_labels)
    return delta_euc, delta_hyp


# --------------------------------------------------------------------------
# Per-layer, per-attack discriminability
# --------------------------------------------------------------------------
def per_layer_auroc(delta_benign, delta_attack):
    """AUROC of a single layer's delta value separating benign from one
    attack's trajectories. delta_* are 1-D arrays (one value per prompt,
    for a fixed layer). Higher |delta| = more malicious-like is the
    intended direction, but we let roc_auc_score find the right polarity
    by taking max(auc, 1-auc) -- what matters here is discriminative power,
    not which sign happens to mean 'attack' at this particular layer."""
    y = np.concatenate([np.zeros(len(delta_benign)), np.ones(len(delta_attack))])
    scores = np.concatenate([delta_benign, delta_attack])
    if len(np.unique(y)) < 2:
        return np.nan
    auc = roc_auc_score(y, scores)
    return max(auc, 1 - auc)


def per_layer_auroc_table(delta_euc, delta_hyp, source_labels, layer_range=None):
    """Returns a dict: attack_name -> (auroc_euc[L], auroc_hyp[L])."""
    is_benign = source_labels == "benign"
    attacks = sorted(set(source_labels) - {"benign"})
    L = delta_euc.shape[1]
    layers = range(L) if layer_range is None else layer_range
    table = {}
    for atk in attacks:
        mask = source_labels == atk
        auc_e = np.array([per_layer_auroc(delta_euc[is_benign, l], delta_euc[mask, l])
                           for l in layers])
        auc_h = np.array([per_layer_auroc(delta_hyp[is_benign, l], delta_hyp[mask, l])
                           for l in layers])
        table[atk] = (auc_e, auc_h)
    return table


# --------------------------------------------------------------------------
# Per-layer worst-case geometry mask + leave-one-attack-out validation
# --------------------------------------------------------------------------
def select_layer_mask(auroc_table):
    """For each layer, pick whichever geometry has the higher MINIMUM AUROC
    across all attacks (worst-case-oriented, not mean-oriented). Returns a
    boolean array, True = use hyperbolic at that layer, False = Euclidean."""
    attacks = list(auroc_table.keys())
    L = len(auroc_table[attacks[0]][0])
    min_euc = np.min([auroc_table[a][0] for a in attacks], axis=0)  # (L,)
    min_hyp = np.min([auroc_table[a][1] for a in attacks], axis=0)
    return min_hyp > min_euc, min_euc, min_hyp


def leave_one_attack_out_validation(delta_euc, delta_hyp, source_labels):
    """For each attack, select the layer mask using every OTHER attack, then
    report the held-out attack's AUROC (per layer, mask-selected geometry)
    to see whether the mask generalizes to an attack it never saw."""
    all_attacks = sorted(set(source_labels) - {"benign"})
    results = {}
    for held_out in all_attacks:
        train_attacks = [a for a in all_attacks if a != held_out]
        train_mask = np.isin(source_labels, train_attacks + ["benign"])
        table_train = per_layer_auroc_table(delta_euc[train_mask], delta_hyp[train_mask],
                                             source_labels[train_mask])
        use_hyp, _, _ = select_layer_mask(table_train)

        is_benign = source_labels == "benign"
        held_mask = source_labels == held_out
        L = delta_euc.shape[1]
        held_auroc = np.array([
            per_layer_auroc(
                (delta_hyp if use_hyp[l] else delta_euc)[is_benign, l],
                (delta_hyp if use_hyp[l] else delta_euc)[held_mask, l],
            ) for l in range(L)
        ])
        results[held_out] = held_auroc
    return results


# --------------------------------------------------------------------------
# End-to-end comparison: IsolationForest trained on benign trajectories
# only (matches the paper's zero-jailbreak-training-data design), scored
# against every attack, for four trajectory constructions
# --------------------------------------------------------------------------
def isoforest_auroc(delta_benign_train, delta_benign_test, delta_attack, seed=RNG_SEED):
    clf = IsolationForest(random_state=seed, n_estimators=200)
    clf.fit(delta_benign_train)
    scores_benign = -clf.score_samples(delta_benign_test)  # higher = more anomalous
    scores_attack = -clf.score_samples(delta_attack)
    y = np.concatenate([np.zeros(len(scores_benign)), np.ones(len(scores_attack))])
    scores = np.concatenate([scores_benign, scores_attack])
    return roc_auc_score(y, scores)


def end_to_end_comparison(delta_euc, delta_hyp, source_labels, use_hyp_mask,
                           benign_train_frac=0.5, seed=RNG_SEED):
    """Compares four trajectory constructions head-to-head, all scored with
    the same benign-only IsolationForest protocol the paper uses:
      - euclidean-only (32-dim)
      - hyperbolic-only (32-dim)
      - concatenated dual-geometry (64-dim)
      - worst-case-masked (32-dim, per-layer selected geometry)
    """
    rng_local = np.random.default_rng(seed)
    is_benign = source_labels == "benign"
    benign_idx = np.where(is_benign)[0]
    rng_local.shuffle(benign_idx)
    n_train = int(len(benign_idx) * benign_train_frac)
    train_idx, test_benign_idx = benign_idx[:n_train], benign_idx[n_train:]

    concat = np.hstack([delta_euc, delta_hyp])
    masked = np.where(use_hyp_mask[None, :], delta_hyp, delta_euc)

    variants = {
        "euclidean_only": delta_euc,
        "hyperbolic_only": delta_hyp,
        "concatenated_dual": concat,
        "worst_case_masked": masked,
    }
    attacks = sorted(set(source_labels) - {"benign"})
    results = {name: {} for name in variants}
    for name, feats in variants.items():
        for atk in attacks:
            atk_idx = np.where(source_labels == atk)[0]
            auc = isoforest_auroc(feats[train_idx], feats[test_benign_idx], feats[atk_idx], seed)
            results[name][atk] = auc
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    import torch
    import pandas as pd

    if not (os.path.exists(REFERENCE_PATH) and os.path.exists(TEST_PATH)):
        print(f"Error: expected {REFERENCE_PATH} and {TEST_PATH}. "
              f"Adjust the paths at the top of this script if your files "
              f"are named differently.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    ref_data = torch.load(REFERENCE_PATH, map_location="cpu")
    ref_activations = ref_data["background_layered_activations"].to(dtype=torch.float64).numpy()
    ref_labels = ref_data["labels"].numpy()

    test_data = torch.load(TEST_PATH, map_location="cpu")
    test_activations = test_data["test_layered_activations"].to(dtype=torch.float64).numpy()
    source_labels = np.array(test_data["test_source"])

    L = ref_activations.shape[1]
    scales_per_layer = [DEFAULT_SCALE] * L
    if os.path.exists(HEADROOM_CSV):
        df_prev = pd.read_csv(HEADROOM_CSV).sort_values("layer")
        if len(df_prev) == L:
            scales_per_layer = df_prev["best_scale"].tolist()
            print(f"Loaded per-layer scale from {HEADROOM_CSV}")

    print("Building delta trajectories (Euclidean + Hyperbolic) for all test prompts...")
    delta_euc, delta_hyp = build_delta_trajectories(
        ref_activations, ref_labels, test_activations, scales_per_layer
    )

    print("\n=== Per-layer, per-attack AUROC ===")
    table = per_layer_auroc_table(delta_euc, delta_hyp, source_labels)
    rows = []
    for atk, (auc_e, auc_h) in table.items():
        for l in range(L):
            rows.append(dict(attack=atk, layer=l, auroc_euc=auc_e[l], auroc_hyp=auc_h[l]))
    df_layer = pd.DataFrame(rows)
    df_layer.to_csv(os.path.join(OUT_DIR, "per_layer_per_attack_auroc.csv"), index=False)
    print(f"Written to {OUT_DIR}/per_layer_per_attack_auroc.csv")

    use_hyp_mask, min_euc, min_hyp = select_layer_mask(table)
    print(f"\nWorst-case-selected mask: hyperbolic at "
          f"{use_hyp_mask.sum()}/{L} layers -> {np.where(use_hyp_mask)[0].tolist()}")

    print("\n=== Leave-one-attack-out validation of the mask ===")
    loao = leave_one_attack_out_validation(delta_euc, delta_hyp, source_labels)
    for atk, auc_per_layer in loao.items():
        print(f"  held out {atk:>10s}: mean layer AUROC = {np.nanmean(auc_per_layer):.4f}")

    print("\n=== End-to-end comparison (IsolationForest, benign-only training) ===")
    e2e = end_to_end_comparison(delta_euc, delta_hyp, source_labels, use_hyp_mask)
    df_e2e = pd.DataFrame(e2e)
    df_e2e.to_csv(os.path.join(OUT_DIR, "end_to_end_auroc_comparison.csv"))
    print(df_e2e.round(3).to_string())
    print(f"\nWritten to {OUT_DIR}/end_to_end_auroc_comparison.csv")
    print("\nRemember: this table is against a STATIC attack suite. Validate "
          "the winning variant against an adaptive (GCG_adapt-style) attack "
          "before claiming improved robustness to optimization.")


if __name__ == "__main__":
    main()
