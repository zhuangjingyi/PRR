"""
t-SNE comparison: PRR vs Baseline.
Generates multiple diverse views for paper selection.

Features:
  - Side-by-side PRR vs Baseline t-SNE (same classes, same seed)
  - Multiple random class subsets for diversity (pick the best)
  - Multiple perplexity values for different granularity
  - Convex hulls around same-class points for cluster visibility
  - Query-gallery connecting lines for intra-class compactness

Usage (standalone, via test.py --tsne):
  python test.py --config_file configs/night600/PRR.yml --tsne
  (automatically compares cfg.TEST.WEIGHT vs --model_without if provided)

Output: {OUTPUT_DIR}/tsne/
  compare_XX_perpYY.png   — side-by-side comparison (per subset & perplexity)
  PRR_samples.png         — multi-subset grid for PRR only
  baseline_samples.png    — multi-subset grid for baseline only
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import os

# ---- global style ----
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 7,
    "figure.dpi": 180, "savefig.dpi": 180,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _get_colors(n):
    """Distinguishable colors cycling through tab10 + Set3."""
    bases = list(plt.cm.tab10.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Paired.colors)
    return [bases[i % len(bases)] for i in range(n)]


# ═══════════════════════════════════════════════════════
# Main entry: called from processor.do_inference
# ═══════════════════════════════════════════════════════
def plot_tsne(features, labels, modalities, save_dir, title="Feature Embedding",
              max_samples_per_class=50, n_views=16, n_classes_per_view=12,
              random_seed=42):
    """
    Generate many diverse t-SNE views by randomly sampling classes and running
    independent t-SNE on each subset. No group figures, no global overview.

    Args:
        features:    [N, D] L2-normalized features
        labels:      [N] person ID (0-indexed)
        modalities:  [N] 0=query, 1=gallery
        save_dir:    output directory
        title:       plot title prefix
        max_samples_per_class: max samples per ID per view
        n_views:     how many random-subset views to generate
        n_classes_per_view: how many IDs to include in each view
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.RandomState(random_seed)
    unique_ids = np.unique(labels)
    n_total = len(unique_ids)
    # filter to IDs with enough samples for meaningful clustering
    min_samples = 50
    id_counts = {pid: (labels == pid).sum() for pid in unique_ids}
    valid_ids = np.array([pid for pid, c in id_counts.items() if c >= min_samples])
    skipped = n_total - len(valid_ids)
    if skipped > 0:
        print(f"t-SNE: {n_total} classes → {len(valid_ids)} with ≥{min_samples} samples (skipped {skipped})")
    unique_ids = valid_ids
    n_total = len(unique_ids)

    print(f"  Generating {n_views} views, {n_classes_per_view} classes each")

    for s in range(n_views):
        seed_s = random_seed + s * 73 + 11
        rng_s = np.random.RandomState(seed_s)

        # pick random classes
        n_pick = min(n_classes_per_view, n_total)
        picked = rng_s.choice(unique_ids, n_pick, replace=False)
        picked = np.sort(picked)

        # sample features for these classes
        keep = np.zeros(len(features), dtype=bool)
        for pid in picked:
            idx = np.where(labels == pid)[0]
            if len(idx) > max_samples_per_class:
                idx = rng_s.choice(idx, max_samples_per_class, replace=False)
            keep[idx] = True

        X_sub = features[keep]
        L_sub = labels[keep]
        M_sub = modalities[keep]

        # PCA + t-SNE
        if X_sub.shape[1] > 50:
            X_sub = PCA(n_components=50, random_state=seed_s).fit_transform(X_sub)
        perp = min(30, X_sub.shape[0] // 3)  # adaptive perplexity
        perp = max(5, perp)
        X_tsne_s = manifold.TSNE(
            n_components=2, init='pca', random_state=seed_s,
            perplexity=perp, learning_rate='auto', max_iter=2000
        ).fit_transform(X_sub)

        _draw(X_tsne_s, L_sub, M_sub,
              title=title, save_path=os.path.join(save_dir, f"tsne_{s+1:02d}.png"))
        print(f"  Saved: tsne_{s+1:02d}.png  ({X_sub.shape[0]} pts, {n_pick} IDs, perp={perp})")

    print(f"t-SNE done → {save_dir}  ({n_views} figures)")


# ═══════════════════════════════════════════════════════
# Drawing function
# ═══════════════════════════════════════════════════════
def _draw(X, labels, modalities, title, save_path):
    """Single clean t-SNE figure — no hulls, no clutter."""
    uids = np.unique(labels); n_ids = len(uids)
    colors = _get_colors(n_ids)
    fig, ax = plt.subplots(figsize=(11, 8))

    for idx, pid in enumerate(uids):
        c = colors[idx]
        mq = (labels == pid) & (modalities == 0)
        mg = (labels == pid) & (modalities == 1)
        if mq.sum():
            ax.scatter(X[mq, 0], X[mq, 1], s=55, c=[c], marker='o',
                       edgecolors='white', linewidth=0.4, alpha=0.88, label=f"ID {pid}")
        if mg.sum():
            ax.scatter(X[mg, 0], X[mg, 1], s=55, c=[c], marker='^',
                       edgecolors='white', linewidth=0.4, alpha=0.88)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.5); s.set_color('#cccccc')

    # modality legend only (clean, no ID legend clutter)
    hm = [
        Line2D([0], [0], marker='o', color='#333', label='query', markersize=8,
               markerfacecolor='#333', markeredgewidth=0, lw=0),
        Line2D([0], [0], marker='^', color='#333', label='gallery', markersize=8,
               markerfacecolor='#333', markeredgewidth=0, lw=0),
    ]
    ax.legend(handles=hm, loc='upper left', fontsize=10, title="Modality",
              title_fontsize=10, framealpha=0.85, edgecolor='#dddddd')

    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight'); plt.close()


