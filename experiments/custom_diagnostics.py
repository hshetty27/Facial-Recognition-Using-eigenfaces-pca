"""
Diagnostics for the custom demo (demo_data/ + unlabelled mystery images).

Nothing here changes the official method (nearest training face in PCA space).
It only measures it, using ONE fitted PCA per training set:

  1. k comparison: U_k = U[:, :k] of the same SVD, k = 2, 4, 5, 7, 10, 15, 20, 30
     - held-out accuracy on the 48 / 12 split (seed 42, 3 per person), as in `demo.py train`
     - leave-one-out accuracy on all 60 images (one SVD per left-out image, shared by all k)
  2. nearest training face  vs  nearest class mean  (m_c = U_k^T (mu_c - mu), diagnostic only)
     for the held-out images and for the mystery images
  3. training faces in the (PC1, PC2) plane
  4. framing of the final 78 x 64 vectors: face-space residual and edge-energy statistics,
     plus a side-by-side figure of training images and the mystery images

The mystery images carry NO labels here: only predictions and distances are printed.

Run from the project root:
    python -m experiments.custom_diagnostics
    python -m experiments.custom_diagnostics --data-dir demo_data --mystery-dir mystery
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.custom_demo import (
    class_mean_projections, face_space_residual, leave_one_out, load_labelled_folder,
    nearest_class_mean, prepare_custom_split, train_model, truncate_model,
)
from src.data_loader import IMAGE_EXTENSIONS, PROJECT_ROOT, load_image
from src.pca import explained_variance_ratio
from src.preprocessing import IMAGE_SHAPE, preprocess_image, vector_to_image
from src.projection import project_new_face
from src.recognition import euclidean_distances, recognize
from src.visualization import ACCENT, INK, MUTED, SERIES_COLORS, _recessive_axes, plot_projection_2d

KS = [2, 4, 5, 7, 10, 15, 20, 30]
OUT = PROJECT_ROOT / "outputs" / "demo"


def nn_per_identity(z: np.ndarray, model: dict) -> tuple[str, dict]:
    """Official rule: D_c = min over identity c of ||z - z_i||, prediction = identity of argmin_i."""
    distances = euclidean_distances(z, model["Z_train"])
    labels = model["labels_train"]
    predicted, _, _ = recognize(z, model["Z_train"], labels)
    identities = sorted(set(labels.tolist()), key=str.lower)
    return predicted, {c: float(distances[labels == c].min()) for c in identities}


def edge_stats(x: np.ndarray) -> dict:
    """
    Where the image structure is, from the 78 x 64 vector only (no face detection):
    G = |horizontal difference| + |vertical difference| of the image, then
      col_extent  = fraction of the width holding the central 80 % of edge energy (columns)
      row_centroid = energy-weighted mean row / height  (0 = top, 1 = bottom)
      flat_fraction = fraction of pixels with G below 0.02 (smooth regions, e.g. plain background)
    """
    image = vector_to_image(x, IMAGE_SHAPE)
    gx = np.abs(np.diff(image, axis=1))[:-1, :]
    gy = np.abs(np.diff(image, axis=0))[:, :-1]
    G = gx + gy
    cols = np.cumsum(G.sum(axis=0)) / G.sum()
    rows = G.sum(axis=1)
    lo, hi = np.searchsorted(cols, 0.1), np.searchsorted(cols, 0.9)
    return {
        "col_extent": (hi - lo) / G.shape[1],
        "row_centroid": float(np.sum(rows * np.arange(len(rows))) / rows.sum() / G.shape[0]),
        "flat_fraction": float(np.mean(G < 0.02)),
    }


def mystery_files(mystery_dir: Path) -> list[Path]:
    return sorted(p for p in mystery_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main(data_dir: Path, mystery_dir: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = prepare_custom_split(data_dir, IMAGE_SHAPE, n_test_per_person=3, seed=42)
    X_all, labels_all, paths_all, _ = load_labelled_folder(data_dir, IMAGE_SHAPE)
    mysteries = mystery_files(mystery_dir)
    X_mys = {p.name: preprocess_image(load_image(p), IMAGE_SHAPE) for p in mysteries}

    fits = {
        "48": (split["X_train"], train_model(split["X_train"], split["labels_train"], split["train_paths"], 10**6)),
        "60": (X_all, train_model(X_all, labels_all, paths_all, 10**6)),
    }

    # ---------- 1. verification of the recognition rule against an explicit re-computation ----------
    X_tr, full48 = fits["48"]
    model = truncate_model(full48, 30)
    mu = X_tr.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(X_tr - mu, full_matrices=False)
    assert np.allclose(model["mean_face"], mu)
    assert np.allclose(np.abs(model["U_k"]), np.abs(U[:, :30]), atol=1e-8)
    for name, x in X_mys.items():
        z = U[:, :30].T @ (x - mu[:, 0])
        Z = U[:, :30].T @ (X_tr - mu)
        d = np.linalg.norm(Z - z[:, None], axis=0)
        explicit = {c: d[split["labels_train"] == c].min() for c in np.unique(split["labels_train"])}
        _, ours = nn_per_identity(project_new_face(x, model["mean_face"], model["U_k"]), model)
        assert all(np.isclose(ours[c], explicit[c]) for c in explicit)
        print(f"verified (k=30, 48 training images): {name:<28} " +
              "  ".join(f"{c} {v:.3f}" for c, v in ours.items()))

    # ---------- 2. k comparison ----------
    loo = leave_one_out(X_all, labels_all, KS)
    rows, mystery_rows = [], []
    for k in KS:
        m48 = truncate_model(full48, k)
        ids, M48 = class_mean_projections(X_tr, split["labels_train"], m48)
        nn_ok = ncm_ok = 0
        for j, truth in enumerate(split["labels_test"]):
            z = project_new_face(split["X_test"][:, j], m48["mean_face"], m48["U_k"])
            nn_ok += nn_per_identity(z, m48)[0] == truth
            ncm_ok += nearest_class_mean(z, M48, ids)[0] == truth
        n = len(split["labels_test"])
        m60 = truncate_model(fits["60"][1], k)
        rows.append({"k": k, "variance_retained_48": round(m48["explained_variance"], 4),
                     "heldout_nn_correct": nn_ok, "heldout_total": n, "heldout_nn_accuracy": round(nn_ok / n, 4),
                     "heldout_ncm_correct": ncm_ok, "heldout_ncm_accuracy": round(ncm_ok / n, 4),
                     "variance_retained_60": round(m60["explained_variance"], 4),
                     "loo60_nn_correct": loo[k], "loo60_total": X_all.shape[1],
                     "loo60_nn_accuracy": round(loo[k] / X_all.shape[1], 4)})
        for tag, (X_fit, full) in fits.items():
            m = truncate_model(full, k)
            ids, M = class_mean_projections(X_fit, m["labels_train"], m)
            for name, x in X_mys.items():
                z = project_new_face(x, m["mean_face"], m["U_k"])
                nn_pred, nn_d = nn_per_identity(z, m)
                ncm_pred, ncm_d = nearest_class_mean(z, M, ids)
                mystery_rows.append({"train_images": tag, "k": k, "image": name,
                                     "nn_prediction": nn_pred, "ncm_prediction": ncm_pred,
                                     **{f"nn_D_{c}": round(v, 3) for c, v in nn_d.items()},
                                     **{f"ncm_d_{c}": round(v, 3) for c, v in ncm_d.items()},
                                     "face_space_residual": round(face_space_residual(x, m), 3)})

    for path, table in [(OUT / "k_comparison.csv", rows), (OUT / "mystery_predictions.csv", mystery_rows)]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"saved {path.relative_to(PROJECT_ROOT)}")

    print("\n  k  var48   held-out NN   held-out NCM   var60   LOO-60 NN")
    for r in rows:
        print(f"{r['k']:3d}  {100 * r['variance_retained_48']:5.1f}%  {r['heldout_nn_correct']:2d}/{r['heldout_total']}"
              f" {100 * r['heldout_nn_accuracy']:5.1f}%  {r['heldout_ncm_correct']:2d}/{r['heldout_total']}"
              f" {100 * r['heldout_ncm_accuracy']:5.1f}%   {100 * r['variance_retained_60']:5.1f}%"
              f"  {r['loo60_nn_correct']:2d}/{r['loo60_total']} {100 * r['loo60_nn_accuracy']:5.1f}%")
    for tag in fits:
        print(f"\nmystery predictions, PCA fitted on {tag} images  (NN = nearest training face, NCM = nearest class mean)")
        print("  k  " + "".join(f"{name[:16]:<34}" for name in X_mys))
        for k in KS:
            cells = [r for r in mystery_rows if r["train_images"] == tag and r["k"] == k]
            print(f"{k:3d}  " + "".join(f"NN {r['nn_prediction']:<9} NCM {r['ncm_prediction']:<9}       " for r in cells))

    # accuracy vs k figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label, color, marker in [("heldout_nn_accuracy", "held-out 12, nearest training face", ACCENT, "o"),
                                      ("heldout_ncm_accuracy", "held-out 12, nearest class mean (diagnostic)", SERIES_COLORS[1], "s"),
                                      ("loo60_nn_accuracy", "leave-one-out 60, nearest training face", SERIES_COLORS[2], "^")]:
        ax.plot(KS, [100 * r[key] for r in rows], color=color, marker=marker, linewidth=2, markersize=7,
                markeredgecolor="white", label=label)
    ax2 = ax.twinx()
    ax2.plot(KS, [100 * r["variance_retained_48"] for r in rows], color=MUTED, linestyle="--", linewidth=1.5)
    ax2.set_ylabel("variance retained, 48-image PCA (%), dashed", color=INK)
    ax2.set_ylim(0, 105)
    ax.set_xticks(KS)
    ax.set_ylim(0, 105)
    ax.set_xlabel("number of components k  (U_k = U[:, :k] of one SVD)")
    ax.set_ylabel("recognition accuracy (%)")
    ax.set_title("Custom dataset: accuracy vs k")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _recessive_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_k_custom.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- 3. PCA plane ----------
    full60 = fits["60"][1]
    ids = sorted(set(labels_all.tolist()), key=str.lower)
    fig = plot_projection_2d(full48["Z_train"], full48["labels_train"], ids, save_path=None)
    ax = fig.axes[0]
    ax.collections[0].set_label("_nolegend_")   # every training face is highlighted: no "other people"
    ax.set_title("48 training faces in the (PC1, PC2) plane; crosses = mystery images (unlabelled)")
    for name, x in X_mys.items():
        z = project_new_face(x, full48["mean_face"], full48["U_k"][:, :2])
        ax.scatter(z[0], z[1], marker="X", s=90, color=INK, zorder=4)
        ax.annotate(Path(name).stem.replace("_extra", ""), (z[0], z[1]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(OUT / "custom_pca_space.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\nPC1/PC2 separation (48-image PCA): centroid of each identity and spread (RMS distance to centroid)")
    Z2 = full48["Z_train"][:2]
    centroids = {}
    for c in ids:
        P = Z2[:, full48["labels_train"] == c]
        centroids[c] = P.mean(axis=1)
        print(f"  {c:<10} centroid ({centroids[c][0]:7.2f}, {centroids[c][1]:7.2f})  "
              f"spread {np.sqrt(np.mean(np.sum((P - centroids[c][:, None]) ** 2, axis=0))):5.2f}")
    print("  pairwise centroid distances: " + ", ".join(
        f"{a}-{b} {np.linalg.norm(centroids[a] - centroids[b]):.2f}" for i, a in enumerate(ids) for b in ids[i + 1:]))
    print(f"  PC1 + PC2 explain {100 * explained_variance_ratio(full48['singular_values'])[1][1]:.1f}% of the variance")

    # ---------- 4. framing ----------
    print("\nFraming of the final 78 x 64 vectors (60-image PCA, k = 30 for the residual)")
    print("  image                           residual  col_extent  row_centroid  flat_fraction")
    m30 = truncate_model(full60, 30)
    loo_res = []
    for j in range(X_all.shape[1]):   # residual of each known image w.r.t. a PCA fitted WITHOUT it
        keep = np.arange(X_all.shape[1]) != j
        m = train_model(X_all[:, keep], labels_all[keep], [""] * 59, 30)
        loo_res.append(face_space_residual(X_all[:, j], m))
    stats = {c: [] for c in ids}
    for j, (c, p) in enumerate(zip(labels_all, paths_all)):
        stats[c].append([loo_res[j], *edge_stats(X_all[:, j]).values()])
    framing_rows = []
    for c in ids:
        a = np.array(stats[c])
        print(f"  {c + ' (15 imgs, mean +- sd)':<30} " + "  ".join(f"{m:.2f}+-{s:.2f}" for m, s in zip(a.mean(0), a.std(0))))
        framing_rows.append({"image": f"{c} (mean of 15)", **dict(zip(
            ["residual", "col_extent", "row_centroid", "flat_fraction"], np.round(a.mean(0), 3)))})
    for name, x in X_mys.items():
        e = edge_stats(x)
        r = face_space_residual(x, m30)
        print(f"  {name:<30} {r:9.2f}  {e['col_extent']:10.2f}  {e['row_centroid']:12.2f}  {e['flat_fraction']:13.2f}")
        framing_rows.append({"image": name, "residual": round(r, 3), **{k: round(v, 3) for k, v in e.items()}})
    with open(OUT / "framing_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(framing_rows[0]))
        writer.writeheader()
        writer.writerows(framing_rows)

    # comparison figure: one representative (image 01) per identity + every mystery image, same 78 x 64 grid
    panels = [(f"{c} 01 (training)", X_all[:, [i for i, l in enumerate(labels_all) if l == c][0]]) for c in ids]
    panels += [(f"{Path(n).stem}\n(mystery, unlabelled)", x) for n, x in X_mys.items()]
    fig, axes = plt.subplots(2, len(panels), figsize=(2.1 * len(panels), 6))
    for col, (title, x) in enumerate(panels):
        image = vector_to_image(x, IMAGE_SHAPE)
        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=1)
        for frac in (0.25, 0.5, 0.75):   # reference grid, same position in every panel
            axes[0, col].axhline(frac * IMAGE_SHAPE[0] - 0.5, color=ACCENT, linewidth=0.6, alpha=0.7)
            axes[0, col].axvline(frac * IMAGE_SHAPE[1] - 0.5, color=ACCENT, linewidth=0.6, alpha=0.7)
        axes[0, col].set_title(title, fontsize=8)
        axes[0, col].axis("off")
        G = np.abs(np.diff(image, axis=1))[:-1, :] + np.abs(np.diff(image, axis=0))[:, :-1]
        axes[1, col].imshow(G, cmap="magma")
        e = edge_stats(x)
        axes[1, col].set_title(f"edge energy |grad I|\nextent {e['col_extent']:.2f}, row c. {e['row_centroid']:.2f}", fontsize=7)
        axes[1, col].axis("off")
    fig.suptitle("Final 78 x 64 representations (what PCA actually sees). Blue lines: 1/4, 1/2, 3/4 of height and width.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "framing_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # all 60 + mystery as final 78 x 64 vectors
    n_cols = 10
    items = [(f"{l} {Path(p).stem}", X_all[:, j]) for j, (l, p) in enumerate(zip(labels_all, paths_all))]
    items += [(Path(n).stem, x) for n, x in X_mys.items()]
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.75 * n_rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, x) in zip(axes.flat, items):
        ax.imshow(vector_to_image(x, IMAGE_SHAPE), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=6)
    fig.suptitle("All known images and mystery images after preprocessing (78 x 64)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "all_preprocessed_78x64.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("\nfigures saved in outputs/demo: accuracy_vs_k_custom.png, custom_pca_space.png, "
          "framing_comparison.png, all_preprocessed_78x64.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "demo_data")
    parser.add_argument("--mystery-dir", type=Path, default=PROJECT_ROOT / "mystery")
    args = parser.parse_args()
    main(args.data_dir, args.mystery_dir)
