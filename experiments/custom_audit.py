"""
Clean-baseline audit of the custom dataset (demo_data/): dataset checks, PCA spectrum,
accuracy vs k under three clearly separated protocols, leakage and preprocessing checks.

The mystery images are NOT loaded or recognised here. Their files are only hashed, to
confirm that none of them is byte-identical to a training image.

Nothing here changes the official method. Everything uses the project functions:
    mu = (1/N) X 1,  X_c = X - mu,  X_c = U Sigma V^T,  lambda_i = sigma_i^2 / (N - 1),
    z = U_k^T (x - mu),  identity = label of argmin_i ||z - z_i||_2  (nearest training image)

Evaluation protocols (never mixed):
  - held-out:      the project split (3 images per person, seed 42 -> 48 train / 12 test);
                   PCA fitted on the 48 training images only.
                   Also repeated over seeds 0..99 (same split function) to show how much a
                   12-image test set fluctuates.
  - leave-one-out: for each of the 60 images, PCA refitted on the other 59.
  - training:      every image recognised against a reference set that CONTAINS it
                   (distance 0 to itself), so 100 % by construction; reported only to show
                   that it is not evidence of anything.

Run from the project root:
    python -m experiments.custom_audit
"""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.custom_demo import (
    check_no_overlap, leave_one_out, load_labelled_folder, prepare_custom_split, recognize_image,
    train_model, truncate_model,
)
from src.data_loader import IMAGE_EXTENSIONS, PROJECT_ROOT, discover_images, load_image
from src.pca import (
    center_data, choose_k, compute_mean_face, compute_svd, eigenvalues_from_singular_values,
    explained_variance_ratio,
)
from src.preprocessing import IMAGE_SHAPE, preprocess_image, vector_to_image
from src.projection import project_data, reconstruct
from src.recognition import recognize_faces
from src.visualization import (
    ACCENT, INK, MUTED, SERIES_COLORS, _recessive_axes, plot_cumulative_variance, plot_eigenfaces,
    plot_explained_variance, plot_mean_face, plot_projection_2d, plot_reconstructions,
)

KS = [2, 4, 5, 7, 10, 15, 20, 30]
THRESHOLDS = [0.80, 0.90, 0.95, 0.99]
N_TEST, SEED = 3, 42                       # the project's held-out protocol (demo.py defaults)
REPEAT_SEEDS = range(100)
TAG = "NEW cropped dataset"
OUT = PROJECT_ROOT / "outputs" / "demo" / "audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. dataset
# ---------------------------------------------------------------------------

def verify_dataset(data_dir: Path, mystery_dir: Path) -> dict:
    paths, labels = discover_images(data_dir)
    again, _ = discover_images(data_dir)
    stray = [f.name for d in data_dir.iterdir() if d.is_dir() for f in d.iterdir()
             if f.suffix.lower() not in IMAGE_EXTENSIONS]
    sizes, formats = Counter(), Counter()
    for p in paths:
        with Image.open(p) as im:
            sizes[im.size] += 1
            formats[im.format] += 1
    hashes = Counter(sha256(p) for p in paths)
    mystery_hits = []
    if mystery_dir.is_dir():
        train_hashes = set(hashes)
        mystery_hits = [p.name for p in mystery_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS and sha256(p) in train_hashes]
    counts = Counter(labels)
    return {
        "paths": paths, "labels": labels, "counts": counts,
        "n_identities": len(counts), "n_images": len(paths),
        "sizes": dict(sizes), "formats": dict(formats), "stray_files": stray,
        "duplicates": sum(c - 1 for c in hashes.values()),
        "deterministic": [str(p) for p in paths] == [str(p) for p in again],
        "mystery_in_training": mystery_hits,
    }


# ---------------------------------------------------------------------------
# 2. recognition vs k
# ---------------------------------------------------------------------------

def nn_correct(X_test: np.ndarray, labels_test: np.ndarray, model: dict) -> int:
    predicted, _, _ = recognize_faces(X_test, model["mean_face"], model["U_k"], model["Z_train"], model["labels_train"])
    return int(np.sum(predicted == labels_test))


def heldout_counts(X: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """One project-style split; one PCA on the training columns; NN accuracy for every k in KS."""
    from src.data_loader import split_indices
    is_test = split_indices(labels, N_TEST, seed)
    full = train_model(X[:, ~is_test], labels[~is_test], [""] * int((~is_test).sum()), max(KS))
    return {k: nn_correct(X[:, is_test], labels[is_test], truncate_model(full, min(k, full["k"]))) for k in KS}


# ---------------------------------------------------------------------------

def main(data_dir: Path, mystery_dir: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    # ---------- 1. dataset ----------
    ds = verify_dataset(data_dir, mystery_dir)
    say(f"DATASET ({TAG}: {data_dir.relative_to(PROJECT_ROOT) if data_dir.is_relative_to(PROJECT_ROOT) else data_dir})")
    say(f"  identities: {ds['n_identities']}   images: {ds['n_images']}")
    for person, n in ds["counts"].items():
        say(f"    {person:<12} {n}")
    say(f"  file formats: {ds['formats']}   pixel sizes (w x h): {ds['sizes']}")
    say(f"  non-image files in identity folders: {ds['stray_files'] or 'none'}")
    say(f"  byte-identical duplicate files: {ds['duplicates']}")
    say(f"  deterministic order (two scans identical): {ds['deterministic']}")
    say(f"  mystery files byte-identical to a training file (hash only): {ds['mystery_in_training'] or 'none'}")
    ok = (ds["n_identities"] == 4 and ds["n_images"] == 60 and set(ds["counts"].values()) == {15}
          and not ds["stray_files"] and ds["duplicates"] == 0 and ds["deterministic"])
    if not ok:
        raise SystemExit("Dataset verification FAILED - stopping before training.")

    X, labels, paths, warnings = load_labelled_folder(data_dir, IMAGE_SHAPE)
    say(f"  aspect-ratio warnings: {warnings or 'none'}")

    # ---------- 2. PCA on all 60 images ----------
    mu = compute_mean_face(X)
    X_c = center_data(X, mu)
    U, S, Vt = compute_svd(X_c)
    full = train_model(X, labels, paths, 10**6, IMAGE_SHAPE)       # same quantities, via the project path
    assert np.allclose(full["mean_face"], mu) and np.allclose(full["U_k"], U[:, : full["rank"]])
    N, r = X.shape[1], full["rank"]
    lam = eigenvalues_from_singular_values(S, N)
    gram = np.sort(np.linalg.eigvalsh(X_c.T @ X_c / (N - 1)))[::-1]      # independent check of lambda_i
    evr, cumulative = explained_variance_ratio(S)
    say(f"\nPCA ({TAG}, all {N} images)")
    say(f"  X: {X.shape}   mu: {mu.shape}   X_c: {X_c.shape}   U: {U.shape}   S: {S.shape}   V^T: {Vt.shape}")
    say(f"  rank(X_c) = {r}  (= N - 1 = {N - 1}: centering removes one dimension, X_c 1 = 0)")
    say(f"  usable principal components: {r}   (sigma_60 = {S[-1]:.2e}, numerically zero)")
    say(f"  max |X_c 1| = {np.abs(X_c.sum(axis=1)).max():.1e};  max |U^T U - I| = {np.abs(U.T @ U - np.eye(U.shape[1])).max():.1e}")
    say(f"  lambda_i = sigma_i^2/(N-1) vs eigenvalues of X_c^T X_c/(N-1): max abs diff {np.abs(lam - gram).max():.1e}")
    say(f"  total variance trace(C) = sum lambda_i = {lam.sum():.4f}")
    say("\n  first components:   i   sigma_i   lambda_i    EVR_i   cumulative")
    for i in range(10):
        say(f"                    {i + 1:3d}  {S[i]:8.4f}  {lam[i]:8.5f}  {100 * evr[i]:6.2f}%  {100 * cumulative[i]:6.2f}%")
    say("\n  variance retained:  " + "  ".join(f"k={k}: {100 * cumulative[k - 1]:.2f}%" for k in KS))
    say("  thresholds:         " + "  ".join(f"{int(100 * t)}% -> k = {choose_k(cumulative[:r], t)}" for t in THRESHOLDS))
    with open(OUT / "spectrum.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "sigma", "lambda", "evr", "cumulative"])
        for i in range(r):
            w.writerow([i + 1, f"{S[i]:.6f}", f"{lam[i]:.8f}", f"{evr[i]:.6f}", f"{cumulative[i]:.6f}"])

    # ---------- 3. recognition vs k ----------
    split = prepare_custom_split(data_dir, IMAGE_SHAPE, N_TEST, SEED)
    m48 = train_model(split["X_train"], split["labels_train"], split["train_paths"], max(KS), IMAGE_SHAPE)
    _, cum48 = explained_variance_ratio(m48["singular_values"])
    held = {k: nn_correct(split["X_test"], split["labels_test"], truncate_model(m48, k)) for k in KS}
    n_test = len(split["labels_test"])
    held_wrong = {}
    for k in KS:
        m = truncate_model(m48, k)
        pred, _, _ = recognize_faces(split["X_test"], m["mean_face"], m["U_k"], m["Z_train"], m["labels_train"])
        held_wrong[k] = [f"{Path(p).parent.name}/{Path(p).name}->{q}"
                         for p, t, q in zip(split["test_paths"], split["labels_test"], pred) if q != t]
    repeated = np.array([[heldout_counts(X, labels, s)[k] for k in KS] for s in REPEAT_SEEDS])  # (seeds, len(KS))
    loo = leave_one_out(X, labels, KS, IMAGE_SHAPE)
    train_acc = {k: nn_correct(X, labels, truncate_model(full, k)) for k in KS}

    say(f"\nRECOGNITION ({TAG}; nearest training image in PCA space)")
    say(f"  held-out split: {N_TEST} per person, seed {SEED} -> {split['X_train'].shape[1]} train / {n_test} test")
    say("  test images: " + ", ".join(f"{Path(p).parent.name}/{Path(p).name}" for p in split["test_paths"]))
    say("\n    k | var(48) | held-out (seed 42)  wrong | held-out, 100 seeds: mean  min  max | var(60) | LOO-60     wrong | training (60)")
    rows = []
    for j, k in enumerate(KS):
        rep = repeated[:, j] / n_test
        say(f"  {k:3d} | {100 * cum48[k - 1]:6.2f}% | {held[k]:2d}/{n_test} {100 * held[k] / n_test:5.1f}%  {n_test - held[k]:5d} |"
            f"            {100 * rep.mean():5.1f}% {100 * rep.min():5.1f} {100 * rep.max():5.1f} | {100 * cumulative[k - 1]:6.2f}% |"
            f" {loo[k]:2d}/{N} {100 * loo[k] / N:5.1f}% {N - loo[k]:4d} | {train_acc[k]}/{N}")
        rows.append({"k": k, "variance_retained_48": round(cum48[k - 1], 4),
                     "heldout_seed42_correct": held[k], "heldout_seed42_incorrect": n_test - held[k],
                     "heldout_seed42_accuracy": round(held[k] / n_test, 4),
                     "heldout_100seeds_mean_accuracy": round(rep.mean(), 4),
                     "heldout_100seeds_min": round(rep.min(), 4), "heldout_100seeds_max": round(rep.max(), 4),
                     "heldout_100seeds_perfect_fraction": round(float(np.mean(rep == 1.0)), 2),
                     "variance_retained_60": round(cumulative[k - 1], 4),
                     "loo60_correct": loo[k], "loo60_incorrect": N - loo[k], "loo60_accuracy": round(loo[k] / N, 4),
                     "training_correct": train_acc[k], "heldout_seed42_errors": "; ".join(held_wrong[k])})
    for k in KS:
        if held_wrong[k]:
            say(f"  held-out errors at k={k}: {', '.join(held_wrong[k])}")
    with open(OUT / "k_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------- 4. leakage ----------
    say("\nLEAKAGE CHECKS")
    test_set = {str(p) for p in split["test_paths"]}
    checks = {
        "held-out mu = mean of the 48 training columns only":
            np.allclose(m48["mean_face"], split["X_train"].mean(axis=1, keepdims=True)),
        "held-out mu differs from the 60-image mean (test images not in mu)":
            not np.allclose(m48["mean_face"], mu),
        "held-out U_k = SVD of the 48 centered training columns only":
            np.allclose(np.abs(m48["U_k"]), np.abs(compute_svd(split["X_train"] - m48["mean_face"])[0][:, : m48["k"]])),
        "reference set Z_train has 48 columns, no test file among them":
            m48["Z_train"].shape[1] == 48 and not (test_set & set(m48["train_files"].tolist())),
        "no test file byte-identical to a training file":
            check_no_overlap(split["train_paths"], split["test_paths"]) is None,
        "no test vector equal to a training vector":
            min(np.linalg.norm(split["X_train"] - split["X_test"][:, [j]], axis=0).min() for j in range(n_test)) > 1e-9,
    }
    # leave-one-out: refit one fold explicitly and compare with the fold inside leave_one_out
    keep = np.arange(N) != 0
    fold = train_model(X[:, keep], labels[keep], [""] * (N - 1), 30)
    checks["LOO fold: mu and U from the 59 other images only"] = (
        np.allclose(fold["mean_face"], X[:, keep].mean(axis=1, keepdims=True))
        and fold["Z_train"].shape[1] == N - 1 and fold["rank"] == N - 2)
    checks["LOO counts reproduce with an explicit refit (k = 10)"] = loo[10] == sum(
        nn_correct(X[:, [j]], labels[[j]], truncate_model(
            train_model(X[:, np.arange(N) != j], labels[np.arange(N) != j], [""] * (N - 1), 10), 10))
        for j in range(N))
    checks["no mystery file is a training file (hash only)"] = not ds["mystery_in_training"]
    for name, passed in checks.items():
        say(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    # ---------- 5. preprocessing ----------
    say("\nPREPROCESSING CHECKS")
    same = all(np.array_equal(X[:, j], preprocess_image(load_image(p), IMAGE_SHAPE)) for j, p in enumerate(paths))
    via_recognize = all(np.array_equal(recognize_image(m48, p)["x"], preprocess_image(load_image(p), IMAGE_SHAPE))
                        for p in split["test_paths"])
    pchecks = {
        "every column of X = preprocess_image(load_image(file))": same,
        "held-out/new images go through the same preprocess_image (recognize_image)": via_recognize,
        f"all source images have h/w = {IMAGE_SHAPE[0]}/{IMAGE_SHAPE[1]} exactly (centre-crop is a no-op, no stretching)":
            all(h * IMAGE_SHAPE[1] == w * IMAGE_SHAPE[0] for (w, h) in ds["sizes"]),
        f"d = {IMAGE_SHAPE[0]} x {IMAGE_SHAPE[1]} = {IMAGE_SHAPE[0] * IMAGE_SHAPE[1]}, values in [0, 1]":
            X.shape[0] == 4992 and X.min() >= 0 and X.max() <= 1,
    }
    for name, passed in pchecks.items():
        say(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    say("  pipeline: EXIF-upright -> grayscale (ITU-R 601 luma) -> centre-crop to 78:64 -> bilinear resize "
        "78 x 64 -> /255 -> row-major flatten. No enhancement, sharpening or equalisation.")

    # ---------- 6. figures ----------
    shape = IMAGE_SHAPE
    plot_mean_face(mu, shape, save_path=None).axes[0].set_title(f"Mean face\n{TAG}, N = {N}")
    plt.gcf().savefig(OUT / "mean_face.png", dpi=120, bbox_inches="tight")
    plot_eigenfaces(U[:, :10], shape, title=f"Eigenfaces u_1 ... u_10  ({TAG}, N = {N})",
                    save_path=OUT / "eigenfaces.png")
    fig = plot_explained_variance(evr[:r], n_show=r)
    fig.axes[0].set_title(f"Explained variance per component  ({TAG}, {r} components)")
    fig.savefig(OUT / "explained_variance.png", dpi=120, bbox_inches="tight")
    fig = plot_cumulative_variance(cumulative[:r], THRESHOLDS)
    fig.axes[0].set_title(f"Cumulative explained variance  ({TAG})")
    fig.savefig(OUT / "cumulative_variance.png", dpi=120, bbox_inches="tight")
    rec_ks = [2, 5, 10, 20, 30, r]
    ids = sorted(set(labels.tolist()), key=str.lower)
    first = [int(np.flatnonzero(labels == c)[0]) for c in ids]
    originals = X[:, first]
    recs = [reconstruct(project_data(originals - mu, U[:, :k]), mu, U[:, :k]) for k in rec_ks]
    fig = plot_reconstructions(originals, recs, rec_ks, shape, row_titles=[f"{c} 01" for c in ids])
    fig.suptitle(f"PCA reconstruction  x_hat = U_k U_k^T (x - mu) + mu   ({TAG}, training images)")
    fig.savefig(OUT / "reconstructions.png", dpi=120, bbox_inches="tight")
    fig = plot_projection_2d(full["Z_train"], labels, ids)
    ax = fig.axes[0]
    ax.collections[0].set_label("_nolegend_")
    ax.set_title(f"All {N} training faces in the (PC1, PC2) plane  ({TAG})\n"
                 f"PC1 + PC2 = {100 * cumulative[1]:.1f}% of variance; variance is not the same as identity")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(OUT / "pca_space_pc1_pc2.png", dpi=120, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    rep_mean = repeated.mean(axis=0) / n_test
    for values, label, color, marker in [
        ([held[k] / n_test for k in KS], f"held-out 12 (seed {SEED})", ACCENT, "o"),
        (rep_mean, "held-out 12, mean over 100 seeds", SERIES_COLORS[1], "s"),
        ([loo[k] / N for k in KS], "leave-one-out 60", SERIES_COLORS[2], "^")]:
        ax.plot(KS, 100 * np.asarray(values), color=color, marker=marker, linewidth=2, markersize=7,
                markeredgecolor="white", label=label)
    ax.fill_between(KS, 100 * repeated.min(axis=0) / n_test, 100 * repeated.max(axis=0) / n_test,
                    color=SERIES_COLORS[1], alpha=0.12, linewidth=0, label="held-out, min-max over 100 seeds")
    ax2 = ax.twinx()
    ax2.plot(KS, [100 * cumulative[k - 1] for k in KS], color=MUTED, linestyle="--", linewidth=1.5)
    ax2.set_ylabel("variance retained, 60-image PCA (%), dashed", color=INK)
    ax2.set_ylim(0, 105)
    ax.set_xticks(KS)
    ax.set_ylim(0, 105)
    ax.set_xlabel("number of components k")
    ax.set_ylabel("nearest-neighbour accuracy (%)")
    ax.set_title(f"Accuracy vs k  ({TAG})")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _recessive_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_k.png", dpi=120, bbox_inches="tight")
    n_cols = 15
    fig, axes = plt.subplots(4, n_cols, figsize=(1.1 * n_cols, 1.55 * 4))
    for ax, j in zip(axes.flat, range(N)):
        ax.imshow(vector_to_image(X[:, j], shape), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{labels[j]} {Path(paths[j]).stem}", fontsize=6)
        ax.axis("off")
    fig.suptitle(f"All {N} training images after preprocessing (78 x 64)  ({TAG})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "all_training_78x64.png", dpi=110, bbox_inches="tight")
    plt.close("all")

    (OUT / "audit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved in {OUT.relative_to(PROJECT_ROOT)}: audit_summary.txt, spectrum.csv, k_comparison.csv, "
          "mean_face.png, eigenfaces.png, explained_variance.png, cumulative_variance.png, reconstructions.png, "
          "pca_space_pc1_pc2.png, accuracy_vs_k.png, all_training_78x64.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "demo_data")
    parser.add_argument("--mystery-dir", type=Path, default=PROJECT_ROOT / "mystery",
                        help="only hashed (overlap check); never loaded or recognised")
    args = parser.parse_args()
    main(args.data_dir.resolve(), args.mystery_dir.resolve())
