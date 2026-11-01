"""
Phase 4 experiment: nearest-neighbour face recognition in PCA space, for several k.

    1. load data, fixed split (seed 42)          6. Z_train = U_k^T X_c
    2. mu from X_train ONLY                       7. z_test  = U_k^T (x_test - mu)   (training mu, U_k)
    3. X_c = X_train - mu                         8. nearest neighbour: argmin_i ||z_test - z_i||
    4. X_c = U Sigma V^T (X_train ONLY)           9. accuracy, variance retained, distance statistics
    5. for each k in K_VALUES: U_k = U[:, :k]    10. save results and figures

The test faces are used for nothing except steps 7-9. PCA is never recomputed with test data.
The k values were fixed in advance from the Phase 3 variance analysis. If several k tie for
the best accuracy, the smallest k is reported (fewest dimensions).

Run from the project root:   python -m experiments.run_recognition
"""

import csv

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.data_loader import DATASET_DIR, PROJECT_ROOT, download_att_faces, load_dataset, train_test_split
from src.pca import center_data, compute_mean_face, compute_svd, explained_variance_ratio, get_principal_components
from src.preprocessing import IMAGE_SHAPE
from src.projection import project_data
from src.recognition import accuracy, confusion_matrix, recognize_faces
from src.visualization import (
    plot_accuracy_vs_k, plot_confusion_matrix, plot_projection_2d,
    plot_recognition_examples, plot_results_table, plot_variance_vs_k,
)

N_TEST_PER_PERSON = 3
SEED = 42
K_VALUES = [5, 10, 20, 30, 50, 66]
N_EXAMPLES = 10                                   # the first 10 test faces, not hand-picked
HIGHLIGHT_PEOPLE = ["s1", "s2", "s3", "s4", "s5", "s6"]  # fixed: the first six people

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def main() -> None:
    # 1. Data and fixed split
    download_att_faces()
    X, labels = load_dataset(DATASET_DIR, IMAGE_SHAPE)
    X_train, labels_train, X_test, labels_test = train_test_split(X, labels, N_TEST_PER_PERSON, SEED)
    classes = sorted(set(labels_train.tolist()), key=lambda s: int(s[1:]))

    # 2-4. PCA learned from the TRAINING faces only
    mu = compute_mean_face(X_train)
    X_c = center_data(X_train, mu)
    U, S, _ = compute_svd(X_c)
    _, cumulative = explained_variance_ratio(S)
    assert mu.shape == (X_train.shape[0], 1) and U.shape[1] == X_train.shape[1]  # PCA sized by N_train

    lines = [
        "FACE RECOGNITION IN PCA SPACE (nearest neighbour, Euclidean distance)",
        f"training faces: {X_train.shape[1]}  (X_train {X_train.shape})",
        f"test faces:     {X_test.shape[1]}  (X_test {X_test.shape})",
        f"people: {len(classes)}; split seed {SEED}; {N_TEST_PER_PERSON} test images per person",
        "PCA (mu, U, Sigma) computed from X_train only; test faces projected with the training mu and U_k.",
        "",
        "    k   variance   accuracy   correct   mean d   median d    min d    max d",
    ]

    # 5-9. Recognition for each k
    rows, runs = [], {}
    for k in K_VALUES:
        U_k = get_principal_components(U, k)
        Z_train = project_data(X_c, U_k)                                   # (k, 280)
        predictions, nearest, distances = recognize_faces(X_test, mu, U_k, Z_train, labels_train)
        acc = accuracy(predictions, labels_test)
        row = {
            "k": k,
            "explained_variance": float(cumulative[k - 1]),
            "accuracy": acc,
            "n_correct": int(np.sum(predictions == labels_test)),
            "n_test": len(labels_test),
            "mean_distance": float(np.mean(distances)),
            "median_distance": float(np.median(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
        }
        rows.append(row)
        runs[k] = (Z_train, predictions, nearest, distances)
        lines.append(f"  {k:3d}    {100 * row['explained_variance']:5.1f}%    {100 * acc:5.1f}%   "
                     f"{row['n_correct']:3d}/{row['n_test']}   {row['mean_distance']:6.3f}   "
                     f"{row['median_distance']:6.3f}   {row['min_distance']:6.3f}   {row['max_distance']:6.3f}")

    best = max(rows, key=lambda r: (r["accuracy"], -r["k"]))  # highest accuracy, then smallest k
    best_k = best["k"]
    Z_best, pred_best, nearest_best, dist_best = runs[best_k]
    correct = pred_best == labels_test
    lines += [
        "",
        f"best k = {best_k}: accuracy {100 * best['accuracy']:.1f}% ({best['n_correct']}/{best['n_test']}), "
        f"variance retained {100 * best['explained_variance']:.1f}%",
        f"mean nearest distance at k = {best_k}: correct {dist_best[correct].mean():.3f}, "
        f"wrong {dist_best[~correct].mean():.3f}" if (~correct).any() else "",
        "",
        f"misrecognised test faces at k = {best_k} (test column: true -> predicted, distance):",
    ]
    lines += [f"  {j:3d}: {labels_test[j]} -> {pred_best[j]}  (d = {dist_best[j]:.3f})"
              for j in np.flatnonzero(~correct)]
    print("\n".join(lines))

    # 10. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "recognition_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "explained_variance", "accuracy", "mean_distance", "median_distance"])
        for r in rows:
            writer.writerow([r["k"], f"{r['explained_variance']:.6f}", f"{r['accuracy']:.6f}",
                             f"{r['mean_distance']:.6f}", f"{r['median_distance']:.6f}"])
    (RESULTS_DIR / "recognition_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figures
    plot_accuracy_vs_k(K_VALUES, [r["accuracy"] for r in rows], save_path=FIGURES_DIR / "accuracy_vs_k.png")
    plot_variance_vs_k(cumulative, K_VALUES, save_path=FIGURES_DIR / "explained_variance_vs_k.png")
    plot_results_table(rows, save_path=FIGURES_DIR / "recognition_results_table.png")
    examples = list(range(N_EXAMPLES))
    plot_recognition_examples(
        X_test[:, examples], X_train[:, nearest_best[examples]],
        labels_test[examples].tolist(), pred_best[examples].tolist(), dist_best[examples].tolist(),
        IMAGE_SHAPE, title=f"Recognition examples: first {N_EXAMPLES} test faces (k = {best_k})",
        save_path=FIGURES_DIR / "recognition_examples.png",
    )
    plot_confusion_matrix(
        confusion_matrix(labels_test, pred_best, classes), classes,
        title=f"Confusion matrix, k = {best_k} (accuracy {100 * best['accuracy']:.1f}%, 120 test faces)",
        save_path=FIGURES_DIR / f"confusion_matrix_k{best_k}.png",
    )
    plot_projection_2d(runs[K_VALUES[0]][0], labels_train, HIGHLIGHT_PEOPLE,
                       save_path=FIGURES_DIR / "pca_space_pc1_pc2.png")
    print(f"\nResults saved to {RESULTS_DIR.relative_to(PROJECT_ROOT)}, figures to {FIGURES_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
