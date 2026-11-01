"""
Plots that make the mathematics visible.

All figures are saved under outputs/figures/. Only matplotlib is used.
Every function returns the matplotlib Figure; it is saved if save_path is given.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import vector_to_image


def _save(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")


def plot_faces(
    images: list[np.ndarray],
    titles: list[str],
    n_cols: int = 8,
    suptitle: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Show a grid of grayscale images.

    Input:  list of arrays of shape (h, w) with values in [0, 1], one title per image.
    Output: the Figure (and a saved PNG if save_path is given).

    vmin=0, vmax=1 fixes the grey scale, so the same pixel value always
    has the same shade in every panel.
    """
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.7 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_vector_roundtrip(
    original: np.ndarray,
    vector: np.ndarray,
    image_shape: tuple[int, int],
    title: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Verify flattening visually: original image | vector x in R^d | x reshaped to (h, w).

    Input:  original image array (any size), vector x of shape (d,), image_shape (h, w).

    The middle panel plots x as d values against their index; the right panel
    should show the same face as the left, at the reduced resolution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 3, 1]})
    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Original {original.shape[0]} x {original.shape[1]} (h x w)")
    axes[1].plot(vector, linewidth=0.4, color="black")
    axes[1].set_title(f"Column of X:  x in R^{vector.size}")
    axes[1].set_xlabel("pixel index (row-major)")
    axes[1].set_ylabel("intensity")
    axes[1].set_xlim(0, vector.size)
    axes[1].set_ylim(0, 1)
    axes[2].imshow(vector_to_image(vector, image_shape), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"x reshaped to {image_shape[0]} x {image_shape[1]} (h x w)")
    for ax in (axes[0], axes[2]):
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_one_per_person(
    X: np.ndarray,
    labels: np.ndarray,
    image_shape: tuple[int, int],
    n_cols: int = 10,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Show the first column of X belonging to each person (reshaped from the vector).

    Input:  X of shape (d, N), labels of shape (N,), image_shape (h, w).
    """
    people = list(dict.fromkeys(labels.tolist()))  # unique, in column order
    first_columns = [int(np.flatnonzero(labels == p)[0]) for p in people]
    images = [vector_to_image(X[:, j], image_shape) for j in first_columns]
    return plot_faces(images, people, n_cols=n_cols,
                      suptitle="One face per person (columns of X reshaped)", save_path=save_path)


# ---------------------------------------------------------------------------
# PCA figures
# ---------------------------------------------------------------------------

def plot_mean_face(mu: np.ndarray, image_shape: tuple[int, int], save_path: Path | None = None) -> plt.Figure:
    """
    Display the mean face mu (shape (d, 1)) reshaped to (h, w).

    mu is an average of faces, so its values lie in [0, 1] and are shown on the
    same fixed grey scale as the original faces.
    """
    fig, ax = plt.subplots(figsize=(3, 3.6))
    ax.imshow(vector_to_image(mu.ravel(), image_shape), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Mean Face")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_eigenfaces(
    U_k: np.ndarray,
    image_shape: tuple[int, int],
    n_cols: int = 5,
    title: str = "Eigenfaces",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Display principal directions as images: column u_j of U_k reshaped to (h, w).

    Input:  U_k of shape (d, k), image_shape (h, w) with h * w = d.

    Eigenfaces are NOT faces of real people. Each u_j is a unit vector in R^d whose
    entries can be negative: it is a pattern of pixel changes away from the mean face,
    and u_1 is the pattern along which the training faces vary the most.

    Display only: each eigenface is linearly rescaled by imshow to the full grey range
    (darkest entry = black, brightest = white). U_k itself is not modified.
    The sign of each u_j is arbitrary (if u_j is an eigenvector, so is -u_j), so an
    eigenface and its "photographic negative" carry the same information.
    """
    k = U_k.shape[1]
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for j, ax in enumerate(axes.flat[:k]):
        ax.imshow(vector_to_image(U_k[:, j], image_shape), cmap="gray")  # autoscaled for display
        ax.set_title(f"u_{j + 1}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_explained_variance(evr: np.ndarray, n_show: int = 50, save_path: Path | None = None) -> plt.Figure:
    """
    Bar chart of EVR_i = sigma_i^2 / sum_j sigma_j^2 for the first n_show components.

    Input:  evr of shape (r,).
    """
    n_show = min(n_show, len(evr))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(np.arange(1, n_show + 1), 100 * evr[:n_show], color="#4C72B0")
    ax.set_xlabel("principal component i")
    ax.set_ylabel("explained variance (%)")
    ax.set_title(f"Explained Variance by Principal Component (first {n_show} of {len(evr)})")
    ax.set_xlim(0.3, n_show + 0.7)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_cumulative_variance(
    cumulative: np.ndarray,
    thresholds: list[float],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Cumulative explained variance against k, with the smallest k for each threshold marked.

    Input:  cumulative of shape (r,), where cumulative[k - 1] is the fraction kept by k components.
    """
    from src.pca import choose_k

    ks = np.arange(1, len(cumulative) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, 100 * cumulative, color="black", linewidth=1.5)
    for threshold in thresholds:
        k = choose_k(cumulative, threshold)
        ax.axhline(100 * threshold, color="gray", linestyle=":", linewidth=0.8)
        ax.plot(k, 100 * cumulative[k - 1], "o", color="#C44E52")
        ax.annotate(f"{threshold:.0%}: k = {k}", (k, 100 * cumulative[k - 1]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.set_xlabel("number of components k")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlim(0, len(cumulative) + 1)
    ax.set_ylim(0, 101)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_singular_values(singular_values: np.ndarray, save_path: Path | None = None) -> plt.Figure:
    """
    Singular values sigma_i on a logarithmic axis.

    The last singular value is ~0 because centering makes rank(X_c) <= N - 1.
    """
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ks = np.arange(1, len(singular_values) + 1)
    ax.semilogy(ks, np.maximum(singular_values, 1e-16), ".", markersize=3, color="black")
    ax.set_xlabel("index i")
    ax.set_ylabel("sigma_i (log scale)")
    ax.set_title("Singular Values of the Centered Training Matrix X_c")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_reconstructions(
    originals: np.ndarray,
    reconstructions: list[np.ndarray],
    ks: list[int],
    image_shape: tuple[int, int],
    row_titles: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Show faces next to their PCA reconstructions x_hat = U_k U_k^T (x - mu) + mu for several k.

    Input:  originals of shape (d, m): m faces, one per row of the figure;
            reconstructions[j] of shape (d, m), made with ks[j] components.

    Each panel title gives the Euclidean error || x - x_hat ||_2. Reconstructions can
    leave [0, 1] slightly; they are clipped to [0, 1] for display only.
    """
    m = originals.shape[1]
    n_cols = 1 + len(ks)
    fig, axes = plt.subplots(m, n_cols, figsize=(1.9 * n_cols, 2.5 * m), squeeze=False)
    for i in range(m):
        axes[i, 0].imshow(vector_to_image(originals[:, i], image_shape), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("original" if row_titles is None else f"original ({row_titles[i]})", fontsize=8)
        for j, (k, X_hat) in enumerate(zip(ks, reconstructions), start=1):
            error = np.linalg.norm(originals[:, i] - X_hat[:, i])
            shown = np.clip(vector_to_image(X_hat[:, i], image_shape), 0.0, 1.0)
            axes[i, j].imshow(shown, cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"k = {k}\nerror {error:.2f}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("PCA Reconstruction  x_hat = U_k U_k^T (x - mu) + mu")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Recognition figures
# ---------------------------------------------------------------------------

# Categorical palette (validated for colour-vision deficiency) and one sequential hue.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P"]
ACCENT = "#2a78d6"
MUTED = "#b5b5b0"
INK = "#3a3a38"


def _recessive_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_projection_2d(
    Z: np.ndarray,
    labels: np.ndarray,
    highlight: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Scatter plot of faces using their first two PCA coordinates (rows 0 and 1 of Z).

    Input:  Z of shape (k, N) with k >= 2; labels of shape (N,); a few people to highlight.

    This is NOT a new dimensionality-reduction method: the points are exactly the
    first two coordinates z_1i = u_1^T (x_i - mu), z_2i = u_2^T (x_i - mu) already
    computed by PCA. Labels are used only for colouring. All faces are drawn in grey;
    the highlighted people get a colour AND a marker shape, so identity is not colour-only.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(Z[0], Z[1], s=14, color=MUTED, linewidths=0, label="other people")
    for person, color, marker in zip(highlight, SERIES_COLORS, SERIES_MARKERS):
        mask = labels == person
        ax.scatter(Z[0, mask], Z[1, mask], s=46, color=color, marker=marker,
                   edgecolors="white", linewidths=1.0, label=person, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.set_xlabel("PC1 coordinate  z_1 = u_1^T (x - mu)")
    ax.set_ylabel("PC2 coordinate  z_2 = u_2^T (x - mu)")
    ax.set_title("Training faces in the plane of the first two principal components")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _recessive_axes(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_accuracy_vs_k(ks: list[int], accuracies: list[float], save_path: Path | None = None) -> plt.Figure:
    """Recognition accuracy on the test faces for each number of components k."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, 100 * np.array(accuracies), color=ACCENT, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5)
    for k, a in zip(ks, accuracies):
        ax.annotate(f"{100 * a:.1f}%", (k, 100 * a), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of PCA components k")
    ax.set_ylabel("Recognition accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Recognition Accuracy vs Number of Principal Components")
    _recessive_axes(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_variance_vs_k(cumulative: np.ndarray, ks: list[int], save_path: Path | None = None) -> plt.Figure:
    """Cumulative explained variance (training data) with the evaluated k values marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    k_max = max(ks) + 20
    ax.plot(np.arange(1, k_max + 1), 100 * cumulative[:k_max], color=MUTED, linewidth=2)
    ax.plot(ks, [100 * cumulative[k - 1] for k in ks], "o", color=ACCENT, markersize=8,
            markeredgecolor="white", markeredgewidth=1.5)
    for k in ks:
        ax.annotate(f"{100 * cumulative[k - 1]:.1f}%", (k, 100 * cumulative[k - 1]),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color=INK)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of PCA components k")
    ax.set_ylabel("Explained variance retained (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Explained Variance vs Number of Principal Components")
    _recessive_axes(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_results_table(rows: list[dict], save_path: Path | None = None) -> plt.Figure:
    """Render the k | variance | accuracy | distance table as a figure."""
    header = ["k", "variance retained", "accuracy", "correct / total", "mean NN distance", "median NN distance"]
    cells = [[str(r["k"]), f"{100 * r['explained_variance']:.1f}%", f"{100 * r['accuracy']:.1f}%",
              f"{r['n_correct']} / {r['n_test']}", f"{r['mean_distance']:.3f}", f"{r['median_distance']:.3f}"]
             for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.45 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d8d8d4")
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("PCA-space nearest-neighbour recognition (120 test faces)", fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_recognition_examples(
    test_faces: np.ndarray,
    matched_faces: np.ndarray,
    true_labels: list[str],
    predicted_labels: list[str],
    distances: list[float],
    image_shape: tuple[int, int],
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Pairs of (test face, nearest training face in PCA space), one pair per column.

    Input:  test_faces and matched_faces of shape (d, m); labels and distances of length m.
    The result is written as text (CORRECT / WRONG) so it is not signalled by colour alone.
    """
    m = test_faces.shape[1]
    fig, axes = plt.subplots(2, m, figsize=(1.9 * m, 5.8), squeeze=False)
    for j in range(m):
        correct = true_labels[j] == predicted_labels[j]
        axes[0, j].imshow(vector_to_image(test_faces[:, j], image_shape), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"test: {true_labels[j]}", fontsize=8)
        axes[1, j].imshow(vector_to_image(matched_faces[:, j], image_shape), cmap="gray", vmin=0, vmax=1)
        axes[1, j].text(0.5, -0.04, f"nearest: {predicted_labels[j]}\nd = {distances[j]:.2f}\n"
                        f"{'CORRECT' if correct else 'WRONG'}", transform=axes[1, j].transAxes,
                        ha="center", va="top", fontsize=8, weight="normal" if correct else "bold")
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].text(-0.15, 0.5, "test face", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.15, 0.5, "nearest training face", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion_matrix(counts: np.ndarray, classes: list[str], title: str, save_path: Path | None = None) -> plt.Figure:
    """
    40 x 40 confusion matrix: rows = true identity, columns = predicted identity.
    Single-hue sequential colour (white = 0); every non-zero cell also shows its count.
    """
    n = len(classes)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(counts, cmap="Blues", vmin=0, vmax=counts.max())
    for a in range(n):
        for b in range(n):
            if counts[a, b]:
                ax.text(b, a, str(counts[a, b]), ha="center", va="center", fontsize=6,
                        color="white" if counts[a, b] > counts.max() / 2 else INK)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=6)
    ax.set_yticks(range(n), classes, fontsize=6)
    ax.set_xlabel("Predicted Identity")
    ax.set_ylabel("True Identity")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.04, label="number of test faces")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_single_recognition(
    test_face: np.ndarray,
    nearest_face: np.ndarray,
    distances: np.ndarray,
    train_labels: np.ndarray,
    true_label: str,
    predicted_label: str,
    image_shape: tuple[int, int],
    k: int,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    One test face through the pipeline: test face | nearest training face | distances to all
    training faces (sorted), with the true person's training faces marked.
    """
    fig = plt.figure(figsize=(12, 3.8))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.15, 3])
    ax_test, ax_near, ax_dist = fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[3])
    ax_test.imshow(vector_to_image(test_face, image_shape), cmap="gray", vmin=0, vmax=1)
    ax_test.set_title(f"Test face\ntrue identity: {true_label}", fontsize=9)
    ax_near.imshow(vector_to_image(nearest_face, image_shape), cmap="gray", vmin=0, vmax=1)
    ax_near.set_title(f"Nearest training face\npredicted: {predicted_label}", fontsize=9)
    for ax in (ax_test, ax_near):
        ax.axis("off")

    order = np.argsort(distances)
    same = train_labels[order] == true_label
    ranks = np.arange(1, len(distances) + 1)
    ax_dist.scatter(ranks[~same], distances[order][~same], s=10, color=MUTED, linewidths=0, label="other people")
    ax_dist.scatter(ranks[same], distances[order][same], s=46, color=ACCENT, marker="D",
                    edgecolors="white", linewidths=1.0, label=f"training faces of {true_label}", zorder=3)
    ax_dist.set_xlabel("training faces, sorted by distance")
    ax_dist.set_ylabel(f"|| z - z_i ||  (k = {k})")
    ax_dist.set_title("Distance in PCA space to all training faces", fontsize=9)
    ax_dist.legend(frameon=False, fontsize=8, loc="lower right")
    _recessive_axes(ax_dist)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Custom demo figure
# ---------------------------------------------------------------------------

def plot_demo_recognition(
    x: np.ndarray,
    mu: np.ndarray,
    z: np.ndarray,
    nearest_face: np.ndarray,
    per_identity: dict[str, float],
    predicted: str,
    image_shape: tuple[int, int],
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    The recognition pipeline for one input image, left to right:
        input x  ->  mean face mu  ->  x - mu  ->  z = U_k^T (x - mu)  ->  nearest training face,
    and below it the smallest Euclidean distance from z to each identity.

    The bottom chart shows DISTANCES (smaller = closer), not probabilities or confidences.
    """
    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.15, 1], hspace=0.45, wspace=0.3)
    panels = [
        (vector_to_image(x, image_shape), "1. input face x", dict(vmin=0, vmax=1)),
        (vector_to_image(mu[:, 0], image_shape), "2. training mean face mu", dict(vmin=0, vmax=1)),
        (vector_to_image(x - mu[:, 0], image_shape), "3. centered face x - mu\n(rescaled for display)", {}),
    ]
    for col, (image, label, scale) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image, cmap="gray", **scale)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    ax_z = fig.add_subplot(grid[0, 3])
    ax_z.bar(np.arange(1, len(z) + 1), z, color=ACCENT, width=0.8)
    ax_z.axhline(0, color=INK, linewidth=0.6)
    ax_z.set_title(f"4. z = U_k^T (x - mu)\n{len(z)} PCA coordinates", fontsize=9)
    ax_z.set_xlabel("component j", fontsize=8)
    ax_z.tick_params(labelsize=7)
    _recessive_axes(ax_z)

    ax_n = fig.add_subplot(grid[0, 4])
    ax_n.imshow(vector_to_image(nearest_face, image_shape), cmap="gray", vmin=0, vmax=1)
    ax_n.set_title(f"5. nearest training face\npredicted: {predicted}", fontsize=9)
    ax_n.axis("off")

    ax_d = fig.add_subplot(grid[1, 1:4])
    names = list(per_identity)
    values = [per_identity[n] for n in names]
    colors = [ACCENT if n == predicted else MUTED for n in names]
    ax_d.barh(names, values, color=colors, height=0.6)
    for i, (name, value) in enumerate(zip(names, values)):
        ax_d.text(value, i, f"  {value:.2f}" + ("  <- nearest" if name == predicted else ""),
                  va="center", fontsize=9, color=INK)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, max(values) * 1.35)
    ax_d.set_xlabel("smallest Euclidean distance ||z - z_i|| to that person's training faces (not a probability)")
    ax_d.set_title("6. distance to each identity in PCA space", fontsize=9)
    _recessive_axes(ax_d)
    fig.suptitle(title)
    _save(fig, save_path)
    return fig
