"""
Phase 3 experiment: PCA of the training faces via the SVD.

    X_train -> mu -> X_c -> X_c = U Sigma V^T -> lambda, EVR -> k -> eigenfaces, reconstructions

Run from the project root:   python -m experiments.run_pca

Only the TRAINING faces are used. No recognition is performed.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.data_loader import DATASET_DIR, PROJECT_ROOT, download_att_faces, load_dataset, train_test_split
from src.pca import (
    center_data, choose_k, compute_mean_face, compute_svd,
    eigenvalues_from_singular_values, explained_variance_ratio, get_principal_components, numerical_rank,
)
from src.preprocessing import IMAGE_SHAPE
from src.projection import project_data, reconstruct, relative_reconstruction_error
from src.visualization import (
    plot_cumulative_variance, plot_eigenfaces, plot_explained_variance,
    plot_mean_face, plot_reconstructions, plot_singular_values,
)

N_TEST_PER_PERSON = 3
SEED = 42
THRESHOLDS = [0.80, 0.90, 0.95, 0.99]
K_VALUES = [5, 10, 20, 30, 50, 100]
RECONSTRUCTION_KS = [5, 20, 50]
RECONSTRUCTION_PEOPLE = ["s1", "s14", "s27"]

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def main() -> None:
    download_att_faces()
    X, labels = load_dataset(DATASET_DIR, IMAGE_SHAPE)
    X_train, labels_train, _, _ = train_test_split(X, labels, N_TEST_PER_PERSON, SEED)
    d, N = X_train.shape
    lines: list[str] = []

    def report(text: str = "") -> None:
        print(text)
        lines.append(text)

    report("PCA OF TRAINING FACES (AT&T, seed 42, 7 train images per person)")
    report(f"image shape (h x w):     {IMAGE_SHAPE[0]} x {IMAGE_SHAPE[1]}")
    report(f"X_train shape (d x N):   {X_train.shape}")

    # 1. Mean face and centering
    mu = compute_mean_face(X_train)
    X_c = center_data(X_train, mu)
    report(f"mean face mu shape:      {mu.shape}")
    report(f"X_c shape:               {X_c.shape}")

    # 2. SVD
    U, S, Vt = compute_svd(X_c)
    r = len(S)
    rank = numerical_rank(S, X_c.shape)
    report(f"U shape:                 {U.shape}")
    report(f"singular values:         {S.shape[0]}  (r = min(d, N) = {min(d, N)})")
    report(f"V^T shape:               {Vt.shape}")
    report(f"numerical rank of X_c:   {rank}  (<= N - 1 = {N - 1} because X_c 1 = 0)")
    report(f"smallest singular value: {S[-1]:.3e}  (largest: {S[0]:.4f})")

    # 3. Eigenvalues and explained variance
    lam = eigenvalues_from_singular_values(S, N)
    evr, cumulative = explained_variance_ratio(S)
    report(f"total variance trace(C) = sum lambda_i = {lam.sum():.4f}")
    report()
    report("  i    sigma_i    lambda_i    EVR_i     cumulative")
    for i in range(10):
        report(f"{i + 1:3d}  {S[i]:9.4f}  {lam[i]:9.5f}  {evr[i]:8.4f}  {cumulative[i]:8.4f}")
    report()

    # 4. Choice of k
    report("Cumulative explained variance (and relative reconstruction error of X_train):")
    report("    k   cumulative   ||X - X_hat||_F / ||X_c||_F")
    for k in [k for k in K_VALUES if k <= rank]:
        U_k = get_principal_components(U, k)
        X_hat = reconstruct(project_data(X_c, U_k), mu, U_k)
        rel = relative_reconstruction_error(X_train, X_hat, mu)
        report(f"  {k:3d}   {cumulative[k - 1]:8.4f}     {rel:.4f}   (sqrt(1 - cumulative) = {np.sqrt(1 - cumulative[k - 1]):.4f})")
    report()
    report("Smallest k reaching each variance threshold:")
    for threshold in THRESHOLDS:
        report(f"  {threshold:.0%}: k = {choose_k(cumulative, threshold)}")
    report()

    # 5. Numerical verification on the real data
    U_10 = get_principal_components(U, 10)
    eig_residual = np.linalg.norm(X_c @ (X_c.T @ U_10) / (N - 1) - U_10 * lam[:10]) / np.linalg.norm(U_10 * lam[:10])
    report("Verification:")
    report(f"  max |row mean of X_c|             = {np.abs(X_c.mean(axis=1)).max():.2e}")
    report(f"  ||X_c - U S V^T||_F / ||X_c||_F    = {np.linalg.norm(X_c - U @ np.diag(S) @ Vt) / np.linalg.norm(X_c):.2e}")
    report(f"  max |U^T U - I|                   = {np.abs(U.T @ U - np.eye(r)).max():.2e}")
    report(f"  ||C U_10 - U_10 Lambda_10|| (rel.) = {eig_residual:.2e}   (C applied as X_c (X_c^T u) / (N-1))")
    report(f"  |sum lambda - sum of pixel variances| = {abs(lam.sum() - np.var(X_train, axis=1, ddof=1).sum()):.2e}")
    U_rank = get_principal_components(U, rank)
    Z_rank = project_data(X_c, U_rank)
    X_full = reconstruct(Z_rank, mu, U_rank)
    report(f"  ||X_c 1|| (columns sum to zero)   = {np.linalg.norm(X_c @ np.ones(N)):.2e}")
    report(f"  |sum EVR - 1|                     = {abs(evr.sum() - 1):.2e}")
    report(f"  ||X - X_hat||_F / ||X||_F, k = {rank} = {np.linalg.norm(X_train - X_full) / np.linalg.norm(X_train):.2e}"
           f"   (Z shape {Z_rank.shape})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pca_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savetxt(
        RESULTS_DIR / "pca_spectrum.csv",
        np.column_stack([np.arange(1, r + 1), S, lam, evr, cumulative]),
        delimiter=",", header="i,sigma_i,lambda_i,evr_i,cumulative_evr", comments="", fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10f"],
    )

    # 6. Figures
    plot_mean_face(mu, IMAGE_SHAPE, save_path=FIGURES_DIR / "mean_face.png")
    plot_eigenfaces(get_principal_components(U, 10), IMAGE_SHAPE,
                    title="First 10 Eigenfaces (principal directions u_1 ... u_10)",
                    save_path=FIGURES_DIR / "eigenfaces_first_10.png")
    plot_explained_variance(evr, n_show=50, save_path=FIGURES_DIR / "explained_variance.png")
    plot_cumulative_variance(cumulative, THRESHOLDS, save_path=FIGURES_DIR / "cumulative_explained_variance.png")
    plot_singular_values(S, save_path=FIGURES_DIR / "singular_values.png")

    columns = [int(np.flatnonzero(labels_train == person)[0]) for person in RECONSTRUCTION_PEOPLE]
    ks = [k for k in RECONSTRUCTION_KS if k <= rank] + [rank]
    reconstructions = []
    for k in ks:
        U_k = get_principal_components(U, k)
        reconstructions.append(reconstruct(project_data(X_c[:, columns], U_k), mu, U_k))
    plot_reconstructions(X_train[:, columns], reconstructions, ks, IMAGE_SHAPE,
                         row_titles=RECONSTRUCTION_PEOPLE, save_path=FIGURES_DIR / "reconstructions.png")

    print(f"\nResults saved to {RESULTS_DIR.relative_to(PROJECT_ROOT)}, figures to {FIGURES_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
