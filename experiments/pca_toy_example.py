"""
PCA by hand: a 2 x 5 example that can be checked with pen and paper.

Think of each column as a tiny "image" with only d = 2 pixels; there are N = 5 images.
The same functions as in the face experiment are used, step by step:

    X -> mean -> centered X -> SVD -> eigenvalues -> principal directions -> projection

Run from the project root:   python -m experiments.pca_toy_example

Expected hand results (derived in the printout):
    mu = (5, 5)^T
    X_c X_c^T = [[20, 12], [12, 20]]  ->  sigma^2 = 32, 8  ->  lambda = 32/4 = 8, 8/4 = 2
    u_1 = (1, 1)/sqrt(2),  u_2 = (1, -1)/sqrt(2)   (each possibly multiplied by -1)
    explained variance: 80 %, 20 %
"""

import numpy as np

from src.pca import (
    center_data, compute_mean_face, compute_svd,
    eigenvalues_from_singular_values, explained_variance_ratio, get_principal_components,
)
from src.projection import project_data, reconstruct, reconstruction_error

np.set_printoptions(precision=4, suppress=True)


def show(name: str, value) -> None:
    print(f"{name} =\n{np.asarray(value)}\n")


def main() -> None:
    X = np.array([[2.0, 4.0, 5.0, 6.0, 8.0],
                  [4.0, 2.0, 5.0, 8.0, 6.0]])
    d, N = X.shape
    print(f"STEP 0. Data: d = {d} pixels, N = {N} images (one per column)\n")
    show("X", X)

    print("STEP 1. Mean  mu = (1/N) X 1")
    mu = compute_mean_face(X)
    show("mu", mu)
    print("  by hand: (2+4+5+6+8)/5 = 5,  (4+2+5+8+6)/5 = 5\n")

    print("STEP 2. Centering  X_c = X - mu")
    X_c = center_data(X, mu)
    show("X_c", X_c)
    print("  each row sums to 0:", X_c.sum(axis=1), "\n")

    print("STEP 3. (For checking only) covariance  C = X_c X_c^T / (N-1)")
    show("X_c X_c^T", X_c @ X_c.T)
    print("  by hand: row1.row1 = 9+1+0+1+9 = 20, row1.row2 = 3+3+0+3+3 = 12, row2.row2 = 20")
    show("C", X_c @ X_c.T / (N - 1))
    print("  [[20,12],[12,20]] has eigenvectors (1,1) and (1,-1) with eigenvalues 20+12 = 32 and 20-12 = 8\n")

    print("STEP 4. SVD  X_c = U Sigma V^T")
    U, S, Vt = compute_svd(X_c)
    show("U", U)
    show("singular values sigma", S)
    show("V^T", Vt)
    print(f"  sigma^2 = {S**2}  (= eigenvalues of X_c X_c^T: 32 and 8)")
    print(f"  sqrt(32) = {np.sqrt(32):.4f},  sqrt(8) = {np.sqrt(8):.4f},  1/sqrt(2) = {1 / np.sqrt(2):.4f}")
    show("check: U Sigma V^T", U @ np.diag(S) @ Vt)
    show("check: U^T U (should be I)", U.T @ U)
    print("  Note: the SVD may return -u instead of u; both are valid unit eigenvectors.\n")

    print("STEP 5. Eigenvalues  lambda_i = sigma_i^2 / (N-1)")
    lam = eigenvalues_from_singular_values(S, N)
    show("lambda", lam)
    C = X_c @ X_c.T / (N - 1)
    for i in range(2):
        print(f"  check C u_{i + 1} = lambda_{i + 1} u_{i + 1}:  {C @ U[:, i]}  vs  {lam[i] * U[:, i]}")
    print()

    print("STEP 6. Explained variance  EVR_i = sigma_i^2 / sum sigma_j^2")
    evr, cumulative = explained_variance_ratio(S)
    show("EVR", evr)
    show("cumulative", cumulative)
    print("  by hand: 32/40 = 0.8, 8/40 = 0.2\n")

    print("STEP 7. Keep k = 1 principal direction and project  Z = U_1^T X_c")
    U_1 = get_principal_components(U, 1)
    show("U_1", U_1)
    Z = project_data(X_c, U_1)
    show("Z (one coordinate per image)", Z)
    print("  by hand with u_1 = (1,1)/sqrt(2): z_i = (x_c,1i + x_c,2i)/sqrt(2)")
    print(f"  e.g. image 1: (-3 + -1)/sqrt(2) = {-4 / np.sqrt(2):.4f}  (sign flips if the SVD returned -u_1)")
    print(f"  variance of Z = sum z^2 / (N-1) = {np.sum(Z**2) / (N - 1):.4f}  (= lambda_1 = 8)\n")

    print("STEP 8. Reconstruct from k = 1:  X_hat = U_1 Z + mu")
    X_hat = reconstruct(Z, mu, U_1)
    show("X_hat", X_hat)
    errors = reconstruction_error(X, X_hat)
    show("per-image error ||x_i - x_hat_i||", errors)
    print(f"  sum of squared errors = {np.sum(errors**2):.4f}  (= discarded sigma_2^2 = 8)")
    print("  The points are moved onto the line through mu in direction u_1: 2-D data -> 1 number each.")


if __name__ == "__main__":
    main()
