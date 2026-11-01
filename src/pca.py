"""
Principal Component Analysis via the Singular Value Decomposition.

Convention: X has shape (d, N), one face per COLUMN.

    mu         = (1/N) X 1                             mean face, shape (d, 1)
    X_c        = X - mu                                mu subtracted from every column
    X_c        = U Sigma V^T                           thin SVD, r = min(d, N)
    C          = (1/(N-1)) X_c X_c^T = U [Sigma^2/(N-1)] U^T
    C u_i      = lambda_i u_i,   lambda_i = sigma_i^2 / (N-1)

So the columns of U are eigenvectors of the sample covariance matrix C
(the principal directions, displayed as "eigenfaces"), and they are
orthonormal: U^T U = I.

The (d x d) matrix C is never formed: for d = 4992 it would have about
24.9 million entries. The SVD of X_c gives its eigenvectors and eigenvalues
directly. numpy.linalg.svd is the only library routine used for the core
mathematics; every other step is written out explicitly.
"""

import numpy as np


def compute_mean_face(X: np.ndarray) -> np.ndarray:
    """
    Mean face  mu = (1/N) sum_{i=1}^{N} x_i = (1/N) X 1,  with 1 = (1, ..., 1)^T in R^N.

    Input:  X of shape (d, N), one face per column.
    Output: mu of shape (d, 1). It is kept 2-D so that X - mu subtracts mu from every column.

    Interpretation: entry p of mu is the average intensity of pixel p over all
    N faces; mu is the centre of the cloud of N points in R^d.
    """
    d, N = X.shape
    ones = np.ones((N, 1))
    return (X @ ones) / N


def center_data(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Centered data  X_c = X - mu,  i.e. column i becomes x_c,i = x_i - mu.

    Input:  X of shape (d, N); mean of shape (d, 1).
    Output: X_c of shape (d, N). Each row of X_c (one pixel) has mean 0, so X_c 1 = 0.

    Interpretation: moves the origin to the mean face. PCA then describes how faces
    VARY around the average face rather than the average face itself.
    The mean must be the TRAINING mean, also when centering test faces later.
    """
    if mean.shape != (X.shape[0], 1):
        raise ValueError(f"mean must have shape ({X.shape[0]}, 1), got {mean.shape}")
    return X - mean  # broadcasting: the (d, 1) column is subtracted from each of the N columns


def compute_svd(X_centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin singular value decomposition  X_c = U Sigma V^T.

    Input:  X_c of shape (d, N); let r = min(d, N).
    Output: U  of shape (d, r) - orthonormal columns u_1, ..., u_r (principal directions)
            S  of shape (r,)   - singular values sigma_1 >= ... >= sigma_r >= 0 (diagonal of Sigma)
            Vt of shape (r, N) - orthonormal rows

    For the training faces: d = 4992, N = 280, so U is (4992, 280), S has 280
    entries and Vt is (280, 280).

    Why at most r = min(d, N) components: rank(X_c) <= min(d, N), since the rank
    cannot exceed the number of rows or of columns. Moreover X_c 1 = 0 (the columns
    sum to the zero vector after centering), so rank(X_c) <= N - 1 and the last
    singular value is (numerically) zero when d >= N.
    """
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return U, S, Vt


def eigenvalues_from_singular_values(singular_values: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Covariance eigenvalues  lambda_i = sigma_i^2 / (N - 1).

    Derivation: X_c X_c^T = U Sigma V^T V Sigma U^T = U Sigma^2 U^T, so
        C = (1/(N-1)) X_c X_c^T = U [Sigma^2 / (N-1)] U^T,
    which is an eigendecomposition of C:  C u_i = (sigma_i^2 / (N-1)) u_i.

    Input:  singular values of shape (r,), n_samples = N (number of columns of X_c).
    Output: lambda of shape (r,), in decreasing order.

    Interpretation: lambda_i is the variance of the faces along direction u_i.
    """
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for the sample covariance (N - 1 > 0)")
    return singular_values**2 / (n_samples - 1)


def explained_variance_ratio(singular_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fraction of total variance along each principal direction, and its running total.

        EVR_i        = sigma_i^2 / sum_j sigma_j^2  ( = lambda_i / sum_j lambda_j )
        cumulative_k = EVR_1 + ... + EVR_k

    The factor 1/(N-1) cancels, so singular values can be used directly.
    The total sum_j lambda_j = trace(C) is the total variance of the data.

    Input:  singular values of shape (r,).
    Output: (evr, cumulative), both of shape (r,); sum(evr) = 1, cumulative is
            non-decreasing and ends at 1. cumulative[k - 1] is the fraction of
            variance kept by the first k components.
    """
    squared = singular_values**2
    evr = squared / squared.sum()
    return evr, np.cumsum(evr)


def choose_k(cumulative_variance: np.ndarray, threshold: float) -> int:
    """
    Smallest number of components k whose cumulative explained variance >= threshold.

    Input:  cumulative explained variance of shape (r,) (from explained_variance_ratio);
            threshold in (0, 1], e.g. 0.90.
    Output: k with 1 <= k <= r, such that cumulative_variance[k - 1] >= threshold
            (k counts components; the array is 0-indexed).

    A tolerance of 1e-12 absorbs floating-point rounding (so threshold = 1.0 works).
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    reached = np.flatnonzero(cumulative_variance >= threshold - 1e-12)
    return int(reached[0]) + 1


def numerical_rank(singular_values: np.ndarray, matrix_shape: tuple[int, int]) -> int:
    """
    Number of singular values that are non-zero up to floating-point precision.

    A singular value counts as zero if sigma_i <= sigma_1 * max(d, N) * machine epsilon
    (the same tolerance as numpy.linalg.matrix_rank). For centered data with d >= N
    the result is at most N - 1, because X_c 1 = 0.

    Input:  singular values of shape (r,) (sorted, decreasing); matrix_shape = (d, N).
    Output: integer rank, 0 <= rank <= r.
    """
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0
    tolerance = singular_values[0] * max(matrix_shape) * np.finfo(float).eps
    return int(np.sum(singular_values > tolerance))


def get_principal_components(U: np.ndarray, k: int) -> np.ndarray:
    """
    First k principal directions  U_k = [u_1 | ... | u_k] = U[:, :k].

    Input:  U of shape (d, r); integer k with 1 <= k <= r.
    Output: U_k of shape (d, k), orthonormal columns: U_k^T U_k = I_k.

    u_1 is the unit direction of maximum variance, u_2 the direction of maximum
    remaining variance orthogonal to u_1, and so on. Reshaped to (78, 64), each
    column is an "eigenface".
    """
    if not 1 <= k <= U.shape[1]:
        raise ValueError(f"k must be between 1 and {U.shape[1]}, got {k}")
    return U[:, :k]
